"""Materialize typed cXG feature-family tables from the Gold JSON snapshot."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from google.cloud import bigquery

from opponent_adjusted.features.cxg.contracts import (
    event_candidate_names,
    three_sixty_candidate_names_for_families,
)

PROJECT = "oam-varun-260819"
LOCATION = "europe-west2"
DATASET = "oam_features"
SOURCE_TABLE = "cxg_shot_features"

BASE_COLUMN_CANDIDATES = (
    ("event_id", ("event_id",)),
    ("match_id", ("match_id",)),
    ("team_id", ("team_id",)),
    ("player_id", ("player_id",)),
    ("shot_x_sb", ("shot_x_sb", "shot_x")),
    ("shot_y_sb", ("shot_y_sb", "shot_y")),
    ("outcome_name", ("outcome_name", "shot_outcome_name")),
    ("is_goal", ("is_goal",)),
    ("has_360_frame", ("has_360_frame",)),
    ("event_context_valid", ("event_context_valid",)),
    ("data_version", ("data_version",)),
    ("silver_schema_version", ("silver_schema_version",)),
    ("feature_version", ("feature_version",)),
    ("materialized_at", ("materialized_at",)),
)

FAMILY_TABLES = {
    "base": "cxg_shot_base_features",
    "geometry": "cxg_shot_geometry_features",
    "event_context": "cxg_event_context_features",
    "buildup": "cxg_buildup_features",
    "defensive_360": "cxg_defensive_360_features",
    "goalkeeper_360": "cxg_goalkeeper_360_features",
    "line_shape_360": "cxg_line_shape_360_features",
}

EVENT_CONTEXT_FEATURES = set(event_candidate_names())
GOALKEEPER_360_FEATURES = set(three_sixty_candidate_names_for_families(("F4", "F9"))) | {"gk_setness_proxy"}
LINE_SHAPE_360_FEATURES = set(three_sixty_candidate_names_for_families(("F2", "F5", "F6", "F11", "F13", "F14")))
DEFENSIVE_360_FEATURES = (
    set(three_sixty_candidate_names_for_families(("F1", "F3", "F7", "F8", "F10", "F12", "F15")))
    - GOALKEEPER_360_FEATURES
)
THREE_SIXTY_FEATURES = DEFENSIVE_360_FEATURES | GOALKEEPER_360_FEATURES | LINE_SHAPE_360_FEATURES

# Physical shot-state fields may be present in future snapshots. Contextual E/F
# fields are classified by the locked taxonomy above, never by fuzzy names.
GEOMETRY_FEATURES = {
    "shot_distance_sb",
    "shot_angle_rad",
    "body_part_name",
    "technique_name",
    "shot_type_name",
    "first_time",
    "open_goal",
    "one_on_one",
    "follows_dribble",
    "under_pressure",
}


@dataclass(frozen=True)
class FeatureColumn:
    source_key: str
    column_name: str
    family: str
    value_type: str


def _column_name(feature_key: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", feature_key).strip("_").lower()
    if not name:
        name = "feature"
    if name[0].isdigit():
        name = f"f_{name}"
    return name[:300]


def _classify(feature_key: str) -> str:
    key = feature_key.lower()
    if key in EVENT_CONTEXT_FEATURES:
        return "event_context"
    if key in GOALKEEPER_360_FEATURES:
        return "goalkeeper_360"
    if key in LINE_SHAPE_360_FEATURES:
        return "line_shape_360"
    if key in DEFENSIVE_360_FEATURES:
        return "defensive_360"
    if key in GEOMETRY_FEATURES:
        return "geometry"
    raise ValueError(f"Feature key is not in the locked CxG taxonomy: {feature_key}")


def _json_object(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _infer_type(values: list[object]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "FLOAT64"
    if all(isinstance(value, bool) for value in present):
        return "BOOL"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
        return "FLOAT64"
    return "STRING"


def _discover_features(client: bigquery.Client, source: str) -> list[FeatureColumn]:
    values_by_key: dict[str, list[object]] = defaultdict(list)
    rows = client.query(
        f"""
        SELECT feature_values_json
        FROM `{source}`
        WHERE feature_values_json IS NOT NULL
        """,
        location=LOCATION,
    ).result()
    for row in rows:
        for key, value in _json_object(row.feature_values_json).items():
            values_by_key[key].append(value)

    used_names: dict[str, int] = defaultdict(int)
    columns: list[FeatureColumn] = []
    for key in sorted(values_by_key):
        base_name = _column_name(key)
        used_names[base_name] += 1
        name = base_name if used_names[base_name] == 1 else f"{base_name}_{used_names[base_name]}"
        columns.append(
            FeatureColumn(
                source_key=key,
                column_name=name,
                family=_classify(key),
                value_type=_infer_type(values_by_key[key]),
            )
        )
    return columns


def _source_columns(client: bigquery.Client, dataset_ref: str, table_name: str) -> set[str]:
    project_id, dataset_id = dataset_ref.split(".", 1)
    sql = f"""
    SELECT column_name
    FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = @table_name
    """
    config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("table_name", "STRING", table_name)]
    )
    return {row.column_name for row in client.query(sql, job_config=config, location=LOCATION).result()}


def _base_select_columns(source_columns: set[str]) -> list[str]:
    select_columns: list[str] = []
    for output_name, candidates in BASE_COLUMN_CANDIDATES:
        for candidate in candidates:
            if candidate in source_columns:
                if candidate == output_name:
                    select_columns.append(f"`{candidate}`")
                else:
                    select_columns.append(f"`{candidate}` AS `{output_name}`")
                break
    return select_columns


def _json_path(feature_key: str) -> str:
    escaped_key = feature_key.replace("\\", "\\\\").replace('"', '\\"')
    return f"'$.\"{escaped_key}\"'"


def _json_expr(feature: FeatureColumn) -> str:
    value = f"JSON_VALUE(PARSE_JSON(JSON_VALUE(feature_values_json), wide_number_mode=>'round'), {_json_path(feature.source_key)})"
    if feature.value_type == "BOOL":
        expr = f"SAFE_CAST({value} AS BOOL)"
    elif feature.value_type == "FLOAT64":
        expr = f"SAFE_CAST({value} AS FLOAT64)"
    else:
        expr = value
    return f"{expr} AS `{feature.column_name}`"


def _materialize_table(
    client: bigquery.Client,
    source: str,
    destination: str,
    select_columns: list[str],
    where_clause: str | None = None,
) -> None:
    where_sql = f"\nWHERE {where_clause}" if where_clause else ""
    join_sep = ",\n      "
    sql = f"""
    CREATE OR REPLACE TABLE `{destination}` AS
    SELECT
      {join_sep.join(select_columns)}
    FROM `{source}`{where_sql}
    """
    client.query(sql, location=LOCATION).result()


def _create_training_view(client: bigquery.Client, dataset_ref: str) -> None:
    table = lambda name: f"`{dataset_ref}.{name}`"
    metadata = {"event_id", "match_id", "data_version", "feature_version", "materialized_at"}
    joins = []
    select_parts = ["base.*"]
    aliases = {
        "geometry": "geom",
        "event_context": "ctx",
        "buildup": "build",
        "defensive_360": "def360",
        "goalkeeper_360": "gk360",
        "line_shape_360": "line360",
    }
    for family in ["geometry", "event_context", "buildup", "defensive_360", "goalkeeper_360", "line_shape_360"]:
        table_name = FAMILY_TABLES[family]
        columns = _source_columns(client, dataset_ref, table_name)
        feature_columns = sorted(columns - metadata)
        alias = aliases[family]
        joins.append(f"LEFT JOIN {table(table_name)} AS {alias} USING(event_id)")
        if feature_columns:
            select_parts.extend(f"{alias}.`{column}`" for column in feature_columns)

    select_sep = ",\n      "
    joins_sep = " "
    sql = f"""
    CREATE OR REPLACE VIEW `{dataset_ref}.cxg_training_matrix_v1` AS
    SELECT
      {select_sep.join(select_parts)}
    FROM {table(FAMILY_TABLES['base'])} AS base
    {joins_sep.join(joins)}
    """
    client.query(sql, location=LOCATION).result()


def main() -> None:
    global LOCATION

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", default=PROJECT)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--source-table", default=SOURCE_TABLE)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOCATION = args.location
    client = bigquery.Client(project=args.project_id)
    dataset_ref = f"{args.project_id}.{args.dataset}"
    source = f"{dataset_ref}.{args.source_table}"

    features = _discover_features(client, source)
    by_family: dict[str, list[FeatureColumn]] = defaultdict(list)
    type_counts: dict[str, int] = defaultdict(int)
    for feature in features:
        by_family[feature.family].append(feature)
        type_counts[feature.value_type] += 1

    result = {
        "source": source,
        "dataset": dataset_ref,
        "feature_count": len(features),
        "families": {family: len(by_family[family]) for family in FAMILY_TABLES},
        "types": dict(sorted(type_counts.items())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    source_columns = _source_columns(client, dataset_ref, args.source_table)
    common = _base_select_columns(source_columns)
    _materialize_table(client, source, f"{dataset_ref}.{FAMILY_TABLES['base']}", common)

    for family, table_name in FAMILY_TABLES.items():
        if family == "base":
            continue
        select_columns = [
            "event_id",
            "match_id",
            "data_version",
            "feature_version",
            "materialized_at",
            *[_json_expr(feature) for feature in by_family[family]],
        ]
        where_clause = "has_360_frame" if family.endswith("_360") else None
        _materialize_table(client, source, f"{dataset_ref}.{table_name}", select_columns, where_clause)

    _create_training_view(client, dataset_ref)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


