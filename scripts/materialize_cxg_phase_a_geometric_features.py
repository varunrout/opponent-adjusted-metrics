"""Phase A geometric/categorical defender features (CxG+ only, new candidate features).

Adds 6 new columns to the EXISTING `oam_features.cxg_defensive_360_features` table
(ALTER TABLE ADD COLUMN IF NOT EXISTS + scoped UPDATE...FROM a staging table, keyed on
event_id) -- deliberately NOT a `CREATE OR REPLACE TABLE` rebuild, since that table's
existing 33 columns are computed by the frozen F1-F14 JSON-blob pipeline in
`materialize_cxg_feature_family_tables.py`, which this script does not touch or re-run.

Landed in `cxg_defensive_360_features` rather than `cxg_line_shape_360_features` because
these are per-defender-IDENTITY features (which specific defender, their role, their
distance to a teammate) -- the same category as that table's existing
`nearest_defender_distance`/`nearest_defender_distance_delta`. `line_shape_360` holds
team-SHAPE features (defensive line/width/hull), a different unit of analysis.

  - nearest_defender_role                categorical (GK/CB/Fullback_WingBack/Midfield/Attack)
  - nearest_defender_zone_displacement   continuous, metres
  - second_nearest_defender_role         categorical, same taxonomy
  - nearest_defender_gap                 continuous, metres
  - nearest_defender_rank_null_reason    explains why the two second-nearest-dependent
                                          columns are null (never a silent/fabricated null)
  - phase_a_materialized_at              lineage timestamp for just these new columns

After the update, re-runs the EXISTING (unmodified) `_create_training_view` from
`materialize_cxg_feature_family_tables.py` so `cxg_training_matrix_v1` picks up the new
columns -- that function already introspects each family table's live columns at call time,
so simply calling it again (not editing it) is sufficient.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from google.cloud import bigquery

from opponent_adjusted.features.cxg.phase_a_geometric import (
    DefenderCandidate,
    compute_shot_features,
    role_centroids,
)
from materialize_cxg_feature_family_tables import _create_training_view  # noqa: E402

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
TARGET_TABLE = "cxg_defensive_360_features"
STAGING_TABLE = "_phase_a_geometric_staging"

NEW_COLUMNS = [
    ("nearest_defender_role", "STRING"),
    ("nearest_defender_zone_displacement", "FLOAT64"),
    ("second_nearest_defender_role", "STRING"),
    ("nearest_defender_gap", "FLOAT64"),
    ("nearest_defender_rank_null_reason", "STRING"),
    ("phase_a_materialized_at", "STRING"),
]

COHORT_QUERY = f"""
SELECT event_id, shot_x_sb, shot_y_sb
FROM `{PROJECT}.{FEATURE_DATASET}.cxg_shot_base_features`
WHERE has_360_frame AND data_version = @data_version AND silver_schema_version = @schema_version
"""

DEFENDERS_QUERY = f"""
SELECT shot_event_id, position_name, x, y
FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND teammate = FALSE AND x IS NOT NULL AND y IS NOT NULL
  AND shot_event_id IN UNNEST(@shot_ids)
"""

# Dataset-wide, every event with a usable position_name + coordinates -- not just shot
# freeze-frame defenders -- per Step 1's "typical/expected zone centroid" definition. Step 1
# confirmed no attack-direction flip is needed (raw location_x/y is already per-event
# goal-oriented, the same convention `goalward_progress_sb`/`defensive_line_depth` rely on),
# so this is a single dataset-wide aggregation, not split by period/team.
ROLE_CENTROID_STATS_QUERY = f"""
SELECT position_name, COUNT(*) AS n, SUM(location_x) AS sum_x, SUM(location_y) AS sum_y
FROM `{PROJECT}.{CORE_DATASET}.events`
WHERE data_version = @data_version AND silver_schema_version = @schema_version
  AND position_name IS NOT NULL AND location_x IS NOT NULL AND location_y IS NOT NULL
GROUP BY position_name
"""


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params(extra: list | None = None) -> list:
    base = [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]
    return base + (extra or [])


def _rows(client: bigquery.Client, sql: str, job_config: bigquery.QueryJobConfig) -> list:
    return list(client.query(sql, job_config=job_config, location=LOCATION).result())


def _ensure_columns(client: bigquery.Client) -> None:
    add_clauses = ", ".join(f"ADD COLUMN IF NOT EXISTS `{name}` {sql_type}" for name, sql_type in NEW_COLUMNS)
    client.query(f"ALTER TABLE `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}` {add_clauses}", location=LOCATION).result()


def main() -> None:
    client = _client()
    now = datetime.now(UTC).isoformat()

    print("[phase_a] fetching 360-eligible cohort...")
    cohort_rows = _rows(client, COHORT_QUERY, bigquery.QueryJobConfig(query_parameters=_params()))
    print(f"[phase_a] cohort size: {len(cohort_rows)}")
    shot_ids = [r.event_id for r in cohort_rows]

    print("[phase_a] fetching defenders + role-centroid stats...")
    defender_rows = _rows(
        client, DEFENDERS_QUERY,
        bigquery.QueryJobConfig(query_parameters=_params([bigquery.ArrayQueryParameter("shot_ids", "STRING", shot_ids)])),
    )
    centroid_rows = _rows(client, ROLE_CENTROID_STATS_QUERY, bigquery.QueryJobConfig(query_parameters=_params()))

    position_stats = {r.position_name: (r.n, float(r.sum_x), float(r.sum_y)) for r in centroid_rows}
    centroids = role_centroids(position_stats)
    print(f"[phase_a] role centroids: {centroids}")

    defenders_by_shot: dict[str, list] = {}
    for r in defender_rows:
        defenders_by_shot.setdefault(r.shot_event_id, []).append(DefenderCandidate(position_name=r.position_name, x=float(r.x), y=float(r.y)))

    staging_rows = []
    for shot in cohort_rows:
        candidates = tuple(defenders_by_shot.get(shot.event_id, ()))
        feats = compute_shot_features(float(shot.shot_x_sb), float(shot.shot_y_sb), candidates, centroids)
        staging_rows.append({"event_id": shot.event_id, **feats, "phase_a_materialized_at": now})

    print(f"[phase_a] computed features for {len(staging_rows)} shots")

    staging_ref = f"{PROJECT}.{FEATURE_DATASET}.{STAGING_TABLE}"
    staging_schema = [bigquery.SchemaField("event_id", "STRING", mode="REQUIRED")] + [
        bigquery.SchemaField(name, sql_type) for name, sql_type in NEW_COLUMNS
    ]
    client.delete_table(staging_ref, not_found_ok=True)
    client.create_table(bigquery.Table(staging_ref, schema=staging_schema))
    job_config = bigquery.LoadJobConfig(schema=staging_schema, write_disposition="WRITE_TRUNCATE")
    client.load_table_from_json(staging_rows, staging_ref, job_config=job_config, location=LOCATION).result()

    print("[phase_a] ensuring target columns exist...")
    _ensure_columns(client)

    print("[phase_a] updating target table from staging...")
    set_clause = ", ".join(f"tgt.`{name}` = src.`{name}`" for name, _ in NEW_COLUMNS)
    update_sql = f"""
    UPDATE `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}` AS tgt
    SET {set_clause}
    FROM `{staging_ref}` AS src
    WHERE tgt.event_id = src.event_id
    """
    client.query(update_sql, location=LOCATION).result()
    client.delete_table(staging_ref, not_found_ok=True)

    print("[phase_a] refreshing cxg_training_matrix_v1 view (existing, unmodified function)...")
    _create_training_view(client, f"{PROJECT}.{FEATURE_DATASET}")

    verify_sql = f"""
    SELECT COUNT(*) AS n, COUNTIF(nearest_defender_role IS NOT NULL) AS n_role,
           COUNTIF(second_nearest_defender_role IS NOT NULL) AS n_second_role,
           COUNTIF(nearest_defender_gap IS NOT NULL) AS n_gap
    FROM `{PROJECT}.{FEATURE_DATASET}.{TARGET_TABLE}`
    """
    summary = dict(list(client.query(verify_sql, location=LOCATION).result())[0].items())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
