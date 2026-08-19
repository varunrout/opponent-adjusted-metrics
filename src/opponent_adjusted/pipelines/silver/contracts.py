"""Explicit schema contracts for StatsBomb Silver v1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

SILVER_SCHEMA_VERSION = "statsbomb_silver_v1_2"


@dataclass(frozen=True)
class ColumnContract:
    name: str
    arrow_type: str
    nullable: bool = True
    repeated: bool = False


@dataclass(frozen=True)
class TableContract:
    name: str
    key: list[str]
    columns: list[ColumnContract]


def _col(
    name: str, arrow_type: str, nullable: bool = True, repeated: bool = False
) -> ColumnContract:
    return ColumnContract(name=name, arrow_type=arrow_type, nullable=nullable, repeated=repeated)


LINEAGE = [
    _col("data_version", "string", nullable=False),
    _col("silver_schema_version", "string", nullable=False),
]

CONTRACTS: dict[str, TableContract] = {
    "competitions": TableContract(
        name="competitions",
        key=["competition_id", "season_id", "data_version", "silver_schema_version"],
        columns=[
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("competition_name", "string"),
            _col("competition_gender", "string"),
            _col("country_name", "string"),
            _col("season_name", "string"),
            _col("match_updated", "string"),
            _col("match_available", "string"),
            _col("match_updated_360", "string"),
            _col("match_available_360", "string"),
            *LINEAGE,
        ],
    ),
    "matches": TableContract(
        name="matches",
        key=["match_id", "data_version", "silver_schema_version"],
        columns=[
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("match_date", "string"),
            _col("kick_off", "string"),
            _col("home_team_id", "int64"),
            _col("home_team_name", "string"),
            _col("away_team_id", "int64"),
            _col("away_team_name", "string"),
            _col("home_score", "int64"),
            _col("away_score", "int64"),
            _col("competition_stage", "string"),
            _col("stadium", "string"),
            _col("referee", "string"),
            _col("match_status", "string"),
            _col("match_status_360", "string"),
            _col("last_updated", "string"),
            _col("last_updated_360", "string"),
            *LINEAGE,
        ],
    ),
    "teams": TableContract(
        name="teams",
        key=["team_id", "data_version", "silver_schema_version"],
        columns=[_col("team_id", "int64", nullable=False), _col("team_name", "string"), *LINEAGE],
    ),
    "players": TableContract(
        name="players",
        key=["player_id", "data_version", "silver_schema_version"],
        columns=[
            _col("player_id", "int64", nullable=False),
            _col("player_name", "string"),
            *LINEAGE,
        ],
    ),
    "starting_xi_players": TableContract(
        name="starting_xi_players",
        key=["event_id", "lineup_ordinal", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("event_index", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("team_name", "string"),
            _col("formation", "int64"),
            _col("lineup_ordinal", "int64", nullable=False),
            _col("player_id", "int64", nullable=False),
            _col("player_name", "string"),
            _col("position_id", "int64"),
            _col("position_name", "string"),
            _col("jersey_number", "int64"),
            *LINEAGE,
        ],
    ),
    "events": TableContract(
        name="events",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("event_index", "int64", nullable=False),
            _col("period", "int64"),
            _col("minute", "int64"),
            _col("second", "int64"),
            _col("timestamp", "string"),
            _col("duration", "float64"),
            _col("event_type_id", "int64"),
            _col("event_type_name", "string"),
            _col("possession_id", "int64"),
            _col("possession_team_id", "int64"),
            _col("possession_team_name", "string"),
            _col("team_id", "int64"),
            _col("team_name", "string"),
            _col("player_id", "int64"),
            _col("player_name", "string"),
            _col("position_id", "int64"),
            _col("position_name", "string"),
            _col("play_pattern_id", "int64"),
            _col("play_pattern_name", "string"),
            _col("under_pressure", "bool"),
            _col("counterpress", "bool"),
            _col("off_camera", "bool"),
            _col("out", "bool"),
            _col("location_x", "float64"),
            _col("location_y", "float64"),
            _col("related_event_ids", "string", repeated=True),
            *LINEAGE,
        ],
    ),
    "shots": TableContract(
        name="shots",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("location_x", "float64"),
            _col("location_y", "float64"),
            _col("end_x", "float64"),
            _col("end_y", "float64"),
            _col("end_z", "float64"),
            _col("statsbomb_xg", "float64"),
            _col("outcome_id", "int64"),
            _col("outcome_name", "string"),
            _col("body_part_id", "int64"),
            _col("body_part_name", "string"),
            _col("technique_id", "int64"),
            _col("technique_name", "string"),
            _col("shot_type_id", "int64"),
            _col("shot_type_name", "string"),
            _col("first_time", "bool"),
            _col("key_pass_id", "string"),
            _col("aerial_won", "bool"),
            _col("follows_dribble", "bool"),
            _col("open_goal", "bool"),
            _col("one_on_one", "bool"),
            _col("deflected", "bool"),
            _col("saved_off_target", "bool"),
            _col("saved_to_post", "bool"),
            *LINEAGE,
        ],
    ),
    "passes": TableContract(
        name="passes",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("recipient_id", "int64"),
            _col("recipient_name", "string"),
            _col("length", "float64"),
            _col("angle", "float64"),
            _col("height_id", "int64"),
            _col("height_name", "string"),
            _col("end_x", "float64"),
            _col("end_y", "float64"),
            _col("pass_type_id", "int64"),
            _col("pass_type_name", "string"),
            _col("outcome_id", "int64"),
            _col("outcome_name", "string"),
            _col("technique_id", "int64"),
            _col("technique_name", "string"),
            _col("body_part_id", "int64"),
            _col("body_part_name", "string"),
            _col("assisted_shot_id", "string"),
            _col("shot_assist", "bool"),
            _col("goal_assist", "bool"),
            _col("cross", "bool"),
            _col("cut_back", "bool"),
            _col("switch", "bool"),
            _col("through_ball", "bool"),
            _col("inswinging", "bool"),
            _col("outswinging", "bool"),
            _col("straight", "bool"),
            _col("deflected", "bool"),
            _col("miscommunication", "bool"),
            _col("no_touch", "bool"),
            _col("aerial_won", "bool"),
            *LINEAGE,
        ],
    ),
    "carries": TableContract(
        name="carries",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("end_x", "float64"),
            _col("end_y", "float64"),
            *LINEAGE,
        ],
    ),
    "dribbles": TableContract(
        name="dribbles",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("outcome_id", "int64"),
            _col("outcome_name", "string"),
            _col("nutmeg", "bool"),
            _col("overrun", "bool"),
            _col("no_touch", "bool"),
            *LINEAGE,
        ],
    ),
    "pressures": TableContract(
        name="pressures",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("period", "int64"),
            _col("minute", "int64"),
            _col("second", "int64"),
            _col("timestamp", "string"),
            _col("duration", "float64"),
            *LINEAGE,
        ],
    ),
    "ball_receipts": TableContract(
        name="ball_receipts",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("outcome_id", "int64"),
            _col("outcome_name", "string"),
            *LINEAGE,
        ],
    ),
    "disciplinary_events": TableContract(
        name="disciplinary_events",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("event_type", "string"),
            _col("card_id", "int64"),
            _col("card_name", "string"),
            _col("period", "int64"),
            _col("minute", "int64"),
            _col("second", "int64"),
            _col("timestamp", "string"),
            _col("event_index", "int64"),
            *LINEAGE,
        ],
    ),
    "substitutions": TableContract(
        name="substitutions",
        key=["event_id", "data_version", "silver_schema_version"],
        columns=[
            _col("event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("team_id", "int64"),
            _col("player_id", "int64"),
            _col("replacement_id", "int64"),
            _col("replacement_name", "string"),
            _col("outcome_id", "int64"),
            _col("outcome_name", "string"),
            _col("period", "int64"),
            _col("minute", "int64"),
            _col("second", "int64"),
            _col("timestamp", "string"),
            _col("event_index", "int64"),
            *LINEAGE,
        ],
    ),
    "possessions": TableContract(
        name="possessions",
        key=["match_id", "possession_id", "data_version", "silver_schema_version"],
        columns=[
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("possession_id", "int64", nullable=False),
            _col("possession_team_id", "int64"),
            _col("possession_team_name", "string"),
            _col("start_event_id", "string"),
            _col("end_event_id", "string"),
            _col("start_event_index", "int64"),
            _col("end_event_index", "int64"),
            _col("start_period", "int64"),
            _col("end_period", "int64"),
            _col("start_timestamp", "string"),
            _col("end_timestamp", "string"),
            _col("start_minute", "int64"),
            _col("start_second", "int64"),
            _col("end_minute", "int64"),
            _col("end_second", "int64"),
            _col("event_count", "int64"),
            _col("start_event_type", "string"),
            _col("end_event_type", "string"),
            _col("start_play_pattern", "string"),
            _col("end_play_pattern", "string"),
            *LINEAGE,
        ],
    ),
    "three_sixty_frames": TableContract(
        name="three_sixty_frames",
        key=["match_id", "event_uuid", "data_version", "silver_schema_version"],
        columns=[
            _col("match_id", "int64", nullable=False),
            _col("event_uuid", "string", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("visible_area", "float64", repeated=True),
            _col("frame_player_count", "int64", nullable=False),
            *LINEAGE,
        ],
    ),
    "three_sixty_players": TableContract(
        name="three_sixty_players",
        key=[
            "match_id",
            "event_uuid",
            "frame_player_ordinal",
            "data_version",
            "silver_schema_version",
        ],
        columns=[
            _col("match_id", "int64", nullable=False),
            _col("event_uuid", "string", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("frame_player_ordinal", "int64", nullable=False),
            _col("teammate", "bool"),
            _col("actor", "bool"),
            _col("keeper", "bool"),
            _col("x", "float64"),
            _col("y", "float64"),
            *LINEAGE,
        ],
    ),
    "shot_freeze_frame_players": TableContract(
        name="shot_freeze_frame_players",
        key=[
            "shot_event_id",
            "freeze_frame_player_ordinal",
            "data_version",
            "silver_schema_version",
        ],
        columns=[
            _col("shot_event_id", "string", nullable=False),
            _col("match_id", "int64", nullable=False),
            _col("competition_id", "int64", nullable=False),
            _col("season_id", "int64", nullable=False),
            _col("freeze_frame_player_ordinal", "int64", nullable=False),
            _col("player_id", "int64"),
            _col("player_name", "string"),
            _col("position_id", "int64"),
            _col("position_name", "string"),
            _col("teammate", "bool"),
            _col("x", "float64"),
            _col("y", "float64"),
            *LINEAGE,
        ],
    ),
}


def _arrow_type(type_name: str, repeated: bool) -> pa.DataType:
    mapping: dict[str, pa.DataType] = {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    base = mapping[type_name]
    if repeated:
        return pa.list_(base)
    return base


def table_arrow_schema(table_name: str) -> pa.Schema:
    table = CONTRACTS[table_name]
    fields = [
        pa.field(c.name, _arrow_type(c.arrow_type, c.repeated), nullable=c.nullable)
        for c in table.columns
    ]
    return pa.schema(fields)


def table_bq_schema(table_name: str) -> list[Any]:
    from google.cloud import bigquery  # type: ignore[import-untyped]

    arrow_to_bq = {
        "string": "STRING",
        "int64": "INT64",
        "float64": "FLOAT64",
        "bool": "BOOL",
    }
    fields: list[bigquery.SchemaField] = []
    for c in CONTRACTS[table_name].columns:
        mode = "REPEATED" if c.repeated else ("NULLABLE" if c.nullable else "REQUIRED")
        fields.append(bigquery.SchemaField(c.name, arrow_to_bq[c.arrow_type], mode=mode))
    return fields


def write_contract_json(path: Path) -> Path:
    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "silver_schema_version": SILVER_SCHEMA_VERSION,
        "tables": [
            {
                "table": t.name,
                "grain_key": t.key,
                "columns": [
                    {
                        "name": c.name,
                        "arrow_type": c.arrow_type,
                        "nullable": c.nullable,
                        "repeated": c.repeated,
                    }
                    for c in t.columns
                ],
            }
            for t in CONTRACTS.values()
        ],
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
