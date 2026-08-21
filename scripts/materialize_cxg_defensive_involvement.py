"""Materialize `oam_analysis.cxg_defensive_involvement_v1` (ODI Phase 1, step 1).

One row per shot with a resolvable nearest defender (from
`shot_freeze_frame_players`, `teammate = FALSE`, minimum-distance `player_id`
to the shot location -- Phase 0 confirmed defender `player_id` is 100%
populated, so no fuzzy geometric matching is used). This table spans the
FULL shot corpus (not just the 360-eligible cohort): the rolling ODI
aggregator needs every shot a defender has faced in the match/tournament,
not only 360-flagged ones, to compute an accurate trailing window.

Excludes period 5 (penalty shootout) shots, matching the existing repository
convention of excluding shootouts from match-clock-based context (see
`goal_beneficiary_team_id` in `event_context.py`, which the E-family also
excludes period 5 from).

Also writes a local exclusion-diagnostics JSON to
`audit_outputs/cxg_analysis/odi_defprofile_spike/raw/involvement_exclusions.json`
recording, rather than silently dropping, the count of shots excluded at
each stage (no freeze-frame row at all; freeze-frame present but no
resolvable defender; missing coordinates).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
FEATURE_VERSION = "cxg_odi_involvement_v1"

TABLE = f"{PROJECT}.{ANALYSIS_DATASET}.cxg_defensive_involvement_v1"
EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "audit_outputs"
    / "cxg_analysis"
    / "odi_defprofile_spike"
    / "raw"
    / "involvement_exclusions.json"
)

_TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(?:\.(\d+))?$")


def _event_clock_s(minute: int | None, second: int | None, timestamp: str | None) -> float | None:
    """Same formula as `event_context.event_clock_s` (decoupled reimplementation;
    frozen E-family module is not imported/modified by this new pipeline)."""
    if not isinstance(minute, int) or not isinstance(second, int):
        return None
    if minute < 0 or second < 0:
        return None
    fraction = 0.0
    if timestamp is not None:
        match = _TIMESTAMP_RE.fullmatch(timestamp)
        if match is None:
            return None
        if match.group(1):
            fraction = float(f"0.{match.group(1)}")
    return minute * 60.0 + second + fraction


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _params() -> list[bigquery.ScalarQueryParameter]:
    return [
        bigquery.ScalarQueryParameter("data_version", "STRING", DATA_VERSION),
        bigquery.ScalarQueryParameter("schema_version", "STRING", SCHEMA_VERSION),
    ]


DIAGNOSTICS_QUERY = f"""
WITH shot_period AS (
  SELECT s.event_id AS shot_event_id, e.period
  FROM `{PROJECT}.{CORE_DATASET}.shots` s
  JOIN `{PROJECT}.{CORE_DATASET}.events` e
    ON s.event_id = e.event_id AND e.data_version = @data_version AND e.silver_schema_version = @schema_version
  WHERE s.data_version = @data_version AND s.silver_schema_version = @schema_version
),
freeze_present AS (
  SELECT DISTINCT shot_event_id
  FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
  WHERE data_version = @data_version AND silver_schema_version = @schema_version
),
defender_present AS (
  SELECT DISTINCT shot_event_id
  FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
  WHERE data_version = @data_version AND silver_schema_version = @schema_version
    AND teammate = FALSE AND player_id IS NOT NULL
)
SELECT
  COUNTIF(sp.period IS NULL OR sp.period != 5) AS total_shots_excl_shootout,
  COUNTIF((sp.period IS NULL OR sp.period != 5) AND fp.shot_event_id IS NOT NULL) AS with_freeze_frame,
  COUNTIF((sp.period IS NULL OR sp.period != 5) AND fp.shot_event_id IS NULL) AS excluded_no_freeze_frame,
  COUNTIF((sp.period IS NULL OR sp.period != 5) AND fp.shot_event_id IS NOT NULL AND dp.shot_event_id IS NOT NULL) AS with_resolvable_defender,
  COUNTIF((sp.period IS NULL OR sp.period != 5) AND fp.shot_event_id IS NOT NULL AND dp.shot_event_id IS NULL) AS excluded_freeze_frame_no_defender,
  COUNTIF(sp.period = 5) AS excluded_shootout_shots
FROM shot_period sp
LEFT JOIN freeze_present fp USING (shot_event_id)
LEFT JOIN defender_present dp USING (shot_event_id)
"""

INVOLVEMENT_QUERY = f"""
WITH defenders AS (
  SELECT
    f.shot_event_id, f.match_id, f.competition_id, f.season_id,
    f.player_id, f.position_id, f.position_name, f.freeze_frame_player_ordinal,
    SQRT(POW(f.x - s.location_x, 2) + POW(f.y - s.location_y, 2)) AS distance
  FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players` f
  JOIN `{PROJECT}.{CORE_DATASET}.shots` s
    ON f.shot_event_id = s.event_id
    AND s.data_version = @data_version AND s.silver_schema_version = @schema_version
  WHERE f.data_version = @data_version AND f.silver_schema_version = @schema_version
    AND f.teammate = FALSE AND f.player_id IS NOT NULL
    AND f.x IS NOT NULL AND f.y IS NOT NULL
    AND s.location_x IS NOT NULL AND s.location_y IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY f.shot_event_id ORDER BY distance ASC, f.freeze_frame_player_ordinal ASC
  ) = 1
)
SELECT
  d.shot_event_id, d.match_id, d.competition_id, d.season_id,
  e.period, e.minute, e.second, e.timestamp,
  d.player_id, d.position_id, d.position_name, d.distance,
  IF(s.team_id = m.home_team_id, m.away_team_id, m.home_team_id) AS defending_team_id,
  s.statsbomb_xg, s.outcome_name
FROM defenders d
JOIN `{PROJECT}.{CORE_DATASET}.shots` s
  ON d.shot_event_id = s.event_id AND s.data_version = @data_version AND s.silver_schema_version = @schema_version
JOIN `{PROJECT}.{CORE_DATASET}.events` e
  ON d.shot_event_id = e.event_id AND e.data_version = @data_version AND e.silver_schema_version = @schema_version
JOIN `{PROJECT}.{CORE_DATASET}.matches` m
  ON d.match_id = m.match_id AND m.data_version = @data_version AND m.silver_schema_version = @schema_version
WHERE e.period IS NOT NULL AND e.period != 5
"""


def main() -> None:
    client = _client()
    job_config = bigquery.QueryJobConfig(query_parameters=_params())

    diagnostics = [
        dict(r.items())
        for r in client.query(DIAGNOSTICS_QUERY, job_config=job_config, location=LOCATION).result()
    ][0]
    EXCLUSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXCLUSIONS_PATH.write_text(json.dumps(diagnostics, indent=2, default=str), encoding="utf8")
    print("involvement diagnostics:", json.dumps(diagnostics))

    rows_out: list[dict] = []
    for r in client.query(INVOLVEMENT_QUERY, job_config=job_config, location=LOCATION).result(
        page_size=20000
    ):
        clock_s = _event_clock_s(r.minute, r.second, r.timestamp)
        rows_out.append(
            {
                "shot_event_id": r.shot_event_id,
                "match_id": r.match_id,
                "competition_id": r.competition_id,
                "season_id": r.season_id,
                "period": r.period,
                "shot_timestamp_seconds": clock_s,
                "player_id": r.player_id,
                "team_id": r.defending_team_id,
                "position_id": r.position_id,
                "position_name": r.position_name,
                "nearest_defender_distance_native": r.distance,
                "statsbomb_xg": r.statsbomb_xg,
                # None (not False) when outcome_name is missing, matching the
                # existing `is_goal` convention in materialize_gold_cxg.py.
                "is_goal": (r.outcome_name == "Goal") if r.outcome_name else None,
                "data_version": DATA_VERSION,
                "silver_schema_version": SCHEMA_VERSION,
                "feature_version": FEATURE_VERSION,
                "materialized_at": datetime.now(UTC).isoformat(),
            }
        )

    schema = [
        bigquery.SchemaField("shot_event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("match_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("competition_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("season_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("period", "INT64"),
        bigquery.SchemaField("shot_timestamp_seconds", "FLOAT64"),
        bigquery.SchemaField("player_id", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("team_id", "INT64"),
        bigquery.SchemaField("position_id", "INT64"),
        bigquery.SchemaField("position_name", "STRING"),
        bigquery.SchemaField("nearest_defender_distance_native", "FLOAT64"),
        bigquery.SchemaField("statsbomb_xg", "FLOAT64"),
        bigquery.SchemaField("is_goal", "BOOL"),
        bigquery.SchemaField("data_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("silver_schema_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("feature_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("materialized_at", "TIMESTAMP", mode="REQUIRED"),
    ]
    job_config_load = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_json(rows_out, TABLE, job_config=job_config_load, location=LOCATION)
    job.result()
    print(json.dumps({"table": TABLE, "rows": len(rows_out), "feature_version": FEATURE_VERSION}))


if __name__ == "__main__":
    main()
