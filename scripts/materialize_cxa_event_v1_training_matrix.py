"""Materialize the CxA (event-only) P_create training matrix.

One row per pass in `oam_core.passes` (silver_schema_version-filtered per the CxA
feasibility audit; see docs/cxa_data_feasibility_audit.md). Completed and non-completed
passes are both kept -- a non-completed pass cannot have created a chance, so it is a
legitimate Y_create=FALSE row, not a leak or a row to drop.

Y_create is derived from `shots.key_pass_id` (NOT `passes.shot_assist`, which
undercounts by ~9%, see audit point 3). Y_goal is a secondary/illustrative label for a
later P_convert stage and is not used to select or evaluate P_create.

Features are strictly pre-reception: pass geometry (own columns on `oam_core.passes`)
and possession/phase context (joined from `oam_core.events`, since `passes` itself
carries neither `minute`/`second`/`possession_id`/`play_pattern_name` nor
`start_x`/`start_y`). Receiver identity and receiving-team identity are intentionally
excluded from the SELECT list -- fold-safe nuisance attributes only, never published
model features (same rule CxG applied to shooter/defending-team identity).

Writes via CREATE OR REPLACE TABLE (full rebuild each run; this is a pure derived
feature table with no manually-inserted rows to lose, same class of table as CxG's
`cxg_analysis_event_v1` surface rebuilds).
"""

from __future__ import annotations

import json

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
FEATURE_VERSION = "cxa_event_v1"

TABLE = f"{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix"

SQL = f"""
CREATE OR REPLACE TABLE `{TABLE}` AS
WITH base AS (
  SELECT
    e.event_id AS pass_event_id,
    e.match_id,
    e.competition_id,
    e.season_id,
    e.period,
    e.minute,
    e.second,
    e.possession_id,
    e.play_pattern_name,
    e.team_id AS passer_team_id,
    e.player_id AS passer_player_id,
    e.location_x AS start_x,
    e.location_y AS start_y,
    e.related_event_ids,
    p.length AS pass_length,
    p.angle AS pass_angle,
    p.end_x,
    p.end_y,
    p.height_name AS pass_height_name,
    p.pass_type_name,
    p.technique_name AS pass_technique_name,
    p.body_part_name AS pass_body_part_name,
    COALESCE(p.through_ball, FALSE) AS is_through_ball,
    COALESCE(p.switch, FALSE) AS is_switch,
    COALESCE(p.cross, FALSE) AS is_cross,
    COALESCE(p.cut_back, FALSE) AS is_cut_back,
    p.outcome_name AS pass_outcome_name,
    p.outcome_name IS NULL AS is_completed
  FROM `{PROJECT}.{CORE_DATASET}.events` e
  JOIN `{PROJECT}.{CORE_DATASET}.passes` p
    ON p.event_id = e.event_id AND p.silver_schema_version = e.silver_schema_version
  WHERE e.silver_schema_version = '{SCHEMA_VERSION}'
),
labeled AS (
  SELECT
    b.* EXCEPT (related_event_ids),
    s.event_id IS NOT NULL AS y_create,
    COALESCE(s.outcome_name = 'Goal', FALSE) AS y_goal
  FROM base b
  LEFT JOIN `{PROJECT}.{CORE_DATASET}.shots` s
    ON s.key_pass_id = b.pass_event_id AND s.silver_schema_version = '{SCHEMA_VERSION}'
)
SELECT
  *,
  '{DATA_VERSION}' AS data_version,
  '{SCHEMA_VERSION}' AS silver_schema_version,
  '{FEATURE_VERSION}' AS feature_version,
  CURRENT_TIMESTAMP() AS materialized_at
FROM labeled
"""

ROW_COUNT_SQL = f"SELECT COUNT(*) AS n, COUNTIF(y_create) AS n_create FROM `{TABLE}`"


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    job = client.query(SQL, location=LOCATION)
    job.result()
    counts = list(client.query(ROW_COUNT_SQL, location=LOCATION).result())[0]
    print(
        json.dumps(
            {
                "table": TABLE,
                "rows": counts["n"],
                "y_create_positive": counts["n_create"],
                "feature_version": FEATURE_VERSION,
            }
        )
    )


if __name__ == "__main__":
    main()
