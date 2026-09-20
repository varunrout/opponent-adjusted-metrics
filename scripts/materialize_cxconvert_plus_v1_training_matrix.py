"""Materialize the CxA+ P_convert training matrix.

Same logic as materialize_cxconvert_event_v1_training_matrix.py (see that script's
docstring for the shared design: 1:1 key_pass_id->shot join, y_goal definition, shot
freeze-frame GK/defender features, leakage exclusions), applied to
`cxa_plus_v1_training_matrix`'s `y_create = TRUE` population instead of the event-only
one. Additionally carries over the reception-time 360 context P_create's CxA+ pipeline
already computed (`receipt_event_id`, `reception_frame_player_count`, `receiver_x`,
`receiver_y`, `reception_nearest_opponent_distance_m`,
`reception_opponents_within_5m`, `reception_opponents_within_8m`,
`reception_opponents_visible`, `reception_teammates_visible`) -- a different moment
(reception) than the shot-freeze-frame features (the shot itself), complementary, not
redundant, both carried unfiltered.
"""

from __future__ import annotations

import json

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURES_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
DATA_VERSION = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"
SCHEMA_VERSION = "statsbomb_silver_v1_2"
FEATURE_VERSION = "cxconvert_plus_v1"

NATIVE_TO_METRE_X = 105.0 / 120.0
NATIVE_TO_METRE_Y = 68.0 / 80.0

TABLE = f"{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix"

SQL = f"""
CREATE OR REPLACE TABLE `{TABLE}` AS
WITH creates AS (
  SELECT
    pass_event_id, match_id, competition_id, season_id, period, minute, second,
    possession_id, play_pattern_name, passer_team_id, passer_player_id,
    start_x, start_y,
    pass_length, pass_angle,
    end_x AS pass_end_x, end_y AS pass_end_y,
    pass_height_name, pass_type_name, pass_technique_name, pass_body_part_name,
    is_through_ball, is_switch, is_cross, is_cut_back,
    receipt_event_id, reception_frame_player_count, receiver_x, receiver_y,
    reception_nearest_opponent_distance_m, reception_opponents_within_5m,
    reception_opponents_within_8m, reception_opponents_visible, reception_teammates_visible
  FROM `{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix`
  WHERE y_create = TRUE
),
shot_base AS (
  SELECT
    c.*,
    s.event_id AS shot_event_id,
    s.location_x AS shot_x_sb,
    s.location_y AS shot_y_sb,
    s.end_x AS shot_end_x,
    s.end_y AS shot_end_y,
    s.end_z AS shot_end_z,
    s.body_part_name AS shot_body_part_name,
    s.technique_name AS shot_technique_name,
    s.shot_type_name AS shot_type_name,
    s.statsbomb_xg AS statsbomb_xg,
    COALESCE(s.first_time, FALSE) AS shot_first_time,
    COALESCE(s.aerial_won, FALSE) AS shot_aerial_won,
    COALESCE(s.follows_dribble, FALSE) AS shot_follows_dribble,
    COALESCE(s.open_goal, FALSE) AS shot_open_goal,
    COALESCE(s.one_on_one, FALSE) AS shot_one_on_one,
    (s.outcome_name = 'Goal') AS y_goal
  FROM creates c
  JOIN `{PROJECT}.{CORE_DATASET}.shots` s
    ON s.key_pass_id = c.pass_event_id AND s.silver_schema_version = '{SCHEMA_VERSION}'
),
shot_events AS (
  SELECT
    event_id AS shot_event_id,
    minute AS shot_minute,
    second AS shot_second,
    COALESCE(under_pressure, FALSE) AS shot_under_pressure,
    COALESCE(counterpress, FALSE) AS shot_counterpress
  FROM `{PROJECT}.{CORE_DATASET}.events`
  WHERE silver_schema_version = '{SCHEMA_VERSION}'
),
freeze_players AS (
  SELECT match_id, shot_event_id, teammate, position_name, x, y
  FROM `{PROJECT}.{CORE_DATASET}.shot_freeze_frame_players`
  WHERE silver_schema_version = '{SCHEMA_VERSION}'
),
gk AS (
  SELECT match_id, shot_event_id, x AS shot_gk_x, y AS shot_gk_y
  FROM freeze_players
  WHERE teammate = FALSE AND position_name = 'Goalkeeper'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY match_id, shot_event_id ORDER BY shot_event_id) = 1
),
frame_counts AS (
  SELECT match_id, shot_event_id,
    COUNT(*) AS shot_frame_player_count,
    COUNTIF(teammate = FALSE) AS shot_defenders_visible
  FROM freeze_players
  GROUP BY 1, 2
),
defender_geometry AS (
  SELECT sb.pass_event_id,
    COUNTIF(fp.teammate = FALSE AND SQRT(
      POW((fp.x - sb.shot_x_sb) * {NATIVE_TO_METRE_X}, 2)
      + POW((fp.y - sb.shot_y_sb) * {NATIVE_TO_METRE_Y}, 2)
    ) <= 5.0) AS shot_defenders_within_5m,
    COUNTIF(fp.teammate = FALSE AND SQRT(
      POW((fp.x - sb.shot_x_sb) * {NATIVE_TO_METRE_X}, 2)
      + POW((fp.y - sb.shot_y_sb) * {NATIVE_TO_METRE_Y}, 2)
    ) <= 8.0) AS shot_defenders_within_8m
  FROM shot_base sb
  JOIN freeze_players fp ON fp.match_id = sb.match_id AND fp.shot_event_id = sb.shot_event_id
  GROUP BY 1
)
SELECT
  sb.* EXCEPT (shot_event_id),
  sb.shot_event_id,
  se.shot_minute, se.shot_second, se.shot_under_pressure, se.shot_counterpress,
  gk.shot_gk_x, gk.shot_gk_y,
  SAFE.SQRT(
    POW((gk.shot_gk_x - sb.shot_x_sb) * {NATIVE_TO_METRE_X}, 2)
    + POW((gk.shot_gk_y - sb.shot_y_sb) * {NATIVE_TO_METRE_Y}, 2)
  ) AS shot_gk_distance_m,
  fc.shot_frame_player_count,
  fc.shot_defenders_visible,
  dg.shot_defenders_within_5m,
  dg.shot_defenders_within_8m,
  m.split,
  '{DATA_VERSION}' AS data_version,
  '{SCHEMA_VERSION}' AS silver_schema_version,
  '{FEATURE_VERSION}' AS feature_version,
  CURRENT_TIMESTAMP() AS materialized_at
FROM shot_base sb
LEFT JOIN shot_events se ON se.shot_event_id = sb.shot_event_id
LEFT JOIN gk ON gk.match_id = sb.match_id AND gk.shot_event_id = sb.shot_event_id
LEFT JOIN frame_counts fc ON fc.match_id = sb.match_id AND fc.shot_event_id = sb.shot_event_id
LEFT JOIN defender_geometry dg ON dg.pass_event_id = sb.pass_event_id
JOIN `{PROJECT}.{ANALYSIS_DATASET}.cxa_match_splits_v1` m ON m.match_id = sb.match_id
"""

ROW_COUNT_SQL = f"SELECT COUNT(*) AS n, COUNTIF(y_goal) AS n_goal FROM `{TABLE}`"


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    client.query(SQL, location=LOCATION).result()
    counts = list(client.query(ROW_COUNT_SQL, location=LOCATION).result())[0]
    print(
        json.dumps(
            {
                "table": TABLE,
                "rows": counts["n"],
                "y_goal_positive": counts["n_goal"],
                "feature_version": FEATURE_VERSION,
            }
        )
    )


if __name__ == "__main__":
    main()
