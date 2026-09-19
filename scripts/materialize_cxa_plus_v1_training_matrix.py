"""Materialize the CxA+ (event + 360-at-reception) P_create training matrix.

Same base population/features as `materialize_cxa_event_v1_training_matrix.py`,
restricted to completed passes whose reception boundary resolves to a real
`ball_receipts` row via `events.related_event_ids` (never `event_index + 1` -- see
docs/cxa_data_feasibility_audit.md point 4 for why that is a leakage/correctness trap)
AND whose receipt event has a StatsBomb 360 frame.

360 features are joined at `(match_id, event_uuid = receipt_event_id)` -- the frame
captured AT the reception event, never a later one. Because the frame belongs to the
Ball Receipt event itself, `three_sixty_players.teammate`/`actor` are already oriented
relative to the receiver (the frame's own actor) -- no CxG-style orient_players()
re-expression is needed here (CxG's frames are sometimes captured at a different event
than the shot being scored).

v1 ships direction-independent reception-pressure features only (nearest-opponent
distance, opponent counts within radius bands, frame player counts). Goal-relative
features (visible goal angle at reception, defenders between receiver and goal) are
deferred -- this codebase has no validated attacking-direction convention for a pass
reception (CxG's `GOAL_X = 120.0` is documented as a shot-geometry-specific
convention), and guessing one is exactly the kind of unvalidated assumption the
feasibility audit exists to prevent.

Distance-band features reuse CxG's documented approximate-metres bridge
(`NATIVE_TO_METRE_X/Y` under an assumed 105x68 standard pitch, see
`opponent_adjusted.features.cxg.three_sixty_frame.DISTANCE_UNIT_CONTRACT`) rather than
inventing a second, undocumented distance convention.
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
FEATURE_VERSION = "cxa_plus_v1"

# Same approximate-metres bridge CxG documents in three_sixty_frame.py.
NATIVE_TO_METRE_X = 105.0 / 120.0
NATIVE_TO_METRE_Y = 68.0 / 80.0

TABLE = f"{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix"

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
    COALESCE(p.cut_back, FALSE) AS is_cut_back
  FROM `{PROJECT}.{CORE_DATASET}.events` e
  JOIN `{PROJECT}.{CORE_DATASET}.passes` p
    ON p.event_id = e.event_id AND p.silver_schema_version = e.silver_schema_version
  WHERE e.silver_schema_version = '{SCHEMA_VERSION}'
    AND p.outcome_name IS NULL  -- completed passes only: a receipt boundary requires a reception
),
receipt_link AS (
  -- one row per (pass_event_id, candidate related event), then narrowed to the one
  -- that is a genuine ball_receipts row for that same match
  SELECT
    b.pass_event_id,
    b.match_id,
    br.event_id AS receipt_event_id
  FROM base b, UNNEST(b.related_event_ids) AS rel_id
  JOIN `{PROJECT}.{CORE_DATASET}.ball_receipts` br
    ON br.event_id = rel_id
    AND br.match_id = b.match_id
    AND br.silver_schema_version = '{SCHEMA_VERSION}'
),
receipt_with_frame AS (
  -- a pass's related_event_ids should resolve to exactly one ball_receipts row, but
  -- dedupe defensively (keep the lowest event_id deterministically) so this join can
  -- never fan out the one-row-per-pass grain if StatsBomb ever links more than one.
  SELECT
    rl.pass_event_id,
    rl.receipt_event_id,
    rl.match_id,
    f.frame_player_count AS reception_frame_player_count
  FROM receipt_link rl
  JOIN `{PROJECT}.{CORE_DATASET}.three_sixty_frames` f
    ON f.event_uuid = rl.receipt_event_id
    AND f.match_id = rl.match_id
    AND f.silver_schema_version = '{SCHEMA_VERSION}'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY rl.pass_event_id ORDER BY rl.receipt_event_id) = 1
),
frame_players AS (
  SELECT match_id, event_uuid AS receipt_event_id, teammate, actor, keeper, x, y
  FROM `{PROJECT}.{CORE_DATASET}.three_sixty_players`
  WHERE silver_schema_version = '{SCHEMA_VERSION}'
),
receiver_pos AS (
  SELECT match_id, receipt_event_id, x AS receiver_x, y AS receiver_y
  FROM frame_players
  WHERE actor = TRUE
  QUALIFY ROW_NUMBER() OVER (PARTITION BY match_id, receipt_event_id ORDER BY receipt_event_id) = 1
),
opponent_geometry AS (
  SELECT
    rp.match_id,
    rp.receipt_event_id,
    rp.receiver_x,
    rp.receiver_y,
    MIN(SQRT(
      POW((fp.x - rp.receiver_x) * {NATIVE_TO_METRE_X}, 2)
      + POW((fp.y - rp.receiver_y) * {NATIVE_TO_METRE_Y}, 2)
    )) AS reception_nearest_opponent_distance_m,
    COUNTIF(SQRT(
      POW((fp.x - rp.receiver_x) * {NATIVE_TO_METRE_X}, 2)
      + POW((fp.y - rp.receiver_y) * {NATIVE_TO_METRE_Y}, 2)
    ) <= 5.0) AS reception_opponents_within_5m,
    COUNTIF(SQRT(
      POW((fp.x - rp.receiver_x) * {NATIVE_TO_METRE_X}, 2)
      + POW((fp.y - rp.receiver_y) * {NATIVE_TO_METRE_Y}, 2)
    ) <= 8.0) AS reception_opponents_within_8m,
    COUNT(*) AS reception_opponents_visible
  FROM receiver_pos rp
  JOIN frame_players fp
    ON fp.match_id = rp.match_id
    AND fp.receipt_event_id = rp.receipt_event_id
    AND fp.teammate = FALSE
    AND fp.x IS NOT NULL AND fp.y IS NOT NULL
  GROUP BY 1, 2, 3, 4
),
teammate_counts AS (
  SELECT match_id, receipt_event_id, COUNT(*) AS reception_teammates_visible
  FROM frame_players
  WHERE teammate = TRUE
  GROUP BY 1, 2
),
plus_rows AS (
  SELECT
    b.*,
    rwf.receipt_event_id,
    rwf.reception_frame_player_count,
    og.receiver_x,
    og.receiver_y,
    og.reception_nearest_opponent_distance_m,
    og.reception_opponents_within_5m,
    og.reception_opponents_within_8m,
    COALESCE(og.reception_opponents_visible, 0) AS reception_opponents_visible,
    COALESCE(tc.reception_teammates_visible, 0) AS reception_teammates_visible
  FROM base b
  JOIN receipt_with_frame rwf ON rwf.pass_event_id = b.pass_event_id
  LEFT JOIN opponent_geometry og
    ON og.match_id = rwf.match_id AND og.receipt_event_id = rwf.receipt_event_id
  LEFT JOIN teammate_counts tc
    ON tc.match_id = rwf.match_id AND tc.receipt_event_id = rwf.receipt_event_id
),
labeled AS (
  SELECT
    pr.* EXCEPT (related_event_ids),
    s.event_id IS NOT NULL AS y_create,
    COALESCE(s.outcome_name = 'Goal', FALSE) AS y_goal
  FROM plus_rows pr
  LEFT JOIN `{PROJECT}.{CORE_DATASET}.shots` s
    ON s.key_pass_id = pr.pass_event_id AND s.silver_schema_version = '{SCHEMA_VERSION}'
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
