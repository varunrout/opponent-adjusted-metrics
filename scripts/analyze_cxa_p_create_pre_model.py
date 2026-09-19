"""Pre-model target and feature analysis for CxA P_create.

Sits between the CxA feature pipeline (scripts/materialize_cxa_event_v1_training_matrix.py,
materialize_cxa_plus_v1_training_matrix.py) and any future model training. Reads only
`oam_features.cxa_event_v1_training_matrix` / `oam_features.cxa_plus_v1_training_matrix`
(plus `oam_core.competitions` for readable slice labels) -- writes no tables, changes
no BigQuery state. Rubric mirrors the earlier (deleted, pre-methodology-lock) CxA
diagnostic layer's structure -- target usability, target sparsity, per-feature signal,
redundancy, slice stability, leakage/eligibility, modelling recommendations -- not its
code or its data assumptions (that prior attempt ran on an `action_features` table this
project does not have).

Writes one JSON file per analysis section to
`audit_outputs/cxa_analysis/pre_model_study/<section>.json` and prints a summary. The
findings are written up in `docs/analysis/cxa_p_create_pre_model_analysis.md` by hand
from this script's output, not auto-rendered -- this task is a real read of the numbers,
not a template fill.

Does not train, score, or select a model. Does not make a modelling decision.
"""

from __future__ import annotations

import json
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix`"
PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix`"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxa_analysis" / "pre_model_study"

QUERIES: dict[str, str] = {
    "target_cross_tab": f"""
        SELECT 'cxa_event_v1' AS track, y_create, y_goal, COUNT(*) n FROM {EVENT_MATRIX} GROUP BY 1,2,3
        UNION ALL
        SELECT 'cxa_plus_v1' AS track, y_create, y_goal, COUNT(*) n FROM {PLUS_MATRIX} GROUP BY 1,2,3
        ORDER BY track, y_create, y_goal
    """,
    "anomaly_row_event": f"""
        SELECT pass_event_id, match_id, competition_id, season_id, pass_outcome_name, is_completed,
               y_create, y_goal, passer_team_id, passer_player_id, start_x, start_y, end_x, end_y,
               minute, second, possession_id
        FROM {EVENT_MATRIX}
        WHERE is_completed = false AND y_create = true
    """,
    "sparsity_by_competition_event": f"""
        SELECT c.competition_name, c.competition_id, m.season_id, COUNT(*) n, COUNTIF(m.y_create) n_create,
               ROUND(COUNTIF(m.y_create)/COUNT(*)*100,3) create_pct, COUNT(DISTINCT m.match_id) n_matches
        FROM {EVENT_MATRIX} m
        JOIN `{PROJECT}.{CORE_DATASET}.competitions` c
          ON c.competition_id = m.competition_id AND c.season_id = m.season_id
          AND c.silver_schema_version = '{SCHEMA_VERSION}'
        GROUP BY 1,2,3 ORDER BY 1,3
    """,
    "sparsity_by_competition_plus": f"""
        SELECT c.competition_name, m.competition_id, m.season_id, COUNT(*) n, COUNTIF(m.y_create) n_create,
               ROUND(COUNTIF(m.y_create)/COUNT(*)*100,3) create_pct, COUNT(DISTINCT m.match_id) n_matches
        FROM {PLUS_MATRIX} m
        JOIN `{PROJECT}.{CORE_DATASET}.competitions` c
          ON c.competition_id = m.competition_id AND c.season_id = m.season_id
          AND c.silver_schema_version = '{SCHEMA_VERSION}'
        GROUP BY 1,2,3 ORDER BY 1,3
    """,
    "sparsity_by_match_event": f"""
        WITH per_match AS (
          SELECT match_id, COUNT(*) n, COUNTIF(y_create) n_create FROM {EVENT_MATRIX} GROUP BY match_id
        )
        SELECT COUNT(*) total_matches, COUNTIF(n_create=0) matches_zero,
               COUNTIF(n_create BETWEEN 1 AND 5) matches_1_5, MIN(n_create) min_c, MAX(n_create) max_c,
               APPROX_QUANTILES(n_create,4) q
        FROM per_match
    """,
    "sparsity_by_match_plus": f"""
        WITH per_match AS (
          SELECT match_id, COUNT(*) n, COUNTIF(y_create) n_create FROM {PLUS_MATRIX} GROUP BY match_id
        )
        SELECT COUNT(*) total_matches, COUNTIF(n_create=0) matches_zero,
               COUNTIF(n_create BETWEEN 1 AND 5) matches_1_5, MIN(n_create) min_c, MAX(n_create) max_c,
               APPROX_QUANTILES(n_create,4) q
        FROM per_match
    """,
    "missingness_event": f"""
        SELECT COUNT(*) n,
          COUNTIF(pass_length IS NULL) null_length, COUNTIF(pass_angle IS NULL) null_angle,
          COUNTIF(start_x IS NULL) null_start_x, COUNTIF(start_y IS NULL) null_start_y,
          COUNTIF(end_x IS NULL) null_end_x, COUNTIF(end_y IS NULL) null_end_y,
          COUNTIF(pass_height_name IS NULL) null_height, COUNTIF(pass_type_name IS NULL) null_type,
          COUNTIF(pass_technique_name IS NULL) null_technique, COUNTIF(pass_body_part_name IS NULL) null_bodypart,
          COUNTIF(play_pattern_name IS NULL) null_playpattern, COUNTIF(minute IS NULL) null_minute,
          COUNTIF(second IS NULL) null_second, COUNTIF(possession_id IS NULL) null_possession
        FROM {EVENT_MATRIX}
    """,
    "numeric_signal_event": f"""
        SELECT y_create, COUNT(*) n,
          ROUND(AVG(pass_length),2) avg_length, ROUND(APPROX_QUANTILES(pass_length,2)[OFFSET(1)],2) med_length,
          ROUND(AVG(ABS(pass_angle)),3) avg_abs_angle,
          ROUND(AVG(start_x),2) avg_start_x, ROUND(AVG(start_y),2) avg_start_y,
          ROUND(AVG(end_x),2) avg_end_x, ROUND(AVG(end_y),2) avg_end_y,
          ROUND(AVG(minute),2) avg_minute,
          ROUND(AVG(CAST(is_through_ball AS INT64))*100,3) pct_through_ball,
          ROUND(AVG(CAST(is_switch AS INT64))*100,3) pct_switch,
          ROUND(AVG(CAST(is_cross AS INT64))*100,3) pct_cross,
          ROUND(AVG(CAST(is_cut_back AS INT64))*100,3) pct_cutback
        FROM {EVENT_MATRIX} GROUP BY y_create ORDER BY y_create
    """,
    "categorical_signal_height": f"""
        SELECT pass_height_name, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,3) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 4 DESC
    """,
    "categorical_signal_technique": f"""
        SELECT COALESCE(pass_technique_name,'(null=regular)') technique, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,3) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 4 DESC
    """,
    "categorical_signal_bodypart": f"""
        SELECT COALESCE(pass_body_part_name,'(null)') bodypart, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,3) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 4 DESC
    """,
    "categorical_signal_playpattern": f"""
        SELECT play_pattern_name, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,3) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 4 DESC
    """,
    "categorical_signal_passtype": f"""
        SELECT COALESCE(pass_type_name,'(null=open play)') ptype, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,3) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 4 DESC
    """,
    "redundancy_event": f"""
        SELECT
          ROUND(CORR(start_x, end_x),4) corr_startx_endx, ROUND(CORR(start_y, end_y),4) corr_starty_endy,
          ROUND(CORR(pass_length, end_x),4) corr_length_endx, ROUND(CORR(pass_length, start_x),4) corr_length_startx,
          ROUND(CORR(pass_angle, start_y),4) corr_angle_starty, ROUND(CORR(pass_angle, end_y),4) corr_angle_endy,
          ROUND(CORR(start_x, start_y),4) corr_startx_starty, ROUND(CORR(end_x, end_y),4) corr_endx_endy,
          ROUND(CORR(pass_length, pass_angle),4) corr_length_angle
        FROM {EVENT_MATRIX}
    """,
    "redundancy_plus_receiver_vs_end": f"""
        SELECT ROUND(CORR(end_x, receiver_x),4) corr_endx_receiverx,
               ROUND(CORR(end_y, receiver_y),4) corr_endy_receivery,
               COUNTIF(receiver_x IS NULL) null_receiver_x,
               COUNTIF(reception_nearest_opponent_distance_m IS NULL) null_nearest_dist,
               COUNT(*) n
        FROM {PLUS_MATRIX}
    """,
    "signal_360_plus": f"""
        SELECT y_create, COUNT(*) n,
          COUNTIF(reception_nearest_opponent_distance_m IS NULL) null_nearest,
          ROUND(AVG(reception_nearest_opponent_distance_m),3) avg_nearest_opp_dist,
          ROUND(APPROX_QUANTILES(reception_nearest_opponent_distance_m,2)[OFFSET(1)],3) med_nearest_opp_dist,
          ROUND(AVG(reception_opponents_within_5m),3) avg_opp_5m,
          ROUND(AVG(reception_opponents_within_8m),3) avg_opp_8m,
          ROUND(AVG(reception_teammates_visible),3) avg_teammates_visible,
          ROUND(AVG(reception_opponents_visible),3) avg_opponents_visible,
          ROUND(AVG(reception_frame_player_count),3) avg_frame_player_count
        FROM {PLUS_MATRIX} GROUP BY y_create ORDER BY y_create
    """,
    "slice_stability_event": f"""
        SELECT competition_id,
          ROUND(AVG(CASE WHEN y_create THEN start_x END),2) avg_startx_create,
          ROUND(AVG(CASE WHEN NOT y_create THEN start_x END),2) avg_startx_nocreate,
          ROUND(AVG(CASE WHEN y_create THEN CAST(is_cross AS INT64) END)*100,2) pct_cross_create,
          ROUND(AVG(CASE WHEN NOT y_create THEN CAST(is_cross AS INT64) END)*100,2) pct_cross_nocreate,
          ROUND(AVG(CASE WHEN y_create THEN CAST(is_through_ball AS INT64) END)*100,2) pct_tb_create,
          ROUND(AVG(CASE WHEN NOT y_create THEN CAST(is_through_ball AS INT64) END)*100,2) pct_tb_nocreate
        FROM {EVENT_MATRIX} GROUP BY 1 ORDER BY 1
    """,
    "slice_stability_plus": f"""
        SELECT competition_id,
          ROUND(AVG(CASE WHEN y_create THEN reception_nearest_opponent_distance_m END),2) dist_create,
          ROUND(AVG(CASE WHEN NOT y_create THEN reception_nearest_opponent_distance_m END),2) dist_nocreate,
          ROUND(AVG(CASE WHEN y_create THEN reception_opponents_within_5m END),2) opp5_create,
          ROUND(AVG(CASE WHEN NOT y_create THEN reception_opponents_within_5m END),2) opp5_nocreate
        FROM {PLUS_MATRIX} GROUP BY 1 ORDER BY 1
    """,
    "leakage_is_completed_vs_ycreate": f"""
        SELECT is_completed, COUNT(*) n, COUNTIF(y_create) n_create,
               ROUND(COUNTIF(y_create)/COUNT(*)*100,4) create_pct
        FROM {EVENT_MATRIX} GROUP BY 1
    """,
}


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    for name, sql in QUERIES.items():
        rows = [dict(r.items()) for r in client.query(sql, location=LOCATION).result()]
        (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(rows, indent=2, default=str))
        summary[name] = len(rows)
        print(f"{name}: {len(rows)} row(s) -> {OUTPUT_DIR / f'{name}.json'}")
    print(json.dumps({"sections_written": summary}, indent=2))


if __name__ == "__main__":
    main()
