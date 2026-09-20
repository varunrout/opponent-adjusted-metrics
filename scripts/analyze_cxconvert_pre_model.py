"""Pre-model target and feature analysis for CxA P_convert.

Sits between the P_convert feature pipeline
(scripts/materialize_cxconvert_event_v1_training_matrix.py,
materialize_cxconvert_plus_v1_training_matrix.py) and any future model training. Reads
only `oam_features.cxconvert_event_v1_training_matrix` /
`cxconvert_plus_v1_training_matrix` (plus `oam_core.competitions` for readable slice
labels) -- writes no tables, changes no BigQuery state. Rubric mirrors
`docs/analysis/cxa_p_create_pre_model_analysis.md` (P_create's own pre-model analysis)
exactly: target usability/sparsity, per-feature signal, thin-support re-check,
redundancy, leakage re-confirmation, modelling recommendations.

Writes one JSON file per analysis section to
`audit_outputs/cxconvert_analysis/pre_model_study/<section>.json`. The findings are
written up by hand in `docs/analysis/cxa_p_convert_pre_model_analysis.md` from this
script's output, not auto-rendered.

Full-population exploratory analysis only -- no train/validation split-awareness (the
split policy's own "full-population EDA is valid exploratory work" carve-out). Does
not train, score, or select a model.
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

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxconvert_event_v1_training_matrix`"
PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix`"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "pre_model_study"


def cat_rate_sql(matrix: str, col: str) -> str:
    return f"""
        SELECT {col} AS level, COUNT(*) n, COUNTIF(y_goal) n_goal,
               ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {matrix} GROUP BY 1 ORDER BY n DESC
    """


def bool_rate_sql(matrix: str, col: str) -> str:
    return f"""
        SELECT {col} AS level, COUNT(*) n, COUNTIF(y_goal) n_goal,
               ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {matrix} GROUP BY 1 ORDER BY level
    """


QUERIES: dict[str, str] = {
    # --- 1. target analysis, both tracks ---
    "target_overall": f"""
        SELECT 'event' track, COUNT(*) n, COUNTIF(y_goal) n_goal, ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct FROM {EVENT_MATRIX}
        UNION ALL
        SELECT 'plus' track, COUNT(*) n, COUNTIF(y_goal) n_goal, ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct FROM {PLUS_MATRIX}
    """,
    "target_by_shot_body_part_event": cat_rate_sql(EVENT_MATRIX, "shot_body_part_name"),
    "target_by_shot_body_part_plus": cat_rate_sql(PLUS_MATRIX, "shot_body_part_name"),
    "target_by_shot_technique_event": cat_rate_sql(EVENT_MATRIX, "shot_technique_name"),
    "target_by_shot_technique_plus": cat_rate_sql(PLUS_MATRIX, "shot_technique_name"),
    "target_by_pass_type_event": cat_rate_sql(EVENT_MATRIX, "pass_type_name"),
    "target_by_pass_type_plus": cat_rate_sql(PLUS_MATRIX, "pass_type_name"),
    "target_by_play_pattern_event": cat_rate_sql(EVENT_MATRIX, "play_pattern_name"),
    "target_by_play_pattern_plus": cat_rate_sql(PLUS_MATRIX, "play_pattern_name"),
    "target_by_competition_event": f"""
        SELECT c.competition_name, m.competition_id, m.season_id, COUNT(*) n, COUNTIF(m.y_goal) n_goal,
               ROUND(COUNTIF(m.y_goal)/COUNT(*)*100,3) goal_pct, COUNT(DISTINCT m.match_id) n_matches
        FROM {EVENT_MATRIX} m
        JOIN `{PROJECT}.{CORE_DATASET}.competitions` c
          ON c.competition_id = m.competition_id AND c.season_id = m.season_id
          AND c.silver_schema_version = '{SCHEMA_VERSION}'
        GROUP BY 1,2,3 ORDER BY 1,3
    """,
    "target_by_competition_plus": f"""
        SELECT c.competition_name, m.competition_id, m.season_id, COUNT(*) n, COUNTIF(m.y_goal) n_goal,
               ROUND(COUNTIF(m.y_goal)/COUNT(*)*100,3) goal_pct, COUNT(DISTINCT m.match_id) n_matches
        FROM {PLUS_MATRIX} m
        JOIN `{PROJECT}.{CORE_DATASET}.competitions` c
          ON c.competition_id = m.competition_id AND c.season_id = m.season_id
          AND c.silver_schema_version = '{SCHEMA_VERSION}'
        GROUP BY 1,2,3 ORDER BY 1,3
    """,
    "sparsity_by_match_event": f"""
        WITH per_match AS (SELECT match_id, COUNT(*) n, COUNTIF(y_goal) n_goal FROM {EVENT_MATRIX} GROUP BY match_id)
        SELECT COUNT(*) total_matches, COUNTIF(n_goal=0) matches_zero, MIN(n_goal) min_g, MAX(n_goal) max_g,
               APPROX_QUANTILES(n_goal,4) q, APPROX_QUANTILES(n,4) q_rows
        FROM per_match
    """,
    "sparsity_by_match_plus": f"""
        WITH per_match AS (SELECT match_id, COUNT(*) n, COUNTIF(y_goal) n_goal FROM {PLUS_MATRIX} GROUP BY match_id)
        SELECT COUNT(*) total_matches, COUNTIF(n_goal=0) matches_zero, MIN(n_goal) min_g, MAX(n_goal) max_g,
               APPROX_QUANTILES(n_goal,4) q, APPROX_QUANTILES(n,4) q_rows
        FROM per_match
    """,
    # --- 2. univariate numeric signal ---
    "numeric_signal_event": f"""
        SELECT y_goal, COUNT(*) n,
          ROUND(AVG(start_x),3) avg_start_x, ROUND(AVG(start_y),3) avg_start_y,
          ROUND(AVG(pass_end_x),3) avg_pass_end_x, ROUND(AVG(pass_end_y),3) avg_pass_end_y,
          ROUND(AVG(pass_length),3) avg_pass_length, ROUND(AVG(ABS(pass_angle)),3) avg_abs_pass_angle,
          ROUND(AVG(minute),3) avg_minute, ROUND(AVG(shot_x_sb),3) avg_shot_x, ROUND(AVG(shot_y_sb),3) avg_shot_y,
          ROUND(AVG(ABS(shot_y_sb-40)),3) avg_shot_abs_y_from_center,
          ROUND(AVG(shot_end_x),3) avg_shot_end_x, ROUND(AVG(shot_end_y),3) avg_shot_end_y,
          ROUND(AVG(shot_end_z),3) avg_shot_end_z, COUNTIF(shot_end_z IS NULL) null_end_z,
          ROUND(AVG(statsbomb_xg),4) avg_statsbomb_xg,
          ROUND(AVG(shot_gk_distance_m),3) avg_gk_distance_m,
          ROUND(AVG(shot_defenders_within_5m),3) avg_defenders_5m,
          ROUND(AVG(shot_defenders_within_8m),3) avg_defenders_8m,
          ROUND(AVG(shot_defenders_visible),3) avg_defenders_visible,
          ROUND(AVG(shot_frame_player_count),3) avg_frame_player_count
        FROM {EVENT_MATRIX} GROUP BY y_goal ORDER BY y_goal
    """,
    "numeric_signal_event_dist_to_goal": f"""
        SELECT y_goal, COUNT(*) n,
          ROUND(AVG(SQRT(POW(120-shot_x_sb,2)+POW(40-shot_y_sb,2))),3) avg_dist_to_goal_center_native,
          ROUND(AVG(SQRT(POW(120-shot_x_sb,2)+POW(40-shot_y_sb,2))*(105.0/120.0)),3) avg_dist_to_goal_center_m
        FROM {EVENT_MATRIX} GROUP BY y_goal ORDER BY y_goal
    """,
    "numeric_signal_plus": f"""
        SELECT y_goal, COUNT(*) n,
          ROUND(AVG(start_x),3) avg_start_x, ROUND(AVG(pass_end_x),3) avg_pass_end_x,
          ROUND(AVG(pass_length),3) avg_pass_length,
          ROUND(AVG(reception_nearest_opponent_distance_m),3) avg_reception_nearest_opp_m,
          ROUND(AVG(reception_opponents_within_5m),3) avg_reception_opp_5m,
          ROUND(AVG(reception_opponents_within_8m),3) avg_reception_opp_8m,
          ROUND(AVG(reception_opponents_visible),3) avg_reception_opp_visible,
          ROUND(AVG(reception_teammates_visible),3) avg_reception_teammates_visible,
          ROUND(AVG(reception_frame_player_count),3) avg_reception_frame_count,
          ROUND(AVG(shot_x_sb),3) avg_shot_x, ROUND(AVG(shot_end_z),3) avg_shot_end_z,
          COUNTIF(shot_end_z IS NULL) null_end_z,
          ROUND(AVG(statsbomb_xg),4) avg_statsbomb_xg,
          ROUND(AVG(shot_gk_distance_m),3) avg_gk_distance_m,
          ROUND(AVG(shot_defenders_within_5m),3) avg_defenders_5m,
          ROUND(AVG(shot_defenders_within_8m),3) avg_defenders_8m
        FROM {PLUS_MATRIX} GROUP BY y_goal ORDER BY y_goal
    """,
    # --- categorical / boolean signal, event track ---
    "cat_pass_height_event": cat_rate_sql(EVENT_MATRIX, "pass_height_name"),
    "cat_pass_type_event": cat_rate_sql(EVENT_MATRIX, "pass_type_name"),
    "cat_pass_technique_event": cat_rate_sql(EVENT_MATRIX, "pass_technique_name"),
    "cat_pass_body_part_event": cat_rate_sql(EVENT_MATRIX, "pass_body_part_name"),
    "bool_is_through_ball_event": bool_rate_sql(EVENT_MATRIX, "is_through_ball"),
    "bool_is_switch_event": bool_rate_sql(EVENT_MATRIX, "is_switch"),
    "bool_is_cross_event": bool_rate_sql(EVENT_MATRIX, "is_cross"),
    "bool_is_cut_back_event": bool_rate_sql(EVENT_MATRIX, "is_cut_back"),
    "cat_shot_body_part_event": cat_rate_sql(EVENT_MATRIX, "shot_body_part_name"),
    "cat_shot_technique_event": cat_rate_sql(EVENT_MATRIX, "shot_technique_name"),
    "cat_shot_type_event": cat_rate_sql(EVENT_MATRIX, "shot_type_name"),
    "bool_shot_first_time_event": bool_rate_sql(EVENT_MATRIX, "shot_first_time"),
    "bool_shot_aerial_won_event": bool_rate_sql(EVENT_MATRIX, "shot_aerial_won"),
    "bool_shot_follows_dribble_event": bool_rate_sql(EVENT_MATRIX, "shot_follows_dribble"),
    "bool_shot_open_goal_event": bool_rate_sql(EVENT_MATRIX, "shot_open_goal"),
    "bool_shot_one_on_one_event": bool_rate_sql(EVENT_MATRIX, "shot_one_on_one"),
    "bool_shot_under_pressure_event": bool_rate_sql(EVENT_MATRIX, "shot_under_pressure"),
    "bool_shot_counterpress_event": bool_rate_sql(EVENT_MATRIX, "shot_counterpress"),
    # --- categorical / boolean signal, plus track (pass-context + shot attrs, same fields available) ---
    "cat_pass_height_plus": cat_rate_sql(PLUS_MATRIX, "pass_height_name"),
    "cat_pass_type_plus": cat_rate_sql(PLUS_MATRIX, "pass_type_name"),
    "cat_pass_technique_plus": cat_rate_sql(PLUS_MATRIX, "pass_technique_name"),
    "cat_pass_body_part_plus": cat_rate_sql(PLUS_MATRIX, "pass_body_part_name"),
    "bool_is_through_ball_plus": bool_rate_sql(PLUS_MATRIX, "is_through_ball"),
    "bool_is_switch_plus": bool_rate_sql(PLUS_MATRIX, "is_switch"),
    "bool_is_cross_plus": bool_rate_sql(PLUS_MATRIX, "is_cross"),
    "bool_is_cut_back_plus": bool_rate_sql(PLUS_MATRIX, "is_cut_back"),
    "cat_shot_body_part_plus": cat_rate_sql(PLUS_MATRIX, "shot_body_part_name"),
    "cat_shot_technique_plus": cat_rate_sql(PLUS_MATRIX, "shot_technique_name"),
    "cat_shot_type_plus": cat_rate_sql(PLUS_MATRIX, "shot_type_name"),
    "bool_shot_first_time_plus": bool_rate_sql(PLUS_MATRIX, "shot_first_time"),
    "bool_shot_aerial_won_plus": bool_rate_sql(PLUS_MATRIX, "shot_aerial_won"),
    "bool_shot_follows_dribble_plus": bool_rate_sql(PLUS_MATRIX, "shot_follows_dribble"),
    "bool_shot_open_goal_plus": bool_rate_sql(PLUS_MATRIX, "shot_open_goal"),
    "bool_shot_one_on_one_plus": bool_rate_sql(PLUS_MATRIX, "shot_one_on_one"),
    "bool_shot_under_pressure_plus": bool_rate_sql(PLUS_MATRIX, "shot_under_pressure"),
    "bool_shot_counterpress_plus": bool_rate_sql(PLUS_MATRIX, "shot_counterpress"),
    # --- 3. thin-support / near-constant re-check ---
    "thin_support_event": f"""
        SELECT COUNT(*) n,
          COUNTIF(shot_follows_dribble) n_follows_dribble,
          COUNTIF(shot_type_name != 'Open Play') n_non_open_play,
          COUNTIF(shot_open_goal) n_open_goal,
          COUNTIF(pass_body_part_name = 'No Touch') n_notouch,
          COUNTIF(pass_technique_name = 'Straight') n_technique_straight,
          COUNTIF(shot_gk_x IS NULL) n_null_gk,
          COUNTIF(is_cut_back) n_cut_back,
          COUNTIF(pass_type_name IS NOT NULL) n_pass_type_tagged
        FROM {EVENT_MATRIX}
    """,
    "thin_support_plus": f"""
        SELECT COUNT(*) n,
          COUNTIF(shot_follows_dribble) n_follows_dribble,
          COUNTIF(shot_type_name != 'Open Play') n_non_open_play,
          COUNTIF(shot_open_goal) n_open_goal,
          COUNTIF(pass_body_part_name = 'No Touch') n_notouch,
          COUNTIF(shot_gk_x IS NULL) n_null_gk,
          COUNTIF(is_cut_back) n_cut_back
        FROM {PLUS_MATRIX}
    """,
    # --- 4. redundancy / correlation ---
    "redundancy_numeric_event": f"""
        SELECT
          ROUND(CORR(pass_end_x, shot_x_sb),4) corr_pass_end_x_shot_x,
          ROUND(CORR(pass_end_y, shot_y_sb),4) corr_pass_end_y_shot_y,
          ROUND(CORR(shot_x_sb, shot_end_x),4) corr_shot_x_shot_end_x,
          ROUND(CORR(shot_y_sb, shot_end_y),4) corr_shot_y_shot_end_y,
          ROUND(CORR(shot_gk_distance_m, shot_defenders_within_5m),4) corr_gkdist_def5m,
          ROUND(CORR(shot_gk_distance_m, shot_defenders_within_8m),4) corr_gkdist_def8m,
          ROUND(CORR(shot_defenders_within_5m, shot_defenders_within_8m),4) corr_def5m_def8m,
          ROUND(CORR(shot_defenders_visible, shot_frame_player_count),4) corr_defvis_framecount,
          ROUND(CORR(start_x, pass_end_x),4) corr_startx_passendx,
          ROUND(CORR(pass_length, start_x),4) corr_passlength_startx,
          ROUND(CORR(statsbomb_xg, shot_gk_distance_m),4) corr_xg_gkdist,
          ROUND(CORR(statsbomb_xg, shot_defenders_within_5m),4) corr_xg_def5m
        FROM {EVENT_MATRIX}
    """,
    "redundancy_numeric_plus": f"""
        SELECT
          ROUND(CORR(pass_end_x, receiver_x),4) corr_pass_end_x_receiver_x,
          ROUND(CORR(receiver_x, shot_x_sb),4) corr_receiver_x_shot_x,
          ROUND(CORR(reception_nearest_opponent_distance_m, shot_gk_distance_m),4) corr_receptiondist_gkdist,
          ROUND(CORR(reception_opponents_within_5m, shot_defenders_within_5m),4) corr_receptionopp5m_shotdef5m,
          ROUND(CORR(shot_gk_distance_m, shot_defenders_within_5m),4) corr_gkdist_def5m,
          ROUND(CORR(statsbomb_xg, shot_gk_distance_m),4) corr_xg_gkdist
        FROM {PLUS_MATRIX}
    """,
    "overlap_one_on_one_vs_open_goal_event": f"""
        SELECT shot_one_on_one, shot_open_goal, COUNT(*) n, COUNTIF(y_goal) n_goal,
               ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {EVENT_MATRIX} GROUP BY 1,2 ORDER BY 1,2
    """,
    "overlap_first_time_vs_pass_technique_event": f"""
        SELECT shot_first_time, pass_technique_name, COUNT(*) n, COUNTIF(y_goal) n_goal,
               ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {EVENT_MATRIX} GROUP BY 1,2 ORDER BY 1,2
    """,
    "overlap_is_cross_vs_shot_body_part_event": f"""
        SELECT is_cross, shot_body_part_name, COUNT(*) n, COUNTIF(y_goal) n_goal,
               ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {EVENT_MATRIX} GROUP BY 1,2 ORDER BY 1,2
    """,
    "overlap_under_pressure_vs_counterpress_event": f"""
        SELECT shot_under_pressure, shot_counterpress, COUNT(*) n, COUNTIF(y_goal) n_goal
        FROM {EVENT_MATRIX} GROUP BY 1,2 ORDER BY 1,2
    """,
    # --- 5. leakage re-confirmation ---
    "leakage_is_completed_event": f"""
        SELECT is_completed, COUNT(*) n, COUNTIF(y_goal) n_goal, ROUND(COUNTIF(y_goal)/COUNT(*)*100,3) goal_pct
        FROM {EVENT_MATRIX} GROUP BY 1
    """,
    "leakage_statsbomb_xg_correlation_event": f"""
        SELECT ROUND(CORR(statsbomb_xg, CAST(y_goal AS INT64)),4) corr_xg_ygoal,
               ROUND(AVG(statsbomb_xg),4) avg_xg_all,
               ROUND(AVG(CASE WHEN y_goal THEN statsbomb_xg END),4) avg_xg_goal,
               ROUND(AVG(CASE WHEN NOT y_goal THEN statsbomb_xg END),4) avg_xg_nogoal
        FROM {EVENT_MATRIX}
    """,
    "leakage_statsbomb_xg_correlation_plus": f"""
        SELECT ROUND(CORR(statsbomb_xg, CAST(y_goal AS INT64)),4) corr_xg_ygoal,
               ROUND(AVG(statsbomb_xg),4) avg_xg_all,
               ROUND(AVG(CASE WHEN y_goal THEN statsbomb_xg END),4) avg_xg_goal,
               ROUND(AVG(CASE WHEN NOT y_goal THEN statsbomb_xg END),4) avg_xg_nogoal
        FROM {PLUS_MATRIX}
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
        print(f"{name}: {len(rows)} row(s)")
    print(json.dumps({"sections_written": summary}, indent=2))


if __name__ == "__main__":
    main()
