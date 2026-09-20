"""Deep EDA on the LOCKED CxA P_convert feature sets only -- one level deeper than the
pre-model analysis's signal/redundancy checks and the feature-lock docs' train-vs-
validation confirmation. Does not repeat target usability, sparsity, per-feature
signal-vs-target, or any redundancy pair already covered in
docs/analysis/cxa_p_convert_pre_model_analysis.md.

Covers, for the event-only locked-15 (`cxconvert_event_v1_training_matrix`) and
CxA+ locked-15 (`cxconvert_plus_v1_training_matrix`):

1. Numeric locked-feature distributions (mean/median/std/min/max/p1/p5/p95/p99),
   pitch-bounds sanity (0-120 for x-coordinates; >=0 for distance/count features),
   decile histogram for shape.
2. Categorical/boolean locked-feature level counts on THIS SCOPE (train+validation,
   not the full matrix including test -- see the scope note below), flagging any level
   under 100 rows.
3. Full pairwise correlation matrix across the whole locked set, encoded together
   (numeric as-is, boolean/categorical-level as 0/1).
4. Interaction/overlap crosstabs for selected locked boolean/categorical pairs.

SCOPE, deliberately narrower than P_create's own locked-feature EDA: this task's
instructions explicitly restrict every check here to `split IN ('train', 'validation')`
-- test stays sealed, unlike P_create's own locked-feature EDA, which ran on the true
full population (including test) under the split policy's "full-dataset analysis is
valid exploratory work" carve-out. That carve-out still applies in principle, but this
task's explicit constraint ("full population (train+validation) only... test stays
sealed") governs here and is followed literally, not silently loosened to match
P_create's precedent.

Writes no BigQuery tables, does not train, score, or select a model, and does not
reopen either locked feature list -- it characterizes features already locked, nothing
more.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"

EVENT_MATRIX = f"(SELECT * FROM `{PROJECT}.{FEATURES_DATASET}.cxconvert_event_v1_training_matrix` WHERE split IN ('train','validation'))"
PLUS_MATRIX = f"(SELECT * FROM `{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix` WHERE split IN ('train','validation'))"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "locked_feature_eda"

DIST_EXPR = "SQRT(POW(120-shot_x_sb,2)+POW(40-shot_y_sb,2))*(105.0/120.0)"

# --- Locked-feature column definitions -------------------------------------------

EVENT_NUMERIC: dict[str, str] = {
    "start_x": "start_x",
    "pass_end_x": "pass_end_x",
    "shot_x_sb": "shot_x_sb",
    "shot_dist_to_goal_m": DIST_EXPR,
    "shot_gk_distance_m": "shot_gk_distance_m",
    "shot_defenders_within_5m": "shot_defenders_within_5m",
    "shot_defenders_within_8m": "shot_defenders_within_8m",
}
EVENT_BOOL_COLS: dict[str, str] = {
    "is_through_ball": "is_through_ball",
    "shot_one_on_one": "shot_one_on_one",
    "is_cross": "is_cross",
    "shot_first_time": "shot_first_time",
    "shot_open_goal": "shot_open_goal",
    "shot_technique_Lob": "(shot_technique_name = 'Lob')",
    "shot_technique_DivingHeader": "(shot_technique_name = 'Diving Header')",
    "shot_technique_Backheel": "(shot_technique_name = 'Backheel')",
    "shot_technique_Volley": "(shot_technique_name = 'Volley')",
    "is_cut_back": "is_cut_back",
    "pass_type_Corner": "(pass_type_name = 'Corner')",
}

PLUS_NUMERIC: dict[str, str] = {
    "start_x": "start_x",
    "pass_end_x": "pass_end_x",
    "shot_x_sb": "shot_x_sb",
    "shot_dist_to_goal_m": DIST_EXPR,
    "shot_gk_distance_m": "shot_gk_distance_m",
    "reception_nearest_opponent_distance_m": "reception_nearest_opponent_distance_m",
    "reception_opponents_within_5m": "reception_opponents_within_5m",
}
PLUS_BOOL_COLS: dict[str, str] = {
    "is_through_ball": "is_through_ball",
    "shot_one_on_one": "shot_one_on_one",
    "is_cross": "is_cross",
    "shot_first_time": "shot_first_time",
    "shot_open_goal": "shot_open_goal",
    "shot_technique_Lob": "(shot_technique_name = 'Lob')",
    "is_cut_back": "is_cut_back",
    "pass_type_Corner": "(pass_type_name = 'Corner')",
}

# Pairs already known/accepted from the feature-lock docs -- not "new" if surfaced here.
KNOWN_PAIRS = {
    ("shot_x_sb", "shot_dist_to_goal_m"),
    ("shot_dist_to_goal_m", "shot_gk_distance_m"),
    ("shot_x_sb", "shot_gk_distance_m"),
    ("start_x", "pass_end_x"),
}

NUMERIC_DIST_TEMPLATE = """
SELECT
  '{name}' AS feature, COUNT(*) AS n, COUNTIF({expr} IS NULL) AS n_null,
  ROUND(AVG({expr}), 3) AS mean, ROUND(APPROX_QUANTILES({expr}, 100)[OFFSET(50)], 3) AS median,
  ROUND(STDDEV({expr}), 3) AS stddev, MIN({expr}) AS min_val, MAX({expr}) AS max_val,
  ROUND(APPROX_QUANTILES({expr}, 100)[OFFSET(1)], 3) AS p1,
  ROUND(APPROX_QUANTILES({expr}, 100)[OFFSET(5)], 3) AS p5,
  ROUND(APPROX_QUANTILES({expr}, 100)[OFFSET(95)], 3) AS p95,
  ROUND(APPROX_QUANTILES({expr}, 100)[OFFSET(99)], 3) AS p99
FROM {matrix}
"""

DECILE_HIST_TEMPLATE = "SELECT '{name}' AS feature, APPROX_QUANTILES({expr}, 10) AS decile_edges FROM {matrix}"

PITCH_X_BOUNDS_TEMPLATE = """
SELECT '{name}' AS feature, COUNTIF({expr} < 0 OR {expr} > 120) AS n_out_of_bounds,
  MIN({expr}) AS min_val, MAX({expr}) AS max_val
FROM {matrix}
"""

NONNEG_BOUNDS_TEMPLATE = """
SELECT '{name}' AS feature, COUNTIF({expr} < 0) AS n_negative, MIN({expr}) AS min_val, MAX({expr}) AS max_val
FROM {matrix}
"""

PITCH_X_FEATURES = {"start_x", "pass_end_x", "shot_x_sb"}
NONNEG_FEATURES = {"shot_dist_to_goal_m", "shot_gk_distance_m", "shot_defenders_within_5m", "shot_defenders_within_8m", "reception_nearest_opponent_distance_m", "reception_opponents_within_5m"}


def numeric_distribution(client, matrix, cols: dict[str, str]) -> list[dict]:
    rows = []
    for name, expr in cols.items():
        sql = NUMERIC_DIST_TEMPLATE.format(name=name, expr=expr, matrix=matrix)
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def decile_histograms(client, matrix, cols: dict[str, str]) -> list[dict]:
    rows = []
    for name, expr in cols.items():
        sql = DECILE_HIST_TEMPLATE.format(name=name, expr=expr, matrix=matrix)
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def bounds_checks(client, matrix, cols: dict[str, str]) -> list[dict]:
    rows = []
    for name, expr in cols.items():
        if name in PITCH_X_FEATURES:
            sql = PITCH_X_BOUNDS_TEMPLATE.format(name=name, expr=expr, matrix=matrix)
        elif name in NONNEG_FEATURES:
            sql = NONNEG_BOUNDS_TEMPLATE.format(name=name, expr=expr, matrix=matrix)
        else:
            continue
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def categorical_levels(client, matrix, bool_cols: dict[str, str]) -> list[dict]:
    rows = []
    for name, expr in bool_cols.items():
        sql = f"""
        SELECT '{name}' AS feature, COUNT(*) AS total_n, COUNTIF({expr}) AS n_true,
               ROUND(COUNTIF({expr}) / COUNT(*) * 100, 4) AS pct_true
        FROM {matrix}
        """
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def correlation_matrix(client, matrix, numeric_cols: dict[str, str], bool_cols: dict[str, str]) -> list[dict]:
    exprs: dict[str, str] = {name: f"CAST({expr} AS FLOAT64)" for name, expr in numeric_cols.items()}
    exprs.update({name: f"CAST({expr} AS INT64)" for name, expr in bool_cols.items()})
    names = list(exprs.keys())
    pairs = list(itertools.combinations(names, 2))
    select_parts = [f"ROUND(CORR({exprs[a]}, {exprs[b]}), 4) AS r__{a}__{b}" for a, b in pairs]
    sql = f"SELECT {', '.join(select_parts)} FROM {matrix}"
    row = dict(list(client.query(sql, location=LOCATION).result())[0].items())
    return [{"feature_a": a, "feature_b": b, "r": row[f"r__{a}__{b}"]} for a, b in pairs]


def interaction_overlap(client, matrix, flag_a, flag_b, name_a, name_b) -> dict:
    sql = f"""
    SELECT
      COUNTIF({flag_a} AND {flag_b}) AS both_true,
      COUNTIF({flag_a} AND NOT {flag_b}) AS a_only,
      COUNTIF(NOT {flag_a} AND {flag_b}) AS b_only,
      COUNTIF(NOT {flag_a} AND NOT {flag_b}) AS neither,
      COUNTIF({flag_a}) AS a_total,
      COUNTIF({flag_b}) AS b_total
    FROM {matrix}
    """
    row = dict(list(client.query(sql, location=LOCATION).result())[0].items())
    row["feature_a"] = name_a
    row["feature_b"] = name_b
    return row


EVENT_INTERACTIONS = [
    ("is_cross", "(shot_technique_name = 'Lob')", "is_cross", "shot_technique=Lob"),
    ("shot_one_on_one", "shot_open_goal", "shot_one_on_one", "shot_open_goal"),
    ("shot_first_time", "is_through_ball", "shot_first_time", "is_through_ball"),
    ("is_cross", "is_cut_back", "is_cross", "is_cut_back"),
    ("(pass_type_name = 'Corner')", "is_cross", "pass_type=Corner", "is_cross"),
    ("(pass_type_name = 'Corner')", "shot_open_goal", "pass_type=Corner", "shot_open_goal"),
]
PLUS_INTERACTIONS = [
    ("is_cross", "(shot_technique_name = 'Lob')", "is_cross", "shot_technique=Lob"),
    ("shot_one_on_one", "shot_open_goal", "shot_one_on_one", "shot_open_goal"),
    ("shot_first_time", "is_through_ball", "shot_first_time", "is_through_ball"),
    ("is_cross", "is_cut_back", "is_cross", "is_cut_back"),
    ("(pass_type_name = 'Corner')", "is_cross", "pass_type=Corner", "is_cross"),
    ("(pass_type_name = 'Corner')", "shot_open_goal", "pass_type=Corner", "shot_open_goal"),
]


def run_track(client, track, matrix, numeric_cols, bool_cols, interactions):
    print(f"--- {track} ---")
    dist = numeric_distribution(client, matrix, numeric_cols)
    hist = decile_histograms(client, matrix, numeric_cols)
    bounds = bounds_checks(client, matrix, numeric_cols)
    levels = categorical_levels(client, matrix, bool_cols)
    corr = correlation_matrix(client, matrix, numeric_cols, bool_cols)
    flagged_dup = [r for r in corr if abs(r["r"]) >= 0.8]
    flagged_moderate = [
        r for r in corr
        if 0.4 <= abs(r["r"]) < 0.8
        and (r["feature_a"], r["feature_b"]) not in KNOWN_PAIRS
        and (r["feature_b"], r["feature_a"]) not in KNOWN_PAIRS
    ]
    interaction_results = [interaction_overlap(client, matrix, a, b, na, nb) for a, b, na, nb in interactions]
    print(f"numeric: {len(dist)}, categorical: {len(levels)}, corr pairs: {len(corr)}, "
          f">=0.8: {len(flagged_dup)}, 0.4-0.8 new: {len(flagged_moderate)}, interactions: {len(interaction_results)}")
    return {
        "numeric_distribution": dist,
        "decile_histograms": hist,
        "bounds_checks": bounds,
        "categorical_levels": levels,
        "correlation_matrix": corr,
        "flagged_duplicate_pairs": flagged_dup,
        "flagged_moderate_pairs": flagged_moderate,
        "interactions": interaction_results,
    }


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_result = run_track(client, "event", EVENT_MATRIX, EVENT_NUMERIC, EVENT_BOOL_COLS, EVENT_INTERACTIONS)
    (OUTPUT_DIR / "event_result.json").write_text(json.dumps(event_result, indent=2, default=str))

    plus_result = run_track(client, "plus", PLUS_MATRIX, PLUS_NUMERIC, PLUS_BOOL_COLS, PLUS_INTERACTIONS)
    (OUTPUT_DIR / "plus_result.json").write_text(json.dumps(plus_result, indent=2, default=str))

    # cross-track: shared numeric features, split by track, to compare distributions
    shared_numeric = [n for n in EVENT_NUMERIC if n in PLUS_NUMERIC]
    shared_bool = [n for n in EVENT_BOOL_COLS if n in PLUS_BOOL_COLS]
    print(f"shared numeric features: {shared_numeric}")
    print(f"shared boolean features: {shared_bool}")

    print("Done.")


if __name__ == "__main__":
    main()
