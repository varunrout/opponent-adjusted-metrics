"""Pre-modelling EDA scoped to the CxA P_create LOCKED feature set only -- one level
deeper than the signal/redundancy checks already done in
docs/analysis/cxa_p_create_pre_model_analysis.md (target usability, sparsity,
per-feature signal-vs-target, and the pass_length/angle/start/end and receiver_x,y vs
end_x,y redundancy pairs). Does not repeat any of that.

Covers, for the event-only locked-10 (`cxa_event_v1_training_matrix`, full population,
608,722 rows) and separately for the CxA+ locked-9 + held-out No Touch
(`cxa_plus_v1_training_matrix`, full population, 133,143 rows):

1. Numeric locked-feature distributions (mean/median/std/min/max/p1/p5/p95/p99),
   pitch-bounds sanity (0-120 x 0-80), decile histogram for shape/bimodality.
2. Categorical/boolean locked-feature level counts and percentages on the full
   population, flagging any level under 100 total rows.
3. Full pairwise correlation matrix across the whole locked set encoded together
   (numeric as-is, categorical one-hot per *locked level only*, booleans as 0/1) --
   the first time the locked set has been checked together rather than in the ad hoc
   pairs the pre-model analysis checked.
4. Interaction/overlap crosstabs for the two strongest set-piece flags (is_cross,
   pass_type_name=Corner) and the two open-play flags (is_through_ball, is_cut_back).

Per docs/cxa_split_policy_and_parallel_plan.md ("full-dataset analysis (EDA, null
profiling, summary stats) is valid exploratory work"), this runs on the full
population, not train-only -- feature promotion and model evaluation (which do require
train-only) were already done in the feature-lock docs and are not repeated here.

Writes no BigQuery tables, does not train, score, or select a model, and does not
reopen the locked feature list -- it only characterizes features already locked.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix`"
PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix`"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxa_analysis" / "locked_feature_eda"

# --- Locked-feature column definitions, one-hot per CONFIRMED LOCKED LEVEL only -----

EVENT_NUMERIC = ["start_x", "end_x"]
EVENT_BOOL_COLS: dict[str, str] = {
    "is_cross": "is_cross",
    "is_through_ball": "is_through_ball",
    "pass_type_Corner": "(pass_type_name = 'Corner')",
    "pass_type_FreeKick": "(pass_type_name = 'Free Kick')",
    "play_pattern_Counter": "(play_pattern_name = 'From Counter')",
    "play_pattern_Corner": "(play_pattern_name = 'From Corner')",
    "pass_technique_Outswinging": "(pass_technique_name = 'Outswinging')",
    "is_switch": "is_switch",
    "is_cut_back": "is_cut_back",
    "pass_bodypart_NoTouch": "(pass_body_part_name = 'No Touch')",
}

PLUS_NUMERIC = [
    "start_x",
    "end_x",
    "reception_nearest_opponent_distance_m",
    "reception_opponents_within_5m",
    "reception_opponents_within_8m",  # context only, not itself locked
]
PLUS_BOOL_COLS: dict[str, str] = {
    "is_cross": "is_cross",
    "is_through_ball": "is_through_ball",
    "pass_type_Corner": "(pass_type_name = 'Corner')",
    "pass_type_FreeKick": "(pass_type_name = 'Free Kick')",
    "play_pattern_Counter": "(play_pattern_name = 'From Counter')",
    "play_pattern_Corner": "(play_pattern_name = 'From Corner')",
    "pass_technique_Outswinging": "(pass_technique_name = 'Outswinging')",
    "pass_technique_Straight": "(pass_technique_name = 'Straight')",
    "is_cut_back": "is_cut_back",
    "is_switch": "is_switch",
    "pass_bodypart_NoTouch": "(pass_body_part_name = 'No Touch')",  # held out, kept for context
}

# Pairs already checked and accepted elsewhere -- not "new" findings if they show up here.
KNOWN_PAIRS = {("start_x", "end_x")}

NUMERIC_DIST_TEMPLATE = """
SELECT
  '{col}' AS feature, COUNT(*) AS n, COUNTIF({col} IS NULL) AS n_null,
  ROUND(AVG({col}), 3) AS mean, ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(50)], 3) AS median,
  ROUND(STDDEV({col}), 3) AS stddev, MIN({col}) AS min_val, MAX({col}) AS max_val,
  ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(1)], 3) AS p1,
  ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(5)], 3) AS p5,
  ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(95)], 3) AS p95,
  ROUND(APPROX_QUANTILES({col}, 100)[OFFSET(99)], 3) AS p99
FROM {matrix}
"""

DECILE_HIST_TEMPLATE = """
SELECT '{col}' AS feature, APPROX_QUANTILES({col}, 10) AS decile_edges
FROM {matrix}
"""


def numeric_distribution(client: bigquery.Client, matrix: str, columns: list[str]) -> list[dict]:
    rows = []
    for col in columns:
        sql = NUMERIC_DIST_TEMPLATE.format(col=col, matrix=matrix)
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def decile_histograms(client: bigquery.Client, matrix: str, columns: list[str]) -> list[dict]:
    rows = []
    for col in columns:
        sql = DECILE_HIST_TEMPLATE.format(col=col, matrix=matrix)
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def categorical_levels(client: bigquery.Client, matrix: str, bool_cols: dict[str, str]) -> list[dict]:
    rows = []
    for name, expr in bool_cols.items():
        sql = f"""
        SELECT '{name}' AS feature, COUNT(*) AS total_n, COUNTIF({expr}) AS n_true,
               ROUND(COUNTIF({expr}) / COUNT(*) * 100, 4) AS pct_true
        FROM {matrix}
        """
        rows.append(dict(list(client.query(sql, location=LOCATION).result())[0].items()))
    return rows


def correlation_matrix(
    client: bigquery.Client, matrix: str, numeric_cols: list[str], bool_cols: dict[str, str]
) -> list[dict]:
    exprs: dict[str, str] = {c: f"CAST({c} AS FLOAT64)" for c in numeric_cols}
    exprs.update({name: f"CAST({expr} AS INT64)" for name, expr in bool_cols.items()})
    names = list(exprs.keys())
    pairs = list(itertools.combinations(names, 2))
    select_parts = [
        f"ROUND(CORR({exprs[a]}, {exprs[b]}), 4) AS r__{a}__{b}" for a, b in pairs
    ]
    sql = f"SELECT {', '.join(select_parts)} FROM {matrix}"
    row = dict(list(client.query(sql, location=LOCATION).result())[0].items())
    results = []
    for a, b in pairs:
        key = f"r__{a}__{b}"
        results.append({"feature_a": a, "feature_b": b, "r": row[key]})
    return results


def interaction_overlap(client: bigquery.Client, matrix: str, flag_a: str, flag_b: str, name_a: str, name_b: str) -> dict:
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


def run_track(client: bigquery.Client, track: str, matrix: str, numeric_cols: list[str], bool_cols: dict[str, str]) -> dict:
    print(f"--- {track} ---")
    dist = numeric_distribution(client, matrix, numeric_cols)
    print(f"numeric distribution: {len(dist)} feature(s)")
    hist = decile_histograms(client, matrix, numeric_cols)
    print(f"decile histograms: {len(hist)} feature(s)")
    levels = categorical_levels(client, matrix, bool_cols)
    print(f"categorical levels: {len(levels)} feature(s)")
    corr = correlation_matrix(client, matrix, numeric_cols, bool_cols)
    flagged = [
        r for r in corr
        if abs(r["r"]) >= 0.5 and (r["feature_a"], r["feature_b"]) not in KNOWN_PAIRS
        and (r["feature_b"], r["feature_a"]) not in KNOWN_PAIRS
    ]
    print(f"correlation pairs: {len(corr)}, |r|>=0.5 and new: {len(flagged)}")

    return {
        "numeric_distribution": dist,
        "decile_histograms": hist,
        "categorical_levels": levels,
        "correlation_matrix": corr,
        "flagged_new_correlations": flagged,
    }


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    event_result = run_track(client, "event-only (cxa_event_v1_training_matrix)", EVENT_MATRIX, EVENT_NUMERIC, EVENT_BOOL_COLS)
    (OUTPUT_DIR / "event_eda.json").write_text(json.dumps(event_result, indent=2, default=str))

    plus_result = run_track(client, "cxa+ (cxa_plus_v1_training_matrix)", PLUS_MATRIX, PLUS_NUMERIC, PLUS_BOOL_COLS)
    (OUTPUT_DIR / "plus_eda.json").write_text(json.dumps(plus_result, indent=2, default=str))

    # Interaction overlap: two strongest set-piece flags vs two open-play flags, both tracks.
    overlaps = {}
    for track, matrix in (("event", EVENT_MATRIX), ("plus", PLUS_MATRIX)):
        pairs = [
            ("is_cross", "(pass_type_name = 'Corner')", "is_cross", "pass_type_name=Corner"),
            ("is_cross", "is_through_ball", "is_cross", "is_through_ball"),
            ("is_cross", "is_cut_back", "is_cross", "is_cut_back"),
            ("(pass_type_name = 'Corner')", "is_through_ball", "pass_type_name=Corner", "is_through_ball"),
            ("(pass_type_name = 'Corner')", "is_cut_back", "pass_type_name=Corner", "is_cut_back"),
            ("is_through_ball", "is_cut_back", "is_through_ball", "is_cut_back"),
        ]
        overlaps[track] = [
            interaction_overlap(client, matrix, a_expr, b_expr, a_name, b_name)
            for a_expr, b_expr, a_name, b_name in pairs
        ]
    (OUTPUT_DIR / "interaction_overlap.json").write_text(json.dumps(overlaps, indent=2, default=str))
    print("interaction overlap: written")

    print("\nDone. Output written under", OUTPUT_DIR)


if __name__ == "__main__":
    main()
