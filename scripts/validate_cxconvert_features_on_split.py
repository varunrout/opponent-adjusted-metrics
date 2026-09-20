"""Train-vs-validation confirmation for CxA P_convert candidate features (both tracks).

Reads only `split IN ('train', 'validation')` from `oam_features.cxconvert_event_v1_
training_matrix` / `cxconvert_plus_v1_training_matrix` (the `split` column is already
joined in at materialization time, no separate join to the splits table needed here).
Test stays sealed -- never queried by this script.

Per docs/analysis/cxa_p_convert_pre_model_analysis.md section 9, three groups of
candidate features need this treatment:

1. "Locked" features (large effect + large support on the full population) -- this
   task still recomputes their train and validation effect and confirms direction
   holds on both, rather than waving them through (unlike P_create's own event-only
   lock, which skipped this step for its population-strong features -- this task's
   instructions are explicitly stricter).
2. "Needs split-validation" features -- subject to a promotion rule.
3. `shot_end_z` -- handled separately (see the `end_z` section), not a lift/gap
   statistic.

Lift definition for boolean/categorical flags: `lift = rate(y_goal | flag=TRUE) /
rate(y_goal | flag=FALSE)` (the flagged group's rate over its own complement's, not the
whole-population rate) -- same convention P_create's own split-validation scripts used.
For an INVERSE-direction feature (validation rate lower than complement, e.g.
`is_switch`, `pass_type_name=Corner`), the same ratio is reported as a "suppression"
value: `suppression = rate(y_goal | flag=FALSE) / rate(y_goal | flag=TRUE)` (the
complement's rate over the flagged group's), so both directions are expressed as a
ratio >1 that grows with effect size, and both use the same promotion-rule shape.

Numeric "gap" features are reported as `mean(y_goal=TRUE) - mean(y_goal=FALSE)`, signed
so a feature whose pre-model direction was "higher value -> more goals" has a positive
train gap, and a "lower value -> more goals" feature (e.g. `shot_gk_distance_m`,
`reception_nearest_opponent_distance_m`) has a negative train gap.

Promotion rule for "needs split-validation" features, ADAPTED from P_create's own fixed
2.0x-of-baseline floor:
1. Direction must not flip: train effect and validation effect must have the same sign
   (gap) or both be > 1x (lift/suppression ratio).
2. Validation effect magnitude must be >= 50% of train effect magnitude. Magnitude for
   a lift/suppression ratio is `(ratio - 1)` (the excess over "no effect"); magnitude
   for a gap is `abs(gap)`.

This is a deliberate relaxation of P_create's own rule (validation lift >= a fixed
2.0x), stated and justified explicitly, not silently reused: P_create's CxA+ validation
split alone had 19,490-92,977 rows; this population's validation splits are far smaller
(1,720 event rows / 159 goals, 420 CxA+ rows / 46 goals -- see
docs/cxa_convert_split_policy_and_plan.md). A fixed 2.0x floor calibrated against
P_create's much larger, higher-base-rate splits would reject genuine signal here purely
because of sample size, not because the effect is weaker. A relative floor (validation
retains at least half of train's excess-over-baseline) scales with how strong the
feature's own train-split signal was, which is the right comparison at this support
level -- but see the doc's own explicit caveat wherever a CxA+ call rests on single-
digit validation positive counts (the ratio can still be mechanically "confirmed" while
resting on very few rows; that judgment call is made in the write-up, not by this
script).

Does not fit a model. Test split is never read.
"""

from __future__ import annotations

import json
from pathlib import Path

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxconvert_event_v1_training_matrix`"
PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix`"

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "audit_outputs" / "cxconvert_analysis" / "feature_lock"

MIN_VALIDATION_RETENTION = 0.5  # validation excess-over-baseline must be >= 50% of train's

# name -> (SQL boolean expression, expected_direction: "elevated" or "suppressed")
LOCKED_BOOL_FEATURES: dict[str, tuple[str, str]] = {
    "is_through_ball": ("is_through_ball", "elevated"),
    "shot_one_on_one": ("shot_one_on_one", "elevated"),
    "is_cross": ("is_cross", "elevated"),
    "shot_first_time": ("shot_first_time", "elevated"),
}

NEEDS_VALIDATION_BOOL_FEATURES: dict[str, tuple[str, str]] = {
    "shot_open_goal": ("shot_open_goal", "elevated"),
    "shot_technique_name=Lob": ("(shot_technique_name = 'Lob')", "elevated"),
    "shot_technique_name=Diving Header": ("(shot_technique_name = 'Diving Header')", "elevated"),
    "shot_technique_name=Backheel": ("(shot_technique_name = 'Backheel')", "elevated"),
    "shot_technique_name=Volley": ("(shot_technique_name = 'Volley')", "elevated"),
    "is_cut_back": ("is_cut_back", "elevated"),
    "pass_body_part_name=No Touch": ("(pass_body_part_name = 'No Touch')", "elevated"),
    "pass_technique_name=Straight": ("(pass_technique_name = 'Straight')", "elevated"),
    "is_switch": ("is_switch", "suppressed"),
    "pass_type_name=Corner": ("(pass_type_name = 'Corner')", "suppressed"),
}

# name -> (SQL numeric expression, expected_direction: "higher" or "lower" for goal=TRUE)
LOCKED_NUMERIC_FEATURES: dict[str, tuple[str, str]] = {
    "start_x": ("start_x", "higher"),
    "pass_end_x": ("pass_end_x", "higher"),
    "shot_x_sb": ("shot_x_sb", "higher"),
    "shot_dist_to_goal_m": (
        "SQRT(POW(120-shot_x_sb,2)+POW(40-shot_y_sb,2))*(105.0/120.0)",
        "lower",
    ),
    "shot_gk_distance_m": ("shot_gk_distance_m", "lower"),
    "shot_defenders_within_5m": ("shot_defenders_within_5m", "higher"),
    "shot_defenders_within_8m": ("shot_defenders_within_8m", "higher"),
}

PLUS_ONLY_NUMERIC_FEATURES: dict[str, tuple[str, str]] = {
    "reception_nearest_opponent_distance_m": ("reception_nearest_opponent_distance_m", "lower"),
    "reception_opponents_within_5m": ("reception_opponents_within_5m", "higher"),
}

BOOL_SQL_TEMPLATE = """
WITH joined AS (
  SELECT split, y_goal, {expr} AS flag
  FROM {matrix}
  WHERE split IN ('train', 'validation')
),
totals AS (SELECT split, COUNT(*) total_n, COUNTIF(y_goal) total_goal FROM joined GROUP BY split),
true_group AS (
  SELECT split, COUNTIF(flag) n_true, COUNTIF(flag AND y_goal) n_true_goal
  FROM joined GROUP BY split
)
SELECT t.split, tg.n_true, tg.n_true_goal, SAFE_DIVIDE(tg.n_true_goal, tg.n_true) rate_true,
  t.total_n - tg.n_true n_false, t.total_goal - tg.n_true_goal n_false_goal,
  SAFE_DIVIDE(t.total_goal - tg.n_true_goal, t.total_n - tg.n_true) rate_false
FROM totals t JOIN true_group tg USING (split)
ORDER BY split
"""

NUMERIC_SQL_TEMPLATE = """
SELECT split, y_goal, COUNT(*) n, AVG({expr}) avg_val
FROM {matrix}
WHERE split IN ('train', 'validation')
GROUP BY split, y_goal
ORDER BY split, y_goal
"""

END_Z_SQL_TEMPLATE = """
SELECT split, y_goal, COUNT(*) n, COUNTIF(shot_end_z IS NULL) n_null,
  AVG(shot_end_z) avg_end_z_nonnull
FROM {matrix}
WHERE split IN ('train', 'validation')
GROUP BY split, y_goal
ORDER BY split, y_goal
"""


def run_bool_feature(client, matrix, name, expr, direction):
    sql = BOOL_SQL_TEMPLATE.format(expr=expr, matrix=matrix)
    rows = {r["split"]: dict(r.items()) for r in client.query(sql, location=LOCATION).result()}
    out = {"direction": direction}
    for split in ("train", "validation"):
        r = rows[split]
        ratio = None
        if direction == "elevated" and r["rate_false"]:
            ratio = r["rate_true"] / r["rate_false"] if r["rate_false"] else None
        elif direction == "suppressed" and r["rate_true"]:
            ratio = r["rate_false"] / r["rate_true"] if r["rate_true"] else None
        out[split] = {**r, "ratio": ratio}
    return out


def run_numeric_feature(client, matrix, name, expr, direction):
    sql = NUMERIC_SQL_TEMPLATE.format(expr=expr, matrix=matrix)
    rows = {(r["split"], r["y_goal"]): dict(r.items()) for r in client.query(sql, location=LOCATION).result()}
    out = {"direction": direction}
    for split in ("train", "validation"):
        true_r = rows.get((split, True))
        false_r = rows.get((split, False))
        gap = true_r["avg_val"] - false_r["avg_val"] if true_r and false_r else None
        out[split] = {
            "n_true": true_r["n"] if true_r else None,
            "avg_true": true_r["avg_val"] if true_r else None,
            "n_false": false_r["n"] if false_r else None,
            "avg_false": false_r["avg_val"] if false_r else None,
            "gap": gap,
        }
    return out


def run_end_z(client, matrix):
    sql = END_Z_SQL_TEMPLATE.format(matrix=matrix)
    rows = {(r["split"], r["y_goal"]): dict(r.items()) for r in client.query(sql, location=LOCATION).result()}
    out = {}
    for split in ("train", "validation"):
        true_r = rows.get((split, True))
        false_r = rows.get((split, False))
        out[split] = {
            "n_true": true_r["n"] if true_r else None,
            "n_null_true": true_r["n_null"] if true_r else None,
            "avg_end_z_true_nonnull": true_r["avg_end_z_nonnull"] if true_r else None,
            "n_false": false_r["n"] if false_r else None,
            "n_null_false": false_r["n_null"] if false_r else None,
            "null_pct_false": (false_r["n_null"] / false_r["n"] * 100) if false_r and false_r["n"] else None,
            "avg_end_z_false_nonnull": false_r["avg_end_z_nonnull"] if false_r else None,
        }
    return out


def check_promotion(stat: dict, is_bool: bool) -> dict:
    train, val = stat["train"], stat["validation"]
    if is_bool:
        train_val_ok = train["ratio"] is not None and train["ratio"] > 1.0
        val_ok = val["ratio"] is not None and val["ratio"] > 1.0
        train_excess = (train["ratio"] - 1.0) if train["ratio"] else 0.0
        val_excess = (val["ratio"] - 1.0) if val["ratio"] else 0.0
    else:
        expected_sign = 1 if stat["direction"] == "higher" else -1
        train_val_ok = train["gap"] is not None and (train["gap"] * expected_sign) > 0
        val_ok = val["gap"] is not None and (val["gap"] * expected_sign) > 0
        train_excess = abs(train["gap"]) if train["gap"] is not None else 0.0
        val_excess = abs(val["gap"]) if val["gap"] is not None else 0.0
    retention = (val_excess / train_excess) if train_excess else 0.0
    promoted = train_val_ok and val_ok and retention >= MIN_VALIDATION_RETENTION
    return {
        "direction_holds_train": train_val_ok,
        "direction_holds_validation": val_ok,
        "train_excess": train_excess,
        "validation_excess": val_excess,
        "retention": retention,
        "promoted": promoted,
    }


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for track, matrix, has_plus_features in (("event", EVENT_MATRIX, False), ("plus", PLUS_MATRIX, True)):
        result: dict[str, object] = {"locked": {}, "needs_validation": {}}

        for name, (expr, direction) in LOCKED_BOOL_FEATURES.items():
            result["locked"][name] = run_bool_feature(client, matrix, name, expr, direction)
        for name, (expr, direction) in LOCKED_NUMERIC_FEATURES.items():
            result["locked"][name] = run_numeric_feature(client, matrix, name, expr, direction)
        if has_plus_features:
            for name, (expr, direction) in PLUS_ONLY_NUMERIC_FEATURES.items():
                result["locked"][name] = run_numeric_feature(client, matrix, name, expr, direction)

        for name, (expr, direction) in NEEDS_VALIDATION_BOOL_FEATURES.items():
            stat = run_bool_feature(client, matrix, name, expr, direction)
            stat["promotion_check"] = check_promotion(stat, is_bool=True)
            result["needs_validation"][name] = stat

        result["shot_end_z"] = run_end_z(client, matrix)

        out_path = OUTPUT_DIR / f"split_validation_{track}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"{track}: wrote {out_path}")

        print(f"\n=== {track} track: needs-validation promotion results ===")
        for name, stat in result["needs_validation"].items():
            c = stat["promotion_check"]
            verdict = "PROMOTE" if c["promoted"] else "DROP"
            print(
                f"  {name:<40} train_excess={c['train_excess']:.3f} "
                f"val_excess={c['validation_excess']:.3f} retention={c['retention']:.2f} -> {verdict}"
            )


if __name__ == "__main__":
    main()
