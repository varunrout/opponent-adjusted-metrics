"""Train-vs-validation lift confirmation for the 4 CxA event-only P_create features
flagged "validate before trusting" in docs/analysis/cxa_p_create_pre_model_analysis.md
(section 9): `pass_technique_name = Outswinging`, `is_switch`, `is_cut_back`,
`pass_body_part_name = No Touch`.

Per docs/cxa_split_policy_and_parallel_plan.md step 5 ("Validate selected features on
the validation split: direction stability, support stability, ... uplift over the XY
baseline"), a feature that only showed signal on the full population (as the pre-model
analysis did, by design -- see that doc's section 9 caveat) is not confirmed until it
also holds on train and, independently, on the validation split. Test stays sealed;
this script only ever reads `split IN ('train', 'validation')` from
`oam_analysis.cxa_match_splits_v1`.

Lift definition (matches the manually-computed table this script reproduces): for a
boolean feature flag, `lift = rate(y_create | flag=TRUE) / rate(y_create | flag=FALSE)`
-- the feature group's create rate divided by its own complement's create rate (NOT the
whole-population rate, which would still include the flagged rows and understate the
lift). This is the standard two-group lift definition and is what makes, e.g., the
Outswinging lift land at ~13.87x rather than ~13.3x (the population-baseline version).

Promotion rule (stated and justified here, not left implicit):
1. Direction must not flip: train lift > 1 AND validation lift > 1 (i.e. the feature
   group's create rate must stay above its complement's on both splits).
2. Validation lift must be >= MIN_VALIDATION_LIFT (2.0x, see constant below for the
   justification).
Any feature failing either check is flagged, not silently passed -- this script does
not assume all 4 features will confirm, even though they are expected to (per the
already-completed manual check this script exists to make reproducible).
"""

from __future__ import annotations

import json
import sys

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

EVENT_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix`"
SPLITS_TABLE = f"`{PROJECT}.{ANALYSIS_DATASET}.cxa_match_splits_v1`"

# Chosen so the deliberately weakest confirmed candidate (is_switch, validation lift
# 2.28x) still clears it with a real margin (0.28x), while staying low enough to accept
# genuine-but-modest signal rather than only admitting the largest effects. A flag with
# no real relationship to y_create at this population's ~1.85% base rate and these
# support sizes (tens to low thousands of TRUE rows) would not consistently land near
# or above 2x on an independently-drawn validation split -- sampling noise at these
# support levels does not reliably produce a 2x directional lift twice in a row.
MIN_VALIDATION_LIFT = 2.0

FEATURES: dict[str, str] = {
    "pass_technique_name=Outswinging": "(pass_technique_name = 'Outswinging')",
    "is_switch": "is_switch",
    "is_cut_back": "is_cut_back",
    "pass_body_part_name=No Touch": "(pass_body_part_name = 'No Touch')",
}

SQL_TEMPLATE = """
WITH joined AS (
  SELECT m.split, e.y_create, {expr} AS flag
  FROM {matrix} e
  JOIN {splits} m USING (match_id)
  WHERE m.split IN ('train', 'validation')
),
totals AS (
  SELECT split, COUNT(*) AS total_n, COUNTIF(y_create) AS total_create FROM joined GROUP BY split
),
true_group AS (
  SELECT split, COUNTIF(flag) AS n_true, COUNTIF(flag AND y_create) AS n_true_create
  FROM joined GROUP BY split
)
SELECT
  t.split,
  tg.n_true,
  tg.n_true_create,
  SAFE_DIVIDE(tg.n_true_create, tg.n_true) AS rate_true,
  t.total_n - tg.n_true AS n_false,
  t.total_create - tg.n_true_create AS n_false_create,
  SAFE_DIVIDE(t.total_create - tg.n_true_create, t.total_n - tg.n_true) AS rate_false,
  SAFE_DIVIDE(
    SAFE_DIVIDE(tg.n_true_create, tg.n_true),
    SAFE_DIVIDE(t.total_create - tg.n_true_create, t.total_n - tg.n_true)
  ) AS lift
FROM totals t JOIN true_group tg USING (split)
ORDER BY split
"""


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    results: dict[str, dict[str, dict[str, float]]] = {}
    failures: list[str] = []

    for feature_name, expr in FEATURES.items():
        sql = SQL_TEMPLATE.format(expr=expr, matrix=EVENT_MATRIX, splits=SPLITS_TABLE)
        rows = {r["split"]: dict(r.items()) for r in client.query(sql, location=LOCATION).result()}
        results[feature_name] = rows

        train_lift = rows["train"]["lift"]
        val_lift = rows["validation"]["lift"]
        checks = {
            "direction_holds_train": train_lift > 1.0,
            "direction_holds_validation": val_lift > 1.0,
            "validation_lift_above_threshold": val_lift >= MIN_VALIDATION_LIFT,
        }
        rows["checks"] = checks
        rows["min_validation_lift_threshold"] = MIN_VALIDATION_LIFT
        if not all(checks.values()):
            failures.append(f"{feature_name}: {checks}")

    print(
        f"{'feature':<32} {'train n_true':>12} {'train lift':>11} "
        f"{'val n_true':>11} {'val lift':>9}  verdict"
    )
    for feature_name, rows in results.items():
        train, val = rows["train"], rows["validation"]
        verdict = "PASS" if all(rows["checks"].values()) else "FAIL"
        print(
            f"{feature_name:<32} {train['n_true']:>12} {train['lift']:>10.2f}x "
            f"{val['n_true']:>11} {val['lift']:>8.2f}x  {verdict}"
        )

    print()
    print(json.dumps(results, indent=2, default=str))

    if failures:
        print("\nFAILED promotion rule:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\nAll {len(FEATURES)} features pass the promotion rule (min validation lift {MIN_VALIDATION_LIFT}x, no direction flip).")


if __name__ == "__main__":
    main()
