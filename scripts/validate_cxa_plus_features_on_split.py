"""Train-vs-validation confirmation for CxA+'s own candidate feature set, against
`oam_analysis.cxa_match_splits_v1` filtered to the 166 tournament matches present in
`oam_features.cxa_plus_v1_training_matrix` (train=119/95,083 rows,
validation=24/19,490 rows, test=23 matches sealed, never queried here).

Same structure and promotion-rule pattern as
`scripts/validate_cxa_event_features_on_split.py` (event-only track, branch
`analysis/cxa-p-create-feature-lock`), reused rather than reinvented:

- Boolean/categorical features: lift = rate(y_create | flag=TRUE) /
  rate(y_create | flag=FALSE) (the feature group's create rate over its own
  complement's -- see that script's docstring for why this, not the whole-population
  rate, is the definition that matches the manually-computed numbers).
- Promotion rule: direction must not flip (lift > 1x on both train and validation) AND
  validation lift >= MIN_VALIDATION_LIFT (2.0x, same threshold and justification as the
  event-only script -- unchanged here since it is a property of the lift statistic
  itself, not of which track it's applied to).

CxA+ adds two continuous 360 features that are not boolean flags, so "lift between a
flag and its complement" does not apply. These are validated by the same target-group
comparison the pre-model analysis already used for them (mean value at
`y_create=TRUE` vs `y_create=FALSE`), with an analogous, separately-justified
threshold:

- `reception_nearest_opponent_distance_m`: lower is stronger signal (tighter marking
  on a created chance), so the statistic is `gap = mean(dist | y_create=FALSE) -
  mean(dist | y_create=TRUE)` and the promotion rule is `gap > 0` (direction) AND
  `gap >= MIN_VALIDATION_DISTANCE_GAP_M` (1.0m -- chosen so the observed validation gap
  (~3.5m) clears it with a wide margin while the floor itself is still a clearly
  non-trivial fraction of a metre at this population's approximate-metres pitch scale,
  well above rounding/measurement noise).
- `reception_opponents_within_5m`: higher is stronger signal, so this is reported as a
  group-mean ratio (`lift = mean(count | y_create=TRUE) / mean(count |
  y_create=FALSE)`) and reuses the same MIN_VALIDATION_LIFT / direction rule as the
  boolean features, since it is already a ratio statistic.

Does not train, score, or select a model.
"""

from __future__ import annotations

import json
import sys

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
FEATURES_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

PLUS_MATRIX = f"`{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix`"
SPLITS_TABLE = f"`{PROJECT}.{ANALYSIS_DATASET}.cxa_match_splits_v1`"

# Same threshold, same justification as validate_cxa_event_features_on_split.py: low
# enough to accept genuine-but-modest signal, high enough that sampling noise at these
# support sizes would not reliably clear it twice (train and validation) by chance.
MIN_VALIDATION_LIFT = 2.0
# See module docstring for the justification specific to a metre-gap statistic.
MIN_VALIDATION_DISTANCE_GAP_M = 1.0

BOOLEAN_FEATURES: dict[str, str] = {
    "is_cross": "is_cross",
    "is_through_ball": "is_through_ball",
    "pass_type_name=Corner": "(pass_type_name = 'Corner')",
    "play_pattern_name=From Counter": "(play_pattern_name = 'From Counter')",
    "pass_technique_name=Outswinging": "(pass_technique_name = 'Outswinging')",
    "is_cut_back": "is_cut_back",
    "is_switch": "is_switch",
    "pass_body_part_name=No Touch": "(pass_body_part_name = 'No Touch')",
}

BOOLEAN_SQL_TEMPLATE = """
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

NUMERIC_SQL = f"""
WITH joined AS (
  SELECT m.split, e.y_create,
    e.reception_nearest_opponent_distance_m AS dist,
    e.reception_opponents_within_5m AS opp5
  FROM {PLUS_MATRIX} e
  JOIN {SPLITS_TABLE} m USING (match_id)
  WHERE m.split IN ('train', 'validation')
)
SELECT
  split,
  AVG(CASE WHEN y_create THEN dist END) AS dist_create,
  AVG(CASE WHEN NOT y_create THEN dist END) AS dist_nocreate,
  AVG(CASE WHEN NOT y_create THEN dist END) - AVG(CASE WHEN y_create THEN dist END) AS dist_gap,
  AVG(CASE WHEN y_create THEN opp5 END) AS opp5_create,
  AVG(CASE WHEN NOT y_create THEN opp5 END) AS opp5_nocreate,
  SAFE_DIVIDE(AVG(CASE WHEN y_create THEN opp5 END), AVG(CASE WHEN NOT y_create THEN opp5 END)) AS opp5_lift
FROM joined GROUP BY split ORDER BY split
"""


def run_boolean_features(client: bigquery.Client) -> tuple[dict, list[str]]:
    results: dict[str, dict] = {}
    failures: list[str] = []
    for feature_name, expr in BOOLEAN_FEATURES.items():
        sql = BOOLEAN_SQL_TEMPLATE.format(expr=expr, matrix=PLUS_MATRIX, splits=SPLITS_TABLE)
        rows = {r["split"]: dict(r.items()) for r in client.query(sql, location=LOCATION).result()}
        train_lift, val_lift = rows["train"]["lift"], rows["validation"]["lift"]
        checks = {
            "direction_holds_train": train_lift > 1.0,
            "direction_holds_validation": val_lift > 1.0,
            "validation_lift_above_threshold": val_lift >= MIN_VALIDATION_LIFT,
        }
        rows["checks"] = checks
        rows["min_validation_lift_threshold"] = MIN_VALIDATION_LIFT
        results[feature_name] = rows
        if not all(checks.values()):
            failures.append(f"{feature_name}: {checks}")
    return results, failures


def run_numeric_features(client: bigquery.Client) -> tuple[dict, list[str]]:
    rows = {r["split"]: dict(r.items()) for r in client.query(NUMERIC_SQL, location=LOCATION).result()}
    failures: list[str] = []

    dist_checks = {
        "direction_holds_train": rows["train"]["dist_gap"] > 0,
        "direction_holds_validation": rows["validation"]["dist_gap"] > 0,
        "validation_gap_above_threshold": rows["validation"]["dist_gap"] >= MIN_VALIDATION_DISTANCE_GAP_M,
    }
    if not all(dist_checks.values()):
        failures.append(f"reception_nearest_opponent_distance_m: {dist_checks}")

    opp5_checks = {
        "direction_holds_train": rows["train"]["opp5_lift"] > 1.0,
        "direction_holds_validation": rows["validation"]["opp5_lift"] > 1.0,
        "validation_lift_above_threshold": rows["validation"]["opp5_lift"] >= MIN_VALIDATION_LIFT,
    }
    if not all(opp5_checks.values()):
        failures.append(f"reception_opponents_within_5m: {opp5_checks}")

    results = {
        "reception_nearest_opponent_distance_m": {
            "train": rows["train"],
            "validation": rows["validation"],
            "checks": dist_checks,
            "min_validation_gap_threshold_m": MIN_VALIDATION_DISTANCE_GAP_M,
        },
        "reception_opponents_within_5m": {
            "train": rows["train"],
            "validation": rows["validation"],
            "checks": opp5_checks,
            "min_validation_lift_threshold": MIN_VALIDATION_LIFT,
        },
    }
    return results, failures


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    bool_results, bool_failures = run_boolean_features(client)
    numeric_results, numeric_failures = run_numeric_features(client)
    all_results = {**bool_results, **numeric_results}
    all_failures = bool_failures + numeric_failures

    print(f"{'feature':<34} {'train stat':>14} {'validation stat':>16}  verdict")
    for feature_name, rows in bool_results.items():
        verdict = "PASS" if all(rows["checks"].values()) else "FAIL"
        print(
            f"{feature_name:<34} {rows['train']['lift']:>13.2f}x "
            f"{rows['validation']['lift']:>15.2f}x  {verdict}"
        )
    dist = numeric_results["reception_nearest_opponent_distance_m"]
    verdict = "PASS" if all(dist["checks"].values()) else "FAIL"
    print(
        f"{'reception_nearest_opponent_distance_m (gap)':<34} "
        f"{dist['train']['dist_gap']:>12.2f}m {dist['validation']['dist_gap']:>15.2f}m  {verdict}"
    )
    opp5 = numeric_results["reception_opponents_within_5m"]
    verdict = "PASS" if all(opp5["checks"].values()) else "FAIL"
    print(
        f"{'reception_opponents_within_5m':<34} {opp5['train']['opp5_lift']:>13.2f}x "
        f"{opp5['validation']['opp5_lift']:>15.2f}x  {verdict}"
    )

    print()
    print(json.dumps(all_results, indent=2, default=str))

    if all_failures:
        print("\nFAILED promotion rule:")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"\nAll {len(all_results)} CxA+ features pass the promotion rule.")


if __name__ == "__main__":
    main()
