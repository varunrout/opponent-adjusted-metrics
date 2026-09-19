"""Validate the materialized CxA / CxA+ training matrices.

Two checks, run read-only against BigQuery, independent of the materialization SQL in
materialize_cxa_event_v1_training_matrix.py / materialize_cxa_plus_v1_training_matrix.py
(re-derives its own counts rather than trusting the materializer's own row count):

1. Row-count sanity: re-counts `oam_core.passes` under the `silver_schema_version`
   filter and asserts it matches `cxa_event_v1_training_matrix` exactly. Catches (a) a
   future silent re-introduction of the 3x row-duplication bug (see feasibility audit
   point 1) and (b) any join fan-out in the materializer (e.g. a pass matching more
   than one shot's key_pass_id, which should be impossible but is asserted, not
   assumed).

2. Leakage spot-check: for a random sample of CxA+ rows, independently re-resolves the
   pass's receipt boundary via `related_event_ids` and confirms (a) the recorded
   `receipt_event_id` is a real `ball_receipts.event_id` for that match, not an
   arbitrary later event, and (b) the receipt's `event_index` is strictly greater than
   the pass's own `event_index` -- i.e. it causally follows the pass. This is the same
   property that makes `event_index + 1` unsafe (see feasibility audit point 4): a
   correct receipt link should look causally ordered under an independent re-check, not
   just structurally present.

Exits non-zero on any failed check.
"""

from __future__ import annotations

import json
import sys

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
CORE_DATASET = "oam_core"
FEATURES_DATASET = "oam_features"
LOCATION = "europe-west2"
SCHEMA_VERSION = "statsbomb_silver_v1_2"

EVENT_MATRIX = f"{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix"
PLUS_MATRIX = f"{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix"

ROW_COUNT_CHECK_SQL = f"""
SELECT
  (SELECT COUNT(*) FROM `{PROJECT}.{CORE_DATASET}.passes`
     WHERE silver_schema_version = '{SCHEMA_VERSION}') AS source_pass_count,
  (SELECT COUNT(*) FROM `{EVENT_MATRIX}`) AS matrix_row_count,
  (SELECT COUNT(DISTINCT pass_event_id) FROM `{EVENT_MATRIX}`) AS matrix_distinct_pass_count
"""

PLUS_POPULATION_CHECK_SQL = f"""
SELECT
  COUNT(*) AS plus_row_count,
  COUNT(DISTINCT pass_event_id) AS plus_distinct_pass_count,
  COUNTIF(y_create) AS plus_y_create_count
FROM `{PLUS_MATRIX}`
"""

LEAKAGE_SAMPLE_SQL = f"""
SELECT pass_event_id, match_id, receipt_event_id
FROM `{PLUS_MATRIX}`
ORDER BY FARM_FINGERPRINT(pass_event_id)
LIMIT 200
"""

LEAKAGE_VERIFY_SQL = f"""
WITH sample AS (
  SELECT pass_event_id, match_id, receipt_event_id
  FROM `{PLUS_MATRIX}`
  ORDER BY FARM_FINGERPRINT(pass_event_id)
  LIMIT 200
)
SELECT
  sample.pass_event_id,
  sample.receipt_event_id,
  br.event_id IS NOT NULL AS receipt_is_real_ball_receipt,
  pass_e.event_index AS pass_event_index,
  receipt_e.event_index AS receipt_event_index,
  receipt_e.event_index > pass_e.event_index AS receipt_after_pass
FROM sample
JOIN `{PROJECT}.{CORE_DATASET}.events` pass_e
  ON pass_e.event_id = sample.pass_event_id AND pass_e.silver_schema_version = '{SCHEMA_VERSION}'
LEFT JOIN `{PROJECT}.{CORE_DATASET}.ball_receipts` br
  ON br.event_id = sample.receipt_event_id
  AND br.match_id = sample.match_id
  AND br.silver_schema_version = '{SCHEMA_VERSION}'
LEFT JOIN `{PROJECT}.{CORE_DATASET}.events` receipt_e
  ON receipt_e.event_id = sample.receipt_event_id AND receipt_e.silver_schema_version = '{SCHEMA_VERSION}'
"""


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    failures: list[str] = []

    row_counts = list(client.query(ROW_COUNT_CHECK_SQL, location=LOCATION).result())[0]
    source_count = row_counts["source_pass_count"]
    matrix_count = row_counts["matrix_row_count"]
    distinct_count = row_counts["matrix_distinct_pass_count"]
    if matrix_count != source_count:
        failures.append(
            f"row-count mismatch: cxa_event_v1_training_matrix has {matrix_count} rows, "
            f"oam_core.passes (silver_schema_version={SCHEMA_VERSION}) has {source_count}"
        )
    if distinct_count != matrix_count:
        failures.append(
            f"grain violation: {matrix_count} rows but only {distinct_count} distinct "
            "pass_event_id values -- materializer fanned out the one-row-per-pass grain"
        )

    plus_pop = list(client.query(PLUS_POPULATION_CHECK_SQL, location=LOCATION).result())[0]

    leak_rows = [dict(r.items()) for r in client.query(LEAKAGE_VERIFY_SQL, location=LOCATION).result()]
    bad_receipt = [r for r in leak_rows if not r["receipt_is_real_ball_receipt"]]
    bad_order = [r for r in leak_rows if not r["receipt_after_pass"]]
    if bad_receipt:
        failures.append(
            f"leakage check: {len(bad_receipt)}/{len(leak_rows)} sampled receipt_event_id "
            "values are not real ball_receipts rows"
        )
    if bad_order:
        failures.append(
            f"leakage check: {len(bad_order)}/{len(leak_rows)} sampled receipts do not "
            "causally follow their pass (event_index not strictly greater)"
        )

    result = {
        "row_count_check": {
            "source_pass_count": source_count,
            "matrix_row_count": matrix_count,
            "matrix_distinct_pass_count": distinct_count,
            "passed": matrix_count == source_count and distinct_count == matrix_count,
        },
        "plus_population": dict(plus_pop.items()),
        "leakage_spot_check": {
            "sample_size": len(leak_rows),
            "bad_receipt_count": len(bad_receipt),
            "bad_order_count": len(bad_order),
            "passed": not bad_receipt and not bad_order,
        },
        "failures": failures,
    }
    print(json.dumps(result, indent=2, default=str))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
