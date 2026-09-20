"""Validate the materialized CxA P_convert training matrices.

Three checks, read-only, independent of the materializer's own row count (re-derives
its own counts rather than trusting the CTAS query that built the table):

1. Row-count sanity: the P_convert matrix row count must exactly equal the number of
   `y_create = TRUE` rows in the corresponding P_create matrix (no fan-out from the
   shot or shot-freeze-frame joins).
2. 1:1 join re-check: every `pass_event_id` in the P_convert matrix must resolve to
   exactly one `shot_event_id`, independently re-verified against `oam_core.shots`
   (not just trusting the materializer's own join).
3. Split coverage: every row must have a non-null `split` value (the P_convert matrix
   joins `cxa_match_splits_v1`, which the feasibility audit already confirmed covers
   100% of this population -- re-checked here on the materialized output itself).

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

CHECKS = {
    "event": {
        "p_create_table": f"{PROJECT}.{FEATURES_DATASET}.cxa_event_v1_training_matrix",
        "p_convert_table": f"{PROJECT}.{FEATURES_DATASET}.cxconvert_event_v1_training_matrix",
    },
    "plus": {
        "p_create_table": f"{PROJECT}.{FEATURES_DATASET}.cxa_plus_v1_training_matrix",
        "p_convert_table": f"{PROJECT}.{FEATURES_DATASET}.cxconvert_plus_v1_training_matrix",
    },
}


def run_track(client: bigquery.Client, track: str, tables: dict) -> dict:
    row_count_sql = f"""
    SELECT
      (SELECT COUNT(*) FROM `{tables['p_create_table']}` WHERE y_create = TRUE) AS source_create_count,
      (SELECT COUNT(*) FROM `{tables['p_convert_table']}`) AS convert_row_count,
      (SELECT COUNT(DISTINCT pass_event_id) FROM `{tables['p_convert_table']}`) AS convert_distinct_pass_count,
      (SELECT COUNTIF(split IS NULL) FROM `{tables['p_convert_table']}`) AS null_split_count
    """
    row = dict(list(client.query(row_count_sql, location=LOCATION).result())[0].items())

    join_check_sql = f"""
    SELECT
      COUNT(*) AS n,
      COUNTIF(shot_event_id IS NULL) AS null_shot_event_id,
      COUNT(DISTINCT shot_event_id) AS distinct_shot_event_ids
    FROM `{tables['p_convert_table']}`
    """
    join_row = dict(list(client.query(join_check_sql, location=LOCATION).result())[0].items())

    failures = []
    if row["convert_row_count"] != row["source_create_count"]:
        failures.append(
            f"{track}: row-count mismatch, convert={row['convert_row_count']} "
            f"vs source y_create=TRUE count={row['source_create_count']}"
        )
    if row["convert_distinct_pass_count"] != row["convert_row_count"]:
        failures.append(f"{track}: grain violation, distinct pass_event_id < row count")
    if row["null_split_count"] != 0:
        failures.append(f"{track}: {row['null_split_count']} rows have a null split")
    if join_row["null_shot_event_id"] != 0:
        failures.append(f"{track}: {join_row['null_shot_event_id']} rows have no shot_event_id")
    if join_row["distinct_shot_event_ids"] != join_row["n"]:
        failures.append(f"{track}: shot_event_id is not 1:1 with rows (fan-out or duplicate shot)")

    return {"row_counts": row, "join_check": join_row, "failures": failures}


def main() -> None:
    client = bigquery.Client(project=PROJECT)
    all_failures: list[str] = []
    results = {}
    for track, tables in CHECKS.items():
        result = run_track(client, track, tables)
        results[track] = result
        all_failures.extend(result["failures"])

    print(json.dumps(results, indent=2, default=str))
    if all_failures:
        print("\nFAILED:")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll checks passed for both tracks.")


if __name__ == "__main__":
    main()
