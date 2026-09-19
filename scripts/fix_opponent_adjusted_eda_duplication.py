"""Fix the pre-existing 4x row-duplication bug for the 4 original `opponent_adjusted`-family
features (`defensive_profile_cluster`, `gk_odi`, `mean_backline_odi`, `nearest_defender_odi`)
across all 5 tables `materialize_cxg_opponent_adjusted_analysis.py` writes to (the 4 EDA-
appendix tables the task named, plus `cxg_feature_inventory_v1`, which shares the identical
root cause and was found duplicated too during investigation).

ROOT CAUSE (confirmed, not assumed): `materialize_cxg_opponent_adjusted_analysis.py` writes
every one of these 5 tables via plain `INSERT INTO` with no `DELETE` anywhere in the file
(confirmed by reading the whole script -- zero `DELETE` statements). It was run 4 times
historically (4 distinct `materialized_at` timestamps per feature, confirmed live, with
IDENTICAL row_count/null_count/etc. each time -- the underlying data never changed between
runs, only the script was re-invoked without an idempotency guard). `cxg_split_univariate_v1`
is NOT affected by this same bug -- its rows are legitimately run_id-tagged historical
snapshots (the established convention for that one table), not literal duplicates, and is
correctly left untouched here.

Fix: for each table, keep exactly the most-recently-materialized row per feature (content is
identical across the 4 historical copies, confirmed live, so "most recent" is a safe,
lossless choice) -- fetch the deduped rows, scoped-DELETE all rows for these 4 features, then
re-insert exactly the deduped set. `materialize_cxg_opponent_adjusted_analysis.py` itself is
separately patched (this task) to add a scoped DELETE-before-INSERT guard so this can't recur.
"""

from __future__ import annotations

import datetime as dt
import json

from google.cloud import bigquery


def _json_safe(value: object) -> object:
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    return value

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"
FAMILY = "opponent_adjusted"
FEATURES = ["defensive_profile_cluster", "gk_odi", "mean_backline_odi", "nearest_defender_odi"]


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _count(client: bigquery.Client, table: str) -> int:
    sql = f"SELECT COUNT(*) FROM {q(table)} WHERE feature_family = '{FAMILY}' AND column_name IN UNNEST({FEATURES!r})"
    return int(list(client.query(sql, location=LOCATION).result())[0][0])


def _dedupe_table(client: bigquery.Client, table: str, partition_cols: list[str]) -> dict:
    before = _count(client, table)

    partition_sql = ", ".join(f"`{c}`" for c in partition_cols)
    dedup_sql = f"""
    SELECT * EXCEPT(rn) FROM (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_sql} ORDER BY materialized_at DESC) AS rn
      FROM {q(table)}
      WHERE feature_family = '{FAMILY}' AND column_name IN UNNEST({FEATURES!r})
    )
    WHERE rn = 1
    """
    deduped_rows = [dict(r.items()) for r in client.query(dedup_sql, location=LOCATION).result()]

    client.query(
        f"DELETE FROM {q(table)} WHERE feature_family = '{FAMILY}' AND column_name IN UNNEST({FEATURES!r})",
        location=LOCATION,
    ).result()

    if deduped_rows:
        table_ref = client.get_table(f"{PROJECT}.{ANALYSIS_DATASET}.{table}")
        schema = table_ref.schema
        clean_rows = [{k: _json_safe(r.get(k)) for k in (f.name for f in schema)} for r in deduped_rows]
        client.load_table_from_json(
            clean_rows, f"{PROJECT}.{ANALYSIS_DATASET}.{table}",
            job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND"),
            location=LOCATION,
        ).result()

    after = _count(client, table)
    return {"before": before, "after": after, "kept": len(deduped_rows)}


def main() -> None:
    client = _client()
    results = {}

    results["cxg_null_profile_v1"] = _dedupe_table(client, "cxg_null_profile_v1", ["feature_family", "table_name", "column_name"])
    results["cxg_summary_stats_v1"] = _dedupe_table(client, "cxg_summary_stats_v1", ["feature_family", "table_name", "column_name"])
    results["cxg_eda_distribution_bins_v1"] = _dedupe_table(client, "cxg_eda_distribution_bins_v1", ["feature_family", "column_name", "bin_type", "bin_label"])
    results["cxg_univariate_target_v1"] = _dedupe_table(client, "cxg_univariate_target_v1", ["feature_family", "column_name"])
    results["cxg_feature_inventory_v1"] = _dedupe_table(client, "cxg_feature_inventory_v1", ["feature_family", "source_table", "column_name"])

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
