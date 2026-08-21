"""Extend the CxG+ analysis surfaces with the new `opponent_adjusted` family.

Registers ODI (nearest_defender_odi, mean_backline_odi, gk_odi) and
defensive_profile_cluster as a fourth governed 360-population family,
alongside defensive_360/goalkeeper_360/line_shape_360 -- see
`opponent_adjusted.features.cxg.contracts.OPPONENT_ADJUSTED_FAMILIES`.

This is a surgical, additive extension of the existing governed analysis
tables (`CxGAnalysisMaterializer` in `analysis/cxg.py`), NOT a call to its
`run()` method: `run()` unconditionally rebuilds `cxg_correlation_v1`
(bivariate) and recomputes `cxg_univariate_target_v1` on the full dataset,
both of which are explicitly out of scope for this extension (see
`docs/cxg_split_policy_and_parallel_plan.md` -- bivariate/correlation stays
reverted, and univariate for this family must be train-split-only). Every
statement below is an INSERT of new rows for the new family only; no
existing row for any other family is touched.

Steps 1-3 (inventory, null profile, summary stats, EDA bins) are computed
full-dataset -- explicitly permitted by the split policy doc ("Full-dataset
EDA, null reports, summary statistics ... are acceptable because they
describe the corpus"). Step 6 (univariate target) is computed TRAIN-SPLIT
ONLY, joined through `cxg_plus_360_model_matrix_v1.split = 'train'` -- the
one step the split policy requires to be split-aware to avoid overfitting
feature selection to the full dataset.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
LOCATION = "europe-west2"

ODI_TABLE = "cxg_odi_features_v1"
CLUSTERS_TABLE = "cxg_defensive_profile_clusters_v1"
MODEL_MATRIX = "cxg_plus_360_model_matrix_v1"

# The 4 governed opponent_adjusted candidates. `column` is the physical
# column read; `name` is the governed candidate name (identical except for
# the cluster label, which is stored as `cluster_label` but registered under
# the taxonomy name `defensive_profile_cluster`).
NUMERIC_FEATURES = [
    {"name": "nearest_defender_odi", "column": "nearest_defender_odi", "table": ODI_TABLE, "data_type": "FLOAT64"},
    {"name": "mean_backline_odi", "column": "mean_backline_odi", "table": ODI_TABLE, "data_type": "FLOAT64"},
    {"name": "gk_odi", "column": "gk_odi", "table": ODI_TABLE, "data_type": "FLOAT64"},
]
CATEGORICAL_FEATURE = {
    "name": "defensive_profile_cluster",
    "column": "cluster_label",
    "table": CLUSTERS_TABLE,
    "data_type": "INT64",
}
FAMILY = "opponent_adjusted"


def q(table: str) -> str:
    return f"`{PROJECT}.{ANALYSIS_DATASET}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _run(client: bigquery.Client, sql: str) -> None:
    client.query(sql, location=LOCATION).result()


def _count(client: bigquery.Client, sql: str) -> int:
    rows = list(client.query(sql, location=LOCATION).result())
    return int(rows[0][0]) if rows else 0


def materialize_inventory(client: bigquery.Client) -> None:
    now = datetime.now(UTC).isoformat()
    selects = []
    for f in NUMERIC_FEATURES:
        selects.append(
            f"SELECT '{FAMILY}' AS feature_family, '{f['table']}' AS source_table, "
            f"'{f['name']}' AS column_name, '{f['data_type']}' AS data_type, 'feature' AS column_role, "
            f"TRUE AS is_numeric, FALSE AS is_categorical, TIMESTAMP('{now}') AS materialized_at"
        )
    c = CATEGORICAL_FEATURE
    selects.append(
        f"SELECT '{FAMILY}' AS feature_family, '{c['table']}' AS source_table, "
        f"'{c['name']}' AS column_name, '{c['data_type']}' AS data_type, 'feature' AS column_role, "
        f"FALSE AS is_numeric, TRUE AS is_categorical, TIMESTAMP('{now}') AS materialized_at"
    )
    _run(client, f"INSERT INTO {q('cxg_feature_inventory_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_null_profile(client: bigquery.Client) -> None:
    selects = []
    for f in NUMERIC_FEATURES:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{f['table']}' AS table_name, '{f['name']}' AS column_name,
          COUNT(*) AS row_count,
          COUNTIF(`{f['column']}` IS NULL) AS null_count,
          COUNTIF(`{f['column']}` IS NOT NULL) AS non_null_count,
          SAFE_DIVIDE(COUNTIF(`{f['column']}` IS NULL), COUNT(*)) AS null_pct,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(f['table'])}
        """)
    c = CATEGORICAL_FEATURE
    selects.append(f"""
    SELECT
      '{FAMILY}' AS feature_family, '{c['table']}' AS table_name, '{c['name']}' AS column_name,
      COUNT(*) AS row_count,
      COUNTIF(`{c['column']}` IS NULL) AS null_count,
      COUNTIF(`{c['column']}` IS NOT NULL) AS non_null_count,
      SAFE_DIVIDE(COUNTIF(`{c['column']}` IS NULL), COUNT(*)) AS null_pct,
      CURRENT_TIMESTAMP() AS materialized_at
    FROM {q(c['table'])}
    """)
    _run(client, f"INSERT INTO {q('cxg_null_profile_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_summary_stats(client: bigquery.Client) -> None:
    selects = []
    for f in NUMERIC_FEATURES:
        col = f"`{f['column']}`"
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{f['table']}' AS table_name, '{f['name']}' AS column_name,
          '{f['data_type']}' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF({col} IS NOT NULL) AS non_null_count,
          AVG({col}) AS mean_value,
          STDDEV({col}) AS stddev_value,
          MIN({col}) AS min_value,
          APPROX_QUANTILES({col}, 100)[OFFSET(25)] AS p25_value,
          APPROX_QUANTILES({col}, 100)[OFFSET(50)] AS median_value,
          APPROX_QUANTILES({col}, 100)[OFFSET(75)] AS p75_value,
          MAX({col}) AS max_value,
          COUNT(DISTINCT {col}) AS distinct_count,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(f['table'])}
        WHERE {col} IS NOT NULL
        """)
    # Categorical (defensive_profile_cluster): report distinct_count only, no numeric moments
    # -- treating a cluster label's mean/stddev as a physical quantity would be meaningless.
    c = CATEGORICAL_FEATURE
    col = f"`{c['column']}`"
    selects.append(f"""
    SELECT
      '{FAMILY}' AS feature_family, '{c['table']}' AS table_name, '{c['name']}' AS column_name,
      '{c['data_type']}' AS data_type,
      COUNT(*) AS row_count,
      COUNTIF({col} IS NOT NULL) AS non_null_count,
      NULL AS mean_value, NULL AS stddev_value, NULL AS min_value,
      NULL AS p25_value, NULL AS median_value, NULL AS p75_value, NULL AS max_value,
      COUNT(DISTINCT {col}) AS distinct_count,
      CURRENT_TIMESTAMP() AS materialized_at
    FROM {q(c['table'])}
    """)
    _run(client, f"INSERT INTO {q('cxg_summary_stats_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_distribution_bins(client: bigquery.Client) -> None:
    selects = []
    for f in NUMERIC_FEATURES:
        col = f"`{f['column']}`"
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{f['name']}' AS column_name, 'quantile' AS bin_type,
          CAST(bin_number AS STRING) AS bin_label, COUNT(*) AS row_count, CURRENT_TIMESTAMP() AS materialized_at
        FROM (
          SELECT NTILE(20) OVER (ORDER BY {col}) AS bin_number
          FROM {q(f['table'])}
          WHERE {col} IS NOT NULL
        )
        GROUP BY bin_label
        """)
    # Categorical: 4 category bins + an explicit 'null' bin (deliberate deviation from the
    # existing STRING-categorical pattern, which filters nulls out -- the task asks for the
    # null bin to be visible here since geometry-ineligibility is a first-class outcome).
    c = CATEGORICAL_FEATURE
    col = f"`{c['column']}`"
    selects.append(f"""
    SELECT
      '{FAMILY}' AS feature_family, '{c['name']}' AS column_name, 'category' AS bin_type,
      COALESCE(CAST({col} AS STRING), 'null') AS bin_label, COUNT(*) AS row_count, CURRENT_TIMESTAMP() AS materialized_at
    FROM {q(c['table'])}
    GROUP BY bin_label
    """)
    _run(client, f"INSERT INTO {q('cxg_eda_distribution_bins_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_univariate_target_train_only(client: bigquery.Client) -> None:
    """Extends cxg_univariate_target_v1 with 4 opponent_adjusted rows, TRAIN SPLIT ONLY.

    Every other row in this table is full-dataset (360-cohort or event-wide); these 4
    rows are computed against `cxg_plus_360_model_matrix_v1 WHERE split = 'train'` only,
    per the split policy's train-only feature-confirmation discipline. This is a
    deliberate, documented population difference within the same table -- flagged here
    and in the closure report since the table itself carries no split column.
    """
    selects = []
    for f in NUMERIC_FEATURES:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{f['name']}' AS column_name, '{f['data_type']}' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(x.`{f['column']}` IS NOT NULL) AS non_null_count,
          AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate,
          AVG(x.`{f['column']}`) AS mean_when_available,
          AVG(IF(m.is_goal, x.`{f['column']}`, NULL)) AS mean_for_goals,
          AVG(IF(NOT m.is_goal, x.`{f['column']}`, NULL)) AS mean_for_non_goals,
          CORR(x.`{f['column']}`, IF(m.is_goal, 1.0, 0.0)) AS point_biserial_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(MODEL_MATRIX)} m
        JOIN {q(f['table'])} x ON m.event_id = x.event_id
        WHERE m.split = 'train'
        """)
    c = CATEGORICAL_FEATURE
    selects.append(f"""
    SELECT
      '{FAMILY}' AS feature_family, '{c['name']}' AS column_name, '{c['data_type']}' AS data_type,
      COUNT(*) AS row_count,
      COUNTIF(x.`{c['column']}` IS NOT NULL) AS non_null_count,
      AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate,
      NULL AS mean_when_available, NULL AS mean_for_goals, NULL AS mean_for_non_goals,
      NULL AS point_biserial_corr,
      CURRENT_TIMESTAMP() AS materialized_at
    FROM {q(MODEL_MATRIX)} m
    JOIN {q(c['table'])} x ON m.event_id = x.event_id
    WHERE m.split = 'train'
    """)
    _run(client, f"INSERT INTO {q('cxg_univariate_target_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_split_univariate(client: bigquery.Client, run_id: str) -> None:
    """Extends cxg_split_univariate_v1 (already split/track-tagged) for all 3 splits --
    train rows are the feature-confirmation numbers; validation/test rows are descriptive
    stability visibility only, matching this table's existing convention for every other
    family (never used on their own to select features)."""
    selects = []
    for f in NUMERIC_FEATURES:
        selects.append(f"""
        SELECT
          '{run_id}' AS run_id, 'cxg_plus' AS track, m.split,
          '{FAMILY}' AS feature_family, '{f['name']}' AS column_name, '{f['data_type']}' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(x.`{f['column']}` IS NULL) AS null_count,
          COUNTIF(x.`{f['column']}` IS NOT NULL) AS non_null_count,
          SAFE_DIVIDE(COUNTIF(x.`{f['column']}` IS NULL), COUNT(*)) AS null_pct,
          COUNTIF(m.is_goal) AS goal_count,
          AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate,
          CORR(x.`{f['column']}`, IF(m.is_goal, 1.0, 0.0)) AS point_biserial_corr,
          ABS(CORR(x.`{f['column']}`, IF(m.is_goal, 1.0, 0.0))) AS abs_signal,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(MODEL_MATRIX)} m
        JOIN {q(f['table'])} x ON m.event_id = x.event_id
        GROUP BY m.split
        """)
    _run(client, f"INSERT INTO {q('cxg_split_univariate_v1')}\n" + "\nUNION ALL\n".join(selects))


def main() -> None:
    client = _client()
    run_id = "cxg-analysis-opponent-adjusted-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    before = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)}")
        for t in (
            "cxg_feature_inventory_v1",
            "cxg_null_profile_v1",
            "cxg_summary_stats_v1",
            "cxg_eda_distribution_bins_v1",
            "cxg_univariate_target_v1",
            "cxg_split_univariate_v1",
        )
    }

    materialize_inventory(client)
    materialize_null_profile(client)
    materialize_summary_stats(client)
    materialize_distribution_bins(client)
    materialize_univariate_target_train_only(client)
    materialize_split_univariate(client, run_id)

    after = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)}")
        for t in before
    }
    print(json.dumps({"run_id": run_id, "before": before, "after": after, "delta": {t: after[t] - before[t] for t in before}}, indent=2))


if __name__ == "__main__":
    main()
