"""Backfill the 4 EDA-appendix analysis tables for the 5 new CxG+ candidates (Phase A/B),
which the v2 pool requalification task explicitly did not touch (it only refreshed
`feature_correlation_heatmap`/`pca_scree`/`bivariate_significance_grid`).

Scoped, idempotent extension of `cxg_null_profile_v1`, `cxg_summary_stats_v1`,
`cxg_eda_distribution_bins_v1`, `cxg_univariate_target_v1` -- adds rows for exactly the 5
new column names under `feature_family='opponent_adjusted'`; the 18 already-covered
opponent_adjusted-family features (ODI trio + defensive_profile_cluster) and every other
family's rows are untouched. Follows `scripts/materialize_cxg_opponent_adjusted_analysis.py`'s
exact conventions (categorical bin_type='category' with an explicit 'null' bin; continuous
bin_type='quantile' via NTILE(20); univariate target computed TRAIN-SPLIT ONLY per the split
policy) -- but adds a scoped DELETE before each INSERT, since that precedent script has no
idempotency guard of its own (confirmed by inspection: re-running it duplicates rows).

Also refreshes `cxg_analysis_opponent_adjusted_v1` (the surface table `cxg_charts.py`'s
`_eda_bins` numeric branch queries live for histogram values) -- discovered mid-task to be
completely unpopulated (0 rows, and missing all 5 new columns), which would otherwise either
silently render blank continuous-feature histograms (pre-existing behavior for the 3 ODI
features) or hard-fail with "Unrecognized name" for the 2 new continuous features (columns
that don't exist at all). This is a necessary enabling fix for Part 2's chart deliverable,
not new analysis -- CREATE OR REPLACE, matching the existing convention for analysis surface
tables, scoped to exactly this one table.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from google.cloud import bigquery

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

SOURCE_TABLE = "cxg_defensive_360_features"
MODEL_MATRIX = "cxg_plus_360_model_matrix_v1"
FAMILY = "opponent_adjusted"

NEW_CONTINUOUS = ["nearest_defender_zone_displacement", "nearest_defender_gap"]
NEW_CATEGORICAL = ["nearest_defender_role", "second_nearest_defender_role", "nearest_defender_style_archetype"]
NEW_FEATURES = NEW_CONTINUOUS + NEW_CATEGORICAL


def q(table: str, dataset: str = ANALYSIS_DATASET) -> str:
    return f"`{PROJECT}.{dataset}.{table}`"


def _client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT)


def _run(client: bigquery.Client, sql: str) -> None:
    client.query(sql, location=LOCATION).result()


def _count(client: bigquery.Client, sql: str) -> int:
    rows = list(client.query(sql, location=LOCATION).result())
    return int(rows[0][0]) if rows else 0


def _delete_scoped(client: bigquery.Client, table: str, cols: list[str]) -> None:
    _run(
        client,
        f"DELETE FROM {q(table)} WHERE feature_family = '{FAMILY}' AND column_name IN UNNEST({cols!r})",
    )


def materialize_null_profile(client: bigquery.Client) -> None:
    _delete_scoped(client, "cxg_null_profile_v1", NEW_FEATURES)
    selects = []
    for col in NEW_FEATURES:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{SOURCE_TABLE}' AS table_name, '{col}' AS column_name,
          COUNT(*) AS row_count,
          COUNTIF(`{col}` IS NULL) AS null_count,
          COUNTIF(`{col}` IS NOT NULL) AS non_null_count,
          SAFE_DIVIDE(COUNTIF(`{col}` IS NULL), COUNT(*)) AS null_pct,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(SOURCE_TABLE, FEATURE_DATASET)}
        """)
    _run(client, f"INSERT INTO {q('cxg_null_profile_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_summary_stats(client: bigquery.Client) -> None:
    _delete_scoped(client, "cxg_summary_stats_v1", NEW_FEATURES)
    selects = []
    for col in NEW_CONTINUOUS:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{SOURCE_TABLE}' AS table_name, '{col}' AS column_name,
          'FLOAT64' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(`{col}` IS NOT NULL) AS non_null_count,
          AVG(`{col}`) AS mean_value,
          STDDEV(`{col}`) AS stddev_value,
          MIN(`{col}`) AS min_value,
          APPROX_QUANTILES(`{col}`, 100)[OFFSET(25)] AS p25_value,
          APPROX_QUANTILES(`{col}`, 100)[OFFSET(50)] AS median_value,
          APPROX_QUANTILES(`{col}`, 100)[OFFSET(75)] AS p75_value,
          MAX(`{col}`) AS max_value,
          COUNT(DISTINCT `{col}`) AS distinct_count,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(SOURCE_TABLE, FEATURE_DATASET)}
        WHERE `{col}` IS NOT NULL
        """)
    for col in NEW_CATEGORICAL:
        # Categorical: distinct_count only, no numeric moments -- matches
        # defensive_profile_cluster's own convention (a category label's mean/stddev is
        # not a meaningful physical quantity).
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{SOURCE_TABLE}' AS table_name, '{col}' AS column_name,
          'STRING' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(`{col}` IS NOT NULL) AS non_null_count,
          NULL AS mean_value, NULL AS stddev_value, NULL AS min_value,
          NULL AS p25_value, NULL AS median_value, NULL AS p75_value, NULL AS max_value,
          COUNT(DISTINCT `{col}`) AS distinct_count,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(SOURCE_TABLE, FEATURE_DATASET)}
        """)
    _run(client, f"INSERT INTO {q('cxg_summary_stats_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_distribution_bins(client: bigquery.Client) -> None:
    _delete_scoped(client, "cxg_eda_distribution_bins_v1", NEW_FEATURES)
    selects = []
    for col in NEW_CONTINUOUS:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{col}' AS column_name, 'quantile' AS bin_type,
          CAST(bin_number AS STRING) AS bin_label, COUNT(*) AS row_count, CURRENT_TIMESTAMP() AS materialized_at
        FROM (
          SELECT NTILE(20) OVER (ORDER BY `{col}`) AS bin_number
          FROM {q(SOURCE_TABLE, FEATURE_DATASET)}
          WHERE `{col}` IS NOT NULL
        )
        GROUP BY bin_label
        """)
    for col in NEW_CATEGORICAL:
        # Explicit 'null' bin included -- matches the opponent_adjusted-family extension's
        # deliberate deviation from the base materializer's STRING-categorical path (which
        # drops nulls); geometry-ineligibility / below-clustering-threshold is a first-class
        # outcome worth surfacing, not hiding.
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{col}' AS column_name, 'category' AS bin_type,
          COALESCE(`{col}`, 'null') AS bin_label, COUNT(*) AS row_count, CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(SOURCE_TABLE, FEATURE_DATASET)}
        GROUP BY bin_label
        """)
    _run(client, f"INSERT INTO {q('cxg_eda_distribution_bins_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_univariate_target(client: bigquery.Client) -> None:
    """TRAIN SPLIT ONLY, per the split policy this table's existing opponent_adjusted rows
    already follow (see materialize_cxg_opponent_adjusted_analysis.py's own docstring)."""
    _delete_scoped(client, "cxg_univariate_target_v1", NEW_FEATURES)
    selects = []
    for col in NEW_CONTINUOUS:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{col}' AS column_name, 'FLOAT64' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(x.`{col}` IS NOT NULL) AS non_null_count,
          AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate,
          AVG(x.`{col}`) AS mean_when_available,
          AVG(IF(m.is_goal, x.`{col}`, NULL)) AS mean_for_goals,
          AVG(IF(NOT m.is_goal, x.`{col}`, NULL)) AS mean_for_non_goals,
          CORR(x.`{col}`, IF(m.is_goal, 1.0, 0.0)) AS point_biserial_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(MODEL_MATRIX)} m
        JOIN {q(SOURCE_TABLE, FEATURE_DATASET)} x ON m.event_id = x.event_id
        WHERE m.split = 'train'
        """)
    for col in NEW_CATEGORICAL:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{col}' AS column_name, 'STRING' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(x.`{col}` IS NOT NULL) AS non_null_count,
          AVG(IF(m.is_goal, 1.0, 0.0)) AS goal_rate,
          NULL AS mean_when_available, NULL AS mean_for_goals, NULL AS mean_for_non_goals,
          NULL AS point_biserial_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(MODEL_MATRIX)} m
        JOIN {q(SOURCE_TABLE, FEATURE_DATASET)} x ON m.event_id = x.event_id
        WHERE m.split = 'train'
        """)
    _run(client, f"INSERT INTO {q('cxg_univariate_target_v1')}\n" + "\nUNION ALL\n".join(selects))


def refresh_surface_table(client: bigquery.Client) -> None:
    """`cxg_analysis_opponent_adjusted_v1` is a VIEW (not a table -- `get_table().num_rows`
    reports 0 for any view regardless of actual content, the same metadata-API quirk already
    seen with `cxg_training_matrix_v1`; querying it directly confirms 3,960 real rows). Not
    broken, just missing the 5 new columns. `CREATE OR REPLACE VIEW`, adding one more LEFT
    JOIN to the same source table Phase A/B already write to, otherwise identical to its
    existing (live, working) definition -- confirmed via `get_table().view_query` before
    writing this, not assumed."""
    sql = f"""
    CREATE OR REPLACE VIEW {q('cxg_analysis_opponent_adjusted_v1')} AS
    SELECT
      a.event_id, a.match_id, a.player_id, a.team_id, a.is_goal, a.outcome_name,
      a.shot_x_sb, a.shot_y_sb, a.has_360_frame,
      o.nearest_defender_odi, o.mean_backline_odi, o.gk_odi,
      c.cluster_label AS defensive_profile_cluster,
      c.split AS defensive_profile_cluster_split,
      c.cluster_model_version AS defensive_profile_cluster_model_version,
      d.nearest_defender_role, d.nearest_defender_zone_displacement,
      d.second_nearest_defender_role, d.nearest_defender_gap,
      d.nearest_defender_style_archetype
    FROM {q('cxg_analysis_360_v1')} a
    LEFT JOIN {q('cxg_odi_features_v1')} o ON a.event_id = o.event_id
    LEFT JOIN {q('cxg_defensive_profile_clusters_v1')} c ON a.event_id = c.event_id
    LEFT JOIN {q(SOURCE_TABLE, FEATURE_DATASET)} d ON a.event_id = d.event_id
    """
    _run(client, sql)


def main() -> None:
    client = _client()

    before = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)} WHERE feature_family='{FAMILY}'")
        for t in ("cxg_null_profile_v1", "cxg_summary_stats_v1", "cxg_eda_distribution_bins_v1", "cxg_univariate_target_v1")
    }

    materialize_null_profile(client)
    materialize_summary_stats(client)
    materialize_distribution_bins(client)
    materialize_univariate_target(client)
    refresh_surface_table(client)

    after = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)} WHERE feature_family='{FAMILY}'")
        for t in before
    }
    surface_n = _count(client, f"SELECT COUNT(*) FROM {q('cxg_analysis_opponent_adjusted_v1')}")

    print(json.dumps({"before": before, "after": after, "delta": {t: after[t] - before[t] for t in before}, "surface_table_rows": surface_n}, indent=2))


if __name__ == "__main__":
    main()
