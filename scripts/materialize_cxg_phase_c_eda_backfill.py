"""Backfill the 4 EDA-appendix tables for Phase C's 3 requalification-surviving features
(`defensive_action_rate_30m`, `territorial_dominance_last_15m`, `cross_match_defensive_rate`)
-- confirmed live before writing this script: `cxg_summary_stats_v1` (and the other 3
EDA-appendix tables) have zero rows for any of these 3 column names, same gap pattern as the
earlier Phase A/B EDA backfill (`materialize_cxg_v2_eda_backfill.py` is the precedent this
script follows for scope/method).

Family = `event_context` (where these 3 columns physically live, `cxg_event_context_features`
-- confirmed via `oam_features.INFORMATION_SCHEMA.COLUMNS` before writing this). Unlike the
opponent_adjusted family (CxG+-only, has its own bolt-on script), `event_context` IS one of
the 7 families the canonical `CxGAnalysisMaterializer` (`src/opponent_adjusted/analysis/
cxg.py`) already owns -- but that class's `run()` unconditionally `CREATE OR REPLACE`s
`cxg_chart_registry_v1`/`cxg_correlation_v1`/etc. wholesale, which would silently wipe every
v2/v3/Phase-C chart-registry row and analysis-table row this session has built up via the
INSERT-only copy-forward convention. This script therefore does NOT call `.run()` -- it
reuses the class's own `_load_fields()`/`_materialize_analysis_surfaces()` methods directly
(both pure `CREATE OR REPLACE TABLE` rebuilds of derived surface tables with no manually-
inserted rows to lose, safe to fully rebuild) to refresh `cxg_analysis_event_v1` with the 3
new columns, and otherwise writes the 4 EDA-appendix tables via the SAME scoped-DELETE-then-
INSERT idempotent pattern already established by `materialize_cxg_v2_eda_backfill.py` and
this session's own `fix_opponent_adjusted_eda_duplication.py` idempotency-guard fix.

BOTH TRACKS: unlike `opponent_adjusted` (CxG+-only), `event_context` has never been track-
split in these 4 tables (confirmed live: every existing event_context row is a single,
unscoped stat over the full population -- these tables carry no `track` column at all).
Followed that same established convention rather than inventing a track split no other
event_context feature has: one row per feature, computed over the full population
(`cxg_event_context_features`, all 15,737 shots = the CxG event-wide population; CxG+ is
strictly the `has_360_frame` subset of it, so this single stat is valid context for both
tracks, exactly like every other event_context feature already in these tables).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from google.cloud import bigquery

from opponent_adjusted.analysis.cxg import CxGAnalysisMaterializer

PROJECT = "oam-varun-260819"
ANALYSIS_DATASET = "oam_analysis"
FEATURE_DATASET = "oam_features"
LOCATION = "europe-west2"

FAMILY = "event_context"
SOURCE_TABLE = "cxg_event_context_features"
SURFACE_TABLE = "cxg_analysis_event_v1"
NEW_FEATURES = ["defensive_action_rate_30m", "territorial_dominance_last_15m", "cross_match_defensive_rate"]


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
    cols_sql = ", ".join(f"'{c}'" for c in cols)
    _run(client, f"DELETE FROM {q(table)} WHERE feature_family = '{FAMILY}' AND column_name IN ({cols_sql})")


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
    for col in NEW_FEATURES:
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
    _run(client, f"INSERT INTO {q('cxg_summary_stats_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_distribution_bins(client: bigquery.Client) -> None:
    _delete_scoped(client, "cxg_eda_distribution_bins_v1", NEW_FEATURES)
    selects = []
    for col in NEW_FEATURES:
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
    _run(client, f"INSERT INTO {q('cxg_eda_distribution_bins_v1')}\n" + "\nUNION ALL\n".join(selects))


def materialize_univariate_target(client: bigquery.Client) -> None:
    """Full-dataset (not train-split-only) -- matches every other event_context row already
    in this table (event_context is not opponent_adjusted's train-only split-policy case)."""
    _delete_scoped(client, "cxg_univariate_target_v1", NEW_FEATURES)
    selects = []
    for col in NEW_FEATURES:
        selects.append(f"""
        SELECT
          '{FAMILY}' AS feature_family, '{col}' AS column_name, 'FLOAT64' AS data_type,
          COUNT(*) AS row_count,
          COUNTIF(`{col}` IS NOT NULL) AS non_null_count,
          AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
          AVG(`{col}`) AS mean_when_available,
          AVG(IF(is_goal, `{col}`, NULL)) AS mean_for_goals,
          AVG(IF(NOT is_goal, `{col}`, NULL)) AS mean_for_non_goals,
          CORR(`{col}`, IF(is_goal, 1.0, 0.0)) AS point_biserial_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {q(SURFACE_TABLE)}
        """)
    _run(client, f"INSERT INTO {q('cxg_univariate_target_v1')}\n" + "\nUNION ALL\n".join(selects))


def main() -> None:
    client = _client()

    before = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)} WHERE feature_family = '{FAMILY}' AND column_name IN ({', '.join(repr(c) for c in NEW_FEATURES)})")
        for t in ("cxg_null_profile_v1", "cxg_summary_stats_v1", "cxg_eda_distribution_bins_v1", "cxg_univariate_target_v1")
    }

    print("[surface] refreshing cxg_analysis_event_v1 (CREATE OR REPLACE, reusing CxGAnalysisMaterializer's own methods)...")
    materializer = CxGAnalysisMaterializer()
    fields = materializer._load_fields()
    materializer._materialize_analysis_surfaces(fields)

    materialize_null_profile(client)
    materialize_summary_stats(client)
    materialize_distribution_bins(client)
    materialize_univariate_target(client)

    after = {
        t: _count(client, f"SELECT COUNT(*) FROM {q(t)} WHERE feature_family = '{FAMILY}' AND column_name IN ({', '.join(repr(c) for c in NEW_FEATURES)})")
        for t in before
    }
    surface_cols = _count(client, f"""
        SELECT COUNT(*) FROM `{PROJECT}.{ANALYSIS_DATASET}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{SURFACE_TABLE}' AND column_name IN ({', '.join(repr(c) for c in NEW_FEATURES)})
    """)
    print(json.dumps({"before": before, "after": after, "delta": {t: after[t] - before[t] for t in before}, "surface_table_new_columns": surface_cols}, indent=2))


if __name__ == "__main__":
    main()
