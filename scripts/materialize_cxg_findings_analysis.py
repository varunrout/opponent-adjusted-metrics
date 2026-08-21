"""Materialize findings-grade CxG analysis tables from the clean analysis surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from google.cloud import bigquery

from opponent_adjusted.analysis.cxg import ANALYSIS_DATASET, LOCATION, PROJECT_ID

ACTIVE_RUN_ID = "cxg-analysis-20260820T201934Z"
CONTEXT_FAMILIES = {"event_context", "defensive_360", "goalkeeper_360", "line_shape_360"}
CONTROL_FAMILY = "base_identity_target"
MIN_PAIR_SUPPORT_EVENT = 1000
MIN_PAIR_SUPPORT_360 = 400
MAX_NUMERIC_FEATURES_PER_FAMILY = 28
MAX_INTERACTION_FEATURES_PER_FAMILY = 12


@dataclass(frozen=True)
class Feature:
    family: str
    column: str
    data_type: str
    non_null_count: int
    null_pct: float
    abs_signal: float

    @property
    def is_numeric(self) -> bool:
        return self.data_type.upper() in {"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC"}

    @property
    def is_categorical(self) -> bool:
        return self.data_type.upper() in {"STRING", "BOOL"}


def q(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "\\'") + "'"


class FindingsMaterializer:
    def __init__(self, run_id: str, dry_run: bool = False) -> None:
        self.run_id = run_id
        self.dry_run = dry_run
        self.bq = bigquery.Client(project=PROJECT_ID)

    def run(self) -> dict[str, object]:
        features = self.load_features()
        self.materialize_binned_target_response(features)
        self.materialize_cross_family_correlations(features)
        self.materialize_pair_interactions(features)
        self.materialize_redundancy_clusters()
        self.materialize_decision_manifest()
        return self.validate_manifest()

    def query(self, sql: str) -> None:
        if self.dry_run:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bq.query(sql, location=LOCATION, job_config=job_config)
            return
        self.bq.query(sql, location=LOCATION).result()

    def load_features(self) -> list[Feature]:
        sql = f"""
        SELECT
          i.feature_family,
          i.column_name,
          i.data_type,
          COALESCE(n.non_null_count, 0) AS non_null_count,
          COALESCE(n.null_pct, 1.0) AS null_pct,
          ABS(COALESCE(u.point_biserial_corr, 0.0)) AS abs_signal
        FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_feature_inventory_v1')} i
        LEFT JOIN {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_null_profile_v1')} n
          ON i.feature_family = n.feature_family AND i.column_name = n.column_name
        LEFT JOIN {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_univariate_target_v1')} u
          ON i.feature_family = u.feature_family AND i.column_name = u.column_name
        WHERE i.column_role = 'feature'
          AND i.feature_family IN ({', '.join(quote_literal(f) for f in sorted(CONTEXT_FAMILIES | {CONTROL_FAMILY}))})
        ORDER BY i.feature_family, i.column_name
        """
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return [
            Feature(
                family=row.feature_family,
                column=row.column_name,
                data_type=row.data_type,
                non_null_count=int(row.non_null_count or 0),
                null_pct=float(row.null_pct or 0),
                abs_signal=float(row.abs_signal or 0),
            )
            for row in rows
        ]

    def surface_for(self, family: str) -> str:
        return "cxg_analysis_360_v1" if family.endswith("_360") else "cxg_analysis_event_v1"

    def materialize_binned_target_response(self, features: list[Feature]) -> None:
        target = q(PROJECT_ID, ANALYSIS_DATASET, "cxg_binned_target_response_v1")
        self.query(f"""
        CREATE OR REPLACE TABLE {target} AS
        SELECT
          CAST(NULL AS STRING) AS run_id,
          CAST(NULL AS STRING) AS feature_family,
          CAST(NULL AS STRING) AS column_name,
          CAST(NULL AS STRING) AS data_type,
          CAST(NULL AS STRING) AS bin_type,
          CAST(NULL AS STRING) AS bin_label,
          CAST(NULL AS FLOAT64) AS bin_min_value,
          CAST(NULL AS FLOAT64) AS bin_max_value,
          CAST(NULL AS INT64) AS row_count,
          CAST(NULL AS INT64) AS goal_count,
          CAST(NULL AS FLOAT64) AS goal_rate,
          CAST(NULL AS FLOAT64) AS baseline_goal_rate,
          CAST(NULL AS FLOAT64) AS lift_vs_baseline,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM UNNEST([]) AS empty_row
        """)
        selects: list[str] = []
        for feature in features:
            surface = q(PROJECT_ID, ANALYSIS_DATASET, self.surface_for(feature.family))
            col = f"`{feature.column}`"
            run = quote_literal(self.run_id)
            fam = quote_literal(feature.family)
            name = quote_literal(feature.column)
            dtype = quote_literal(feature.data_type)
            if feature.is_numeric and feature.non_null_count >= 50:
                value = f"IF({col} IS NULL, NULL, IF({col}, 1.0, 0.0))" if feature.data_type.upper() == "BOOL" else f"CAST({col} AS FLOAT64)"
                selects.append(f"""
                SELECT
                  {run} AS run_id,
                  {fam} AS feature_family,
                  {name} AS column_name,
                  {dtype} AS data_type,
                  'quantile' AS bin_type,
                  CAST(bin_number AS STRING) AS bin_label,
                  MIN(value) AS bin_min_value,
                  MAX(value) AS bin_max_value,
                  COUNT(*) AS row_count,
                  COUNTIF(is_goal) AS goal_count,
                  AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
                  AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER() AS baseline_goal_rate,
                  SAFE_DIVIDE(AVG(IF(is_goal, 1.0, 0.0)), AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER()) AS lift_vs_baseline,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM (
                  SELECT {value} AS value, is_goal, NTILE(10) OVER (ORDER BY {value}) AS bin_number
                  FROM {surface}
                  WHERE {col} IS NOT NULL
                )
                GROUP BY bin_number
                """)
            elif feature.is_categorical and feature.non_null_count >= 50:
                selects.append(f"""
                SELECT
                  {run} AS run_id,
                  {fam} AS feature_family,
                  {name} AS column_name,
                  {dtype} AS data_type,
                  'category' AS bin_type,
                  CAST({col} AS STRING) AS bin_label,
                  CAST(NULL AS FLOAT64) AS bin_min_value,
                  CAST(NULL AS FLOAT64) AS bin_max_value,
                  COUNT(*) AS row_count,
                  COUNTIF(is_goal) AS goal_count,
                  AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
                  AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER() AS baseline_goal_rate,
                  SAFE_DIVIDE(AVG(IF(is_goal, 1.0, 0.0)), AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER()) AS lift_vs_baseline,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {surface}
                WHERE {col} IS NOT NULL
                GROUP BY bin_label
                QUALIFY ROW_NUMBER() OVER (PARTITION BY {name} ORDER BY row_count DESC) <= 30
                """)
        for offset in range(0, len(selects), 80):
            chunk = selects[offset : offset + 80]
            if chunk:
                self.query(f"INSERT INTO {target}\n" + "\nUNION ALL\n".join(chunk))

    def eligible_numeric(self, features: list[Feature], family: str) -> list[Feature]:
        selected = [f for f in features if f.family == family and f.is_numeric]
        selected = sorted(selected, key=lambda f: (f.abs_signal, f.non_null_count), reverse=True)
        return selected[:MAX_NUMERIC_FEATURES_PER_FAMILY]

    def materialize_cross_family_correlations(self, features: list[Feature]) -> None:
        target = q(PROJECT_ID, ANALYSIS_DATASET, "cxg_cross_family_correlation_v1")
        self.query(f"""
        CREATE OR REPLACE TABLE {target} AS
        SELECT
          CAST(NULL AS STRING) AS run_id,
          CAST(NULL AS STRING) AS left_family,
          CAST(NULL AS STRING) AS left_column,
          CAST(NULL AS STRING) AS right_family,
          CAST(NULL AS STRING) AS right_column,
          CAST(NULL AS INT64) AS pair_non_null_count,
          CAST(NULL AS FLOAT64) AS pearson_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM UNNEST([]) AS empty_row
        """)
        families = [CONTROL_FAMILY, "event_context", "defensive_360", "goalkeeper_360", "line_shape_360"]
        selects: list[str] = []
        for i, left_family in enumerate(families):
            for right_family in families[i + 1 :]:
                if left_family.endswith("_360") != right_family.endswith("_360") and left_family not in {CONTROL_FAMILY, "event_context"}:
                    continue
                surface = q(PROJECT_ID, ANALYSIS_DATASET, "cxg_analysis_360_v1" if right_family.endswith("_360") else "cxg_analysis_event_v1")
                min_support = MIN_PAIR_SUPPORT_360 if right_family.endswith("_360") else MIN_PAIR_SUPPORT_EVENT
                left_features = self.eligible_numeric(features, left_family)[:18]
                right_features = self.eligible_numeric(features, right_family)[:18]
                for left in left_features:
                    for right in right_features:
                        if left.column == right.column:
                            continue
                        selects.append(f"""
                        SELECT
                          {quote_literal(self.run_id)} AS run_id,
                          {quote_literal(left.family)} AS left_family,
                          {quote_literal(left.column)} AS left_column,
                          {quote_literal(right.family)} AS right_family,
                          {quote_literal(right.column)} AS right_column,
                          COUNTIF(`{left.column}` IS NOT NULL AND `{right.column}` IS NOT NULL) AS pair_non_null_count,
                          CORR(CAST(`{left.column}` AS FLOAT64), CAST(`{right.column}` AS FLOAT64)) AS pearson_corr,
                          CURRENT_TIMESTAMP() AS materialized_at
                        FROM {surface}
                        HAVING pair_non_null_count >= {min_support}
                        """)
        for offset in range(0, len(selects), 80):
            chunk = selects[offset : offset + 80]
            if chunk:
                self.query(f"INSERT INTO {target}\n" + "\nUNION ALL\n".join(chunk))

    def materialize_pair_interactions(self, features: list[Feature]) -> None:
        target = q(PROJECT_ID, ANALYSIS_DATASET, "cxg_pair_interaction_target_v1")
        self.query(f"""
        CREATE OR REPLACE TABLE {target} AS
        SELECT
          CAST(NULL AS STRING) AS run_id,
          CAST(NULL AS STRING) AS feature_family,
          CAST(NULL AS STRING) AS left_column,
          CAST(NULL AS STRING) AS right_column,
          CAST(NULL AS STRING) AS left_bin,
          CAST(NULL AS STRING) AS right_bin,
          CAST(NULL AS INT64) AS row_count,
          CAST(NULL AS INT64) AS goal_count,
          CAST(NULL AS FLOAT64) AS goal_rate,
          CAST(NULL AS FLOAT64) AS baseline_goal_rate,
          CAST(NULL AS FLOAT64) AS lift_vs_baseline,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM UNNEST([]) AS empty_row
        """)
        selects: list[str] = []
        for family in ["event_context", "defensive_360", "goalkeeper_360", "line_shape_360"]:
            selected = self.eligible_numeric(features, family)[:MAX_INTERACTION_FEATURES_PER_FAMILY]
            surface = q(PROJECT_ID, ANALYSIS_DATASET, self.surface_for(family))
            min_support = 80 if family.endswith("_360") else 180
            for i, left in enumerate(selected):
                for right in selected[i + 1 :]:
                    selects.append(f"""
                    SELECT * FROM (
                      SELECT
                        {quote_literal(self.run_id)} AS run_id,
                        {quote_literal(family)} AS feature_family,
                        {quote_literal(left.column)} AS left_column,
                        {quote_literal(right.column)} AS right_column,
                        CAST(left_bin AS STRING) AS left_bin,
                        CAST(right_bin AS STRING) AS right_bin,
                        COUNT(*) AS row_count,
                        COUNTIF(is_goal) AS goal_count,
                        AVG(IF(is_goal, 1.0, 0.0)) AS goal_rate,
                        AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER() AS baseline_goal_rate,
                        SAFE_DIVIDE(AVG(IF(is_goal, 1.0, 0.0)), AVG(AVG(IF(is_goal, 1.0, 0.0))) OVER()) AS lift_vs_baseline,
                        CURRENT_TIMESTAMP() AS materialized_at
                      FROM (
                        SELECT
                          is_goal,
                          NTILE(4) OVER (ORDER BY CAST(`{left.column}` AS FLOAT64)) AS left_bin,
                          NTILE(4) OVER (ORDER BY CAST(`{right.column}` AS FLOAT64)) AS right_bin
                        FROM {surface}
                        WHERE `{left.column}` IS NOT NULL AND `{right.column}` IS NOT NULL
                      )
                      GROUP BY left_bin, right_bin
                    )
                    WHERE row_count >= {min_support}
                    """)
        for offset in range(0, len(selects), 50):
            chunk = selects[offset : offset + 50]
            if chunk:
                self.query(f"INSERT INTO {target}\n" + "\nUNION ALL\n".join(chunk))

    def materialize_redundancy_clusters(self) -> None:
        self.query(f"""
        CREATE OR REPLACE TABLE {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_redundancy_pairs_v1')} AS
        SELECT
          {quote_literal(self.run_id)} AS run_id,
          left_family,
          left_column,
          right_family,
          right_column,
          pair_non_null_count,
          pearson_corr,
          ABS(pearson_corr) AS abs_corr,
          CASE
            WHEN ABS(pearson_corr) >= 0.95 THEN 'duplicate_or_near_duplicate'
            WHEN ABS(pearson_corr) >= 0.85 THEN 'strong_redundancy'
            WHEN ABS(pearson_corr) >= 0.75 THEN 'moderate_redundancy'
            ELSE 'not_redundant'
          END AS redundancy_band,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM (
          SELECT left_family, left_column, right_family, right_column, pair_non_null_count, pearson_corr
          FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_correlation_v1')}
          UNION ALL
          SELECT left_family, left_column, right_family, right_column, pair_non_null_count, pearson_corr
          FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_cross_family_correlation_v1')}
        )
        WHERE pearson_corr IS NOT NULL AND ABS(pearson_corr) >= 0.75
        """)

    def materialize_decision_manifest(self) -> None:
        self.query(f"""
        CREATE OR REPLACE TABLE {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_feature_decision_manifest_v1')} AS
        WITH base AS (
          SELECT
            i.feature_family,
            i.column_name,
            i.data_type,
            n.non_null_count,
            n.null_pct,
            u.point_biserial_corr,
            ABS(COALESCE(u.point_biserial_corr, 0.0)) AS abs_signal,
            COALESCE(r.max_abs_corr, 0.0) AS max_abs_corr
          FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_feature_inventory_v1')} i
          LEFT JOIN {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_null_profile_v1')} n USING(feature_family, column_name)
          LEFT JOIN {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_univariate_target_v1')} u USING(feature_family, column_name)
          LEFT JOIN (
            SELECT column_name, MAX(abs_corr) AS max_abs_corr
            FROM (
              SELECT left_column AS column_name, ABS(pearson_corr) AS abs_corr FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_correlation_v1')}
              UNION ALL
              SELECT right_column AS column_name, ABS(pearson_corr) AS abs_corr FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_correlation_v1')}
              UNION ALL
              SELECT left_column AS column_name, ABS(pearson_corr) AS abs_corr FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_cross_family_correlation_v1')}
              UNION ALL
              SELECT right_column AS column_name, ABS(pearson_corr) AS abs_corr FROM {q(PROJECT_ID, ANALYSIS_DATASET, 'cxg_cross_family_correlation_v1')}
            )
            GROUP BY column_name
          ) r USING(column_name)
          WHERE i.column_role = 'feature'
            AND i.feature_family IN ({', '.join(quote_literal(f) for f in sorted(CONTEXT_FAMILIES | {CONTROL_FAMILY}))})
        )
        SELECT
          {quote_literal(self.run_id)} AS run_id,
          *,
          CASE
            WHEN feature_family = {quote_literal(CONTROL_FAMILY)} THEN 'control'
            WHEN non_null_count < 400 THEN 'watchlist_sparse'
            WHEN null_pct > 0.80 THEN 'watchlist_sparse'
            WHEN abs_signal >= 0.10 AND max_abs_corr < 0.95 THEN 'candidate_keep'
            WHEN abs_signal >= 0.05 THEN 'candidate_review'
            WHEN max_abs_corr >= 0.95 THEN 'redundant_review'
            ELSE 'weak_signal_review'
          END AS preliminary_decision,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM base
        """)

    def validate_manifest(self) -> dict[str, object]:
        tables = [
            "cxg_binned_target_response_v1",
            "cxg_cross_family_correlation_v1",
            "cxg_pair_interaction_target_v1",
            "cxg_redundancy_pairs_v1",
            "cxg_feature_decision_manifest_v1",
        ]
        counts = {}
        if not self.dry_run:
            for table in tables:
                rows = list(
                    self.bq.query(
                        f"SELECT COUNT(*) AS n FROM {q(PROJECT_ID, ANALYSIS_DATASET, table)}",
                        location=LOCATION,
                    ).result()
                )
                counts[table] = rows[0].n
        return {
            "run_id": self.run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "dry_run": self.dry_run,
            "tables": counts,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=ACTIVE_RUN_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    materializer = FindingsMaterializer(run_id=args.run_id, dry_run=args.dry_run)
    print(json.dumps(materializer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
