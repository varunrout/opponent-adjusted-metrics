"""CxG analysis table and chart-spec materialization.

The analysis layer is intentionally data-first: BigQuery stores canonical
analysis outputs, while Cloud Storage stores dashboard-ready chart specs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from google.cloud import bigquery, storage


PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
FEATURE_DATASET = "oam_features"
ANALYSIS_DATASET = "oam_analysis"
DEFAULT_FEATURE_VERSION = "cxg_gold_v1_e13_f1_f15"


FAMILY_TABLES: dict[str, str] = {
    "base_identity_target": "cxg_shot_base_features",
    "shot_geometry": "cxg_shot_geometry_features",
    "event_context": "cxg_event_context_features",
    "buildup": "cxg_buildup_features",
    "defensive_360": "cxg_defensive_360_features",
    "goalkeeper_360": "cxg_goalkeeper_360_features",
    "line_shape_360": "cxg_line_shape_360_features",
}

ID_COLUMNS = {
    "event_id",
    "match_id",
    "player_id",
    "team_id",
    "data_version",
    "feature_version",
    "silver_schema_version",
    "materialized_at",
}
TARGET_COLUMNS = {"is_goal", "outcome_name"}


@dataclass(frozen=True)
class FieldInfo:
    family: str
    table_name: str
    column_name: str
    data_type: str

    @property
    def is_numeric(self) -> bool:
        return self.data_type.upper() in {"INT64", "FLOAT64", "NUMERIC", "BIGNUMERIC"}

    @property
    def is_boolean(self) -> bool:
        return self.data_type.upper() == "BOOL"

    @property
    def is_categorical(self) -> bool:
        return self.data_type.upper() in {"STRING", "BOOL"}

    @property
    def role(self) -> str:
        if self.column_name in TARGET_COLUMNS:
            return "target"
        if self.column_name in ID_COLUMNS:
            return "lineage"
        return "feature"


def _quote(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class CxGAnalysisMaterializer:
    def __init__(
        self,
        project_id: str = PROJECT_ID,
        location: str = LOCATION,
        feature_dataset: str = FEATURE_DATASET,
        analysis_dataset: str = ANALYSIS_DATASET,
        artifact_bucket: str | None = None,
        artifact_prefix: str = "analysis/cxg",
        feature_version: str = DEFAULT_FEATURE_VERSION,
        local_output_dir: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        self.project_id = project_id
        self.location = location
        self.feature_dataset = feature_dataset
        self.analysis_dataset = analysis_dataset
        self.artifact_bucket = artifact_bucket
        self.artifact_prefix = artifact_prefix.strip("/")
        self.feature_version = feature_version
        self.local_output_dir = local_output_dir
        self.dry_run = dry_run
        self.bq = bigquery.Client(project=project_id)
        self.gcs = storage.Client(project=project_id) if artifact_bucket else None
        self.run_id = "cxg-analysis-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    def run(self) -> dict[str, object]:
        self._ensure_analysis_dataset()
        fields = self._load_fields()
        self._materialize_analysis_surfaces(fields)
        self._materialize_inventory(fields)
        self._materialize_null_profile(fields)
        self._materialize_summary_stats(fields)
        self._materialize_distribution_bins(fields)
        self._materialize_univariate_target(fields)
        self._materialize_correlation(fields)
        self._materialize_family_summary(fields)
        chart_specs = self._build_chart_specs(fields)
        self._persist_chart_specs(chart_specs)
        self._materialize_chart_registry(chart_specs)
        manifest = self._write_run_manifest(fields, chart_specs)
        return manifest

    def _query(self, sql: str) -> None:
        if self.dry_run:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            self.bq.query(sql, location=self.location, job_config=job_config)
            return
        self.bq.query(sql, location=self.location).result()

    def _ensure_analysis_dataset(self) -> None:
        dataset_id = f"{self.project_id}.{self.analysis_dataset}"
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = self.location
        if not self.dry_run:
            self.bq.create_dataset(dataset, exists_ok=True)

    def _load_fields(self) -> list[FieldInfo]:
        rows: list[FieldInfo] = []
        sql = f"""
        SELECT table_name, column_name, data_type
        FROM `{self.project_id}.{self.feature_dataset}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name IN ({", ".join(repr(v) for v in FAMILY_TABLES.values())})
        ORDER BY table_name, ordinal_position
        """
        for row in self.bq.query(sql, location=self.location).result():
            family = next(k for k, v in FAMILY_TABLES.items() if v == row.table_name)
            rows.append(FieldInfo(family, row.table_name, row.column_name, row.data_type))
        return rows

    def _materialize_analysis_surfaces(self, fields: Iterable[FieldInfo]) -> None:
        training = _quote(self.project_id, self.feature_dataset, "cxg_training_matrix_v1")
        event_surface = _quote(self.project_id, self.analysis_dataset, "cxg_analysis_event_v1")
        plus_surface = _quote(self.project_id, self.analysis_dataset, "cxg_analysis_360_v1")
        field_list = list(fields)
        base_cols = [
            "event_id",
            "match_id",
            "player_id",
            "team_id",
            "is_goal",
            "outcome_name",
            "shot_x_sb",
            "shot_y_sb",
            "has_360_frame",
            "data_version",
            "feature_version",
            "silver_schema_version",
        ]
        event_families = {"base_identity_target", "shot_geometry", "event_context", "buildup"}
        plus_families = event_families | {"defensive_360", "goalkeeper_360", "line_shape_360"}

        def columns_for(families: set[str]) -> str:
            seen = set()
            ordered = []
            for column in base_cols:
                seen.add(column)
                ordered.append(f"`{column}`")
            for field in field_list:
                if field.family not in families or field.role != "feature":
                    continue
                if field.column_name in seen:
                    continue
                seen.add(field.column_name)
                ordered.append(f"`{field.column_name}`")
            ordered.append("CURRENT_TIMESTAMP() AS analysis_materialized_at")
            return ",\n          ".join(ordered)

        self._query(f"""
        CREATE OR REPLACE TABLE {event_surface} AS
        SELECT
          {columns_for(event_families)}
        FROM {training}
        WHERE feature_version = '{self.feature_version}'
        """)

        self._query(f"""
        CREATE OR REPLACE TABLE {plus_surface} AS
        SELECT
          {columns_for(plus_families)}
        FROM {training}
        WHERE feature_version = '{self.feature_version}'
          AND has_360_frame
        """)

    def _materialize_inventory(self, fields: Iterable[FieldInfo]) -> None:
        rows = []
        for field in fields:
            rows.append(
                "SELECT "
                f"'{field.family}' AS feature_family, "
                f"'{field.table_name}' AS source_table, "
                f"'{field.column_name}' AS column_name, "
                f"'{field.data_type}' AS data_type, "
                f"'{field.role}' AS column_role, "
                f"{str(field.is_numeric).upper()} AS is_numeric, "
                f"{str(field.is_categorical).upper()} AS is_categorical, "
                f"TIMESTAMP('{_now_iso()}') AS materialized_at"
            )
        sql = " UNION ALL\n".join(rows)
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_feature_inventory_v1')} AS\n{sql}"
        )

    def _materialize_null_profile(self, fields: Iterable[FieldInfo]) -> None:
        selects = []
        for field in fields:
            table = _quote(self.project_id, self.feature_dataset, field.table_name)
            col = f"`{field.column_name}`"
            selects.append(f"""
            SELECT
              '{field.family}' AS feature_family,
              '{field.table_name}' AS table_name,
              '{field.column_name}' AS column_name,
              COUNT(*) AS row_count,
              COUNTIF({col} IS NULL) AS null_count,
              COUNTIF({col} IS NOT NULL) AS non_null_count,
              SAFE_DIVIDE(COUNTIF({col} IS NULL), COUNT(*)) AS null_pct,
              CURRENT_TIMESTAMP() AS materialized_at
            FROM {table}
            """)
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_null_profile_v1')} AS\n"
            + "\nUNION ALL\n".join(selects)
        )

    def _materialize_summary_stats(self, fields: Iterable[FieldInfo]) -> None:
        selects = []
        for field in fields:
            if field.role != "feature":
                continue
            table = _quote(self.project_id, self.feature_dataset, field.table_name)
            col = f"`{field.column_name}`"
            if field.is_numeric or field.is_boolean:
                value = f"IF({col} IS NULL, NULL, IF({col}, 1.0, 0.0))" if field.is_boolean else f"CAST({col} AS FLOAT64)"
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.table_name}' AS table_name,
                  '{field.column_name}' AS column_name,
                  '{field.data_type}' AS data_type,
                  COUNT(*) AS row_count,
                  COUNTIF({col} IS NOT NULL) AS non_null_count,
                  AVG({value}) AS mean_value,
                  STDDEV({value}) AS stddev_value,
                  MIN({value}) AS min_value,
                  APPROX_QUANTILES({value}, 100)[OFFSET(25)] AS p25_value,
                  APPROX_QUANTILES({value}, 100)[OFFSET(50)] AS median_value,
                  APPROX_QUANTILES({value}, 100)[OFFSET(75)] AS p75_value,
                  MAX({value}) AS max_value,
                  COUNT(DISTINCT {col}) AS distinct_count,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {table}
                WHERE {col} IS NOT NULL
                """)
            elif field.data_type.upper() == "STRING":
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.table_name}' AS table_name,
                  '{field.column_name}' AS column_name,
                  '{field.data_type}' AS data_type,
                  COUNT(*) AS row_count,
                  COUNTIF({col} IS NOT NULL) AS non_null_count,
                  NULL AS mean_value,
                  NULL AS stddev_value,
                  NULL AS min_value,
                  NULL AS p25_value,
                  NULL AS median_value,
                  NULL AS p75_value,
                  NULL AS max_value,
                  COUNT(DISTINCT {col}) AS distinct_count,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {table}
                """)
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_summary_stats_v1')} AS\n"
            + "\nUNION ALL\n".join(selects)
        )

    def _materialize_distribution_bins(self, fields: Iterable[FieldInfo]) -> None:
        selects = []
        for field in fields:
            if field.role != "feature":
                continue
            table = _quote(self.project_id, self.feature_dataset, field.table_name)
            col = f"`{field.column_name}`"
            if field.is_numeric:
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.column_name}' AS column_name,
                  'quantile' AS bin_type,
                  CAST(bin_number AS STRING) AS bin_label,
                  COUNT(*) AS row_count,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM (
                  SELECT NTILE(20) OVER (ORDER BY {col}) AS bin_number
                  FROM {table}
                  WHERE {col} IS NOT NULL
                )
                GROUP BY bin_label
                """)
            elif field.is_categorical:
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.column_name}' AS column_name,
                  'category' AS bin_type,
                  CAST({col} AS STRING) AS bin_label,
                  COUNT(*) AS row_count,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {table}
                WHERE {col} IS NOT NULL
                GROUP BY bin_label
                """)
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_eda_distribution_bins_v1')} AS\n"
            + "\nUNION ALL\n".join(selects)
        )


    def _surface_for_family(self, family: str) -> str:
        if family.endswith("_360"):
            return "cxg_analysis_360_v1"
        return "cxg_analysis_event_v1"

    def _materialize_univariate_target(self, fields: Iterable[FieldInfo]) -> None:
        selects = []
        for field in fields:
            if field.role != "feature":
                continue
            surface = _quote(self.project_id, self.analysis_dataset, self._surface_for_family(field.family))
            col = f"`{field.column_name}`"
            if field.is_numeric or field.is_boolean:
                value = f"IF({col} IS NULL, NULL, IF({col}, 1.0, 0.0))" if field.is_boolean else f"CAST({col} AS FLOAT64)"
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.column_name}' AS column_name,
                  '{field.data_type}' AS data_type,
                  COUNT(*) AS row_count,
                  COUNTIF({col} IS NOT NULL) AS non_null_count,
                  AVG(IF(is_goal IS NULL, NULL, IF(is_goal, 1.0, 0.0))) AS goal_rate,
                  AVG({value}) AS mean_when_available,
                  AVG(IF(is_goal, {value}, NULL)) AS mean_for_goals,
                  AVG(IF(NOT is_goal, {value}, NULL)) AS mean_for_non_goals,
                  CORR({value}, IF(is_goal IS NULL, NULL, IF(is_goal, 1.0, 0.0))) AS point_biserial_corr,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {surface}
                """)
            elif field.data_type.upper() == "STRING":
                selects.append(f"""
                SELECT
                  '{field.family}' AS feature_family,
                  '{field.column_name}' AS column_name,
                  '{field.data_type}' AS data_type,
                  COUNT(*) AS row_count,
                  COUNTIF({col} IS NOT NULL) AS non_null_count,
                  AVG(IF(is_goal IS NULL, NULL, IF(is_goal, 1.0, 0.0))) AS goal_rate,
                  NULL AS mean_when_available,
                  NULL AS mean_for_goals,
                  NULL AS mean_for_non_goals,
                  NULL AS point_biserial_corr,
                  CURRENT_TIMESTAMP() AS materialized_at
                FROM {surface}
                """)
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_univariate_target_v1')} AS\n"
            + "\nUNION ALL\n".join(selects)
        )

    def _materialize_correlation(self, fields: Iterable[FieldInfo]) -> None:
        target = _quote(self.project_id, self.analysis_dataset, "cxg_correlation_v1")
        self._query(f"""
        CREATE OR REPLACE TABLE {target} AS
        SELECT
          CAST(NULL AS STRING) AS left_family,
          CAST(NULL AS STRING) AS left_column,
          CAST(NULL AS STRING) AS right_family,
          CAST(NULL AS STRING) AS right_column,
          CAST(NULL AS INT64) AS pair_non_null_count,
          CAST(NULL AS FLOAT64) AS pearson_corr,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM UNNEST([]) AS empty_row
        """)

        numeric = [f for f in fields if f.role == "feature" and f.is_numeric]
        max_features_per_family = 28
        for family in sorted({field.family for field in numeric}):
            surface_name = self._surface_for_family(family)
            surface = _quote(self.project_id, self.analysis_dataset, surface_name)
            candidates = [field for field in numeric if field.family == family]
            if len(candidates) > max_features_per_family:
                columns = ", ".join(
                    f"COUNTIF(`{field.column_name}` IS NOT NULL) AS `{field.column_name}`"
                    for field in candidates
                )
                rows = list(self.bq.query(f"SELECT {columns} FROM {surface}", location=self.location).result())
                counts = dict(rows[0].items()) if rows else {}
                candidates = sorted(candidates, key=lambda f: counts.get(f.column_name, 0), reverse=True)[:max_features_per_family]
            selects = []
            for left in candidates:
                for right in candidates:
                    if left.column_name >= right.column_name:
                        continue
                    selects.append(f"""
                    SELECT
                      '{left.family}' AS left_family,
                      '{left.column_name}' AS left_column,
                      '{right.family}' AS right_family,
                      '{right.column_name}' AS right_column,
                      COUNTIF(`{left.column_name}` IS NOT NULL AND `{right.column_name}` IS NOT NULL) AS pair_non_null_count,
                      CORR(CAST(`{left.column_name}` AS FLOAT64), CAST(`{right.column_name}` AS FLOAT64)) AS pearson_corr,
                      CURRENT_TIMESTAMP() AS materialized_at
                    FROM {surface}
                    """)
            for offset in range(0, len(selects), 120):
                chunk = selects[offset : offset + 120]
                if chunk:
                    self._query(f"INSERT INTO {target}\n" + "\nUNION ALL\n".join(chunk))


    def _materialize_family_summary(self, fields: Iterable[FieldInfo]) -> None:
        self._query(f"""
        CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_family_summary_v1')} AS
        SELECT
          i.feature_family,
          COUNTIF(i.column_role = 'feature') AS feature_count,
          COUNTIF(i.is_numeric AND i.column_role = 'feature') AS numeric_feature_count,
          COUNTIF(i.is_categorical AND i.column_role = 'feature') AS categorical_feature_count,
          AVG(IF(i.column_role = 'feature', n.null_pct, NULL)) AS avg_null_pct,
          MAX(IF(i.column_role = 'feature', n.null_pct, NULL)) AS max_null_pct,
          COUNTIF(i.column_role = 'feature' AND n.null_pct = 1) AS all_null_feature_count,
          CURRENT_TIMESTAMP() AS materialized_at
        FROM {_quote(self.project_id, self.analysis_dataset, 'cxg_feature_inventory_v1')} i
        LEFT JOIN {_quote(self.project_id, self.analysis_dataset, 'cxg_null_profile_v1')} n
          ON i.source_table = n.table_name AND i.column_name = n.column_name
        GROUP BY i.feature_family
        """)

    def _build_chart_specs(self, fields: Iterable[FieldInfo]) -> list[dict[str, object]]:
        specs: list[dict[str, object]] = []
        feature_families = sorted({field.family for field in fields if field.role == "feature"})
        non_empty_families = [
            family
            for family in feature_families
            if any(field.family == family and field.role == "feature" for field in fields)
        ]
        for family in non_empty_families:
            specs.extend(
                [
                    self._chart_spec(family, "null_profile_bar", "bar", "cxg_null_profile_v1"),
                    self._chart_spec(family, "summary_box", "box", "cxg_summary_stats_v1"),
                    self._chart_spec(family, "eda_histogram", "histogram", "cxg_eda_distribution_bins_v1"),
                    self._chart_spec(family, "target_lift_bar", "bar", "cxg_univariate_target_v1"),
                ]
            )
        specs.extend(
            [
                self._chart_spec("base_identity_target", "shot_pitch_density", "pitch_heatmap", "cxg_analysis_event_v1"),
                self._chart_spec("defensive_360", "defender_pressure_pitch", "pitch_scatter", "cxg_analysis_360_v1"),
                self._chart_spec("goalkeeper_360", "gk_position_pitch", "pitch_scatter", "cxg_analysis_360_v1"),
                self._chart_spec("line_shape_360", "line_shape_corr_heatmap", "heatmap", "cxg_correlation_v1"),
            ]
        )
        return specs

    def _chart_spec(self, family: str, chart_name: str, chart_type: str, backing_table: str) -> dict[str, object]:
        full_chart_name = f"{_safe_name(family)}_{_safe_name(chart_name)}"
        uri = None
        if self.artifact_bucket:
            uri = f"gs://{self.artifact_bucket}/{self.artifact_prefix}/{self.run_id}/charts/{full_chart_name}.json"
        return {
            "run_id": self.run_id,
            "feature_family": family,
            "chart_name": full_chart_name,
            "chart_type": chart_type,
            "chart_library": "plotly",
            "backing_table": f"{self.project_id}.{self.analysis_dataset}.{backing_table}",
            "artifact_uri": uri,
            "created_at": _now_iso(),
            "spec": {
                "data_source": f"{self.project_id}.{self.analysis_dataset}.{backing_table}",
                "family_filter": family,
                "chart_type": chart_type,
                "dashboard_ready": True,
            },
        }


    def _persist_chart_specs(self, specs: Iterable[dict[str, object]]) -> None:
        if self.local_output_dir:
            chart_dir = self.local_output_dir / self.run_id / "charts"
            chart_dir.mkdir(parents=True, exist_ok=True)
            for spec in specs:
                (chart_dir / f"{spec['chart_name']}.json").write_text(
                    json.dumps(spec, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        if not self.artifact_bucket or self.dry_run:
            return
        assert self.gcs is not None
        bucket = self.gcs.bucket(self.artifact_bucket)
        for spec in specs:
            blob_name = f"{self.artifact_prefix}/{self.run_id}/charts/{spec['chart_name']}.json"
            bucket.blob(blob_name).upload_from_string(
                json.dumps(spec, indent=2, sort_keys=True),
                content_type="application/json",
            )

    def _materialize_chart_registry(self, specs: Iterable[dict[str, object]]) -> None:
        rows = []
        for spec in specs:
            rows.append(
                "SELECT "
                f"'{spec['run_id']}' AS run_id, "
                f"'{spec['feature_family']}' AS feature_family, "
                f"'{spec['chart_name']}' AS chart_name, "
                f"'{spec['chart_type']}' AS chart_type, "
                f"'{spec['chart_library']}' AS chart_library, "
                f"'{spec['backing_table']}' AS backing_table, "
                f"{json.dumps(spec['artifact_uri']) if spec['artifact_uri'] else 'NULL'} AS artifact_uri, "
                "CURRENT_TIMESTAMP() AS materialized_at"
            )
        self._query(
            f"CREATE OR REPLACE TABLE {_quote(self.project_id, self.analysis_dataset, 'cxg_chart_registry_v1')} AS\n"
            + "\nUNION ALL\n".join(rows)
        )

    def _write_run_manifest(
        self, fields: Iterable[FieldInfo], specs: Iterable[dict[str, object]]
    ) -> dict[str, object]:
        manifest = {
            "run_id": self.run_id,
            "generated_at": _now_iso(),
            "project_id": self.project_id,
            "location": self.location,
            "feature_dataset": self.feature_dataset,
            "analysis_dataset": self.analysis_dataset,
            "feature_version": self.feature_version,
            "feature_columns_seen": sum(1 for field in fields if field.role == "feature"),
            "chart_specs": sum(1 for _ in specs),
            "artifact_bucket": self.artifact_bucket,
            "artifact_prefix": self.artifact_prefix,
            "dry_run": self.dry_run,
        }
        if self.local_output_dir:
            run_dir = self.local_output_dir / self.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if self.artifact_bucket and not self.dry_run:
            assert self.gcs is not None
            blob_name = f"{self.artifact_prefix}/{self.run_id}/manifest.json"
            self.gcs.bucket(self.artifact_bucket).blob(blob_name).upload_from_string(
                json.dumps(manifest, indent=2),
                content_type="application/json",
            )
        return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize CxG analysis tables and chart specs.")
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--feature-dataset", default=FEATURE_DATASET)
    parser.add_argument("--analysis-dataset", default=ANALYSIS_DATASET)
    parser.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    parser.add_argument("--artifact-bucket", default=None)
    parser.add_argument("--artifact-prefix", default="analysis/cxg")
    parser.add_argument("--local-output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    materializer = CxGAnalysisMaterializer(
        project_id=args.project_id,
        location=args.location,
        feature_dataset=args.feature_dataset,
        analysis_dataset=args.analysis_dataset,
        artifact_bucket=args.artifact_bucket,
        artifact_prefix=args.artifact_prefix,
        feature_version=args.feature_version,
        local_output_dir=Path(args.local_output_dir) if args.local_output_dir else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(materializer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

