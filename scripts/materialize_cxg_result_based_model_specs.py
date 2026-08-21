"""Materialize result-based above-baseline CxG/CxG+ model specs only.

This script uses existing split-aware univariate, bivariate, and multivariate
analysis outputs to define candidate model specs above the true XY baselines.
It does not train, evaluate, freeze, calibrate, promote, or package models.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
ANALYSIS_DATASET = "oam_analysis"
RUN_ID = "cxg-analysis-20260820T201934Z"
SPEC_VERSION = "cxg_above_baseline_model_spec_v1"


def table(name: str) -> str:
    return f"{PROJECT_ID}.{ANALYSIS_DATASET}.{name}"


def bq_table(name: str) -> str:
    return f"`{PROJECT_ID}.{ANALYSIS_DATASET}.{name}`"


class ResultBasedModelSpecMaterializer:
    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.output_dir = output_dir / run_id / "result_based_model_specs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.materialized_at = datetime.now(UTC).isoformat()

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def run(self) -> dict[str, object]:
        specs = [
            self.build_spec(
                track="cxg_event",
                spec_name="cxg_event_above_baseline_event_context_v1",
                baseline_name="cxg_event_true_baseline_xy",
                population_table=table("cxg_event_model_matrix_v1"),
                population_filter="all CxG shots",
                eligible_stages=["candidate_model"],
                feature_scope="baseline XY plus validation-supported event_context shortlist",
            ),
            self.build_spec(
                track="cxg_plus",
                spec_name="cxg_plus_above_baseline_360_context_v1",
                baseline_name="cxg_plus_true_baseline_xy",
                population_table=table("cxg_plus_360_model_matrix_v1"),
                population_filter="has_360_frame CxG+ shots only",
                eligible_stages=["candidate_model", "multivariate_experiment"],
                feature_scope="baseline XY plus validation-supported 360 contextual families",
            ),
        ]
        self.load_rows("cxg_above_baseline_model_spec_v1", specs)
        manifest_path = self.write_manifest(specs)
        report_path = self.write_report(specs)
        validation = self.validate()
        return {
            "run_id": self.run_id,
            "spec_version": SPEC_VERSION,
            "model_spec_table": table("cxg_above_baseline_model_spec_v1"),
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "validation": validation,
        }

    def build_spec(
        self,
        track: str,
        spec_name: str,
        baseline_name: str,
        population_table: str,
        population_filter: str,
        eligible_stages: list[str],
        feature_scope: str,
    ) -> dict[str, object]:
        reference = self.choose_multivariate_reference(track, eligible_stages)
        features = [f.strip() for f in str(reference["features"]).split(",") if f.strip()]
        uni = self.univariate_evidence(track, features)
        bi = self.bivariate_evidence(track, features)
        return {
            "run_id": self.run_id,
            "spec_version": SPEC_VERSION,
            "track": track,
            "model_spec_name": spec_name,
            "model_spec_status": "above_baseline_candidate_spec_ready",
            "baseline_name": baseline_name,
            "population_table": population_table,
            "population_filter": population_filter,
            "target_column": "is_goal",
            "feature_names": ",".join(features),
            "feature_count": len(features),
            "feature_scope": feature_scope,
            "source_univariate_table": table("cxg_split_univariate_v1"),
            "source_bivariate_table": table("cxg_split_bivariate_v1"),
            "source_multivariate_table": table("cxg_split_multivariate_results_v1"),
            "multivariate_reference_model": str(reference["model_name"]),
            "multivariate_reference_stage": str(reference["model_stage"]),
            "selection_split": "validation",
            "training_split": "train",
            "test_split_policy": "not used for this spec definition",
            "top_univariate_validation_features": ",".join(uni["top_features"]),
            "top_bivariate_validation_pairs": "; ".join(bi["top_pairs"]),
            "univariate_overlap_count": int(uni["overlap_count"]),
            "bivariate_overlap_pair_count": int(bi["overlap_pair_count"]),
            "selection_basis": (
                "Spec is selected from split-aware validation-screened multivariate candidates, "
                "then documented against overlapping validation univariate and bivariate evidence. "
                "This is a candidate model spec only, not a production model."
            ),
            "not_a_model_claim": "no estimator locked; no calibration; no artifact; no serving contract; no production freeze",
            "materialized_at": self.materialized_at,
        }

    def choose_multivariate_reference(self, track: str, eligible_stages: list[str]) -> pd.Series:
        stage_list = ", ".join(f"'{stage}'" for stage in eligible_stages)
        df = self.query_df(f"""
        SELECT model_name, model_stage, track, feature_count, features, auc, log_loss, brier
        FROM {bq_table('cxg_split_multivariate_results_v1')}
        WHERE run_id = '{self.run_id}'
          AND split = 'validation'
          AND track = '{track}'
          AND model_stage IN ({stage_list})
          AND auc IS NOT NULL
        ORDER BY auc DESC, log_loss ASC, feature_count ASC
        LIMIT 1
        """)
        if df.empty:
            raise RuntimeError(f"No eligible validation multivariate reference found for {track}")
        return df.iloc[0]

    def univariate_evidence(self, track: str, features: list[str]) -> dict[str, object]:
        quoted = ", ".join("'" + f.replace("'", "\\'") + "'" for f in features)
        df = self.query_df(f"""
        SELECT column_name, feature_family, abs_signal, point_biserial_corr, non_null_count
        FROM {bq_table('cxg_split_univariate_v1')}
        WHERE run_id = '{self.run_id}'
          AND split = 'validation'
          AND track = '{track}'
          AND column_name IN ({quoted})
          AND abs_signal IS NOT NULL
        ORDER BY abs_signal DESC, non_null_count DESC
        LIMIT 12
        """)
        return {
            "overlap_count": len(df),
            "top_features": df["column_name"].astype(str).tolist() if not df.empty else [],
        }

    def bivariate_evidence(self, track: str, features: list[str]) -> dict[str, object]:
        quoted = ", ".join("'" + f.replace("'", "\\'") + "'" for f in features)
        df = self.query_df(f"""
        SELECT left_column, right_column, lift_vs_split_baseline, row_count
        FROM {bq_table('cxg_split_bivariate_v1')}
        WHERE run_id = '{self.run_id}'
          AND split = 'validation'
          AND track = '{track}'
          AND left_column IN ({quoted})
          AND right_column IN ({quoted})
          AND lift_vs_split_baseline IS NOT NULL
        ORDER BY lift_vs_split_baseline DESC, row_count DESC
        LIMIT 8
        """)
        pairs = []
        if not df.empty:
            for _, row in df.iterrows():
                pairs.append(f"{row['left_column']} + {row['right_column']}")
        return {"overlap_pair_count": len(df), "top_pairs": pairs}

    def load_rows(self, table_name: str, rows: list[dict[str, object]]) -> None:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        self.bq.load_table_from_json(rows, table(table_name), job_config=job_config, location=LOCATION).result()

    def write_manifest(self, specs: list[dict[str, object]]) -> Path:
        path = self.output_dir / "cxg_above_baseline_model_specs_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "spec_version": SPEC_VERSION,
                    "materialized_at": self.materialized_at,
                    "model_specs": specs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def write_report(self, specs: list[dict[str, object]]) -> Path:
        path = self.output_dir / "cxg_above_baseline_model_specs.md"
        lines = [
            "# CxG / CxG+ Above-Baseline Model Specs",
            "",
            f"Run: `{self.run_id}`",
            f"Spec version: `{SPEC_VERSION}`",
            "",
            "Only model specs are defined here. No estimator, calibration, model artifact, serving contract, production freeze, or final evaluation is created.",
            "",
            "## Specs",
            "",
            "| Track | Spec | Baseline | Population | Features | Multivariate reference |",
            "|---|---|---|---|---:|---|",
        ]
        for spec in specs:
            lines.append(
                f"| {spec['track']} | `{spec['model_spec_name']}` | `{spec['baseline_name']}` | "
                f"{spec['population_filter']} | {spec['feature_count']} | `{spec['multivariate_reference_model']}` |"
            )
        lines.extend(["", "## Feature Lists", ""])
        for spec in specs:
            lines.append(f"### {spec['track']} - {spec['model_spec_name']}")
            lines.append("")
            for feature in str(spec["feature_names"]).split(","):
                lines.append(f"- `{feature}`")
            lines.append("")
            lines.append(f"Top univariate evidence: `{spec['top_univariate_validation_features']}`")
            lines.append("")
            lines.append(f"Top bivariate evidence: `{spec['top_bivariate_validation_pairs']}`")
            lines.append("")
        lines.extend([
            "## Saved Outputs",
            "",
            f"- BigQuery: `{table('cxg_above_baseline_model_spec_v1')}`",
            f"- Manifest: `{self.output_dir / 'cxg_above_baseline_model_specs_manifest.json'}`",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def validate(self) -> dict[str, object]:
        rows = list(self.bq.query(f"SELECT COUNT(*) AS n FROM {bq_table('cxg_above_baseline_model_spec_v1')}", location=LOCATION).result())
        row_count = int(rows[0]["n"])
        manifest_exists = (self.output_dir / "cxg_above_baseline_model_specs_manifest.json").exists()
        report_exists = (self.output_dir / "cxg_above_baseline_model_specs.md").exists()
        checks = {
            "model_spec_rows": row_count,
            "manifest_exists": manifest_exists,
            "report_exists": report_exists,
            "passed": row_count == 2 and manifest_exists and report_exists,
        }
        if not checks["passed"]:
            raise RuntimeError(f"Result-based model spec validation failed: {checks}")
        return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", default="audit_outputs/cxg_analysis")
    args = parser.parse_args()
    materializer = ResultBasedModelSpecMaterializer(args.run_id, Path(args.output_dir))
    print(json.dumps(materializer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
