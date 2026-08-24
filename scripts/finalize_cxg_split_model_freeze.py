"""Freeze CxG/CxG+ split-aware model choices and write final test report."""

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
FREEZE_VERSION = "cxg_split_model_freeze_v1"


def table(name: str) -> str:
    return f"{PROJECT_ID}.{ANALYSIS_DATASET}.{name}"


def bq_table(name: str) -> str:
    return f"`{PROJECT_ID}.{ANALYSIS_DATASET}.{name}`"


class SplitModelFreezer:
    def __init__(self, run_id: str, output_dir: Path) -> None:
        self.run_id = run_id
        self.output_dir = output_dir / run_id / "split_model_freeze"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.frozen_at = datetime.now(UTC).isoformat()

    def query_df(self, sql: str) -> pd.DataFrame:
        rows = list(self.bq.query(sql, location=LOCATION).result())
        return pd.DataFrame([dict(row.items()) for row in rows]) if rows else pd.DataFrame()

    def run(self) -> dict[str, object]:
        results = self.load_results()
        if results.empty:
            raise RuntimeError("No split multivariate results found")
        selected = [
            self.choose_event_model(results),
            self.choose_plus_model(results),
        ]
        freeze_rows = [self.freeze_row(results, choice) for choice in selected]
        self.materialize_freeze_table(freeze_rows)
        manifest_path = self.write_manifest(freeze_rows)
        report_path = self.write_report(results, freeze_rows)
        validation = self.validate()
        return {
            "run_id": self.run_id,
            "freeze_version": FREEZE_VERSION,
            "frozen_models": [row["selected_model_name"] for row in freeze_rows],
            "freeze_table": table("cxg_final_model_freeze_v1"),
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "validation": validation,
        }

    def load_results(self) -> pd.DataFrame:
        return self.query_df(f"""
        SELECT
          run_id, model_name, model_stage, track, split, feature_count, features,
          row_count, goal_count, goal_rate, auc, log_loss, brier,
          trained_on_split, evaluated_at
        FROM {bq_table('cxg_split_multivariate_results_v1')}
        WHERE run_id = '{self.run_id}'
        """)

    def choose_event_model(self, results: pd.DataFrame) -> dict[str, str]:
        validation = results[
            (results["track"] == "cxg_event")
            & (results["split"] == "validation")
            & (results["model_stage"] == "candidate_model")
        ].copy()
        if validation.empty:
            raise RuntimeError("No CxG event candidate model found")
        best = validation.sort_values(["auc", "log_loss"], ascending=[False, True]).iloc[0]
        return {
            "track": "cxg_event",
            "selected_model_name": str(best["model_name"]),
            "selection_basis": "Best eligible CxG event validation candidate over true XY baseline.",
            "final_decision": "freeze_for_final_test_report",
        }

    def choose_plus_model(self, results: pd.DataFrame) -> dict[str, str]:
        validation = results[
            (results["track"] == "cxg_plus")
            & (results["split"] == "validation")
            & (results["model_stage"].isin(["candidate_model", "multivariate_experiment"]))
        ].copy()
        if validation.empty:
            raise RuntimeError("No CxG+ eligible model found")
        best = validation.sort_values(["auc", "log_loss"], ascending=[False, True]).iloc[0]
        return {
            "track": "cxg_plus",
            "selected_model_name": str(best["model_name"]),
            "selection_basis": "Best eligible CxG+ validation model; selected over selected-core because validation AUC was higher.",
            "final_decision": "freeze_for_final_test_report",
        }

    def metric_row(self, results: pd.DataFrame, model_name: str, split: str) -> pd.Series:
        row = results[(results["model_name"] == model_name) & (results["split"] == split)]
        if row.empty:
            raise RuntimeError(f"Missing {split} row for {model_name}")
        return row.iloc[0]

    def baseline_model(self, track: str) -> str:
        return "cxg_event_base_xy" if track == "cxg_event" else "cxg_plus_base_xy"

    def freeze_row(self, results: pd.DataFrame, choice: dict[str, str]) -> dict[str, object]:
        model = choice["selected_model_name"]
        track = choice["track"]
        baseline = self.baseline_model(track)
        train = self.metric_row(results, model, "train")
        val = self.metric_row(results, model, "validation")
        test = self.metric_row(results, model, "test")
        base_val = self.metric_row(results, baseline, "validation")
        base_test = self.metric_row(results, baseline, "test")
        return {
            "run_id": self.run_id,
            "freeze_version": FREEZE_VERSION,
            "track": track,
            "selected_model_name": model,
            "model_stage": str(val["model_stage"]),
            "feature_count": int(val["feature_count"]),
            "features": str(val["features"]),
            "baseline_model_name": baseline,
            "selection_basis": choice["selection_basis"],
            "final_decision": choice["final_decision"],
            "trained_on_split": str(val["trained_on_split"]),
            "train_row_count": int(train["row_count"]),
            "train_goal_count": int(train["goal_count"]),
            "train_auc": self.num(train["auc"]),
            "train_log_loss": self.num(train["log_loss"]),
            "train_brier": self.num(train["brier"]),
            "validation_row_count": int(val["row_count"]),
            "validation_goal_count": int(val["goal_count"]),
            "validation_auc": self.num(val["auc"]),
            "validation_log_loss": self.num(val["log_loss"]),
            "validation_brier": self.num(val["brier"]),
            "baseline_validation_auc": self.num(base_val["auc"]),
            "validation_auc_delta_vs_baseline": self.num(val["auc"] - base_val["auc"]),
            "test_row_count": int(test["row_count"]),
            "test_goal_count": int(test["goal_count"]),
            "test_auc": self.num(test["auc"]),
            "test_log_loss": self.num(test["log_loss"]),
            "test_brier": self.num(test["brier"]),
            "baseline_test_auc": self.num(base_test["auc"]),
            "test_auc_delta_vs_baseline": self.num(test["auc"] - base_test["auc"]),
            "frozen_at": self.frozen_at,
        }

    @staticmethod
    def num(value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

    def materialize_freeze_table(self, rows: list[dict[str, object]]) -> None:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        self.bq.load_table_from_json(rows, table("cxg_final_model_freeze_v1"), job_config=job_config, location=LOCATION).result()

    def write_manifest(self, rows: list[dict[str, object]]) -> Path:
        path = self.output_dir / "cxg_final_model_freeze_manifest.json"
        path.write_text(json.dumps({"run_id": self.run_id, "freeze_version": FREEZE_VERSION, "models": rows}, indent=2), encoding="utf-8")
        return path

    def write_report(self, results: pd.DataFrame, rows: list[dict[str, object]]) -> Path:
        path = self.output_dir / "cxg_final_test_report.md"
        lines = [
            "# CxG / CxG+ Final Split-Aware Model Freeze and Test Report",
            "",
            f"Run: `{self.run_id}`",
            f"Freeze version: `{FREEZE_VERSION}`",
            f"Frozen at: `{self.frozen_at}`",
            "",
            "## Freeze Decisions",
            "",
            "| Track | Frozen model | Stage | Features | Validation AUC | Baseline val AUC | Val delta | Test AUC | Baseline test AUC | Test delta | Test log loss | Test Brier |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['track']} | `{row['selected_model_name']}` | {row['model_stage']} | {row['feature_count']} | "
                f"{row['validation_auc']:.4f} | {row['baseline_validation_auc']:.4f} | {row['validation_auc_delta_vs_baseline']:.4f} | "
                f"{row['test_auc']:.4f} | {row['baseline_test_auc']:.4f} | {row['test_auc_delta_vs_baseline']:.4f} | "
                f"{row['test_log_loss']:.4f} | {row['test_brier']:.4f} |"
            )
        lines.extend([
            "",
            "## Interpretation",
            "",
            "- CxG event freezes the event-context candidate because validation AUC improves over the shot-location true baseline.",
            "- CxG+ freezes the best validation 360-family model rather than selected-core because validation AUC is materially stronger.",
            "- Test results are reported once from the already materialized split-aware audit table; further model-choice iteration should go back to validation only.",
            "",
            "## Frozen Feature Lists",
            "",
        ])
        for row in rows:
            features = [f.strip() for f in str(row["features"]).split(",") if f.strip()]
            lines.append(f"### {row['track']} - {row['selected_model_name']}")
            lines.append("")
            for feature in features:
                lines.append(f"- `{feature}`")
            lines.append("")
        lines.extend([
            "## Saved Outputs",
            "",
            f"- BigQuery: `{table('cxg_final_model_freeze_v1')}`",
            f"- Local manifest: `{self.output_dir / 'cxg_final_model_freeze_manifest.json'}`",
            f"- Local report: `{path}`",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def validate(self) -> dict[str, object]:
        row = self.query_df(f"SELECT COUNT(*) AS n FROM {bq_table('cxg_final_model_freeze_v1')} WHERE run_id = '{self.run_id}'")
        count = int(row.iloc[0]["n"]) if not row.empty else 0
        manifest_exists = (self.output_dir / "cxg_final_model_freeze_manifest.json").exists()
        report_exists = (self.output_dir / "cxg_final_test_report.md").exists()
        checks = {
            "freeze_table_rows": count,
            "manifest_exists": manifest_exists,
            "report_exists": report_exists,
            "passed": count == 2 and manifest_exists and report_exists,
        }
        if not checks["passed"]:
            raise RuntimeError(f"Freeze validation failed: {checks}")
        return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", default="audit_outputs/cxg_analysis")
    args = parser.parse_args()
    freezer = SplitModelFreezer(args.run_id, Path(args.output_dir))
    print(json.dumps(freezer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
