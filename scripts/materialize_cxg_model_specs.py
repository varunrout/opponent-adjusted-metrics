"""Materialize CxG/CxG+ model specs and baseline definitions.

This intentionally does not train, freeze, promote, calibrate, or evaluate a
production model. It records the next modelling contract only.
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

from google.cloud import bigquery

PROJECT_ID = "oam-varun-260819"
LOCATION = "europe-west2"
ANALYSIS_DATASET = "oam_analysis"
RUN_ID = "cxg-analysis-20260820T201934Z"
SPEC_VERSION = "cxg_model_spec_v1"

CXG_EVENT_BASELINE_FEATURES = ["shot_x_sb", "shot_y_sb"]
CXG_PLUS_BASELINE_FEATURES = ["shot_x_sb", "shot_y_sb"]

CXG_EVENT_CANDIDATE_FEATURES = [
    "shot_x_sb",
    "shot_y_sb",
    "last_action_interval_s",
    "previous_action_time_gap_s",
    "last_box_entry_to_shot_s",
    "possession_directness_proxy",
    "regain_height_speed_interaction",
    "box_actions_before_shot",
    "match_minute",
    "first_box_entry_to_shot_s",
    "possession_start_goal_distance",
]

CXG_PLUS_CANDIDATE_FEATURES = [
    "shot_x_sb",
    "shot_y_sb",
    "defenders_between_ball_and_goal",
    "defensive_reset_index",
    "rest_defence_reset_fraction",
    "shooter_space_previous_linked_event",
    "pre_shot_receiver_space",
    "nearest_defender_distance_delta",
    "local_density_delta",
    "gk_distance_to_shooter",
    "visible_goal_angle_proxy",
    "goal_mouth_defender_count",
    "visible_goal_angle_delta",
    "estimated_goalface_occlusion",
    "defensive_line_depth",
    "defensive_hull_area",
    "shot_corridor_occlusion",
    "min_defensive_compactness_sequence",
    "defensive_centroid_x",
    "defensive_width",
    "defensive_compactness",
    "max_goal_exposure",
    "defensive_length",
    "mean_defensive_compactness_last_3",
    "goal_exposure_decay",
]


def table(name: str) -> str:
    return f"{PROJECT_ID}.{ANALYSIS_DATASET}.{name}"


class ModelSpecMaterializer:
    def __init__(self, run_id: str, output_dir: Path, drop_misleading_freeze: bool) -> None:
        self.run_id = run_id
        self.drop_misleading_freeze = drop_misleading_freeze
        self.output_dir = output_dir / run_id / "model_specs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bq = bigquery.Client(project=PROJECT_ID)
        self.materialized_at = datetime.now(UTC).isoformat()

    def run(self) -> dict[str, object]:
        dropped = []
        if self.drop_misleading_freeze:
            dropped = self.drop_misleading_tables()
        baseline_rows = self.baseline_rows()
        model_rows = self.model_rows()
        self.load_rows("cxg_baseline_spec_v1", baseline_rows)
        self.load_rows("cxg_model_spec_v1", model_rows)
        manifest_path = self.write_manifest(baseline_rows, model_rows, dropped)
        report_path = self.write_report(baseline_rows, model_rows, dropped)
        validation = self.validate()
        return {
            "run_id": self.run_id,
            "spec_version": SPEC_VERSION,
            "baseline_table": table("cxg_baseline_spec_v1"),
            "model_spec_table": table("cxg_model_spec_v1"),
            "dropped_misleading_tables": dropped,
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "validation": validation,
        }

    def drop_misleading_tables(self) -> list[str]:
        tables = ["cxg_final_model_freeze_v1"]
        for name in tables:
            self.bq.delete_table(table(name), not_found_ok=True)
        return tables

    def baseline_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_id": self.run_id,
                "spec_version": SPEC_VERSION,
                "track": "cxg_event",
                "baseline_name": "cxg_event_true_baseline_xy",
                "baseline_type": "true_baseline",
                "population_table": table("cxg_event_model_matrix_v1"),
                "population_filter": "all CxG shots",
                "target_column": "is_goal",
                "feature_names": ",".join(CXG_EVENT_BASELINE_FEATURES),
                "feature_count": len(CXG_EVENT_BASELINE_FEATURES),
                "purpose": "Minimum honest shot-location baseline for event-wide CxG model comparison.",
                "training_split": "train",
                "selection_split": "validation",
                "test_split_policy": "reserved for one final locked evaluation only",
                "materialized_at": self.materialized_at,
            },
            {
                "run_id": self.run_id,
                "spec_version": SPEC_VERSION,
                "track": "cxg_plus",
                "baseline_name": "cxg_plus_true_baseline_xy",
                "baseline_type": "true_baseline",
                "population_table": table("cxg_plus_360_model_matrix_v1"),
                "population_filter": "has_360_frame CxG+ shots only",
                "target_column": "is_goal",
                "feature_names": ",".join(CXG_PLUS_BASELINE_FEATURES),
                "feature_count": len(CXG_PLUS_BASELINE_FEATURES),
                "purpose": "Minimum honest shot-location baseline for 360-eligible CxG+ model comparison.",
                "training_split": "train",
                "selection_split": "validation",
                "test_split_policy": "reserved for one final locked evaluation only",
                "materialized_at": self.materialized_at,
            },
        ]

    def model_rows(self) -> list[dict[str, object]]:
        return [
            {
                "run_id": self.run_id,
                "spec_version": SPEC_VERSION,
                "track": "cxg_event",
                "model_spec_name": "cxg_event_candidate_event_context_v1",
                "model_spec_status": "candidate_spec_ready",
                "population_table": table("cxg_event_model_matrix_v1"),
                "population_filter": "all CxG shots",
                "target_column": "is_goal",
                "baseline_name": "cxg_event_true_baseline_xy",
                "feature_names": ",".join(CXG_EVENT_CANDIDATE_FEATURES),
                "feature_count": len(CXG_EVENT_CANDIDATE_FEATURES),
                "required_preprocessing": "numeric imputation and scaling; final estimator not locked",
                "estimator_status": "not_selected",
                "calibration_status": "not_started",
                "artifact_status": "not_created",
                "serving_status": "not_started",
                "selection_rule": "compare against cxg_event_true_baseline_xy on validation only before any final test use",
                "materialized_at": self.materialized_at,
            },
            {
                "run_id": self.run_id,
                "spec_version": SPEC_VERSION,
                "track": "cxg_plus",
                "model_spec_name": "cxg_plus_candidate_360_context_v1",
                "model_spec_status": "candidate_spec_ready",
                "population_table": table("cxg_plus_360_model_matrix_v1"),
                "population_filter": "has_360_frame CxG+ shots only",
                "target_column": "is_goal",
                "baseline_name": "cxg_plus_true_baseline_xy",
                "feature_names": ",".join(CXG_PLUS_CANDIDATE_FEATURES),
                "feature_count": len(CXG_PLUS_CANDIDATE_FEATURES),
                "required_preprocessing": "numeric imputation and scaling; final estimator not locked",
                "estimator_status": "not_selected",
                "calibration_status": "not_started",
                "artifact_status": "not_created",
                "serving_status": "not_started",
                "selection_rule": "compare against cxg_plus_true_baseline_xy on validation only before any final test use",
                "materialized_at": self.materialized_at,
            },
        ]

    def load_rows(self, table_name: str, rows: list[dict[str, object]]) -> None:
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        self.bq.load_table_from_json(rows, table(table_name), job_config=job_config, location=LOCATION).result()

    def write_manifest(self, baseline_rows: list[dict[str, object]], model_rows: list[dict[str, object]], dropped: list[str]) -> Path:
        path = self.output_dir / "cxg_model_specs_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "spec_version": SPEC_VERSION,
                    "materialized_at": self.materialized_at,
                    "dropped_misleading_tables": dropped,
                    "baselines": baseline_rows,
                    "model_specs": model_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def write_report(self, baseline_rows: list[dict[str, object]], model_rows: list[dict[str, object]], dropped: list[str]) -> Path:
        path = self.output_dir / "cxg_model_specs.md"
        lines = [
            "# CxG / CxG+ Model Specs and Baselines",
            "",
            f"Run: `{self.run_id}`",
            f"Spec version: `{SPEC_VERSION}`",
            f"Materialized at: `{self.materialized_at}`",
            "",
            "This is a modelling contract only. It does not freeze, promote, calibrate, or evaluate a production model.",
            "",
            "## Baselines",
            "",
            "| Track | Baseline | Population | Features | Purpose |",
            "|---|---|---|---:|---|",
        ]
        for row in baseline_rows:
            lines.append(f"| {row['track']} | `{row['baseline_name']}` | {row['population_filter']} | {row['feature_count']} | {row['purpose']} |")
        lines.extend([
            "",
            "## Candidate Specs",
            "",
            "| Track | Candidate spec | Population | Features | Status |",
            "|---|---|---|---:|---|",
        ])
        for row in model_rows:
            lines.append(f"| {row['track']} | `{row['model_spec_name']}` | {row['population_filter']} | {row['feature_count']} | {row['model_spec_status']} |")
        lines.extend([
            "",
            "## Guardrails",
            "",
            "- Baseline for both tracks is shot location only: `shot_x_sb`, `shot_y_sb`.",
            "- CxG event population is all shots in `cxg_event_model_matrix_v1`.",
            "- CxG+ population is only `has_360_frame` shots in `cxg_plus_360_model_matrix_v1`.",
            "- Model choice must use validation only until a future locked final-evaluation protocol is explicitly created.",
            "- Estimator, calibration, model artifact, serving contract and model card are not done yet.",
        ])
        if dropped:
            lines.extend(["", "## Correction", ""])
            lines.append("Removed misleading prior final-freeze table(s): " + ", ".join(f"`{name}`" for name in dropped) + ".")
        lines.extend([
            "",
            "## Saved Outputs",
            "",
            f"- BigQuery: `{table('cxg_baseline_spec_v1')}`",
            f"- BigQuery: `{table('cxg_model_spec_v1')}`",
            f"- Manifest: `{self.output_dir / 'cxg_model_specs_manifest.json'}`",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def validate(self) -> dict[str, object]:
        baseline_count = self.count_rows("cxg_baseline_spec_v1")
        model_count = self.count_rows("cxg_model_spec_v1")
        freeze_exists = self.table_exists("cxg_final_model_freeze_v1")
        manifest_exists = (self.output_dir / "cxg_model_specs_manifest.json").exists()
        report_exists = (self.output_dir / "cxg_model_specs.md").exists()
        checks = {
            "baseline_spec_rows": baseline_count,
            "model_spec_rows": model_count,
            "misleading_freeze_table_exists": freeze_exists,
            "manifest_exists": manifest_exists,
            "report_exists": report_exists,
            "passed": baseline_count == 2 and model_count == 2 and not freeze_exists and manifest_exists and report_exists,
        }
        if not checks["passed"]:
            raise RuntimeError(f"Model spec validation failed: {checks}")
        return checks

    def count_rows(self, table_name: str) -> int:
        rows = list(self.bq.query(f"SELECT COUNT(*) AS n FROM `{table(table_name)}`", location=LOCATION).result())
        return int(rows[0]["n"])

    def table_exists(self, table_name: str) -> bool:
        try:
            self.bq.get_table(table(table_name))
            return True
        except Exception:
            return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output-dir", default="audit_outputs/cxg_analysis")
    parser.add_argument("--keep-misleading-freeze", action="store_true")
    args = parser.parse_args()
    materializer = ModelSpecMaterializer(
        args.run_id,
        Path(args.output_dir),
        drop_misleading_freeze=not args.keep_misleading_freeze,
    )
    print(json.dumps(materializer.run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
