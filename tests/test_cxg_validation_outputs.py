import json
from pathlib import Path

import pandas as pd

from scripts.validate_cxg_outputs import CxGValidationPaths, validate_cxg_outputs


def _validation_paths(tmp_path: Path) -> CxGValidationPaths:
    reports_dir = tmp_path / "outputs" / "modeling" / "cxg" / "reports"
    return CxGValidationPaths(
        predictions_path=tmp_path
        / "outputs"
        / "modeling"
        / "cxg"
        / "predictions"
        / "shot_predictions.parquet",
        metrics_path=reports_dir / "metrics.json",
        validation_summary_path=reports_dir / "validation_summary.json",
        calibration_table_path=reports_dir / "calibration_table.csv",
        slice_metrics_path=reports_dir / "slice_metrics.csv",
    )


def _prediction_frame(include_baseline: bool = True) -> pd.DataFrame:
    rows = []
    for i in range(24):
        is_goal = int(i % 5 == 0)
        row = {
            "shot_id": i,
            "match_id": i // 4,
            "is_goal": is_goal,
            "cxg_raw": 0.08 + (0.22 * is_goal) + ((i % 4) * 0.015),
            "pressure_state": "under_pressure" if i % 3 == 0 else "no_pressure",
            "shot_type": "Open Play" if i % 4 else "Free Kick",
            "body_part": "Head" if i % 6 == 0 else "Right Foot",
            "score_state": "drawing" if i % 2 else "leading",
            "shot_distance": 8.0 + i,
        }
        if include_baseline:
            row["statsbomb_xg"] = 0.06 + (0.18 * is_goal) + ((i % 3) * 0.01)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_inputs(paths: CxGValidationPaths, predictions: pd.DataFrame) -> None:
    paths.predictions_path.parent.mkdir(parents=True)
    paths.metrics_path.parent.mkdir(parents=True)
    predictions.to_parquet(paths.predictions_path, index=False)
    paths.metrics_path.write_text(
        json.dumps(
            {
                "brier_mean": 0.08,
                "log_loss_mean": 0.32,
                "auc_mean": 0.81,
                "n_rows": len(predictions),
                "n_splits": 3,
                "folds": [
                    {"fold": 1, "brier": 0.07, "log_loss": 0.3, "auc": 0.8},
                    {"fold": 2, "brier": 0.09, "log_loss": 0.34, "auc": 0.82},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_cxg_validation_outputs_are_created(tmp_path: Path):
    paths = _validation_paths(tmp_path)
    _write_inputs(paths, _prediction_frame(include_baseline=True))

    summary = validate_cxg_outputs(paths, n_bins=5, min_slice_size=2)

    assert paths.validation_summary_path.exists()
    assert paths.calibration_table_path.exists()
    assert paths.slice_metrics_path.exists()
    assert summary["main_model"]["row_count"] == 24
    assert summary["baseline_comparison"]["status"] == "computed"
    assert summary["baseline_comparison"]["baseline_column"] == "statsbomb_xg"
    assert summary["fold_metrics"]["status"] == "computed"
    assert summary["grouped_validation"]["status"] == "computed"
    assert len(pd.read_csv(paths.calibration_table_path)) == 5
    assert not pd.read_csv(paths.slice_metrics_path).empty


def test_cxg_validation_skips_baseline_when_column_missing(tmp_path: Path):
    paths = _validation_paths(tmp_path)
    _write_inputs(paths, _prediction_frame(include_baseline=False))

    summary = validate_cxg_outputs(paths, n_bins=4, min_slice_size=2)

    assert summary["baseline_comparison"]["status"] == "skipped_no_baseline_column"
    assert "No provider or baseline xG column" in summary["baseline_comparison"]["reason"]


def test_cxg_validation_handles_missing_optional_slice_columns(tmp_path: Path):
    paths = _validation_paths(tmp_path)
    predictions = _prediction_frame(include_baseline=False)[["shot_id", "is_goal", "cxg_raw"]]
    _write_inputs(paths, predictions)

    summary = validate_cxg_outputs(paths, n_bins=3, min_slice_size=2)

    assert summary["grouped_validation"]["status"] == "skipped_no_match_id"
    assert summary["slices"]["slice_row_count"] == 0
    assert pd.read_csv(paths.slice_metrics_path).empty


def test_cxg_validation_handles_single_class_metrics(tmp_path: Path):
    paths = _validation_paths(tmp_path)
    predictions = _prediction_frame(include_baseline=False)
    predictions["is_goal"] = 0
    _write_inputs(paths, predictions)

    summary = validate_cxg_outputs(paths, n_bins=4, min_slice_size=2)

    assert summary["main_model"]["goal_count"] == 0
    assert summary["main_model"]["log_loss"] >= 0
    assert summary["main_model"]["roc_auc"] is None
    assert summary["main_model"]["roc_auc_status"] == "skipped_single_class"
