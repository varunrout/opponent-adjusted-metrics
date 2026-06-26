#!/usr/bin/env python
"""Generate validation reports from CxG prediction outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


DEFAULT_MODELING_DIR = Path("outputs") / "modeling" / "cxg"
PREDICTION_COLUMN = "cxg_raw"
TARGET_COLUMN = "is_goal"
BASELINE_CANDIDATES = (
    "statsbomb_xg",
    "provider_xg",
    "baseline_xg",
    "xg",
    "shot_xg",
)
SLICE_COLUMNS = (
    "pressure_state",
    "shot_type",
    "body_part",
    "score_state",
    "simple_state",
    "minute_bucket_label",
    "set_piece_category",
)
SLICE_METRIC_COLUMNS = (
    "slice_column",
    "slice_value",
    "row_count",
    "goal_count",
    "goal_rate",
    "mean_predicted_cxg",
    "brier",
    "log_loss",
    "roc_auc",
    "roc_auc_status",
    "baseline_status",
)


@dataclass(frozen=True)
class CxGValidationPaths:
    """Input and output paths for generated CxG validation reports."""

    predictions_path: Path
    metrics_path: Path
    validation_summary_path: Path
    calibration_table_path: Path
    slice_metrics_path: Path

    @classmethod
    def from_modeling_dir(cls, modeling_dir: Path = DEFAULT_MODELING_DIR) -> "CxGValidationPaths":
        reports_dir = modeling_dir / "reports"
        return cls(
            predictions_path=modeling_dir / "predictions" / "shot_predictions.parquet",
            metrics_path=reports_dir / "metrics.json",
            validation_summary_path=reports_dir / "validation_summary.json",
            calibration_table_path=reports_dir / "calibration_table.csv",
            slice_metrics_path=reports_dir / "slice_metrics.csv",
        )


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _prepare_eval_frame(df: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    missing = [column for column in (TARGET_COLUMN, prediction_column) if column not in df.columns]
    if missing:
        raise ValueError(f"CxG predictions are missing required columns: {missing}")

    eval_df = df.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    if eval_df.empty:
        raise ValueError("CxG predictions contain no rows with target and prediction values")

    eval_df[TARGET_COLUMN] = eval_df[TARGET_COLUMN].astype(int)
    eval_df[prediction_column] = eval_df[prediction_column].astype(float).clip(0.0, 1.0)
    if "shot_distance" in eval_df.columns:
        eval_df["distance_bucket"] = pd.cut(
            eval_df["shot_distance"],
            bins=[-np.inf, 12.0, 20.0, np.inf],
            labels=["close", "medium", "long"],
        ).astype("object")
        eval_df["distance_bucket"] = eval_df["distance_bucket"].fillna("unknown")
    return eval_df


def _metric_summary(df: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    y_true = df[TARGET_COLUMN].astype(int).to_numpy()
    y_pred = df[prediction_column].astype(float).clip(0.0, 1.0).to_numpy()
    both_classes = len(np.unique(y_true)) == 2

    summary: dict[str, Any] = {
        "row_count": int(len(df)),
        "goal_count": int(y_true.sum()),
        "goal_rate": float(y_true.mean()),
        "mean_predicted_cxg": float(y_pred.mean()),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "roc_auc": None,
        "roc_auc_status": "skipped_single_class",
    }
    if both_classes:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        summary["roc_auc_status"] = "computed"
    return summary


def _fold_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    folds = metrics.get("folds")
    if not isinstance(folds, list) or not folds:
        return {"status": "skipped_no_fold_metrics"}

    summary: dict[str, Any] = {"status": "computed", "fold_count": len(folds)}
    for key in ("brier", "log_loss", "auc"):
        values = [
            float(row[key])
            for row in folds
            if isinstance(row, dict) and key in row and pd.notna(row[key])
        ]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return summary


def _grouped_summary(df: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    if "match_id" not in df.columns:
        return {"status": "skipped_no_match_id"}

    group_sizes = df.groupby("match_id", dropna=False).size()
    return {
        "status": "computed",
        "group_column": "match_id",
        "group_count": int(group_sizes.size),
        "min_rows_per_group": int(group_sizes.min()),
        "median_rows_per_group": float(group_sizes.median()),
        "max_rows_per_group": int(group_sizes.max()),
        "overall_grouped_rows": int(group_sizes.sum()),
        "metrics": _metric_summary(df, prediction_column),
    }


def _baseline_summary(df: pd.DataFrame) -> dict[str, Any]:
    baseline_column = next(
        (
            column
            for column in BASELINE_CANDIDATES
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )
    if baseline_column is None:
        return {
            "status": "skipped_no_baseline_column",
            "reason": "No provider or baseline xG column was present.",
            "candidate_columns": list(BASELINE_CANDIDATES),
        }

    baseline_df = df.dropna(subset=[baseline_column]).copy()
    baseline_df[baseline_column] = baseline_df[baseline_column].astype(float).clip(0.0, 1.0)
    summary = _metric_summary(baseline_df, baseline_column)
    summary.update({"status": "computed", "baseline_column": baseline_column})
    return summary


def _calibration_table(
    df: pd.DataFrame,
    prediction_column: str,
    *,
    n_bins: int,
) -> pd.DataFrame:
    y_pred = df[prediction_column].astype(float).clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_pred, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    records = []
    for bin_id in range(n_bins):
        subset = df.loc[bin_ids == bin_id]
        if subset.empty:
            records.append(
                {
                    "bin": bin_id + 1,
                    "lower_bound": float(bins[bin_id]),
                    "upper_bound": float(bins[bin_id + 1]),
                    "row_count": 0,
                    "mean_predicted_cxg": np.nan,
                    "goal_rate": np.nan,
                    "calibration_error": np.nan,
                }
            )
            continue
        mean_pred = float(subset[prediction_column].mean())
        goal_rate = float(subset[TARGET_COLUMN].mean())
        records.append(
            {
                "bin": bin_id + 1,
                "lower_bound": float(bins[bin_id]),
                "upper_bound": float(bins[bin_id + 1]),
                "row_count": int(len(subset)),
                "mean_predicted_cxg": mean_pred,
                "goal_rate": goal_rate,
                "calibration_error": abs(goal_rate - mean_pred),
            }
        )
    return pd.DataFrame(records)


def _slice_metric_rows(
    df: pd.DataFrame,
    prediction_column: str,
    *,
    min_slice_size: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    slice_columns = [
        column for column in (*SLICE_COLUMNS, "distance_bucket") if column in df.columns
    ]

    for column in slice_columns:
        values = df[column].fillna("unknown").astype(str)
        for value, subset in df.groupby(values, dropna=False):
            if len(subset) < min_slice_size:
                continue
            metrics = _metric_summary(subset, prediction_column)
            metrics.update(
                {
                    "slice_column": column,
                    "slice_value": str(value),
                    "baseline_status": _baseline_summary(subset)["status"],
                }
            )
            records.append(metrics)
    return records


def validate_cxg_outputs(
    paths: CxGValidationPaths,
    *,
    prediction_column: str = PREDICTION_COLUMN,
    n_bins: int = 10,
    min_slice_size: int = 1,
) -> dict[str, Any]:
    """Generate validation reports for generated CxG predictions."""

    predictions = pd.read_parquet(paths.predictions_path)
    eval_df = _prepare_eval_frame(predictions, prediction_column)
    metrics = _read_json_if_exists(paths.metrics_path)

    paths.validation_summary_path.parent.mkdir(parents=True, exist_ok=True)

    calibration = _calibration_table(eval_df, prediction_column, n_bins=n_bins)
    calibration.to_csv(paths.calibration_table_path, index=False)

    slice_metrics = pd.DataFrame(
        _slice_metric_rows(eval_df, prediction_column, min_slice_size=min_slice_size),
        columns=SLICE_METRIC_COLUMNS,
    )
    slice_metrics.to_csv(paths.slice_metrics_path, index=False)

    validation_summary = {
        "prediction_column": prediction_column,
        "inputs": {
            "predictions_path": str(paths.predictions_path),
            "metrics_path": str(paths.metrics_path),
            "metrics_json_present": paths.metrics_path.exists(),
        },
        "outputs": {
            "validation_summary": str(paths.validation_summary_path),
            "calibration_table": str(paths.calibration_table_path),
            "slice_metrics": str(paths.slice_metrics_path),
        },
        "main_model": _metric_summary(eval_df, prediction_column),
        "baseline_comparison": _baseline_summary(eval_df),
        "fold_metrics": _fold_summary(metrics),
        "grouped_validation": _grouped_summary(eval_df, prediction_column),
        "calibration": {
            "bin_count": n_bins,
            "non_empty_bin_count": int((calibration["row_count"] > 0).sum()),
            "mean_absolute_calibration_error": float(
                calibration["calibration_error"].dropna().mean()
            ),
        },
        "slices": {
            "min_slice_size": min_slice_size,
            "slice_row_count": int(len(slice_metrics)),
            "slice_columns": (
                sorted(slice_metrics["slice_column"].unique().tolist())
                if not slice_metrics.empty
                else []
            ),
        },
    }

    paths.validation_summary_path.write_text(
        json.dumps(_json_safe(validation_summary), indent=2),
        encoding="utf-8",
    )
    return validation_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CxG validation reports")
    parser.add_argument("--modeling-dir", type=Path, default=DEFAULT_MODELING_DIR)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--prediction-column", default=PREDICTION_COLUMN)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-slice-size", type=int, default=1)
    args = parser.parse_args()

    paths = CxGValidationPaths.from_modeling_dir(args.modeling_dir)
    reports_dir = args.reports_dir or paths.validation_summary_path.parent
    paths = CxGValidationPaths(
        predictions_path=args.predictions or paths.predictions_path,
        metrics_path=args.metrics or paths.metrics_path,
        validation_summary_path=reports_dir / "validation_summary.json",
        calibration_table_path=reports_dir / "calibration_table.csv",
        slice_metrics_path=reports_dir / "slice_metrics.csv",
    )

    summary = validate_cxg_outputs(
        paths,
        prediction_column=args.prediction_column,
        n_bins=args.bins,
        min_slice_size=args.min_slice_size,
    )
    print(json.dumps(_json_safe(summary["outputs"]), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
