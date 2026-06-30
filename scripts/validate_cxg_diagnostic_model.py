#!/usr/bin/env python
"""Validate diagnostic-informed CxG against the baseline CxG model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

DEFAULT_BASELINE_DIR = Path("outputs/modeling/cxg/baseline")
DEFAULT_DIAGNOSTIC_DIR = Path("outputs/modeling/cxg/diagnostic_v1")
DEFAULT_ANALYSIS_DIR = Path("outputs/analysis/cxg")
DEFAULT_OUTPUT_DIR = Path("outputs/validation/cxg/diagnostic_v1")
TARGET_COLUMN = "is_goal"
BASELINE_PREDICTION_COLUMN = "cxg_raw"
DIAGNOSTIC_PREDICTION_COLUMN = "predicted_cxg"
MODEL_COLUMN = "model_version"
PREDICTION_COLUMN = "predicted_probability"
SLICE_COLUMNS = (
    "body_part",
    "technique",
    "shot_type",
    "play_pattern",
    "set_piece_category",
    "pressure_state",
    "under_pressure",
    "minute_bucket",
    "minute_bucket_label",
    "score_state",
    "def_label",
)
ID_COLUMNS = ("shot_id", "event_id", "match_id", "team_id", "player_id")


@dataclass(frozen=True)
class ValidationPaths:
    """Input and output paths for diagnostic CxG validation."""

    baseline_predictions: Path
    baseline_metrics: Path
    baseline_validation_summary: Path
    baseline_calibration_table: Path
    baseline_slice_metrics: Path
    baseline_metadata: Path
    diagnostic_predictions: Path
    diagnostic_model_comparison: Path
    diagnostic_fold_metrics: Path
    diagnostic_training_summary: Path
    diagnostic_selected_metadata: Path
    diagnostic_feature_contract: Path
    diagnostic_resolved_features: Path
    diagnostic_feature_group_summary: Path
    diagnostic_excluded_columns: Path
    analysis_report: Path
    analysis_slice_stability_dir: Path
    analysis_leakage_dir: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        baseline_dir: Path = DEFAULT_BASELINE_DIR,
        diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
        analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "ValidationPaths":
        return cls(
            baseline_predictions=baseline_dir / "predictions" / "shot_predictions.parquet",
            baseline_metrics=baseline_dir / "reports" / "metrics.json",
            baseline_validation_summary=baseline_dir / "reports" / "validation_summary.json",
            baseline_calibration_table=baseline_dir / "reports" / "calibration_table.csv",
            baseline_slice_metrics=baseline_dir / "reports" / "slice_metrics.csv",
            baseline_metadata=baseline_dir / "models" / "contextual_model.json",
            diagnostic_predictions=diagnostic_dir
            / "predictions"
            / "cross_validated_predictions.parquet",
            diagnostic_model_comparison=diagnostic_dir / "reports" / "model_comparison.csv",
            diagnostic_fold_metrics=diagnostic_dir / "reports" / "fold_metrics.csv",
            diagnostic_training_summary=diagnostic_dir / "reports" / "training_summary.json",
            diagnostic_selected_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            diagnostic_feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            diagnostic_resolved_features=diagnostic_dir / "diagnostics" / "resolved_features.json",
            diagnostic_feature_group_summary=diagnostic_dir
            / "diagnostics"
            / "feature_group_summary.csv",
            diagnostic_excluded_columns=diagnostic_dir / "diagnostics" / "excluded_columns.csv",
            analysis_report=analysis_dir / "report.md",
            analysis_slice_stability_dir=analysis_dir / "04_slice_stability",
            analysis_leakage_dir=analysis_dir / "06_leakage_checks",
            output_dir=output_dir,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
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


def _clip_probabilities(values: pd.Series) -> tuple[pd.Series, bool, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    clipped = numeric.clip(1e-15, 1 - 1e-15)
    changed = numeric.notna() & (numeric != clipped)
    return clipped, bool(changed.any()), int(changed.sum())


def safe_metric_summary(
    df: pd.DataFrame,
    *,
    model_version: str,
    prediction_column: str = PREDICTION_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """Compute safe probability metrics for a model prediction frame."""

    required = [target_column, prediction_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Validation frame is missing required columns: {missing}")
    eval_df = df.copy()
    probs_raw = pd.to_numeric(eval_df[prediction_column], errors="coerce")
    null_count = int(probs_raw.isna().sum())
    eval_df = eval_df.loc[probs_raw.notna() & eval_df[target_column].notna()].copy()
    if eval_df.empty:
        raise ValueError(f"{model_version} has no rows with target and prediction values")
    probabilities, clipping_used, clipping_count = _clip_probabilities(eval_df[prediction_column])
    y_true = eval_df[target_column].astype(int).to_numpy()
    y_pred = probabilities.to_numpy(dtype=float)
    both_classes = len(np.unique(y_true)) == 2
    summary: dict[str, Any] = {
        "model_version": model_version,
        "row_count": int(len(eval_df)),
        "goal_count": int(y_true.sum()),
        "goal_rate": float(y_true.mean()),
        "mean_predicted_probability": float(y_pred.mean()),
        "brier": float(brier_score_loss(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_pred, labels=[0, 1])),
        "roc_auc": np.nan,
        "roc_auc_status": "skipped_single_class",
        "expected_calibration_error": np.nan,
        "probability_min": float(np.nanmin(probs_raw.to_numpy(dtype=float))),
        "probability_max": float(np.nanmax(probs_raw.to_numpy(dtype=float))),
        "probability_null_count": null_count,
        "probability_clipping_used": clipping_used,
        "probability_clipping_count": clipping_count,
    }
    if both_classes:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        summary["roc_auc_status"] = "computed"
    return summary


def calibration_bins(
    df: pd.DataFrame,
    *,
    model_version: str,
    n_bins: int = 10,
    prediction_column: str = PREDICTION_COLUMN,
) -> pd.DataFrame:
    probabilities = pd.to_numeric(df[prediction_column], errors="coerce").clip(0.0, 1.0)
    target = df[TARGET_COLUMN].astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        subset = df.loc[bin_ids == bin_id]
        subset_probs = probabilities.loc[subset.index]
        subset_target = target.loc[subset.index]
        if subset.empty:
            rows.append(
                {
                    "model_version": model_version,
                    "bin_id": bin_id + 1,
                    "probability_lower": float(bins[bin_id]),
                    "probability_upper": float(bins[bin_id + 1]),
                    "rows": 0,
                    "goals": 0,
                    "actual_goal_rate": np.nan,
                    "mean_predicted_probability": np.nan,
                    "calibration_error": np.nan,
                    "absolute_calibration_error": np.nan,
                }
            )
            continue
        actual = float(subset_target.mean())
        mean_pred = float(subset_probs.mean())
        rows.append(
            {
                "model_version": model_version,
                "bin_id": bin_id + 1,
                "probability_lower": float(bins[bin_id]),
                "probability_upper": float(bins[bin_id + 1]),
                "rows": int(len(subset)),
                "goals": int(subset_target.sum()),
                "actual_goal_rate": actual,
                "mean_predicted_probability": mean_pred,
                "calibration_error": actual - mean_pred,
                "absolute_calibration_error": abs(actual - mean_pred),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(calibration: pd.DataFrame) -> float:
    non_empty = calibration.loc[calibration["rows"] > 0]
    if non_empty.empty:
        return np.nan
    total_rows = float(non_empty["rows"].sum())
    weighted = non_empty["absolute_calibration_error"] * non_empty["rows"]
    return float(weighted.sum() / total_rows)


def normalize_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if BASELINE_PREDICTION_COLUMN not in df.columns:
        raise ValueError(f"Baseline predictions missing {BASELINE_PREDICTION_COLUMN}")
    normalized = df.copy()
    normalized[MODEL_COLUMN] = "baseline"
    normalized[PREDICTION_COLUMN] = normalized[BASELINE_PREDICTION_COLUMN]
    return normalized


def normalize_diagnostic_predictions(
    df: pd.DataFrame,
    *,
    selected_model: str,
    baseline_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if "model_candidate" not in df.columns:
        raise ValueError("Diagnostic predictions must include model_candidate")
    selected = df.loc[df["model_candidate"] == selected_model].copy()
    if selected.empty:
        raise ValueError(f"No diagnostic predictions found for selected model {selected_model}")
    selected[MODEL_COLUMN] = f"diagnostic_v1:{selected_model}"
    selected[PREDICTION_COLUMN] = selected[DIAGNOSTIC_PREDICTION_COLUMN]
    if baseline_context is not None:
        selected = _enrich_slice_columns(selected, baseline_context)
    return selected


def _enrich_slice_columns(df: pd.DataFrame, baseline_context: pd.DataFrame) -> pd.DataFrame:
    missing_slice_columns = [column for column in SLICE_COLUMNS if column not in df.columns]
    if not missing_slice_columns:
        return df
    join_keys = [column for column in ("shot_id", "event_id") if column in df.columns]
    if not join_keys:
        join_keys = [
            column
            for column in ("match_id", "team_id", "player_id")
            if column in df.columns and column in baseline_context.columns
        ]
    if not join_keys:
        return df
    context_columns = [
        column
        for column in (*join_keys, *missing_slice_columns)
        if column in baseline_context.columns
    ]
    if len(context_columns) <= len(join_keys):
        return df
    context = baseline_context[context_columns].drop_duplicates(subset=join_keys)
    return df.merge(context, on=join_keys, how="left", suffixes=("", "_baseline"))


def selected_diagnostic_model(metadata: dict[str, Any], training_summary: dict[str, Any]) -> str:
    selected = metadata.get("selected_model") or training_summary.get("selected_model")
    if not selected:
        raise ValueError("Could not identify selected diagnostic model")
    return str(selected)


def slice_calibration(
    frames: list[pd.DataFrame],
    *,
    min_rows: int = 1,
    slice_columns: tuple[str, ...] = SLICE_COLUMNS,
) -> tuple[pd.DataFrame, list[str]]:
    available_columns = sorted(
        {column for column in slice_columns if any(column in frame.columns for frame in frames)}
    )
    missing_columns = [column for column in slice_columns if column not in available_columns]
    rows = []
    for frame in frames:
        model_version = str(frame[MODEL_COLUMN].iloc[0])
        for column in available_columns:
            if column not in frame.columns:
                continue
            values = frame[column].fillna("unknown").astype(str)
            for value, subset in frame.groupby(values, dropna=False):
                if len(subset) < min_rows:
                    continue
                metrics = safe_metric_summary(subset, model_version=model_version)
                warning = sample_size_warning(len(subset))
                rows.append(
                    {
                        "model_version": model_version,
                        "slice_column": column,
                        "slice_value": str(value),
                        "rows": metrics["row_count"],
                        "goals": metrics["goal_count"],
                        "goal_rate": metrics["goal_rate"],
                        "mean_predicted_probability": metrics["mean_predicted_probability"],
                        "calibration_error": metrics["goal_rate"]
                        - metrics["mean_predicted_probability"],
                        "brier": metrics["brier"],
                        "log_loss": metrics["log_loss"],
                        "roc_auc": metrics["roc_auc"],
                        "roc_auc_status": metrics["roc_auc_status"],
                        "sample_size_warning": warning,
                        "interpretation": _slice_interpretation(metrics, warning),
                    }
                )
    columns = [
        "model_version",
        "slice_column",
        "slice_value",
        "rows",
        "goals",
        "goal_rate",
        "mean_predicted_probability",
        "calibration_error",
        "brier",
        "log_loss",
        "roc_auc",
        "roc_auc_status",
        "sample_size_warning",
        "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns), missing_columns


def sample_size_warning(rows: int) -> str:
    if rows < 30:
        return "very_sparse"
    if rows < 100:
        return "sparse"
    return "sufficient"


def _slice_interpretation(metrics: dict[str, Any], warning: str) -> str:
    error = abs(metrics["goal_rate"] - metrics["mean_predicted_probability"])
    if warning != "sufficient":
        return f"{warning} sample; interpret directionally"
    if error <= 0.03:
        return "well_calibrated"
    if error <= 0.08:
        return "monitor_calibration"
    return "large_calibration_gap"


def fold_stability(fold_metrics: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if fold_metrics.empty:
        return pd.DataFrame(
            columns=[
                "model_candidate",
                "selected_model",
                "fold_count",
                "brier_mean",
                "brier_std",
                "log_loss_mean",
                "log_loss_std",
                "roc_auc_mean",
                "roc_auc_std",
                "fold_stability_status",
            ]
        )
    summary = fold_metrics.groupby("model_candidate", as_index=False).agg(
        fold_count=("fold", "nunique"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
    )
    summary["selected_model"] = summary["model_candidate"] == selected_model
    summary["fold_stability_status"] = summary.apply(_fold_status, axis=1)
    return summary[
        [
            "model_candidate",
            "selected_model",
            "fold_count",
            "brier_mean",
            "brier_std",
            "log_loss_mean",
            "log_loss_std",
            "roc_auc_mean",
            "roc_auc_std",
            "fold_stability_status",
        ]
    ]


def _fold_status(row: pd.Series) -> str:
    log_std = float(row.get("log_loss_std") or 0.0)
    brier_std = float(row.get("brier_std") or 0.0)
    auc_std = float(row.get("roc_auc_std") or 0.0)
    if log_std > 0.08 or brier_std > 0.025 or auc_std > 0.08:
        return "unstable"
    if log_std > 0.04 or brier_std > 0.0125 or auc_std > 0.04:
        return "moderate_variance"
    return "stable"


def leakage_status(
    excluded_columns: pd.DataFrame,
    resolved_features: dict[str, Any],
    feature_group_summary: pd.DataFrame,
    *,
    missing_governance_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    missing_governance_artifacts = missing_governance_artifacts or []
    if missing_governance_artifacts:
        return {
            "status": "unknown",
            "governance_status": "unknown",
            "leakage_status": "unknown",
            "missing_governance_artifacts": missing_governance_artifacts,
            "leakage_or_reference_features_used": [],
            "source_available": {},
            "synthetic_default_excluded": {},
            "synthetic_default_excluded_features": [],
        }
    excluded = set()
    if not excluded_columns.empty and "column" in excluded_columns.columns:
        excluded = set(excluded_columns["column"].dropna().astype(str))
    training_features = set()
    training_payload = resolved_features.get("training_features", {})
    if isinstance(training_payload, dict):
        for values in training_payload.values():
            training_features.update(str(value) for value in values)
    leakage_used = sorted(excluded.intersection(training_features))
    synthetic_excluded: list[str] = []
    if not feature_group_summary.empty and "excluded_reason" in feature_group_summary.columns:
        mask = feature_group_summary["excluded_reason"].fillna("") != ""
        synthetic_excluded = (
            feature_group_summary.loc[mask, "feature"].dropna().astype(str).tolist()
        )
    status = "failed" if leakage_used else "passed"
    return {
        "status": status,
        "governance_status": status,
        "leakage_status": status,
        "missing_governance_artifacts": [],
        "leakage_or_reference_features_used": leakage_used,
        "source_available": resolved_features.get("source_available", {}),
        "synthetic_default_excluded": resolved_features.get("synthetic_default_excluded", {}),
        "synthetic_default_excluded_features": synthetic_excluded,
    }


def promotion_recommendation(
    baseline_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
    *,
    selected_fold_status: str,
    leakage: dict[str, Any],
    slice_calibration_df: pd.DataFrame,
) -> dict[str, Any]:
    reasons: list[str] = []
    limitations: list[str] = []
    leakage_failed = leakage.get("status") != "passed"
    missing_governance = leakage.get("missing_governance_artifacts") or []
    invalid_probability = (
        diagnostic_metrics["probability_null_count"] > 0
        or diagnostic_metrics["probability_min"] < 0.0
        or diagnostic_metrics["probability_max"] > 1.0
    )
    severe_slice_gap = _has_severe_slice_gap(slice_calibration_df)
    improves_log_loss = diagnostic_metrics["log_loss"] <= baseline_metrics["log_loss"]
    improves_brier = diagnostic_metrics["brier"] <= baseline_metrics["brier"]
    ece_ok = diagnostic_metrics["expected_calibration_error"] <= (
        baseline_metrics["expected_calibration_error"] + 0.005
    )

    if leakage_failed:
        if missing_governance:
            reasons.append(
                "Required feature-governance artifacts are missing: "
                + ", ".join(str(path) for path in missing_governance)
            )
        else:
            reasons.append("Forbidden leakage/reference features were used.")
    if invalid_probability:
        reasons.append("Diagnostic probabilities contain invalid or missing values.")
    if selected_fold_status == "unstable":
        reasons.append("Selected diagnostic model has unstable fold metrics.")
    if severe_slice_gap:
        reasons.append("Slice calibration shows major degradation on sufficient samples.")

    if leakage_failed or invalid_probability or selected_fold_status == "unstable":
        recommendation = "do_not_promote"
    elif improves_log_loss and improves_brier and ece_ok and not severe_slice_gap:
        recommendation = "promote"
        reasons.append("Diagnostic model matches or improves log loss, Brier, and calibration.")
    elif (
        not severe_slice_gap
        and selected_fold_status in {"stable", "moderate_variance"}
        and (improves_log_loss or improves_brier)
    ):
        recommendation = "provisional_promote"
        reasons.append(
            "Diagnostic model has mixed metric results but acceptable governance and stability."
        )
    else:
        recommendation = "needs_revision"
        reasons.append("Diagnostic model is worse on key metrics or slice calibration.")

    if not improves_log_loss:
        limitations.append("Diagnostic log loss is worse than baseline.")
    if not improves_brier:
        limitations.append("Diagnostic Brier score is worse than baseline.")
    if not ece_ok:
        limitations.append("Diagnostic calibration error is worse than baseline beyond tolerance.")
    if severe_slice_gap:
        limitations.append(
            "At least one sufficient slice has an absolute calibration gap above 0.12."
        )
    if missing_governance:
        limitations.append(
            "Feature-governance artifacts are required promotion gates and were missing."
        )

    return {
        "recommendation": recommendation,
        "reasons": reasons,
        "baseline_metrics": baseline_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "known_limitations": limitations,
        "next_issue": "#57 if promoted/provisional, otherwise revise #55",
    }


def _has_severe_slice_gap(slice_calibration_df: pd.DataFrame) -> bool:
    if slice_calibration_df.empty:
        return False
    diagnostic = slice_calibration_df[
        slice_calibration_df["model_version"].astype(str).str.startswith("diagnostic_v1:")
    ]
    sufficient = diagnostic.loc[diagnostic["sample_size_warning"] == "sufficient"]
    if sufficient.empty:
        return False
    return bool(sufficient["calibration_error"].abs().gt(0.12).any())


def validate_diagnostic_cxg(
    paths: ValidationPaths,
    *,
    n_bins: int = 10,
    min_slice_rows: int = 1,
    make_plots: bool = True,
) -> dict[str, Path]:
    """Run diagnostic CxG validation and write artifacts."""

    baseline_raw = pd.read_parquet(paths.baseline_predictions)
    diagnostic_raw = pd.read_parquet(paths.diagnostic_predictions)
    selected_metadata = _read_json(paths.diagnostic_selected_metadata)
    training_summary = _read_json(paths.diagnostic_training_summary)
    selected_model = selected_diagnostic_model(selected_metadata, training_summary)

    baseline = normalize_baseline_predictions(baseline_raw)
    diagnostic = normalize_diagnostic_predictions(
        diagnostic_raw,
        selected_model=selected_model,
        baseline_context=baseline_raw,
    )
    baseline_metrics = safe_metric_summary(baseline, model_version="baseline")
    diagnostic_model_version = f"diagnostic_v1:{selected_model}"
    diagnostic_metrics = safe_metric_summary(diagnostic, model_version=diagnostic_model_version)

    calibration = pd.concat(
        [
            calibration_bins(baseline, model_version="baseline", n_bins=n_bins),
            calibration_bins(
                diagnostic,
                model_version=diagnostic_model_version,
                n_bins=n_bins,
            ),
        ],
        ignore_index=True,
    )
    ece_by_model = calibration.groupby("model_version").apply(
        expected_calibration_error, include_groups=False
    )
    baseline_metrics["expected_calibration_error"] = float(ece_by_model.loc["baseline"])
    diagnostic_metrics["expected_calibration_error"] = float(
        ece_by_model.loc[diagnostic_model_version]
    )

    comparison = pd.DataFrame([baseline_metrics, diagnostic_metrics])
    fold_metrics = pd.read_csv(paths.diagnostic_fold_metrics)
    folds = fold_stability(fold_metrics, selected_model)
    selected_fold = folds.loc[folds["selected_model"]]
    selected_fold_status = (
        str(selected_fold.iloc[0]["fold_stability_status"])
        if not selected_fold.empty
        else "unstable"
    )
    slices, missing_slice_columns = slice_calibration(
        [baseline, diagnostic],
        min_rows=min_slice_rows,
    )
    governance = _load_governance_artifacts(paths)
    leakage = leakage_status(
        governance["excluded_columns"],
        governance["resolved_features"],
        governance["feature_group_summary"],
        missing_governance_artifacts=governance["missing_governance_artifacts"],
    )
    recommendation = promotion_recommendation(
        baseline_metrics,
        diagnostic_metrics,
        selected_fold_status=selected_fold_status,
        leakage=leakage,
        slice_calibration_df=slices,
    )

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = paths.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "validation_summary": paths.output_dir / "validation_summary.json",
        "model_comparison_validation": paths.output_dir / "model_comparison_validation.csv",
        "fold_stability": paths.output_dir / "fold_stability.csv",
        "calibration_bins": paths.output_dir / "calibration_bins.csv",
        "slice_calibration": paths.output_dir / "slice_calibration.csv",
        "promotion_recommendation": paths.output_dir / "promotion_recommendation.json",
        "validation_report": paths.output_dir / "validation_report.md",
        "calibration_curve": plots_dir / "calibration_curve.png",
        "predicted_vs_actual_by_slice": plots_dir / "predicted_vs_actual_by_slice.png",
    }
    comparison.to_csv(output_paths["model_comparison_validation"], index=False)
    folds.to_csv(output_paths["fold_stability"], index=False)
    calibration.to_csv(output_paths["calibration_bins"], index=False)
    slices.to_csv(output_paths["slice_calibration"], index=False)
    output_paths["promotion_recommendation"].write_text(
        json.dumps(_json_safe(recommendation), indent=2),
        encoding="utf-8",
    )
    summary = _validation_summary(
        paths,
        selected_model,
        baseline_metrics,
        diagnostic_metrics,
        missing_slice_columns,
        leakage,
        recommendation,
        output_paths,
    )
    output_paths["validation_summary"].write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    output_paths["validation_report"].write_text(
        _validation_report(
            paths,
            selected_model,
            comparison,
            folds,
            slices,
            calibration,
            leakage,
            recommendation,
            missing_slice_columns,
        ),
        encoding="utf-8",
    )
    if make_plots:
        _write_plots(calibration, slices, output_paths)
    return output_paths


def _load_governance_artifacts(paths: ValidationPaths) -> dict[str, Any]:
    required = {
        "resolved_features": paths.diagnostic_resolved_features,
        "excluded_columns": paths.diagnostic_excluded_columns,
        "feature_group_summary": paths.diagnostic_feature_group_summary,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        return {
            "resolved_features": {},
            "excluded_columns": pd.DataFrame(),
            "feature_group_summary": pd.DataFrame(),
            "missing_governance_artifacts": missing,
        }
    return {
        "resolved_features": _read_json(paths.diagnostic_resolved_features),
        "excluded_columns": pd.read_csv(paths.diagnostic_excluded_columns),
        "feature_group_summary": pd.read_csv(paths.diagnostic_feature_group_summary),
        "missing_governance_artifacts": [],
    }


def _validation_summary(
    paths: ValidationPaths,
    selected_model: str,
    baseline_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
    missing_slice_columns: list[str],
    leakage: dict[str, Any],
    recommendation: dict[str, Any],
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "selected_diagnostic_model": selected_model,
        "inputs": {
            "baseline_predictions": str(paths.baseline_predictions),
            "diagnostic_predictions": str(paths.diagnostic_predictions),
            "selected_model_metadata": str(paths.diagnostic_selected_metadata),
            "feature_contract": str(paths.diagnostic_feature_contract),
            "resolved_features": str(paths.diagnostic_resolved_features),
            "feature_group_summary": str(paths.diagnostic_feature_group_summary),
            "excluded_columns": str(paths.diagnostic_excluded_columns),
            "analysis_report": str(paths.analysis_report),
            "analysis_slice_stability_dir": str(paths.analysis_slice_stability_dir),
            "analysis_leakage_dir": str(paths.analysis_leakage_dir),
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "baseline_metrics": baseline_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "missing_slice_columns": missing_slice_columns,
        "governance_status": leakage.get("governance_status", leakage.get("status", "unknown")),
        "missing_governance_artifacts": leakage.get("missing_governance_artifacts", []),
        "leakage_status": leakage.get("leakage_status", leakage.get("status", "unknown")),
        "leakage_feature_governance": leakage,
        "promotion_recommendation": recommendation["recommendation"],
    }


def _validation_report(
    paths: ValidationPaths,
    selected_model: str,
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    slices: pd.DataFrame,
    calibration: pd.DataFrame,
    leakage: dict[str, Any],
    recommendation: dict[str, Any],
    missing_slice_columns: list[str],
) -> str:
    comparison_lines = [
        f"- `{row.model_version}`: log loss {row.log_loss:.4f}, Brier {row.brier:.4f}, "
        f"ECE {row.expected_calibration_error:.4f}, ROC AUC "
        f"{row.roc_auc if pd.notna(row.roc_auc) else 'skipped'}"
        for row in comparison.itertuples()
    ]
    selected_fold = folds.loc[folds["selected_model"]]
    fold_status = (
        str(selected_fold.iloc[0]["fold_stability_status"])
        if not selected_fold.empty
        else "not available"
    )
    slice_count = 0 if slices.empty else len(slices)
    non_empty_bins = int((calibration["rows"] > 0).sum()) if not calibration.empty else 0
    source_available = leakage.get("source_available", {})
    synthetic_excluded = leakage.get("synthetic_default_excluded", {})
    leakage_used = leakage.get("leakage_or_reference_features_used", [])
    missing_governance = leakage.get("missing_governance_artifacts", [])
    return "\n".join(
        [
            "# Diagnostic CxG Validation",
            "",
            "## 1. Purpose",
            "",
            "This validation compares the existing baseline CxG model with the selected "
            "diagnostic-informed CxG model from `diagnostic_v1`. It evaluates probability "
            "quality, calibration, fold stability, slice behavior, and feature governance "
            "before any promotion work in #57.",
            "",
            "## 2. Inputs",
            "",
            f"- Baseline predictions: `{paths.baseline_predictions}`",
            f"- Diagnostic cross-validated predictions: `{paths.diagnostic_predictions}`",
            f"- Selected diagnostic metadata: `{paths.diagnostic_selected_metadata}`",
            f"- Diagnostic analysis report: `{paths.analysis_report}`",
            "",
            "## 3. Baseline vs Diagnostic Model Comparison",
            "",
            *comparison_lines,
            "",
            "## 4. Calibration Analysis",
            "",
            f"Calibration was evaluated with {non_empty_bins} non-empty model/bin rows. "
            "Expected calibration error is weighted by bin support, so larger bins carry "
            "more influence than sparse bins.",
            "",
            "## 5. Fold Stability",
            "",
            f"The selected diagnostic model `{selected_model}` has fold stability status "
            f"`{fold_status}`. Status thresholds are: stable when fold standard deviations "
            "are low, moderate_variance for watch-list variance, and unstable for large "
            "log-loss, Brier, or ROC AUC spread.",
            "",
            "## 6. Slice-Level Calibration",
            "",
            f"Slice validation produced {slice_count} model/slice rows. Missing optional "
            f"slice columns: {', '.join(missing_slice_columns) or 'none'}. Sparse slices "
            "are labelled so they are interpreted directionally rather than as hard "
            "promotion evidence.",
            "",
            "## 7. Leakage / Feature Governance Check",
            "",
            "The validation uses the #55 diagnostics artifacts rather than recomputing "
            "training eligibility. Governance status is "
            f"`{leakage.get('governance_status', leakage.get('status', 'unknown'))}`. "
            f"Missing governance artifacts: {', '.join(missing_governance) or 'none'}. "
            "Source-available features were:",
            "",
            f"- Numeric: {', '.join(source_available.get('numeric', [])) or 'none'}",
            f"- Binary: {', '.join(source_available.get('binary', [])) or 'none'}",
            f"- Categorical: {', '.join(source_available.get('categorical', [])) or 'none'}",
            "",
            "Synthetic default features excluded from training were:",
            "",
            f"- Numeric: {', '.join(synthetic_excluded.get('numeric', [])) or 'none'}",
            f"- Binary: {', '.join(synthetic_excluded.get('binary', [])) or 'none'}",
            f"- Categorical: {', '.join(synthetic_excluded.get('categorical', [])) or 'none'}",
            "",
            f"Leakage/reference features used: {', '.join(leakage_used) or 'none'}. "
            "This matters because calibration gains are only trustworthy when the model "
            "uses pre-shot, source-governed features rather than post-shot or reference "
            "signals.",
            "",
            "## 8. Model Selection Trade-Offs",
            "",
            "The diagnostic model is assessed on log loss, Brier score, calibration error, "
            "fold stability, and slice calibration. ROC AUC is included as context but is "
            "not sufficient for promotion because it measures rank ordering rather than "
            "probability quality.",
            "",
            "## 9. Promotion Recommendation",
            "",
            f"Recommendation: `{recommendation['recommendation']}`.",
            "",
            *[f"- {reason}" for reason in recommendation.get("reasons", [])],
            "",
            "## 10. Remaining Work for #57",
            "",
            "If this model is promoted or provisionally promoted, #57 should handle final "
            "prediction generation, result promotion, and any player/team reporting. If "
            "the recommendation is revision, #55 feature or candidate decisions should be "
            "updated before promotion.",
            "",
        ]
    )


def _write_plots(
    calibration: pd.DataFrame,
    slices: pd.DataFrame,
    output_paths: dict[str, Path],
) -> None:
    plt.figure(figsize=(7, 5))
    for model_version, subset in calibration.loc[calibration["rows"] > 0].groupby("model_version"):
        plt.plot(
            subset["mean_predicted_probability"],
            subset["actual_goal_rate"],
            marker="o",
            label=str(model_version),
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual goal rate")
    plt.title("Does diagnostic CxG improve calibration?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_paths["calibration_curve"])
    plt.close()

    plt.figure(figsize=(9, 5))
    if slices.empty:
        plt.text(0.5, 0.5, "No slice rows available", ha="center", va="center")
        plt.axis("off")
    else:
        sufficient = slices.loc[slices["sample_size_warning"] != "very_sparse"].copy()
        if sufficient.empty:
            sufficient = slices.copy()
        sufficient["label"] = (
            sufficient["slice_column"].astype(str) + "=" + sufficient["slice_value"].astype(str)
        )
        top = sufficient.reindex(sufficient["rows"].sort_values(ascending=False).index).head(12)
        x = np.arange(len(top))
        plt.scatter(x, top["goal_rate"], label="Actual", color="#1f77b4")
        plt.scatter(
            x,
            top["mean_predicted_probability"],
            label="Predicted",
            color="#ff7f0e",
        )
        plt.xticks(x, top["label"], rotation=45, ha="right")
        plt.ylabel("Probability")
        plt.title("Predicted vs actual goal rate by slice")
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_paths["predicted_vs_actual_by_slice"])
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate diagnostic CxG against baseline")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--min-slice-rows", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = validate_diagnostic_cxg(
        ValidationPaths.from_roots(
            baseline_dir=args.baseline_dir,
            diagnostic_dir=args.diagnostic_dir,
            analysis_dir=args.analysis_dir,
            output_dir=args.output_dir,
        ),
        n_bins=args.bins,
        min_slice_rows=args.min_slice_rows,
        make_plots=not args.skip_plots,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
