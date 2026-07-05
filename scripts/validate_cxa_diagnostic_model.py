#!/usr/bin/env python
"""Validate diagnostic CxA against the existing baseline CxA outputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "shot_created"
DIAGNOSTIC_PREDICTION_COLUMN = "predicted_shot_created_probability"
BASELINE_PREDICTION_CANDIDATES = (
    "predicted_shot_created_probability",
    "predicted_chance_action",
    "predicted_cxa",
    "baseline_probability",
)
SLICE_COLUMNS = (
    "action_type",
    "is_pass",
    "is_carry",
    "is_progressive",
    "enters_final_third",
    "enters_penalty_area",
    "start_third",
    "end_third",
    "team_id",
    "player_id",
)
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.50)
TOP_CUTS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class CxAValidationPaths:
    diagnostic_predictions: Path
    selected_model_metadata: Path
    model_comparison: Path
    feature_contract: Path
    baseline_predictions: Path
    baseline_metrics: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        diagnostic_dir: Path = Path("outputs/modeling/cxa/diagnostic_v1"),
        baseline_dir: Path = Path("outputs/modeling/cxa"),
        output_dir: Path = Path("outputs/validation/cxa/diagnostic_v1"),
    ) -> "CxAValidationPaths":
        return cls(
            diagnostic_predictions=diagnostic_dir
            / "predictions"
            / "cross_validated_predictions.parquet",
            selected_model_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            model_comparison=diagnostic_dir / "reports" / "model_comparison.csv",
            feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            baseline_predictions=baseline_dir / "predictions" / "action_predictions.parquet",
            baseline_metrics=baseline_dir / "reports" / "metrics.json",
            output_dir=output_dir,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def detect_baseline_prediction_column(df: pd.DataFrame) -> str:
    for column in BASELINE_PREDICTION_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Could not find a baseline CxA prediction column. Expected one of "
        f"{BASELINE_PREDICTION_CANDIDATES}."
    )


def selected_diagnostic_model(metadata: dict[str, Any]) -> str:
    selected = (
        metadata.get("selected_model_candidate")
        or metadata.get("selected_model")
        or metadata.get("selected_candidate")
    )
    if not selected:
        raise ValueError("selected_model_metadata.json does not identify the selected CxA model")
    return str(selected)


def choose_join_key(diagnostic: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    if "action_id" in diagnostic.columns and "action_id" in baseline.columns:
        return ["action_id"]
    if "event_id" in diagnostic.columns and "event_id" in baseline.columns:
        if "match_id" in diagnostic.columns and "match_id" in baseline.columns:
            return ["match_id", "event_id"]
        return ["event_id"]
    raise ValueError("No safe join key found for CxA validation predictions")


def _clip_probabilities(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").clip(1e-15, 1 - 1e-15)


def expected_calibration_error(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    n_bins: int = 10,
) -> float:
    frame = pd.DataFrame({"target": y_true.astype(int), "probability": probabilities})
    frame = frame.dropna(subset=["target", "probability"])
    if frame.empty:
        return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(frame["probability"].clip(0.0, 1.0), bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    total = len(frame)
    ece = 0.0
    for bin_id in range(n_bins):
        subset = frame.loc[bin_ids == bin_id]
        if subset.empty:
            continue
        ece += (len(subset) / total) * abs(
            float(subset["probability"].mean()) - float(subset["target"].mean())
        )
    return float(ece)


def top_k_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    top_fraction: float,
) -> dict[str, Any]:
    frame = pd.DataFrame({"target": y_true.astype(int), "probability": probabilities}).dropna()
    if frame.empty:
        return {"precision": np.nan, "recall": np.nan, "positive_count_captured": 0}
    selected_count = max(1, int(np.ceil(len(frame) * top_fraction)))
    selected = frame.sort_values("probability", ascending=False).head(selected_count)
    positives = int(frame["target"].sum())
    captured = int(selected["target"].sum())
    return {
        "precision": float(captured / selected_count) if selected_count else np.nan,
        "recall": float(captured / positives) if positives else np.nan,
        "positive_count_captured": captured,
    }


def metric_summary(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    model_name: str,
) -> dict[str, Any]:
    required = {TARGET_COLUMN, prediction_column}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{model_name} validation frame is missing required columns: {missing}")
    eval_df = df.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    if eval_df.empty:
        raise ValueError(f"{model_name} has no valid target/prediction rows")
    y_true = eval_df[TARGET_COLUMN].astype(int)
    probabilities_raw = pd.to_numeric(eval_df[prediction_column], errors="coerce")
    probabilities = _clip_probabilities(probabilities_raw)
    both_classes = y_true.nunique() == 2
    threshold_labels = (probabilities >= 0.5).astype(int)
    top_1 = top_k_metrics(y_true, probabilities, top_fraction=0.01)
    top_5 = top_k_metrics(y_true, probabilities, top_fraction=0.05)
    summary: dict[str, Any] = {
        "model": model_name,
        "row_count": int(len(eval_df)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(probabilities.mean()),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "roc_auc": np.nan,
        "average_precision": np.nan,
        "expected_calibration_error": expected_calibration_error(y_true, probabilities),
        "calibration_error": abs(float(probabilities.mean()) - float(y_true.mean())),
        "precision_at_0_5": float(precision_score(y_true, threshold_labels, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, threshold_labels, zero_division=0)),
        "precision_at_top_1pct": top_1["precision"],
        "recall_at_top_1pct": top_1["recall"],
        "precision_at_top_5pct": top_5["precision"],
        "recall_at_top_5pct": top_5["recall"],
    }
    if both_classes:
        summary["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        summary["average_precision"] = float(average_precision_score(y_true, probabilities))
    return summary


def calibration_summary(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    model_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = df.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    probabilities = pd.to_numeric(frame[prediction_column], errors="coerce").clip(0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    rows = []
    for bin_id in range(n_bins):
        subset = frame.loc[bin_ids == bin_id]
        subset_probs = probabilities.loc[subset.index]
        if subset.empty:
            rows.append(
                {
                    "model": model_name,
                    "bin": bin_id + 1,
                    "row_count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue
        mean_pred = float(subset_probs.mean())
        observed = float(subset[TARGET_COLUMN].astype(int).mean())
        rows.append(
            {
                "model": model_name,
                "bin": bin_id + 1,
                "row_count": int(len(subset)),
                "mean_predicted_probability": mean_pred,
                "observed_positive_rate": observed,
                "absolute_gap": abs(mean_pred - observed),
            }
        )
    return pd.DataFrame(rows)


def threshold_summary(
    df: pd.DataFrame,
    *,
    prediction_column: str,
    model_name: str,
) -> pd.DataFrame:
    frame = df.dropna(subset=[TARGET_COLUMN, prediction_column]).copy()
    y_true = frame[TARGET_COLUMN].astype(int)
    probabilities = pd.to_numeric(frame[prediction_column], errors="coerce").clip(0.0, 1.0)
    positives = int(y_true.sum())
    rows = []
    for threshold in THRESHOLDS:
        mask = probabilities >= threshold
        selected_count = int(mask.sum())
        captured = int(y_true.loc[mask].sum()) if selected_count else 0
        rows.append(
            {
                "model": model_name,
                "threshold_or_cut": f"{threshold:.2f}",
                "selected_count": selected_count,
                "precision": float(captured / selected_count) if selected_count else np.nan,
                "recall": float(captured / positives) if positives else np.nan,
                "positive_count_captured": captured,
                "share_of_actions_selected": (
                    float(selected_count / len(frame)) if len(frame) else 0.0
                ),
            }
        )
    sorted_frame = frame.assign(_probability=probabilities).sort_values(
        "_probability", ascending=False
    )
    for cut in TOP_CUTS:
        selected_count = max(1, int(np.ceil(len(sorted_frame) * cut))) if len(sorted_frame) else 0
        selected = sorted_frame.head(selected_count)
        captured = int(selected[TARGET_COLUMN].astype(int).sum()) if selected_count else 0
        rows.append(
            {
                "model": model_name,
                "threshold_or_cut": f"top_{int(cut * 100)}pct",
                "selected_count": selected_count,
                "precision": float(captured / selected_count) if selected_count else np.nan,
                "recall": float(captured / positives) if positives else np.nan,
                "positive_count_captured": captured,
                "share_of_actions_selected": (
                    float(selected_count / len(frame)) if len(frame) else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def join_predictions(
    diagnostic: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    selected_model: str,
    baseline_prediction_column: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    diagnostic_selected = diagnostic.loc[diagnostic["model_candidate"] == selected_model].copy()
    join_key = choose_join_key(diagnostic_selected, baseline)
    baseline_columns = list(dict.fromkeys(join_key + [TARGET_COLUMN, baseline_prediction_column]))
    for column in SLICE_COLUMNS:
        if column in baseline.columns and column not in baseline_columns:
            baseline_columns.append(column)
    joined = diagnostic_selected.merge(
        baseline[baseline_columns],
        on=join_key,
        how="left",
        suffixes=("_diagnostic", "_baseline"),
    )
    if f"{TARGET_COLUMN}_diagnostic" in joined.columns:
        joined[TARGET_COLUMN] = joined[f"{TARGET_COLUMN}_diagnostic"]
    elif TARGET_COLUMN in diagnostic_selected.columns:
        joined[TARGET_COLUMN] = joined[TARGET_COLUMN]
    if f"{DIAGNOSTIC_PREDICTION_COLUMN}_diagnostic" in joined.columns:
        diagnostic_joined_column = f"{DIAGNOSTIC_PREDICTION_COLUMN}_diagnostic"
    else:
        diagnostic_joined_column = DIAGNOSTIC_PREDICTION_COLUMN
    if f"{baseline_prediction_column}_baseline" in joined.columns:
        baseline_joined_column = f"{baseline_prediction_column}_baseline"
    else:
        baseline_joined_column = baseline_prediction_column
    joined = joined.rename(
        columns={
            diagnostic_joined_column: "diagnostic_probability",
            baseline_joined_column: "baseline_probability",
        }
    )
    if f"{TARGET_COLUMN}_baseline" in joined.columns:
        target_mismatch = int(
            (
                joined[f"{TARGET_COLUMN}_baseline"].notna()
                & (
                    joined[TARGET_COLUMN].astype(float)
                    != joined[f"{TARGET_COLUMN}_baseline"].astype(float)
                )
            ).sum()
        )
    else:
        target_mismatch = 0
    join_quality = {
        "diagnostic_row_count": int(len(diagnostic_selected)),
        "baseline_row_count": int(len(baseline)),
        "joined_row_count": int(joined["baseline_probability"].notna().sum()),
        "baseline_join_rate": (
            float(joined["baseline_probability"].notna().mean()) if len(joined) else 0.0
        ),
        "duplicate_action_id_count": (
            int(diagnostic_selected["action_id"].duplicated().sum())
            if "action_id" in diagnostic_selected.columns
            else 0
        ),
        "missing_action_id_count": (
            int(diagnostic_selected["action_id"].isna().sum())
            if "action_id" in diagnostic_selected.columns
            else int(len(diagnostic_selected))
        ),
        "target_mismatch_count": target_mismatch,
        "diagnostic_prediction_null_count": int(joined["diagnostic_probability"].isna().sum()),
        "baseline_prediction_null_count": int(joined["baseline_probability"].isna().sum()),
        "diagnostic_prediction_outside_0_1_count": int(
            (~pd.to_numeric(joined["diagnostic_probability"], errors="coerce").between(0, 1)).sum()
        ),
        "baseline_prediction_outside_0_1_count": int(
            (
                joined["baseline_probability"].notna()
                & ~pd.to_numeric(joined["baseline_probability"], errors="coerce").between(0, 1)
            ).sum()
        ),
    }
    return joined, join_key, join_quality


def quality_checks(join_quality: dict[str, Any], *, candidate_matches: bool) -> pd.DataFrame:
    checks = [
        (
            "joined_row_count",
            join_quality["joined_row_count"],
            join_quality["joined_row_count"] > 0,
        ),
        (
            "baseline_join_rate",
            join_quality["baseline_join_rate"],
            join_quality["baseline_join_rate"] >= 0.98,
        ),
        (
            "diagnostic_prediction_nulls",
            join_quality["diagnostic_prediction_null_count"],
            join_quality["diagnostic_prediction_null_count"] == 0,
        ),
        (
            "baseline_prediction_nulls",
            join_quality["baseline_prediction_null_count"],
            join_quality["baseline_prediction_null_count"] == 0,
        ),
        (
            "diagnostic_probability_bounds",
            join_quality["diagnostic_prediction_outside_0_1_count"],
            join_quality["diagnostic_prediction_outside_0_1_count"] == 0,
        ),
        (
            "baseline_probability_bounds",
            join_quality["baseline_prediction_outside_0_1_count"],
            join_quality["baseline_prediction_outside_0_1_count"] == 0,
        ),
        (
            "target_mismatch_count",
            join_quality["target_mismatch_count"],
            join_quality["target_mismatch_count"] == 0,
        ),
        ("selected_candidate_match", candidate_matches, candidate_matches),
    ]
    rows = []
    for name, value, passed in checks:
        rows.append(
            {
                "check_name": name,
                "value": value,
                "status": "passed" if passed else "failed",
                "severity": "info" if passed else "blocker",
                "notes": "" if passed else f"{name} failed validation quality requirements.",
            }
        )
    return pd.DataFrame(rows)


def slice_summary(
    joined: pd.DataFrame,
    *,
    min_rows: int = 500,
) -> pd.DataFrame:
    rows = []
    for column in SLICE_COLUMNS:
        if column not in joined.columns:
            continue
        for value, subset in joined.groupby(column, dropna=False):
            if len(subset) < min_rows:
                continue
            if subset[TARGET_COLUMN].nunique() < 2:
                baseline_ap = np.nan
                diagnostic_ap = np.nan
            else:
                baseline_ap = average_precision_score(
                    subset[TARGET_COLUMN].astype(int),
                    _clip_probabilities(subset["baseline_probability"]),
                )
                diagnostic_ap = average_precision_score(
                    subset[TARGET_COLUMN].astype(int),
                    _clip_probabilities(subset["diagnostic_probability"]),
                )
            baseline_log_loss = log_loss(
                subset[TARGET_COLUMN].astype(int),
                _clip_probabilities(subset["baseline_probability"]),
                labels=[0, 1],
            )
            diagnostic_log_loss = log_loss(
                subset[TARGET_COLUMN].astype(int),
                _clip_probabilities(subset["diagnostic_probability"]),
                labels=[0, 1],
            )
            baseline_brier = brier_score_loss(
                subset[TARGET_COLUMN].astype(int),
                _clip_probabilities(subset["baseline_probability"]),
            )
            diagnostic_brier = brier_score_loss(
                subset[TARGET_COLUMN].astype(int),
                _clip_probabilities(subset["diagnostic_probability"]),
            )
            rows.append(
                {
                    "slice_name": column,
                    "slice_value": value,
                    "row_count": int(len(subset)),
                    "positive_rate": float(subset[TARGET_COLUMN].astype(int).mean()),
                    "baseline_log_loss": float(baseline_log_loss),
                    "diagnostic_log_loss": float(diagnostic_log_loss),
                    "baseline_brier": float(baseline_brier),
                    "diagnostic_brier": float(diagnostic_brier),
                    "baseline_average_precision": (
                        float(baseline_ap) if not pd.isna(baseline_ap) else np.nan
                    ),
                    "diagnostic_average_precision": (
                        float(diagnostic_ap) if not pd.isna(diagnostic_ap) else np.nan
                    ),
                    "diagnostic_minus_baseline_log_loss": float(
                        diagnostic_log_loss - baseline_log_loss
                    ),
                    "diagnostic_minus_baseline_brier": float(diagnostic_brier - baseline_brier),
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def promotion_recommendation(
    *,
    baseline_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
    checks: pd.DataFrame,
    slices: pd.DataFrame,
) -> dict[str, Any]:
    failed_blockers = checks.loc[
        (checks["severity"] == "blocker") & (checks["status"] == "failed"), "check_name"
    ].tolist()
    if failed_blockers:
        return {
            "recommendation": "blocked",
            "reasons": [f"Hard blocker failed: {name}" for name in failed_blockers],
            "known_limitations": [],
        }

    log_loss_delta = diagnostic_metrics["log_loss"] - baseline_metrics["log_loss"]
    brier_delta = diagnostic_metrics["brier"] - baseline_metrics["brier"]
    ap_delta = diagnostic_metrics["average_precision"] - baseline_metrics["average_precision"]
    ece_delta = (
        diagnostic_metrics["expected_calibration_error"]
        - baseline_metrics["expected_calibration_error"]
    )
    severe_slice_regressions = 0
    if not slices.empty and "diagnostic_minus_baseline_log_loss" in slices:
        severe_slice_regressions = int((slices["diagnostic_minus_baseline_log_loss"] > 0.03).sum())

    reasons = [
        f"log_loss_delta={log_loss_delta:.6f}",
        f"brier_delta={brier_delta:.6f}",
        f"average_precision_delta={ap_delta:.6f}",
        f"expected_calibration_error_delta={ece_delta:.6f}",
        f"severe_slice_regressions={severe_slice_regressions}",
    ]
    if log_loss_delta <= 1e-6 and brier_delta <= 1e-6 and ap_delta > 0 and ece_delta <= 0.01:
        recommendation = "promote"
    elif ap_delta > 0.005 and log_loss_delta <= 0.02 and brier_delta <= 0.01:
        recommendation = "provisional_promote"
    else:
        recommendation = "needs_revision"
    return {
        "recommendation": recommendation,
        "reasons": reasons,
        "known_limitations": (
            ["Some slice regressions should be reviewed."] if severe_slice_regressions else []
        ),
    }


def _metric_comparison_frame(
    baseline_metrics: dict[str, Any],
    diagnostic_metrics: dict[str, Any],
) -> pd.DataFrame:
    metrics = [
        "log_loss",
        "brier",
        "roc_auc",
        "average_precision",
        "positive_rate",
        "mean_predicted_probability",
        "expected_calibration_error",
        "calibration_error",
        "precision_at_0_5",
        "recall_at_0_5",
        "precision_at_top_1pct",
        "recall_at_top_1pct",
        "precision_at_top_5pct",
        "recall_at_top_5pct",
    ]
    rows = []
    for metric in metrics:
        baseline = baseline_metrics.get(metric)
        diagnostic = diagnostic_metrics.get(metric)
        rows.append(
            {
                "metric": metric,
                "baseline": baseline,
                "diagnostic": diagnostic,
                "diagnostic_minus_baseline": (
                    diagnostic - baseline
                    if baseline is not None and diagnostic is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _validation_report(
    *,
    selected_model: str,
    baseline_prediction_column: str,
    join_key: list[str],
    comparison: pd.DataFrame,
    checks: pd.DataFrame,
    recommendation: dict[str, Any],
) -> str:
    comparison_preview = comparison.to_string(index=False)
    failed_checks = checks.loc[checks["status"] == "failed", "check_name"].tolist()
    return "\n".join(
        [
            "# Diagnostic CxA Validation Report",
            "",
            "## Executive summary",
            f"- Selected diagnostic model: `{selected_model}`.",
            f"- Recommendation: `{recommendation['recommendation']}`.",
            "- This PR validates but does not promote result outputs.",
            "",
            "## Baseline vs diagnostic comparison",
            "```text",
            comparison_preview,
            "```",
            "",
            "## Join quality",
            f"- Join key: `{'+'.join(join_key)}`.",
            f"- Baseline prediction column: `{baseline_prediction_column}`.",
            "",
            "## Prediction quality checks",
            f"- Failed checks: {', '.join(failed_checks) if failed_checks else 'none'}.",
            "",
            "## Metrics comparison",
            "- Metrics include log loss, Brier, ROC AUC, average precision, calibration, and top-k precision/recall.",
            "",
            "## Calibration summary",
            "- Calibration bins are written to `calibration_summary.csv`.",
            "",
            "## Threshold/top-k analysis",
            "- Threshold and top-percentile summaries are written to `threshold_summary.csv`.",
            "",
            "## Slice analysis",
            "- Slice summaries use only available columns and skip small slices.",
            "",
            "## Promotion recommendation",
            *[f"- {reason}" for reason in recommendation["reasons"]],
            "",
            "## Limitations",
            "- Diagnostic CxA predicts `shot_created`.",
            "- `created_shot_cxg` and `cxa_value` are not model features.",
            "- CxA+ and Advanced CxA come later.",
            "",
            "## Next recommended PR",
            "- If promoted, generate governed promoted CxA outputs.",
            "",
        ]
    )


def validate_cxa_diagnostic(
    paths: CxAValidationPaths,
    *,
    min_slice_rows: int = 500,
) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "validation_summary": paths.output_dir / "validation_summary.json",
        "promotion_recommendation": paths.output_dir / "promotion_recommendation.json",
        "validation_report": paths.output_dir / "validation_report.md",
        "baseline_vs_diagnostic_metrics": paths.output_dir / "baseline_vs_diagnostic_metrics.csv",
        "calibration_summary": paths.output_dir / "calibration_summary.csv",
        "threshold_summary": paths.output_dir / "threshold_summary.csv",
        "slice_summary": paths.output_dir / "slice_summary.csv",
        "validation_quality_checks": paths.output_dir / "validation_quality_checks.csv",
    }
    required_paths = {
        "diagnostic_predictions": paths.diagnostic_predictions,
        "selected_model_metadata": paths.selected_model_metadata,
        "model_comparison": paths.model_comparison,
        "feature_contract": paths.feature_contract,
        "baseline_predictions": paths.baseline_predictions,
        "baseline_metrics": paths.baseline_metrics,
    }
    missing_inputs = [
        {"check_name": f"{name}_exists", "path": path}
        for name, path in required_paths.items()
        if not path.exists()
    ]
    if missing_inputs:
        checks = pd.DataFrame(
            [
                {
                    "check_name": item["check_name"],
                    "value": item["path"].as_posix(),
                    "status": "failed",
                    "severity": "blocker",
                    "notes": "Required validation input is missing.",
                }
                for item in missing_inputs
            ]
        )
        recommendation = {
            "recommendation": "blocked",
            "reasons": [f"Missing required input: {item['path']}" for item in missing_inputs],
            "known_limitations": [],
        }
        summary = {
            "metric": "cxa",
            "model_version": "diagnostic_v1",
            "selected_diagnostic_model": None,
            "baseline_prediction_column": None,
            "join_key": [],
            "row_counts": {},
            "quality_check_summary": {
                "failed_blocker_count": int(len(checks)),
                "failed_checks": checks["check_name"].tolist(),
            },
            "baseline_metrics": {},
            "diagnostic_metrics": {},
            "metric_deltas": {},
            "promotion_recommendation": "blocked",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inputs": required_paths,
            "outputs": output_paths,
        }
        output_paths["validation_summary"].write_text(
            json.dumps(_json_safe(summary), indent=2),
            encoding="utf-8",
        )
        output_paths["promotion_recommendation"].write_text(
            json.dumps(_json_safe(recommendation), indent=2),
            encoding="utf-8",
        )
        output_paths["validation_report"].write_text(
            "\n".join(
                [
                    "# Diagnostic CxA Validation Report",
                    "",
                    "## Executive summary",
                    "- Validation is blocked because required inputs are missing.",
                    "",
                    "## Promotion recommendation",
                    "- `blocked`",
                    "",
                    "## Missing inputs",
                    *[f"- `{item['path']}`" for item in missing_inputs],
                    "",
                ]
            ),
            encoding="utf-8",
        )
        pd.DataFrame().to_csv(output_paths["baseline_vs_diagnostic_metrics"], index=False)
        pd.DataFrame().to_csv(output_paths["calibration_summary"], index=False)
        pd.DataFrame().to_csv(output_paths["threshold_summary"], index=False)
        pd.DataFrame().to_csv(output_paths["slice_summary"], index=False)
        checks.to_csv(output_paths["validation_quality_checks"], index=False)
        return output_paths

    metadata = _read_json(paths.selected_model_metadata)
    selected_model = selected_diagnostic_model(metadata)
    diagnostic = _read_table(paths.diagnostic_predictions)
    baseline = _read_table(paths.baseline_predictions)
    baseline_prediction_column = detect_baseline_prediction_column(baseline)
    candidate_matches = bool((diagnostic.get("model_candidate") == selected_model).any())
    joined, join_key, join_quality = join_predictions(
        diagnostic,
        baseline,
        selected_model=selected_model,
        baseline_prediction_column=baseline_prediction_column,
    )
    joined_eval = joined.dropna(subset=["baseline_probability"]).copy()
    baseline_metrics = metric_summary(
        joined_eval,
        prediction_column="baseline_probability",
        model_name="baseline",
    )
    diagnostic_metrics = metric_summary(
        joined_eval,
        prediction_column="diagnostic_probability",
        model_name="diagnostic_v1",
    )
    comparison = _metric_comparison_frame(baseline_metrics, diagnostic_metrics)
    calibration = pd.concat(
        [
            calibration_summary(
                joined_eval,
                prediction_column="baseline_probability",
                model_name="baseline",
            ),
            calibration_summary(
                joined_eval,
                prediction_column="diagnostic_probability",
                model_name="diagnostic_v1",
            ),
        ],
        ignore_index=True,
    )
    thresholds = pd.concat(
        [
            threshold_summary(
                joined_eval,
                prediction_column="baseline_probability",
                model_name="baseline",
            ),
            threshold_summary(
                joined_eval,
                prediction_column="diagnostic_probability",
                model_name="diagnostic_v1",
            ),
        ],
        ignore_index=True,
    )
    slices = slice_summary(joined_eval, min_rows=min_slice_rows)
    checks = quality_checks(join_quality, candidate_matches=candidate_matches)
    recommendation = promotion_recommendation(
        baseline_metrics=baseline_metrics,
        diagnostic_metrics=diagnostic_metrics,
        checks=checks,
        slices=slices,
    )
    metric_deltas = {
        row["metric"]: row["diagnostic_minus_baseline"]
        for row in comparison.to_dict(orient="records")
    }
    summary = {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "selected_diagnostic_model": selected_model,
        "baseline_prediction_column": baseline_prediction_column,
        "join_key": join_key,
        "row_counts": join_quality,
        "quality_check_summary": {
            "failed_blocker_count": int(
                ((checks["severity"] == "blocker") & (checks["status"] == "failed")).sum()
            ),
            "failed_checks": checks.loc[checks["status"] == "failed", "check_name"].tolist(),
        },
        "baseline_metrics": baseline_metrics,
        "diagnostic_metrics": diagnostic_metrics,
        "metric_deltas": metric_deltas,
        "promotion_recommendation": recommendation["recommendation"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "diagnostic_predictions": paths.diagnostic_predictions,
            "selected_model_metadata": paths.selected_model_metadata,
            "model_comparison": paths.model_comparison,
            "feature_contract": paths.feature_contract,
            "baseline_predictions": paths.baseline_predictions,
            "baseline_metrics": paths.baseline_metrics,
        },
        "outputs": output_paths,
    }
    output_paths["validation_summary"].write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    output_paths["promotion_recommendation"].write_text(
        json.dumps(
            _json_safe(
                {
                    **recommendation,
                    "baseline_metrics": baseline_metrics,
                    "diagnostic_metrics": diagnostic_metrics,
                    "metric_deltas": metric_deltas,
                    "next_issue": (
                        "generate governed promoted CxA outputs if promoted/provisional; "
                        "otherwise revise diagnostic training"
                    ),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths["validation_report"].write_text(
        _validation_report(
            selected_model=selected_model,
            baseline_prediction_column=baseline_prediction_column,
            join_key=join_key,
            comparison=comparison,
            checks=checks,
            recommendation=recommendation,
        ),
        encoding="utf-8",
    )
    comparison.to_csv(output_paths["baseline_vs_diagnostic_metrics"], index=False)
    calibration.to_csv(output_paths["calibration_summary"], index=False)
    thresholds.to_csv(output_paths["threshold_summary"], index=False)
    slices.to_csv(output_paths["slice_summary"], index=False)
    checks.to_csv(output_paths["validation_quality_checks"], index=False)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=Path("outputs/modeling/cxa/diagnostic_v1"),
    )
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/modeling/cxa"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/validation/cxa/diagnostic_v1"),
    )
    parser.add_argument("--min-slice-rows", type=int, default=500)
    args = parser.parse_args()

    outputs = validate_cxa_diagnostic(
        CxAValidationPaths.from_roots(
            diagnostic_dir=args.diagnostic_dir,
            baseline_dir=args.baseline_dir,
            output_dir=args.output_dir,
        ),
        min_slice_rows=args.min_slice_rows,
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
