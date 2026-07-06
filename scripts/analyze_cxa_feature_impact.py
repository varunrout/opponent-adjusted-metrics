#!/usr/bin/env python
"""Analyze governed diagnostic CxA feature impact without retraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

try:
    from scripts.run_cxa_diagnostic_training import _coerce_binary_frame as _coerce_binary_frame
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_cxa_diagnostic_training import _coerce_binary_frame as _coerce_binary_frame


DEFAULT_FEATURE_PATH = Path("feature_store/cxa/action_features.parquet")
DEFAULT_DIAGNOSTIC_DIR = Path("outputs/modeling/cxa/diagnostic_v1")
DEFAULT_RESULTS_DIR = Path("outputs/results/cxa/diagnostic_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_DIAGNOSTIC_DIR / "feature_impact"
MODEL_VERSION = "diagnostic_v1"
TARGET_COLUMN = "shot_created"
PREDICTION_COLUMN = "predicted_shot_created_probability"
DIAGNOSTIC_VALUE_COLUMN = "diagnostic_cxa"
PROMOTION_ALLOWED = {"promoted", "provisionally_promoted"}
PROMOTION_BLOCKED = {"blocked", "needs_revision"}
EXPLICIT_FORBIDDEN_COLUMNS = {
    "shot_created",
    "created_shot_cxg",
    "cxa_value",
    "created_shot_id",
}
RESULT_PREDICTION_COLUMNS = {
    "predicted_shot_created_probability",
    "diagnostic_cxa",
    "prediction_source",
    "model_version",
    "selected_model_candidate",
    "promotion_status",
    "promotion_recommendation",
    "created_shot_cxg_reference",
    "created_shot_id_reference",
}
REQUIRED_GROUP_ORDER = [
    "numeric",
    "binary",
    "categorical",
    "progression/location",
    "zone-entry",
    "action-type/context",
    "pressure",
    "time/sequence",
]
TOP_EXAMPLE_COLUMNS = [
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "sequence_id",
    "possession",
    "action_type",
    "shot_created",
    "predicted_shot_created_probability",
    "diagnostic_cxa",
    "is_progressive",
    "enters_final_third",
    "enters_penalty_area",
    "start_zone",
    "end_zone",
    "score_state",
]


@dataclass(frozen=True)
class CxAFeatureImpactPaths:
    """Input and output paths for governed diagnostic CxA feature impact."""

    selected_model: Path
    selected_model_metadata: Path
    feature_contract: Path
    action_predictions: Path
    model_promotion_summary: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "CxAFeatureImpactPaths":
        return cls(
            selected_model=diagnostic_dir / "models" / "selected_model.joblib",
            selected_model_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            action_predictions=results_dir / "action_predictions.parquet",
            model_promotion_summary=results_dir / "model_promotion_summary.json",
            output_dir=output_dir,
        )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if pd.isna(value) and not isinstance(value, (bool, str)):
        return None
    return value


def selected_model_candidate(metadata: dict[str, Any]) -> str:
    candidate = metadata.get("selected_model_candidate", metadata.get("selected_model"))
    if not candidate:
        raise ValueError("selected_model_metadata.json does not include a selected model candidate")
    return str(candidate)


def _required_selected_features(
    contract: dict[str, Any],
    promotion_summary: dict[str, Any],
) -> dict[str, list[str]]:
    contract_groups = contract.get("selected_feature_candidates", {})
    if not isinstance(contract_groups, dict) or not contract_groups:
        raise ValueError("feature_contract.json is missing selected_feature_candidates")

    governance_selected = promotion_summary.get("governance_summary", {}).get("selected_features")
    governance_set = {str(feature) for feature in governance_selected or []}

    grouped: dict[str, list[str]] = {}
    for key in ("numeric", "binary", "categorical"):
        raw = [str(feature) for feature in contract_groups.get(key, [])]
        grouped[key] = [
            feature for feature in raw if not governance_set or feature in governance_set
        ]

    flattened = flatten_feature_groups(grouped)
    if governance_set:
        missing = sorted(governance_set.difference(flattened))
        extra = sorted(set(flattened).difference(governance_set))
        if missing or extra:
            raise ValueError(
                "Promotion summary selected features do not match feature contract candidates: "
                f"missing={missing}, extra={extra}"
            )
    if not flattened:
        raise ValueError("No governed diagnostic CxA features were resolved")
    return grouped


def flatten_feature_groups(grouped: dict[str, list[str]]) -> list[str]:
    ordered = (
        grouped.get("numeric", []) + grouped.get("binary", []) + grouped.get("categorical", [])
    )
    return list(dict.fromkeys(str(feature) for feature in ordered))


def forbidden_columns(
    contract: dict[str, Any], prediction_columns: set[str] | None = None
) -> set[str]:
    excluded = contract.get("excluded_columns", {})
    columns = set(EXPLICIT_FORBIDDEN_COLUMNS)
    for key in (
        "target_columns",
        "reference_only_columns",
        "output_prediction_columns",
        "identifier_columns",
        "requires_review_columns",
        "excluded_unknown_columns",
    ):
        columns.update(str(column) for column in excluded.get(key, []))
    columns.update(str(column) for column in prediction_columns or set())
    return columns


def model_impact_features(
    grouped_features: dict[str, list[str]],
    *,
    forbidden: set[str],
) -> list[str]:
    selected = flatten_feature_groups(grouped_features)
    forbidden_used = sorted(set(selected).intersection(forbidden))
    if forbidden_used:
        raise ValueError(
            "Forbidden/reference columns were selected as model features: "
            + ", ".join(forbidden_used)
        )
    return selected


def validate_artifact_consistency(
    metadata: dict[str, Any],
    contract: dict[str, Any],
    promotion_summary: dict[str, Any],
    grouped_features: dict[str, list[str]],
) -> None:
    metadata_candidate = selected_model_candidate(metadata)
    promotion_candidate = str(promotion_summary.get("selected_model_candidate", ""))
    validation_candidate = str(promotion_summary.get("validation_selected_model", ""))
    expected_count = int(
        promotion_summary.get("governance_summary", {}).get("selected_feature_count", 0)
    )
    actual_count = len(flatten_feature_groups(grouped_features))

    if promotion_candidate and metadata_candidate != promotion_candidate:
        raise ValueError(
            "selected_model_metadata and model_promotion_summary disagree on selected model: "
            f"{metadata_candidate} != {promotion_candidate}"
        )
    if validation_candidate and metadata_candidate != validation_candidate:
        raise ValueError(
            "selected_model_metadata and model_promotion_summary disagree on validation model: "
            f"{metadata_candidate} != {validation_candidate}"
        )
    if (
        metadata.get("metric")
        and contract.get("metric")
        and metadata.get("metric") != contract.get("metric")
    ):
        raise ValueError("selected_model_metadata and feature_contract disagree on metric")
    if (
        metadata.get("model_version")
        and contract.get("model_version")
        and metadata.get("model_version") != contract.get("model_version")
    ):
        raise ValueError("selected_model_metadata and feature_contract disagree on model_version")
    if expected_count and expected_count != actual_count:
        raise ValueError(
            "model_promotion_summary selected_feature_count does not match resolved governed features: "
            f"{expected_count} != {actual_count}"
        )
    for key, group_name in (
        ("numeric_feature_count", "numeric"),
        ("binary_feature_count", "binary"),
        ("categorical_feature_count", "categorical"),
    ):
        if metadata.get(key) is not None and int(metadata[key]) != len(
            grouped_features[group_name]
        ):
            raise ValueError(
                f"selected_model_metadata {key} does not match resolved {group_name} features"
            )


def validate_promotion_status(promotion_summary: dict[str, Any]) -> list[str]:
    status = str(promotion_summary.get("promotion_status", "")).strip()
    recommendation = str(promotion_summary.get("promotion_recommendation", "")).strip()
    gate_passed = bool(promotion_summary.get("promotion_gate_passed"))
    if status in PROMOTION_BLOCKED or recommendation in PROMOTION_BLOCKED or not gate_passed:
        raise ValueError(
            "Diagnostic CxA feature impact analysis requires a promoted or provisionally promoted model"
        )
    warnings: list[str] = []
    if status == "provisionally_promoted":
        warnings.append(
            "Model is provisionally promoted; current baseline comparison remains reference-only/in-sample."
        )
    return warnings


def align_selected_feature_matrix(
    feature_frame: pd.DataFrame, selected_features: list[str]
) -> pd.DataFrame:
    missing = [feature for feature in selected_features if feature not in feature_frame.columns]
    if missing:
        raise ValueError(f"Selected governed features are missing from action features: {missing}")
    return feature_frame[selected_features].copy()


def predict_probabilities(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(matrix))
        if probabilities.ndim == 2:
            if probabilities.shape[1] == 1:
                return probabilities[:, 0].astype(float)
            return probabilities[:, 1].astype(float)
        return probabilities.astype(float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(matrix), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    predictions = np.asarray(model.predict(matrix), dtype=float)
    return predictions.astype(float)


def probability_metrics(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    metrics: dict[str, Any] = {
        "row_count": int(len(probs)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "mean_predicted_probability": float(probs.mean()) if len(probs) else np.nan,
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
        "roc_auc": np.nan,
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    return metrics


def sampled_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(feature_frame) <= sample_size:
        return feature_frame.reset_index(drop=True).copy()
    return (
        feature_frame.sample(n=sample_size, random_state=random_state)
        .sort_index()
        .reset_index(drop=True)
    )


def _feature_group_lookup(grouped_features: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for key in ("numeric", "binary", "categorical"):
        for feature in grouped_features[key]:
            lookup[feature] = key

    for feature in (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "length",
        "angle",
        "x_progression",
        "y_progression",
        "start_zone",
        "end_zone",
        "start_third",
        "end_third",
    ):
        lookup[feature] = "progression/location"
    for feature in ("enters_final_third", "enters_penalty_area", "enters_zone14"):
        lookup[feature] = "zone-entry"
    for feature in (
        "action_type",
        "body_part",
        "pass_height",
        "play_pattern",
        "prior_action_type",
        "score_state",
        "set_piece_phase",
        "is_cross",
        "is_through_ball",
        "is_pass",
        "is_carry",
        "is_dribble",
        "is_cutback",
        "switches_play",
    ):
        lookup[feature] = "action-type/context"
    for feature in ("under_pressure", "carry_under_pressure"):
        lookup[feature] = "pressure"
    for feature in (
        "minute",
        "second",
        "action_position",
        "sequence_length_so_far",
        "seconds_since_possession_start",
    ):
        lookup[feature] = "time/sequence"
    return lookup


def required_feature_groups(grouped_features: dict[str, list[str]]) -> dict[str, list[str]]:
    selected = set(flatten_feature_groups(grouped_features))
    groups = {
        "numeric": [feature for feature in grouped_features["numeric"] if feature in selected],
        "binary": [feature for feature in grouped_features["binary"] if feature in selected],
        "categorical": [
            feature for feature in grouped_features["categorical"] if feature in selected
        ],
        "progression/location": [
            feature
            for feature in (
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "length",
                "angle",
                "x_progression",
                "y_progression",
                "start_zone",
                "end_zone",
                "start_third",
                "end_third",
            )
            if feature in selected
        ],
        "zone-entry": [
            feature
            for feature in ("enters_final_third", "enters_penalty_area", "enters_zone14")
            if feature in selected
        ],
        "action-type/context": [
            feature
            for feature in (
                "action_type",
                "body_part",
                "pass_height",
                "play_pattern",
                "prior_action_type",
                "score_state",
                "set_piece_phase",
                "is_cross",
                "is_through_ball",
                "is_pass",
                "is_carry",
                "is_dribble",
                "is_cutback",
                "switches_play",
            )
            if feature in selected
        ],
        "pressure": [
            feature for feature in ("under_pressure", "carry_under_pressure") if feature in selected
        ],
        "time/sequence": [
            feature
            for feature in (
                "minute",
                "second",
                "action_position",
                "sequence_length_so_far",
                "seconds_since_possession_start",
            )
            if feature in selected
        ],
    }
    return groups


def _permuted_metric(
    model: Any,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    *,
    columns: list[str],
    baseline_metric: float,
    metric_name: str,
    n_repeats: int,
    random_state: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_state)
    values: list[float] = []
    mean_probability_deltas: list[float] = []
    baseline_probs = predict_probabilities(model, matrix)
    baseline_mean_probability = float(baseline_probs.mean())
    for _ in range(n_repeats):
        permuted = matrix.copy()
        for column in columns:
            shuffled = permuted[column].to_numpy(copy=True)
            rng.shuffle(shuffled)
            permuted[column] = shuffled
        probs = predict_probabilities(model, permuted)
        metrics = probability_metrics(y_true, probs)
        values.append(float(metrics[metric_name]))
        mean_probability_deltas.append(
            abs(float(metrics["mean_predicted_probability"]) - baseline_mean_probability)
        )
    return float(np.mean(values)), float(np.mean(mean_probability_deltas))


def permutation_feature_impact(
    model: Any,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    grouped_features: dict[str, list[str]],
    *,
    n_repeats: int,
    random_state: int,
    metric_name: str = "log_loss",
) -> pd.DataFrame:
    baseline_metrics = probability_metrics(y_true, predict_probabilities(model, matrix))
    baseline_metric = float(baseline_metrics[metric_name])
    group_lookup = _feature_group_lookup(grouped_features)
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(matrix.columns):
        permuted_metric, mean_probability_shift = _permuted_metric(
            model,
            matrix,
            y_true,
            columns=[feature],
            baseline_metric=baseline_metric,
            metric_name=metric_name,
            n_repeats=n_repeats,
            random_state=random_state + index + 1,
        )
        rows.append(
            {
                "feature_name": feature,
                "feature_group": group_lookup.get(feature, "other"),
                "baseline_metric": baseline_metric,
                "permuted_metric": permuted_metric,
                "impact": permuted_metric - baseline_metric,
                "mean_probability_shift": mean_probability_shift,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "feature_group",
                "baseline_metric",
                "permuted_metric",
                "impact",
                "mean_probability_shift",
                "rank",
            ]
        )
    result = result.sort_values(
        ["impact", "mean_probability_shift", "feature_name"], ascending=[False, False, True]
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def feature_group_impact(
    model: Any,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    grouped_features: dict[str, list[str]],
    *,
    n_repeats: int,
    random_state: int,
    metric_name: str = "log_loss",
) -> pd.DataFrame:
    baseline_metrics = probability_metrics(y_true, predict_probabilities(model, matrix))
    baseline_metric = float(baseline_metrics[metric_name])
    rows: list[dict[str, Any]] = []
    for index, group_name in enumerate(REQUIRED_GROUP_ORDER):
        features = [
            feature for feature in grouped_features.get(group_name, []) if feature in matrix.columns
        ]
        if not features:
            rows.append(
                {
                    "feature_group": group_name,
                    "feature_count": 0,
                    "feature_names": "",
                    "baseline_metric": baseline_metric,
                    "permuted_metric": None,
                    "impact": None,
                    "mean_probability_shift": None,
                    "status": "skipped_no_features",
                }
            )
            continue
        permuted_metric, mean_probability_shift = _permuted_metric(
            model,
            matrix,
            y_true,
            columns=features,
            baseline_metric=baseline_metric,
            metric_name=metric_name,
            n_repeats=n_repeats,
            random_state=random_state + (index + 1) * 100,
        )
        rows.append(
            {
                "feature_group": group_name,
                "feature_count": len(features),
                "feature_names": ", ".join(features),
                "baseline_metric": baseline_metric,
                "permuted_metric": permuted_metric,
                "impact": permuted_metric - baseline_metric,
                "mean_probability_shift": mean_probability_shift,
                "status": "computed",
            }
        )
    result = pd.DataFrame(rows)
    result["sort_impact"] = result["impact"].fillna(-np.inf)
    result = result.sort_values(["sort_impact", "feature_group"], ascending=[False, True]).drop(
        columns=["sort_impact"]
    )
    return result.reset_index(drop=True)


def top_feature_examples(
    action_predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    if "action_id" not in action_predictions.columns or "action_id" not in feature_frame.columns:
        raise ValueError("Both action predictions and action features must include action_id")
    merge_columns = [
        column
        for column in feature_frame.columns
        if column not in action_predictions.columns
        and column
        in {
            "is_progressive",
            "enters_final_third",
            "enters_penalty_area",
            "start_zone",
            "end_zone",
            "score_state",
        }
    ]
    merged = action_predictions.merge(
        feature_frame[["action_id", *merge_columns]],
        on="action_id",
        how="left",
    )

    rows: list[pd.DataFrame] = []

    def add_examples(category: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        selected_columns = [column for column in TOP_EXAMPLE_COLUMNS if column in frame.columns]
        subset = frame.nlargest(
            top_n,
            (
                DIAGNOSTIC_VALUE_COLUMN
                if DIAGNOSTIC_VALUE_COLUMN in frame.columns
                else PREDICTION_COLUMN
            ),
        )[selected_columns].copy()
        subset.insert(0, "example_category", category)
        subset.insert(1, "example_rank", np.arange(1, len(subset) + 1))
        rows.append(subset)

    add_examples("highest_diagnostic_cxa_actions", merged)
    if "is_progressive" in merged.columns:
        add_examples(
            "high_impact_progressive_actions",
            merged[merged["is_progressive"].fillna(False).astype(bool)],
        )
    final_third_mask = pd.Series(False, index=merged.index)
    if "enters_final_third" in merged.columns:
        final_third_mask = final_third_mask | merged["enters_final_third"].fillna(False).astype(
            bool
        )
    if "enters_penalty_area" in merged.columns:
        final_third_mask = final_third_mask | merged["enters_penalty_area"].fillna(False).astype(
            bool
        )
    if final_third_mask.any():
        add_examples("final_third_or_box_entry_actions", merged[final_third_mask])

    if not rows:
        return pd.DataFrame(columns=["example_category", "example_rank", *TOP_EXAMPLE_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def build_feature_impact_report(
    *,
    summary: dict[str, Any],
    feature_impact: pd.DataFrame,
    group_impact: pd.DataFrame,
) -> str:
    top_features = feature_impact.head(10)
    top_feature_lines = (
        [
            (
                f"- `{row.feature_name}` ({row.feature_group}) "
                f"impact `{row.impact:.6f}` from baseline log loss `{row.baseline_metric:.6f}` "
                f"to permuted log loss `{row.permuted_metric:.6f}`"
            )
            for row in top_features.itertuples(index=False)
        ]
        if not top_features.empty
        else ["- No governed impact features were available."]
    )
    group_lines = (
        [
            (
                f"- `{row.feature_group}`: {row.feature_count} features, "
                f"impact `{row.impact:.6f}`"
                if row.status == "computed"
                else f"- `{row.feature_group}`: skipped (no selected features)"
            )
            for row in group_impact.itertuples(index=False)
        ]
        if not group_impact.empty
        else ["- No feature-group impact rows were generated."]
    )
    warnings = summary.get("warnings", [])
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- None."]
    return "\n".join(
        [
            "# Diagnostic CxA Feature Impact Report",
            "",
            "## Executive summary",
            "Diagnostic CxA estimates probability that an action creates a shot. This report "
            "explains the currently selected governed diagnostic model with lightweight "
            "post-training perturbation analysis only.",
            f"- Selected model: `{summary['selected_model_candidate']}`",
            f"- Promotion status: `{summary['promotion_status']}`",
            f"- Sample rows used for impact analysis: `{summary['analysis_row_count']}`",
            *warning_lines,
            "",
            "## Model and promotion status",
            f"- Metric: `{summary['metric']}`",
            f"- Model version: `{summary['model_version']}`",
            f"- Promotion recommendation: `{summary['promotion_recommendation']}`",
            f"- Promotion gate passed: `{summary['promotion_gate_passed']}`",
            (
                "- The model is provisionally promoted."
                if summary["promotion_status"] == "provisionally_promoted"
                else "- The model is fully promoted."
            ),
            "",
            "## Top feature drivers",
            *top_feature_lines,
            "",
            "## Feature group impact",
            *group_lines,
            "",
            "## Football interpretation",
            "Positive impact means log loss worsened after the selected feature or feature group "
            "was permuted, so larger positive values indicate stronger dependence of the trained "
            "diagnostic model on that governed signal family.",
            "",
            "## Leakage/reference-only guard",
            "The analysis uses only governed diagnostic features from the diagnostic feature "
            "contract and promotion summary. `created_shot_cxg` and `cxa_value` are not model "
            "features. Identifiers, prediction outputs, requires-review columns, and "
            "excluded-unknown columns are excluded before scoring.",
            "",
            "## Limitations",
            "This analysis does not retrain the model, change validation, or change result "
            "generation. It is a local sensitivity read on the provisionally promoted model and "
            "does not establish causal football effects. CxA+ and Advanced CxA come later.",
            "",
            "## Next recommended PR",
            f"- `{summary['next_recommended_pr']}`",
            "",
        ]
    )


def analyze_cxa_feature_impact(
    *,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    paths: CxAFeatureImpactPaths | None = None,
    sample_size: int = 20000,
    n_repeats: int = 2,
    random_state: int = 42,
    top_n_examples: int = 15,
) -> dict[str, Path]:
    paths = paths or CxAFeatureImpactPaths.from_roots()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(paths.selected_model_metadata)
    contract = _read_json(paths.feature_contract)
    promotion_summary = _read_json(paths.model_promotion_summary)
    warnings = validate_promotion_status(promotion_summary)

    action_predictions = pd.read_parquet(paths.action_predictions)
    if action_predictions.empty:
        raise ValueError(
            "Governed diagnostic CxA action predictions are required for feature impact"
        )
    if "action_id" not in action_predictions.columns:
        raise ValueError("Governed diagnostic CxA action predictions are missing action_id")

    grouped_features = _required_selected_features(contract, promotion_summary)
    validate_artifact_consistency(metadata, contract, promotion_summary, grouped_features)

    forbidden = forbidden_columns(contract, prediction_columns=RESULT_PREDICTION_COLUMNS)
    selected_features = model_impact_features(grouped_features, forbidden=forbidden)

    feature_frame = pd.read_parquet(feature_path)
    if TARGET_COLUMN not in feature_frame.columns:
        raise ValueError(f"Action features are missing required target column {TARGET_COLUMN}")

    matrix_source = sampled_feature_frame(
        feature_frame, sample_size=sample_size, random_state=random_state
    )
    matrix = align_selected_feature_matrix(matrix_source, selected_features)
    y_true = matrix_source[TARGET_COLUMN].astype(int)
    model = joblib.load(paths.selected_model)

    promoted_metrics = probability_metrics(y_true, predict_probabilities(model, matrix))
    required_groups = required_feature_groups(grouped_features)
    feature_impact = permutation_feature_impact(
        model,
        matrix,
        y_true,
        grouped_features,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    group_impact = feature_group_impact(
        model,
        matrix,
        y_true,
        required_groups,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    examples = top_feature_examples(action_predictions, feature_frame, top_n=top_n_examples)

    outputs = {
        "feature_impact_summary_csv": paths.output_dir / "feature_impact_summary.csv",
        "feature_group_impact_csv": paths.output_dir / "feature_group_impact.csv",
        "top_feature_examples_csv": paths.output_dir / "top_feature_examples.csv",
        "feature_impact_report": paths.output_dir / "feature_impact_report.md",
        "feature_impact_summary_json": paths.output_dir / "feature_impact_summary.json",
    }

    feature_impact.to_csv(outputs["feature_impact_summary_csv"], index=False)
    group_impact.to_csv(outputs["feature_group_impact_csv"], index=False)
    examples.to_csv(outputs["top_feature_examples_csv"], index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": str(contract.get("metric", metadata.get("metric", "cxa"))),
        "model_version": str(
            contract.get("model_version", metadata.get("model_version", MODEL_VERSION))
        ),
        "selected_model_candidate": selected_model_candidate(metadata),
        "promotion_status": promotion_summary.get("promotion_status"),
        "promotion_recommendation": promotion_summary.get("promotion_recommendation"),
        "promotion_gate_passed": promotion_summary.get("promotion_gate_passed"),
        "warnings": warnings,
        "analysis_row_count": int(len(matrix_source)),
        "sample_size_requested": int(sample_size),
        "n_repeats": int(n_repeats),
        "random_state": int(random_state),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "forbidden_features_used": sorted(set(selected_features).intersection(forbidden)),
        "selected_feature_groups": required_groups,
        "baseline_metric_name": "log_loss",
        "baseline_metric_value": promoted_metrics["log_loss"],
        "promoted_metrics_on_sample": promoted_metrics,
        "top_feature_driver": (
            feature_impact.iloc[0]["feature_name"] if not feature_impact.empty else None
        ),
        "top_feature_group_driver": (
            group_impact[group_impact["status"] == "computed"].iloc[0]["feature_group"]
            if not group_impact[group_impact["status"] == "computed"].empty
            else None
        ),
        "top_examples_row_count": int(len(examples)),
        "next_recommended_pr": "analysis/cxa-promoted-portfolio-summary",
        "sources": {
            "feature_path": str(feature_path),
            "selected_model": str(paths.selected_model),
            "selected_model_metadata": str(paths.selected_model_metadata),
            "feature_contract": str(paths.feature_contract),
            "action_predictions": str(paths.action_predictions),
            "model_promotion_summary": str(paths.model_promotion_summary),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }

    _write_json(outputs["feature_impact_summary_json"], summary)
    outputs["feature_impact_report"].write_text(
        build_feature_impact_report(
            summary=summary,
            feature_impact=feature_impact,
            group_impact=group_impact,
        ),
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze promoted diagnostic CxA feature impact")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n-examples", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = CxAFeatureImpactPaths.from_roots(
        diagnostic_dir=args.diagnostic_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    outputs = analyze_cxa_feature_impact(
        feature_path=args.feature_path,
        paths=paths,
        sample_size=args.sample_size,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        top_n_examples=args.top_n_examples,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
