#!/usr/bin/env python
"""Generate governed diagnostic CxA result outputs from the selected model."""

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

try:
    from scripts.run_cxa_diagnostic_training import _coerce_binary_frame as _coerce_binary_frame
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_cxa_diagnostic_training import _coerce_binary_frame as _coerce_binary_frame

DEFAULT_FEATURE_PATH = Path("feature_store") / "cxa" / "action_features.parquet"
DEFAULT_DIAGNOSTIC_DIR = Path("outputs") / "modeling" / "cxa" / "diagnostic_v1"
DEFAULT_VALIDATION_DIR = Path("outputs") / "validation" / "cxa" / "diagnostic_v1"
DEFAULT_BASELINE_DIR = Path("outputs") / "modeling" / "cxa" / "baseline"
DEFAULT_OUTPUT_DIR = Path("outputs") / "results" / "cxa" / "diagnostic_v1"
MODEL_VERSION = "diagnostic_v1"
METRIC = "cxa"
TARGET_COLUMN = "shot_created"
PREDICTION_COLUMN = "predicted_shot_created_probability"
ALLOWED_RECOMMENDATIONS = {"promote", "provisional_promote"}
REFERENCE_COLUMNS = {
    "created_shot_cxg",
    "created_shot_id",
    "cxa_value",
}
REQUIRED_ACTION_COLUMNS = (
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    TARGET_COLUMN,
    PREDICTION_COLUMN,
    "diagnostic_cxa",
    "prediction_source",
    "model_version",
    "selected_model_candidate",
    "promotion_status",
    "promotion_recommendation",
)
ACTION_CONTEXT_COLUMNS = (
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "sequence_id",
    "possession",
    "action_type",
    TARGET_COLUMN,
)
OPTIONAL_ID_COLUMNS = ("event_id", "match_id", "team_id", "player_id", "sequence_id", "possession")


@dataclass(frozen=True)
class CxAResultPaths:
    """Input and output paths for diagnostic CxA result generation."""

    feature_path: Path
    selected_model: Path
    selected_model_metadata: Path
    feature_contract: Path
    validation_summary: Path
    promotion_recommendation: Path
    baseline_vs_diagnostic_metrics: Path
    baseline_predictions: Path
    legacy_baseline_predictions: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        *,
        feature_path: Path = DEFAULT_FEATURE_PATH,
        diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
        validation_dir: Path = DEFAULT_VALIDATION_DIR,
        baseline_dir: Path = DEFAULT_BASELINE_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "CxAResultPaths":
        legacy_baseline_dir = (
            baseline_dir.parent if baseline_dir.name == "baseline" else baseline_dir
        )
        return cls(
            feature_path=feature_path,
            selected_model=diagnostic_dir / "models" / "selected_model.joblib",
            selected_model_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            validation_summary=validation_dir / "validation_summary.json",
            promotion_recommendation=validation_dir / "promotion_recommendation.json",
            baseline_vs_diagnostic_metrics=validation_dir / "baseline_vs_diagnostic_metrics.csv",
            baseline_predictions=baseline_dir / "predictions" / "action_predictions.parquet",
            legacy_baseline_predictions=legacy_baseline_dir
            / "predictions"
            / "action_predictions.parquet",
            output_dir=output_dir,
        )


@dataclass(frozen=True)
class CxAResultOutputs:
    action_predictions: Path
    player_cxa_summary_csv: Path
    player_cxa_summary_parquet: Path
    team_cxa_summary_csv: Path
    team_cxa_summary_parquet: Path
    sequence_cxa_summary_csv: Path
    sequence_cxa_summary_parquet: Path
    top_players_by_cxa: Path
    team_cxa_rankings: Path
    baseline_vs_diagnostic_summary: Path
    model_promotion_summary: Path
    prediction_quality_checks: Path
    cxa_results_report: Path

    @property
    def full_output_artifacts(self) -> tuple[Path, ...]:
        """Scoring outputs that must not survive a blocked result run."""

        return (
            self.action_predictions,
            self.player_cxa_summary_csv,
            self.player_cxa_summary_parquet,
            self.team_cxa_summary_csv,
            self.team_cxa_summary_parquet,
            self.sequence_cxa_summary_csv,
            self.sequence_cxa_summary_parquet,
            self.top_players_by_cxa,
            self.team_cxa_rankings,
            self.baseline_vs_diagnostic_summary,
        )


def output_paths(output_dir: Path) -> CxAResultOutputs:
    return CxAResultOutputs(
        action_predictions=output_dir / "action_predictions.parquet",
        player_cxa_summary_csv=output_dir / "player_cxa_summary.csv",
        player_cxa_summary_parquet=output_dir / "player_cxa_summary.parquet",
        team_cxa_summary_csv=output_dir / "team_cxa_summary.csv",
        team_cxa_summary_parquet=output_dir / "team_cxa_summary.parquet",
        sequence_cxa_summary_csv=output_dir / "sequence_cxa_summary.csv",
        sequence_cxa_summary_parquet=output_dir / "sequence_cxa_summary.parquet",
        top_players_by_cxa=output_dir / "top_players_by_cxa.csv",
        team_cxa_rankings=output_dir / "team_cxa_rankings.csv",
        baseline_vs_diagnostic_summary=output_dir / "baseline_vs_diagnostic_summary.csv",
        model_promotion_summary=output_dir / "model_promotion_summary.json",
        prediction_quality_checks=output_dir / "prediction_quality_checks.csv",
        cxa_results_report=output_dir / "cxa_results_report.md",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, bool | str):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _existing_path(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    return fallback


def selected_model_name(metadata: dict[str, Any]) -> str:
    selected = (
        metadata.get("selected_model_candidate")
        or metadata.get("selected_model")
        or metadata.get("selected_candidate")
    )
    if not selected:
        raise ValueError("selected_model_metadata.json does not identify the selected CxA model")
    return str(selected)


def validation_selected_model(validation_summary: dict[str, Any]) -> str | None:
    selected = (
        validation_summary.get("selected_diagnostic_model")
        or validation_summary.get("selected_model_candidate")
        or validation_summary.get("selected_model")
    )
    return str(selected) if selected else None


def resolve_feature_columns(
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    contract: dict[str, Any],
) -> list[str]:
    """Resolve model input features without inferring beyond governed metadata/contract."""

    feature_candidates: list[str] = []
    for key in ("selected_features", "feature_columns"):
        values = metadata.get(key)
        if isinstance(values, list):
            feature_candidates.extend(str(value) for value in values)
    selected = contract.get("selected_feature_candidates", {})
    for group in ("numeric", "binary", "categorical"):
        feature_candidates.extend(str(column) for column in selected.get(group, []))
    feature_candidates = list(dict.fromkeys(feature_candidates))

    forbidden = set()
    excluded = contract.get("excluded_columns", {})
    if isinstance(excluded, dict):
        for columns in excluded.values():
            forbidden.update(str(column) for column in columns)
    forbidden.update({TARGET_COLUMN, "created_shot_cxg", "created_shot_id", "cxa_value"})

    features = [
        column
        for column in feature_candidates
        if column in frame.columns and column not in forbidden and frame[column].notna().any()
    ]
    if not features:
        raise ValueError("No governed CxA diagnostic model features are available for scoring")
    return features


def _positive_class_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(frame), dtype=float)


def promotion_status(recommendation: str) -> tuple[str, str, bool]:
    if recommendation == "promote":
        return "promoted", "promoted_model", True
    if recommendation == "provisional_promote":
        return "provisionally_promoted", "provisional_promoted_model", True
    return "blocked", "blocked", False


def validation_metric_deltas(validation_summary: dict[str, Any]) -> dict[str, Any]:
    deltas = validation_summary.get("metric_deltas")
    if isinstance(deltas, dict):
        return deltas
    baseline = validation_summary.get("baseline_metrics", {})
    diagnostic = validation_summary.get("diagnostic_metrics", {})
    if not isinstance(baseline, dict) or not isinstance(diagnostic, dict):
        return {}
    return {
        metric: diagnostic.get(metric) - baseline.get(metric)
        for metric in sorted(set(baseline) & set(diagnostic))
        if isinstance(baseline.get(metric), int | float)
        and isinstance(diagnostic.get(metric), int | float)
    }


def _check_row(
    name: str,
    value: Any,
    *,
    status: str,
    severity: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "check_name": name,
        "value": value,
        "status": status,
        "severity": severity,
        "notes": notes,
    }


def preflight_checks(
    paths: CxAResultPaths,
    *,
    recommendation: str | None,
    current_selected_model: str | None,
    validation_selected: str | None,
    validation_summary: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_paths = {
        "selected_model_artifact": paths.selected_model,
        "selected_model_metadata": paths.selected_model_metadata,
        "feature_contract": paths.feature_contract,
        "validation_summary": paths.validation_summary,
        "promotion_recommendation": paths.promotion_recommendation,
        "feature_table": paths.feature_path,
    }
    for name, path in required_paths.items():
        exists = path.exists()
        rows.append(
            _check_row(
                name,
                path.as_posix(),
                status="passed" if exists else "failed",
                severity="info" if exists else "blocker",
                notes="" if exists else "Required CxA result input is missing.",
            )
        )

    recommendation_allowed = recommendation in ALLOWED_RECOMMENDATIONS
    rows.append(
        _check_row(
            "validation_recommendation_allowed",
            recommendation,
            status="passed" if recommendation_allowed else "failed",
            severity="info" if recommendation_allowed else "blocker",
            notes=(
                ""
                if recommendation_allowed
                else "Validation recommendation does not allow promoted CxA result outputs."
            ),
        )
    )
    model_matches = bool(
        current_selected_model
        and validation_selected
        and current_selected_model == validation_selected
    )
    rows.append(
        _check_row(
            "selected_model_matches_validation",
            model_matches,
            status="passed" if model_matches else "failed",
            severity="info" if model_matches else "blocker",
            notes=(
                ""
                if model_matches
                else "Selected model metadata does not match the model validated in CxA validation."
            ),
        )
    )
    if recommendation == "provisional_promote":
        rows.append(
            _check_row(
                "provisional_promotion",
                recommendation,
                status="warning",
                severity="warning",
                notes="Validation recommendation is provisional_promote, so outputs are provisionally promoted.",
            )
        )
    if validation_summary:
        baseline_fair = bool(validation_summary.get("baseline_is_fair_comparator"))
        strict_enabled = bool(validation_summary.get("strict_promotion_comparison_enabled"))
        rows.append(
            _check_row(
                "baseline_is_fair_comparator",
                baseline_fair,
                status="passed" if baseline_fair else "warning",
                severity="info" if baseline_fair else "warning",
                notes=(
                    ""
                    if baseline_fair
                    else "Baseline comparison is reference-only because baseline predictions are in-sample."
                ),
            )
        )
        rows.append(
            _check_row(
                "strict_promotion_comparison_enabled",
                strict_enabled,
                status="passed" if strict_enabled else "warning",
                severity="info" if strict_enabled else "warning",
                notes=(
                    ""
                    if strict_enabled
                    else "Strict promotion comparison is disabled by validation provenance checks."
                ),
            )
        )
    return pd.DataFrame(rows)


def prediction_quality_checks(
    predictions: pd.DataFrame,
    *,
    preflight: pd.DataFrame,
) -> pd.DataFrame:
    rows = preflight.to_dict("records")
    probability = pd.to_numeric(predictions[PREDICTION_COLUMN], errors="coerce")
    rows.extend(
        [
            _check_row(
                "row_count",
                len(predictions),
                status="passed" if len(predictions) else "failed",
                severity="info" if len(predictions) else "blocker",
                notes="" if len(predictions) else "No action rows were scored.",
            ),
            _check_row(
                "prediction_null_count",
                int(probability.isna().sum()),
                status="passed" if probability.notna().all() else "failed",
                severity="info" if probability.notna().all() else "blocker",
                notes=(
                    ""
                    if probability.notna().all()
                    else "Predicted CxA probabilities contain nulls."
                ),
            ),
            _check_row(
                "probability_outside_0_1_count",
                int((~probability.between(0, 1)).sum()),
                status="passed" if probability.between(0, 1).all() else "failed",
                severity="info" if probability.between(0, 1).all() else "blocker",
                notes=(
                    ""
                    if probability.between(0, 1).all()
                    else "Predicted probabilities must be in [0, 1]."
                ),
            ),
            _check_row(
                "action_id_missing_count",
                (
                    int(predictions["action_id"].isna().sum())
                    if "action_id" in predictions
                    else len(predictions)
                ),
                status=(
                    "passed"
                    if "action_id" in predictions and predictions["action_id"].notna().all()
                    else "failed"
                ),
                severity=(
                    "info"
                    if "action_id" in predictions and predictions["action_id"].notna().all()
                    else "blocker"
                ),
                notes=(
                    ""
                    if "action_id" in predictions
                    else "action_id is required for result outputs."
                ),
            ),
            _check_row(
                "duplicate_action_id_count",
                (
                    int(predictions["action_id"].duplicated().sum())
                    if "action_id" in predictions
                    else len(predictions)
                ),
                status=(
                    "passed"
                    if "action_id" in predictions
                    and predictions["action_id"].duplicated().sum() == 0
                    else "failed"
                ),
                severity=(
                    "info"
                    if "action_id" in predictions
                    and predictions["action_id"].duplicated().sum() == 0
                    else "blocker"
                ),
                notes="Duplicate action_id values block governed CxA result outputs.",
            ),
        ]
    )
    for column in OPTIONAL_ID_COLUMNS:
        if column in predictions:
            missing = int(predictions[column].isna().sum())
            rows.append(
                _check_row(
                    f"{column}_missing_count",
                    missing,
                    status="passed" if missing == 0 else "warning",
                    severity="info" if missing == 0 else "warning",
                    notes=(
                        ""
                        if missing == 0
                        else f"Optional identifier `{column}` has missing values."
                    ),
                )
            )
    if "created_shot_id_reference" in predictions:
        missing = int(predictions["created_shot_id_reference"].isna().sum())
        rows.append(
            _check_row(
                "created_shot_id_reference_missing_count",
                missing,
                status="warning" if missing else "passed",
                severity="warning" if missing else "info",
                notes=(
                    "Sparse created_shot_id is expected because most CxA actions do not create shots."
                    if missing
                    else ""
                ),
            )
        )
    return pd.DataFrame(rows)


def has_blockers(checks: pd.DataFrame) -> bool:
    return bool(((checks["severity"] == "blocker") & (checks["status"] == "failed")).any())


def build_action_predictions(
    feature_frame: pd.DataFrame,
    model: Any,
    feature_columns: list[str],
    *,
    selected_model: str,
    recommendation: str,
    status: str,
    prediction_source: str,
) -> pd.DataFrame:
    probabilities = _positive_class_probability(model, feature_frame[feature_columns])
    predictions = pd.DataFrame(index=feature_frame.index)
    for column in ACTION_CONTEXT_COLUMNS:
        if column in feature_frame.columns:
            predictions[column] = feature_frame[column]
    if TARGET_COLUMN not in predictions:
        predictions[TARGET_COLUMN] = np.nan
    predictions[PREDICTION_COLUMN] = probabilities
    predictions["diagnostic_cxa"] = probabilities
    predictions["prediction_source"] = prediction_source
    predictions["model_version"] = MODEL_VERSION
    predictions["selected_model_candidate"] = selected_model
    predictions["promotion_status"] = status
    predictions["promotion_recommendation"] = recommendation
    if "created_shot_cxg" in feature_frame.columns:
        predictions["created_shot_cxg_reference"] = feature_frame["created_shot_cxg"]
    if "created_shot_id" in feature_frame.columns:
        predictions["created_shot_id_reference"] = feature_frame["created_shot_id"]
    for column in REQUIRED_ACTION_COLUMNS:
        if column not in predictions.columns:
            predictions[column] = np.nan
    return predictions.reset_index(drop=True)


def _summary_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def player_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in predictions.columns:
        return _summary_frame(
            [
                "player_id",
                "team_id",
                "actions",
                "shot_creating_actions",
                "total_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "max_diagnostic_cxa",
                "rank",
            ]
        )
    group_cols = ["player_id"]
    if "team_id" in predictions.columns:
        group_cols.append("team_id")
    grouped = predictions.groupby(group_cols, dropna=False).agg(
        actions=("diagnostic_cxa", "size"),
        shot_creating_actions=(TARGET_COLUMN, "sum"),
        total_diagnostic_cxa=("diagnostic_cxa", "sum"),
        mean_diagnostic_cxa=("diagnostic_cxa", "mean"),
        max_diagnostic_cxa=("diagnostic_cxa", "max"),
    )
    output = grouped.reset_index().sort_values("total_diagnostic_cxa", ascending=False)
    output["rank"] = np.arange(1, len(output) + 1)
    return output


def team_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if "team_id" not in predictions.columns:
        return _summary_frame(
            [
                "team_id",
                "actions",
                "shot_creating_actions",
                "total_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "max_diagnostic_cxa",
                "rank",
            ]
        )
    output = (
        predictions.groupby(["team_id"], dropna=False)
        .agg(
            actions=("diagnostic_cxa", "size"),
            shot_creating_actions=(TARGET_COLUMN, "sum"),
            total_diagnostic_cxa=("diagnostic_cxa", "sum"),
            mean_diagnostic_cxa=("diagnostic_cxa", "mean"),
            max_diagnostic_cxa=("diagnostic_cxa", "max"),
        )
        .reset_index()
        .sort_values("total_diagnostic_cxa", ascending=False)
    )
    output["rank"] = np.arange(1, len(output) + 1)
    return output


def sequence_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if "sequence_id" not in predictions.columns:
        return _summary_frame(
            [
                "sequence_id",
                "match_id",
                "team_id",
                "possession",
                "actions",
                "shot_creating_actions",
                "total_diagnostic_cxa",
                "max_diagnostic_cxa",
                "mean_diagnostic_cxa",
                "sequence_led_to_shot",
                "rank",
            ]
        )
    group_cols = ["sequence_id"]
    for column in ("match_id", "team_id", "possession"):
        if column in predictions.columns:
            group_cols.append(column)
    output = (
        predictions.groupby(group_cols, dropna=False)
        .agg(
            actions=("diagnostic_cxa", "size"),
            shot_creating_actions=(TARGET_COLUMN, "sum"),
            total_diagnostic_cxa=("diagnostic_cxa", "sum"),
            max_diagnostic_cxa=("diagnostic_cxa", "max"),
            mean_diagnostic_cxa=("diagnostic_cxa", "mean"),
        )
        .reset_index()
        .sort_values("total_diagnostic_cxa", ascending=False)
    )
    output["sequence_led_to_shot"] = (output["shot_creating_actions"] > 0).astype(int)
    output["rank"] = np.arange(1, len(output) + 1)
    return output


def baseline_vs_diagnostic_summary(
    validation_metrics: pd.DataFrame,
    validation_summary: dict[str, Any],
) -> pd.DataFrame:
    if validation_metrics.empty:
        output = pd.DataFrame(
            columns=[
                "metric",
                "baseline",
                "diagnostic",
                "diagnostic_minus_baseline",
                "baseline_prediction_provenance",
                "baseline_is_fair_comparator",
            ]
        )
    else:
        output = validation_metrics.copy()
    output["baseline_prediction_provenance"] = validation_summary.get(
        "baseline_prediction_provenance"
    )
    output["baseline_is_fair_comparator"] = bool(
        validation_summary.get("baseline_is_fair_comparator")
    )
    return output


def prediction_summary(predictions: pd.DataFrame | None) -> dict[str, Any]:
    if predictions is None or predictions.empty or PREDICTION_COLUMN not in predictions:
        return {
            "row_count": 0,
            "prediction_null_count": None,
            "probability_min": None,
            "probability_max": None,
            "probability_mean": None,
            "total_diagnostic_cxa": None,
        }
    probability = pd.to_numeric(predictions[PREDICTION_COLUMN], errors="coerce")
    return {
        "row_count": int(len(predictions)),
        "prediction_null_count": int(probability.isna().sum()),
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "total_diagnostic_cxa": float(predictions["diagnostic_cxa"].sum()),
    }


def promotion_summary(
    *,
    paths: CxAResultPaths,
    outputs: CxAResultOutputs,
    selected_model: str | None,
    validation_selected: str | None,
    recommendation: str | None,
    status: str,
    gate_passed: bool,
    validation_summary_payload: dict[str, Any] | None,
    checks: pd.DataFrame,
    predictions: pd.DataFrame | None,
    feature_columns: list[str],
) -> dict[str, Any]:
    validation_summary_payload = validation_summary_payload or {}
    failed_blockers = checks.loc[
        (checks["severity"] == "blocker") & (checks["status"] == "failed"), "check_name"
    ].tolist()
    known_limitations = []
    if recommendation == "provisional_promote":
        known_limitations.append(
            "Model is provisionally promoted because validation marked the current baseline comparison as reference-only/in-sample."
        )
    if not validation_summary_payload.get("baseline_is_fair_comparator", False):
        known_limitations.append(
            "Baseline comparison is not a strict fair comparator because baseline predictions are full-data/in-sample."
        )
    if failed_blockers:
        known_limitations.extend(f"Blocked by quality check: {check}" for check in failed_blockers)
    return {
        "metric": METRIC,
        "model_version": MODEL_VERSION,
        "selected_model_candidate": selected_model,
        "validation_selected_model": validation_selected,
        "validation_recommendation": recommendation,
        "promotion_status": status,
        "promotion_gate_passed": bool(gate_passed),
        "baseline_is_fair_comparator": bool(
            validation_summary_payload.get("baseline_is_fair_comparator")
        ),
        "strict_promotion_comparison_enabled": bool(
            validation_summary_payload.get("strict_promotion_comparison_enabled")
        ),
        "known_limitations": known_limitations,
        "row_count": int(len(predictions)) if predictions is not None else 0,
        "prediction_summary": prediction_summary(predictions),
        "governance_summary": {
            "status": "passed" if not failed_blockers else "failed",
            "selected_feature_count": len(feature_columns),
            "selected_features": feature_columns,
            "forbidden_features_used": sorted(set(feature_columns).intersection(REFERENCE_COLUMNS)),
            "failed_blocker_checks": failed_blockers,
        },
        "validation_metric_deltas": validation_metric_deltas(validation_summary_payload),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "feature_path": paths.feature_path,
            "selected_model": paths.selected_model,
            "selected_model_metadata": paths.selected_model_metadata,
            "feature_contract": paths.feature_contract,
            "validation_summary": paths.validation_summary,
            "promotion_recommendation": paths.promotion_recommendation,
            "baseline_vs_diagnostic_metrics": paths.baseline_vs_diagnostic_metrics,
            "baseline_predictions": _existing_path(
                paths.baseline_predictions,
                paths.legacy_baseline_predictions,
            ),
        },
        "outputs": {
            "action_predictions": outputs.action_predictions,
            "player_cxa_summary": outputs.player_cxa_summary_csv,
            "team_cxa_summary": outputs.team_cxa_summary_csv,
            "sequence_cxa_summary": outputs.sequence_cxa_summary_csv,
            "model_promotion_summary": outputs.model_promotion_summary,
            "prediction_quality_checks": outputs.prediction_quality_checks,
            "cxa_results_report": outputs.cxa_results_report,
        },
    }


def _results_report(
    *,
    summary: dict[str, Any],
    checks: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> str:
    failed = checks.loc[checks["status"] == "failed", "check_name"].tolist()
    warnings = checks.loc[checks["status"] == "warning", "check_name"].tolist()
    row_count = 0 if predictions is None else len(predictions)
    return "\n".join(
        [
            "# Diagnostic CxA Results Report",
            "",
            "## Executive summary",
            "- This PR generates result outputs; it does not train or validate a model.",
            f"- Selected diagnostic model: `{summary.get('selected_model_candidate')}`.",
            f"- Promotion status: `{summary.get('promotion_status')}`.",
            "- `diagnostic_cxa` is the model-estimated probability that an action creates a shot.",
            "",
            "## Promotion status",
            f"- Validation recommendation: `{summary.get('validation_recommendation')}`.",
            f"- Promotion gate passed: `{summary.get('promotion_gate_passed')}`.",
            "",
            "## Validation basis",
            "- The model is provisionally promoted when validation recommendation is `provisional_promote`.",
            "- The current CxA baseline comparison is reference-only because baseline predictions are full-data/in-sample while diagnostic predictions are out-of-fold.",
            "",
            "## Known caveat about in-sample baseline",
            "- Strict promotion comparison is disabled when validation marks the baseline as an in-sample comparator.",
            "- Outputs are therefore labelled provisionally promoted rather than fully promoted.",
            "",
            "## Action prediction output",
            f"- Action rows written: {row_count}.",
            "- `created_shot_cxg` and `created_shot_id` may appear only as clearly named reference columns.",
            "",
            "## Player/team/sequence aggregates",
            "- Player, team, and sequence summaries aggregate `diagnostic_cxa` from action-level predictions.",
            "",
            "## Prediction quality checks",
            f"- Failed checks: {', '.join(failed) if failed else 'none'}.",
            f"- Warnings: {', '.join(warnings) if warnings else 'none'}.",
            "",
            "## Model governance",
            "- The feature matrix is built from diagnostic feature contract candidates and selected model metadata.",
            "- `created_shot_cxg`, `cxa_value`, identifiers, prediction outputs, and target columns are not model features.",
            "",
            "## Limitations",
            "- CxA+ and Advanced CxA come later.",
            "- This result layer estimates shot-creation probability, not downstream shot value attribution.",
            "",
            "## Next recommended PR",
            "- Add portfolio/reporting views for provisionally promoted diagnostic CxA outputs, then extend CxA+ attribution when the value method is defined.",
            "",
        ]
    )


def write_blocked_outputs(
    *,
    outputs: CxAResultOutputs,
    paths: CxAResultPaths,
    checks: pd.DataFrame,
    selected_model: str | None,
    validation_selected: str | None,
    recommendation: str | None,
    validation_summary_payload: dict[str, Any] | None,
    feature_columns: list[str] | None = None,
) -> dict[str, Path]:
    remove_full_output_artifacts(outputs)
    summary = promotion_summary(
        paths=paths,
        outputs=outputs,
        selected_model=selected_model,
        validation_selected=validation_selected,
        recommendation=recommendation,
        status="blocked",
        gate_passed=False,
        validation_summary_payload=validation_summary_payload,
        checks=checks,
        predictions=None,
        feature_columns=feature_columns or [],
    )
    _write_json(outputs.model_promotion_summary, summary)
    checks.to_csv(outputs.prediction_quality_checks, index=False)
    outputs.cxa_results_report.write_text(
        _results_report(summary=summary, checks=checks, predictions=None),
        encoding="utf-8",
    )
    return {
        "model_promotion_summary": outputs.model_promotion_summary,
        "prediction_quality_checks": outputs.prediction_quality_checks,
        "cxa_results_report": outputs.cxa_results_report,
    }


def remove_full_output_artifacts(outputs: CxAResultOutputs) -> None:
    """Remove stale scoring outputs before writing a blocked governance result."""

    for path in outputs.full_output_artifacts:
        if path.exists():
            path.unlink()


def generate_cxa_diagnostic_results(paths: CxAResultPaths) -> dict[str, Path]:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = output_paths(paths.output_dir)

    metadata = (
        _read_json(paths.selected_model_metadata) if paths.selected_model_metadata.exists() else {}
    )
    validation_summary_payload = (
        _read_json(paths.validation_summary) if paths.validation_summary.exists() else {}
    )
    recommendation_payload = (
        _read_json(paths.promotion_recommendation)
        if paths.promotion_recommendation.exists()
        else {}
    )
    recommendation = recommendation_payload.get("recommendation")
    selected_model = selected_model_name(metadata) if metadata else None
    validation_selected = (
        validation_selected_model(validation_summary_payload)
        if validation_summary_payload
        else None
    )
    status, prediction_source, gate_candidate = promotion_status(str(recommendation))
    preflight = preflight_checks(
        paths,
        recommendation=str(recommendation) if recommendation is not None else None,
        current_selected_model=selected_model,
        validation_selected=validation_selected,
        validation_summary=validation_summary_payload,
    )
    if has_blockers(preflight):
        return write_blocked_outputs(
            outputs=outputs,
            paths=paths,
            checks=preflight,
            selected_model=selected_model,
            validation_selected=validation_selected,
            recommendation=str(recommendation) if recommendation is not None else None,
            validation_summary_payload=validation_summary_payload,
        )

    contract = _read_json(paths.feature_contract)
    feature_frame = _read_table(paths.feature_path)
    feature_columns = resolve_feature_columns(feature_frame, metadata, contract)
    model = joblib.load(paths.selected_model)
    predictions = build_action_predictions(
        feature_frame,
        model,
        feature_columns,
        selected_model=str(selected_model),
        recommendation=str(recommendation),
        status=status,
        prediction_source=prediction_source,
    )
    checks = prediction_quality_checks(predictions, preflight=preflight)
    if has_blockers(checks) or not gate_candidate:
        return write_blocked_outputs(
            outputs=outputs,
            paths=paths,
            checks=checks,
            selected_model=selected_model,
            validation_selected=validation_selected,
            recommendation=str(recommendation),
            validation_summary_payload=validation_summary_payload,
            feature_columns=feature_columns,
        )

    players = player_summary(predictions)
    teams = team_summary(predictions)
    sequences = sequence_summary(predictions)
    validation_metrics = (
        pd.read_csv(paths.baseline_vs_diagnostic_metrics)
        if paths.baseline_vs_diagnostic_metrics.exists()
        else pd.DataFrame()
    )
    baseline_summary = baseline_vs_diagnostic_summary(
        validation_metrics,
        validation_summary_payload,
    )
    summary = promotion_summary(
        paths=paths,
        outputs=outputs,
        selected_model=selected_model,
        validation_selected=validation_selected,
        recommendation=str(recommendation),
        status=status,
        gate_passed=True,
        validation_summary_payload=validation_summary_payload,
        checks=checks,
        predictions=predictions,
        feature_columns=feature_columns,
    )

    predictions.to_parquet(outputs.action_predictions, index=False)
    players.to_csv(outputs.player_cxa_summary_csv, index=False)
    players.to_parquet(outputs.player_cxa_summary_parquet, index=False)
    teams.to_csv(outputs.team_cxa_summary_csv, index=False)
    teams.to_parquet(outputs.team_cxa_summary_parquet, index=False)
    sequences.to_csv(outputs.sequence_cxa_summary_csv, index=False)
    sequences.to_parquet(outputs.sequence_cxa_summary_parquet, index=False)
    players.head(25).to_csv(outputs.top_players_by_cxa, index=False)
    teams.to_csv(outputs.team_cxa_rankings, index=False)
    baseline_summary.to_csv(outputs.baseline_vs_diagnostic_summary, index=False)
    _write_json(outputs.model_promotion_summary, summary)
    checks.to_csv(outputs.prediction_quality_checks, index=False)
    outputs.cxa_results_report.write_text(
        _results_report(summary=summary, checks=checks, predictions=predictions),
        encoding="utf-8",
    )

    return {
        "action_predictions": outputs.action_predictions,
        "player_cxa_summary_csv": outputs.player_cxa_summary_csv,
        "player_cxa_summary_parquet": outputs.player_cxa_summary_parquet,
        "team_cxa_summary_csv": outputs.team_cxa_summary_csv,
        "team_cxa_summary_parquet": outputs.team_cxa_summary_parquet,
        "sequence_cxa_summary_csv": outputs.sequence_cxa_summary_csv,
        "sequence_cxa_summary_parquet": outputs.sequence_cxa_summary_parquet,
        "top_players_by_cxa": outputs.top_players_by_cxa,
        "team_cxa_rankings": outputs.team_cxa_rankings,
        "baseline_vs_diagnostic_summary": outputs.baseline_vs_diagnostic_summary,
        "model_promotion_summary": outputs.model_promotion_summary,
        "prediction_quality_checks": outputs.prediction_quality_checks,
        "cxa_results_report": outputs.cxa_results_report,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = CxAResultPaths.from_roots(
        feature_path=args.feature_path,
        diagnostic_dir=args.diagnostic_dir,
        validation_dir=args.validation_dir,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
    )
    outputs = generate_cxa_diagnostic_results(paths)
    print("Generated diagnostic CxA result outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
