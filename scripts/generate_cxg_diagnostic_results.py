#!/usr/bin/env python
"""Generate final diagnostic-informed CxG result outputs."""

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
    from scripts.run_cxg_diagnostic_training import load_raw_and_modeling_features
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_cxg_diagnostic_training import load_raw_and_modeling_features

DEFAULT_DIAGNOSTIC_DIR = Path("outputs/modeling/cxg/diagnostic_v1")
DEFAULT_VALIDATION_DIR = Path("outputs/validation/cxg/diagnostic_v1")
DEFAULT_BASELINE_DIR = Path("outputs/modeling/cxg/baseline")
DEFAULT_OUTPUT_DIR = Path("outputs/results/cxg/diagnostic_v1")
MODEL_VERSION = "diagnostic_v1"
ALLOWED_PROMOTION_RECOMMENDATIONS = {"promote", "provisional_promote"}
BLOCKING_QUALITY_CHECKS = {
    "row_count",
    "prediction_null_count",
    "outside_0_1_count",
    "missing_shot_id_count",
    "model_loaded",
    "governance_artifacts_present",
}
CONTEXT_COLUMNS = (
    "shot_id",
    "event_id",
    "match_id",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "opponent_team_id",
    "opponent_team_name",
    "is_goal",
    "shot_distance",
    "shot_angle",
    "body_part",
    "technique",
    "shot_type",
    "play_pattern",
    "under_pressure",
    "pressure_state",
    "minute",
    "minute_bucket",
    "minute_bucket_label",
    "score_state",
    "def_label",
)


@dataclass(frozen=True)
class ResultPaths:
    """Input and output paths for diagnostic CxG result generation."""

    selected_model: Path
    selected_model_metadata: Path
    feature_contract: Path
    resolved_features: Path
    feature_group_summary: Path
    excluded_columns: Path
    validation_summary: Path
    model_comparison_validation: Path
    fold_stability: Path
    calibration_bins: Path
    slice_calibration: Path
    promotion_recommendation: Path
    validation_report: Path
    baseline_predictions: Path
    baseline_player: Path
    baseline_team: Path
    baseline_metrics: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
        validation_dir: Path = DEFAULT_VALIDATION_DIR,
        baseline_dir: Path = DEFAULT_BASELINE_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "ResultPaths":
        return cls(
            selected_model=diagnostic_dir / "models" / "selected_model.joblib",
            selected_model_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            resolved_features=diagnostic_dir / "diagnostics" / "resolved_features.json",
            feature_group_summary=diagnostic_dir / "diagnostics" / "feature_group_summary.csv",
            excluded_columns=diagnostic_dir / "diagnostics" / "excluded_columns.csv",
            validation_summary=validation_dir / "validation_summary.json",
            model_comparison_validation=validation_dir / "model_comparison_validation.csv",
            fold_stability=validation_dir / "fold_stability.csv",
            calibration_bins=validation_dir / "calibration_bins.csv",
            slice_calibration=validation_dir / "slice_calibration.csv",
            promotion_recommendation=validation_dir / "promotion_recommendation.json",
            validation_report=validation_dir / "validation_report.md",
            baseline_predictions=baseline_dir / "predictions" / "shot_predictions.parquet",
            baseline_player=baseline_dir / "aggregates" / "player_cxg.parquet",
            baseline_team=baseline_dir / "aggregates" / "team_cxg.parquet",
            baseline_metrics=baseline_dir / "reports" / "metrics.json",
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
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if pd.isna(value) and not isinstance(value, bool | str):
        return None
    return value


def selected_model_name(metadata: dict[str, Any]) -> str:
    selected = metadata.get("selected_model")
    if not selected:
        raise ValueError("Selected model metadata does not include selected_model")
    return str(selected)


def selected_feature_columns(metadata: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    """Return selected model features exactly as recorded by training metadata."""

    selected = selected_model_name(metadata)
    candidates = metadata.get("model_candidates", [])
    selected_candidate = next(
        (candidate for candidate in candidates if candidate.get("name") == selected),
        None,
    )
    feature_groups = (
        selected_candidate.get("features", {}) if isinstance(selected_candidate, dict) else {}
    )
    if not feature_groups:
        feature_groups = metadata.get("resolved_features", {}).get("training_features", {})
    grouped = {
        group: [str(column) for column in feature_groups.get(group, [])]
        for group in ("numeric", "binary", "categorical")
    }
    features = grouped["numeric"] + grouped["binary"] + grouped["categorical"]
    if not features:
        raise ValueError("Selected model metadata does not contain selected feature columns")
    return list(dict.fromkeys(features)), grouped


def governance_summary(
    paths: ResultPaths,
    metadata: dict[str, Any],
    contract: dict[str, Any],
    resolved_features: dict[str, Any] | None,
    feature_group_summary: pd.DataFrame | None,
    excluded_columns: pd.DataFrame | None,
) -> dict[str, Any]:
    """Summarise feature-governance state for the promoted model gate."""

    required_paths = [paths.resolved_features, paths.feature_group_summary, paths.excluded_columns]
    missing = [str(path) for path in required_paths if not path.exists()]
    selected_features, grouped = selected_feature_columns(metadata)
    forbidden = set(contract.get("reference_only_columns", []))
    forbidden.update(contract.get("excluded_leakage_columns", []))
    if excluded_columns is not None and "column" in excluded_columns.columns:
        forbidden.update(excluded_columns["column"].dropna().astype(str))
    synthetic_excluded = _flatten_feature_groups(
        (resolved_features or {}).get("synthetic_default_excluded", {})
    )
    forbidden_used = sorted(set(selected_features).intersection(forbidden))
    synthetic_excluded_used = sorted(set(selected_features).intersection(synthetic_excluded))
    source_available = (resolved_features or {}).get("source_available", {})
    synthetic_defaults = (resolved_features or {}).get("synthetic_default_features", {})
    status = "passed"
    if missing:
        status = "failed"
    if forbidden_used or synthetic_excluded_used:
        status = "failed"
    return {
        "status": status,
        "missing_governance_artifacts": missing,
        "selected_features": grouped,
        "selected_feature_count": len(selected_features),
        "forbidden_features_used": forbidden_used,
        "synthetic_default_excluded_features_used": synthetic_excluded_used,
        "source_available": source_available,
        "synthetic_default_features": synthetic_defaults,
        "synthetic_default_excluded": (resolved_features or {}).get(
            "synthetic_default_excluded", {}
        ),
        "feature_group_rows": (
            0 if feature_group_summary is None else int(len(feature_group_summary))
        ),
    }


def _flatten_feature_groups(payload: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    if isinstance(payload, dict):
        for columns in payload.values():
            values.update(str(column) for column in columns)
    return values


def promotion_status(recommendation: str, *, allow_non_promoted: bool) -> tuple[str, bool]:
    if recommendation == "promote":
        return "promoted", True
    if recommendation == "provisional_promote":
        return "provisionally_promoted", True
    if allow_non_promoted:
        return "exploratory", False
    return "blocked", False


def validation_selected_model(paths: ResultPaths) -> str | None:
    """Return the diagnostic model name that validation evaluated, when available."""

    if not paths.validation_summary.exists():
        return None
    validation_summary = _read_json(paths.validation_summary)
    selected = validation_summary.get("selected_diagnostic_model")
    return str(selected) if selected else None


def blocking_reasons(
    *,
    selected_model: str,
    validation_selected: str | None,
    recommendation: str,
    governance: dict[str, Any],
    allow_non_promoted: bool,
) -> list[str]:
    """Return promotion/result-generation blockers that cannot be bypassed."""

    reasons: list[str] = []
    if validation_selected != selected_model:
        reasons.append(
            "Validation selected model was missing or stale, so promoted outputs were blocked."
        )
    if governance.get("status") != "passed":
        reasons.append("Feature governance failed; scoring outputs are blocked.")
    if recommendation not in ALLOWED_PROMOTION_RECOMMENDATIONS and not allow_non_promoted:
        reasons.append(
            f"Validation recommendation `{recommendation}` is not allowed for promoted outputs."
        )
    return reasons


def validation_model_state(
    selected_model: str,
    validation_selected: str | None,
) -> dict[str, Any]:
    matches = validation_selected == selected_model
    return {
        "validation_model_matches_selected": matches,
        "validation_selected_model": validation_selected,
        "current_selected_model": selected_model,
        "stale_validation_detected": not matches,
    }


def generate_cxg_diagnostic_results(
    *,
    input_path: Path | None = None,
    paths: ResultPaths | None = None,
    allow_non_promoted: bool = False,
) -> dict[str, Path]:
    """Generate diagnostic CxG results and write result artifacts."""

    paths = paths or ResultPaths.from_roots()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = paths.output_dir / "model_promotion_summary.json"
    report_path = paths.output_dir / "cxg_results_report.md"

    metadata = _read_json(paths.selected_model_metadata)
    contract = _read_json(paths.feature_contract)
    recommendation_payload = _read_json(paths.promotion_recommendation)
    recommendation = str(recommendation_payload.get("recommendation", "unknown"))
    selected_model = selected_model_name(metadata)
    validated_model = validation_selected_model(paths)
    validation_state = validation_model_state(selected_model, validated_model)
    selected_features, _ = selected_feature_columns(metadata)
    resolved_features = (
        _read_json(paths.resolved_features) if paths.resolved_features.exists() else {}
    )
    feature_group_summary = (
        pd.read_csv(paths.feature_group_summary)
        if paths.feature_group_summary.exists()
        else pd.DataFrame()
    )
    excluded_columns = (
        pd.read_csv(paths.excluded_columns) if paths.excluded_columns.exists() else pd.DataFrame()
    )
    governance = governance_summary(
        paths,
        metadata,
        contract,
        resolved_features,
        feature_group_summary,
        excluded_columns,
    )
    blockers = blocking_reasons(
        selected_model=selected_model,
        validation_selected=validated_model,
        recommendation=recommendation,
        governance=governance,
        allow_non_promoted=allow_non_promoted,
    )
    status, gate_passed = promotion_status(
        recommendation,
        allow_non_promoted=allow_non_promoted,
    )
    governance_failed = governance.get("status") != "passed"
    stale_validation = bool(validation_state["stale_validation_detected"])
    validation_rejected = recommendation not in ALLOWED_PROMOTION_RECOMMENDATIONS
    if allow_non_promoted and stale_validation and not governance_failed:
        status = "exploratory"
        gate_passed = False
    elif blockers:
        status = "blocked"
        gate_passed = False
    else:
        gate_passed = gate_passed and not validation_rejected

    output_paths = _output_paths(paths.output_dir)
    if status == "blocked":
        summary = _promotion_summary(
            paths=paths,
            selected_model=selected_model,
            recommendation=recommendation,
            status=status,
            gate_passed=False,
            governance=governance,
            quality_checks={},
            validation_payloads=_validation_payloads(paths),
            validation_state=validation_state,
            output_paths={"model_promotion_summary": summary_path, "results_report": report_path},
            known_limitations=_blocked_limitations(
                recommendation,
                governance,
                blockers=blockers,
            ),
        )
        _write_json(summary_path, summary)
        report_path.write_text(_results_report(summary, blocked=True), encoding="utf-8")
        return {"model_promotion_summary": summary_path, "results_report": report_path}

    model = joblib.load(paths.selected_model)
    frame, resolved_input, _ = load_raw_and_modeling_features(input_path)
    missing_features = [column for column in selected_features if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Selected model features missing from input frame: {missing_features}")
    feature_matrix = frame[selected_features]
    probs = _predict_probabilities(model, feature_matrix)
    shots = build_shot_predictions(
        frame,
        probs,
        selected_model=selected_model,
        promotion_status_value=status,
        validation_recommendation=recommendation,
        prediction_source=prediction_source_for_status(status),
    )
    shots, baseline_join = join_baseline_predictions(shots, paths.baseline_predictions)
    player_summary = build_entity_summary(shots, "player")
    team_summary = build_entity_summary(shots, "team")
    quality = prediction_quality_checks(
        shots,
        baseline_join_rate=baseline_join["join_rate"],
        model_loaded=True,
        promotion_gate_passed=gate_passed,
        governance_artifacts_present=governance["status"] == "passed",
        validation_recommendation=recommendation,
    )
    quality_payload = _quality_payload(quality)
    quality_blockers = blocking_quality_failures(quality)
    if status in {"promoted", "provisionally_promoted"} and has_blocking_quality_failures(quality):
        status = "blocked"
        gate_passed = False
        summary = _promotion_summary(
            paths=paths,
            selected_model=selected_model,
            recommendation=recommendation,
            status=status,
            gate_passed=False,
            governance=governance,
            quality_checks=quality_payload,
            validation_payloads=_validation_payloads(paths),
            validation_state=validation_state,
            output_paths={"model_promotion_summary": summary_path, "results_report": report_path},
            input_path=resolved_input,
            baseline_join=baseline_join,
            known_limitations=_known_limitations_for_quality_blockers(quality_blockers),
        )
        _write_json(summary_path, summary)
        report_path.write_text(_results_report(summary, blocked=True), encoding="utf-8")
        return {"model_promotion_summary": summary_path, "results_report": report_path}

    baseline_summary = baseline_vs_diagnostic_summary(shots, baseline_join)

    _write_result_tables(
        output_paths, shots, player_summary, team_summary, quality, baseline_summary
    )
    summary = _promotion_summary(
        paths=paths,
        selected_model=selected_model,
        recommendation=recommendation,
        status=status,
        gate_passed=gate_passed,
        governance=governance,
        quality_checks=quality_payload,
        validation_payloads=_validation_payloads(paths),
        validation_state=validation_state,
        output_paths=output_paths,
        input_path=resolved_input,
        baseline_join=baseline_join,
        known_limitations=_known_limitations(
            status,
            recommendation,
            baseline_join,
            governance,
            validation_state=validation_state,
        ),
    )
    _write_json(output_paths["model_promotion_summary"], summary)
    output_paths["results_report"].write_text(_results_report(summary), encoding="utf-8")
    return output_paths


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "shot_predictions": output_dir / "shot_predictions.parquet",
        "player_cxg_summary_csv": output_dir / "player_cxg_summary.csv",
        "team_cxg_summary_csv": output_dir / "team_cxg_summary.csv",
        "player_cxg_summary_parquet": output_dir / "player_cxg_summary.parquet",
        "team_cxg_summary_parquet": output_dir / "team_cxg_summary.parquet",
        "top_players_by_cxg": output_dir / "top_players_by_cxg.csv",
        "team_cxg_rankings": output_dir / "team_cxg_rankings.csv",
        "baseline_vs_diagnostic_summary": output_dir / "baseline_vs_diagnostic_summary.csv",
        "model_promotion_summary": output_dir / "model_promotion_summary.json",
        "prediction_quality_checks": output_dir / "prediction_quality_checks.csv",
        "results_report": output_dir / "cxg_results_report.md",
    }


def _predict_probabilities(model: Any, feature_matrix: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_matrix)[:, 1]
    else:
        probs = model.predict(feature_matrix)
    return np.asarray(probs, dtype=float)


def build_shot_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    selected_model: str,
    promotion_status_value: str,
    validation_recommendation: str,
    prediction_source: str,
) -> pd.DataFrame:
    prediction = pd.DataFrame(index=frame.index)
    for column in CONTEXT_COLUMNS:
        if column in frame.columns:
            prediction[column] = frame[column].to_numpy()
    prediction["predicted_cxg"] = probabilities
    prediction["model_version"] = MODEL_VERSION
    prediction["selected_model_candidate"] = selected_model
    prediction["prediction_source"] = prediction_source
    prediction["promotion_status"] = promotion_status_value
    prediction["validation_recommendation"] = validation_recommendation
    return prediction.reset_index(drop=True)


def prediction_source_for_status(status: str) -> str:
    if status in {"promoted", "provisionally_promoted"}:
        return "promoted_model"
    if status == "exploratory":
        return "exploratory_model"
    raise ValueError(f"No shot prediction source is defined for promotion status `{status}`.")


def join_baseline_predictions(
    shots: pd.DataFrame,
    baseline_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shots = shots.copy()
    shots["baseline_cxg"] = np.nan
    shots["cxg_delta_vs_baseline"] = np.nan
    if not baseline_path.exists():
        return shots, {"status": "missing", "join_key": None, "join_rate": 0.0}
    baseline = pd.read_parquet(baseline_path)
    if "cxg_raw" not in baseline.columns:
        return shots, {"status": "missing_probability", "join_key": None, "join_rate": 0.0}
    join_columns = _baseline_join_columns(shots, baseline)
    if not join_columns:
        return shots, {"status": "no_join_key", "join_key": None, "join_rate": 0.0}
    baseline_subset = baseline[join_columns + ["cxg_raw"]].drop_duplicates(join_columns)
    merged = shots.drop(columns=["baseline_cxg", "cxg_delta_vs_baseline"]).merge(
        baseline_subset.rename(columns={"cxg_raw": "baseline_cxg"}),
        on=join_columns,
        how="left",
    )
    merged["cxg_delta_vs_baseline"] = merged["predicted_cxg"] - merged["baseline_cxg"]
    join_rate = float(merged["baseline_cxg"].notna().mean()) if len(merged) else 0.0
    return merged, {"status": "joined", "join_key": join_columns, "join_rate": join_rate}


def _baseline_join_columns(shots: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    for column in ("shot_id", "event_id"):
        if column in shots.columns and column in baseline.columns:
            return [column]
    fallback = ["match_id", "player_id", "minute", "shot_distance", "shot_angle"]
    if all(column in shots.columns and column in baseline.columns for column in fallback):
        return fallback
    return []


def build_entity_summary(shots: pd.DataFrame, entity: str) -> pd.DataFrame:
    entity_id = f"{entity}_id"
    entity_name = f"{entity}_name"
    if entity_id not in shots.columns:
        return pd.DataFrame()
    group_columns = [entity_id]
    for optional in (entity_name, "team_id", "team_name"):
        if optional in shots.columns and optional not in group_columns:
            group_columns.append(optional)
    summary = (
        shots.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            shots=("predicted_cxg", "size"),
            goals=("is_goal", "sum") if "is_goal" in shots.columns else ("predicted_cxg", "size"),
            total_cxg=("predicted_cxg", "sum"),
            mean_cxg_per_shot=("predicted_cxg", "mean"),
            baseline_total_cxg=("baseline_cxg", lambda values: values.sum(min_count=1)),
        )
        .sort_values(["total_cxg", "shots"], ascending=False)
        .reset_index(drop=True)
    )
    if "is_goal" not in shots.columns:
        summary["goals"] = np.nan
    summary["goals_minus_cxg"] = summary["goals"] - summary["total_cxg"]
    if shots["baseline_cxg"].notna().any():
        summary["total_cxg_delta_vs_baseline"] = (
            summary["total_cxg"] - summary["baseline_total_cxg"]
        )
    else:
        summary["baseline_total_cxg"] = np.nan
        summary["total_cxg_delta_vs_baseline"] = np.nan
    summary["rank_total_cxg"] = summary["total_cxg"].rank(method="min", ascending=False).astype(int)
    summary["rank_mean_cxg_per_shot"] = (
        summary["mean_cxg_per_shot"].rank(method="min", ascending=False).astype(int)
    )
    if entity == "player":
        summary["cxg_per_90"] = np.nan
    ordered = _entity_column_order(entity, summary.columns)
    return summary[ordered].sort_values(["total_cxg", "shots"], ascending=False)


def _entity_column_order(entity: str, columns: pd.Index) -> list[str]:
    base = list(
        dict.fromkeys(
            [
                f"{entity}_id",
                f"{entity}_name",
                "team_id",
                "team_name",
                "shots",
                "goals",
                "total_cxg",
                "mean_cxg_per_shot",
                "goals_minus_cxg",
                "cxg_per_90",
                "baseline_total_cxg",
                "total_cxg_delta_vs_baseline",
                "rank_total_cxg",
                "rank_mean_cxg_per_shot",
            ]
        )
    )
    return [column for column in base if column in columns] + [
        column for column in columns if column not in base
    ]


def prediction_quality_checks(
    shots: pd.DataFrame,
    *,
    baseline_join_rate: float,
    model_loaded: bool,
    promotion_gate_passed: bool,
    governance_artifacts_present: bool,
    validation_recommendation: str,
) -> pd.DataFrame:
    # Per-column ID checks — shot_id is required; others are preserved if available.
    missing_shot_id_count = (
        int(shots["shot_id"].isna().sum()) if "shot_id" in shots.columns else len(shots)
    )
    missing_match_id_count = (
        int(shots["match_id"].isna().sum()) if "match_id" in shots.columns else 0
    )
    missing_team_id_count = int(shots["team_id"].isna().sum()) if "team_id" in shots.columns else 0
    missing_player_id_count = (
        int(shots["player_id"].isna().sum()) if "player_id" in shots.columns else 0
    )
    # event_id is optional — absent when the prediction source is the feature store
    # rather than a DB-backed scored predictions table.
    missing_event_id_count = (
        int(shots["event_id"].isna().sum()) if "event_id" in shots.columns else len(shots)
    )
    duplicate_shot_count = (
        int(shots["shot_id"].duplicated().sum()) if "shot_id" in shots.columns else 0
    )
    probabilities = (
        shots["predicted_cxg"] if "predicted_cxg" in shots.columns else pd.Series(dtype=float)
    )
    rows = [
        _check("row_count", len(shots), "passed" if len(shots) > 0 else "failed", "Rows scored."),
        _check(
            "prediction_null_count",
            int(probabilities.isna().sum()),
            "passed" if int(probabilities.isna().sum()) == 0 else "failed",
            "Predicted probabilities should be non-null.",
        ),
        _check("probability_min", probabilities.min(), "passed", "Minimum predicted probability."),
        _check("probability_max", probabilities.max(), "passed", "Maximum predicted probability."),
        _check(
            "outside_0_1_count",
            int(((probabilities < 0) | (probabilities > 1)).sum()),
            "passed" if int(((probabilities < 0) | (probabilities > 1)).sum()) == 0 else "failed",
            "Predicted probabilities should be within [0, 1].",
        ),
        _check(
            "target_null_count",
            int(shots["is_goal"].isna().sum()) if "is_goal" in shots.columns else len(shots),
            (
                "passed"
                if "is_goal" in shots.columns and int(shots["is_goal"].isna().sum()) == 0
                else "warning"
            ),
            "Goal target is retained for audit.",
        ),
        _check(
            "missing_shot_id_count",
            missing_shot_id_count,
            "passed" if missing_shot_id_count == 0 else "failed",
            "shot_id is required for all scored rows.",
        ),
        _check(
            "missing_match_id_count",
            missing_match_id_count,
            "passed" if missing_match_id_count == 0 else "warning",
            "match_id should be present for all scored rows.",
        ),
        _check(
            "missing_team_id_count",
            missing_team_id_count,
            "passed" if missing_team_id_count == 0 else "warning",
            "team_id should be present when available from the feature source.",
        ),
        _check(
            "missing_player_id_count",
            missing_player_id_count,
            "passed" if missing_player_id_count == 0 else "warning",
            "player_id should be present when available from the feature source.",
        ),
        _check(
            "missing_event_id_count",
            missing_event_id_count,
            "info",
            "event_id is optional; absent when the prediction source is the feature store.",
        ),
        _check(
            "duplicate_shot_id_count",
            duplicate_shot_count,
            "passed" if duplicate_shot_count == 0 else "warning",
            "Duplicate shot identifiers are not expected.",
        ),
        _check(
            "baseline_join_rate",
            baseline_join_rate,
            "passed" if baseline_join_rate >= 0.95 else "warning",
            "Share of diagnostic predictions joined to baseline CxG.",
        ),
        _check(
            "model_loaded",
            model_loaded,
            "passed" if model_loaded else "failed",
            "Model load state.",
        ),
        _check(
            "promotion_gate_passed",
            promotion_gate_passed,
            "passed" if promotion_gate_passed else "warning",
            "Validation promotion gate state.",
        ),
        _check(
            "governance_artifacts_present",
            governance_artifacts_present,
            "passed" if governance_artifacts_present else "failed",
            "Required feature-governance artifacts are present and passed.",
        ),
        _check(
            "validation_recommendation",
            validation_recommendation,
            (
                "passed"
                if validation_recommendation in ALLOWED_PROMOTION_RECOMMENDATIONS
                else "warning"
            ),
            "Validation recommendation controlling promotion.",
        ),
    ]
    return pd.DataFrame(rows, columns=["check_name", "check_value", "status", "details"])


def _check(name: str, value: Any, status: str, details: str) -> dict[str, Any]:
    return {"check_name": name, "check_value": value, "status": status, "details": details}


def baseline_vs_diagnostic_summary(
    shots: pd.DataFrame,
    baseline_join: dict[str, Any],
) -> pd.DataFrame:
    if not shots["baseline_cxg"].notna().any():
        return pd.DataFrame(
            [
                {
                    "metric": "baseline_join_rate",
                    "value": baseline_join["join_rate"],
                    "status": baseline_join["status"],
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "metric": "baseline_join_rate",
                "value": baseline_join["join_rate"],
                "status": "joined",
            },
            {
                "metric": "diagnostic_total_cxg",
                "value": shots["predicted_cxg"].sum(),
                "status": "computed",
            },
            {
                "metric": "baseline_total_cxg",
                "value": shots["baseline_cxg"].sum(),
                "status": "computed",
            },
            {
                "metric": "total_cxg_delta_vs_baseline",
                "value": shots["cxg_delta_vs_baseline"].sum(),
                "status": "computed",
            },
        ]
    )


def _write_result_tables(
    output_paths: dict[str, Path],
    shots: pd.DataFrame,
    player_summary: pd.DataFrame,
    team_summary: pd.DataFrame,
    quality: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> None:
    shots.to_parquet(output_paths["shot_predictions"], index=False)
    player_summary.to_csv(output_paths["player_cxg_summary_csv"], index=False)
    team_summary.to_csv(output_paths["team_cxg_summary_csv"], index=False)
    player_summary.to_parquet(output_paths["player_cxg_summary_parquet"], index=False)
    team_summary.to_parquet(output_paths["team_cxg_summary_parquet"], index=False)
    player_summary.head(25).to_csv(output_paths["top_players_by_cxg"], index=False)
    team_summary.to_csv(output_paths["team_cxg_rankings"], index=False)
    baseline_summary.to_csv(output_paths["baseline_vs_diagnostic_summary"], index=False)
    quality.to_csv(output_paths["prediction_quality_checks"], index=False)


def blocking_quality_failures(quality: pd.DataFrame) -> list[str]:
    failed = quality.loc[
        quality["check_name"].isin(BLOCKING_QUALITY_CHECKS) & (quality["status"] == "failed"),
        "check_name",
    ]
    return sorted(str(name) for name in failed.tolist())


def has_blocking_quality_failures(quality: pd.DataFrame) -> bool:
    return bool(blocking_quality_failures(quality))


def _known_limitations_for_quality_blockers(failed_checks: list[str]) -> list[str]:
    checks = ", ".join(f"`{name}`" for name in failed_checks)
    return [
        "Promoted/provisionally promoted result outputs were blocked due to failed hard quality checks: "
        + checks
        + "."
    ]


def _quality_payload(quality: pd.DataFrame) -> dict[str, Any]:
    return {
        str(row.check_name): {"value": row.check_value, "status": row.status}
        for row in quality.itertuples()
    }


def _validation_payloads(paths: ResultPaths) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if paths.validation_summary.exists():
        payload["validation_summary"] = _read_json(paths.validation_summary)
    if paths.promotion_recommendation.exists():
        payload["promotion_recommendation"] = _read_json(paths.promotion_recommendation)
    if paths.model_comparison_validation.exists():
        comparison = pd.read_csv(paths.model_comparison_validation)
        payload["model_comparison"] = comparison.to_dict(orient="records")
    return payload


def _promotion_summary(
    *,
    paths: ResultPaths,
    selected_model: str,
    recommendation: str,
    status: str,
    gate_passed: bool,
    governance: dict[str, Any],
    quality_checks: dict[str, Any],
    validation_payloads: dict[str, Any],
    validation_state: dict[str, Any],
    output_paths: dict[str, Path],
    input_path: Path | None = None,
    baseline_join: dict[str, Any] | None = None,
    known_limitations: list[str] | None = None,
) -> dict[str, Any]:
    comparison_rows = validation_payloads.get("model_comparison", [])
    validation_metrics = _validation_metrics(comparison_rows)
    return {
        "model_version": MODEL_VERSION,
        "selected_model_candidate": selected_model,
        "validation_recommendation": recommendation,
        "promotion_status": status,
        "promotion_gate_passed": gate_passed,
        "validation_model_matches_selected": validation_state["validation_model_matches_selected"],
        "validation_selected_model": validation_state["validation_selected_model"],
        "current_selected_model": validation_state["current_selected_model"],
        "stale_validation_detected": validation_state["stale_validation_detected"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "shot_features": str(input_path) if input_path else None,
            "selected_model": str(paths.selected_model),
            "selected_model_metadata": str(paths.selected_model_metadata),
            "feature_contract": str(paths.feature_contract),
            "resolved_features": str(paths.resolved_features),
            "feature_group_summary": str(paths.feature_group_summary),
            "excluded_columns": str(paths.excluded_columns),
            "promotion_recommendation": str(paths.promotion_recommendation),
            "baseline_predictions": str(paths.baseline_predictions),
        },
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "validation_metrics": validation_metrics,
        "baseline_comparison": baseline_join or {"status": "not_generated", "join_rate": 0.0},
        "governance_summary": governance,
        "quality_checks": quality_checks,
        "known_limitations": known_limitations or [],
        "next_steps": _next_steps(status),
    }


def _validation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for row in rows:
        model_version = str(row.get("model_version", "unknown"))
        metrics[model_version] = {
            key: row.get(key)
            for key in (
                "row_count",
                "goal_count",
                "goal_rate",
                "mean_predicted_probability",
                "brier",
                "log_loss",
                "roc_auc",
                "expected_calibration_error",
            )
            if key in row
        }
    return metrics


def _blocked_limitations(
    recommendation: str,
    governance: dict[str, Any],
    *,
    blockers: list[str] | None = None,
) -> list[str]:
    limitations = list(blockers or [])
    if not limitations and recommendation not in ALLOWED_PROMOTION_RECOMMENDATIONS:
        limitations.append(
            f"Validation recommendation `{recommendation}` is not allowed for promoted outputs."
        )
    missing = governance.get("missing_governance_artifacts", [])
    if missing:
        message = "Required governance artifacts are missing: " + ", ".join(missing)
        if message not in limitations:
            limitations.append(message)
    if governance.get("forbidden_features_used"):
        message = "Selected model metadata includes forbidden features: " + ", ".join(
            governance["forbidden_features_used"]
        )
        if message not in limitations:
            limitations.append(message)
    if governance.get("synthetic_default_excluded_features_used"):
        message = (
            "Selected model metadata includes synthetic default features that training "
            "excluded: " + ", ".join(governance["synthetic_default_excluded_features_used"])
        )
        if message not in limitations:
            limitations.append(message)
    return limitations


def _known_limitations(
    status: str,
    recommendation: str,
    baseline_join: dict[str, Any],
    governance: dict[str, Any],
    *,
    validation_state: dict[str, Any],
) -> list[str]:
    limitations: list[str] = []
    if status == "exploratory":
        limitations.append(
            f"Outputs were generated with --allow-non-promoted after validation returned `{recommendation}`."
        )
    if status == "exploratory" and validation_state.get("stale_validation_detected"):
        limitations.append(
            "Exploratory outputs were generated despite stale validation because "
            "--allow-non-promoted was used."
        )
    if baseline_join.get("join_rate", 0.0) < 1.0:
        limitations.append("Baseline comparison is partial or unavailable for some shot rows.")
    synthetic_defaults = _flatten_feature_groups(governance.get("synthetic_default_features", {}))
    if synthetic_defaults:
        limitations.append(
            "Some diagnostic contract features were synthetic defaults during training and remain monitored."
        )
    return limitations


def _next_steps(status: str) -> list[str]:
    if status in {"promoted", "provisionally_promoted"}:
        return [
            "Use these shot, player, and team outputs in the portfolio/dashboard layer.",
            "Continue monitoring calibration and slice behavior as more data is added.",
        ]
    if status == "exploratory":
        return [
            "Treat outputs as exploratory only.",
            "Revise diagnostic training or validation before promoting to a dashboard surface.",
        ]
    return [
        "Do not use diagnostic_v1 as promoted CxG output.",
        "Revise training or validation before regenerating final results.",
    ]


def _results_report(summary: dict[str, Any], *, blocked: bool = False) -> str:
    governance = summary.get("governance_summary", {})
    baseline = summary.get("baseline_comparison", {})
    quality = summary.get("quality_checks", {})
    limitations = summary.get("known_limitations", [])
    quality_lines = [
        f"- `{name}`: {detail.get('status')} ({detail.get('value')})"
        for name, detail in quality.items()
    ] or ["- Quality checks were not generated because promotion was blocked."]
    limitation_lines = [f"- {item}" for item in limitations] or [
        "- No material limitations recorded."
    ]
    return "\n".join(
        [
            "# Diagnostic CxG Results",
            "",
            "## 1. Purpose",
            "",
            "This layer turns the validation-reviewed diagnostic CxG model into final "
            "shot-level predictions and player/team summaries for downstream portfolio "
            "or dashboard use.",
            "",
            "## 2. Promotion Decision",
            "",
            f"- Validation recommendation: `{summary['validation_recommendation']}`",
            f"- Promotion status: `{summary['promotion_status']}`",
            f"- Promotion gate passed: `{summary['promotion_gate_passed']}`",
            "",
            "Promoted outputs are generated only when validation recommends `promote` "
            "or `provisional_promote`. Rejected models can only produce exploratory "
            "outputs when explicitly requested.",
            "",
            "## 3. Inputs Used",
            "",
            f"- Selected model: `{summary['inputs']['selected_model']}`",
            f"- Feature contract: `{summary['inputs']['feature_contract']}`",
            f"- Validation recommendation: `{summary['inputs']['promotion_recommendation']}`",
            f"- Shot features: `{summary['inputs']['shot_features']}`",
            "",
            "## 4. Model and Validation Summary",
            "",
            f"The selected diagnostic candidate is `{summary['selected_model_candidate']}`. "
            "Validation metrics are carried through from the validation layer rather than "
            "recomputed here.",
            "",
            "## 5. Shot-Level Prediction Output",
            "",
            _artifact_sentence(summary, "shot_predictions", blocked),
            "",
            "## 6. Player-Level CxG Summary",
            "",
            _artifact_sentence(summary, "player_cxg_summary_csv", blocked),
            "",
            "## 7. Team-Level CxG Summary",
            "",
            _artifact_sentence(summary, "team_cxg_summary_csv", blocked),
            "",
            "## 8. Baseline vs Diagnostic Differences",
            "",
            f"Baseline join status is `{baseline.get('status')}` with join rate "
            f"{float(baseline.get('join_rate', 0.0)):.3f}. Diagnostic deltas are included "
            "where baseline shot predictions can be joined safely.",
            "",
            "## 9. Prediction Quality Checks",
            "",
            *quality_lines,
            "",
            "## 10. Limitations",
            "",
            *limitation_lines,
            "",
            "## 11. How This Feeds the Portfolio / Dashboard Layer",
            "",
            "The results files provide stable shot, player, and team surfaces that can be "
            "loaded directly by portfolio reporting or dashboard code once the promotion "
            "status is promoted or provisionally promoted.",
            "",
            "## Governance Note",
            "",
            f"Feature governance status is `{governance.get('status')}`. Forbidden features "
            f"used: `{governance.get('forbidden_features_used', [])}`. Synthetic default "
            f"excluded features used: `{governance.get('synthetic_default_excluded_features_used', [])}`.",
        ]
    )


def _artifact_sentence(summary: dict[str, Any], key: str, blocked: bool) -> str:
    if blocked:
        return "Not generated because the promotion gate blocked promoted outputs."
    return f"Generated at `{summary['outputs'].get(key)}`."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Optional CxG shot feature table.")
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-non-promoted",
        action="store_true",
        help="Generate exploratory outputs even when validation did not recommend promotion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ResultPaths.from_roots(
        diagnostic_dir=args.diagnostic_dir,
        validation_dir=args.validation_dir,
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
    )
    outputs = generate_cxg_diagnostic_results(
        input_path=args.input,
        paths=paths,
        allow_non_promoted=args.allow_non_promoted,
    )
    print(f"Wrote diagnostic CxG result artifacts to {paths.output_dir}")
    if "shot_predictions" not in outputs:
        print("Promotion gate blocked promoted prediction outputs.")


if __name__ == "__main__":
    main()
