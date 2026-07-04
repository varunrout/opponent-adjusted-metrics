#!/usr/bin/env python
"""Analyze promoted diagnostic CxG feature impact without retraining."""

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

DEFAULT_FEATURE_PATH = Path("feature_store/cxg/shot_features.parquet")
DEFAULT_DIAGNOSTIC_DIR = Path("outputs/modeling/cxg/diagnostic_v1")
DEFAULT_RESULTS_DIR = Path("outputs/results/cxg/diagnostic_v1")
DEFAULT_OUTPUT_DIR = DEFAULT_DIAGNOSTIC_DIR / "feature_impact"
MODEL_VERSION = "diagnostic_v1"
TARGET_COLUMN = "is_goal"
PREDICTION_COLUMN = "predicted_cxg"
REFERENCE_ONLY_COLUMNS = {"statsbomb_xg"}
FEATURE_ALIASES = {"location_x": "shot_x", "location_y": "shot_y"}
IDENTIFIER_COLUMNS = {
    "shot_id",
    "event_id",
    "match_id",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "opponent_team_id",
    "opponent_team_name",
}
CATEGORY_LIFT_COLUMNS = (
    "body_part",
    "technique",
    "shot_type",
    "play_pattern",
    "set_piece_category",
    "set_piece_phase",
    "pressure_state",
    "score_state",
    "def_label",
    "minute_bucket_label",
)
FOOTBALL_FEATURE_GROUPS = {
    "geometry": [
        "shot_distance",
        "shot_angle",
        "centrality",
        "distance_to_goal_line",
        "location_x",
        "location_y",
    ],
    "shot_execution": ["first_time", "body_part", "technique", "shot_type"],
    "game_state": [
        "score_diff_at_shot",
        "is_leading",
        "is_trailing",
        "is_drawing",
        "score_state",
        "simple_state",
    ],
    "time_context": ["minute", "minute_bucket", "minute_bucket_label"],
    "possession_context": [
        "time_gap_seconds",
        "possession_sequence_length",
        "possession_duration",
        "previous_action_gap",
        "possession_match",
    ],
    "pressure_defensive_context": [
        "under_pressure",
        "recent_def_actions_count",
        "pressure_proxy_score",
        "pressure_state",
        "def_label",
    ],
    "set_piece_play_pattern": ["play_pattern", "set_piece_category", "set_piece_phase"],
    "team_player_identifiers_only": [
        "shot_id",
        "event_id",
        "match_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "opponent_team_id",
        "opponent_team_name",
    ],
}


@dataclass(frozen=True)
class FeatureImpactPaths:
    """Input and output paths for promoted CxG feature-impact analysis."""

    selected_model: Path
    selected_model_metadata: Path
    feature_contract: Path
    resolved_features: Path
    feature_group_summary: Path
    shot_predictions: Path
    model_promotion_summary: Path
    baseline_vs_diagnostic_summary: Path
    output_dir: Path

    @classmethod
    def from_roots(
        cls,
        diagnostic_dir: Path = DEFAULT_DIAGNOSTIC_DIR,
        results_dir: Path = DEFAULT_RESULTS_DIR,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ) -> "FeatureImpactPaths":
        return cls(
            selected_model=diagnostic_dir / "models" / "selected_model.joblib",
            selected_model_metadata=diagnostic_dir / "models" / "selected_model_metadata.json",
            feature_contract=diagnostic_dir / "contracts" / "feature_contract.json",
            resolved_features=diagnostic_dir / "diagnostics" / "resolved_features.json",
            feature_group_summary=diagnostic_dir / "diagnostics" / "feature_group_summary.csv",
            shot_predictions=results_dir / "shot_predictions.parquet",
            model_promotion_summary=results_dir / "model_promotion_summary.json",
            baseline_vs_diagnostic_summary=results_dir / "baseline_vs_diagnostic_summary.csv",
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
        return None if np.isnan(value) else float(value)
    if pd.isna(value) and not isinstance(value, bool | str):
        return None
    return value


def selected_features_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return the selected diagnostic features exactly as training recorded them."""

    features = metadata.get("selected_features")
    if not features:
        groups = metadata.get("selected_feature_groups", {})
        features = (
            list(groups.get("numeric", []))
            + list(groups.get("binary", []))
            + list(groups.get("categorical", []))
        )
    if not features:
        selected = metadata.get("selected_model")
        for candidate in metadata.get("model_candidates", []):
            if candidate.get("name") == selected:
                groups = candidate.get("features", {})
                features = (
                    list(groups.get("numeric", []))
                    + list(groups.get("binary", []))
                    + list(groups.get("categorical", []))
                )
                break
    cleaned = [str(feature) for feature in features or []]
    if not cleaned:
        raise ValueError("selected_model_metadata.json does not include selected features")
    return list(dict.fromkeys(cleaned))


def model_impact_features(selected_features: list[str]) -> list[str]:
    """Return selected features eligible for impact scoring."""

    excluded = IDENTIFIER_COLUMNS | REFERENCE_ONLY_COLUMNS
    return [feature for feature in selected_features if feature not in excluded]


def align_selected_feature_matrix(
    feature_frame: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Build the selected model matrix, applying known source aliases only."""

    matrix_source = feature_frame.copy()
    for expected, alias in FEATURE_ALIASES.items():
        if expected not in matrix_source.columns and alias in matrix_source.columns:
            matrix_source[expected] = matrix_source[alias]
    missing = [feature for feature in selected_features if feature not in matrix_source]
    if missing:
        raise ValueError(f"Selected features missing from feature frame: {missing}")
    return matrix_source[selected_features]


def map_feature_groups(selected_features: list[str]) -> dict[str, list[str]]:
    """Map selected model features to football interpretation groups."""

    selected = set(model_impact_features(selected_features))
    grouped = {
        group: [feature for feature in features if feature in selected]
        for group, features in FOOTBALL_FEATURE_GROUPS.items()
    }
    known = {feature for features in FOOTBALL_FEATURE_GROUPS.values() for feature in features}
    grouped["other_selected_features"] = sorted(selected - known)
    return grouped


def predict_probabilities(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities from a saved model or pipeline."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        probabilities = np.asarray(probabilities)
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
    """Compute safe probability metrics for feature-impact comparisons."""

    y = np.asarray(y_true, dtype=int)
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1 - 1e-15)
    metrics: dict[str, Any] = {
        "row_count": int(len(probs)),
        "goal_count": int(y.sum()),
        "goal_rate": float(y.mean()) if len(y) else np.nan,
        "mean_predicted_probability": float(probs.mean()) if len(probs) else np.nan,
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probs)),
        "roc_auc": np.nan,
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, probs))
    return metrics


def permutation_importance(
    model: Any,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    features: list[str],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute model-agnostic permutation impact for selected features."""

    rng = np.random.default_rng(random_state)
    baseline_probs = predict_probabilities(model, matrix)
    baseline = probability_metrics(y_true, baseline_probs)
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in matrix.columns:
            continue
        repeat_rows = []
        for repeat in range(n_repeats):
            permuted = matrix.copy()
            values = permuted[feature].to_numpy(copy=True)
            rng.shuffle(values)
            permuted[feature] = values
            probs = predict_probabilities(model, permuted)
            metrics = probability_metrics(y_true, probs)
            repeat_rows.append(
                {
                    "feature": feature,
                    "repeat": repeat + 1,
                    "log_loss_delta": metrics["log_loss"] - baseline["log_loss"],
                    "brier_delta": metrics["brier"] - baseline["brier"],
                    "roc_auc_delta": _safe_delta(baseline["roc_auc"], metrics["roc_auc"]),
                    "mean_probability_delta": (
                        metrics["mean_predicted_probability"]
                        - baseline["mean_predicted_probability"]
                    ),
                    "absolute_probability_delta_mean": float(np.abs(probs - baseline_probs).mean()),
                }
            )
        repeat_df = pd.DataFrame(repeat_rows)
        summary = repeat_df.drop(columns=["repeat"]).groupby("feature", as_index=False).mean()
        std = (
            repeat_df.groupby("feature", as_index=False)["log_loss_delta"]
            .std()
            .rename(columns={"log_loss_delta": "log_loss_delta_std"})
        )
        rows.extend(summary.merge(std, on="feature", how="left").to_dict("records"))
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_group",
                "log_loss_delta",
                "brier_delta",
                "roc_auc_delta",
                "mean_probability_delta",
                "absolute_probability_delta_mean",
                "log_loss_delta_std",
                "impact_rank",
            ]
        )
    result = pd.DataFrame(rows)
    result["feature_group"] = result["feature"].map(feature_to_group_map())
    result["impact_rank"] = (
        result["log_loss_delta"].rank(method="first", ascending=False).astype(int)
    )
    return result.sort_values(["impact_rank", "feature"]).reset_index(drop=True)


def _safe_delta(before: Any, after: Any) -> float:
    if pd.isna(before) or pd.isna(after):
        return np.nan
    return float(before - after)


def feature_to_group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, features in FOOTBALL_FEATURE_GROUPS.items():
        for feature in features:
            mapping[feature] = group
    return mapping


def neutralize_feature_group(matrix: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Replace a group of features with neutral/default values for perturbation."""

    neutralized = matrix.copy()
    for feature in features:
        if feature not in neutralized.columns:
            continue
        series = neutralized[feature]
        if pd.api.types.is_numeric_dtype(series):
            value = float(series.median()) if series.notna().any() else 0.0
        else:
            mode = series.dropna().mode()
            value = mode.iloc[0] if not mode.empty else "unknown"
        neutralized[feature] = value
    return neutralized


def group_perturbation_summary(
    model: Any,
    matrix: pd.DataFrame,
    y_true: pd.Series,
    grouped_features: dict[str, list[str]],
) -> pd.DataFrame:
    """Estimate group impact by neutralizing each selected feature group."""

    baseline_probs = predict_probabilities(model, matrix)
    baseline = probability_metrics(y_true, baseline_probs)
    rows = []
    for group, features in grouped_features.items():
        present = [feature for feature in features if feature in matrix.columns]
        if not present or group == "team_player_identifiers_only":
            continue
        perturbed = neutralize_feature_group(matrix, present)
        probs = predict_probabilities(model, perturbed)
        metrics = probability_metrics(y_true, probs)
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(present),
                "features": ", ".join(present),
                "log_loss_delta": metrics["log_loss"] - baseline["log_loss"],
                "brier_delta": metrics["brier"] - baseline["brier"],
                "roc_auc_delta": _safe_delta(baseline["roc_auc"], metrics["roc_auc"]),
                "mean_probability_delta": (
                    metrics["mean_predicted_probability"] - baseline["mean_predicted_probability"]
                ),
                "absolute_probability_delta_mean": float(np.abs(probs - baseline_probs).mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_group",
                "feature_count",
                "features",
                "log_loss_delta",
                "brier_delta",
                "roc_auc_delta",
                "mean_probability_delta",
                "absolute_probability_delta_mean",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["log_loss_delta", "absolute_probability_delta_mean"], ascending=False)
        .reset_index(drop=True)
    )


def category_lift_table(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Summarise CxG lift by an interpretable football category."""

    required = {column, TARGET_COLUMN, PREDICTION_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Cannot build category lift for {column}; missing {sorted(missing)}")
    working = frame.copy()
    working[column] = working[column].fillna("unknown").astype(str)
    grouped = working.groupby(column, dropna=False)
    result = grouped.agg(
        shots=(PREDICTION_COLUMN, "size"),
        goals=(TARGET_COLUMN, "sum"),
        mean_predicted_cxg=(PREDICTION_COLUMN, "mean"),
        total_predicted_cxg=(PREDICTION_COLUMN, "sum"),
    ).reset_index()
    result = result.rename(columns={column: "category"})
    result.insert(0, "category_column", column)
    result["goal_rate"] = result["goals"] / result["shots"]
    if "baseline_cxg" in working.columns:
        baseline = grouped.agg(
            mean_baseline_cxg=("baseline_cxg", "mean"),
            total_baseline_cxg=("baseline_cxg", "sum"),
        ).reset_index(drop=True)
        result = pd.concat([result, baseline], axis=1)
        result["mean_delta_vs_baseline"] = (
            result["mean_predicted_cxg"] - result["mean_baseline_cxg"]
        )
        result["total_delta_vs_baseline"] = (
            result["total_predicted_cxg"] - result["total_baseline_cxg"]
        )
    return result.sort_values(["shots", "total_predicted_cxg"], ascending=False).reset_index(
        drop=True
    )


def result_integrity_checks(
    feature_frame: pd.DataFrame,
    shot_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Check identifier integrity in source features and promoted outputs."""

    checks: dict[str, Any] = {}
    for frame_name, frame in (
        ("feature_frame", feature_frame),
        ("shot_predictions", shot_predictions),
    ):
        for column in ("shot_id", "player_id", "team_id"):
            checks[f"{frame_name}_{column}_exists"] = column in frame.columns
            checks[f"{frame_name}_{column}_missing_count"] = (
                int(frame[column].isna().sum()) if column in frame.columns else None
            )
    return checks


def analyze_cxg_feature_impact(
    *,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    paths: FeatureImpactPaths | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, Path]:
    """Run promoted diagnostic CxG feature-impact analysis and write artifacts."""

    paths = paths or FeatureImpactPaths.from_roots()
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(paths.selected_model_metadata)
    contract = _read_json(paths.feature_contract)
    resolved_features = _read_json(paths.resolved_features)
    promotion_summary = _read_json(paths.model_promotion_summary)
    feature_group_summary = pd.read_csv(paths.feature_group_summary)
    model = joblib.load(paths.selected_model)
    feature_frame = pd.read_parquet(feature_path)
    shot_predictions = pd.read_parquet(paths.shot_predictions)
    baseline_summary = (
        pd.read_csv(paths.baseline_vs_diagnostic_summary)
        if paths.baseline_vs_diagnostic_summary.exists()
        else pd.DataFrame()
    )

    selected_features = selected_features_from_metadata(metadata)
    if "statsbomb_xg" in selected_features:
        raise ValueError("statsbomb_xg is reference-only and cannot be a selected model feature")
    if TARGET_COLUMN not in feature_frame:
        raise ValueError(f"Feature frame missing required target column {TARGET_COLUMN}")

    impact_features = model_impact_features(selected_features)
    matrix = align_selected_feature_matrix(feature_frame, selected_features)
    y_true = feature_frame[TARGET_COLUMN].astype(int)
    baseline_probs = predict_probabilities(model, matrix)
    promoted_metrics = probability_metrics(y_true, baseline_probs)
    grouped_features = map_feature_groups(selected_features)
    permutation = permutation_importance(
        model,
        matrix,
        y_true,
        impact_features,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    group_summary = group_perturbation_summary(model, matrix, y_true, grouped_features)

    category_frame = _category_frame(feature_frame, shot_predictions)
    skipped_tables: list[str] = []
    category_outputs: dict[str, str] = {}
    for column in CATEGORY_LIFT_COLUMNS:
        if column not in category_frame.columns:
            skipped_tables.append(column)
            continue
        table = category_lift_table(category_frame, column)
        output_path = paths.output_dir / f"category_lift_{column}.csv"
        table.to_csv(output_path, index=False)
        category_outputs[column] = str(output_path)

    integrity = result_integrity_checks(feature_frame, shot_predictions)
    selected_model = str(metadata.get("selected_model", "unknown"))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_VERSION,
        "selected_model_candidate": selected_model,
        "promotion_status": promotion_summary.get("promotion_status"),
        "promotion_gate_passed": promotion_summary.get("promotion_gate_passed"),
        "baseline_join_rate": promotion_summary.get("baseline_comparison", {}).get(
            "baseline_join_rate"
        ),
        "governance_status": promotion_summary.get("governance_summary", {}).get("status"),
        "selected_feature_count": len(selected_features),
        "impact_feature_count": len(impact_features),
        "selected_features": selected_features,
        "impact_features": impact_features,
        "identifier_columns_selected": sorted(
            set(selected_features).intersection(IDENTIFIER_COLUMNS)
        ),
        "reference_columns_selected": sorted(
            set(selected_features).intersection(REFERENCE_ONLY_COLUMNS)
        ),
        "selected_feature_groups": grouped_features,
        "promoted_metrics_on_feature_frame": promoted_metrics,
        "baseline_vs_diagnostic_summary": _baseline_summary_records(baseline_summary),
        "skipped_category_lift_tables": skipped_tables,
        "category_lift_outputs": category_outputs,
        "result_integrity_checks": integrity,
        "diagnostic_sources": {
            "selected_model": str(paths.selected_model),
            "selected_model_metadata": str(paths.selected_model_metadata),
            "feature_contract": str(paths.feature_contract),
            "resolved_features": str(paths.resolved_features),
            "feature_group_summary": str(paths.feature_group_summary),
            "feature_path": str(feature_path),
            "shot_predictions": str(paths.shot_predictions),
            "model_promotion_summary": str(paths.model_promotion_summary),
        },
        "governance_artifact_context": {
            "contract_version": contract.get("version"),
            "resolved_source_available": resolved_features.get("source_available", {}),
            "feature_group_summary_rows": int(len(feature_group_summary)),
        },
    }

    outputs = {
        "permutation_importance": paths.output_dir / "permutation_importance.csv",
        "group_perturbation_summary": paths.output_dir / "group_perturbation_summary.csv",
        "feature_impact_summary": paths.output_dir / "feature_impact_summary.json",
        "feature_impact_report": paths.output_dir / "feature_impact_report.md",
    }
    permutation.to_csv(outputs["permutation_importance"], index=False)
    group_summary.to_csv(outputs["group_perturbation_summary"], index=False)
    _write_json(outputs["feature_impact_summary"], summary)
    outputs["feature_impact_report"].write_text(
        build_feature_impact_report(
            summary=summary,
            permutation=permutation,
            group_summary=group_summary,
        ),
        encoding="utf-8",
    )
    return outputs


def _category_frame(feature_frame: pd.DataFrame, shot_predictions: pd.DataFrame) -> pd.DataFrame:
    id_columns = [column for column in ("shot_id", "event_id") if column in feature_frame.columns]
    if id_columns and all(column in shot_predictions.columns for column in id_columns[:1]):
        key = id_columns[0]
        category_columns = [
            column
            for column in CATEGORY_LIFT_COLUMNS
            if column in feature_frame.columns and column not in shot_predictions.columns
        ]
        return shot_predictions.merge(
            feature_frame[[key, *category_columns]],
            on=key,
            how="left",
        )
    return shot_predictions.copy()


def _baseline_summary_records(baseline_summary: pd.DataFrame) -> list[dict[str, Any]]:
    if baseline_summary.empty:
        return []
    return baseline_summary.to_dict("records")


def build_feature_impact_report(
    *,
    summary: dict[str, Any],
    permutation: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    """Build a portfolio-ready Markdown feature-impact report."""

    top_features = permutation.head(20)
    top_feature_lines = (
        [
            (
                f"- `{row.feature}` ({row.feature_group}): "
                f"log loss delta {row.log_loss_delta:.6f}, "
                f"Brier delta {row.brier_delta:.6f}"
            )
            for row in top_features.itertuples(index=False)
        ]
        if not top_features.empty
        else ["- No permutation features were available."]
    )
    group_lines = (
        [
            (
                f"- `{row.feature_group}`: {row.feature_count} features, "
                f"log loss delta {row.log_loss_delta:.6f}, "
                f"mean absolute probability move {row.absolute_probability_delta_mean:.6f}"
            )
            for row in group_summary.itertuples(index=False)
        ]
        if not group_summary.empty
        else ["- No non-empty feature groups were available for perturbation."]
    )
    integrity = summary["result_integrity_checks"]
    metrics = summary.get("promoted_metrics_on_feature_frame", {})
    skipped = summary.get("skipped_category_lift_tables", [])
    category_outputs = summary.get("category_lift_outputs", {})
    return "\n".join(
        [
            "# Promoted Diagnostic CxG Feature Impact Report",
            "",
            "## Purpose",
            "This post-promotion analysis explains which governed, selected CxG features "
            "most influence the promoted diagnostic model. It does not retrain the model, "
            "change validation thresholds, or alter result generation.",
            "",
            "## Model Context",
            f"- Model version: `{summary['model_version']}`",
            f"- Selected candidate: `{summary['selected_model_candidate']}`",
            f"- Promotion status: `{summary.get('promotion_status')}`",
            f"- Promotion gate passed: `{summary.get('promotion_gate_passed')}`",
            f"- Governance status: `{summary.get('governance_status')}`",
            f"- Baseline join rate: `{summary.get('baseline_join_rate')}`",
            f"- Selected feature count: `{summary.get('selected_feature_count')}`",
            "",
            "## Headline Metrics",
            f"- Log loss: `{metrics.get('log_loss')}`",
            f"- Brier: `{metrics.get('brier')}`",
            f"- ROC AUC: `{metrics.get('roc_auc')}`",
            f"- Goal rate: `{metrics.get('goal_rate')}`",
            "",
            "## Selected Feature Groups",
            *[
                f"- `{group}`: {', '.join(features) if features else 'none'}"
                for group, features in summary["selected_feature_groups"].items()
            ],
            "",
            "## Top 20 Permutation Importance Features",
            *top_feature_lines,
            "",
            "## Group Perturbation Ranking",
            *group_lines,
            "",
            "## Football Interpretation",
            "Positive log-loss and Brier deltas mean performance worsened when a feature "
            "or group was disrupted, so larger positive values indicate stronger model "
            "impact. Group perturbation is an ablation-style perturbation of the already "
            "trained model, not a retraining experiment.",
            "",
            "## Category Lift Summary",
            f"Generated category lift tables: {', '.join(category_outputs) or 'none'}.",
            f"Skipped category lift tables: {', '.join(skipped) or 'none'}.",
            "",
            "## Data Integrity Checks",
            f"- Feature frame `shot_id` missing: `{integrity.get('feature_frame_shot_id_missing_count')}`",
            f"- Feature frame `player_id` missing: `{integrity.get('feature_frame_player_id_missing_count')}`",
            f"- Feature frame `team_id` missing: `{integrity.get('feature_frame_team_id_missing_count')}`",
            f"- Promoted shots `shot_id` missing: `{integrity.get('shot_predictions_shot_id_missing_count')}`",
            f"- Promoted shots `player_id` missing: `{integrity.get('shot_predictions_player_id_missing_count')}`",
            f"- Promoted shots `team_id` missing: `{integrity.get('shot_predictions_team_id_missing_count')}`",
            "",
            "## Governance Notes",
            "`statsbomb_xg` remains reference-only and excluded from model-impact features. "
            "Identifier columns are retained for reporting and aggregation integrity, not "
            "permutation or perturbation scoring.",
            "",
            "## Limitations",
            "Permutation and perturbation impact are local analyses of the promoted model "
            "on the current feature frame. They explain sensitivity of this trained model; "
            "they do not prove causal football effects or choose a new model.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze promoted diagnostic CxG feature impact")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = FeatureImpactPaths.from_roots(
        diagnostic_dir=args.diagnostic_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    outputs = analyze_cxg_feature_impact(
        feature_path=args.feature_path,
        paths=paths,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )
    print(json.dumps({name: str(path) for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
