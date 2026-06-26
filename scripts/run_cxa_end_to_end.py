#!/usr/bin/env python
"""Train, evaluate, score, and export the baseline CxA model."""

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from opponent_adjusted.config import settings

FEATURE_STORE_INPUT = settings.feature_store_path / "cxa" / "action_features.parquet"
DEFAULT_OUTPUT_DIR = Path("outputs") / "modeling" / "cxa"
CONTRACT_PATH = Path("configs") / "feature_contracts" / "cxa_v1.json"
MODEL_VERSION_PREFIX = "cxa_baseline"
TARGET_COLUMN = "shot_created"
VALUE_COLUMN = "created_shot_cxg"


@dataclass(frozen=True)
class CxARunOutputs:
    """Paths emitted by the CxA baseline run."""

    model_path: Path
    metadata_path: Path
    metrics_path: Path
    predictions_path: Path
    player_aggregates_path: Path
    team_aggregates_path: Path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported CxA input format: {path.suffix}")


def _load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _feature_columns(contract: dict[str, Any], df: pd.DataFrame) -> dict[str, list[str]]:
    forbidden = set(contract.get("forbidden_training_features", []))
    numeric = [
        col
        for col in contract.get("required_numeric_features", [])
        + contract.get("optional_context_features", [])
        if col in df.columns
        and col not in forbidden
        and pd.api.types.is_numeric_dtype(df[col])
        and df[col].notna().any()
    ]
    binary = [
        col
        for col in contract.get("required_binary_features", [])
        if col in df.columns and col not in forbidden
    ]
    categorical = [
        col
        for col in contract.get("required_categorical_features", [])
        if col in df.columns and col not in forbidden
    ]
    return {"numeric": numeric, "binary": binary, "categorical": categorical}


def _check_leakage_columns(contract: dict[str, Any], feature_groups: dict[str, list[str]]) -> None:
    forbidden = set(contract.get("forbidden_training_features", [])) | set(
        contract.get("prohibited_leakage_columns", [])
    )
    model_features = set().union(*[set(values) for values in feature_groups.values()])
    leaked = sorted(model_features & forbidden)
    if leaked:
        raise ValueError(f"CxA model features include prohibited leakage columns: {leaked}")


def _build_model(
    feature_groups: dict[str, list[str]],
    *,
    use_dummy: bool,
) -> Pipeline:
    transformers = []
    if feature_groups["numeric"]:
        transformers.append(
            (
                "num",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                feature_groups["numeric"],
            )
        )
    if feature_groups["binary"]:
        transformers.append(
            (
                "bin",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                feature_groups["binary"],
            )
        )
    if feature_groups["categorical"]:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                feature_groups["categorical"],
            )
        )
    if not transformers:
        raise ValueError("No contract-aligned CxA model features found")

    estimator: DummyClassifier | LogisticRegression
    if use_dummy:
        estimator = DummyClassifier(strategy="prior")
    else:
        estimator = LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs")
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", estimator)])


def _safe_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "row_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "mean_predicted_probability": float(np.mean(y_pred)) if len(y_pred) else 0.0,
        "brier": float(brier_score_loss(y_true, y_pred)) if len(y_true) else None,
        "log_loss": None,
        "log_loss_status": "skipped_single_class",
        "roc_auc": None,
        "roc_auc_status": "skipped_single_class",
    }
    if len(np.unique(y_true)) == 2:
        metrics["log_loss"] = float(log_loss(y_true, y_pred, labels=[0, 1]))
        metrics["log_loss_status"] = "computed"
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        metrics["roc_auc_status"] = "computed"
    return metrics


def _positive_class_probability(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.named_steps["model"].classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(frame), dtype=float)


def _cross_validated_predictions(
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    feature_cols = (
        feature_groups["numeric"] + feature_groups["binary"] + feature_groups["categorical"]
    )
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        prior = np.full(len(df), float(y.mean()) if len(y) else 0.0)
        return prior, [{"fold": 1, "status": "skipped_single_class"}]

    class_counts = pd.Series(y).value_counts()
    n_splits = int(max(2, min(5, class_counts.min())))
    groups = df["match_id"].to_numpy() if "match_id" in df.columns else None
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splits = list(GroupKFold(n_splits=n_splits).split(df[feature_cols], y, groups))
        split_kind = "group_kfold_match_id"
    else:
        splits = list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(
                df[feature_cols], y
            )
        )
        split_kind = "stratified_kfold"

    predictions = np.zeros(len(df), dtype=float)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y[train_idx]
        use_dummy = len(np.unique(y_train)) < 2
        model = _build_model(feature_groups, use_dummy=use_dummy)
        model.fit(df.iloc[train_idx][feature_cols], y_train)
        probs = _positive_class_probability(model, df.iloc[test_idx][feature_cols])
        predictions[test_idx] = probs
        row = _safe_classification_metrics(y[test_idx], probs)
        row.update({"fold": fold, "split_kind": split_kind})
        folds.append(row)
    return predictions, folds


def _aggregate(scored: pd.DataFrame, entity_id: str) -> pd.DataFrame:
    if entity_id not in scored.columns:
        return pd.DataFrame()
    return (
        scored.groupby(entity_id, dropna=False)
        .agg(
            actions_count=("action_id", "count"),
            shot_creating_actions=("shot_created", "sum"),
            summed_created_shot_cxg=("created_shot_cxg", "sum"),
            summed_predicted_cxa=("predicted_cxa", "sum"),
            summed_baseline_cxa=("baseline_cxa", "sum"),
            summed_cxa_above_baseline=("cxa_above_baseline", "sum"),
            avg_predicted_cxa=("predicted_cxa", "mean"),
        )
        .reset_index()
        .sort_values("summed_predicted_cxa", ascending=False)
    )


def train_evaluate_score(
    df: pd.DataFrame,
    *,
    model_version: str,
) -> tuple[Pipeline, pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    contract = _load_contract()
    required = {"action_id", TARGET_COLUMN, VALUE_COLUMN, "match_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CxA input is missing required columns: {missing}")

    df = df.dropna(subset=[TARGET_COLUMN, VALUE_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    df[VALUE_COLUMN] = df[VALUE_COLUMN].astype(float).clip(0.0, 1.0)
    feature_groups = _feature_columns(contract, df)
    _check_leakage_columns(contract, feature_groups)
    for column in feature_groups["binary"]:
        df[column] = df[column].astype(float)
    feature_cols = (
        feature_groups["numeric"] + feature_groups["binary"] + feature_groups["categorical"]
    )

    cv_probs, fold_metrics = _cross_validated_predictions(df, feature_groups)
    y = df[TARGET_COLUMN].to_numpy()
    positive_mean_value = (
        float(df.loc[df[TARGET_COLUMN] == 1, VALUE_COLUMN].mean())
        if int(df[TARGET_COLUMN].sum()) > 0
        else 0.0
    )
    baseline_probability = float(df[TARGET_COLUMN].mean()) if len(df) else 0.0
    baseline_value = baseline_probability * positive_mean_value

    use_dummy = len(np.unique(y)) < 2
    final_model = _build_model(feature_groups, use_dummy=use_dummy)
    final_model.fit(df[feature_cols], y)
    full_probs = _positive_class_probability(final_model, df[feature_cols])

    scored = df.copy()
    scored["predicted_shot_created_probability"] = full_probs
    scored["predicted_cxa"] = full_probs * positive_mean_value
    scored["baseline_cxa"] = baseline_value
    scored["cxa_above_baseline"] = scored["predicted_cxa"] - scored["baseline_cxa"]
    scored["model_version"] = model_version

    metrics = _safe_classification_metrics(y, cv_probs)
    metrics.update(
        {
            "target": TARGET_COLUMN,
            "value_column": VALUE_COLUMN,
            "model_version": model_version,
            "positive_mean_created_shot_cxg": positive_mean_value,
            "baseline_probability": baseline_probability,
            "baseline_cxa": baseline_value,
            "folds": fold_metrics,
            "features": feature_groups,
            "estimator": "dummy_prior" if use_dummy else "logistic_regression",
        }
    )
    return final_model, scored, metrics, feature_groups


def run_end_to_end(
    input_path: Path = FEATURE_STORE_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_version: str | None = None,
) -> CxARunOutputs:
    df = _read_table(input_path)
    created_at = datetime.now(timezone.utc).isoformat()
    version = model_version or f"{MODEL_VERSION_PREFIX}_{created_at[:10].replace('-', '')}"
    model, scored, metrics, features = train_evaluate_score(df, model_version=version)

    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    aggregates_dir = output_dir / "aggregates"
    for directory in (models_dir, reports_dir, predictions_dir, aggregates_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "baseline_model.joblib"
    metadata_path = models_dir / "baseline_model.json"
    metrics_path = reports_dir / "metrics.json"
    predictions_path = predictions_dir / "action_predictions.parquet"
    player_path = aggregates_dir / "player_cxa.parquet"
    team_path = aggregates_dir / "team_cxa.parquet"

    joblib.dump(model, model_path)
    scored.to_parquet(predictions_path, index=False)
    _aggregate(scored, "player_id").to_parquet(player_path, index=False)
    _aggregate(scored, "team_id").to_parquet(team_path, index=False)

    metadata = {
        "model_name": "cxa",
        "model_version": version,
        "version": version,
        "model_type": "baseline_action_classifier",
        "target": TARGET_COLUMN,
        "value_column": VALUE_COLUMN,
        "artifact_path": str(model_path),
        "metadata_path": str(metadata_path),
        "created_at": created_at,
        "generated_at": created_at,
        "training_input_path": str(input_path),
        "trained_rows": int(len(scored)),
        "features": features,
        "leakage_guardrails": {
            "contract": str(CONTRACT_PATH),
            "forbidden_training_features_excluded": True,
        },
        "attribution": {
            "target": "first same-team same-possession shot within CxA action window",
            "event_data_only": True,
        },
        "outputs": {
            "predictions": str(predictions_path),
            "player_aggregates": str(player_path),
            "team_aggregates": str(team_path),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return CxARunOutputs(
        model_path, metadata_path, metrics_path, predictions_path, player_path, team_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CxA baseline training/evaluation/export")
    parser.add_argument("--input", type=Path, default=FEATURE_STORE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    outputs = run_end_to_end(args.input, args.output_dir, args.model_version)
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
