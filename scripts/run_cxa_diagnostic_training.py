#!/usr/bin/env python
"""Train diagnostic CxA model candidates from the diagnostic feature contract."""

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

DEFAULT_INPUT_PATH = Path("feature_store") / "cxa" / "action_features.parquet"
DEFAULT_CONTRACT_PATH = (
    Path("outputs") / "modeling" / "cxa" / "diagnostic_v1" / "contracts" / "feature_contract.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "modeling" / "cxa" / "diagnostic_v1"
MODEL_VERSION = "diagnostic_v1"
METRIC = "cxa"
TARGET_COLUMN = "shot_created"
ATTRIBUTION_REFERENCE = "created_shot_cxg"
VALUE_OUTPUT = "cxa_value"
MODEL_CANDIDATES = (
    "logistic_regression",
    "calibrated_logistic_regression",
    "gradient_boosting",
    "calibrated_gradient_boosting_sigmoid",
)
EXCLUDED_BUCKETS = (
    "target_columns",
    "reference_only_columns",
    "output_prediction_columns",
    "leakage_excluded_columns",
    "identifier_columns",
    "requires_review_columns",
    "excluded_unknown_columns",
)
PREDICTION_ID_COLUMNS = (
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "sequence_id",
    "possession",
)


@dataclass(frozen=True)
class DiagnosticTrainingOutputs:
    model_candidates: Path
    selected_model: Path
    selected_model_metadata: Path
    cross_validated_predictions: Path
    model_comparison: Path
    training_report: Path
    training_summary: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            return value
    return value


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported CxA diagnostic training input format: {path.suffix}")


def _read_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CxA diagnostic feature contract must be a JSON object")
    return payload


def resolve_selected_features(
    df: pd.DataFrame,
    contract: dict[str, Any],
) -> dict[str, list[str]]:
    """Resolve contract-selected feature candidates that exist in the input frame."""

    selected = contract.get("selected_feature_candidates", {})
    return {
        group: [
            column
            for column in selected.get(group, [])
            if column in df.columns and df[column].notna().any()
        ]
        for group in ("numeric", "binary", "categorical")
    }


def _all_features(feature_groups: dict[str, list[str]]) -> list[str]:
    return (
        feature_groups.get("numeric", [])
        + feature_groups.get("binary", [])
        + feature_groups.get("categorical", [])
    )


def assert_leakage_guard(
    feature_groups: dict[str, list[str]],
    contract: dict[str, Any],
) -> None:
    """Fail if any model feature appears in a forbidden contract bucket."""

    selected = set(_all_features(feature_groups))
    excluded_columns = contract.get("excluded_columns", {})
    forbidden: set[str] = set()
    for bucket in EXCLUDED_BUCKETS:
        forbidden.update(excluded_columns.get(bucket, []))
    leaked = sorted(selected & forbidden)
    if leaked:
        raise ValueError(f"CxA diagnostic feature leakage guard failed: {leaked}")
    if TARGET_COLUMN in selected or ATTRIBUTION_REFERENCE in selected or VALUE_OUTPUT in selected:
        raise ValueError("CxA diagnostic features include target/reference/output columns")


def _coerce_binary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert bool-like binary columns into numeric values accepted by sklearn."""

    return pd.DataFrame(frame).astype("float64")


def _binary_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("coerce_binary", FunctionTransformer(_coerce_binary_frame, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )


def _linear_preprocessor(feature_groups: dict[str, list[str]]) -> ColumnTransformer:
    transformers = []
    if feature_groups["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_groups["numeric"],
            )
        )
    if feature_groups["binary"]:
        transformers.append(
            (
                "binary",
                _binary_preprocessor(),
                feature_groups["binary"],
            )
        )
    if feature_groups["categorical"]:
        transformers.append(
            (
                "categorical",
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
        raise ValueError("No diagnostic CxA features available for training")
    return ColumnTransformer(transformers)


def _tree_preprocessor(feature_groups: dict[str, list[str]]) -> ColumnTransformer:
    transformers = []
    if feature_groups["numeric"]:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                feature_groups["numeric"],
            )
        )
    if feature_groups["binary"]:
        transformers.append(
            (
                "binary",
                _binary_preprocessor(),
                feature_groups["binary"],
            )
        )
    if feature_groups["categorical"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                feature_groups["categorical"],
            )
        )
    if not transformers:
        raise ValueError("No diagnostic CxA features available for training")
    return ColumnTransformer(transformers)


def _calibrated(estimator: Pipeline, *, method: str, cv: int) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    except TypeError:  # pragma: no cover - older sklearn compatibility
        return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def build_candidate_model(
    candidate_name: str,
    feature_groups: dict[str, list[str]],
    *,
    random_state: int,
    calibration_cv: int = 2,
    use_dummy: bool = False,
) -> Pipeline | CalibratedClassifierCV:
    """Build a deterministic CxA diagnostic candidate model."""

    if use_dummy:
        return Pipeline(
            [
                ("preprocess", _linear_preprocessor(feature_groups)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        )
    if candidate_name in {"logistic_regression", "calibrated_logistic_regression"}:
        pipeline = Pipeline(
            [
                ("preprocess", _linear_preprocessor(feature_groups)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        C=0.5,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        if candidate_name == "calibrated_logistic_regression":
            return _calibrated(pipeline, method="sigmoid", cv=calibration_cv)
        return pipeline
    if candidate_name in {"gradient_boosting", "calibrated_gradient_boosting_sigmoid"}:
        pipeline = Pipeline(
            [
                ("preprocess", _tree_preprocessor(feature_groups)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=50,
                        learning_rate=0.08,
                        l2_regularization=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        if candidate_name == "calibrated_gradient_boosting_sigmoid":
            return _calibrated(pipeline, method="sigmoid", cv=calibration_cv)
        return pipeline
    raise ValueError(f"Unsupported CxA diagnostic candidate: {candidate_name}")


def _positive_class_probability(model: Any, frame: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(frame)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model[-1].classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(frame), dtype=float)


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
    metrics: dict[str, Any] = {
        "row_count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "mean_predicted_probability": float(np.mean(y_pred)) if len(y_pred) else 0.0,
        "brier": float(brier_score_loss(y_true, y_pred)) if len(y_true) else None,
        "log_loss": None,
        "roc_auc": None,
        "average_precision": None,
        "precision_at_threshold": None,
        "recall_at_threshold": None,
        "calibration_error": None,
    }
    if len(y_true):
        metrics["calibration_error"] = abs(
            metrics["mean_predicted_probability"] - metrics["positive_rate"]
        )
        labels = (y_pred >= 0.5).astype(int)
        metrics["precision_at_threshold"] = float(precision_score(y_true, labels, zero_division=0))
        metrics["recall_at_threshold"] = float(recall_score(y_true, labels, zero_division=0))
    if len(np.unique(y_true)) == 2:
        metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        metrics["average_precision"] = float(average_precision_score(y_true, y_pred))
    return metrics


def _split_indices(
    df: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    max_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, str | None]:
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2:
        return [(np.arange(len(df)), np.arange(len(df)))], "single_class_resubstitution", None
    n_splits = int(max(2, min(max_splits, class_counts.min())))
    if "match_id" in df.columns and df["match_id"].nunique(dropna=True) >= n_splits:
        groups = df["match_id"].to_numpy()
        return (
            list(GroupKFold(n_splits=n_splits).split(df, y, groups)),
            "group_kfold_match_id",
            None,
        )
    reason = "match_id missing or has too few unique groups"
    return (
        list(
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(df, y)
        ),
        "stratified_kfold",
        reason,
    )


def _candidate_status(y_train: np.ndarray, candidate_name: str) -> tuple[bool, int]:
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) < 2:
        return True, 2
    min_class = int(class_counts.min())
    return False, min(2, min_class)


def _prediction_frame(
    df: pd.DataFrame,
    *,
    y: np.ndarray,
    predictions: np.ndarray,
    candidate_name: str,
    folds: np.ndarray,
    split_kind: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=df.index)
    for column in PREDICTION_ID_COLUMNS:
        if column in df.columns:
            output[column] = df[column]
    output[TARGET_COLUMN] = y
    output["predicted_shot_created_probability"] = predictions
    output["model_candidate"] = candidate_name
    output["fold"] = folds
    output["split"] = split_kind
    return output.reset_index(drop=True)


def _train_candidate_cv(
    candidate_name: str,
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    splits: list[tuple[np.ndarray, np.ndarray]],
    split_kind: str,
    *,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_cols = _all_features(feature_groups)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    predictions = np.zeros(len(df), dtype=float)
    fold_labels = np.zeros(len(df), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    status = "trained"
    notes = ""
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y[train_idx]
        use_dummy, calibration_cv = _candidate_status(y_train, candidate_name)
        if use_dummy:
            status = "trained_dummy_single_class_fold"
            notes = "At least one fold had a single training class; prior dummy used for that fold."
        model = build_candidate_model(
            candidate_name,
            feature_groups,
            random_state=random_state,
            calibration_cv=calibration_cv,
            use_dummy=use_dummy,
        )
        model.fit(df.iloc[train_idx][feature_cols], y_train)
        probs = _positive_class_probability(model, df.iloc[test_idx][feature_cols])
        predictions[test_idx] = probs
        fold_labels[test_idx] = fold
        fold_metric = _safe_metrics(y[test_idx], probs)
        fold_metric.update({"candidate_name": candidate_name, "fold": fold})
        fold_rows.append(fold_metric)

    metrics = _safe_metrics(y, predictions)
    candidate = {
        "candidate_name": candidate_name,
        "estimator_type": candidate_name,
        "selected_features": feature_cols,
        "numeric_features": feature_groups["numeric"],
        "binary_features": feature_groups["binary"],
        "categorical_features": feature_groups["categorical"],
        "row_count": int(len(df)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "metrics": metrics,
        "fold_metrics": fold_rows,
        "training_config": {
            "split_kind": split_kind,
            "fold_count": len(splits),
            "random_state": random_state,
            "calibration_method": "sigmoid" if "calibrated" in candidate_name else None,
        },
        "leakage_checks": {"passed": True, "source": "diagnostic_v1_feature_contract"},
        "status": status,
        "notes": notes,
    }
    predictions_df = _prediction_frame(
        df,
        y=y,
        predictions=predictions,
        candidate_name=candidate_name,
        folds=fold_labels,
        split_kind=split_kind,
    )
    return candidate, predictions_df


def _comparison(candidates: list[dict[str, Any]], selected_name: str) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        metrics = candidate["metrics"]
        rows.append(
            {
                "candidate_name": candidate["candidate_name"],
                "log_loss": metrics.get("log_loss"),
                "brier": metrics.get("brier"),
                "roc_auc": metrics.get("roc_auc"),
                "average_precision": metrics.get("average_precision"),
                "positive_rate": metrics.get("positive_rate"),
                "mean_predicted_probability": metrics.get("mean_predicted_probability"),
                "calibration_error": metrics.get("calibration_error"),
                "selected": candidate["candidate_name"] == selected_name,
                "notes": candidate.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def key(candidate: dict[str, Any]) -> tuple[float, float, float]:
        metrics = candidate["metrics"]
        logloss = metrics.get("log_loss")
        brier = metrics.get("brier")
        ap = metrics.get("average_precision")
        return (
            float(logloss) if logloss is not None else float("inf"),
            float(brier) if brier is not None else float("inf"),
            -float(ap) if ap is not None else float("inf"),
        )

    return sorted(candidates, key=key)[0]


def _fit_final_model(
    candidate_name: str,
    df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    *,
    random_state: int,
) -> Any:
    feature_cols = _all_features(feature_groups)
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    use_dummy, calibration_cv = _candidate_status(y, candidate_name)
    model = build_candidate_model(
        candidate_name,
        feature_groups,
        random_state=random_state,
        calibration_cv=calibration_cv,
        use_dummy=use_dummy,
    )
    model.fit(df[feature_cols], y)
    return model


def _metadata(
    *,
    selected: dict[str, Any],
    feature_groups: dict[str, list[str]],
    contract: dict[str, Any],
    df: pd.DataFrame,
    input_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    excluded = contract.get("excluded_columns", {})
    y = df[TARGET_COLUMN].astype(int)
    return {
        "metric": METRIC,
        "model_version": MODEL_VERSION,
        "selected_model_candidate": selected["candidate_name"],
        "selected_by": "lowest_log_loss_then_brier_then_average_precision",
        "primary_target": TARGET_COLUMN,
        "attribution_reference": contract.get("attribution_reference", ATTRIBUTION_REFERENCE),
        "value_output": contract.get("value_output", VALUE_OUTPUT),
        "selected_feature_count": len(_all_features(feature_groups)),
        "numeric_feature_count": len(feature_groups["numeric"]),
        "binary_feature_count": len(feature_groups["binary"]),
        "categorical_feature_count": len(feature_groups["categorical"]),
        "excluded_column_summary": {
            bucket: len(excluded.get(bucket, [])) for bucket in EXCLUDED_BUCKETS
        },
        "leakage_guard_passed": True,
        "training_rows": int(len(df)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": input_path.as_posix(),
        "contract_path": contract_path.as_posix(),
    }


def _training_report(
    *,
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> str:
    comparison_preview = comparison[
        [
            "candidate_name",
            "log_loss",
            "brier",
            "roc_auc",
            "average_precision",
            "selected",
        ]
    ].to_string(index=False)
    return "\n".join(
        [
            "# Diagnostic CxA Training Report",
            "",
            "## Executive summary",
            "- This is diagnostic CxA candidate training.",
            f"- Selected model: `{metadata['selected_model_candidate']}`.",
            f"- Training rows: {metadata['training_rows']}",
            "",
            "## Target definition",
            "- The model predicts `shot_created` as the primary binary target.",
            "",
            "## Feature contract used",
            f"- Contract path: `{metadata['contract_path']}`",
            f"- Selected feature count: {metadata['selected_feature_count']}",
            "",
            "## Leakage guard",
            "- The training matrix excludes `created_shot_cxg`, `cxa_value`, identifiers, prediction outputs, requires-review columns, and excluded-unknown columns.",
            "- Leakage guard passed before candidate training.",
            "",
            "## Candidate models",
            "- logistic_regression",
            "- calibrated_logistic_regression",
            "- gradient_boosting",
            "- calibrated_gradient_boosting_sigmoid",
            "",
            "## Metrics",
            "```text",
            comparison_preview,
            "```",
            "",
            "## Selected model",
            f"- Selected by: {metadata['selected_by']}",
            "",
            "## Class imbalance notes",
            f"- Positive rate: {metadata['positive_rate']:.6f}",
            "- Metrics include log loss, Brier score, ROC AUC, average precision, and threshold precision/recall.",
            "",
            "## Limitations",
            "- This PR does not validate or promote the diagnostic CxA model.",
            "- CxA+ progression/state-value enhancement comes later.",
            "",
            "## Next recommended PR",
            "- Validate diagnostic CxA candidates against the fair baseline and decide whether to promote or revise.",
            "",
            "## Feature groups",
            f"- Numeric: {', '.join(feature_groups['numeric'])}",
            f"- Binary: {', '.join(feature_groups['binary'])}",
            f"- Categorical: {', '.join(feature_groups['categorical'])}",
            "",
        ]
    )


def run_cxa_diagnostic_training(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    random_state: int = 42,
    folds: int = 3,
) -> DiagnosticTrainingOutputs:
    """Train diagnostic CxA candidates and write model artifacts."""

    df = _read_table(input_path)
    contract = _read_contract(contract_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"CxA diagnostic training requires target column `{TARGET_COLUMN}`")

    feature_groups = resolve_selected_features(df, contract)
    assert_leakage_guard(feature_groups, contract)
    feature_cols = _all_features(feature_groups)
    if not feature_cols:
        raise ValueError("No contract-selected CxA diagnostic features are available")

    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for directory in (models_dir, predictions_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    y = df[TARGET_COLUMN].astype(int).to_numpy()
    splits, split_kind, fallback_reason = _split_indices(
        df,
        y,
        random_state=random_state,
        max_splits=folds,
    )

    candidates: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_name in MODEL_CANDIDATES:
        print(f"Training CxA diagnostic candidate: {candidate_name}", flush=True)
        candidate, predictions = _train_candidate_cv(
            candidate_name,
            df,
            feature_groups,
            splits,
            split_kind,
            random_state=random_state,
        )
        if fallback_reason:
            candidate["training_config"]["fallback_reason"] = fallback_reason
        candidates.append(candidate)
        prediction_frames.append(predictions)

    selected = _select_candidate(candidates)
    selected_model = _fit_final_model(
        selected["candidate_name"],
        df,
        feature_groups,
        random_state=random_state,
    )
    comparison = _comparison(candidates, selected["candidate_name"])
    metadata = _metadata(
        selected=selected,
        feature_groups=feature_groups,
        contract=contract,
        df=df,
        input_path=input_path,
        contract_path=contract_path,
    )
    training_summary = {
        **metadata,
        "candidate_count": len(candidates),
        "split_kind": split_kind,
        "fallback_reason": fallback_reason,
        "outputs": {
            "model_candidates": (models_dir / "model_candidates.json").as_posix(),
            "selected_model": (models_dir / "selected_model.joblib").as_posix(),
            "selected_model_metadata": (models_dir / "selected_model_metadata.json").as_posix(),
            "cross_validated_predictions": (
                predictions_dir / "cross_validated_predictions.parquet"
            ).as_posix(),
            "model_comparison": (reports_dir / "model_comparison.csv").as_posix(),
            "training_report": (reports_dir / "training_report.md").as_posix(),
            "training_summary": (reports_dir / "training_summary.json").as_posix(),
        },
    }

    outputs = DiagnosticTrainingOutputs(
        model_candidates=models_dir / "model_candidates.json",
        selected_model=models_dir / "selected_model.joblib",
        selected_model_metadata=models_dir / "selected_model_metadata.json",
        cross_validated_predictions=predictions_dir / "cross_validated_predictions.parquet",
        model_comparison=reports_dir / "model_comparison.csv",
        training_report=reports_dir / "training_report.md",
        training_summary=reports_dir / "training_summary.json",
    )
    outputs.model_candidates.write_text(
        json.dumps(_json_safe({"candidates": candidates}), indent=2),
        encoding="utf-8",
    )
    joblib.dump(selected_model, outputs.selected_model)
    outputs.selected_model_metadata.write_text(
        json.dumps(_json_safe(metadata), indent=2),
        encoding="utf-8",
    )
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        outputs.cross_validated_predictions,
        index=False,
    )
    comparison.to_csv(outputs.model_comparison, index=False)
    outputs.training_report.write_text(
        _training_report(metadata=metadata, comparison=comparison, feature_groups=feature_groups),
        encoding="utf-8",
    )
    outputs.training_summary.write_text(
        json.dumps(_json_safe(training_summary), indent=2),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diagnostic CxA model candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    outputs = run_cxa_diagnostic_training(
        input_path=args.input,
        contract_path=args.contract,
        output_dir=args.output_dir,
        random_state=args.random_state,
        folds=args.folds,
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.__dict__.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
