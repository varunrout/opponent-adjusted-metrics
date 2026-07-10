#!/usr/bin/env python
"""Train the first diagnostic CxA+ model from the governed feature matrix."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DEFAULT_MATRIX_PATH = Path(
    "outputs/modeling/cxa_plus/diagnostic_v1/datasets/feature_matrix.parquet"
)
DEFAULT_SUMMARY_PATH = Path(
    "outputs/modeling/cxa_plus/diagnostic_v1/datasets/feature_matrix_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/modeling/cxa_plus/diagnostic_v1")
TARGET_COLUMN = "shot_within_next_5_actions"
GROUP_COLUMN = "match_id"
MODEL_NAME = "regularized_logistic_regression"
IDENTIFIER_COLUMNS = {
    "action_id",
    "event_id",
    "match_id",
    "possession",
    "sequence_id",
    "team_id",
    "player_id",
}
FORBIDDEN_TOKENS = (
    "within_next",
    "downstream",
    "future",
    "outcome",
    "result",
    "predicted",
    "model_",
    "probability",
    "score",
)
OUTPUT_FILES = {
    "model": Path("models/cxa_plus_diagnostic_model.joblib"),
    "metrics": Path("results/diagnostic_metrics.json"),
    "calibration": Path("results/calibration_bins.csv"),
    "coefficients": Path("results/feature_coefficients.csv"),
    "sample": Path("results/prediction_sample.csv"),
    "report": Path("reports/diagnostic_training_report.md"),
}


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    binary: list[str]
    categorical: list[str]

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.binary, *self.categorical]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, (bool, str)):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def load_feature_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CxA+ feature matrix not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature matrix format: {path.suffix}")


def load_feature_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"CxA+ feature matrix summary not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CxA+ feature matrix summary must be a JSON object")
    return payload


def selected_features_from_summary(summary: dict[str, Any]) -> list[str]:
    features = summary.get("eligible_model_features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("feature_matrix_summary.json must contain eligible_model_features")
    return [str(feature) for feature in features]


def assert_feature_leakage_guard(features: list[str]) -> None:
    forbidden: list[str] = []
    for feature in features:
        lower = feature.lower()
        if feature == TARGET_COLUMN:
            forbidden.append(feature)
        elif feature in IDENTIFIER_COLUMNS:
            forbidden.append(feature)
        elif feature.startswith("created_shot_"):
            forbidden.append(feature)
        elif any(token in lower for token in FORBIDDEN_TOKENS):
            forbidden.append(feature)
    if forbidden:
        raise ValueError(f"CxA+ diagnostic feature leakage guard failed: {sorted(forbidden)}")


def resolve_feature_groups(frame: pd.DataFrame, features: list[str]) -> FeatureGroups:
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise ValueError(f"Allowlisted CxA+ model features missing from matrix: {missing}")
    numeric: list[str] = []
    binary: list[str] = []
    categorical: list[str] = []
    for feature in features:
        series = frame[feature]
        non_null = series.dropna()
        unique_values = set(non_null.unique().tolist())
        if pd.api.types.is_bool_dtype(series) or unique_values.issubset(
            {0, 1, 0.0, 1.0, True, False}
        ):
            binary.append(feature)
        elif pd.api.types.is_numeric_dtype(series):
            numeric.append(feature)
        else:
            categorical.append(feature)
    return FeatureGroups(numeric=numeric, binary=binary, categorical=categorical)


def _coerce_binary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(frame).astype("float64")


def build_preprocessor(feature_groups: FeatureGroups) -> ColumnTransformer:
    transformers = []
    if feature_groups.numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_groups.numeric,
            )
        )
    if feature_groups.binary:
        transformers.append(
            (
                "binary",
                Pipeline(
                    [
                        (
                            "coerce",
                            FunctionTransformer(
                                _coerce_binary_frame,
                                validate=False,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                feature_groups.binary,
            )
        )
    if feature_groups.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_groups.categorical,
            )
        )
    if not transformers:
        raise ValueError("No CxA+ model features available after grouping")
    return ColumnTransformer(transformers)


def grouped_train_test_split(
    frame: pd.DataFrame,
    *,
    target_column: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if group_column not in frame.columns:
        raise ValueError(f"CxA+ diagnostic training requires group column `{group_column}`")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = frame[target_column].astype(int)
    groups = frame[group_column]
    train_idx, test_idx = next(splitter.split(frame, y, groups=groups))
    shared = set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))
    if shared:
        raise ValueError(f"Grouped split leaked match_id values: {sorted(shared)[:10]}")
    for split_name, idx in {"train": train_idx, "test": test_idx}.items():
        if int(y.iloc[idx].sum()) == 0:
            raise ValueError(f"CxA+ diagnostic {split_name} split has no positive examples")
    return train_idx, test_idx


def fit_model(frame: pd.DataFrame, features: list[str], train_idx: np.ndarray) -> Pipeline:
    feature_groups = resolve_feature_groups(frame, features)
    model = Pipeline(
        [
            ("preprocess", build_preprocessor(feature_groups)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(frame.iloc[train_idx][features], frame.iloc[train_idx][TARGET_COLUMN].astype(int))
    return model


def predict_positive_probability(
    model: Pipeline, frame: pd.DataFrame, features: list[str]
) -> np.ndarray:
    probabilities = model.predict_proba(frame[features])
    return np.asarray(probabilities[:, 1], dtype=float)


def metric_summary(y_true: pd.Series, y_pred: np.ndarray, baseline_rate: float) -> dict[str, Any]:
    clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
    baseline = np.full(len(y_true), baseline_rate, dtype=float)
    baseline_clipped = np.clip(baseline, 1e-15, 1 - 1e-15)
    metrics = {
        "log_loss": log_loss(y_true, clipped, labels=[0, 1]),
        "brier": brier_score_loss(y_true, clipped),
        "roc_auc": roc_auc_score(y_true, clipped) if y_true.nunique() == 2 else None,
        "average_precision": average_precision_score(y_true, clipped),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(np.mean(clipped)),
        "baseline_positive_rate": baseline_rate,
        "baseline_log_loss": log_loss(y_true, baseline_clipped, labels=[0, 1]),
        "baseline_brier": brier_score_loss(y_true, baseline_clipped),
    }
    metrics["log_loss_lift_over_baseline"] = metrics["baseline_log_loss"] - metrics["log_loss"]
    metrics["brier_lift_over_baseline"] = metrics["baseline_brier"] - metrics["brier"]
    return metrics


def calibration_bins(y_true: pd.Series, y_pred: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"target": y_true.astype(int).to_numpy(), "prediction": y_pred})
    frame["bin"] = pd.cut(
        frame["prediction"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True
    )
    rows = []
    for bin_id, (interval, group) in enumerate(frame.groupby("bin", observed=False), start=1):
        if group.empty:
            lower = float(interval.left)
            upper = float(interval.right)
            rows.append(
                {
                    "bin": bin_id,
                    "probability_lower": lower,
                    "probability_upper": upper,
                    "row_count": 0,
                    "positive_count": 0,
                    "positive_rate": np.nan,
                    "mean_predicted_probability": np.nan,
                    "absolute_calibration_error": np.nan,
                }
            )
            continue
        positive_rate = float(group["target"].mean())
        mean_pred = float(group["prediction"].mean())
        rows.append(
            {
                "bin": bin_id,
                "probability_lower": float(interval.left),
                "probability_upper": float(interval.right),
                "row_count": len(group),
                "positive_count": int(group["target"].sum()),
                "positive_rate": positive_rate,
                "mean_predicted_probability": mean_pred,
                "absolute_calibration_error": abs(positive_rate - mean_pred),
            }
        )
    return pd.DataFrame(rows)


def feature_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    coefficients = estimator.coef_[0]
    return (
        pd.DataFrame({"feature": names, "coefficient": coefficients})
        .assign(abs_coefficient=lambda df: df["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def split_summary(
    frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for split_name, idx in {"train": train_idx, "test": test_idx}.items():
        split = frame.iloc[idx]
        rows[split_name] = {
            "row_count": len(split),
            "match_count": int(split[GROUP_COLUMN].nunique()),
            "positive_count": int(split[TARGET_COLUMN].sum()),
            "positive_rate": float(split[TARGET_COLUMN].mean()),
        }
    rows["shared_match_count"] = int(
        len(
            set(frame.iloc[train_idx][GROUP_COLUMN]).intersection(
                set(frame.iloc[test_idx][GROUP_COLUMN])
            )
        )
    )
    return rows


def prediction_sample(
    frame: pd.DataFrame,
    test_idx: np.ndarray,
    predictions: np.ndarray,
    *,
    n: int = 500,
) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "action_id",
            "event_id",
            "match_id",
            "possession",
            "sequence_id",
            TARGET_COLUMN,
        )
        if column in frame.columns
    ]
    sample = frame.iloc[test_idx][columns].copy()
    sample["predicted_cxa_plus"] = predictions
    return sample.head(n)


def build_report(metrics: dict[str, Any], split: dict[str, Any], features: list[str]) -> str:
    return f"""# Diagnostic CxA+ Training Report

## Executive summary

This diagnostic training run fits the first conservative CxA+ baseline model.
The model predicts `{TARGET_COLUMN}` from the governed CxA+ feature matrix using
only `eligible_model_features` from `feature_matrix_summary.json`.

No model is promoted in this PR. No dashboard logic is changed.

## Target and split

- Target: `{TARGET_COLUMN}`
- Split strategy: grouped train/test split by `match_id`
- Train rows: {split["train"]["row_count"]:,}
- Test rows: {split["test"]["row_count"]:,}
- Train matches: {split["train"]["match_count"]:,}
- Test matches: {split["test"]["match_count"]:,}
- Train positive rate: {split["train"]["positive_rate"]:.6f}
- Test positive rate: {split["test"]["positive_rate"]:.6f}
- Shared match count: {split["shared_match_count"]}

## Feature governance

Feature count: {len(features):,}

The trainer fails if the target, identifier columns, `created_shot_*` reference
columns, downstream/future/window labels, prediction outputs, probability
outputs, outcome/result fields, model outputs, or score fields appear in the
model feature list.

## Metrics

- Log loss: {metrics["log_loss"]:.6f}
- Brier score: {metrics["brier"]:.6f}
- ROC AUC: {_format_metric(metrics["roc_auc"])}
- Average precision / PR AUC: {metrics["average_precision"]:.6f}
- Test positive rate: {metrics["positive_rate"]:.6f}
- Baseline positive-rate log loss: {metrics["baseline_log_loss"]:.6f}
- Baseline positive-rate Brier: {metrics["baseline_brier"]:.6f}
- Log-loss lift over baseline: {metrics["log_loss_lift_over_baseline"]:.6f}
- Brier lift over baseline: {metrics["brier_lift_over_baseline"]:.6f}

## Outputs

- `outputs/modeling/cxa_plus/diagnostic_v1/models/cxa_plus_diagnostic_model.joblib`
- `outputs/modeling/cxa_plus/diagnostic_v1/results/diagnostic_metrics.json`
- `outputs/modeling/cxa_plus/diagnostic_v1/results/calibration_bins.csv`
- `outputs/modeling/cxa_plus/diagnostic_v1/results/feature_coefficients.csv`
- `outputs/modeling/cxa_plus/diagnostic_v1/results/prediction_sample.csv`

## Limitations

This is a diagnostic baseline only. Validation, model comparison, promotion,
and portfolio/dashboard presentation are intentionally out of scope.
"""


def _format_metric(value: Any) -> str:
    return "not available" if value is None else f"{float(value):.6f}"


def run_training(
    *,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    matrix = load_feature_matrix(matrix_path)
    summary = load_feature_summary(summary_path)
    features = selected_features_from_summary(summary)
    assert_feature_leakage_guard(features)
    required = [TARGET_COLUMN, GROUP_COLUMN, *features]
    missing = [column for column in required if column not in matrix.columns]
    if missing:
        raise ValueError(f"CxA+ diagnostic training matrix missing required columns: {missing}")

    train_idx, test_idx = grouped_train_test_split(
        matrix,
        test_size=test_size,
        random_state=random_state,
    )
    feature_groups = resolve_feature_groups(matrix, features)
    model = fit_model(matrix, features, train_idx)
    test_predictions = predict_positive_probability(model, matrix.iloc[test_idx], features)
    y_train = matrix.iloc[train_idx][TARGET_COLUMN].astype(int)
    y_test = matrix.iloc[test_idx][TARGET_COLUMN].astype(int)
    metrics = metric_summary(y_test, test_predictions, baseline_rate=float(y_train.mean()))
    split = split_summary(matrix, train_idx, test_idx)
    payload = {
        "metric": "cxa_plus",
        "model_version": "diagnostic_v1",
        "model_name": MODEL_NAME,
        "target": TARGET_COLUMN,
        "matrix_path": matrix_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features),
        "feature_groups": {
            "numeric": feature_groups.numeric,
            "binary": feature_groups.binary,
            "categorical": feature_groups.categorical,
        },
        "split": split,
        "metrics": metrics,
        "baseline": {
            "strategy": "train positive-rate constant probability",
            "positive_rate": float(y_train.mean()),
        },
        "promotion_status": "not_promoted",
    }

    outputs = {name: output_dir / relative for name, relative in OUTPUT_FILES.items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outputs["model"])
    _write_json(outputs["metrics"], payload)
    calibration_bins(y_test, test_predictions).to_csv(outputs["calibration"], index=False)
    feature_coefficients(model).to_csv(outputs["coefficients"], index=False)
    prediction_sample(matrix, test_idx, test_predictions).to_csv(outputs["sample"], index=False)
    outputs["report"].write_text(build_report(metrics, split, features), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_training(
        matrix_path=args.matrix,
        summary_path=args.summary,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print("Wrote CxA+ diagnostic training outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
