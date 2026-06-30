"""Train diagnostic-informed CxG candidate models.

This training layer sits after pre-model CxG diagnostics and before the fuller
validation/promotion work. It keeps the existing baseline CxG path intact and
writes separate diagnostic_v1 artifacts.
"""

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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from scripts.run_cxg_end_to_end import load_cxg_features
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from run_cxg_end_to_end import load_cxg_features

DEFAULT_CONTRACT_PATH = Path("configs/feature_contracts/cxg_diagnostic_v1.json")
DEFAULT_OUTPUT_DIR = Path("outputs/modeling/cxg/diagnostic_v1")
PREDICTION_ID_COLUMNS = ["shot_id", "event_id", "match_id", "team_id", "player_id"]


@dataclass(frozen=True)
class ResolvedFeatures:
    """Contract features resolved against the actual input frame."""

    numeric: list[str]
    binary: list[str]
    categorical: list[str]
    geometry_numeric: list[str]
    unavailable: dict[str, list[str]]
    excluded_present: list[str]
    reference_present: list[str]

    @property
    def all_features(self) -> list[str]:
        return self.numeric + self.binary + self.categorical


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Collapse infrequent categories before one-hot encoding."""

    def __init__(self, min_count: int = 30, replacement: str = "__rare__") -> None:
        self.min_count = min_count
        self.replacement = replacement
        self.frequent_values_: dict[str, set[str]] = {}

    def fit(self, X: Any, y: Any = None) -> "RareCategoryCollapser":
        frame = _as_frame(X)
        self.frequent_values_ = {}
        for column in frame.columns:
            counts = frame[column].fillna("__missing__").astype(str).value_counts()
            self.frequent_values_[column] = set(counts[counts >= self.min_count].index)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = _as_frame(X).copy()
        for column in frame.columns:
            frequent = self.frequent_values_.get(column, set())
            values = frame[column].fillna("__missing__").astype(str)
            frame[column] = values.where(values.isin(frequent), self.replacement)
        return frame


def _as_frame(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _coerce_binary_frame(X: Any) -> pd.DataFrame:
    return _as_frame(X).astype(float)


def load_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load a diagnostic CxG feature contract."""

    return json.loads(path.read_text(encoding="utf-8"))


def resolve_features(df: pd.DataFrame, contract: dict[str, Any]) -> ResolvedFeatures:
    """Resolve contract features present in an input frame."""

    numeric_contract = list(contract.get("eligible_numeric_features", []))
    binary_contract = list(contract.get("eligible_binary_features", []))
    categorical_contract = list(contract.get("eligible_categorical_features", []))
    reference = list(contract.get("reference_only_columns", []))
    leakage = list(contract.get("excluded_leakage_columns", []))
    target = contract["target_column"]
    group = contract["group_column"]

    forbidden = set(reference + leakage + [target])
    present = set(df.columns)

    def _resolve(columns: list[str]) -> list[str]:
        return [column for column in columns if column in present and column not in forbidden]

    numeric = _resolve(numeric_contract)
    binary = _resolve(binary_contract)
    categorical = _resolve(categorical_contract)
    geometry = [
        column
        for column in contract.get("feature_groups", {}).get("geometry", [])
        if column in numeric
    ]

    excluded_present = [column for column in leakage if column in present]
    reference_present = [column for column in reference if column in present]
    unavailable = {
        "numeric": [column for column in numeric_contract if column not in present],
        "binary": [column for column in binary_contract if column not in present],
        "categorical": [column for column in categorical_contract if column not in present],
        "required": [column for column in (target, group) if column not in present],
    }
    if unavailable["required"]:
        raise ValueError(
            f"CxG diagnostic training missing required columns: {unavailable['required']}"
        )
    return ResolvedFeatures(
        numeric=numeric,
        binary=binary,
        categorical=categorical,
        geometry_numeric=geometry,
        unavailable=unavailable,
        excluded_present=excluded_present,
        reference_present=reference_present,
    )


def validate_no_forbidden_features(features: list[str], contract: dict[str, Any]) -> None:
    """Raise if a model matrix would include leakage or reference columns."""

    forbidden = set(contract.get("reference_only_columns", []))
    forbidden.update(contract.get("excluded_leakage_columns", []))
    forbidden.add(contract["target_column"])
    used = sorted(set(features).intersection(forbidden))
    if used:
        raise ValueError(f"Forbidden leakage/reference columns passed to model matrix: {used}")


def _preprocessor(
    numeric: list[str],
    binary: list[str],
    categorical: list[str],
    *,
    min_category_count: int,
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    binary_pipeline = Pipeline(
        [
            ("coerce", FunctionTransformer(_coerce_binary_frame, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryCollapser(min_count=min_category_count)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("num", numeric_pipeline, numeric))
    if binary:
        transformers.append(("bin", binary_pipeline, binary))
    if categorical:
        transformers.append(("cat", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers)


def _candidate_pipeline(
    candidate_name: str,
    numeric: list[str],
    binary: list[str],
    categorical: list[str],
    *,
    min_category_count: int,
    random_state: int,
) -> Pipeline:
    if candidate_name == "geometry_logistic":
        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        scale = True
    elif candidate_name == "diagnostic_logistic":
        model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        scale = True
    elif candidate_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state)
        scale = False
    elif candidate_name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=120,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        )
        scale = False
    else:
        raise ValueError(f"Unsupported CxG diagnostic candidate: {candidate_name}")
    return Pipeline(
        [
            (
                "preprocess",
                _preprocessor(
                    numeric,
                    binary,
                    categorical,
                    min_category_count=min_category_count,
                    scale_numeric=scale,
                ),
            ),
            ("model", model),
        ]
    )


def _splitter(
    df: pd.DataFrame,
    y: np.ndarray,
    group_column: str,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    if min_class_count < 2:
        raise ValueError("CxG diagnostic training requires at least two examples per class.")
    group_count = int(df[group_column].nunique()) if group_column in df.columns else 0
    n_splits = max(2, min(5, min_class_count, max(group_count, 2)))
    if group_column in df.columns and group_count >= n_splits:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df, y, groups=df[group_column].to_numpy()))
        metadata = {"splitter": "GroupKFold", "n_splits": n_splits, "group_column": group_column}
    else:
        n_splits = max(2, min(5, min_class_count))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(splitter.split(df, y))
        metadata = {
            "splitter": "StratifiedKFold",
            "n_splits": n_splits,
            "fallback_reason": f"{group_column} unavailable or too few groups",
        }
    return splits, metadata


def _fold_metric_row(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    candidate: str,
    fold: int,
    train_rows: int,
    test_rows: int,
    feature_count: int,
    excluded_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_candidate": candidate,
        "fold": fold,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "row_count": test_rows,
        "goal_count": int(y_true.sum()),
        "goal_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "mean_predicted_probability": float(np.mean(probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "log_loss": float(log_loss(y_true, probs, labels=[0, 1])),
        "feature_count": feature_count,
        "excluded_leakage_reference_column_count": excluded_count,
    }
    if len(np.unique(y_true)) == 2:
        row["roc_auc"] = float(roc_auc_score(y_true, probs))
        row["roc_auc_status"] = "computed"
    else:
        row["roc_auc"] = np.nan
        row["roc_auc_status"] = "skipped_single_class_fold"
    return row


def train_diagnostic_candidates(
    df: pd.DataFrame,
    contract: dict[str, Any],
    *,
    min_category_count: int,
    random_state: int,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train CxG diagnostic candidates and return the selected final model plus artifacts."""

    resolved = resolve_features(df, contract)
    y = df[contract["target_column"]].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("CxG diagnostic training data must contain both goals and non-goals.")
    splits, split_metadata = _splitter(df, y, contract["group_column"], random_state)

    candidate_defs = contract.get("model_candidates", [])
    fold_rows = []
    prediction_parts = []
    candidate_metadata = []
    for candidate in candidate_defs:
        name = candidate["name"]
        if candidate.get("feature_scope") == "geometry_only":
            numeric = resolved.geometry_numeric or [
                column for column in ("shot_distance", "shot_angle") if column in resolved.numeric
            ]
            binary: list[str] = []
            categorical: list[str] = []
        else:
            numeric = resolved.numeric
            binary = resolved.binary
            categorical = resolved.categorical
        feature_cols = numeric + binary + categorical
        validate_no_forbidden_features(feature_cols, contract)
        if not feature_cols:
            raise ValueError(f"No features resolved for CxG candidate {name}")

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            if len(np.unique(y[train_idx])) < 2:
                probs = np.full(len(test_idx), np.clip(y[train_idx].mean(), 1e-6, 1 - 1e-6))
            else:
                model = _candidate_pipeline(
                    name,
                    numeric,
                    binary,
                    categorical,
                    min_category_count=min_category_count,
                    random_state=random_state,
                )
                model.fit(df.iloc[train_idx][feature_cols], y[train_idx])
                probs = model.predict_proba(df.iloc[test_idx][feature_cols])[:, 1]
            fold_rows.append(
                _fold_metric_row(
                    y[test_idx],
                    probs,
                    candidate=name,
                    fold=fold,
                    train_rows=len(train_idx),
                    test_rows=len(test_idx),
                    feature_count=len(feature_cols),
                    excluded_count=len(resolved.excluded_present) + len(resolved.reference_present),
                )
            )
            prediction_parts.append(
                _prediction_frame(
                    df.iloc[test_idx],
                    y[test_idx],
                    probs,
                    candidate=name,
                    fold=fold,
                )
            )
        candidate_metadata.append(
            {
                "name": name,
                "type": candidate.get("type"),
                "feature_scope": candidate.get("feature_scope"),
                "features": {"numeric": numeric, "binary": binary, "categorical": categorical},
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    comparison = _comparison_table(fold_metrics)
    selected_name = _select_candidate(comparison)
    selected_meta = next(item for item in candidate_metadata if item["name"] == selected_name)
    final_model = _candidate_pipeline(
        selected_name,
        selected_meta["features"]["numeric"],
        selected_meta["features"]["binary"],
        selected_meta["features"]["categorical"],
        min_category_count=min_category_count,
        random_state=random_state,
    )
    final_features = (
        selected_meta["features"]["numeric"]
        + selected_meta["features"]["binary"]
        + selected_meta["features"]["categorical"]
    )
    final_model.fit(df[final_features], y)
    metadata = {
        "selected_model": selected_name,
        "selected_reason": _selection_reason(comparison, selected_name),
        "split_metadata": split_metadata,
        "resolved_features": {
            "numeric": resolved.numeric,
            "binary": resolved.binary,
            "categorical": resolved.categorical,
            "geometry_numeric": resolved.geometry_numeric,
            "unavailable": resolved.unavailable,
            "excluded_present": resolved.excluded_present,
            "reference_present": resolved.reference_present,
        },
        "model_candidates": candidate_metadata,
    }
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return final_model, metadata, comparison, fold_metrics, predictions


def _prediction_frame(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    candidate: str,
    fold: int,
) -> pd.DataFrame:
    prediction = pd.DataFrame(index=rows.index)
    for column in PREDICTION_ID_COLUMNS:
        if column in rows.columns:
            prediction[column] = rows[column].to_numpy()
    prediction["is_goal"] = y_true
    prediction["predicted_cxg"] = probs
    prediction["model_candidate"] = candidate
    prediction["fold"] = fold
    prediction["prediction_source"] = "cross_validated"
    return prediction.reset_index(drop=True)


def _comparison_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_metrics.groupby("model_candidate", as_index=False).agg(
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        log_loss_mean=("log_loss", "mean"),
        log_loss_std=("log_loss", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        row_count=("row_count", "sum"),
        goal_count=("goal_count", "sum"),
        goal_rate=("goal_rate", "mean"),
        mean_predicted_probability=("mean_predicted_probability", "mean"),
        feature_count=("feature_count", "max"),
        excluded_leakage_reference_column_count=(
            "excluded_leakage_reference_column_count",
            "max",
        ),
    )
    grouped["selection_rank"] = (
        grouped["log_loss_mean"].rank(method="first")
        + grouped["brier_mean"].rank(method="first") * 0.25
    )
    return grouped.sort_values(["selection_rank", "log_loss_mean", "brier_mean"])


def _select_candidate(comparison: pd.DataFrame) -> str:
    return str(comparison.iloc[0]["model_candidate"])


def _selection_reason(comparison: pd.DataFrame, selected: str) -> str:
    row = comparison.loc[comparison["model_candidate"] == selected].iloc[0]
    return (
        f"Selected {selected} provisionally because it had the strongest combined "
        f"log-loss/Brier ranking in training comparison "
        f"(log_loss_mean={row['log_loss_mean']:.4f}, brier_mean={row['brier_mean']:.4f})."
    )


def write_training_artifacts(
    *,
    output_dir: Path,
    contract: dict[str, Any],
    model: Pipeline,
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    input_path: Path,
) -> dict[str, Path]:
    """Write all required diagnostic CxG training artifacts."""

    contracts_dir = output_dir / "contracts"
    diagnostics_dir = output_dir / "diagnostics"
    models_dir = output_dir / "models"
    predictions_dir = output_dir / "predictions"
    reports_dir = output_dir / "reports"
    for directory in (contracts_dir, diagnostics_dir, models_dir, predictions_dir, reports_dir):
        directory.mkdir(parents=True, exist_ok=True)

    contract_path = contracts_dir / "feature_contract.json"
    excluded_path = diagnostics_dir / "excluded_columns.csv"
    resolved_path = diagnostics_dir / "resolved_features.json"
    feature_group_path = diagnostics_dir / "feature_group_summary.csv"
    candidates_path = models_dir / "model_candidates.json"
    metadata_path = models_dir / "selected_model_metadata.json"
    model_path = models_dir / "selected_model.joblib"
    predictions_path = predictions_dir / "cross_validated_predictions.parquet"
    comparison_path = reports_dir / "model_comparison.csv"
    fold_path = reports_dir / "fold_metrics.csv"
    report_path = reports_dir / "training_report.md"
    summary_path = reports_dir / "training_summary.json"

    contract_to_write = dict(contract)
    contract_to_write["selected_model"] = metadata["selected_model"]
    contract_path.write_text(json.dumps(contract_to_write, indent=2), encoding="utf-8")
    candidates_path.write_text(
        json.dumps(metadata["model_candidates"], indent=2),
        encoding="utf-8",
    )
    resolved_features = metadata["resolved_features"]
    resolved_path.write_text(json.dumps(resolved_features, indent=2), encoding="utf-8")
    _excluded_columns_table(resolved_features).to_csv(excluded_path, index=False)
    _feature_group_summary(contract, resolved_features).to_csv(feature_group_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    predictions.to_parquet(predictions_path, index=False)
    joblib.dump(model, model_path)

    selected_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_input_path": str(input_path),
        "artifact_path": str(model_path),
        **metadata,
    }
    metadata_path.write_text(json.dumps(selected_metadata, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_training_summary(selected_metadata, comparison, fold_metrics), indent=2),
        encoding="utf-8",
    )
    report_path.write_text(
        _training_report(contract_to_write, selected_metadata, comparison),
        encoding="utf-8",
    )
    return {
        "feature_contract": contract_path,
        "excluded_columns": excluded_path,
        "resolved_features": resolved_path,
        "feature_group_summary": feature_group_path,
        "model_candidates": candidates_path,
        "model_comparison": comparison_path,
        "fold_metrics": fold_path,
        "selected_model_metadata": metadata_path,
        "selected_model": model_path,
        "cross_validated_predictions": predictions_path,
        "training_report": report_path,
        "training_summary": summary_path,
    }


def _excluded_columns_table(resolved_features: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"column": column, "reason": "reference_only"}
        for column in resolved_features.get("reference_present", [])
    ]
    rows.extend(
        {"column": column, "reason": "excluded_leakage"}
        for column in resolved_features.get("excluded_present", [])
    )
    return pd.DataFrame(rows, columns=["column", "reason"])


def _feature_group_summary(
    contract: dict[str, Any], resolved_features: dict[str, Any]
) -> pd.DataFrame:
    resolved = set(
        resolved_features.get("numeric", [])
        + resolved_features.get("binary", [])
        + resolved_features.get("categorical", [])
    )
    rows = []
    for group_name, columns in contract.get("feature_groups", {}).items():
        available = [column for column in columns if column in resolved]
        rows.append(
            {
                "feature_group": group_name,
                "contract_column_count": len(columns),
                "resolved_column_count": len(available),
                "resolved_columns": ", ".join(available),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_group",
            "contract_column_count",
            "resolved_column_count",
            "resolved_columns",
        ],
    )


def _training_summary(
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
    fold_metrics: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "selected_model": metadata["selected_model"],
        "selected_reason": metadata["selected_reason"],
        "split_metadata": metadata["split_metadata"],
        "candidate_count": int(comparison["model_candidate"].nunique()),
        "fold_count": int(fold_metrics["fold"].nunique()),
        "row_count": int(comparison["row_count"].max()) if not comparison.empty else 0,
        "goal_count": int(comparison["goal_count"].max()) if not comparison.empty else 0,
        "model_comparison": comparison.to_dict(orient="records"),
    }


def _training_report(
    contract: dict[str, Any],
    metadata: dict[str, Any],
    comparison: pd.DataFrame,
) -> str:
    resolved = metadata["resolved_features"]
    return "\n".join(
        [
            "# Diagnostic-Informed CxG Training",
            "",
            "Diagnostic-informed training means the model matrix is built from the pre-model "
            "CxG analysis decisions: eligible features are separated from reference-only and "
            "leakage columns before candidate models are compared.",
            "",
            "## Diagnostic Inputs",
            "",
            "The feature contract references these diagnostic outputs:",
            "",
            *[f"- `{path}`" for path in contract.get("diagnostic_sources", [])],
            "",
            "## Feature Groups Used",
            "",
            f"- Numeric: {', '.join(resolved['numeric']) or 'none'}",
            f"- Binary: {', '.join(resolved['binary']) or 'none'}",
            f"- Categorical: {', '.join(resolved['categorical']) or 'none'}",
            "",
            "## Leakage And Reference Exclusions",
            "",
            f"- Reference-only columns present: {', '.join(resolved['reference_present']) or 'none'}",
            f"- Excluded leakage columns present: {', '.join(resolved['excluded_present']) or 'none'}",
            "",
            "## Candidate Models",
            "",
            *[
                f"- `{row.model_candidate}`: log loss {row.log_loss_mean:.4f}, "
                f"Brier {row.brier_mean:.4f}, ROC AUC {row.roc_auc_mean:.4f}"
                for row in comparison.itertuples()
            ],
            "",
            "## Provisional Selection",
            "",
            metadata["selected_reason"],
            "",
            "## Remaining Work",
            "",
            "Issue #56 should validate the selected model across calibration, reliability, "
            "slices, stability, and diagnostic monitoring. Issue #57 should handle final "
            "prediction/result promotion and reporting.",
            "",
        ]
    )


def run_diagnostic_training(
    *,
    input_path: Path | None = None,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_category_count: int | None = None,
    random_state: int = 42,
) -> dict[str, Path]:
    """Run diagnostic-informed CxG training and write artifacts."""

    contract = load_contract(contract_path)
    if min_category_count is None:
        min_category_count = int(contract.get("sparse_category_rules", {}).get("min_count", 30))
    df, resolved_input = load_cxg_features(input_path)
    resolved = resolve_features(df, contract)
    validate_no_forbidden_features(resolved.all_features, contract)
    model, metadata, comparison, fold_metrics, predictions = train_diagnostic_candidates(
        df,
        contract,
        min_category_count=min_category_count,
        random_state=random_state,
    )
    return write_training_artifacts(
        output_dir=output_dir,
        contract=contract,
        model=model,
        metadata=metadata,
        comparison=comparison,
        fold_metrics=fold_metrics,
        predictions=predictions,
        input_path=resolved_input,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostic-informed CxG training")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-category-count", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_diagnostic_training(
        input_path=args.input,
        contract_path=args.contract,
        output_dir=args.output_dir,
        min_category_count=args.min_category_count,
        random_state=args.random_state,
    )
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
