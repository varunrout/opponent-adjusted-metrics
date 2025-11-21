"""Context-aware CxG model building on geometry, chain, state, and defensive cues."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from opponent_adjusted.analysis.plot_utilities import configure_matplotlib
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
    load_dataset_from_versions,
)


DATASET_ENV_VAR = "CXG_DATASET_VERSION"
RUN_NAME_ENV_VAR = "CXG_CONTEXTUAL_RUN_NAME"

DATASET_FILE_MAP: dict[str, list[str]] = {
    "enriched": [
        "cxg_dataset_enriched.parquet",
        "cxg_dataset_enriched.csv",
    ],
    "filtered": [
        "cxg_dataset_filtered.parquet",
        "cxg_dataset_filtered.csv",
    ],
    "raw": [
        "cxg_dataset.parquet",
        "cxg_dataset.csv",
    ],
}
DATASET_PREFERENCE = ["enriched", "filtered", "raw"]


NUMERIC_FEATURES = [
    "shot_distance",
    "shot_angle",
    "statsbomb_xg",
    "score_diff_at_shot",
    "minute",
    "time_gap_seconds",
    "finishing_bias_logit",
    "finishing_bias_multiplier",
    "concession_bias_logit",
    "concession_bias_multiplier",
    "set_piece_logit",
    "set_piece_multiplier",
    "set_piece_modeled_prob",
    "assist_quality_logit",
    "assist_quality_multiplier",
    "assist_quality_modeled_prob",
    "pressure_logit",
    "pressure_multiplier",
    "pressure_modeled_prob",
    "def_trigger_logit",
    "def_trigger_multiplier",
    "def_trigger_modeled_prob",
]
BINARY_FEATURES = ["is_leading", "is_trailing", "is_drawing", "possession_match"]
CATEGORICAL_FEATURES = [
    "chain_label",
    "pass_style",
    "score_state",
    "simple_state",
    "minute_bucket_label",
    "assist_category",
    "pressure_state",
    "set_piece_category",
    "set_piece_phase",
    "def_label",
]

def _slugify(value: str | None) -> str:
    if not value:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    return slug.strip("-").lower()


def _load_dataset() -> tuple[pd.DataFrame, str]:
    requested = os.getenv(DATASET_ENV_VAR)
    if requested:
        key = requested.lower()
        if key not in DATASET_FILE_MAP:
            raise ValueError(
                f"Unsupported CXG dataset version '{requested}'."
                f" Choose from {list(DATASET_FILE_MAP)}."
            )
        files = DATASET_FILE_MAP[key]
        return load_dataset_from_versions("cxg", files), key

    for key in DATASET_PREFERENCE:
        files = DATASET_FILE_MAP[key]
        try:
            df = load_dataset_from_versions("cxg", files)
            return df, key
        except FileNotFoundError:
            continue
    return load_cxg_modeling_dataset(prefer_filtered=True), "auto"


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "possession_match" in df.columns:
        df["possession_match"] = df["possession_match"].astype(float)
    return df


def _filter_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    def _select(cols: list[str]) -> list[str]:
        selected = []
        for col in cols:
            if col not in df.columns:
                continue
            series = df[col]
            if hasattr(series, "notna") and series.notna().any():
                selected.append(col)
        return selected

    numeric = _select(NUMERIC_FEATURES)
    binary = _select(BINARY_FEATURES)
    categorical = _select(CATEGORICAL_FEATURES)
    return numeric, binary, categorical


def _build_pipeline(
    numeric_features: list[str], binary_features: list[str], categorical_features: list[str]
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    binary_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if binary_features:
        transformers.append(("bin", binary_pipeline, binary_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocess = ColumnTransformer(transformers=transformers)
    model = LogisticRegression(
        penalty="l2",
        C=0.5,
        solver="lbfgs",
        max_iter=1000,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def _train_and_eval(
    df: pd.DataFrame,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
    n_splits: int = 5,
) -> tuple[dict, np.ndarray, np.ndarray]:
    df = df.dropna(subset=["is_goal", "match_id"]).copy()
    y = df["is_goal"].astype(int).to_numpy()
    groups = df["match_id"].to_numpy()
    feature_cols = numeric_features + binary_features + categorical_features
    if not feature_cols:
        raise ValueError("No modeling features detected in dataset")
    X = df[feature_cols]

    pipeline = _build_pipeline(numeric_features, binary_features, categorical_features)
    gkf = GroupKFold(n_splits=n_splits)

    briers, log_losses, aucs = [], [], []
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        probs = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        y_test = y[test_idx]

        briers.append(brier_score_loss(y_test, probs))
        log_losses.append(log_loss(y_test, probs))
        aucs.append(roc_auc_score(y_test, probs))
        y_true_all.append(y_test)
        y_pred_all.append(probs)

    metrics = {
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers)),
        "log_loss_mean": float(np.mean(log_losses)),
        "log_loss_std": float(np.std(log_losses)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }

    return metrics, np.concatenate(y_true_all), np.concatenate(y_pred_all)


def _fit_full_model(
    df: pd.DataFrame,
    numeric_features: list[str],
    binary_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, pd.DataFrame]:
    pipeline = _build_pipeline(numeric_features, binary_features, categorical_features)
    target = df["is_goal"].astype(int)
    features = df[numeric_features + binary_features + categorical_features]
    pipeline.fit(features, target)

    preprocess = pipeline.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    return pipeline, coef_df


def _plot_reliability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    run_label: str | None,
) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_pred, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    empirical = []
    counts = []
    for b in range(10):
        mask = bin_ids == b
        if not np.any(mask):
            empirical.append(np.nan)
            counts.append(0)
            continue
        empirical.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots()
    ax.plot(centers, centers, "--", color="gray", label="Ideal")
    ax.plot(centers, empirical, "o-", label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical goal rate")
    ax.set_title("CxG contextual model reliability")
    ax.legend()
    for x, n in zip(centers, counts):
        ax.annotate(str(n), (x, 0), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=6)
    fig.tight_layout()
    suffix = f"_{run_label}" if run_label else ""
    fig.savefig(out_dir / f"contextual_model_reliability{suffix}.png", dpi=150)
    plt.close(fig)


def _save_metrics(metrics: dict, coef_df: pd.DataFrame, run_label: str | None) -> None:
    base_dir = get_modeling_output_dir("cxg")
    suffix = f"_{run_label}" if run_label else ""
    metrics_path = base_dir / f"contextual_metrics{suffix}.json"
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    coef_path = base_dir / f"contextual_feature_effects{suffix}.csv"
    coef_df.to_csv(coef_path, index=False)


def main() -> None:
    df_raw, dataset_label = _load_dataset()
    df = _prepare_frame(df_raw)
    run_label_env = _slugify(os.getenv(RUN_NAME_ENV_VAR))
    run_suffix = run_label_env or None
    numeric_features, binary_features, categorical_features = _filter_features(df)
    metrics, y_true, y_pred = _train_and_eval(
        df,
        numeric_features,
        binary_features,
        categorical_features,
    )
    _, coef_df = _fit_full_model(df, numeric_features, binary_features, categorical_features)

    plots_dir = get_modeling_output_dir("cxg", subdir="plots")
    _plot_reliability(y_true, y_pred, plots_dir, run_suffix)
    _save_metrics(metrics, coef_df, run_suffix)

    label_msg = run_suffix or "default"
    print(f"Contextual model metrics (dataset={dataset_label}, run={label_msg}):")
    print(json.dumps(metrics, indent=2))
    print("Feature effects and reliability plots saved with matching suffix.")


if __name__ == "__main__":  # pragma: no cover
    main()
