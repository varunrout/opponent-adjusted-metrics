"""Baseline geometry-only CxG model.

Fits a simple regularized logistic regression using only shot distance and
angle (plus an optional penalty flag) as predictors of goal probability.
This serves as the reference model against which contextual features are
measured.

Usage (from repo root):

    poetry run python -m opponent_adjusted.modeling.cxg.baseline_geometry

Outputs:
- Metrics JSON written to `outputs/modeling/cxg/baseline_metrics.json`
- Calibration / reliability plots under `outputs/modeling/cxg/plots/`
"""

from __future__ import annotations

from pathlib import Path

import json
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from opponent_adjusted.analysis.plot_utilities import configure_matplotlib
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)


def _load_dataset() -> pd.DataFrame:
    return load_cxg_modeling_dataset(prefer_filtered=True)


def _prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, y, and group labels (match_id) for CV."""

    # Drop rows without geometry or outcome.
    df = df.dropna(subset=["shot_distance", "shot_angle", "is_goal", "match_id"]).copy()

    X = df[["shot_distance", "shot_angle"]].to_numpy(dtype=float)
    y = df["is_goal"].astype(int).to_numpy()
    groups = df["match_id"].to_numpy()
    return X, y, groups


def _build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )


def _train_and_evaluate(df: pd.DataFrame, n_splits: int = 5) -> dict:
    X, y, groups = _prepare_features(df)

    gkf = GroupKFold(n_splits=n_splits)
    briers: list[float] = []
    log_losses: list[float] = []
    aucs: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []

    for train_idx, test_idx in gkf.split(X, y, groups):
        model = _build_model()
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]

        y_test = y[test_idx]
        briers.append(brier_score_loss(y_test, probs))
        log_losses.append(log_loss(y_test, probs))
        aucs.append(roc_auc_score(y_test, probs))
        y_true_all.append(y_test)
        y_pred_all.append(probs)

    y_true_concat = np.concatenate(y_true_all)
    y_pred_concat = np.concatenate(y_pred_all)

    metrics = {
        "brier_mean": float(np.mean(briers)),
        "brier_std": float(np.std(briers)),
        "log_loss_mean": float(np.mean(log_losses)),
        "log_loss_std": float(np.std(log_losses)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }

    return metrics, y_true_concat, y_pred_concat


def _fit_and_save_full_model(df: pd.DataFrame, out_dir: Path, metrics: dict) -> None:
    X, y, _ = _prepare_features(df)
    model = _build_model()
    model.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "baseline_geometry_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_type": "baseline_geometry",
        "features": ["shot_distance", "shot_angle"],
        "trained_rows": int(len(y)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    metadata_path = out_dir / "baseline_geometry_model.json"
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def _plot_reliability(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    configure_matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bin predictions into deciles and compute empirical goal rate
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(y_pred, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    empirical = []
    counts = []
    for b in range(10):
        mask = bin_ids == b
        if not np.any(mask):
            empirical.append(np.nan)
            counts.append(0)
        else:
            empirical.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))

    fig, ax = plt.subplots()
    ax.plot(bin_centers, bin_centers, "--", color="gray", label="Ideal")
    ax.plot(bin_centers, empirical, "o-", label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical goal rate")
    ax.set_title("CxG baseline geometry reliability")
    ax.legend()

    for x, n in zip(bin_centers, counts):
        ax.annotate(str(n), (x, 0), xytext=(0, -12), textcoords="offset points", ha="center", fontsize=6)

    fig.tight_layout()
    fig_path = out_dir / "baseline_geometry_reliability.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = _load_dataset()
    metrics, y_true, y_pred = _train_and_evaluate(df, n_splits=5)

    base_dir = get_modeling_output_dir("cxg")

    # Save metrics
    metrics_path = base_dir / "baseline_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot reliability
    plots_dir = get_modeling_output_dir("cxg", subdir="plots")
    _plot_reliability(y_true, y_pred, plots_dir)

    model_dir = get_modeling_output_dir("cxg", subdir="models")
    _fit_and_save_full_model(df, model_dir, metrics)

    print("Baseline geometry model metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Reliability plot written to {plots_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
