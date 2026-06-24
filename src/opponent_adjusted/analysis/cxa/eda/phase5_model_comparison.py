"""Phase 5 EDA: Model Comparison & Validation Visualizations.

Visualizations to justify modeling decisions:
- ROC curves for xA models
- Calibration curves
- Feature importance from model coefficients
- Model performance comparison
- Lift charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, brier_score_loss
from sklearn.model_selection import cross_val_predict, StratifiedKFold

logger = logging.getLogger(__name__)

# Sample size for faster CV (use all data if smaller)
MAX_SAMPLE_SIZE = 50000


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _add_assist_labels(passes: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Add is_assist labels by joining with shots data (key_pass_id matches statsbomb_event_id)."""
    shots = pd.read_parquet(repo_root / "feature_store" / "cxa" / "shots.parquet")

    # Get passes that are key passes (led to shots) - key_pass_id matches statsbomb_event_id
    key_pass_ids = set(shots["key_pass_id"].dropna())

    # Get key passes that led to goals
    goal_shots = shots[shots["is_goal"].fillna(False).astype(bool)]
    assist_pass_ids = set(goal_shots["key_pass_id"].dropna())

    # Join on statsbomb_event_id
    passes = passes.copy()
    passes["is_key_pass"] = passes["statsbomb_event_id"].isin(key_pass_ids)
    passes["is_assist"] = passes["statsbomb_event_id"].isin(assist_pass_ids)

    logger.info(f"Found {passes['is_key_pass'].sum():,} key passes leading to shots")
    logger.info(f"Found {passes['is_assist'].sum():,} assists (passes leading to goals)")

    return passes


def plot_roc_curves(passes: pd.DataFrame, features: List[str], output_dir: Path):
    """Plot ROC curves for different feature sets."""

    # Sample for speed if needed
    if len(passes) > MAX_SAMPLE_SIZE:
        # Stratified sampling to preserve class balance
        from sklearn.model_selection import train_test_split

        passes_sample, _ = train_test_split(
            passes, train_size=MAX_SAMPLE_SIZE, stratify=passes["is_assist"], random_state=42
        )
        logger.info(f"Sampled {len(passes_sample):,} passes for ROC curves")
    else:
        passes_sample = passes

    X = passes_sample[features].fillna(0)
    y = passes_sample["is_assist"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Different feature subsets to compare
    feature_sets = {
        "Location only": ["end_x", "end_y"],
        "Location + xT": ["end_x", "end_y", "xt_delta", "end_xt"],
        "All numeric": [f for f in features if f in X.columns],
    }

    colors = ["steelblue", "coral", "green"]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for (name, feat_list), color in zip(feature_sets.items(), colors):
        feat_list = [f for f in feat_list if f in X.columns]
        if len(feat_list) == 0:
            continue

        X_subset = X[feat_list].values

        # Fit model and get predictions (3-fold for speed)
        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"
        )
        y_proba = cross_val_predict(model, X_subset, y, cv=cv, method="predict_proba")[:, 1]

        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: xA Model Feature Comparison", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_comparison.png", dpi=150)
    plt.close()

    logger.info("ROC curves saved")


def plot_calibration_curves(passes: pd.DataFrame, features: List[str], output_dir: Path):
    """Plot calibration curves showing predicted vs actual probabilities."""

    # Sample for speed if needed
    if len(passes) > MAX_SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split

        passes_sample, _ = train_test_split(
            passes, train_size=MAX_SAMPLE_SIZE, stratify=passes["is_assist"], random_state=42
        )
    else:
        passes_sample = passes

    X = passes_sample[features].fillna(0).values
    y = passes_sample["is_assist"].astype(int).values

    # Fit model (3-fold for speed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_proba, n_bins=10, strategy="quantile")

    axes[0].plot(prob_pred, prob_true, "s-", color="steelblue", label="xA Baseline")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Predicted probability distribution
    axes[1].hist(y_proba[y == 0], bins=50, alpha=0.6, label="Non-assist", density=True)
    axes[1].hist(y_proba[y == 1], bins=50, alpha=0.6, label="Assist", density=True)
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Predicted Probability Distribution")
    axes[1].legend()
    axes[1].set_xlim(0, 0.1)  # Most probabilities are very low

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=150)
    plt.close()

    # Brier score
    brier = brier_score_loss(y, y_proba)
    logger.info(f"Brier score: {brier:.6f}")


def plot_feature_importance(passes: pd.DataFrame, features: List[str], output_dir: Path):
    """Plot feature importance from logistic regression coefficients."""

    X = passes[features].fillna(0)
    y = passes["is_assist"].astype(int)

    # Standardize for comparable coefficients
    X_std = (X - X.mean()) / X.std()
    X_std = X_std.fillna(0)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_std, y)

    # Get coefficients
    importance = pd.DataFrame(
        {
            "feature": features,
            "coefficient": model.coef_[0],
            "abs_coefficient": np.abs(model.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=True)

    importance.to_csv(output_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["coral" if c < 0 else "steelblue" for c in importance["coefficient"]]
    ax.barh(importance["feature"], importance["coefficient"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Feature Importance (Logistic Regression Coefficients)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    logger.info("\nTop 5 positive predictors:")
    top_pos = importance.nlargest(5, "coefficient")
    for _, row in top_pos.iterrows():
        logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

    logger.info("\nTop 5 negative predictors:")
    top_neg = importance.nsmallest(5, "coefficient")
    for _, row in top_neg.iterrows():
        logger.info(f"  {row['feature']}: {row['coefficient']:.4f}")

    return importance


def plot_lift_chart(passes: pd.DataFrame, features: List[str], output_dir: Path):
    """Plot lift chart showing model performance by decile."""

    # Sample for speed if needed
    if len(passes) > MAX_SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split

        passes_sample, _ = train_test_split(
            passes, train_size=MAX_SAMPLE_SIZE, stratify=passes["is_assist"], random_state=42
        )
    else:
        passes_sample = passes

    X = passes_sample[features].fillna(0).values
    y = passes_sample["is_assist"].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    # Create deciles
    df = pd.DataFrame({"y": y, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"], q=10, labels=range(10, 0, -1), duplicates="drop")

    lift = (
        df.groupby("decile", observed=True)
        .agg(count=("y", "count"), assists=("y", "sum"), mean_proba=("proba", "mean"))
        .reset_index()
    )

    lift["assist_rate"] = lift["assists"] / lift["count"]
    lift["cumulative_assists"] = lift["assists"].cumsum()
    lift["cumulative_pct"] = lift["cumulative_assists"] / lift["assists"].sum()

    baseline_rate = y.mean()
    lift["lift"] = lift["assist_rate"] / baseline_rate

    lift.to_csv(output_dir / "lift_chart_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Lift by decile
    axes[0].bar(lift["decile"].astype(int), lift["lift"], color="steelblue")
    axes[0].axhline(1, color="red", linestyle="--", label="Baseline")
    axes[0].set_xlabel("Decile (1=highest predicted prob)")
    axes[0].set_ylabel("Lift")
    axes[0].set_title("Lift Chart by Decile")
    axes[0].legend()

    # 2. Cumulative gains
    axes[1].plot(
        range(len(lift) + 1),
        [0] + list(lift["cumulative_pct"]),
        "s-",
        color="steelblue",
        label="Model",
    )
    axes[1].plot([0, 10], [0, 1], "k--", label="Random")
    axes[1].set_xlabel("Decile")
    axes[1].set_ylabel("Cumulative % of Assists Captured")
    axes[1].set_title("Cumulative Gains Chart")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "lift_chart.png", dpi=150)
    plt.close()

    logger.info(f"\nTop decile captures {lift.iloc[0]['cumulative_pct']*100:.1f}% of assists")
    logger.info(f"Top decile lift: {lift.iloc[0]['lift']:.2f}x baseline")


def plot_precision_recall(passes: pd.DataFrame, features: List[str], output_dir: Path):
    """Plot precision-recall curve (better for imbalanced classes)."""

    # Sample for speed if needed
    if len(passes) > MAX_SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split

        passes_sample, _ = train_test_split(
            passes, train_size=MAX_SAMPLE_SIZE, stratify=passes["is_assist"], random_state=42
        )
    else:
        passes_sample = passes

    X = passes_sample[features].fillna(0).values
    y = passes_sample["is_assist"].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, y_proba)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(recall, precision, color="steelblue", lw=2)
    ax.axhline(y.mean(), color="red", linestyle="--", label=f"Baseline ({y.mean():.4f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=150)
    plt.close()


def plot_model_comparison_summary(output_dir: Path):
    """Create summary comparison of all 4 cXA models."""

    # Model characteristics (from our implementation)
    models = pd.DataFrame(
        {
            "Model": ["xA Baseline", "xA+ Passes", "xA+ Actions", "cXA-xG"],
            "Target": ["is_assist", "goal sequences", "goal sequences", "all shots"],
            "Actions": ["Passes only", "Passes only", "Pass+Carry+Dribble", "Pass+Carry"],
            "Credit Method": ["P(assist)", "Softmax log-odds", "Softmax proba", "Softmax × xG"],
            "Calibration": ["Sum=Assists", "Sum=Goals", "Sum=Goals", "Sum=xG"],
            "Coverage": ["All passes", "Goal sequences", "Goal sequences", "All shots"],
        }
    )

    models.to_csv(output_dir / "model_comparison_table.csv", index=False)

    # Create visual comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    table = ax.table(
        cellText=models.values,
        colLabels=models.columns,
        cellLoc="center",
        loc="center",
        colColours=["lightsteelblue"] * len(models.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title("cXA Model Comparison", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("\nModel comparison table saved")


def run_phase5_eda(output_dir: Path = None) -> Dict[str, Any]:
    """Run Phase 5 EDA: Model Comparison & Validation."""

    repo_root = _get_repo_root()
    if output_dir is None:
        output_dir = repo_root / "outputs" / "analysis" / "cxa" / "eda" / "phase5_model_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 5 EDA: Model Comparison & Validation")
    logger.info("=" * 60)

    # Load passes data
    passes = pd.read_parquet(repo_root / "feature_store" / "cxa" / "passes.parquet")
    logger.info(f"Loaded {len(passes):,} passes")

    # Add assist labels from shots data
    passes = _add_assist_labels(passes, repo_root)

    # Define features for modeling
    numeric_features = [
        "end_x",
        "end_y",
        "start_x",
        "start_y",
        "pass_length",
        "xt_delta",
        "end_xt",
        "start_xt",
    ]
    boolean_features = [
        "is_cross",
        "is_through_ball",
        "is_into_box",
        "is_progressive",
        "is_final_third",
        "under_pressure",
    ]

    # Filter to available features
    features = [f for f in numeric_features + boolean_features if f in passes.columns]
    logger.info(f"Using {len(features)} features: {features}")

    results = {}

    # 1. ROC Curves
    logger.info("\n--- 1. ROC Curves ---")
    plot_roc_curves(passes, features, output_dir)

    # 2. Calibration Curves
    logger.info("\n--- 2. Calibration Curves ---")
    plot_calibration_curves(passes, features, output_dir)

    # 3. Feature Importance
    logger.info("\n--- 3. Feature Importance ---")
    importance = plot_feature_importance(passes, features, output_dir)
    results["feature_importance"] = importance

    # 4. Lift Chart
    logger.info("\n--- 4. Lift Chart ---")
    plot_lift_chart(passes, features, output_dir)

    # 5. Precision-Recall
    logger.info("\n--- 5. Precision-Recall ---")
    plot_precision_recall(passes, features, output_dir)

    # 6. Model Comparison Summary
    logger.info("\n--- 6. Model Comparison Summary ---")
    plot_model_comparison_summary(output_dir)

    logger.info(f"\nPhase 5 EDA complete. Outputs saved to {output_dir}")

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_phase5_eda()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
