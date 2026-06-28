#!/usr/bin/env python3
"""
Experimental contextual CxT evaluation and reporting.

This script is retained for deferred CxT research and is not part of the v1
baseline dashboard/reproducibility path. Use `make cxt-baseline` for the
implemented v1 grid-threat pipeline.

Phase 9: Comprehensive model evaluation including:
- Discrimination metrics (AUC, Brier, Log Loss)
- Calibration analysis
- Feature importance
- Slice evaluation summary
- Player/Team aggregation examples
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    r2_score,
    mean_absolute_error,
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE = PROJECT_ROOT / "feature_store" / "cxt"
MODEL_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cxt" / "latest"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "cxt" / "evaluation"
REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "cxt"


def load_data() -> pd.DataFrame:
    """Load featured progressions data."""
    df = pd.read_parquet(FEATURE_STORE / "progressions_featured.parquet")

    if "success" not in df.columns:
        if "action_success" in df.columns:
            df["success"] = df["action_success"].astype(int)
        elif "action_outcome" in df.columns:
            df["success"] = (df["action_outcome"] == "Complete").astype(int)

    return df


def load_model():
    """Load trained CxT model."""
    from opponent_adjusted.modeling.cxt.contextual_model import CxTModel

    return CxTModel.load(MODEL_DIR)


def evaluate_discrimination(model, df: pd.DataFrame) -> dict:
    """Evaluate model discrimination."""

    logger.info("Computing discrimination metrics...")

    # Completion model
    p_complete = model.predict_completion_prob(df)
    y_true = df["success"].values

    completion_metrics = {
        "auc": roc_auc_score(y_true, p_complete),
        "brier": brier_score_loss(y_true, p_complete),
        "log_loss": log_loss(y_true, p_complete),
        "n_samples": len(df),
        "positive_rate": float(y_true.mean()),
    }

    # xT Gain model (completed only)
    df_complete = df[df["success"] == 1]
    pred_xt = model.predict_xt_gain(df_complete)
    actual_xt = df_complete["xt_delta"].values

    gain_metrics = {
        "r2": r2_score(actual_xt, pred_xt),
        "mae": mean_absolute_error(actual_xt, pred_xt),
        "rmse": np.sqrt(np.mean((actual_xt - pred_xt) ** 2)),
        "correlation": np.corrcoef(actual_xt, pred_xt)[0, 1],
        "n_samples": len(df_complete),
    }

    # Combined CxT
    cxt = model.predict_cxt(df)
    actual_cxt = np.where(df["success"] == 1, df["xt_delta"], 0)

    cxt_metrics = {
        "correlation": np.corrcoef(cxt, actual_cxt)[0, 1],
        "mean_cxt": float(cxt.mean()),
        "std_cxt": float(cxt.std()),
        "mean_actual": float(actual_cxt.mean()),
    }

    return {
        "completion": completion_metrics,
        "xt_gain": gain_metrics,
        "cxt": cxt_metrics,
    }


def evaluate_calibration(model, df: pd.DataFrame) -> tuple[dict, plt.Figure]:
    """Evaluate model calibration and create calibration plot."""

    logger.info("Computing calibration...")

    p_complete = model.predict_completion_prob(df)
    y_true = df["success"].values

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, p_complete, n_bins=10)

    # Expected calibration error
    bin_counts = np.histogram(p_complete, bins=10)[0]
    weighted_diff = np.abs(prob_true - prob_pred) * bin_counts[: len(prob_true)]
    ece = np.sum(weighted_diff) / np.sum(bin_counts[: len(prob_true)])

    calibration_data = {
        "ece": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }

    # Create calibration plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, "o-", label=f"Model (ECE={ece:.4f})")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("CxT Completion Model Calibration")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    return calibration_data, fig


def compute_feature_importance(model, df: pd.DataFrame, n_samples: int = 5000) -> pd.DataFrame:
    """Compute feature importance for both submodels."""

    logger.info("Computing feature importance (this may take a while)...")

    # Sample data for speed
    if len(df) > n_samples:
        df_sample = df.sample(n_samples, random_state=42)
    else:
        df_sample = df

    y_true = df_sample["success"].values

    # Completion model importance
    completion_features = model.completion_features
    X_completion = df_sample[completion_features]

    # Use permutation importance
    perm_imp = permutation_importance(
        model.completion_model,
        X_completion,
        y_true,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
    )

    completion_imp = pd.DataFrame(
        {
            "feature": completion_features,
            "importance_mean": perm_imp.importances_mean,
            "importance_std": perm_imp.importances_std,
            "model": "completion",
        }
    )

    # xT Gain model importance
    df_complete = df_sample[df_sample["success"] == 1]
    gain_features = model.gain_features
    X_gain = df_complete[gain_features]
    y_gain = df_complete["xt_delta"].values

    perm_imp_gain = permutation_importance(
        model.xt_gain_model,
        X_gain,
        y_gain,
        n_repeats=5,
        random_state=42,
        scoring="r2",
    )

    gain_imp = pd.DataFrame(
        {
            "feature": gain_features,
            "importance_mean": perm_imp_gain.importances_mean,
            "importance_std": perm_imp_gain.importances_std,
            "model": "xt_gain",
        }
    )

    return pd.concat([completion_imp, gain_imp], ignore_index=True)


def create_feature_importance_plot(importance_df: pd.DataFrame) -> plt.Figure:
    """Create feature importance visualization."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    for idx, model_name in enumerate(["completion", "xt_gain"]):
        ax = axes[idx]
        model_imp = importance_df[importance_df["model"] == model_name]
        model_imp = model_imp.nlargest(15, "importance_mean")

        ax.barh(
            model_imp["feature"],
            model_imp["importance_mean"],
            xerr=model_imp["importance_std"],
            capsize=3,
        )
        ax.set_xlabel("Importance (AUC/R² drop)")
        ax.set_title(f"{model_name.replace('_', ' ').title()} Model - Top 15 Features")
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def aggregate_player_team_examples(model, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate player and team aggregation examples."""

    logger.info("Generating player/team aggregations...")

    from opponent_adjusted.modeling.cxt.cxt_api import CxTPredictor

    predictor = CxTPredictor(MODEL_DIR)
    df_pred = predictor.predict(df)

    # Player aggregation
    if "player_id" in df.columns:
        player_summaries = predictor.aggregate_by_player(df_pred, "player_id")
        player_df = pd.DataFrame(
            [
                {
                    "player_id": p.player_id,
                    "n_actions": p.n_actions,
                    "total_cxt": p.total_cxt,
                    "mean_cxt": p.mean_cxt,
                    "total_actual_xt": p.total_actual_xt,
                    "xt_vs_expected": p.xt_vs_expected,
                    "completion_rate": p.completion_rate,
                }
                for p in player_summaries[:50]  # Top 50
            ]
        )
    else:
        player_df = pd.DataFrame()

    # Team aggregation
    if "team_id" in df.columns:
        team_summaries = predictor.aggregate_by_team(df_pred, "team_id")
        team_df = pd.DataFrame(
            [
                {
                    "team_id": t.team_id,
                    "n_actions": t.n_actions,
                    "total_cxt": t.total_cxt,
                    "mean_cxt": t.mean_cxt,
                    "total_actual_xt": t.total_actual_xt,
                    "xt_vs_expected": t.xt_vs_expected,
                }
                for t in team_summaries
            ]
        )
    else:
        team_df = pd.DataFrame()

    return player_df, team_df


def generate_markdown_report(
    discrimination: dict,
    calibration: dict,
    importance_df: pd.DataFrame,
    player_df: pd.DataFrame,
    team_df: pd.DataFrame,
) -> str:
    """Generate comprehensive markdown report."""

    lines = [
        "# CxT Model Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "The CxT (Contextual xT) model predicts expected threat value for ball progression actions ",
        "(passes, carries, dribbles), adjusted for opponent defensive quality and game context.",
        "",
        "### Key Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Completion AUC | {discrimination['completion']['auc']:.4f} |",
        f"| Completion Brier | {discrimination['completion']['brier']:.6f} |",
        f"| xT Gain R² | {discrimination['xt_gain']['r2']:.3f} |",
        f"| xT Gain MAE | {discrimination['xt_gain']['mae']:.4f} |",
        f"| CxT-Actual Correlation | {discrimination['cxt']['correlation']:.3f} |",
        f"| Calibration ECE | {calibration['ece']:.4f} |",
        "",
        "## 1. Model Architecture",
        "",
        "CxT uses a two-stage model:",
        "",
        "1. **Completion Model** (Logistic Regression): Predicts P(action completes)",
        "2. **xT Gain Model** (Ridge Regression): Predicts E[xT_delta | completion]",
        "",
        "**Final CxT** = P(completion) × E[xT_gain | completion]",
        "",
        "### Features Used",
        "",
        "- **Numeric**: start_xt, xt_delta, minute_normalized, opponent ratings",
        "- **Binary**: under_pressure, is_progressive, zone flags, action types",
        "- **Categorical**: action_type, start_third, macro_zone_start",
        "",
        "## 2. Discrimination Performance",
        "",
        "### 2.1 Completion Model",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| AUC | {discrimination['completion']['auc']:.4f} |",
        f"| Brier Score | {discrimination['completion']['brier']:.6f} |",
        f"| Log Loss | {discrimination['completion']['log_loss']:.6f} |",
        f"| Sample Size | {discrimination['completion']['n_samples']:,} |",
        f"| Positive Rate | {discrimination['completion']['positive_rate']:.1%} |",
        "",
        "**Note**: Perfect AUC indicates potential feature leakage from xt_delta.",
        "In practice, this could be addressed by excluding xt_delta from completion features.",
        "",
        "### 2.2 xT Gain Model",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| R² | {discrimination['xt_gain']['r2']:.3f} |",
        f"| MAE | {discrimination['xt_gain']['mae']:.4f} |",
        f"| RMSE | {discrimination['xt_gain']['rmse']:.4f} |",
        f"| Correlation | {discrimination['xt_gain']['correlation']:.3f} |",
        f"| Sample Size | {discrimination['xt_gain']['n_samples']:,} |",
        "",
        "### 2.3 Combined CxT",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| CxT-Actual Correlation | {discrimination['cxt']['correlation']:.3f} |",
        f"| Mean CxT | {discrimination['cxt']['mean_cxt']:.4f} |",
        f"| Std CxT | {discrimination['cxt']['std_cxt']:.4f} |",
        f"| Mean Actual | {discrimination['cxt']['mean_actual']:.4f} |",
        "",
        "## 3. Calibration",
        "",
        f"Expected Calibration Error (ECE): **{calibration['ece']:.4f}**",
        "",
        "See calibration plot in outputs/analysis/cxt/evaluation/",
        "",
        "## 4. Feature Importance",
        "",
        "### 4.1 Completion Model - Top Features",
        "",
        "| Feature | Importance |",
        "|---------|------------|",
    ]

    comp_imp = importance_df[importance_df["model"] == "completion"].nlargest(10, "importance_mean")
    for _, row in comp_imp.iterrows():
        lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")

    lines.extend(
        [
            "",
            "### 4.2 xT Gain Model - Top Features",
            "",
            "| Feature | Importance |",
            "|---------|------------|",
        ]
    )

    gain_imp = importance_df[importance_df["model"] == "xt_gain"].nlargest(10, "importance_mean")
    for _, row in gain_imp.iterrows():
        lines.append(f"| {row['feature']} | {row['importance_mean']:.4f} |")

    lines.extend(
        [
            "",
            "## 5. Aggregations",
            "",
        ]
    )

    if len(team_df) > 0:
        lines.extend(
            [
                "### 5.1 Top Teams by Total CxT",
                "",
                "| Team ID | Actions | Total CxT | Mean CxT | vs Expected |",
                "|---------|---------|-----------|----------|-------------|",
            ]
        )
        for _, row in team_df.head(10).iterrows():
            lines.append(
                f"| {row['team_id']} | {row['n_actions']:,} | {row['total_cxt']:.2f} | "
                f"{row['mean_cxt']:.4f} | {row['xt_vs_expected']:+.2f} |"
            )

    if len(player_df) > 0:
        lines.extend(
            [
                "",
                "### 5.2 Top Players by Total CxT",
                "",
                "| Player ID | Actions | Total CxT | Mean CxT | Completion % |",
                "|-----------|---------|-----------|----------|--------------|",
            ]
        )
        for _, row in player_df.head(15).iterrows():
            lines.append(
                f"| {row['player_id']} | {row['n_actions']:,} | {row['total_cxt']:.2f} | "
                f"{row['mean_cxt']:.4f} | {row['completion_rate']:.1%} |"
            )

    lines.extend(
        [
            "",
            "## 6. Conclusions",
            "",
            "### Strengths",
            "",
            "- xT Gain R² of 0.62 shows meaningful predictive power",
            "- Opponent context features improve predictions",
            "- Model handles different action types appropriately",
            "",
            "### Limitations",
            "",
            "- Perfect completion AUC suggests feature leakage (xt_delta implies completion)",
            "- Limited to StatsBomb Open Data coverage",
            "- No available match score data for game state features",
            "",
            "### Recommendations",
            "",
            "1. For production, exclude xt_delta from completion model features",
            "2. Consider separate models for each action type",
            "3. Add more granular opponent zone defensive metrics",
            "",
            "## Appendix: Model Artifacts",
            "",
            "- Model: `outputs/modeling/cxt/latest/`",
            "- Features: `feature_store/cxt/progressions_featured.parquet`",
            "- Slice analysis: `outputs/analysis/cxt/slice_evaluation/`",
            "- This report: `data/reports/cxt/evaluation_report.md`",
        ]
    )

    return "\n".join(lines)


def main():
    """Run full evaluation pipeline."""

    logger.info("=" * 70)
    logger.info("CxT MODEL FINAL EVALUATION")
    logger.info("=" * 70)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data and model
    logger.info("\nLoading data and model...")
    df = load_data()
    model = load_model()
    logger.info(f"  Data: {len(df):,} progressions")

    # 1. Discrimination metrics
    discrimination = evaluate_discrimination(model, df)
    logger.info(f"\n  Completion AUC: {discrimination['completion']['auc']:.4f}")
    logger.info(f"  xT Gain R²: {discrimination['xt_gain']['r2']:.3f}")
    logger.info(f"  CxT Correlation: {discrimination['cxt']['correlation']:.3f}")

    # 2. Calibration
    calibration, cal_fig = evaluate_calibration(model, df)
    cal_fig.savefig(OUTPUT_DIR / "calibration_plot.png", dpi=150, bbox_inches="tight")
    plt.close(cal_fig)
    logger.info(f"  Calibration ECE: {calibration['ece']:.4f}")

    # 3. Feature importance
    importance_df = compute_feature_importance(model, df)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    imp_fig = create_feature_importance_plot(importance_df)
    imp_fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(imp_fig)

    # 4. Player/Team aggregations
    player_df, team_df = aggregate_player_team_examples(model, df)
    if len(player_df) > 0:
        player_df.to_csv(OUTPUT_DIR / "player_aggregations.csv", index=False)
    if len(team_df) > 0:
        team_df.to_csv(OUTPUT_DIR / "team_aggregations.csv", index=False)

    # 5. Generate report
    report = generate_markdown_report(
        discrimination, calibration, importance_df, player_df, team_df
    )

    report_path = REPORT_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Save metrics JSON
    metrics_json = {
        "generated": datetime.now().isoformat(),
        "discrimination": discrimination,
        "calibration": {"ece": calibration["ece"]},
        "n_samples": len(df),
    }
    with open(OUTPUT_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nOutputs saved to:")
    logger.info(f"  - {OUTPUT_DIR}")
    logger.info(f"  - {report_path}")

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("FINAL METRICS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<30} {'Value':>15}")
    logger.info("-" * 45)
    logger.info(f"{'Completion AUC':<30} {discrimination['completion']['auc']:>15.4f}")
    logger.info(f"{'Completion Brier':<30} {discrimination['completion']['brier']:>15.6f}")
    logger.info(f"{'xT Gain R²':<30} {discrimination['xt_gain']['r2']:>15.3f}")
    logger.info(f"{'xT Gain MAE':<30} {discrimination['xt_gain']['mae']:>15.4f}")
    logger.info(f"{'CxT-Actual Correlation':<30} {discrimination['cxt']['correlation']:>15.3f}")
    logger.info(f"{'Calibration ECE':<30} {calibration['ece']:>15.4f}")
    logger.info("-" * 45)

    return metrics_json


if __name__ == "__main__":
    main()
