#!/usr/bin/env python
"""
CxT Post-Model Slice Analysis

Validates model performance across key slices after training.
Computes AUC, R², correlation metrics by slice.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score

from opponent_adjusted.config import settings
from opponent_adjusted.modeling.cxt.contextual_model import CxTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_slice(
    df: pd.DataFrame,
    model: CxTModel,
    slice_name: str,
) -> dict:
    """Evaluate model on a slice."""
    if len(df) < 50:
        return {
            "slice": slice_name,
            "n_rows": len(df),
            "skipped": True,
        }
    
    # Ensure success column
    if "success" not in df.columns:
        if "action_success" in df.columns:
            df = df.copy()
            df["success"] = df["action_success"].astype(int)
        else:
            return {"slice": slice_name, "skipped": True, "reason": "no success column"}
    
    # Get predictions
    p_complete = model.predict_completion_prob(df)
    pred_xt = model.predict_xt_gain(df)
    cxt = model.predict_cxt(df)
    
    # Metrics
    result = {
        "slice": slice_name,
        "n_rows": len(df),
        "success_rate": df["success"].mean(),
        "cxt_mean": float(cxt.mean()),
        "cxt_std": float(cxt.std()),
    }
    
    # Completion AUC (only if there's variance)
    if df["success"].nunique() > 1:
        try:
            result["completion_auc"] = roc_auc_score(df["success"], p_complete)
        except Exception:
            result["completion_auc"] = None
    else:
        result["completion_auc"] = None
    
    # xT gain metrics (completed only)
    df_complete = df[df["success"] == 1]
    if len(df_complete) >= 20:
        pred_complete = model.predict_xt_gain(df_complete)
        result["xt_gain_r2"] = r2_score(df_complete["xt_delta"], pred_complete)
        result["xt_gain_mae"] = mean_absolute_error(df_complete["xt_delta"], pred_complete)
    else:
        result["xt_gain_r2"] = None
        result["xt_gain_mae"] = None
    
    # CxT-actual correlation
    actual_xt = np.where(df["success"] == 1, df["xt_delta"], 0)
    if actual_xt.std() > 0 and cxt.std() > 0:
        result["cxt_corr"] = np.corrcoef(cxt, actual_xt)[0, 1]
    else:
        result["cxt_corr"] = None
    
    return result


def run_post_model_analysis(
    df: pd.DataFrame,
    model: CxTModel,
    output_dir: Path,
) -> pd.DataFrame:
    """Run comprehensive post-model slice analysis."""
    
    logger.info("=" * 60)
    logger.info("POST-MODEL SLICE ANALYSIS")
    logger.info("=" * 60)
    
    results = []
    
    # Overall
    results.append(evaluate_slice(df, model, "OVERALL"))
    
    # By action type
    logger.info("\nAnalyzing by action type...")
    for action in df["action_type"].unique():
        slice_df = df[df["action_type"] == action]
        results.append(evaluate_slice(slice_df, model, f"action_{action}"))
    
    # By opponent strength tertile
    logger.info("\nAnalyzing by opponent strength...")
    df_temp = df.copy()
    df_temp["opp_tertile"] = pd.qcut(
        df_temp["opponent_global_rating"], 
        q=3, 
        labels=["weak", "medium", "strong"],
        duplicates="drop"
    )
    for tertile in df_temp["opp_tertile"].unique():
        if pd.notna(tertile):
            slice_df = df_temp[df_temp["opp_tertile"] == tertile]
            results.append(evaluate_slice(slice_df, model, f"opp_{tertile}"))
    
    # By zone (third)
    logger.info("\nAnalyzing by zone...")
    for third in df["start_third"].unique():
        if pd.notna(third):
            slice_df = df[df["start_third"] == third]
            results.append(evaluate_slice(slice_df, model, f"zone_{third}"))
    
    # By pressure
    logger.info("\nAnalyzing by pressure...")
    for pressure in [True, False]:
        slice_df = df[df["under_pressure"] == pressure]
        label = "under_pressure" if pressure else "no_pressure"
        results.append(evaluate_slice(slice_df, model, label))
    
    # By game state (minute buckets)
    logger.info("\nAnalyzing by game state...")
    df_temp = df.copy()
    df_temp["minute_bucket"] = pd.cut(
        df_temp["minute"], 
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
    )
    for bucket in df_temp["minute_bucket"].unique():
        if pd.notna(bucket):
            slice_df = df_temp[df_temp["minute_bucket"] == bucket]
            results.append(evaluate_slice(slice_df, model, f"minute_{bucket}"))
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "post_model_slice_metrics.csv", index=False)
    
    # Generate visualizations
    _plot_slice_metrics(results_df, output_dir)
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("SLICE PERFORMANCE SUMMARY")
    logger.info("=" * 60)
    
    valid_results = results_df[~results_df.get("skipped", False)]
    
    logger.info(f"\nOverall CxT-Actual Correlation: {valid_results.iloc[0]['cxt_corr']:.3f}")
    
    # Worst performing slices
    if "xt_gain_r2" in valid_results.columns:
        r2_results = valid_results[valid_results["xt_gain_r2"].notna()]
        if len(r2_results) > 0:
            worst = r2_results.nsmallest(3, "xt_gain_r2")
            logger.info("\nLowest R² slices:")
            for _, row in worst.iterrows():
                logger.info(f"  {row['slice']}: R²={row['xt_gain_r2']:.3f}, n={row['n_rows']:,}")
    
    return results_df


def _plot_slice_metrics(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate slice metric visualizations."""
    
    valid_df = results_df[~results_df.get("skipped", False)].copy()
    
    # Plot 1: CxT correlation by slice
    fig, ax = plt.subplots(figsize=(12, 8))
    
    valid_df = valid_df.sort_values("cxt_corr", ascending=True)
    y_pos = np.arange(len(valid_df))
    colors = plt.cm.RdYlGn(
        (valid_df["cxt_corr"].fillna(0) - valid_df["cxt_corr"].min()) / 
        (valid_df["cxt_corr"].max() - valid_df["cxt_corr"].min() + 0.001)
    )
    
    ax.barh(y_pos, valid_df["cxt_corr"].fillna(0), color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(valid_df["slice"])
    ax.set_xlabel("CxT-Actual Correlation")
    ax.set_title("CxT Model Correlation by Slice")
    ax.axvline(x=0.6, color="red", linestyle="--", alpha=0.5, label="Target: 0.60")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "slice_cxt_correlation.png", dpi=150)
    plt.close()
    
    # Plot 2: R² by slice
    r2_df = valid_df[valid_df["xt_gain_r2"].notna()].copy()
    if len(r2_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        r2_df = r2_df.sort_values("xt_gain_r2", ascending=True)
        y_pos = np.arange(len(r2_df))
        colors = plt.cm.RdYlGn(
            (r2_df["xt_gain_r2"] - r2_df["xt_gain_r2"].min()) / 
            (r2_df["xt_gain_r2"].max() - r2_df["xt_gain_r2"].min() + 0.001)
        )
        
        ax.barh(y_pos, r2_df["xt_gain_r2"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(r2_df["slice"])
        ax.set_xlabel("xT Gain R²")
        ax.set_title("xT Gain Model R² by Slice")
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Target: 0.50")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "slice_xt_gain_r2.png", dpi=150)
        plt.close()
    
    logger.info("Visualizations saved")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CxT post-model slice analysis")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/modeling/cxt/latest/predictions.parquet"),
        help="Predictions parquet from training",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/modeling/cxt/latest"),
        help="Model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cxt/post_model_slices"),
        help="Output directory",
    )
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("CxT POST-MODEL SLICE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Predictions: {args.predictions}")
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)
    
    # Load predictions (has CxT values already)
    if args.predictions.exists():
        logger.info("\nLoading predictions...")
        df = pd.read_parquet(args.predictions)
        logger.info(f"Loaded {len(df):,} rows")
    else:
        logger.error(f"Predictions file not found: {args.predictions}")
        return 1
    
    # Load model
    logger.info("\nLoading model...")
    model = CxTModel.load(args.model_dir)
    logger.info("Model loaded")
    
    # Run analysis
    results = run_post_model_analysis(df, model, args.output_dir)
    
    # Generate report
    report_path = args.output_dir / "post_model_report.md"
    _generate_report(results, report_path)
    
    elapsed = datetime.now() - start_time
    logger.info(f"\nAnalysis complete in {elapsed}")
    
    return 0


def _generate_report(results_df: pd.DataFrame, output_path: Path) -> None:
    """Generate markdown report."""
    
    lines = [
        "# CxT Post-Model Slice Analysis Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This report validates CxT model performance across different data slices.",
        "",
        "## Slice Metrics",
        "",
        "| Slice | N | Success Rate | CxT Corr | R² | MAE |",
        "|-------|---|--------------|----------|-----|-----|",
    ]
    
    for _, row in results_df.iterrows():
        if row.get("skipped"):
            continue
        
        corr = f"{row.get('cxt_corr', 0):.3f}" if pd.notna(row.get("cxt_corr")) else "N/A"
        r2 = f"{row.get('xt_gain_r2', 0):.3f}" if pd.notna(row.get("xt_gain_r2")) else "N/A"
        mae = f"{row.get('xt_gain_mae', 0):.4f}" if pd.notna(row.get("xt_gain_mae")) else "N/A"
        
        lines.append(
            f"| {row['slice']} | {row['n_rows']:,} | {row['success_rate']:.1%} | "
            f"{corr} | {r2} | {mae} |"
        )
    
    lines.extend([
        "",
        "## Visualizations",
        "",
        "- [CxT Correlation by Slice](slice_cxt_correlation.png)",
        "- [xT Gain R² by Slice](slice_xt_gain_r2.png)",
        "",
        "## Interpretation",
        "",
        "**Target Metrics:**",
        "- CxT-Actual Correlation > 0.60: Good predictive signal",
        "- xT Gain R² > 0.50: Model explains majority of variance",
        "",
    ])
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Report saved: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
