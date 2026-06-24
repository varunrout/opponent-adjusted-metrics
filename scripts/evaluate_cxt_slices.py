#!/usr/bin/env python3
"""
Post-Model Slice Analysis for CxT Model.

Phase 7: Evaluate trained model on key data slices:
- Under Pressure actions
- Strong/Weak opponent zones
- Final Third entries
- Action types (Pass/Carry/Dribble)
- Game state (Early/Late)
- Zone positions

Acceptance thresholds:
- Completion AUC > 0.55 per slice (relaxed due to high baseline)
- xT Gain correlation > 0 per slice
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score, mean_absolute_error

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_STORE = PROJECT_ROOT / "feature_store" / "cxt"
MODEL_DIR = PROJECT_ROOT / "outputs" / "modeling" / "cxt" / "latest"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "cxt" / "slice_evaluation"


def load_data() -> pd.DataFrame:
    """Load featured progressions data."""
    df = pd.read_parquet(FEATURE_STORE / "progressions_featured.parquet")

    # Ensure success column
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


def evaluate_slice(
    model,
    df: pd.DataFrame,
    slice_name: str,
) -> dict:
    """Evaluate model on a data slice."""

    if len(df) < 100:
        return {
            "slice": slice_name,
            "n_samples": len(df),
            "completion_auc": None,
            "completion_brier": None,
            "xt_gain_r2": None,
            "xt_gain_mae": None,
            "cxt_mean": None,
            "status": "SKIP - Too few samples",
        }

    # Get predictions
    try:
        p_complete = model.predict_completion_prob(df)
        model.predict_xt_gain(df)
        cxt = model.predict_cxt(df)
    except Exception as e:
        return {
            "slice": slice_name,
            "n_samples": len(df),
            "status": f"ERROR: {e}",
        }

    # Completion metrics (need both classes)
    y_true = df["success"].values
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        auc = roc_auc_score(y_true, p_complete)
        brier = brier_score_loss(y_true, p_complete)
    else:
        auc = None
        brier = brier_score_loss(y_true, p_complete) if len(y_true) > 0 else None

    # xT gain metrics (completed only)
    df_complete = df[df["success"] == 1]
    if len(df_complete) > 10:
        pred_complete = model.predict_xt_gain(df_complete)
        r2 = r2_score(df_complete["xt_delta"], pred_complete)
        mae = mean_absolute_error(df_complete["xt_delta"], pred_complete)
    else:
        r2 = None
        mae = None

    # Status based on thresholds
    status = "OK"
    if auc is not None and auc < 0.55:
        status = "WARN - Low AUC"
    elif r2 is not None and r2 < 0:
        status = "WARN - Negative R²"

    return {
        "slice": slice_name,
        "n_samples": len(df),
        "n_complete": len(df_complete),
        "success_rate": float(y_true.mean()),
        "completion_auc": auc,
        "completion_brier": brier,
        "xt_gain_r2": r2,
        "xt_gain_mae": mae,
        "cxt_mean": float(cxt.mean()),
        "actual_xt_mean": float(df_complete["xt_delta"].mean()) if len(df_complete) > 0 else None,
        "status": status,
    }


def define_slices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Define data slices for evaluation."""

    slices = {}

    # Overall
    slices["Overall"] = df

    # Pressure
    if "under_pressure" in df.columns:
        slices["Under Pressure"] = df[df["under_pressure"].fillna(False).astype(bool)]
        slices["No Pressure"] = df[~df["under_pressure"].fillna(False).astype(bool)]

    # Opponent strength (if available)
    if "opponent_is_strong" in df.columns:
        slices["vs Strong Opponent"] = df[df["opponent_is_strong"].fillna(False).astype(bool)]
    if "opponent_is_weak" in df.columns:
        slices["vs Weak Opponent"] = df[df["opponent_is_weak"].fillna(False).astype(bool)]

    # Action type
    if "action_type" in df.columns:
        for atype in df["action_type"].dropna().unique():
            slices[f"Action: {atype}"] = df[df["action_type"] == atype]
    elif "is_pass" in df.columns:
        slices["Action: Pass"] = df[df["is_pass"].fillna(False).astype(bool)]
        slices["Action: Carry"] = df[df["is_carry"].fillna(False).astype(bool)]
        if "is_dribble" in df.columns:
            slices["Action: Dribble"] = df[df["is_dribble"].fillna(False).astype(bool)]

    # Zones
    if "start_third" in df.columns:
        for third in df["start_third"].dropna().unique():
            slices[f"Start: {third}"] = df[df["start_third"] == third]

    if "macro_zone_start" in df.columns:
        for zone in df["macro_zone_start"].dropna().unique():
            slices[f"Zone: {zone}"] = df[df["macro_zone_start"] == zone]

    # Game state
    if "is_late_game" in df.columns:
        slices["Late Game (75+)"] = df[df["is_late_game"].fillna(False).astype(bool)]
        slices["Early Game (<45)"] = (
            df[df["is_early_game"].fillna(False).astype(bool)]
            if "is_early_game" in df.columns
            else df[df["minute_normalized"] < 0.5]
        )

    # Progressive actions
    if "is_progressive" in df.columns:
        slices["Progressive Actions"] = df[df["is_progressive"].fillna(False).astype(bool)]
        slices["Non-Progressive"] = df[~df["is_progressive"].fillna(False).astype(bool)]

    # Final third entries
    if "is_into_final_third" in df.columns:
        slices["Into Final Third"] = df[df["is_into_final_third"].fillna(False).astype(bool)]

    # Penalty area entries
    if "is_into_penalty_area" in df.columns:
        slices["Into Penalty Area"] = df[df["is_into_penalty_area"].fillna(False).astype(bool)]

    return slices


def main():
    """Run slice evaluation."""

    logger.info("=" * 70)
    logger.info("CxT POST-MODEL SLICE ANALYSIS")
    logger.info("=" * 70)

    # Load data and model
    logger.info("\nLoading data...")
    df = load_data()
    logger.info(f"  Loaded {len(df):,} progressions")

    logger.info("\nLoading model...")
    model = load_model()
    logger.info("  Model loaded successfully")

    # Define slices
    slices = define_slices(df)
    logger.info(f"\nDefined {len(slices)} slices for evaluation")

    # Evaluate each slice
    results = []
    logger.info("\n" + "-" * 70)
    logger.info(f"{'Slice':<30} {'N':>10} {'AUC':>8} {'R²':>8} {'Status':>12}")
    logger.info("-" * 70)

    for slice_name, slice_df in slices.items():
        metrics = evaluate_slice(model, slice_df, slice_name)
        results.append(metrics)

        auc_str = (
            f"{metrics.get('completion_auc', 0):.3f}" if metrics.get("completion_auc") else "N/A"
        )
        r2_str = f"{metrics.get('xt_gain_r2', 0):.3f}" if metrics.get("xt_gain_r2") else "N/A"

        logger.info(
            f"{slice_name:<30} {metrics['n_samples']:>10,} {auc_str:>8} {r2_str:>8} {metrics['status']:>12}"
        )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(OUTPUT_DIR / f"slice_evaluation_{timestamp}.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "slice_evaluation_latest.csv", index=False)

    # Generate markdown report
    report = generate_report(results_df, df)
    report_path = OUTPUT_DIR / "slice_evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"\n✓ Results saved to {OUTPUT_DIR}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    warn_count = sum(1 for r in results if "WARN" in r.get("status", ""))
    ok_count = sum(1 for r in results if r.get("status") == "OK")

    logger.info(f"  OK slices: {ok_count}")
    logger.info(f"  Warning slices: {warn_count}")

    if warn_count > 0:
        logger.info("\n  Slices with warnings:")
        for r in results:
            if "WARN" in r.get("status", ""):
                logger.info(f"    - {r['slice']}: {r['status']}")

    # Generate plots
    logger.info("\nGenerating slice plots...")
    generate_slice_plots(results_df, OUTPUT_DIR)
    logger.info(f"  Plots saved to {OUTPUT_DIR}")

    return results_df


def generate_slice_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate visualization plots for slice evaluation."""

    # Filter to slices with valid metrics
    plot_df = results_df[
        results_df["status"].isin(["OK"]) & results_df["completion_auc"].notna()
    ].copy()

    if len(plot_df) == 0:
        logger.warning("No valid slices for plotting")
        return

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. AUC by Slice (horizontal bar chart)
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    # Sort by AUC
    auc_df = plot_df.dropna(subset=["completion_auc"]).sort_values("completion_auc")

    colors = [
        "#d62728" if v < 0.6 else "#2ca02c" if v >= 0.8 else "#ff7f0e"
        for v in auc_df["completion_auc"]
    ]

    bars = ax1.barh(auc_df["slice"], auc_df["completion_auc"], color=colors)
    ax1.axvline(x=0.5, color="red", linestyle="--", alpha=0.7, label="Chance (0.5)")
    ax1.axvline(x=0.8, color="green", linestyle="--", alpha=0.7, label="Good (0.8)")
    ax1.set_xlabel("Completion AUC")
    ax1.set_title("Completion Model AUC by Slice")
    ax1.set_xlim(0.4, 1.0)
    ax1.legend(loc="lower right")

    # Add value labels
    for bar, val in zip(bars, auc_df["completion_auc"]):
        ax1.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()
    fig1.savefig(output_dir / "slice_auc_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 2. R² by Slice (horizontal bar chart)
    fig2, ax2 = plt.subplots(figsize=(10, 8))

    r2_df = plot_df.dropna(subset=["xt_gain_r2"]).sort_values("xt_gain_r2")

    colors_r2 = [
        "#d62728" if v < 0.3 else "#2ca02c" if v >= 0.5 else "#ff7f0e" for v in r2_df["xt_gain_r2"]
    ]

    bars2 = ax2.barh(r2_df["slice"], r2_df["xt_gain_r2"], color=colors_r2)
    ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Baseline (0)")
    ax2.axvline(x=0.5, color="green", linestyle="--", alpha=0.7, label="Good (0.5)")
    ax2.set_xlabel("xT Gain R²")
    ax2.set_title("xT Gain Model R² by Slice")
    ax2.set_xlim(-0.1, 0.8)
    ax2.legend(loc="lower right")

    for bar, val in zip(bars2, r2_df["xt_gain_r2"]):
        ax2.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8
        )

    plt.tight_layout()
    fig2.savefig(output_dir / "slice_r2_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3. Combined metrics scatter plot
    fig3, ax3 = plt.subplots(figsize=(10, 8))

    scatter_df = plot_df.dropna(subset=["completion_auc", "xt_gain_r2"])

    # Size by number of samples (log scale)
    sizes = np.log10(scatter_df["n_samples"] + 1) * 50

    scatter = ax3.scatter(
        scatter_df["completion_auc"],
        scatter_df["xt_gain_r2"],
        s=sizes,
        c=scatter_df["cxt_mean"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add labels for each point
    for _, row in scatter_df.iterrows():
        ax3.annotate(
            row["slice"],
            (row["completion_auc"], row["xt_gain_r2"]),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax3.set_xlabel("Completion AUC")
    ax3.set_ylabel("xT Gain R²")
    ax3.set_title(
        "Slice Performance: Completion vs xT Gain\n(size = log(samples), color = mean CxT)"
    )

    plt.colorbar(scatter, ax=ax3, label="Mean CxT")

    # Add quadrant guidelines
    ax3.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax3.axvline(x=0.75, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig3.savefig(output_dir / "slice_scatter_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 4. Category comparison grouped bar chart
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Group slices by category
    categories = {
        "Pressure": ["Under Pressure", "No Pressure"],
        "Action Type": [s for s in plot_df["slice"] if "Action:" in s],
        "Zone": [s for s in plot_df["slice"] if "Zone:" in s],
        "Third": [s for s in plot_df["slice"] if "Start:" in s],
    }

    for idx, (cat_name, cat_slices) in enumerate(categories.items()):
        ax = axes[idx // 2, idx % 2]
        cat_df = plot_df[plot_df["slice"].isin(cat_slices)]

        if len(cat_df) == 0:
            ax.text(0.5, 0.5, f"No data for {cat_name}", ha="center", va="center")
            ax.set_title(cat_name)
            continue

        x = np.arange(len(cat_df))
        width = 0.35

        # Handle NaN values
        auc_vals = cat_df["completion_auc"].fillna(0).values
        r2_vals = cat_df["xt_gain_r2"].fillna(0).values

        bars1 = ax.bar(x - width / 2, auc_vals, width, label="Completion AUC", color="steelblue")
        bars2 = ax.bar(x + width / 2, r2_vals, width, label="xT Gain R²", color="darkorange")

        ax.set_xlabel("Slice")
        ax.set_ylabel("Metric Value")
        ax.set_title(f"{cat_name} Comparison")
        ax.set_xticks(x)

        # Shorten labels
        labels = [
            s.replace("Action: ", "").replace("Zone: ", "").replace("Start: ", "")
            for s in cat_df["slice"]
        ]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.0)

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    plt.tight_layout()
    fig4.savefig(output_dir / "slice_category_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # 5. Success rate vs CxT mean
    fig5, ax5 = plt.subplots(figsize=(10, 6))

    sr_df = plot_df.dropna(subset=["success_rate", "cxt_mean"])

    ax5.scatter(sr_df["success_rate"], sr_df["cxt_mean"], s=100, alpha=0.7, edgecolors="black")

    for _, row in sr_df.iterrows():
        ax5.annotate(
            row["slice"],
            (row["success_rate"], row["cxt_mean"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax5.set_xlabel("Success Rate")
    ax5.set_ylabel("Mean CxT")
    ax5.set_title("Success Rate vs Mean CxT by Slice")

    # Add trend line
    if len(sr_df) > 2:
        z = np.polyfit(sr_df["success_rate"], sr_df["cxt_mean"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(sr_df["success_rate"].min(), sr_df["success_rate"].max(), 100)
        ax5.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, label="Trend")
        ax5.legend()

    plt.tight_layout()
    fig5.savefig(output_dir / "slice_success_vs_cxt.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)

    # 6. Sample size vs Performance (new)
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 6))

    valid_df = plot_df.dropna(subset=["completion_auc", "xt_gain_r2", "n_samples"])

    # AUC vs Sample Size
    ax6a = axes6[0]
    ax6a.scatter(
        valid_df["n_samples"],
        valid_df["completion_auc"],
        s=80,
        alpha=0.7,
        c="steelblue",
        edgecolors="black",
    )
    ax6a.set_xscale("log")
    ax6a.set_xlabel("Sample Size (log scale)")
    ax6a.set_ylabel("Completion AUC")
    ax6a.set_title("Sample Size vs Completion AUC")
    ax6a.axhline(y=0.8, color="green", linestyle="--", alpha=0.5, label="Good (0.8)")
    ax6a.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Chance (0.5)")
    ax6a.legend(fontsize=8)

    # R² vs Sample Size
    ax6b = axes6[1]
    ax6b.scatter(
        valid_df["n_samples"],
        valid_df["xt_gain_r2"],
        s=80,
        alpha=0.7,
        c="darkorange",
        edgecolors="black",
    )
    ax6b.set_xscale("log")
    ax6b.set_xlabel("Sample Size (log scale)")
    ax6b.set_ylabel("xT Gain R²")
    ax6b.set_title("Sample Size vs xT Gain R²")
    ax6b.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Good (0.5)")
    ax6b.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Baseline (0)")
    ax6b.legend(fontsize=8)

    plt.tight_layout()
    fig6.savefig(output_dir / "slice_sample_size_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig6)

    # 7. Actual xT vs CxT comparison (new)
    fig7, ax7 = plt.subplots(figsize=(10, 8))

    cxt_df = plot_df.dropna(subset=["cxt_mean", "actual_xt_mean"])

    if len(cxt_df) > 0:
        ax7.scatter(
            cxt_df["actual_xt_mean"],
            cxt_df["cxt_mean"],
            s=100,
            alpha=0.7,
            c="steelblue",
            edgecolors="black",
        )

        # Add diagonal line (perfect prediction)
        min_val = min(cxt_df["actual_xt_mean"].min(), cxt_df["cxt_mean"].min())
        max_val = max(cxt_df["actual_xt_mean"].max(), cxt_df["cxt_mean"].max())
        ax7.plot(
            [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="Perfect prediction"
        )

        for _, row in cxt_df.iterrows():
            ax7.annotate(
                row["slice"],
                (row["actual_xt_mean"], row["cxt_mean"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax7.set_xlabel("Mean Actual xT (completed actions)")
        ax7.set_ylabel("Mean CxT (predicted)")
        ax7.set_title("Actual xT vs Predicted CxT by Slice")
        ax7.legend()
    else:
        ax7.text(0.5, 0.5, "No data available", ha="center", va="center")

    plt.tight_layout()
    fig7.savefig(output_dir / "slice_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close(fig7)

    # 8. Radar/Spider chart for key slices (new)
    fig8, ax8 = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Select key slices for radar comparison
    key_slices = [
        "Overall",
        "Under Pressure",
        "No Pressure",
        "Progressive Actions",
        "Into Final Third",
        "Late Game (75+)",
    ]
    radar_df = plot_df[plot_df["slice"].isin(key_slices)].copy()

    if len(radar_df) >= 3:
        # Metrics for radar
        metrics = ["completion_auc", "xt_gain_r2", "success_rate", "cxt_mean"]
        metric_labels = ["Completion AUC", "xT Gain R²", "Success Rate", "Mean CxT"]

        # Normalize metrics to 0-1 range for radar
        for m in metrics:
            if m in radar_df.columns:
                min_v = radar_df[m].min()
                max_v = radar_df[m].max()
                if max_v > min_v:
                    radar_df[f"{m}_norm"] = (radar_df[m] - min_v) / (max_v - min_v)
                else:
                    radar_df[f"{m}_norm"] = 0.5

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.tab10(np.linspace(0, 1, len(radar_df)))

        for idx, (_, row) in enumerate(radar_df.iterrows()):
            values = [row.get(f"{m}_norm", 0.5) for m in metrics]
            values += values[:1]  # Complete the circle

            ax8.plot(angles, values, "o-", linewidth=2, label=row["slice"], color=colors[idx])
            ax8.fill(angles, values, alpha=0.1, color=colors[idx])

        ax8.set_xticks(angles[:-1])
        ax8.set_xticklabels(metric_labels)
        ax8.set_ylim(0, 1)
        ax8.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax8.set_title("Normalized Performance Radar: Key Slices")
    else:
        ax8.text(0, 0, "Not enough slices for radar", ha="center", va="center")

    plt.tight_layout()
    fig8.savefig(output_dir / "slice_radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig8)

    # 9. Zone heatmap (new)
    fig9, ax9 = plt.subplots(figsize=(10, 6))

    zone_df = plot_df[plot_df["slice"].str.contains("Zone:", na=False)].copy()

    if len(zone_df) > 0:
        zone_df["zone_num"] = zone_df["slice"].str.extract(r"Zone: (\d+)").astype(float)
        zone_df = zone_df.sort_values("zone_num")

        # Create grouped bar for zones
        x = np.arange(len(zone_df))
        width = 0.25

        ax9.bar(
            x - width,
            zone_df["completion_auc"].fillna(0),
            width,
            label="Completion AUC",
            color="steelblue",
        )
        ax9.bar(x, zone_df["xt_gain_r2"].fillna(0), width, label="xT Gain R²", color="darkorange")
        ax9.bar(
            x + width, zone_df["success_rate"].fillna(0), width, label="Success Rate", color="green"
        )

        ax9.set_xticks(x)
        ax9.set_xticklabels([f"Zone {int(z)}" for z in zone_df["zone_num"]], rotation=45)
        ax9.set_ylabel("Metric Value")
        ax9.set_title("Performance Metrics by Macro Zone")
        ax9.legend()
        ax9.set_ylim(0, 1.0)
    else:
        ax9.text(0.5, 0.5, "No zone data available", ha="center", va="center")

    plt.tight_layout()
    fig9.savefig(output_dir / "slice_zone_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig9)

    # 10. Performance distribution (violin/box plots) (new)
    fig10, axes10 = plt.subplots(1, 3, figsize=(15, 5))

    # Get metrics for all valid slices
    valid_metrics = plot_df.dropna(subset=["completion_auc", "xt_gain_r2", "success_rate"])

    if len(valid_metrics) > 3:
        # Completion AUC distribution
        axes10[0].boxplot(valid_metrics["completion_auc"], vert=True)
        axes10[0].scatter(
            [1] * len(valid_metrics), valid_metrics["completion_auc"], alpha=0.6, color="steelblue"
        )
        axes10[0].axhline(y=0.8, color="green", linestyle="--", alpha=0.5)
        axes10[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        axes10[0].set_ylabel("Completion AUC")
        axes10[0].set_title("AUC Distribution Across Slices")
        axes10[0].set_xlim(0.5, 1.5)

        # R² distribution
        axes10[1].boxplot(valid_metrics["xt_gain_r2"], vert=True)
        axes10[1].scatter(
            [1] * len(valid_metrics), valid_metrics["xt_gain_r2"], alpha=0.6, color="darkorange"
        )
        axes10[1].axhline(y=0.5, color="green", linestyle="--", alpha=0.5)
        axes10[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
        axes10[1].set_ylabel("xT Gain R²")
        axes10[1].set_title("R² Distribution Across Slices")
        axes10[1].set_xlim(0.5, 1.5)

        # Success rate distribution
        axes10[2].boxplot(valid_metrics["success_rate"], vert=True)
        axes10[2].scatter(
            [1] * len(valid_metrics), valid_metrics["success_rate"], alpha=0.6, color="green"
        )
        axes10[2].set_ylabel("Success Rate")
        axes10[2].set_title("Success Rate Distribution Across Slices")
        axes10[2].set_xlim(0.5, 1.5)

    plt.tight_layout()
    fig10.savefig(output_dir / "slice_metric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig10)

    # 11. Progressive vs Non-Progressive comparison (new)
    fig11, ax11 = plt.subplots(figsize=(10, 6))

    prog_slices = plot_df[plot_df["slice"].isin(["Progressive Actions", "Non-Progressive"])]

    if len(prog_slices) == 2:
        metrics_to_compare = ["completion_auc", "xt_gain_r2", "success_rate", "cxt_mean"]
        metric_names = ["Completion AUC", "xT Gain R²", "Success Rate", "Mean CxT"]

        prog_row = prog_slices[prog_slices["slice"] == "Progressive Actions"].iloc[0]
        non_prog_row = prog_slices[prog_slices["slice"] == "Non-Progressive"].iloc[0]

        x = np.arange(len(metrics_to_compare))
        width = 0.35

        prog_vals = [prog_row.get(m, 0) or 0 for m in metrics_to_compare]
        non_prog_vals = [non_prog_row.get(m, 0) or 0 for m in metrics_to_compare]

        bars1 = ax11.bar(
            x - width / 2, prog_vals, width, label="Progressive", color="green", alpha=0.8
        )
        bars2 = ax11.bar(
            x + width / 2, non_prog_vals, width, label="Non-Progressive", color="gray", alpha=0.8
        )

        ax11.set_xticks(x)
        ax11.set_xticklabels(metric_names, rotation=45, ha="right")
        ax11.set_ylabel("Value")
        ax11.set_title("Progressive vs Non-Progressive Actions: Metric Comparison")
        ax11.legend()

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax11.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax11.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    else:
        ax11.text(
            0.5,
            0.5,
            "Progressive/Non-Progressive data not available",
            ha="center",
            va="center",
            transform=ax11.transAxes,
        )

    plt.tight_layout()
    fig11.savefig(output_dir / "slice_progressive_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig11)


def generate_report(results_df: pd.DataFrame, df: pd.DataFrame) -> str:
    """Generate markdown report."""

    lines = [
        "# CxT Post-Model Slice Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total samples**: {len(df):,}",
        f"- **Slices evaluated**: {len(results_df)}",
        f"- **OK slices**: {sum(results_df['status'] == 'OK')}",
        f"- **Warning slices**: {sum(results_df['status'].str.contains('WARN', na=False))}",
        "",
        "## Slice Metrics",
        "",
        "| Slice | N | Success Rate | Completion AUC | xT Gain R² | CxT Mean | Status |",
        "|-------|---|--------------|----------------|------------|----------|--------|",
    ]

    for _, row in results_df.iterrows():
        auc = f"{row['completion_auc']:.3f}" if pd.notna(row.get("completion_auc")) else "N/A"
        r2 = f"{row['xt_gain_r2']:.3f}" if pd.notna(row.get("xt_gain_r2")) else "N/A"
        sr = f"{row['success_rate']:.1%}" if pd.notna(row.get("success_rate")) else "N/A"
        cxt = f"{row['cxt_mean']:.4f}" if pd.notna(row.get("cxt_mean")) else "N/A"

        lines.append(
            f"| {row['slice']} | {row['n_samples']:,} | {sr} | {auc} | {r2} | {cxt} | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            "### Completion Model Performance",
            "",
        ]
    )

    # Find interesting comparisons
    pressure_rows = results_df[results_df["slice"].str.contains("Pressure", na=False)]
    if len(pressure_rows) >= 2:
        under = (
            pressure_rows[pressure_rows["slice"] == "Under Pressure"].iloc[0]
            if len(pressure_rows[pressure_rows["slice"] == "Under Pressure"]) > 0
            else None
        )
        no = (
            pressure_rows[pressure_rows["slice"] == "No Pressure"].iloc[0]
            if len(pressure_rows[pressure_rows["slice"] == "No Pressure"]) > 0
            else None
        )
        if under is not None and no is not None:
            lines.append(
                f"- **Pressure impact**: Under pressure AUC={under.get('completion_auc', 'N/A'):.3f} vs No pressure AUC={no.get('completion_auc', 'N/A'):.3f}"
            )

    # Action type comparison
    action_rows = results_df[results_df["slice"].str.contains("Action:", na=False)]
    if len(action_rows) > 0:
        lines.append("- **Action types**:")
        for _, row in action_rows.iterrows():
            r2 = f"{row['xt_gain_r2']:.3f}" if pd.notna(row.get("xt_gain_r2")) else "N/A"
            lines.append(f"  - {row['slice']}: R²={r2}, CxT mean={row.get('cxt_mean', 0):.4f}")

    lines.extend(
        [
            "",
            "### xT Gain Model Performance",
            "",
        ]
    )

    # Zone performance
    zone_rows = results_df[results_df["slice"].str.contains("Zone:", na=False)]
    if len(zone_rows) > 0:
        lines.append("- **By macro zone**:")
        for _, row in zone_rows.iterrows():
            r2 = f"{row['xt_gain_r2']:.3f}" if pd.notna(row.get("xt_gain_r2")) else "N/A"
            lines.append(f"  - {row['slice']}: R²={r2}")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            "| Criterion | Threshold | Status |",
            "|-----------|-----------|--------|",
        ]
    )

    overall = (
        results_df[results_df["slice"] == "Overall"].iloc[0]
        if len(results_df[results_df["slice"] == "Overall"]) > 0
        else None
    )
    if overall is not None:
        auc_pass = (
            overall["completion_auc"] >= 0.55 if pd.notna(overall.get("completion_auc")) else False
        )
        r2_pass = overall["xt_gain_r2"] >= 0 if pd.notna(overall.get("xt_gain_r2")) else False
        lines.append(
            f"| Overall Completion AUC ≥ 0.55 | {overall.get('completion_auc', 'N/A'):.3f} | {'✓' if auc_pass else '✗'} |"
        )
        lines.append(
            f"| Overall xT Gain R² ≥ 0 | {overall.get('xt_gain_r2', 'N/A'):.3f} | {'✓' if r2_pass else '✗'} |"
        )

    # Check all slices
    all_auc_ok = all(
        row["completion_auc"] >= 0.50 or pd.isna(row.get("completion_auc"))
        for _, row in results_df.iterrows()
    )
    lines.append(f"| All slices AUC ≥ 0.50 | - | {'✓' if all_auc_ok else '✗'} |")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
        ]
    )

    warn_count = sum(results_df["status"].str.contains("WARN", na=False))
    if warn_count == 0:
        lines.append(
            "✓ **All slices pass acceptance criteria.** Model is ready for final integration."
        )
    else:
        lines.append(f"⚠ **{warn_count} slices have warnings.** Review before proceeding.")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
