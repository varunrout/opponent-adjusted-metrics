"""Phase 1: Progressions EDA for CxT.

Analyzes action type distributions, xT delta patterns, and zone transitions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def run_phase1_eda(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Run Phase 1: Progressions analysis.
    
    Args:
        df: Progressions DataFrame
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Progressions EDA")
    logger.info("=" * 60)
    
    results = {}
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Action Type Analysis
    # -------------------------------------------------------------------------
    logger.info("\n1. Action Type Analysis")
    
    action_summary = df.groupby("action_type").agg(
        count=("event_id", "count"),
        mean_xt_delta=("xt_delta", "mean"),
        std_xt_delta=("xt_delta", "std"),
        positive_xt_pct=("is_progressive", "mean"),
        mean_start_xt=("start_xt", "mean"),
        mean_end_xt=("end_xt", "mean"),
    ).round(6)
    
    action_summary["pct_of_total"] = (action_summary["count"] / len(df) * 100).round(2)
    action_summary = action_summary.sort_values("count", ascending=False)
    
    results["action_summary"] = action_summary.to_dict()
    action_summary.to_csv(csv_dir / "action_type_summary.csv")
    
    logger.info(f"\nAction Type Summary:")
    for action, row in action_summary.iterrows():
        logger.info(f"  {action}: {row['count']:,} ({row['pct_of_total']:.1f}%)")
        logger.info(f"    - Mean xT delta: {row['mean_xt_delta']:.6f}")
        logger.info(f"    - Progressive %: {row['positive_xt_pct']*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # 2. xT Delta Distribution
    # -------------------------------------------------------------------------
    logger.info("\n2. xT Delta Distribution")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Overall distribution
    ax1 = axes[0]
    ax1.hist(df["xt_delta"], bins=100, edgecolor="black", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", label="Zero")
    ax1.axvline(x=df["xt_delta"].mean(), color="green", linestyle="--", label=f"Mean: {df['xt_delta'].mean():.4f}")
    ax1.set_xlabel("xT Delta")
    ax1.set_ylabel("Frequency")
    ax1.set_title("xT Delta Distribution (All Actions)")
    ax1.legend()
    
    # By action type
    ax2 = axes[1]
    for action_type in df["action_type"].unique():
        subset = df[df["action_type"] == action_type]["xt_delta"]
        ax2.hist(subset, bins=50, alpha=0.5, label=action_type)
    ax2.set_xlabel("xT Delta")
    ax2.set_ylabel("Frequency")
    ax2.set_title("xT Delta by Action Type")
    ax2.legend()
    
    # Box plot by action type
    ax3 = axes[2]
    df.boxplot(column="xt_delta", by="action_type", ax=ax3)
    ax3.set_xlabel("Action Type")
    ax3.set_ylabel("xT Delta")
    ax3.set_title("xT Delta Box Plot")
    plt.suptitle("")  # Remove automatic super title
    
    plt.tight_layout()
    plt.savefig(plots_dir / "xt_delta_distribution.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 3. Zone Transition Matrix
    # -------------------------------------------------------------------------
    logger.info("\n3. Zone Transition Matrix")
    
    transition_matrix = pd.crosstab(
        df["macro_zone_start"], 
        df["macro_zone_end"],
        normalize="index"
    ).round(3)
    
    results["transition_matrix"] = transition_matrix.to_dict()
    transition_matrix.to_csv(csv_dir / "zone_transition_matrix.csv")
    
    # Plot transition heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        transition_matrix, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd",
        ax=ax
    )
    ax.set_xlabel("End Zone")
    ax.set_ylabel("Start Zone")
    ax.set_title("Zone Transition Probability Matrix")
    plt.tight_layout()
    plt.savefig(plots_dir / "transition_matrix.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. Progressive Actions Analysis
    # -------------------------------------------------------------------------
    logger.info("\n4. Progressive Actions Analysis")
    
    progressive_by_zone = df.groupby("macro_zone_start").agg(
        total_actions=("event_id", "count"),
        progressive_actions=("is_progressive", "sum"),
        progressive_pct=("is_progressive", "mean"),
        mean_xt_delta=("xt_delta", "mean"),
        into_final_third=("is_into_final_third", "sum"),
        into_penalty_area=("is_into_penalty_area", "sum"),
    ).round(4)
    
    progressive_by_zone.to_csv(csv_dir / "progressive_by_zone.csv")
    
    logger.info(f"\nProgressive Actions by Start Zone:")
    for zone, row in progressive_by_zone.iterrows():
        logger.info(f"  Zone {zone}: {row['progressive_pct']*100:.1f}% progressive")
    
    # Plot progressive % by zone
    fig, ax = plt.subplots(figsize=(10, 6))
    zones = progressive_by_zone.index
    progressive_pct = progressive_by_zone["progressive_pct"] * 100
    
    bars = ax.bar(zones, progressive_pct, color="steelblue", edgecolor="black")
    ax.set_xlabel("Start Macro-Zone")
    ax.set_ylabel("Progressive Actions (%)")
    ax.set_title("Progressive Action Rate by Start Zone")
    ax.set_xticks(zones)
    
    # Add value labels
    for bar, pct in zip(bars, progressive_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "progressive_by_zone.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 5. Pressure Analysis
    # -------------------------------------------------------------------------
    logger.info("\n5. Pressure Analysis")
    
    pressure_analysis = df.groupby("under_pressure").agg(
        count=("event_id", "count"),
        mean_xt_delta=("xt_delta", "mean"),
        progressive_pct=("is_progressive", "mean"),
    ).round(4)
    
    results["pressure_analysis"] = pressure_analysis.to_dict()
    pressure_analysis.to_csv(csv_dir / "pressure_analysis.csv")
    
    pressure_pct = df["under_pressure"].mean() * 100
    pressure_xt_diff = (
        pressure_analysis.loc[True, "mean_xt_delta"] - 
        pressure_analysis.loc[False, "mean_xt_delta"]
    )
    
    logger.info(f"  Under pressure: {pressure_pct:.1f}% of actions")
    logger.info(f"  xT delta difference (pressure vs no pressure): {pressure_xt_diff:.6f}")
    
    results["pressure_pct"] = pressure_pct
    results["pressure_xt_diff"] = pressure_xt_diff
    
    # -------------------------------------------------------------------------
    # 6. Zone Heatmaps
    # -------------------------------------------------------------------------
    logger.info("\n6. Zone Heatmaps")
    
    # Create 12x8 grid for start locations
    start_zone_counts = df.groupby(["start_zone_x", "start_zone_y"]).size().reset_index(name="count")
    heatmap_data = np.zeros((8, 12))
    for _, row in start_zone_counts.iterrows():
        heatmap_data[int(row["start_zone_y"]), int(row["start_zone_x"])] = row["count"]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Start location heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", origin="lower")
    ax1.set_xlabel("X Zone (→ Attacking Direction)")
    ax1.set_ylabel("Y Zone")
    ax1.set_title("Action Start Locations")
    plt.colorbar(im1, ax=ax1, label="Count")
    
    # xT delta by start zone
    xt_by_zone = df.groupby(["start_zone_x", "start_zone_y"])["xt_delta"].mean().reset_index()
    xt_heatmap = np.zeros((8, 12))
    for _, row in xt_by_zone.iterrows():
        xt_heatmap[int(row["start_zone_y"]), int(row["start_zone_x"])] = row["xt_delta"]
    
    ax2 = axes[1]
    im2 = ax2.imshow(xt_heatmap, cmap="RdYlGn", aspect="auto", origin="lower", 
                     vmin=-0.02, vmax=0.02)
    ax2.set_xlabel("X Zone (→ Attacking Direction)")
    ax2.set_ylabel("Y Zone")
    ax2.set_title("Mean xT Delta by Start Zone")
    plt.colorbar(im2, ax=ax2, label="Mean xT Delta")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "zone_heatmaps.png", dpi=150)
    plt.close()
    
    logger.info(f"\n✓ Phase 1 complete")
    return results
