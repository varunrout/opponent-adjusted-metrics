"""Phase 2: Outcomes EDA for CxT.

Analyzes completion rates, turnover patterns, and outcome correlations with xT.
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


def run_phase2_eda(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Run Phase 2: Outcomes analysis.
    
    Args:
        df: Progressions DataFrame
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Outcomes EDA")
    logger.info("=" * 60)
    
    results = {}
    plots_dir = output_dir / "plots"
    csv_dir = output_dir / "csv"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Pass Completion Analysis
    # -------------------------------------------------------------------------
    logger.info("\n1. Pass Completion Analysis")
    
    passes = df[df["action_type"] == "pass"].copy()
    passes["is_complete"] = ~passes["action_outcome"].isin(["Incomplete", "Out", "Pass Offside"])
    
    overall_completion = passes["is_complete"].mean()
    results["pass_completion_rate"] = overall_completion
    logger.info(f"  Overall pass completion rate: {overall_completion*100:.1f}%")
    
    # Completion by zone
    completion_by_zone = passes.groupby("macro_zone_start").agg(
        total_passes=("event_id", "count"),
        completed=("is_complete", "sum"),
        completion_rate=("is_complete", "mean"),
        mean_xt_delta=("xt_delta", "mean"),
    ).round(4)
    
    completion_by_zone.to_csv(csv_dir / "pass_completion_by_zone.csv")
    
    logger.info(f"\nPass Completion by Zone:")
    for zone, row in completion_by_zone.iterrows():
        logger.info(f"  Zone {zone}: {row['completion_rate']*100:.1f}% ({row['total_passes']:,} passes)")
    
    # Completion by pressure
    completion_by_pressure = passes.groupby("under_pressure")["is_complete"].agg(["mean", "count"])
    completion_by_pressure.columns = ["completion_rate", "count"]
    
    if True in completion_by_pressure.index and False in completion_by_pressure.index:
        pressure_completion_drop = (
            completion_by_pressure.loc[False, "completion_rate"] - 
            completion_by_pressure.loc[True, "completion_rate"]
        )
        results["pressure_completion_drop"] = pressure_completion_drop
        logger.info(f"  Completion drop under pressure: {pressure_completion_drop*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # 2. Action Outcome Distribution
    # -------------------------------------------------------------------------
    logger.info("\n2. Action Outcome Distribution")
    
    outcome_counts = passes["action_outcome"].value_counts()
    results["pass_outcomes"] = outcome_counts.to_dict()
    
    outcome_df = pd.DataFrame({
        "outcome": outcome_counts.index,
        "count": outcome_counts.values,
        "pct": (outcome_counts / len(passes) * 100).values
    })
    outcome_df.to_csv(csv_dir / "pass_outcome_distribution.csv", index=False)
    
    logger.info(f"\nPass Outcomes:")
    for _, row in outcome_df.head(10).iterrows():
        logger.info(f"  - {row['outcome']}: {row['count']:,} ({row['pct']:.1f}%)")
    
    # -------------------------------------------------------------------------
    # 3. xT Delta by Outcome
    # -------------------------------------------------------------------------
    logger.info("\n3. xT Delta by Outcome")
    
    xt_by_outcome = passes.groupby("action_outcome")["xt_delta"].agg(["mean", "std", "count"])
    xt_by_outcome = xt_by_outcome.sort_values("count", ascending=False)
    xt_by_outcome.to_csv(csv_dir / "xt_delta_by_outcome.csv")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    top_outcomes = xt_by_outcome.head(8)
    colors = ["green" if m > 0 else "red" for m in top_outcomes["mean"]]
    
    bars = ax.barh(range(len(top_outcomes)), top_outcomes["mean"], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_outcomes)))
    ax.set_yticklabels(top_outcomes.index)
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("Mean xT Delta")
    ax.set_title("Mean xT Delta by Pass Outcome")
    
    plt.tight_layout()
    plt.savefig(plots_dir / "xt_by_outcome.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------------------
    # 4. Progressive vs Non-Progressive Patterns
    # -------------------------------------------------------------------------
    logger.info("\n4. Progressive vs Non-Progressive Patterns")
    
    prog_comparison = df.groupby("is_progressive").agg(
        count=("event_id", "count"),
        passes=("action_type", lambda x: (x == "pass").sum()),
        carries=("action_type", lambda x: (x == "carry").sum()),
        under_pressure_pct=("under_pressure", "mean"),
        mean_xt_delta=("xt_delta", "mean"),
    ).round(4)
    
    prog_comparison.to_csv(csv_dir / "progressive_comparison.csv")
    
    logger.info(f"\nProgressive vs Non-Progressive:")
    for is_prog, row in prog_comparison.iterrows():
        label = "Progressive" if is_prog else "Non-Progressive"
        logger.info(f"  {label}: {row['count']:,} actions")
        logger.info(f"    - Under pressure: {row['under_pressure_pct']*100:.1f}%")
        logger.info(f"    - Mean xT delta: {row['mean_xt_delta']:.4f}")
    
    # -------------------------------------------------------------------------
    # 5. Zone Entry Analysis
    # -------------------------------------------------------------------------
    logger.info("\n5. Zone Entry Analysis")
    
    final_third_entries = df["is_into_final_third"].sum()
    penalty_area_entries = df["is_into_penalty_area"].sum()
    
    results["final_third_entries"] = int(final_third_entries)
    results["penalty_area_entries"] = int(penalty_area_entries)
    results["final_third_entry_rate"] = final_third_entries / len(df)
    results["penalty_area_entry_rate"] = penalty_area_entries / len(df)
    
    logger.info(f"  Final third entries: {final_third_entries:,} ({final_third_entries/len(df)*100:.2f}%)")
    logger.info(f"  Penalty area entries: {penalty_area_entries:,} ({penalty_area_entries/len(df)*100:.2f}%)")
    
    # Entry by action type
    entries_by_type = df.groupby("action_type").agg(
        final_third_rate=("is_into_final_third", "mean"),
        penalty_area_rate=("is_into_penalty_area", "mean"),
        count=("event_id", "count"),
    ).round(4)
    
    entries_by_type.to_csv(csv_dir / "entries_by_action_type.csv")
    
    logger.info(f"\nZone Entries by Action Type:")
    for action, row in entries_by_type.iterrows():
        logger.info(f"  {action}: {row['final_third_rate']*100:.1f}% final third, {row['penalty_area_rate']*100:.2f}% pen area")
    
    # -------------------------------------------------------------------------
    # 6. Carry Analysis
    # -------------------------------------------------------------------------
    logger.info("\n6. Carry Analysis")
    
    carries = df[df["action_type"] == "carry"].copy()
    
    carry_stats = {
        "total_carries": len(carries),
        "mean_xt_delta": float(carries["xt_delta"].mean()),
        "progressive_rate": float(carries["is_progressive"].mean()),
        "under_pressure_rate": float(carries["under_pressure"].mean()),
    }
    results["carry_stats"] = carry_stats
    
    logger.info(f"  Total carries: {carry_stats['total_carries']:,}")
    logger.info(f"  Mean xT delta: {carry_stats['mean_xt_delta']:.6f}")
    logger.info(f"  Progressive rate: {carry_stats['progressive_rate']*100:.1f}%")
    
    # Compare carries to passes
    pass_xt = passes["xt_delta"].mean()
    carry_xt = carries["xt_delta"].mean()
    
    logger.info(f"\nCarry vs Pass xT delta:")
    logger.info(f"  - Passes: {pass_xt:.6f}")
    logger.info(f"  - Carries: {carry_xt:.6f}")
    logger.info(f"  - Difference: {carry_xt - pass_xt:.6f}")
    
    logger.info(f"\n✓ Phase 2 complete")
    return results
