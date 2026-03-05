#!/usr/bin/env python
"""cXA Phase 1: Assist Event Hygiene — definitions + base rates.

Purpose
-------
Establish baseline understanding of what becomes an assist:
  - Assist rate by pass type (cross/through/ground), zone, pressure, distance
  - Assists vs "key pass" vs "pass into box" overlap
  - Sequence length stats for goals (1-action vs multi-action)

Outputs (outputs/analysis/cxa/phase1_assist_hygiene/)
-------
data/
  - assist_rate_by_pass_type.csv
  - assist_rate_by_zone.csv
  - assist_rate_by_pressure.csv
  - sequence_length_distribution.csv
  - pass_completion_vs_assist_rate.csv
plots/
  - assist_rate_by_pass_type.png
  - assist_rate_heatmap.png
  - sequence_length_histogram.png
  - pass_type_comparison.png
phase1_assist_hygiene_report.md

Usage
-----
    PYTHONPATH=src python -m opponent_adjusted.analysis.cxa.phase1_assist_hygiene
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def _assign_zone(x: float) -> str:
    """Assign pass origin to third (defensive/middle/attacking)."""
    if x < 40:
        return "Defensive"
    elif x < 80:
        return "Middle"
    else:
        return "Attacking"


def analyze_assist_rate_by_pass_type(passes: pd.DataFrame) -> pd.DataFrame:
    """Compute assist rate by pass type."""
    logger.info("Analyzing assist rate by pass type...")
    
    # Use is_key_pass + sequence_resulted_goal to identify actual assists
    passes = passes.copy()
    passes["is_assist"] = passes["is_key_pass"] & passes["sequence_resulted_goal"]
    
    pass_types = {
        "is_cross": "Cross",
        "is_through_ball": "Through Ball",
        "is_into_box": "Into Box",
        "is_progressive": "Progressive",
    }
    
    results = []
    for col, label in pass_types.items():
        if col not in passes.columns:
            continue
        
        subset = passes[passes[col] == True]
        total = len(subset)
        assists = subset["is_assist"].sum()
        assist_rate = assists / total if total > 0 else 0
        
        results.append({
            "pass_type": label,
            "total_passes": total,
            "assists": assists,
            "assist_rate": assist_rate,
        })
    
    # Add "Other" category
    other_mask = ~(
        passes.get("is_cross", False) |
        passes.get("is_through_ball", False) |
        passes.get("is_into_box", False)
    )
    other = passes[other_mask]
    results.append({
        "pass_type": "Other",
        "total_passes": len(other),
        "assists": other["is_assist"].sum(),
        "assist_rate": other["is_assist"].sum() / len(other) if len(other) > 0 else 0,
    })
    
    return pd.DataFrame(results)


def analyze_assist_rate_by_zone(passes: pd.DataFrame) -> pd.DataFrame:
    """Compute assist rate by pitch zone."""
    logger.info("Analyzing assist rate by zone...")
    
    passes = passes.copy()
    passes["is_assist"] = passes["is_key_pass"] & passes["sequence_resulted_goal"]
    passes["zone"] = passes["start_x"].apply(_assign_zone)
    
    zone_stats = passes.groupby("zone").agg({
        "pass_id": "count",
        "is_assist": ["sum", "mean"]
    }).reset_index()
    
    zone_stats.columns = ["zone", "total_passes", "assists", "assist_rate"]
    zone_stats = zone_stats.sort_values("assist_rate", ascending=False)
    
    return zone_stats


def analyze_assist_rate_by_pressure(passes: pd.DataFrame) -> pd.DataFrame:
    """Compute assist rate by pressure state."""
    logger.info("Analyzing assist rate by pressure...")
    
    if "under_pressure" not in passes.columns:
        logger.warning("No 'under_pressure' column, skipping pressure analysis")
        return pd.DataFrame()
    
    passes = passes.copy()
    passes["is_assist"] = passes["is_key_pass"] & passes["sequence_resulted_goal"]
    
    pressure_stats = passes.groupby("under_pressure").agg({
        "pass_id": "count",
        "is_assist": ["sum", "mean"]
    }).reset_index()
    
    pressure_stats.columns = ["under_pressure", "total_passes", "assists", "assist_rate"]
    pressure_stats["pressure_state"] = pressure_stats["under_pressure"].map({
        True: "Under Pressure",
        False: "Open Play"
    })
    
    return pressure_stats[["pressure_state", "total_passes", "assists", "assist_rate"]]


def analyze_sequence_lengths(sequences: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Analyze sequence length distribution for goals."""
    logger.info("Analyzing sequence lengths...")
    
    seq_goals = sequences[sequences["is_goal"] == True].copy()
    action_goals = actions[actions["is_goal"] == True].copy()
    
    results = []
    
    # Pass sequences
    if "num_passes_in_sequence" in seq_goals.columns:
        pass_lengths = seq_goals["num_passes_in_sequence"].value_counts().sort_index()
        for length, count in pass_lengths.items():
            results.append({
                "sequence_type": "Pass Sequences",
                "length": int(length),
                "count": count,
                "percentage": count / len(seq_goals) * 100,
            })
    
    # Action sequences
    if "num_actions" in action_goals.columns:
        action_lengths = action_goals["num_actions"].value_counts().sort_index()
        for length, count in action_lengths.items():
            results.append({
                "sequence_type": "Action Sequences",
                "length": int(length),
                "count": count,
                "percentage": count / len(action_goals) * 100,
            })
    
    return pd.DataFrame(results)


def plot_assist_rate_by_pass_type(data: pd.DataFrame, out_path: Path) -> None:
    """Plot assist rate by pass type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Total passes
    ax1.barh(data["pass_type"], data["total_passes"], color="#1f77b4", alpha=0.7)
    ax1.set_xlabel("Total Passes")
    ax1.set_title("Pass Volume by Type")
    ax1.invert_yaxis()
    
    # Plot 2: Assist rate
    colors = plt.cm.RdYlGn(data["assist_rate"] / data["assist_rate"].max())
    ax2.barh(data["pass_type"], data["assist_rate"] * 100, color=colors)
    ax2.set_xlabel("Assist Rate (%)")
    ax2.set_title("Assist Conversion Rate by Type")
    ax2.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(data["assist_rate"] * 100):
        ax2.text(v + 0.02, i, f"{v:.2f}%", va="center", fontsize=9)
    
    fig.suptitle("Assist Rate by Pass Type", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_assist_rate_heatmap(passes: pd.DataFrame, out_path: Path) -> None:
    """Plot assist rate heatmap by end location."""
    logger.info("Creating assist rate heatmap...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Bin passes by end location
    x_bins = np.linspace(0, 120, 13)
    y_bins = np.linspace(0, 80, 9)
    
    passes = passes.copy()
    passes["is_assist"] = passes["is_key_pass"] & passes["sequence_resulted_goal"]
    passes["x_bin"] = pd.cut(passes["end_x"], bins=x_bins, labels=False)
    passes["y_bin"] = pd.cut(passes["end_y"], bins=y_bins, labels=False)
    
    # Calculate assist rate per bin
    heatmap_data = passes.groupby(["x_bin", "y_bin"]).agg({
        "is_assist": ["sum", "count", "mean"]
    }).reset_index()
    
    heatmap_data.columns = ["x_bin", "y_bin", "assists", "total", "assist_rate"]
    
    # Filter bins with at least 10 passes
    heatmap_data = heatmap_data[heatmap_data["total"] >= 10]
    
    # Create pivot table for heatmap
    pivot = heatmap_data.pivot_table(
        values="assist_rate",
        index="y_bin",
        columns="x_bin",
        fill_value=0
    )
    
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        cbar_kws={"label": "Assist Rate"},
        ax=ax,
        vmin=0,
        vmax=pivot.max().max(),
    )
    
    ax.set_title("Assist Rate Heatmap by Pass Destination", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position (→ attacking direction)")
    ax.set_ylabel("Y Position")
    ax.invert_yaxis()
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sequence_length_histogram(data: pd.DataFrame, out_path: Path) -> None:
    """Plot sequence length distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for seq_type in data["sequence_type"].unique():
        subset = data[data["sequence_type"] == seq_type]
        ax.bar(
            subset["length"] + (0.2 if seq_type == "Action Sequences" else -0.2),
            subset["count"],
            width=0.4,
            label=seq_type,
            alpha=0.8
        )
    
    ax.set_xlabel("Sequence Length (# actions)")
    ax.set_ylabel("Goal Count")
    ax.set_title("Goal Sequence Length Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown_report(
    out_path: Path,
    pass_type_stats: pd.DataFrame,
    zone_stats: pd.DataFrame,
    pressure_stats: pd.DataFrame,
    sequence_stats: pd.DataFrame,
) -> None:
    """Write Phase 1 markdown report."""
    lines: List[str] = []
    
    lines.append("# cXA Phase 1 — Assist Event Hygiene")
    lines.append("")
    lines.append("## What Becomes an Assist?")
    lines.append("")
    
    # Pass type table
    lines.append("### Assist Rate by Pass Type")
    lines.append("")
    lines.append("| Pass Type | Total Passes | Assists | Assist Rate |")
    lines.append("|-----------|--------------|---------|-------------|")
    for _, row in pass_type_stats.iterrows():
        lines.append(
            f"| {row['pass_type']} | {int(row['total_passes']):,} | "
            f"{int(row['assists'])} | {row['assist_rate']*100:.2f}% |"
        )
    lines.append("")
    
    # Zone table
    lines.append("### Assist Rate by Zone")
    lines.append("")
    lines.append("| Zone | Total Passes | Assists | Assist Rate |")
    lines.append("|------|--------------|---------|-------------|")
    for _, row in zone_stats.iterrows():
        lines.append(
            f"| {row['zone']} | {int(row['total_passes']):,} | "
            f"{int(row['assists'])} | {row['assist_rate']*100:.2f}% |"
        )
    lines.append("")
    
    # Pressure table (if available)
    if not pressure_stats.empty:
        lines.append("### Assist Rate by Pressure State")
        lines.append("")
        lines.append("| Pressure State | Total Passes | Assists | Assist Rate |")
        lines.append("|----------------|--------------|---------|-------------|")
        for _, row in pressure_stats.iterrows():
            lines.append(
                f"| {row['pressure_state']} | {int(row['total_passes']):,} | "
                f"{int(row['assists'])} | {row['assist_rate']*100:.2f}% |"
            )
        lines.append("")
    
    # Sequence lengths
    lines.append("### Goal Sequence Lengths")
    lines.append("")
    for seq_type in sequence_stats["sequence_type"].unique():
        subset = sequence_stats[sequence_stats["sequence_type"] == seq_type]
        lines.append(f"**{seq_type}:**")
        lines.append("")
        for _, row in subset.iterrows():
            lines.append(f"- {int(row['length'])} actions: {int(row['count'])} goals ({row['percentage']:.1f}%)")
        lines.append("")
    
    lines.append("## Key Insights")
    lines.append("")
    
    # Find highest assist rate pass type
    best_type = pass_type_stats.loc[pass_type_stats["assist_rate"].idxmax()]
    lines.append(f"- **Highest assist rate:** {best_type['pass_type']} ({best_type['assist_rate']*100:.2f}%)")
    
    # Find most common sequence length
    if not sequence_stats.empty:
        most_common = sequence_stats.loc[sequence_stats["count"].idxmax()]
        lines.append(f"- **Most common goal sequence:** {int(most_common['length'])} actions ({int(most_common['count'])} goals)")
    
    # Zone insight
    attacking_zone = zone_stats[zone_stats["zone"] == "Attacking"]
    if not attacking_zone.empty:
        rate = attacking_zone.iloc[0]["assist_rate"] * 100
        lines.append(f"- **Attacking third assist rate:** {rate:.2f}%")
    
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_phase1_assist_hygiene(
    feature_store_path: Path | None = None,
    output_path: Path | None = None,
) -> Dict:
    """Run Phase 1 assist hygiene analysis."""
    repo_root = _get_repo_root()
    
    if feature_store_path is None:
        feature_store_path = repo_root / "feature_store" / "cxa"
    if output_path is None:
        output_path = repo_root / "outputs" / "analysis" / "cxa" / "phase1_assist_hygiene"
    
    out_data = output_path / "data"
    out_plots = output_path / "plots"
    out_data.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading datasets...")
    passes = _load_parquet(feature_store_path / "pass_sequences.parquet")
    sequences = _load_parquet(feature_store_path / "sequences.parquet")
    actions = _load_parquet(feature_store_path / "action_sequences.parquet")
    
    # Analyses
    pass_type_stats = analyze_assist_rate_by_pass_type(passes)
    zone_stats = analyze_assist_rate_by_zone(passes)
    pressure_stats = analyze_assist_rate_by_pressure(passes)
    sequence_stats = analyze_sequence_lengths(sequences, actions)
    
    # Save data
    pass_type_stats.to_csv(out_data / "assist_rate_by_pass_type.csv", index=False)
    zone_stats.to_csv(out_data / "assist_rate_by_zone.csv", index=False)
    if not pressure_stats.empty:
        pressure_stats.to_csv(out_data / "assist_rate_by_pressure.csv", index=False)
    sequence_stats.to_csv(out_data / "sequence_length_distribution.csv", index=False)
    
    # Generate plots
    logger.info("Generating plots...")
    plot_assist_rate_by_pass_type(pass_type_stats, out_plots / "assist_rate_by_pass_type.png")
    plot_assist_rate_heatmap(passes, out_plots / "assist_rate_heatmap.png")
    plot_sequence_length_histogram(sequence_stats, out_plots / "sequence_length_histogram.png")
    
    # Write report
    write_markdown_report(
        output_path / "phase1_assist_hygiene_report.md",
        pass_type_stats,
        zone_stats,
        pressure_stats,
        sequence_stats,
    )
    
    logger.info(f"Phase 1 complete. Outputs: {output_path}")
    # Calculate assist stats
    passes["is_assist"] = passes["is_key_pass"] & passes["sequence_resulted_goal"]
    
    return {
        "output_path": str(output_path),
        "total_passes": len(passes),
        "total_assists": int(passes["is_assist"].sum()),
        "overall_assist_rate": passes["is_assist"].mean(),
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    result = run_phase1_assist_hygiene()
    
    print("=" * 72)
    print("cXA Phase 1 — Assist Event Hygiene Summary")
    print("=" * 72)
    print(f"Total passes analyzed: {result['total_passes']:,}")
    print(f"Total assists: {result['total_assists']}")
    print(f"Overall assist rate: {result['overall_assist_rate']*100:.3f}%")
    print()
    print(f"Outputs: {result['output_path']}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
