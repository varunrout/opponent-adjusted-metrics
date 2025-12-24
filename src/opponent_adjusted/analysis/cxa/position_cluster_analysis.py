"""Position-based cluster analysis for cxA.

Infers position groups from pass origin location on the pitch:
- Defenders: passes from defensive third (x < 40)
- Midfielders: passes from middle third (40 <= x < 80)
- Forwards: passes from attacking third (x >= 80)

Then analyzes by team style clusters to show how each team style 
distributes xA creation across position zones.

Outputs: CSV tables and plots under outputs/analysis/cxa/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.analysis.plot_utilities import (
    configure_matplotlib,
    draw_pitch,
    get_analysis_output_dir,
    save_figure,
    STYLE,
)
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Match, Player, Team


# ---------------------------------------------------------------------------
# Position Grouping
# ---------------------------------------------------------------------------

# Pitch zones (StatsBomb coordinates: 120 x 80)
# Based on pass origin x-coordinate
POSITION_GROUPS = {
    "Defenders": (0, 40),      # Defensive third
    "Midfielders": (40, 80),   # Middle third
    "Forwards": (80, 120),     # Attacking third
}


def infer_position_group(pass_start_x: Optional[float]) -> str:
    """Infer position group from pass origin x-coordinate.
    
    Assumes passes originate from the player's general position zone.
    """
    if pd.isna(pass_start_x):
        return "Unknown"
    
    x = float(pass_start_x)
    for group, (x_min, x_max) in POSITION_GROUPS.items():
        if x_min <= x < x_max:
            return group
    if x >= 80:
        return "Forwards"
    return "Unknown"


# Keep old position mapping for reference (not used)
OLD_POSITION_GROUPS = {
    "Defenders": [
        "Goalkeeper",
        "Right Back",
        "Left Back",
        "Center Back",
        "Right Center Back",
        "Left Center Back",
        "Right Wing Back",
        "Left Wing Back",
    ],
    "Midfielders": [
        "Right Midfield",
        "Left Midfield",
        "Right Center Midfield",
        "Left Center Midfield",
        "Center Midfield",
        "Right Defensive Midfield",
        "Left Defensive Midfield",
        "Center Defensive Midfield",
        "Right Attacking Midfield",
        "Left Attacking Midfield",
        "Center Attacking Midfield",
        "Right Wing",
        "Left Wing",
    ],
    "Forwards": [
        "Center Forward",
        "Right Center Forward",
        "Left Center Forward",
        "Striker",
    ],
}


# ---------------------------------------------------------------------------
# Position Extraction from Events JSON (kept for reference, not used)
# ---------------------------------------------------------------------------


def extract_player_positions(events_dir: Path) -> Dict[Tuple[int, int], str]:
    """Deprecated: Player ID mapping is unreliable across datasets.
    
    Kept for reference only.
    """
    return {}


# ---------------------------------------------------------------------------
# Aggregation by Position Group
# ---------------------------------------------------------------------------


def aggregate_by_position_group(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate xA stats by position group inferred from pass origin location.
    
    Position is inferred from where the pass originated on the pitch,
    using pitch thirds: Defenders (x < 40), Midfielders (40-80), Forwards (x >= 80).
    """
    df = df.copy()
    
    # Infer position from pass origin
    df["position_group"] = df["pass_end_x"].apply(infer_position_group)
    
    # Aggregate by position group
    stats = (
        df.groupby("position_group", dropna=False)
        .agg(
            passes=("pass_event_id", "count"),
            shots=("shot_id", pd.Series.nunique),
            xa_plus_sum=("xa_plus", "sum"),
            xa_plus_mean=("xa_plus", "mean"),
            xa_keypass_sum=("xa_keypass", "sum"),
        )
        .reset_index()
    )
    
    total_xa = float(df["xa_plus"].sum()) if len(df) > 0 else 0.0
    stats["xa_plus_share"] = stats["xa_plus_sum"] / total_xa if total_xa > 0 else 0.0
    
    return stats.sort_values("xa_plus_sum", ascending=False)


# ---------------------------------------------------------------------------
# Analysis by Team Style Cluster
# ---------------------------------------------------------------------------


def split_by_team_clusters(
    df: pd.DataFrame,
    team_clusters: pd.DataFrame,
) -> Dict[int, pd.DataFrame]:
    """Split baseline data by team style cluster.
    
    Args:
        df: baseline with team_id, etc.
        team_clusters: DataFrame with columns [team_id, cluster, cluster_label]
    
    Returns:
        Dict mapping cluster_id -> filtered dataframe
    """
    df = df.copy()
    
    # Join team cluster info
    team_cluster_map = dict(zip(team_clusters["team_id"], team_clusters["cluster"]))
    df["team_cluster"] = df["team_id"].map(team_cluster_map)
    
    # Split by cluster
    splits = {}
    for cluster_id in sorted(df["team_cluster"].dropna().unique()):
        splits[int(cluster_id)] = df[df["team_cluster"] == cluster_id]
    
    return splits


def analyze_position_groups_by_team_style(
    baseline_csv: Path,
    team_clusters_csv: Path,
) -> Dict[str, pd.DataFrame]:
    """Main analysis: position groups split by team style clusters.
    
    Returns dict with keys:
    - 'position_overall': position group stats across all data
    - 'position_by_cluster_<id>': position group stats for each team style cluster
    - 'cluster_labels': mapping of cluster_id -> label
    """
    
    # Load data
    baseline_df = pd.read_csv(baseline_csv)
    team_clusters = pd.read_csv(team_clusters_csv)
    
    results = {}
    
    # Overall position stats
    print("Computing overall position group stats...")
    overall_stats = aggregate_by_position_group(baseline_df)
    results["position_overall"] = overall_stats
    
    # By team style cluster
    splits = split_by_team_clusters(baseline_df, team_clusters)
    
    for cluster_id, cluster_df in splits.items():
        label = team_clusters[team_clusters["cluster"] == cluster_id]["cluster_label"].iloc[0]
        print(f"Computing position stats for cluster {cluster_id} ({label})...")
        
        stats = aggregate_by_position_group(cluster_df)
        results[f"position_by_cluster_{cluster_id}"] = stats
    
    results["cluster_labels"] = dict(
        zip(team_clusters["cluster"], team_clusters["cluster_label"])
    )
    
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_position_group_comparison(
    position_overall: pd.DataFrame,
    results_by_cluster: Dict[int, pd.DataFrame],
    cluster_labels: Dict[int, str],
) -> plt.Figure:
    """Multi-panel plot comparing position group stats across team styles.
    
    One row per team style cluster, columns for different metrics.
    """
    n_clusters = len(results_by_cluster)
    fig, axes = plt.subplots(
        n_clusters + 1, 2,
        figsize=(STYLE.fig_width * 1.4, STYLE.fig_height * 0.8 * (n_clusters + 1))
    )
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    colors = {"Defenders": "#1f77b4", "Midfielders": "#ff7f0e", "Forwards": "#2ca02c"}
    
    # --- Overall stats (top row) ---
    ax_left, ax_right = axes[0]
    
    # xA+ share by position
    pos_overall = position_overall[position_overall["position_group"] != "Unknown"]
    ax_left.barh(
        pos_overall["position_group"],
        pos_overall["xa_plus_share"] * 100,
        color=[colors.get(p, "gray") for p in pos_overall["position_group"]],
        alpha=0.7,
    )
    ax_left.set_xlabel("xA+ Share (%)")
    ax_left.set_title("Overall: xA+ Distribution by Position")
    ax_left.grid(True, alpha=0.3, axis="x")
    
    # xA per pass by position
    ax_right.barh(
        pos_overall["position_group"],
        pos_overall["xa_plus_mean"],
        color=[colors.get(p, "gray") for p in pos_overall["position_group"]],
        alpha=0.7,
    )
    ax_right.set_xlabel("Mean xA+ per Pass")
    ax_right.set_title("Overall: Quality by Position")
    ax_right.grid(True, alpha=0.3, axis="x")
    
    # --- By team style cluster ---
    for row_idx, (cluster_id, label) in enumerate(sorted(cluster_labels.items()), 1):
        if cluster_id not in results_by_cluster:
            continue
        
        ax_left, ax_right = axes[row_idx]
        
        pos_stats = results_by_cluster[cluster_id]
        pos_stats = pos_stats[pos_stats["position_group"] != "Unknown"]
        
        # xA+ share
        ax_left.barh(
            pos_stats["position_group"],
            pos_stats["xa_plus_share"] * 100,
            color=[colors.get(p, "gray") for p in pos_stats["position_group"]],
            alpha=0.7,
        )
        ax_left.set_xlabel("xA+ Share (%)")
        ax_left.set_title(f"{label}: xA+ Distribution by Position")
        ax_left.grid(True, alpha=0.3, axis="x")
        
        # xA per pass
        ax_right.barh(
            pos_stats["position_group"],
            pos_stats["xa_plus_mean"],
            color=[colors.get(p, "gray") for p in pos_stats["position_group"]],
            alpha=0.7,
        )
        ax_right.set_xlabel("Mean xA+ per Pass")
        ax_right.set_title(f"{label}: Quality by Position")
        ax_right.grid(True, alpha=0.3, axis="x")
    
    fig.suptitle("Position Group Analysis by Team Style", fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout()
    
    return fig


def plot_position_radar_by_cluster(
    results_by_cluster: Dict[int, pd.DataFrame],
    cluster_labels: Dict[int, str],
) -> plt.Figure:
    """Radar charts showing position group profiles for each team style.
    
    One radar per team style cluster.
    """
    n_clusters = len(results_by_cluster)
    
    # Create subplots with polar projection
    fig = plt.figure(figsize=(STYLE.fig_width * 1.4, STYLE.fig_height * 1.2))
    
    positions_of_interest = ["Defenders", "Midfielders", "Forwards"]
    angles = np.linspace(0, 2 * np.pi, len(positions_of_interest), endpoint=False).tolist()
    angles += angles[:1]
    
    n_cols = min(3, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    for plot_idx, (cluster_id, label) in enumerate(sorted(cluster_labels.items())):
        if cluster_id not in results_by_cluster:
            continue
        
        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection="polar")
        
        pos_stats = results_by_cluster[cluster_id]
        
        # Get xA per pass for each position
        values = []
        for pos in positions_of_interest:
            val = pos_stats[pos_stats["position_group"] == pos]["xa_plus_mean"].values
            values.append(float(val[0]) if len(val) > 0 else 0.0)
        
        # Normalize
        max_val = max(values) if max(values) > 0 else 1
        values = [v / max_val for v in values]
        values += values[:1]
        
        ax.plot(angles, values, "o-", linewidth=2, color="#ff7f0e")
        ax.fill(angles, values, alpha=0.25, color="#ff7f0e")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(positions_of_interest, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(label, fontsize=10, pad=15)
        ax.grid(True)
    
    fig.suptitle("Position Group Profiles by Team Style", fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    
    return fig


# ---------------------------------------------------------------------------
# Main Analysis Runner
# ---------------------------------------------------------------------------


def run_position_cluster_analysis(
    baseline_csv: Path,
    team_clusters_csv: Path,
    events_dir: Path = None,
) -> Dict[str, pd.DataFrame]:
    """Run full position-based cluster analysis.
    
    Args:
        baseline_csv: Path to baseline xA CSV
        team_clusters_csv: Path to team clusters CSV
        events_dir: Unused (deprecated)
    
    Returns:
        Dict of results DataFrames
    """
    
    configure_matplotlib()
    csv_dir = get_analysis_output_dir("cxa", subdir="csv")
    plots_dir = get_analysis_output_dir("cxa", subdir="plots")
    
    # Run analysis
    print("Analyzing position groups by team style...")
    results = analyze_position_groups_by_team_style(
        baseline_csv,
        team_clusters_csv,
    )
    
    # Save CSVs
    print(f"Saving results to {csv_dir}...")
    
    results["position_overall"].to_csv(
        csv_dir / "cxa_position_groups_overall.csv", index=False
    )
    
    for cluster_id in sorted(results.get("cluster_labels", {}).keys()):
        key = f"position_by_cluster_{cluster_id}"
        if key in results:
            results[key].to_csv(
                csv_dir / f"cxa_position_groups_cluster{cluster_id}.csv", index=False
            )
    
    # Generate plots
    print(f"Generating plots to {plots_dir}...")
    
    results_by_cluster = {
        int(k.split("_")[-1]): v
        for k, v in results.items()
        if k.startswith("position_by_cluster_")
    }
    
    fig = plot_position_group_comparison(
        results["position_overall"],
        results_by_cluster,
        results.get("cluster_labels", {}),
    )
    save_figure(fig, "cxa", "cxa_position_groups_by_cluster")
    plt.close(fig)
    
    fig = plot_position_radar_by_cluster(
        results_by_cluster,
        results.get("cluster_labels", {}),
    )
    save_figure(fig, "cxa", "cxa_position_radar_by_cluster")
    plt.close(fig)
    
    print(f"Position cluster analysis complete!")
    
    return results
