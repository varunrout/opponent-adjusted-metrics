"""Player-level cxA analysis.

Provides:
- xA vs actual assists (over/under-performance)
- Player profiles with feature vectors
- Player clustering by creative style

Outputs: CSV tables and plots under outputs/analysis/cxa/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.analysis.plot_utilities import (
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
    STYLE,
)
from opponent_adjusted.db.models import Player, Shot, Team


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_baseline_with_outcomes(
    baseline_csv: Path,
    session: Session,
) -> pd.DataFrame:
    """Load baseline CSV and join shot outcomes from database.

    Adds columns:
    - shot_outcome: Goal, Saved, Blocked, etc.
    - is_goal: boolean flag
    - player_name, team_name: human-readable names
    """
    df = pd.read_csv(baseline_csv)

    # Get shot outcomes
    shot_ids = df["shot_id"].dropna().unique().tolist()
    shot_outcomes: Dict[int, str] = {}

    for row in session.execute(
        select(Shot.id, Shot.outcome).where(Shot.id.in_([int(x) for x in shot_ids]))
    ).all():
        shot_outcomes[int(row.id)] = str(row.outcome)

    df["shot_outcome"] = df["shot_id"].map(shot_outcomes)
    df["is_goal"] = df["shot_outcome"] == "Goal"

    # Add player/team names
    player_ids = [int(x) for x in df["player_id"].dropna().unique().tolist()]
    team_ids = [int(x) for x in df["team_id"].dropna().unique().tolist()]

    player_map: Dict[int, str] = {}
    if player_ids:
        for row in session.execute(
            select(Player.id, Player.name).where(Player.id.in_(player_ids))
        ).all():
            player_map[int(row.id)] = str(row.name)

    team_map: Dict[int, str] = {}
    if team_ids:
        for row in session.execute(
            select(Team.id, Team.name).where(Team.id.in_(team_ids))
        ).all():
            team_map[int(row.id)] = str(row.name)

    df["player_name"] = df["player_id"].map(player_map)
    df["team_name"] = df["team_id"].map(team_map)

    return df


# ---------------------------------------------------------------------------
# xA vs Actual Assists
# ---------------------------------------------------------------------------


def compute_xa_vs_assists(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player-level xA vs actual assists.

    For each player:
    - total_xa_plus: sum of xa_plus across all attributed passes
    - total_xa_keypass: sum of xa_keypass (industry xA definition)
    - actual_assists: count of keypasses that led to goals
    - xa_on_goals: sum of xa_keypass only on passes leading to goals
    - overperformance: actual_assists - xa_on_goals
    """

    # Filter to keypasses only (xa_keypass > 0 means it was the final pass)
    keypasses = df[df["xa_keypass"] > 0].copy()

    player_stats = (
        df.groupby(["player_id", "player_name"], dropna=False)
        .agg(
            total_passes=("pass_event_id", "count"),
            total_xa_plus=("xa_plus", "sum"),
            total_xa_keypass=("xa_keypass", "sum"),
        )
        .reset_index()
    )

    # Assists (keypasses that led to goals)
    goal_keypasses = keypasses[keypasses["is_goal"] == True]
    assist_stats = (
        goal_keypasses.groupby("player_id", dropna=False)
        .agg(
            actual_assists=("pass_event_id", "count"),
            xa_on_goals=("xa_keypass", "sum"),
        )
        .reset_index()
    )

    # Merge
    result = player_stats.merge(assist_stats, on="player_id", how="left")
    result["actual_assists"] = result["actual_assists"].fillna(0).astype(int)
    result["xa_on_goals"] = result["xa_on_goals"].fillna(0.0)

    # Over/under performance
    result["overperformance"] = result["actual_assists"] - result["xa_on_goals"]
    result["conversion_rate"] = result["actual_assists"] / result["total_xa_keypass"].replace(0, np.nan)

    # Sort by total xA
    result = result.sort_values("total_xa_plus", ascending=False)

    return result


def plot_xa_vs_assists(
    player_df: pd.DataFrame,
    *,
    min_xa: float = 0.5,
    top_n_labels: int = 20,
) -> plt.Figure:
    """Enhanced scatter plot of xA (keypass) vs actual assists with quadrant analysis."""

    filtered = player_df[player_df["total_xa_keypass"] >= min_xa].copy()

    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height * 1.1))

    # Color by overperformance
    colors = filtered["overperformance"].values
    scatter = ax.scatter(
        filtered["total_xa_keypass"],
        filtered["actual_assists"],
        c=colors,
        cmap="RdYlGn",
        alpha=0.7,
        s=filtered["total_passes"] * 2 + 30,  # Size by volume
        edgecolors="white",
        linewidth=0.5,
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Over/Under Performance", fontsize=9)

    # Diagonal line (perfect conversion)
    max_val = max(filtered["total_xa_keypass"].max(), filtered["actual_assists"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1.5, label="Perfect conversion")

    # Add quadrant shading
    median_xa = filtered["total_xa_keypass"].median()
    median_assists = filtered["actual_assists"].median()
    ax.axvline(median_xa, color="gray", alpha=0.2, linestyle=":")
    ax.axhline(median_assists, color="gray", alpha=0.2, linestyle=":")

    # Label notable players (top overperformers and underperformers)
    sorted_over = filtered.nlargest(top_n_labels // 2, "overperformance")
    sorted_under = filtered.nsmallest(top_n_labels // 2, "overperformance")
    to_label = pd.concat([sorted_over, sorted_under]).drop_duplicates(subset=["player_id"])

    for _, row in to_label.iterrows():
        name = row["player_name"] if pd.notna(row["player_name"]) else f"ID:{row['player_id']}"
        # Use last name or truncate
        short_name = name.split()[-1] if " " in str(name) else str(name)[:10]
        
        # Position text to avoid overlap
        offset = (3, 3) if row["overperformance"] > 0 else (3, -8)
        ax.annotate(
            short_name,
            (row["total_xa_keypass"], row["actual_assists"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
            fontweight="bold" if abs(row["overperformance"]) > 2 else "normal",
        )

    ax.set_xlabel("Expected Assists (xA)", fontsize=10)
    ax.set_ylabel("Actual Assists", fontsize=10)
    ax.set_title("xA vs Actual Assists\n(size = pass volume, color = over/under performance)", fontsize=11)
    ax.set_xlim(0, max_val)
    ax.set_ylim(-0.3, max_val)
    ax.legend(loc="upper left", fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Player Profiles
# ---------------------------------------------------------------------------


@dataclass
class PlayerProfile:
    """Feature vector for a player's creative style."""

    player_id: int
    player_name: Optional[str]
    team_id: Optional[int]
    team_name: Optional[str]

    # Volume
    total_passes: int
    total_xa_plus: float

    # Efficiency
    xa_per_pass: float

    # Style percentages
    cross_pct: float
    through_ball_pct: float
    under_pressure_pct: float
    ground_pass_pct: float
    high_pass_pct: float
    low_pass_pct: float

    # Spatial (mean end location)
    mean_end_x: float
    mean_end_y: float


def compute_player_profiles(df: pd.DataFrame, min_passes: int = 5) -> pd.DataFrame:
    """Compute feature profiles for each player."""

    profiles: List[dict] = []

    for (player_id, player_name), grp in df.groupby(["player_id", "player_name"], dropna=False):
        if len(grp) < min_passes:
            continue

        team_id = grp["team_id"].mode().iloc[0] if not grp["team_id"].mode().empty else None
        team_name = grp["team_name"].mode().iloc[0] if "team_name" in grp.columns and not grp["team_name"].mode().empty else None

        total_passes = len(grp)
        total_xa_plus = float(grp["xa_plus"].sum())

        profile = {
            "player_id": player_id,
            "player_name": player_name,
            "team_id": team_id,
            "team_name": team_name,
            "total_passes": total_passes,
            "total_xa_plus": total_xa_plus,
            "xa_per_pass": total_xa_plus / total_passes if total_passes > 0 else 0.0,
            "cross_pct": float(grp["is_cross"].sum()) / total_passes,
            "through_ball_pct": float(grp["is_through_ball"].sum()) / total_passes,
            "under_pressure_pct": float(grp["under_pressure"].sum()) / total_passes,
            "ground_pass_pct": (grp["pass_height"] == "Ground Pass").sum() / total_passes,
            "high_pass_pct": (grp["pass_height"] == "High Pass").sum() / total_passes,
            "low_pass_pct": (grp["pass_height"] == "Low Pass").sum() / total_passes,
            "mean_end_x": float(grp["pass_end_x"].mean()) if grp["pass_end_x"].notna().any() else 60.0,
            "mean_end_y": float(grp["pass_end_y"].mean()) if grp["pass_end_y"].notna().any() else 40.0,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ---------------------------------------------------------------------------
# Player Clustering
# ---------------------------------------------------------------------------


CLUSTERING_FEATURES = [
    "xa_per_pass",
    "cross_pct",
    "through_ball_pct",
    "under_pressure_pct",
    "ground_pass_pct",
    "high_pass_pct",
    "mean_end_x",
    "mean_end_y",
]


def cluster_players(
    profiles_df: pd.DataFrame,
    n_clusters: int = 5,
    features: List[str] = CLUSTERING_FEATURES,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Cluster players by creative style using hierarchical clustering.

    Returns:
        - profiles_df with added 'cluster' column
        - linkage matrix for dendrogram plotting
    """

    # Prepare feature matrix
    X = profiles_df[features].copy()
    X = X.fillna(0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering
    linkage_matrix = linkage(X_scaled, method="ward")
    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    profiles_df = profiles_df.copy()
    profiles_df["cluster"] = clusters

    return profiles_df, linkage_matrix


def label_clusters(profiles_df: pd.DataFrame) -> Dict[int, str]:
    """Generate interpretable labels for each cluster based on centroid features."""

    labels = {}

    for cluster_id in profiles_df["cluster"].unique():
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]

        # Compute cluster centroid characteristics
        mean_cross = cluster_data["cross_pct"].mean()
        mean_through = cluster_data["through_ball_pct"].mean()
        mean_xa = cluster_data["xa_per_pass"].mean()
        mean_end_x = cluster_data["mean_end_x"].mean()
        mean_ground = cluster_data["ground_pass_pct"].mean()

        # Assign label based on dominant characteristic
        if mean_cross > 0.4:
            label = "Wide Crossers"
        elif mean_through > 0.1:
            label = "Through-ball Specialists"
        elif mean_end_x > 105:
            label = "Final Third Creators"
        elif mean_ground > 0.6 and mean_xa > 0.1:
            label = "Central Combiners"
        elif mean_xa > 0.12:
            label = "High-Value Creators"
        else:
            label = f"Cluster {cluster_id}"

        labels[cluster_id] = label

    return labels


def plot_player_clusters(
    profiles_df: pd.DataFrame,
    cluster_labels: Dict[int, str],
) -> plt.Figure:
    """Multi-panel visualization of player creative styles by cluster.
    
    Creates a 2x2 grid showing different perspectives on player clustering:
    - Top-left: Cross % vs Through Ball % (passing style)
    - Top-right: Mean End X vs xA per Pass (penetration vs quality)
    - Bottom-left: Radar chart of cluster centroids
    - Bottom-right: Cluster legend with player counts
    """
    fig, axes = plt.subplots(2, 2, figsize=(STYLE.fig_width * 1.4, STYLE.fig_height * 1.4))
    
    n_clusters = len(cluster_labels)
    colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    
    # --- Top-left: Cross % vs Through Ball % ---
    ax = axes[0, 0]
    for i, (cluster_id, label) in enumerate(cluster_labels.items()):
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        ax.scatter(
            cluster_data["cross_pct"] * 100,
            cluster_data["through_ball_pct"] * 100,
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=cluster_data["total_xa_plus"] * 10 + 30,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Cross %")
    ax.set_ylabel("Through Ball %")
    ax.set_title("Passing Style", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Top-right: Mean End X vs xA per Pass ---
    ax = axes[0, 1]
    for i, (cluster_id, label) in enumerate(cluster_labels.items()):
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        ax.scatter(
            cluster_data["mean_end_x"],
            cluster_data["xa_per_pass"],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=cluster_data["total_xa_plus"] * 10 + 30,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Mean Pass End X (yards)")
    ax.set_ylabel("xA per Pass")
    ax.set_title("Penetration vs Quality", fontsize=10)
    ax.axvline(x=102, color="gray", linestyle="--", alpha=0.5, label="18-yard box")
    ax.grid(True, alpha=0.3)
    
    # --- Bottom-left: Radar chart of cluster centroids ---
    # Remove the regular axis and add a polar one
    axes[1, 0].remove()
    ax_radar = fig.add_subplot(2, 2, 3, projection="polar")
    
    radar_features = ["cross_pct", "through_ball_pct", "ground_pass_pct", "xa_per_pass", "mean_end_x"]
    radar_labels = ["Cross %", "Through Ball %", "Ground %", "xA/Pass", "End X"]
    
    # Normalize features to 0-1 for radar
    normalized_centroids = {}
    for feat in radar_features:
        feat_min = profiles_df[feat].min()
        feat_max = profiles_df[feat].max()
        feat_range = feat_max - feat_min if feat_max != feat_min else 1
        for cluster_id in cluster_labels.keys():
            if cluster_id not in normalized_centroids:
                normalized_centroids[cluster_id] = []
            centroid_val = profiles_df[profiles_df["cluster"] == cluster_id][feat].mean()
            normalized_centroids[cluster_id].append((centroid_val - feat_min) / feat_range)
    
    # Create radar
    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    
    for i, (cluster_id, label) in enumerate(cluster_labels.items()):
        values = normalized_centroids[cluster_id] + normalized_centroids[cluster_id][:1]
        ax_radar.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i])
        ax_radar.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=8)
    ax_radar.set_title("Cluster Profiles (Normalized)", fontsize=10, pad=15)
    
    # --- Bottom-right: Legend with details ---
    ax = axes[1, 1]
    ax.axis("off")
    
    # Add colored markers
    y_pos = 0.95
    for i, (cluster_id, label) in enumerate(cluster_labels.items()):
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        n_players = len(cluster_data)
        avg_xa = cluster_data["xa_per_pass"].mean()
        total_xa = cluster_data["total_xa_plus"].sum()
        
        ax.scatter([0.05], [y_pos], c=[colors[i]], s=100, transform=ax.transAxes)
        ax.text(
            0.12, y_pos,
            f"{label}\n  {n_players} players | xA/pass: {avg_xa:.3f} | Total xA+: {total_xa:.1f}",
            transform=ax.transAxes,
            fontsize=9,
            va="center",
            fontfamily="monospace",
        )
        y_pos -= 0.18
    
    ax.set_title("Cluster Summary", fontsize=10)
    
    fig.suptitle("Player Creative Styles Analysis", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    return fig


def plot_cluster_dendrogram(
    linkage_matrix: np.ndarray,
    profiles_df: pd.DataFrame,
    max_labels: int = 30,
) -> plt.Figure:
    """Plot hierarchical clustering dendrogram."""

    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.5, STYLE.fig_height))

    # Use player names as labels (truncated)
    labels = profiles_df["player_name"].fillna("Unknown").astype(str).tolist()
    if len(labels) > max_labels:
        labels = None  # Too many to display

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
    )

    ax.set_title("Player Clustering Dendrogram")
    ax.set_xlabel("Player")
    ax.set_ylabel("Distance")

    return fig


# ---------------------------------------------------------------------------
# Main Analysis Runner
# ---------------------------------------------------------------------------


def run_player_analysis(
    baseline_csv: Path,
    session: Session,
    *,
    min_xa_for_scatter: float = 0.5,
    min_passes_for_profile: int = 5,
    n_clusters: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Run full player analysis pipeline.

    Returns dict of DataFrames for further inspection.
    """

    configure_matplotlib()
    csv_dir = get_analysis_output_dir("cxa", subdir="csv")
    plots_dir = get_analysis_output_dir("cxa", subdir="plots")

    # Load data with outcomes
    print("Loading baseline data with shot outcomes...")
    df = load_baseline_with_outcomes(baseline_csv, session)

    # 1. xA vs Assists
    print("Computing xA vs actual assists...")
    xa_assists = compute_xa_vs_assists(df)
    xa_assists.to_csv(csv_dir / "cxa_player_xa_vs_assists.csv", index=False)

    fig = plot_xa_vs_assists(xa_assists, min_xa=min_xa_for_scatter)
    save_figure(fig, "cxa", "cxa_player_xa_vs_assists")
    plt.close(fig)

    # 2. Player profiles
    print("Computing player profiles...")
    profiles = compute_player_profiles(df, min_passes=min_passes_for_profile)
    profiles.to_csv(csv_dir / "cxa_player_profiles.csv", index=False)

    # 3. Clustering
    print(f"Clustering players (n_clusters={n_clusters})...")
    clustered, linkage_matrix = cluster_players(profiles, n_clusters=n_clusters)
    cluster_labels = label_clusters(clustered)

    # Add label column
    clustered["cluster_label"] = clustered["cluster"].map(cluster_labels)
    clustered.to_csv(csv_dir / "cxa_player_clusters.csv", index=False)

    # Cluster summary
    cluster_summary = (
        clustered.groupby(["cluster", "cluster_label"])
        .agg(
            n_players=("player_id", "count"),
            mean_xa_per_pass=("xa_per_pass", "mean"),
            mean_cross_pct=("cross_pct", "mean"),
            mean_through_ball_pct=("through_ball_pct", "mean"),
            mean_end_x=("mean_end_x", "mean"),
            total_xa_plus=("total_xa_plus", "sum"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(csv_dir / "cxa_player_cluster_summary.csv", index=False)

    # Cluster plots
    fig = plot_player_clusters(clustered, cluster_labels)
    save_figure(fig, "cxa", "cxa_player_clusters_scatter")
    plt.close(fig)

    print(f"Saved player analysis outputs to {csv_dir} and {plots_dir}")

    return {
        "xa_vs_assists": xa_assists,
        "profiles": profiles,
        "clustered": clustered,
        "cluster_summary": cluster_summary,
    }
