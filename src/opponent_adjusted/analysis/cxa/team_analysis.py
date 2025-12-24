"""Team-level cxA analysis.

Provides:
- Team attacking profiles
- Team clustering by creative style
- Team xA breakdowns by pass type/zone

Outputs: CSV tables and plots under outputs/analysis/cxa/
"""

from __future__ import annotations

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
from opponent_adjusted.db.models import Team


# ---------------------------------------------------------------------------
# Team Profiles
# ---------------------------------------------------------------------------


def compute_team_profiles(df: pd.DataFrame, min_passes: int = 20) -> pd.DataFrame:
    """Compute feature profiles for each team's attacking style."""

    # Ensure team_name exists
    if "team_name" not in df.columns:
        df = df.copy()
        df["team_name"] = df["team_id"].astype(str)

    profiles: List[dict] = []

    for (team_id, team_name), grp in df.groupby(["team_id", "team_name"], dropna=False):
        if len(grp) < min_passes:
            continue

        total_passes = len(grp)
        total_xa_plus = float(grp["xa_plus"].sum())
        n_shots = int(grp["shot_id"].nunique())

        profile = {
            "team_id": team_id,
            "team_name": team_name,
            "total_passes": total_passes,
            "n_shots": n_shots,
            "total_xa_plus": total_xa_plus,
            "xa_per_pass": total_xa_plus / total_passes if total_passes > 0 else 0.0,
            "xa_per_shot": total_xa_plus / n_shots if n_shots > 0 else 0.0,
            # Style percentages
            "cross_pct": float(grp["is_cross"].sum()) / total_passes,
            "through_ball_pct": float(grp["is_through_ball"].sum()) / total_passes,
            "under_pressure_pct": float(grp["under_pressure"].sum()) / total_passes,
            "ground_pass_pct": (grp["pass_height"] == "Ground Pass").sum() / total_passes,
            "high_pass_pct": (grp["pass_height"] == "High Pass").sum() / total_passes,
            "low_pass_pct": (grp["pass_height"] == "Low Pass").sum() / total_passes,
            # Spatial tendencies
            "mean_end_x": float(grp["pass_end_x"].mean()) if grp["pass_end_x"].notna().any() else 60.0,
            "mean_end_y": float(grp["pass_end_y"].mean()) if grp["pass_end_y"].notna().any() else 40.0,
            # Pass type mix
            "corner_pct": (grp["pass_type"] == "Corner").sum() / total_passes,
            "free_kick_pct": (grp["pass_type"] == "Free Kick").sum() / total_passes,
        }
        profiles.append(profile)

    return pd.DataFrame(profiles)


# ---------------------------------------------------------------------------
# Team Clustering
# ---------------------------------------------------------------------------


TEAM_CLUSTERING_FEATURES = [
    "xa_per_pass",
    "cross_pct",
    "through_ball_pct",
    "under_pressure_pct",
    "ground_pass_pct",
    "high_pass_pct",
    "mean_end_x",
    "mean_end_y",
]


def cluster_teams(
    profiles_df: pd.DataFrame,
    n_clusters: int = 4,
    features: List[str] = TEAM_CLUSTERING_FEATURES,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Cluster teams by attacking style using hierarchical clustering.

    Returns:
        - profiles_df with added 'cluster' column
        - linkage matrix for dendrogram plotting
    """

    X = profiles_df[features].copy()
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    linkage_matrix = linkage(X_scaled, method="ward")
    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    profiles_df = profiles_df.copy()
    profiles_df["cluster"] = clusters

    return profiles_df, linkage_matrix


def label_team_clusters(profiles_df: pd.DataFrame) -> Dict[int, str]:
    """Generate interpretable labels for each team cluster based on style characteristics."""

    labels = {}

    for cluster_id in profiles_df["cluster"].unique():
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]

        mean_cross = cluster_data["cross_pct"].mean()
        mean_through = cluster_data["through_ball_pct"].mean()
        mean_xa = cluster_data["xa_per_pass"].mean()
        mean_ground = cluster_data["ground_pass_pct"].mean()
        mean_end_x = cluster_data["mean_end_x"].mean()

        # Cluster 2: cross 0.325, ground 0.354 → "Direct / Wide Attacking"
        # Cluster 1: cross 0.284, ground 0.468 → "Wide Play Buildup"
        # Cluster 3: cross 0.239, through 0.048, ground 0.524, xA 0.096 → "Possession Creators"
        # Cluster 4: cross 0.220, ground 0.583, xA 0.076 → "Possession / Conservative"
        
        if mean_cross > 0.31:
            label = "Direct / Wide Attacking"
        elif mean_cross > 0.27:
            label = "Wide Play Buildup"
        elif mean_through > 0.04:
            label = "Possession Creators"
        elif mean_ground > 0.55:
            label = "Possession / Conservative"
        else:
            label = "Balanced Attack"

        labels[cluster_id] = label

    return labels


def plot_team_clusters(
    profiles_df: pd.DataFrame,
    cluster_labels: Dict[int, str],
    x_feature: str = "cross_pct",
    y_feature: str = "through_ball_pct",
) -> plt.Figure:
    """Scatter plot of teams colored by cluster."""

    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))

    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_labels)))

    for i, (cluster_id, label) in enumerate(cluster_labels.items()):
        cluster_data = profiles_df[profiles_df["cluster"] == cluster_id]
        ax.scatter(
            cluster_data[x_feature],
            cluster_data[y_feature],
            c=[colors[i]],
            label=f"{label} (n={len(cluster_data)})",
            alpha=0.7,
            s=cluster_data["total_xa_plus"] * 3 + 50,  # Size by total xA
        )

        # Label team names
        for _, row in cluster_data.iterrows():
            name = str(row["team_name"])[:12] if pd.notna(row["team_name"]) else f"ID:{row['team_id']}"
            ax.annotate(
                name,
                (row[x_feature], row[y_feature]),
                fontsize=6,
                alpha=0.7,
            )

    ax.set_xlabel(x_feature.replace("_", " ").title())
    ax.set_ylabel(y_feature.replace("_", " ").title())
    ax.set_title("Team Attacking Styles")
    ax.legend(loc="best", fontsize=8)

    return fig


def plot_team_dendrogram(
    linkage_matrix: np.ndarray,
    profiles_df: pd.DataFrame,
) -> plt.Figure:
    """Plot hierarchical clustering dendrogram for teams."""

    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.5, STYLE.fig_height))

    labels = profiles_df["team_name"].fillna("Unknown").astype(str).tolist()

    dendrogram(
        linkage_matrix,
        ax=ax,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
    )

    ax.set_title("Team Clustering Dendrogram")
    ax.set_xlabel("Team")
    ax.set_ylabel("Distance")

    return fig


# ---------------------------------------------------------------------------
# Team Style Comparison
# ---------------------------------------------------------------------------


def plot_team_style_radar(
    profiles_df: pd.DataFrame,
    team_ids: List[int],
    features: List[str] = None,
) -> plt.Figure:
    """Radar chart comparing selected teams' styles."""

    if features is None:
        features = ["cross_pct", "through_ball_pct", "ground_pass_pct", "high_pass_pct", "xa_per_pass"]

    # Filter to selected teams
    selected = profiles_df[profiles_df["team_id"].isin(team_ids)]

    if selected.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No teams found", ha="center", va="center")
        return fig

    # Normalize features for radar chart
    feature_data = selected[features].copy()
    for col in features:
        max_val = profiles_df[col].max()
        if max_val > 0:
            feature_data[col] = feature_data[col] / max_val

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_width), subplot_kw=dict(projection="polar"))

    colors = plt.cm.tab10(np.linspace(0, 1, len(team_ids)))

    for i, (_, row) in enumerate(selected.iterrows()):
        values = row[features].tolist()
        values += values[:1]  # Close the polygon

        team_name = str(row["team_name"])[:15] if pd.notna(row["team_name"]) else f"ID:{row['team_id']}"

        ax.plot(angles, values, "o-", linewidth=2, label=team_name, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_", "\n").title() for f in features], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("Team Style Comparison")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=8)

    return fig


# ---------------------------------------------------------------------------
# Main Analysis Runner
# ---------------------------------------------------------------------------


def run_team_analysis(
    baseline_csv: Path,
    session: Session,
    *,
    min_passes: int = 20,
    n_clusters: int = 4,
) -> Dict[str, pd.DataFrame]:
    """Run full team analysis pipeline.

    Returns dict of DataFrames for further inspection.
    """

    configure_matplotlib()
    csv_dir = get_analysis_output_dir("cxa", subdir="csv")
    plots_dir = get_analysis_output_dir("cxa", subdir="plots")

    # Load data
    print("Loading baseline data...")
    df = pd.read_csv(baseline_csv)

    # Add team names if we have session
    team_ids = [int(x) for x in df["team_id"].dropna().unique().tolist()]
    team_map: Dict[int, str] = {}
    if team_ids:
        for row in session.execute(
            select(Team.id, Team.name).where(Team.id.in_(team_ids))
        ).all():
            team_map[int(row.id)] = str(row.name)

    df["team_name"] = df["team_id"].map(team_map)

    # 1. Team profiles
    print("Computing team profiles...")
    profiles = compute_team_profiles(df, min_passes=min_passes)
    profiles.to_csv(csv_dir / "cxa_team_profiles.csv", index=False)

    # 2. Clustering
    print(f"Clustering teams (n_clusters={n_clusters})...")
    clustered, linkage_matrix = cluster_teams(profiles, n_clusters=n_clusters)
    cluster_labels = label_team_clusters(clustered)

    clustered["cluster_label"] = clustered["cluster"].map(cluster_labels)
    clustered.to_csv(csv_dir / "cxa_team_clusters.csv", index=False)

    # Cluster summary
    cluster_summary = (
        clustered.groupby(["cluster", "cluster_label"])
        .agg(
            n_teams=("team_id", "count"),
            mean_xa_per_pass=("xa_per_pass", "mean"),
            mean_cross_pct=("cross_pct", "mean"),
            mean_through_ball_pct=("through_ball_pct", "mean"),
            total_xa_plus=("total_xa_plus", "sum"),
        )
        .reset_index()
    )
    cluster_summary.to_csv(csv_dir / "cxa_team_cluster_summary.csv", index=False)

    # 3. Plots
    print("Generating team plots...")

    fig = plot_team_clusters(clustered, cluster_labels)
    save_figure(fig, "cxa", "cxa_team_clusters_scatter")
    plt.close(fig)

    # Radar chart for top teams by xA
    top_teams = clustered.nlargest(5, "total_xa_plus")["team_id"].tolist()
    fig = plot_team_style_radar(clustered, [int(t) for t in top_teams])
    save_figure(fig, "cxa", "cxa_team_style_radar_top5")
    plt.close(fig)

    print(f"Saved team analysis outputs to {csv_dir} and {plots_dir}")

    return {
        "profiles": profiles,
        "clustered": clustered,
        "cluster_summary": cluster_summary,
    }
