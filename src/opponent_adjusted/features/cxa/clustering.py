"""Player and Team clustering for CxA analysis.

Clusters players by passing style and positional role.
Clusters teams by attacking patterns and xT profiles.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =============================================================================
# PLAYER CLUSTERING
# =============================================================================


def build_player_features(
    passes_df: pd.DataFrame,
    lineups_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build player-level features for clustering.

    Uses passing stats, xT metrics, and positional information.

    Args:
        passes_df: Pass-level data with xT features
        lineups_df: Optional lineup data with position/formation info

    Returns:
        DataFrame with one row per player and clustering features
    """
    logger.info("Building player features for clustering...")

    # Determine player name column
    name_col = "passer_name" if "passer_name" in passes_df.columns else "player_name"

    # Aggregate pass stats per player
    player_stats = (
        passes_df.groupby(["player_id", name_col])
        .agg(
            # Volume
            total_passes=("pass_id", "count"),
            # Completion
            completed_passes=("is_complete", "sum"),
            # Progressive
            progressive_passes=(
                ("is_progressive", "sum")
                if "is_progressive" in passes_df.columns
                else ("pass_id", "count")
            ),
            # Into box
            into_box_passes=(
                ("is_into_box", "sum")
                if "is_into_box" in passes_df.columns
                else ("pass_id", "count")
            ),
            # Crosses
            crosses=(
                ("is_cross", "sum") if "is_cross" in passes_df.columns else ("pass_id", "count")
            ),
            # Through balls
            through_balls=(
                ("is_through_ball", "sum")
                if "is_through_ball" in passes_df.columns
                else ("pass_id", "count")
            ),
            # xT features
            total_xt_gained=(
                ("xt_delta", "sum") if "xt_delta" in passes_df.columns else ("pass_id", "count")
            ),
            mean_xt_delta=(
                ("xt_delta", "mean") if "xt_delta" in passes_df.columns else ("pass_id", "count")
            ),
            max_xt_delta=(
                ("xt_delta", "max") if "xt_delta" in passes_df.columns else ("pass_id", "count")
            ),
            # Spatial - average positions
            mean_start_x=("start_x", "mean"),
            mean_start_y=("start_y", "mean"),
            mean_end_x=("end_x", "mean"),
            mean_end_y=("end_y", "mean"),
            # Sequence involvement
            key_passes=(
                ("is_key_pass", "sum")
                if "is_key_pass" in passes_df.columns
                else ("pass_id", "count")
            ),
            second_assists=(
                ("is_second_assist", "sum")
                if "is_second_assist" in passes_df.columns
                else ("pass_id", "count")
            ),
            # Team info for later joining
            team_id=("team_id", "first"),
        )
        .reset_index()
    )

    # Rename to standard column name
    if name_col != "player_name":
        player_stats = player_stats.rename(columns={name_col: "player_name"})

    # Derived ratios
    player_stats["completion_rate"] = (
        player_stats["completed_passes"] / player_stats["total_passes"]
    ).fillna(0)

    player_stats["progressive_rate"] = (
        player_stats["progressive_passes"] / player_stats["total_passes"]
    ).fillna(0)

    player_stats["into_box_rate"] = (
        player_stats["into_box_passes"] / player_stats["total_passes"]
    ).fillna(0)

    player_stats["cross_rate"] = (player_stats["crosses"] / player_stats["total_passes"]).fillna(0)

    player_stats["through_ball_rate"] = (
        player_stats["through_balls"] / player_stats["total_passes"]
    ).fillna(0)

    player_stats["xt_per_pass"] = (
        player_stats["total_xt_gained"] / player_stats["total_passes"]
    ).fillna(0)

    # Mean pass distance
    player_stats["mean_pass_distance"] = np.sqrt(
        (player_stats["mean_end_x"] - player_stats["mean_start_x"]) ** 2
        + (player_stats["mean_end_y"] - player_stats["mean_start_y"]) ** 2
    )

    # Add positional info from lineups if available
    if lineups_df is not None and not lineups_df.empty:
        player_stats = _add_positional_features(player_stats, lineups_df)
    else:
        player_stats["position_name"] = "Unknown"
        player_stats["formation"] = "Unknown"
        player_stats["position_group"] = "Unknown"

    logger.info(f"Built features for {len(player_stats):,} players")

    return player_stats


def _add_positional_features(
    player_stats: pd.DataFrame,
    lineups_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add position and formation info from lineups."""
    # Determine position column name - lineup uses tactical_position
    pos_col = None
    for col in ["tactical_position", "position_name", "position"]:
        if col in lineups_df.columns:
            pos_col = col
            break

    if pos_col is None:
        player_stats["position_name"] = "Unknown"
        player_stats["formation"] = "Unknown"
        player_stats["position_group"] = "Unknown"
        return player_stats

    # Get most common position for each player
    def safe_mode(x):
        m = x.mode()
        return m.iloc[0] if len(m) > 0 else "Unknown"

    agg_dict = {pos_col: safe_mode}
    if "formation" in lineups_df.columns:
        agg_dict["formation"] = safe_mode

    position_mode = lineups_df.groupby("player_id").agg(agg_dict).reset_index()

    # Rename to standard column
    position_mode = position_mode.rename(columns={pos_col: "position_name"})

    player_stats = player_stats.merge(
        position_mode,
        on="player_id",
        how="left",
    )

    player_stats["position_name"] = player_stats["position_name"].fillna("Unknown")
    if "formation" not in player_stats.columns:
        player_stats["formation"] = "Unknown"
    else:
        player_stats["formation"] = player_stats["formation"].fillna("Unknown")

    # Create position groups
    player_stats["position_group"] = player_stats["position_name"].apply(_get_position_group)

    return player_stats


def _get_position_group(position: str) -> str:
    """Map position name to broader group."""
    if pd.isna(position) or position == "Unknown":
        return "Unknown"

    position = position.lower()

    # Goalkeepers
    if "goalkeeper" in position or position == "gk":
        return "GK"

    # Defenders
    if any(x in position for x in ["back", "defender", "cb", "lb", "rb", "rwb", "lwb"]):
        if "wing" in position or "rwb" in position or "lwb" in position:
            return "Wing Back"
        if "center" in position or "centre" in position or "cb" in position:
            return "Center Back"
        return "Full Back"

    # Midfielders
    if any(x in position for x in ["midfield", "cm", "dm", "am", "cdm", "cam"]):
        if "defensive" in position or "cdm" in position or "dm" in position:
            return "Defensive Mid"
        if "attacking" in position or "cam" in position or "am" in position:
            return "Attacking Mid"
        return "Central Mid"

    # Wingers
    if any(x in position for x in ["wing", "lw", "rw", "lm", "rm"]):
        return "Winger"

    # Forwards
    if any(x in position for x in ["forward", "striker", "cf", "st"]):
        return "Forward"

    return "Unknown"


def cluster_players(
    player_features: pd.DataFrame,
    n_clusters: int = 6,
    features_for_clustering: Optional[list] = None,
    include_position: bool = True,
) -> Tuple[pd.DataFrame, Optional[KMeans], Optional[StandardScaler]]:
    """Cluster players by passing style.

    Args:
        player_features: Player feature DataFrame
        n_clusters: Number of clusters
        features_for_clustering: List of feature columns to use
        include_position: Whether to include position in clustering

    Returns:
        Tuple of (player_df with cluster labels, fitted KMeans, fitted Scaler)
    """
    logger.info(f"Clustering players into {n_clusters} clusters...")

    df = player_features.copy()

    # Default features for clustering
    if features_for_clustering is None:
        features_for_clustering = [
            "completion_rate",
            "progressive_rate",
            "into_box_rate",
            "cross_rate",
            "through_ball_rate",
            "xt_per_pass",
            "mean_start_x",  # Average position on pitch
            "mean_pass_distance",
        ]

    # Add position encoding if requested
    if include_position and "position_group" in df.columns:
        # One-hot encode position groups
        position_dummies = pd.get_dummies(df["position_group"], prefix="pos")
        df = pd.concat([df, position_dummies], axis=1)
        features_for_clustering = features_for_clustering + list(position_dummies.columns)

    # Filter to available features
    available_features = [f for f in features_for_clustering if f in df.columns]

    if len(available_features) < 3:
        logger.warning(f"Not enough features for clustering: {available_features}")
        df["player_cluster"] = 0
        return df, None, None

    # Prepare feature matrix
    X = df[available_features].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["player_cluster"] = kmeans.fit_predict(X_scaled)

    # Add cluster labels
    df = _add_player_cluster_labels(df)

    logger.info("Player cluster distribution:")
    for cluster in sorted(df["player_cluster"].unique()):
        count = (df["player_cluster"] == cluster).sum()
        label = df[df["player_cluster"] == cluster]["player_cluster_label"].iloc[0]
        logger.info(f"  Cluster {cluster} ({label}): {count} players")

    return df, kmeans, scaler


def _add_player_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive labels to player clusters based on characteristics."""
    cluster_profiles = df.groupby("player_cluster").agg(
        {
            "progressive_rate": "mean",
            "into_box_rate": "mean",
            "cross_rate": "mean",
            "xt_per_pass": "mean",
            "mean_start_x": "mean",
            "completion_rate": "mean",
        }
    )

    # Use percentile-based thresholds for better labeling
    thresholds = {
        "mean_start_x": cluster_profiles["mean_start_x"].quantile([0.33, 0.66]).values,
        "progressive_rate": cluster_profiles["progressive_rate"].median(),
        "cross_rate": cluster_profiles["cross_rate"].median(),
        "into_box_rate": cluster_profiles["into_box_rate"].median(),
        "xt_per_pass": cluster_profiles["xt_per_pass"].median(),
        "completion_rate": cluster_profiles["completion_rate"].median(),
    }

    labels = {}
    used_labels = set()

    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]

        # Build label based on distinctive features
        # Deep vs Mid vs Advanced based on mean_start_x
        if profile["mean_start_x"] < thresholds["mean_start_x"][0]:
            zone = "Deep"
        elif profile["mean_start_x"] > thresholds["mean_start_x"][1]:
            zone = "Advanced"
        else:
            zone = "Central"

        # Style based on other features
        if profile["cross_rate"] > thresholds["cross_rate"] * 1.5:
            style = "Wide"
        elif profile["progressive_rate"] > thresholds["progressive_rate"] * 1.2:
            style = "Progressive"
        elif profile["into_box_rate"] > thresholds["into_box_rate"] * 1.2:
            style = "Box Threat"
        elif profile["xt_per_pass"] > thresholds["xt_per_pass"] * 1.2:
            style = "Chance Creator"
        elif profile["completion_rate"] > thresholds["completion_rate"] * 1.05:
            style = "Safe"
        else:
            style = "Circulator"

        label = f"{zone} {style}"

        # Ensure unique labels
        if label in used_labels:
            label = f"{label} {cluster}"
        used_labels.add(label)
        labels[cluster] = label

    df["player_cluster_label"] = df["player_cluster"].map(labels)

    return df


# =============================================================================
# TEAM CLUSTERING
# =============================================================================


def build_team_features(
    possessions_df: pd.DataFrame,
    passes_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build team-level features for clustering.

    Uses possession patterns, xT profiles, and attacking style.

    Args:
        possessions_df: Possession-level data
        passes_df: Optional pass data for additional features

    Returns:
        DataFrame with one row per team
    """
    logger.info("Building team features for clustering...")

    # Aggregate possession stats per team
    team_stats = (
        possessions_df.groupby("team_id")
        .agg(
            # Volume
            total_possessions=("num_passes", "count"),
            total_passes=("num_passes", "sum"),
            # Possession characteristics
            mean_passes_per_poss=("num_passes", "mean"),
            mean_poss_duration=(
                ("duration_seconds", "mean")
                if "duration_seconds" in possessions_df.columns
                else ("num_passes", "mean")
            ),
            # xT
            total_xt_gained=(
                ("total_xt_gained", "sum")
                if "total_xt_gained" in possessions_df.columns
                else ("num_passes", "count")
            ),
            mean_xt_per_poss=(
                ("total_xt_gained", "mean")
                if "total_xt_gained" in possessions_df.columns
                else ("num_passes", "mean")
            ),
            # Shot creation
            possessions_with_shot=(
                ("ended_in_shot", "sum")
                if "ended_in_shot" in possessions_df.columns
                else ("num_passes", "count")
            ),
            possessions_with_goal=(
                ("ended_in_goal", "sum")
                if "ended_in_goal" in possessions_df.columns
                else ("num_passes", "count")
            ),
            total_shot_xg=(
                ("shot_xg", "sum")
                if "shot_xg" in possessions_df.columns
                else ("num_passes", "count")
            ),
            # Pass types in possessions
            progressive_passes=(
                ("num_progressive", "sum")
                if "num_progressive" in possessions_df.columns
                else ("num_passes", "count")
            ),
            into_box_passes=(
                ("num_into_box", "sum")
                if "num_into_box" in possessions_df.columns
                else ("num_passes", "count")
            ),
            crosses=(
                ("num_crosses", "sum")
                if "num_crosses" in possessions_df.columns
                else ("num_passes", "count")
            ),
            # Zone progression
            completion_rate=(
                ("completion_rate", "mean")
                if "completion_rate" in possessions_df.columns
                else ("num_passes", "mean")
            ),
        )
        .reset_index()
    )

    # Derived metrics
    team_stats["shot_creation_rate"] = (
        team_stats["possessions_with_shot"] / team_stats["total_possessions"]
    ).fillna(0)

    team_stats["goal_conversion_rate"] = (
        team_stats["possessions_with_goal"] / team_stats["possessions_with_shot"]
    ).fillna(0)

    team_stats["xt_per_pass"] = (team_stats["total_xt_gained"] / team_stats["total_passes"]).fillna(
        0
    )

    team_stats["progressive_rate"] = (
        team_stats["progressive_passes"] / team_stats["total_passes"]
    ).fillna(0)

    team_stats["cross_rate"] = (team_stats["crosses"] / team_stats["total_passes"]).fillna(0)

    team_stats["directness"] = (
        team_stats["shot_creation_rate"] / team_stats["mean_passes_per_poss"]
    ).fillna(0)

    # Add team names if passes_df available
    if passes_df is not None and "team_name" in passes_df.columns:
        team_names = passes_df[["team_id", "team_name"]].drop_duplicates()
        team_stats = team_stats.merge(team_names, on="team_id", how="left")

    logger.info(f"Built features for {len(team_stats):,} teams")

    return team_stats


def cluster_teams(
    team_features: pd.DataFrame,
    n_clusters: int = 4,
    features_for_clustering: Optional[list] = None,
) -> Tuple[pd.DataFrame, Optional[KMeans], Optional[StandardScaler]]:
    """Cluster teams by attacking style.

    Args:
        team_features: Team feature DataFrame
        n_clusters: Number of clusters
        features_for_clustering: List of feature columns to use

    Returns:
        Tuple of (team_df with cluster labels, fitted KMeans, fitted Scaler)
    """
    logger.info(f"Clustering teams into {n_clusters} clusters...")

    df = team_features.copy()

    # Default features for clustering
    if features_for_clustering is None:
        features_for_clustering = [
            "mean_passes_per_poss",
            "mean_xt_per_poss",
            "shot_creation_rate",
            "progressive_rate",
            "cross_rate",
            "directness",
            "completion_rate",
        ]

    # Filter to available features
    available_features = [f for f in features_for_clustering if f in df.columns]

    if len(available_features) < 3:
        logger.warning(f"Not enough features for clustering: {available_features}")
        df["team_cluster"] = 0
        return df, None, None

    # Prepare feature matrix
    X = df[available_features].fillna(0).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["team_cluster"] = kmeans.fit_predict(X_scaled)

    # Add cluster labels
    df = _add_team_cluster_labels(df)

    logger.info("Team cluster distribution:")
    for cluster in sorted(df["team_cluster"].unique()):
        count = (df["team_cluster"] == cluster).sum()
        label = df[df["team_cluster"] == cluster]["team_cluster_label"].iloc[0]
        logger.info(f"  Cluster {cluster} ({label}): {count} teams")

    return df, kmeans, scaler


def _add_team_cluster_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add descriptive labels to team clusters based on characteristics."""
    cluster_profiles = df.groupby("team_cluster").agg(
        {
            "mean_passes_per_poss": "mean",
            "shot_creation_rate": "mean",
            "progressive_rate": "mean",
            "cross_rate": "mean",
            "directness": "mean",
        }
    )

    labels = {}
    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]

        # Determine label based on profile
        if profile["directness"] > cluster_profiles["directness"].median():
            if profile["cross_rate"] > cluster_profiles["cross_rate"].median():
                labels[cluster] = "Direct Wide"
            else:
                labels[cluster] = "Direct Central"
        else:
            if profile["mean_passes_per_poss"] > cluster_profiles["mean_passes_per_poss"].median():
                labels[cluster] = "Possession Build-up"
            else:
                labels[cluster] = "Balanced"

    df["team_cluster_label"] = df["team_cluster"].map(labels)

    return df


# =============================================================================
# ENRICHMENT FUNCTIONS
# =============================================================================


def enrich_with_player_clusters(
    df: pd.DataFrame,
    player_clusters: pd.DataFrame,
    player_id_col: str = "player_id",
) -> pd.DataFrame:
    """Add player cluster info to a DataFrame.

    Args:
        df: DataFrame to enrich (passes, sequences, etc.)
        player_clusters: Player clustering result
        player_id_col: Name of player ID column

    Returns:
        DataFrame with player_cluster and player_cluster_label columns
    """
    cluster_cols = ["player_id", "player_cluster", "player_cluster_label", "position_group"]
    available_cols = [c for c in cluster_cols if c in player_clusters.columns]

    df = df.merge(
        player_clusters[available_cols],
        left_on=player_id_col,
        right_on="player_id",
        how="left",
        suffixes=("", "_cluster"),
    )

    # Fill missing
    df["player_cluster"] = df["player_cluster"].fillna(-1).astype(int)
    df["player_cluster_label"] = df["player_cluster_label"].fillna("Unknown")

    return df


def enrich_with_team_clusters(
    df: pd.DataFrame,
    team_clusters: pd.DataFrame,
    team_id_col: str = "team_id",
) -> pd.DataFrame:
    """Add team cluster info to a DataFrame.

    Args:
        df: DataFrame to enrich
        team_clusters: Team clustering result
        team_id_col: Name of team ID column

    Returns:
        DataFrame with team_cluster and team_cluster_label columns
    """
    cluster_cols = ["team_id", "team_cluster", "team_cluster_label"]
    available_cols = [c for c in cluster_cols if c in team_clusters.columns]

    df = df.merge(
        team_clusters[available_cols],
        left_on=team_id_col,
        right_on="team_id",
        how="left",
        suffixes=("", "_cluster"),
    )

    # Fill missing
    df["team_cluster"] = df["team_cluster"].fillna(-1).astype(int)
    df["team_cluster_label"] = df["team_cluster_label"].fillna("Unknown")

    return df


def enrich_sequences_with_clusters(
    sequences_df: pd.DataFrame,
    player_clusters: pd.DataFrame,
    team_clusters: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """Enrich assist sequences with cluster labels for each pass position.

    Args:
        sequences_df: Assist sequence data
        player_clusters: Player clustering result
        team_clusters: Team clustering result
        k: Number of pass positions to enrich

    Returns:
        Sequences with cluster columns for each passer
    """
    logger.info("Enriching sequences with cluster labels...")

    df = sequences_df.copy()

    # Add team cluster
    df = enrich_with_team_clusters(df, team_clusters)

    # Add player clusters for each pass position
    cluster_lookup = player_clusters.set_index("player_id")[
        ["player_cluster", "player_cluster_label", "position_group"]
    ].to_dict("index")

    for pos in range(1, k + 1):
        player_col = f"pass{pos}_player_id"

        if player_col not in df.columns:
            continue

        # Add cluster for this position
        df[f"pass{pos}_cluster"] = df[player_col].map(
            lambda x: cluster_lookup.get(x, {}).get("player_cluster", -1)
        )
        df[f"pass{pos}_cluster_label"] = df[player_col].map(
            lambda x: cluster_lookup.get(x, {}).get("player_cluster_label", "Unknown")
        )
        df[f"pass{pos}_position_group"] = df[player_col].map(
            lambda x: cluster_lookup.get(x, {}).get("position_group", "Unknown")
        )

    logger.info(f"Added cluster labels for {k} pass positions")

    return df
