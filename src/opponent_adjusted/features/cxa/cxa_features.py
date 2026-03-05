"""cxA Feature Engineering Module.

Builds features required for cxA submodels:
- Pass completion model (A)
- Shot-within-k hazard model (C)  
- Conditional shot quality model (D)

Features include:
- Pass characteristics
- Spatial/zone features
- Sequence aggregates
- Match state context
- Opponent defensive profile
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# PASS-LEVEL FEATURES (for completion model)
# =============================================================================

def add_pass_difficulty_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pass difficulty features for completion model.
    
    Args:
        df: Passes DataFrame
        
    Returns:
        DataFrame with difficulty features added
    """
    df = df.copy()
    
    # Distance features
    df["pass_distance"] = np.sqrt(
        (df["end_x"] - df["start_x"])**2 + 
        (df["end_y"] - df["start_y"])**2
    )
    
    # Angle to goal (from pass end location)
    # Goal at x=120, y=40 (center)
    df["end_angle_to_goal"] = np.abs(np.arctan2(
        40 - df["end_y"], 
        120 - df["end_x"]
    )) * 180 / np.pi
    
    # Forward/backward direction
    df["is_forward"] = (df["end_x"] > df["start_x"]).astype(int)
    df["is_backward"] = (df["end_x"] < df["start_x"] - 5).astype(int)
    
    # Lateral movement
    df["lateral_distance"] = np.abs(df["end_y"] - df["start_y"])
    
    # Danger zone entry
    df["enters_box"] = (
        (df["end_x"] >= 102) & 
        (df["end_y"] >= 18) & 
        (df["end_y"] <= 62)
    ).astype(int)
    
    # Half-space entry (channels beside the box)
    df["enters_half_space"] = (
        (df["end_x"] >= 90) & 
        ((df["end_y"] <= 30) | (df["end_y"] >= 50))
    ).astype(int)
    
    # Zone 14 entry (central danger zone)
    df["enters_zone14"] = (
        (df["end_x"] >= 90) & (df["end_x"] <= 102) &
        (df["end_y"] >= 30) & (df["end_y"] <= 50)
    ).astype(int)
    
    logger.info("Added pass difficulty features")
    return df


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pressure-related features.
    
    Args:
        df: Passes DataFrame
        
    Returns:
        DataFrame with pressure features added
    """
    df = df.copy()
    
    # Binary pressure
    df["under_pressure_binary"] = df["under_pressure"].fillna(False).astype(int)
    
    # Pressure in different zones
    df["pressure_in_own_third"] = (
        df["under_pressure_binary"] & (df["start_x"] < 40)
    ).astype(int)
    
    df["pressure_in_final_third"] = (
        df["under_pressure_binary"] & (df["start_x"] >= 80)
    ).astype(int)
    
    logger.info("Added pressure features")
    return df


# =============================================================================
# SEQUENCE-LEVEL FEATURES (for shot creation model)
# =============================================================================

def add_sequence_buildup_features(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Add sequence buildup aggregate features.
    
    Args:
        df: Sequences DataFrame
        k: Maximum passes in sequence
        
    Returns:
        DataFrame with buildup features added
    """
    df = df.copy()
    
    # Total xT accumulated in sequence
    xt_cols = [f"pass{i}_xt_delta" for i in range(1, k+1)]
    available_xt = [c for c in xt_cols if c in df.columns]
    df["sequence_total_xt"] = df[available_xt].sum(axis=1, skipna=True)
    
    # Average xT per pass
    df["sequence_avg_xt"] = df["sequence_total_xt"] / df["num_passes_in_sequence"]
    
    # Max single pass xT
    df["sequence_max_xt"] = df[available_xt].max(axis=1, skipna=True)
    
    # Total forward progress
    progress_cols = []
    for i in range(1, k+1):
        start_col = f"pass{i}_start_x"
        end_col = f"pass{i}_end_x"
        if start_col in df.columns and end_col in df.columns:
            col_name = f"pass{i}_progress"
            df[col_name] = df[end_col] - df[start_col]
            progress_cols.append(col_name)
    
    if progress_cols:
        df["sequence_total_progress"] = df[progress_cols].sum(axis=1, skipna=True)
        df["sequence_avg_progress"] = df["sequence_total_progress"] / df["num_passes_in_sequence"]
    
    # Directness: ratio of net progress to total distance traveled
    # (High = direct attack, Low = possession buildup)
    # Use pass1_end_x as shot location (key pass ends where shot starts)
    shot_x_col = "shot_x" if "shot_x" in df.columns else "pass1_end_x"
    first_start_cols = [f"pass{i}_start_x" for i in range(1, k+1) if f"pass{i}_start_x" in df.columns]
    
    if shot_x_col in df.columns and first_start_cols:
        # Find the earliest pass start (furthest back in chain)
        first_pass_start = df[first_start_cols].bfill(axis=1).iloc[:, 0]
        net_progress = df[shot_x_col] - first_pass_start
        total_progress = df["sequence_total_progress"].clip(lower=1) if "sequence_total_progress" in df.columns else 1
        df["sequence_directness"] = net_progress / total_progress
        df["sequence_directness"] = df["sequence_directness"].clip(0, 2)
    
    # Count of progressive passes
    prog_flags = [f"pass{i}_is_progressive" for i in range(1, k+1)]
    available_prog = [c for c in prog_flags if c in df.columns]
    if available_prog:
        df["sequence_progressive_count"] = df[available_prog].sum(axis=1, skipna=True)
    
    # Count of passes into box
    box_flags = [f"pass{i}_is_into_box" for i in range(1, k+1)]
    available_box = [c for c in box_flags if c in df.columns]
    if available_box:
        df["sequence_into_box_count"] = df[available_box].sum(axis=1, skipna=True)
    
    # Count of crosses/through balls
    cross_flags = [f"pass{i}_is_cross" for i in range(1, k+1)]
    available_cross = [c for c in cross_flags if c in df.columns]
    if available_cross:
        df["sequence_cross_count"] = df[available_cross].sum(axis=1, skipna=True)
    
    through_flags = [f"pass{i}_is_through_ball" for i in range(1, k+1)]
    available_through = [c for c in through_flags if c in df.columns]
    if available_through:
        df["sequence_through_ball_count"] = df[available_through].sum(axis=1, skipna=True)
    
    logger.info("Added sequence buildup features")
    return df


def add_receive_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shot receive zone features.
    
    Args:
        df: Sequences DataFrame with shot location
        
    Returns:
        DataFrame with receive zone features added
    """
    df = df.copy()
    
    # Determine shot location columns
    # For sequences: pass1 is the key pass, its end location is where shot is taken
    if "shot_x" in df.columns:
        shot_x = df["shot_x"]
        shot_y = df["shot_y"]
    elif "pass1_end_x" in df.columns:
        shot_x = df["pass1_end_x"]
        shot_y = df["pass1_end_y"]
    else:
        logger.warning("No shot location columns found, skipping receive zone features")
        return df
    
    # Shot location zones
    df["shot_in_box"] = (
        (shot_x >= 102) & 
        (shot_y >= 18) & 
        (shot_y <= 62)
    ).astype(int)
    
    df["shot_in_six_yard"] = (
        (shot_x >= 114) & 
        (shot_y >= 30) & 
        (shot_y <= 50)
    ).astype(int)
    
    # Distance from goal
    df["shot_distance_to_goal"] = np.sqrt(
        (120 - shot_x)**2 + 
        (40 - shot_y)**2
    )
    
    # Angle to goal
    df["shot_angle_to_goal"] = np.abs(np.arctan2(
        40 - shot_y, 
        120 - shot_x
    )) * 180 / np.pi
    
    # Central vs wide shot
    df["shot_central"] = (
        (shot_y >= 25) & (shot_y <= 55)
    ).astype(int)
    
    # Store receive location for reference
    df["receive_x"] = shot_x
    df["receive_y"] = shot_y
    
    logger.info("Added receive zone features")
    return df


# =============================================================================
# MATCH STATE FEATURES
# =============================================================================

def add_match_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add match state context features.
    
    Args:
        df: DataFrame with minute column
        
    Returns:
        DataFrame with match state features added
    """
    df = df.copy()
    
    # Minute column detection
    minute_col = None
    for col in ["shot_minute", "pass1_minute", "minute"]:
        if col in df.columns:
            minute_col = col
            break
    
    if minute_col is None:
        logger.warning("No minute column found")
        return df
    
    # Game phase buckets
    df["minute_bucket"] = pd.cut(
        df[minute_col],
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
    )
    
    # Early/mid/late game
    df["game_phase"] = pd.cut(
        df[minute_col],
        bins=[0, 30, 60, 120],
        labels=["early", "mid", "late"]
    )
    
    # Urgency factor (higher in late game)
    df["urgency"] = (df[minute_col] / 90).clip(0, 1.5)
    
    logger.info("Added match state features")
    return df


# =============================================================================
# OPPONENT FEATURES
# =============================================================================

def compute_opponent_profiles(
    sequences_df: pd.DataFrame,
    team_clusters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute opponent defensive profiles from historical data.
    
    Args:
        sequences_df: Sequences with shot outcomes
        team_clusters_df: Team cluster assignments
        
    Returns:
        DataFrame with opponent defensive metrics
    """
    # Group by opponent team to get defensive stats
    # For now, we use team clusters as proxy for opponent strength
    
    # Merge team defensive cluster info
    # This is a simplified version - full implementation would
    # compute actual defensive stats per opponent
    
    logger.info("Opponent profiles derived from team clusters")
    return team_clusters_df


def add_opponent_features(
    df: pd.DataFrame,
    opponent_profiles: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add opponent-related features.
    
    Uses team clusters as proxy for opponent defensive quality.
    
    Args:
        df: Sequences DataFrame
        opponent_profiles: Optional opponent profiles
        
    Returns:
        DataFrame with opponent features added
    """
    df = df.copy()
    
    # Use team cluster as opponent quality proxy
    # Assumption: playing style correlates with defensive strength
    
    if "team_cluster_label" in df.columns:
        # Encode team style as numeric
        style_map = {
            "Possession Build-up": 0.8,   # Good defense (patient)
            "Balanced": 1.0,              # Average
            "Direct Central": 1.1,        # Slightly weaker
            "Direct Wide": 1.2,           # More open
        }
        df["opponent_quality_proxy"] = df["team_cluster_label"].map(style_map).fillna(1.0)
    
    # If we have actual opponent team ID, we could compute:
    # - Goals conceded per game
    # - xG against per game
    # - Key passes allowed
    # - Shots allowed in box
    
    logger.info("Added opponent features (using team cluster proxy)")
    return df


# =============================================================================
# MAIN FEATURE BUILDER
# =============================================================================

def build_cxa_features(
    passes_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    k: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build all cxA features for passes and sequences.
    
    Args:
        passes_df: Raw passes data
        sequences_df: Raw sequences data
        k: Maximum sequence length
        
    Returns:
        Tuple of (enriched_passes, enriched_sequences)
    """
    logger.info("Building cxA features...")
    
    # Pass-level features
    passes = passes_df.copy()
    passes = add_pass_difficulty_features(passes)
    passes = add_pressure_features(passes)
    
    # Sequence-level features
    sequences = sequences_df.copy()
    sequences = add_sequence_buildup_features(sequences, k)
    sequences = add_receive_zone_features(sequences)
    sequences = add_match_state_features(sequences)
    sequences = add_opponent_features(sequences)
    
    # Summary
    new_pass_cols = len(passes.columns) - len(passes_df.columns)
    new_seq_cols = len(sequences.columns) - len(sequences_df.columns)
    
    logger.info(f"Added {new_pass_cols} pass-level features")
    logger.info(f"Added {new_seq_cols} sequence-level features")
    
    return passes, sequences


def get_completion_model_features() -> list:
    """Get feature columns for pass completion model."""
    return [
        # Pass characteristics
        "pass_distance", "lateral_distance", "pass_length", "pass_angle",
        "is_forward", "is_backward",
        # Difficulty
        "enters_box", "enters_half_space", "enters_zone14",
        "end_angle_to_goal",
        # Type
        "is_cross", "is_through_ball", "is_progressive",
        # Context
        "under_pressure_binary", "pressure_in_final_third",
        # Spatial
        "start_x", "start_y", "xt_delta",
    ]


def get_shot_creation_model_features() -> list:
    """Get feature columns for shot-within-k model."""
    return [
        # Buildup quality
        "sequence_total_xt", "sequence_avg_xt", "sequence_max_xt",
        "sequence_total_progress", "sequence_directness",
        "sequence_progressive_count", "sequence_into_box_count",
        # Pass types
        "sequence_cross_count", "sequence_through_ball_count",
        # Sequence structure
        "num_passes_in_sequence",
        # Key pass features
        "pass1_xt_delta", "pass1_is_into_box", "pass1_is_cross",
        # Context
        "urgency",
        # Opponent
        "opponent_quality_proxy",
        # Clusters
        "team_cluster", "pass1_cluster",
    ]


def get_shot_quality_model_features() -> list:
    """Get feature columns for conditional shot quality model.
    
    Note: shot location is derived from pass1_end_x/y (key pass ends where shot starts).
    We use receive_x/y as the canonical shot location columns created in add_receive_zone_features.
    """
    return [
        # Shot location (receive_x/y are copies of pass1_end_x/y)
        "receive_x", "receive_y", "shot_in_box", "shot_in_six_yard",
        "shot_distance_to_goal", "shot_angle_to_goal", "shot_central",
        # Delivery
        "pass1_is_cross", "pass1_is_through_ball",
        "pass1_end_x", "pass1_end_y",
        # Buildup
        "sequence_total_xt", "sequence_directness",
        # Shot type
        "shot_body_part", "shot_technique", "shot_first_time",
        # Context
        "urgency",
        # Opponent
        "opponent_quality_proxy",
    ]
