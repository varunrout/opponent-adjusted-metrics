"""
Beyond-xT Contribution Features.

Features that capture value creation beyond standard Expected Threat (xT):
- Line-breaking passes (bypassing defensive lines)
- Defenders bypassed (direct 1v1 beating or positional)
- Pressure relief (escaping press)
- Space creation (opening passing lanes)

These features help credit actions that enable chances
even if they don't directly increase xT.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Pitch zones for line-breaking detection
# Y-coordinates for defensive lines (estimated)
DEFENSIVE_LINE_ZONES = {
    "defensive_third": (0, 40),
    "midfield": (40, 80),
    "attacking_third": (80, 120),
}

# Box coordinates
BOX_X_MIN = 102
BOX_Y_MIN = 18
BOX_Y_MAX = 62


def add_contribution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add beyond-xT contribution features to action data.
    
    Args:
        df: Action-level DataFrame with start/end coordinates
        
    Returns:
        DataFrame with contribution features added
    """
    df = df.copy()
    
    # Line-breaking features
    df = add_line_breaking_features(df)
    
    # Space/zone features
    df = add_space_creation_features(df)
    
    # Pressure relief features
    df = add_pressure_relief_features(df)
    
    # Final third entry features
    df = add_final_third_features(df)
    
    # Composite contribution score
    df = compute_contribution_score(df)
    
    logger.info("Added contribution features")
    return df


def add_line_breaking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect line-breaking actions.
    
    A line-breaking action moves the ball past a defensive line:
    - From defensive third past midfield
    - From midfield into attacking third
    - From outside box into box
    """
    df = df.copy()
    
    # Zone classification
    df["start_zone"] = _classify_zone(df["start_x"])
    df["end_zone"] = _classify_zone(df["end_x"])
    
    # Line-breaking detection
    zone_order = {"defensive_third": 0, "midfield": 1, "attacking_third": 2}
    df["start_zone_idx"] = df["start_zone"].map(zone_order).fillna(0)
    df["end_zone_idx"] = df["end_zone"].map(zone_order).fillna(0)
    
    # Breaks line if moves forward at least one zone
    df["breaks_line"] = (df["end_zone_idx"] > df["start_zone_idx"]).astype(int)
    
    # Specifically into attacking third (more valuable)
    df["breaks_into_attack"] = (
        (df["start_zone_idx"] < 2) & (df["end_zone_idx"] == 2)
    ).astype(int)
    
    # Into box
    df["start_in_box"] = _in_box(df["start_x"], df["start_y"])
    df["end_in_box"] = _in_box(df["end_x"], df["end_y"])
    df["breaks_into_box"] = (
        (df["start_in_box"] == 0) & (df["end_in_box"] == 1)
    ).astype(int)
    
    # Clean up intermediate columns
    df = df.drop(columns=["start_zone_idx", "end_zone_idx"], errors="ignore")
    
    return df


def _classify_zone(x_series: pd.Series) -> pd.Series:
    """Classify x-coordinate into pitch zones."""
    conditions = [
        x_series < 40,
        (x_series >= 40) & (x_series < 80),
        x_series >= 80,
    ]
    choices = ["defensive_third", "midfield", "attacking_third"]
    return pd.Series(
        np.select(conditions, choices, default="midfield"),
        index=x_series.index
    )


def _in_box(x_series: pd.Series, y_series: pd.Series) -> pd.Series:
    """Check if coordinates are inside the penalty box."""
    return (
        (x_series >= BOX_X_MIN) & 
        (y_series >= BOX_Y_MIN) & 
        (y_series <= BOX_Y_MAX)
    ).astype(int)


def add_space_creation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to space creation and lane opening.
    
    Space is created by:
    - Lateral movement (switching play)
    - Progressive carries
    - Passes into gaps between defenders
    """
    df = df.copy()
    
    # Lateral movement (width gain)
    df["lateral_distance"] = np.abs(df["end_y"] - df["start_y"])
    
    # Switch play (significant width change from wing to wing)
    df["is_switch"] = (
        (df["lateral_distance"] > 30) &
        (df["action_type"] == "Pass")
    ).astype(int)
    
    # Half-space entry (valuable attacking zones)
    # Half-spaces: y in [18, 30] or [50, 62] when x > 80
    df["enters_half_space"] = (
        (df["end_x"] >= 80) &
        (((df["end_y"] >= 18) & (df["end_y"] <= 30)) |
         ((df["end_y"] >= 50) & (df["end_y"] <= 62)))
    ).astype(int)
    
    # Central channel entry (zone 14)
    df["enters_zone14"] = (
        (df["end_x"] >= 80) & (df["end_x"] <= 102) &
        (df["end_y"] >= 30) & (df["end_y"] <= 50)
    ).astype(int)
    
    # Progressive action (significant x gain)
    x_progress = df["end_x"] - df["start_x"]
    df["is_progressive"] = (x_progress >= 10).astype(int)
    
    # Highly progressive (major territory gain)
    df["is_highly_progressive"] = (x_progress >= 20).astype(int)
    
    return df


def add_pressure_relief_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features related to escaping defensive pressure.
    
    Pressure relief is valuable because:
    - Allows team to reset and attack
    - Creates numerical advantages
    - Opens space for subsequent actions
    """
    df = df.copy()
    
    # Basic pressure escape
    if "under_pressure" not in df.columns:
        df["under_pressure"] = False
    
    df["under_pressure_binary"] = df["under_pressure"].fillna(False).astype(int)
    
    # Successful action under pressure
    df["successful_under_pressure"] = (
        df["under_pressure_binary"] == 1
    ).astype(int)
    
    # Pressure in dangerous zones (more valuable to relieve)
    df["pressure_in_own_third"] = (
        (df["under_pressure_binary"] == 1) & 
        (df["start_x"] < 40)
    ).astype(int)
    
    # Escape from pressure to space (pressure start, no pressure at end implied by progression)
    df["pressure_escape"] = (
        (df["under_pressure_binary"] == 1) & 
        (df["is_progressive"] == 1)
    ).astype(int)
    
    return df


def add_final_third_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features specific to final third actions.
    
    Final third actions are weighted more heavily in xA
    because they're closer to goal-scoring.
    """
    df = df.copy()
    
    # In final third
    df["start_in_final_third"] = (df["start_x"] >= 80).astype(int)
    df["end_in_final_third"] = (df["end_x"] >= 80).astype(int)
    
    # Action entirely in final third
    df["in_final_third"] = (
        (df["start_in_final_third"] == 1) & 
        (df["end_in_final_third"] == 1)
    ).astype(int)
    
    # Distance to goal at end
    df["end_distance_to_goal"] = np.sqrt(
        (120 - df["end_x"])**2 + (40 - df["end_y"])**2
    )
    
    # Angle to goal at end
    df["end_angle_to_goal"] = np.abs(
        np.arctan2(40 - df["end_y"], 120 - df["end_x"])
    ) * 180 / np.pi
    
    # Central angle (0 = direct at goal)
    # More central = better angle
    df["centrality_at_end"] = 90 - df["end_angle_to_goal"]
    
    return df


def compute_contribution_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite contribution score.
    
    This score captures value created beyond xT:
    - Line breaking: +0.3
    - Into box: +0.4
    - Progressive: +0.2
    - Half-space entry: +0.15
    - Zone 14 entry: +0.15
    - Pressure escape: +0.1
    - Switch: +0.1
    """
    df = df.copy()
    
    score = np.zeros(len(df))
    
    # Line breaking
    score += df["breaks_line"].fillna(0) * 0.3
    score += df["breaks_into_box"].fillna(0) * 0.4
    
    # Space creation
    score += df["is_progressive"].fillna(0) * 0.2
    score += df["is_highly_progressive"].fillna(0) * 0.1  # Additional bonus
    score += df["enters_half_space"].fillna(0) * 0.15
    score += df["enters_zone14"].fillna(0) * 0.15
    score += df["is_switch"].fillna(0) * 0.1
    
    # Pressure
    score += df["pressure_escape"].fillna(0) * 0.1
    score += df["successful_under_pressure"].fillna(0) * 0.05
    
    df["contribution_score"] = score
    
    # Normalize to 0-1 range
    max_possible = 0.3 + 0.4 + 0.2 + 0.1 + 0.15 + 0.15 + 0.1 + 0.1 + 0.05
    df["contribution_score_normalized"] = df["contribution_score"] / max_possible
    
    return df


def add_defender_bypass_features(
    df: pd.DataFrame,
    freeze_frame_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add features for defenders bypassed.
    
    Requires freeze-frame data (360 data) to properly count
    defenders between start and end locations.
    
    If freeze-frame not available, uses proxy features.
    
    Args:
        df: Action DataFrame
        freeze_frame_data: Optional 360 freeze-frame data
        
    Returns:
        DataFrame with bypass features
    """
    df = df.copy()
    
    if freeze_frame_data is not None:
        # Full implementation with freeze-frame
        df = _compute_bypass_from_freeze_frame(df, freeze_frame_data)
    else:
        # Proxy: estimate based on action characteristics
        df = _estimate_bypass_proxy(df)
    
    return df


def _estimate_bypass_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate defenders bypassed without freeze-frame data."""
    df = df.copy()
    
    # Proxy based on:
    # - Distance covered (longer = more defenders likely bypassed)
    # - Line-breaking (implies passing defensive line)
    # - Final third entry (defense usually organized there)
    
    bypass_estimate = np.zeros(len(df))
    
    # Distance-based estimate
    distance = np.sqrt(
        (df["end_x"] - df["start_x"])**2 + 
        (df["end_y"] - df["start_y"])**2
    )
    bypass_estimate += np.minimum(distance / 20, 2)  # Max 2 from distance
    
    # Line-breaking bonus
    bypass_estimate += df["breaks_line"].fillna(0) * 1.0
    bypass_estimate += df["breaks_into_box"].fillna(0) * 1.5
    
    # Through balls typically bypass more
    if "is_through_ball" in df.columns:
        bypass_estimate += df["is_through_ball"].fillna(0) * 1.0
    
    # Dribbles explicitly beat defenders
    if "action_type" in df.columns:
        bypass_estimate += (df["action_type"] == "Dribble").astype(int) * 1.0
    
    df["estimated_defenders_bypassed"] = bypass_estimate
    df["defenders_bypassed_proxy"] = np.minimum(bypass_estimate, 5)  # Cap at 5
    
    return df


def _compute_bypass_from_freeze_frame(
    df: pd.DataFrame,
    freeze_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compute actual defenders bypassed using freeze-frame data."""
    # This would require 360 data with opponent positions
    # Implementation depends on data format
    # For now, return placeholder
    df["defenders_bypassed"] = 0
    df["defenders_bypassed_computed"] = True
    return df


# =============================================================================
# CROSS-SPECIFIC FEATURES
# =============================================================================

def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features specific to crosses.
    
    Crosses have different value profiles:
    - Near post vs far post
    - Low vs high
    - Whipped vs floated
    """
    df = df.copy()
    
    if "is_cross" not in df.columns:
        df["is_cross"] = False
    
    crosses = df["is_cross"].fillna(False).astype(bool)
    
    # Cross zone (where it lands)
    # Near post: y < 40, Far post: y > 40
    df["cross_to_near_post"] = (
        crosses & 
        (df["end_y"] < 40) & 
        (df["end_in_box"] == 1)
    ).astype(int)
    
    df["cross_to_far_post"] = (
        crosses & 
        (df["end_y"] > 40) & 
        (df["end_in_box"] == 1)
    ).astype(int)
    
    # Cross from deep vs cutback
    df["cross_from_byline"] = (
        crosses & 
        (df["start_x"] >= 110)
    ).astype(int)
    
    df["cross_cutback"] = (
        crosses & 
        (df["start_x"] >= 102) & 
        (df["end_x"] < df["start_x"])  # Ball comes back
    ).astype(int)
    
    # Cross height (if available)
    if "pass_height" in df.columns:
        df["cross_is_low"] = (
            crosses & 
            (df["pass_height"] == "Low")
        ).astype(int)
        
        df["cross_is_high"] = (
            crosses & 
            (df["pass_height"] == "High")
        ).astype(int)
    
    return df


# =============================================================================
# THROUGH-BALL SPECIFIC FEATURES  
# =============================================================================

def add_through_ball_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features specific to through balls.
    
    Through balls break defensive lines with timing.
    """
    df = df.copy()
    
    if "is_through_ball" not in df.columns:
        df["is_through_ball"] = False
    
    through_balls = df["is_through_ball"].fillna(False).astype(bool)
    
    # Through ball into channel (wide areas behind defense)
    df["through_ball_into_channel"] = (
        through_balls &
        (df["end_x"] >= 90) &
        ((df["end_y"] < 25) | (df["end_y"] > 55))
    ).astype(int)
    
    # Through ball centrally (most dangerous)
    df["through_ball_central"] = (
        through_balls &
        (df["end_x"] >= 90) &
        (df["end_y"] >= 25) &
        (df["end_y"] <= 55)
    ).astype(int)
    
    # Through ball distance (longer = harder to execute)
    through_ball_distance = np.where(
        through_balls,
        np.sqrt(
            (df["end_x"] - df["start_x"])**2 + 
            (df["end_y"] - df["start_y"])**2
        ),
        0
    )
    df["through_ball_distance"] = through_ball_distance
    
    return df
