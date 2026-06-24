"""Possession-level data aggregation for CxA analysis.

Groups passes by possession to create possession-level features:
- Possession duration and pass count
- Spatial progression (start/end zones, xT gained)
- Outcome (shot, turnover, etc.)
- Pass type breakdown
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_possession_dataset(
    passes_df: pd.DataFrame,
    shots_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build possession-level dataset from passes.

    Each row represents a single possession (ball sequence by one team).

    Args:
        passes_df: Pass-level data with xT features
        shots_df: Optional shot data to link outcomes

    Returns:
        DataFrame with one row per possession
    """
    logger.info("Building possession dataset...")

    df = passes_df.copy()

    # Ensure required columns exist
    required_cols = ["match_id", "team_id", "possession", "minute", "second"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by time within possession
    df = df.sort_values(["match_id", "possession", "minute", "second"])

    # Group by possession
    grouped = df.groupby(["match_id", "team_id", "possession"])

    logger.info(f"Aggregating {len(grouped)} possessions...")

    # Aggregate features
    possession_df = grouped.agg(
        # Counts
        num_passes=("pass_id", "count"),
        # Temporal
        start_minute=("minute", "first"),
        start_second=("second", "first"),
        end_minute=("minute", "last"),
        end_second=("second", "last"),
        # Spatial - start
        start_x=("start_x", "first"),
        start_y=("start_y", "first"),
        # Spatial - end (last pass destination)
        end_x=("end_x", "last"),
        end_y=("end_y", "last"),
        # xT features
        total_xt_gained=("xt_delta", "sum") if "xt_delta" in df.columns else ("pass_id", "count"),
        mean_xt_delta=("xt_delta", "mean") if "xt_delta" in df.columns else ("pass_id", "count"),
        max_xt_single_pass=(
            ("xt_delta", "max") if "xt_delta" in df.columns else ("pass_id", "count")
        ),
        start_xt=("start_xt", "first") if "start_xt" in df.columns else ("pass_id", "count"),
        end_xt=("end_xt", "last") if "end_xt" in df.columns else ("pass_id", "count"),
        # Pass types
        num_progressive=(
            ("is_progressive", "sum") if "is_progressive" in df.columns else ("pass_id", "count")
        ),
        num_into_box=(
            ("is_into_box", "sum") if "is_into_box" in df.columns else ("pass_id", "count")
        ),
        num_crosses=("is_cross", "sum") if "is_cross" in df.columns else ("pass_id", "count"),
        num_through_balls=(
            ("is_through_ball", "sum") if "is_through_ball" in df.columns else ("pass_id", "count")
        ),
        num_complete=(
            ("is_complete", "sum") if "is_complete" in df.columns else ("pass_id", "count")
        ),
        # Context
        play_pattern=("play_pattern", "first"),
        competition_id=("competition_id", "first"),
        # Players involved
        num_unique_passers=("player_id", "nunique"),
    ).reset_index()

    # Fix columns if xT wasn't available (they'd be counts instead)
    if "xt_delta" not in df.columns:
        possession_df["total_xt_gained"] = 0.0
        possession_df["mean_xt_delta"] = 0.0
        possession_df["max_xt_single_pass"] = 0.0
        possession_df["start_xt"] = 0.0
        possession_df["end_xt"] = 0.0

    # Derived features
    possession_df["duration_seconds"] = (
        possession_df["end_minute"] - possession_df["start_minute"]
    ) * 60 + (possession_df["end_second"] - possession_df["start_second"])

    possession_df["completion_rate"] = (
        possession_df["num_complete"] / possession_df["num_passes"]
    ).fillna(0)

    possession_df["progressive_rate"] = (
        possession_df["num_progressive"] / possession_df["num_passes"]
    ).fillna(0)

    # Spatial progression
    possession_df["x_progression"] = possession_df["end_x"] - possession_df["start_x"]
    possession_df["total_distance"] = np.sqrt(
        (possession_df["end_x"] - possession_df["start_x"]) ** 2
        + (possession_df["end_y"] - possession_df["start_y"]) ** 2
    )

    # Zone classification
    possession_df["start_zone"] = possession_df["start_x"].apply(_get_third)
    possession_df["end_zone"] = possession_df["end_x"].apply(_get_third)

    # Zone progression category
    possession_df["zone_progression"] = possession_df.apply(
        lambda r: _get_zone_progression(r["start_zone"], r["end_zone"]), axis=1
    )

    # Link to shots if provided
    if shots_df is not None and not shots_df.empty:
        possession_df = _link_shot_outcomes(possession_df, shots_df, passes_df)
    else:
        possession_df["ended_in_shot"] = False
        possession_df["shot_xg"] = None
        possession_df["shot_outcome"] = None
        possession_df["ended_in_goal"] = False

    logger.info(f"Built possession dataset: {len(possession_df):,} possessions")
    logger.info(f"  Mean passes per possession: {possession_df['num_passes'].mean():.1f}")
    logger.info(f"  Mean xT gained: {possession_df['total_xt_gained'].mean():.4f}")

    return possession_df


def _get_third(x: float) -> str:
    """Get pitch third from x coordinate."""
    if pd.isna(x):
        return "Unknown"
    if x < 40:
        return "Defensive"
    elif x < 80:
        return "Middle"
    else:
        return "Attacking"


def _get_zone_progression(start: str, end: str) -> str:
    """Categorize zone progression."""
    zone_order = {"Defensive": 0, "Middle": 1, "Attacking": 2, "Unknown": -1}

    start_idx = zone_order.get(start, -1)
    end_idx = zone_order.get(end, -1)

    if start_idx < 0 or end_idx < 0:
        return "Unknown"

    diff = end_idx - start_idx

    if diff > 0:
        return "Progressed"
    elif diff < 0:
        return "Regressed"
    else:
        return "Same Zone"


def _link_shot_outcomes(
    possession_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    passes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Link possessions to shot outcomes."""
    logger.info("Linking possession outcomes to shots...")

    # Find which possessions ended in shots
    # A shot belongs to a possession if it's in the same match/team/possession

    # Get possession info for each shot via the key pass
    # key_pass_id is a UUID string, so we match on statsbomb_event_id
    shots_with_possession = shots_df.merge(
        passes_df[["statsbomb_event_id", "match_id", "team_id", "possession"]].drop_duplicates(),
        left_on="key_pass_id",
        right_on="statsbomb_event_id",
        how="left",
        suffixes=("", "_pass"),
    )

    # Some shots may not have key passes, use shot's own match/team
    if "match_id_pass" in shots_with_possession.columns:
        shots_with_possession["match_id"] = shots_with_possession["match_id_pass"].fillna(
            shots_with_possession["match_id"]
        )

    # Aggregate by possession
    shot_outcomes = (
        shots_with_possession.groupby(["match_id", "team_id", "possession"])
        .agg(
            num_shots=("shot_id", "count"),
            shot_xg=("statsbomb_xg", "sum"),
            best_shot_xg=("statsbomb_xg", "max"),
            num_goals=("outcome", lambda x: (x == "Goal").sum()),
        )
        .reset_index()
    )

    # Merge with possessions
    possession_df = possession_df.merge(
        shot_outcomes,
        on=["match_id", "team_id", "possession"],
        how="left",
    )

    # Fill missing values
    possession_df["num_shots"] = possession_df["num_shots"].fillna(0).astype(int)
    possession_df["shot_xg"] = possession_df["shot_xg"].fillna(0)
    possession_df["best_shot_xg"] = possession_df["best_shot_xg"].fillna(0)
    possession_df["num_goals"] = possession_df["num_goals"].fillna(0).astype(int)

    possession_df["ended_in_shot"] = possession_df["num_shots"] > 0
    possession_df["ended_in_goal"] = possession_df["num_goals"] > 0

    shots_count = possession_df["ended_in_shot"].sum()
    goals_count = possession_df["ended_in_goal"].sum()
    logger.info(f"  Possessions ending in shot: {shots_count:,}")
    logger.info(f"  Possessions ending in goal: {goals_count:,}")

    return possession_df


def save_possession_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "possessions.csv",
) -> None:
    """Save possession dataset to CSV."""
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved possession dataset to {output_path}")
