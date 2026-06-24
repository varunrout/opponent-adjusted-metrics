"""Ball Progression Data Extraction for CxT Analysis.

Extracts passes, carries, and dribbles with their start/end locations
and computes xT deltas for each action.

This module provides the foundational dataset for CxT modeling.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, literal
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import (
    Event,
    PassEvent,
    CarryEvent,
    DribbleEvent,
    Match,
)
from opponent_adjusted.features.cxt.xt_model import (
    XT_GRID,
    XT_GRID_X,
    XT_GRID_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
)

logger = logging.getLogger(__name__)


# Macro-zone definitions for CxT
# Based on pitch thirds (DEF/MID/ATT) and width (CENTRAL/WIDE)
MACRO_ZONES = {
    # Zone ID: (x_min, x_max, y_min, y_max, name)
    1: (0, 40, 18, 62, "DEF_CENTRAL"),
    2: (0, 40, 0, 18, "DEF_WIDE_LEFT"),
    3: (0, 40, 62, 80, "DEF_WIDE_RIGHT"),
    4: (40, 80, 18, 62, "MID_CENTRAL"),
    5: (40, 80, 0, 18, "MID_WIDE_LEFT"),
    6: (40, 80, 62, 80, "MID_WIDE_RIGHT"),
    7: (80, 120, 18, 62, "ATT_CENTRAL"),
    8: (80, 120, 0, 18, "ATT_WIDE_LEFT"),
    9: (80, 120, 62, 80, "ATT_WIDE_RIGHT"),
}


def get_macro_zone(x: float, y: float) -> int:
    """Assign a macro-zone (1-9) based on pitch location.

    Args:
        x: X coordinate (0-120)
        y: Y coordinate (0-80)

    Returns:
        Macro-zone ID (1-9)
    """
    x = np.clip(x, 0, 120)
    y = np.clip(y, 0, 80)

    for zone_id, (x_min, x_max, y_min, y_max, _) in MACRO_ZONES.items():
        if x_min <= x < x_max and y_min <= y < y_max:
            return zone_id

    # Edge case: exactly at boundary
    return 4  # Default to MID_CENTRAL


def get_macro_zone_name(zone_id: int) -> str:
    """Get the name for a macro-zone ID."""
    return MACRO_ZONES.get(zone_id, (0, 0, 0, 0, "UNKNOWN"))[4]


def _extract_passes(
    session: Session,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Extract all passes with start/end locations.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter

    Returns:
        DataFrame with pass progression data
    """
    logger.info("Extracting passes...")

    stmt = (
        select(
            Event.id.label("event_id"),
            Event.raw_event_id,
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            PassEvent.end_x,
            PassEvent.end_y,
            PassEvent.outcome.label("action_outcome"),
            Event.under_pressure,
            literal("pass").label("action_type"),
        )
        .select_from(Event)
        .join(PassEvent, PassEvent.event_id == Event.id)
        .where(Event.type == "Pass")
    )

    # Apply filters
    if match_ids:
        stmt = stmt.where(Event.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == Event.match_id)
        stmt = stmt.where(Match.competition_id == competition_id)

    result = session.execute(stmt)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    logger.info(f"  Extracted {len(df):,} passes")
    return df


def _extract_carries(
    session: Session,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Extract all carries with start/end locations.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter

    Returns:
        DataFrame with carry progression data
    """
    logger.info("Extracting carries...")

    stmt = (
        select(
            Event.id.label("event_id"),
            Event.raw_event_id,
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            CarryEvent.start_x,
            CarryEvent.start_y,
            CarryEvent.end_x,
            CarryEvent.end_y,
            literal(None).label("action_outcome"),  # Carries don't have outcome
            Event.under_pressure,
            literal("carry").label("action_type"),
        )
        .select_from(Event)
        .join(CarryEvent, CarryEvent.event_id == Event.id)
        .where(Event.type == "Carry")
    )

    # Apply filters
    if match_ids:
        stmt = stmt.where(Event.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == Event.match_id)
        stmt = stmt.where(Match.competition_id == competition_id)

    result = session.execute(stmt)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    logger.info(f"  Extracted {len(df):,} carries")
    return df


def _extract_dribbles(
    session: Session,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Extract successful dribbles.

    Note: Dribbles don't have end locations in StatsBomb data,
    so we use the next event's location as the end location.
    For now, we only include dribbles as markers and skip xT calculation.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter

    Returns:
        DataFrame with dribble data
    """
    logger.info("Extracting dribbles...")

    stmt = (
        select(
            Event.id.label("event_id"),
            Event.raw_event_id,
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            # Dribbles don't have end_location - use start as placeholder
            Event.location_x.label("end_x"),
            Event.location_y.label("end_y"),
            DribbleEvent.outcome.label("action_outcome"),
            Event.under_pressure,
            literal("dribble").label("action_type"),
        )
        .select_from(Event)
        .join(DribbleEvent, DribbleEvent.event_id == Event.id)
        .where(Event.type == "Dribble")
    )

    # Only successful dribbles
    stmt = stmt.where(DribbleEvent.outcome.in_(["Complete", "Won", None]))

    # Apply filters
    if match_ids:
        stmt = stmt.where(Event.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == Event.match_id)
        stmt = stmt.where(Match.competition_id == competition_id)

    result = session.execute(stmt)
    df = pd.DataFrame(result.fetchall(), columns=result.keys())

    logger.info(f"  Extracted {len(df):,} successful dribbles")
    return df


def _add_match_context(df: pd.DataFrame, session: Session) -> pd.DataFrame:
    """Add match-level context (opponent, competition).

    Args:
        df: DataFrame with match_id and team_id
        session: SQLAlchemy session

    Returns:
        DataFrame with match context columns added
    """
    logger.info("Adding match context...")

    # Get match data
    match_ids = df["match_id"].unique().tolist()

    stmt = select(
        Match.id.label("match_id"),
        Match.competition_id,
        Match.home_team_id,
        Match.away_team_id,
        Match.match_date,
    ).where(Match.id.in_(match_ids))

    result = session.execute(stmt)
    matches_df = pd.DataFrame(result.fetchall(), columns=result.keys())

    # Merge match data
    df = df.merge(matches_df, on="match_id", how="left")

    # Determine opponent
    df["opponent_id"] = np.where(
        df["team_id"] == df["home_team_id"], df["away_team_id"], df["home_team_id"]
    )

    # Determine home/away
    df["is_home"] = df["team_id"] == df["home_team_id"]

    # Score tracking placeholder - will be computed from goals in EDA phase
    # For now, use minute-based proxy (late game = more pressure)
    df["minute_normalized"] = df["minute"] / 90.0
    df["is_late_game"] = df["minute"] >= 75

    # Clean up temporary columns
    df = df.drop(columns=["home_team_id", "away_team_id"])

    logger.info(f"  Added match context for {len(matches_df)} matches")
    return df


def _compute_xt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute xT values and deltas for all progressions.

    Args:
        df: DataFrame with start_x, start_y, end_x, end_y

    Returns:
        DataFrame with xT features added
    """
    logger.info("Computing xT features...")

    # Fill NaN coordinates with 0
    start_x = df["start_x"].fillna(0).values
    start_y = df["start_y"].fillna(0).values
    end_x = df["end_x"].fillna(0).values
    end_y = df["end_y"].fillna(0).values

    # Calculate zone indices
    df["start_zone_x"] = np.clip((start_x / PITCH_LENGTH * XT_GRID_X).astype(int), 0, XT_GRID_X - 1)
    df["start_zone_y"] = np.clip((start_y / PITCH_WIDTH * XT_GRID_Y).astype(int), 0, XT_GRID_Y - 1)
    df["end_zone_x"] = np.clip((end_x / PITCH_LENGTH * XT_GRID_X).astype(int), 0, XT_GRID_X - 1)
    df["end_zone_y"] = np.clip((end_y / PITCH_WIDTH * XT_GRID_Y).astype(int), 0, XT_GRID_Y - 1)

    # Lookup xT values
    df["start_xt"] = XT_GRID[df["start_zone_y"].values, df["start_zone_x"].values]
    df["end_xt"] = XT_GRID[df["end_zone_y"].values, df["end_zone_x"].values]
    df["xt_delta"] = df["end_xt"] - df["start_xt"]

    # Calculate macro-zones
    df["macro_zone_start"] = df.apply(
        lambda r: get_macro_zone(
            r["start_x"] if pd.notna(r["start_x"]) else 60,
            r["start_y"] if pd.notna(r["start_y"]) else 40,
        ),
        axis=1,
    )
    df["macro_zone_end"] = df.apply(
        lambda r: get_macro_zone(
            r["end_x"] if pd.notna(r["end_x"]) else 60, r["end_y"] if pd.notna(r["end_y"]) else 40
        ),
        axis=1,
    )

    # Derive progression flags
    df["is_progressive"] = df["xt_delta"] > 0.01
    df["is_into_final_third"] = (df["start_x"] < 80) & (df["end_x"] >= 80)
    df["is_into_penalty_area"] = (df["end_x"] >= 102) & (df["end_y"] >= 18) & (df["end_y"] <= 62)

    # Handle incomplete passes - they lose possession
    if "action_outcome" in df.columns:
        incomplete_mask = df["action_outcome"].isin(["Incomplete", "Out", "Pass Offside"])
        df.loc[incomplete_mask, "xt_delta"] = -df.loc[incomplete_mask, "start_xt"]

    logger.info(f"  Computed xT features for {len(df):,} actions")
    return df


def build_progressions_dataset(
    session: Session,
    *,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
    include_dribbles: bool = True,
) -> pd.DataFrame:
    """Build comprehensive ball progression dataset.

    Extracts passes, carries, and optionally dribbles, then enriches
    with xT values, macro-zones, and match context.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter
        include_dribbles: Whether to include dribbles (default True)

    Returns:
        DataFrame with all progression data
    """
    logger.info("=" * 60)
    logger.info("Building CxT Progressions Dataset")
    logger.info("=" * 60)

    # Extract each action type
    passes_df = _extract_passes(session, match_ids, competition_id)
    carries_df = _extract_carries(session, match_ids, competition_id)

    # Combine
    if include_dribbles:
        dribbles_df = _extract_dribbles(session, match_ids, competition_id)
        df = pd.concat([passes_df, carries_df, dribbles_df], ignore_index=True)
    else:
        df = pd.concat([passes_df, carries_df], ignore_index=True)

    logger.info(f"\nTotal actions extracted: {len(df):,}")
    logger.info(f"  - Passes: {len(passes_df):,}")
    logger.info(f"  - Carries: {len(carries_df):,}")
    if include_dribbles:
        logger.info(f"  - Dribbles: {len(dribbles_df):,}")

    # Add match context
    df = _add_match_context(df, session)

    # Compute xT features
    df = _compute_xt_features(df)

    # Sort by match and timestamp
    df = df.sort_values(["match_id", "period", "minute", "second", "timestamp"])
    df = df.reset_index(drop=True)

    logger.info(f"\nFinal dataset: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df
