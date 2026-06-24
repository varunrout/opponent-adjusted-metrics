"""Pass data extraction for CxA analysis.

Builds a comprehensive pass-level dataset by joining:
- events (core event data)
- passes (pass-specific details)
- raw_events (for play_pattern and other raw fields)
- players (passer and recipient names)
- teams (team names)
- matches (match context)

This module provides the foundational dataset for xT and xA calculations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import (
    Event,
    PassEvent,
    RawEvent,
    Player,
    Team,
    Match,
)

logger = logging.getLogger(__name__)


def build_pass_dataset(
    session: Session,
    *,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
    include_incomplete: bool = True,
) -> pd.DataFrame:
    """Build comprehensive pass dataset from database.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter
        include_incomplete: Whether to include incomplete passes

    Returns:
        DataFrame with all pass data for analysis
    """
    logger.info("Building pass dataset...")

    # Build base query joining events and passes
    stmt = (
        select(
            # Core identity
            Event.id.label("pass_id"),
            Event.raw_event_id,
            Event.match_id,
            Event.team_id,
            Event.player_id,
            PassEvent.recipient_player_id.label("recipient_id"),
            # Temporal
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            # Spatial - origin
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            # Spatial - destination
            PassEvent.end_x,
            PassEvent.end_y,
            PassEvent.length.label("pass_length"),
            PassEvent.angle.label("pass_angle"),
            # Pass type attributes
            PassEvent.pass_height,
            PassEvent.pass_type,
            PassEvent.body_part,
            PassEvent.is_cross,
            PassEvent.is_through_ball,
            PassEvent.outcome.label("pass_outcome"),
            # Context
            Event.under_pressure,
            Event.outcome.label("event_outcome"),
        )
        .select_from(Event)
        .join(PassEvent, PassEvent.event_id == Event.id)
        .where(Event.type == "Pass")
    )

    # Apply filters
    if match_ids:
        stmt = stmt.where(Event.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == Event.match_id).where(
            Match.competition_id == competition_id
        )

    if not include_incomplete:
        stmt = stmt.where(PassEvent.outcome.is_(None) | (PassEvent.outcome == "Complete"))

    # Order for sequence analysis
    stmt = stmt.order_by(Event.match_id, Event.period, Event.minute, Event.second)

    # Execute query
    logger.info("Executing pass query...")
    rows = session.execute(stmt).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        logger.warning("No passes found with given filters")
        return df

    logger.info(f"Retrieved {len(df):,} passes")

    # Enrich with raw event data (play_pattern, position at time of pass)
    df = _enrich_with_raw_data(session, df)

    # Add player and team names
    df = _add_entity_names(session, df)

    # Add match context
    df = _add_match_context(session, df)

    # Derive additional fields
    df = _derive_fields(df)

    logger.info(f"Final pass dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def _enrich_with_raw_data(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Extract play_pattern, position, and statsbomb_event_id from raw event JSON."""
    if df.empty:
        return df

    raw_event_ids = df["raw_event_id"].unique().tolist()
    total_ids = len(raw_event_ids)
    logger.info(f"Enriching {total_ids:,} passes with raw event data...")

    # Query raw events in smaller batches to avoid SQLite limits
    batch_size = 500
    raw_data = {}

    for i in range(0, len(raw_event_ids), batch_size):
        batch_ids = raw_event_ids[i : i + batch_size]
        stmt = select(
            RawEvent.id,
            RawEvent.statsbomb_event_id,
            RawEvent.raw_json,
        ).where(RawEvent.id.in_(batch_ids))
        rows = session.execute(stmt).all()

        for raw_id, statsbomb_id, raw_json in rows:
            play_pattern = raw_json.get("play_pattern", {}).get("name", "Unknown")
            position = raw_json.get("position", {}).get("name", "Unknown")
            raw_data[raw_id] = {
                "statsbomb_event_id": statsbomb_id,
                "play_pattern": play_pattern,
                "passer_position": position,
            }

        if (i + batch_size) % 5000 == 0 or (i + batch_size) >= total_ids:
            logger.info(f"  Processed {min(i + batch_size, total_ids):,}/{total_ids:,} raw events")

    # Map to dataframe - add statsbomb_event_id for sequence linking
    df["statsbomb_event_id"] = df["raw_event_id"].map(
        lambda x: raw_data.get(x, {}).get("statsbomb_event_id")
    )
    df["play_pattern"] = df["raw_event_id"].map(
        lambda x: raw_data.get(x, {}).get("play_pattern", "Unknown")
    )
    df["passer_position"] = df["raw_event_id"].map(
        lambda x: raw_data.get(x, {}).get("passer_position", "Unknown")
    )

    return df


def _add_entity_names(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Add player and team names."""
    if df.empty:
        return df

    # Get unique player IDs (passer + recipient)
    passer_ids = df["player_id"].dropna().unique().tolist()
    recipient_ids = df["recipient_id"].dropna().unique().tolist()
    all_player_ids = list(set(passer_ids + recipient_ids))

    # Query player names
    player_map = {}
    if all_player_ids:
        stmt = select(Player.id, Player.name).where(Player.id.in_(all_player_ids))
        for player_id, name in session.execute(stmt).all():
            player_map[player_id] = name

    df["passer_name"] = df["player_id"].map(player_map)
    df["recipient_name"] = df["recipient_id"].map(player_map)

    # Get team names
    team_ids = df["team_id"].unique().tolist()
    team_map = {}
    if team_ids:
        stmt = select(Team.id, Team.name).where(Team.id.in_(team_ids))
        for team_id, name in session.execute(stmt).all():
            team_map[team_id] = name

    df["team_name"] = df["team_id"].map(team_map)

    return df


def _add_match_context(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Add match-level context (opponent, competition)."""
    if df.empty:
        return df

    match_ids = df["match_id"].unique().tolist()
    logger.info(f"Adding match context for {len(match_ids):,} matches...")

    # Query match info
    stmt = select(
        Match.id,
        Match.home_team_id,
        Match.away_team_id,
        Match.competition_id,
        Match.match_date,
    ).where(Match.id.in_(match_ids))

    match_data = {}
    for row in session.execute(stmt).mappings().all():
        match_data[row["id"]] = dict(row)

    # Create mapping dictionaries for vectorized operations
    home_map = {m: d.get("home_team_id") for m, d in match_data.items()}
    away_map = {m: d.get("away_team_id") for m, d in match_data.items()}
    comp_map = {m: d.get("competition_id") for m, d in match_data.items()}
    date_map = {m: d.get("match_date") for m, d in match_data.items()}

    # Vectorized opponent calculation using numpy where
    df["_home_team_id"] = df["match_id"].map(home_map)
    df["_away_team_id"] = df["match_id"].map(away_map)

    import numpy as np

    df["opponent_team_id"] = np.where(
        df["team_id"] == df["_home_team_id"], df["_away_team_id"], df["_home_team_id"]
    )

    # Drop helper columns
    df = df.drop(columns=["_home_team_id", "_away_team_id"])

    df["competition_id"] = df["match_id"].map(comp_map)
    df["match_date"] = df["match_id"].map(date_map)

    return df


def _derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional useful fields."""
    if df.empty:
        return df

    logger.info("Deriving additional fields...")

    # Pass completion flag
    df["is_complete"] = (df["pass_outcome"].isna() | (df["pass_outcome"] == "Complete")).astype(
        bool
    )

    # Broad position groups (for on-pitch analysis) - vectorized
    df["on_pitch_zone"] = pd.cut(
        df["end_x"].fillna(0), bins=[-1, 40, 80, 121], labels=["Defensive", "Middle", "Attacking"]
    ).astype(str)
    df.loc[df["end_x"].isna(), "on_pitch_zone"] = "Unknown"

    # Set piece flag
    df["is_set_piece"] = df["play_pattern"].isin(
        [
            "From Corner",
            "From Free Kick",
            "From Throw In",
            "From Goal Kick",
            "From Kick Off",
        ]
    )

    # Progressive pass (moves ball significantly toward goal) - vectorized
    df["is_progressive"] = ((df["end_x"].fillna(0) - df["start_x"].fillna(0)) >= 10) & (
        df["end_x"].fillna(0) >= 48
    )

    # Final third pass
    df["is_final_third"] = df["end_x"].fillna(0) >= 80

    # Into box pass (approximation: x > 102, 18 < y < 62)
    df["is_into_box"] = (
        (df["end_x"].fillna(0) >= 102)
        & (df["end_y"].fillna(0) >= 18)
        & (df["end_y"].fillna(0) <= 62)
    )

    return df


def save_pass_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "pass_dataset.csv",
) -> Path:
    """Save pass dataset to CSV.

    Args:
        df: Pass dataset DataFrame
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved pass dataset to {output_path}")

    return output_path
