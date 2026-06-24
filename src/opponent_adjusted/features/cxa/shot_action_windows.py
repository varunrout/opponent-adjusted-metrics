"""Shot action window builder for cXA-xG.

Builds a canonical pre-shot action window for ALL shots (not just goals).
Each row contains one shot with up to N actions (Pass/Carry/Dribble) in the
same possession leading up to it.

This replaces the separate pass-only and action-only sequence builders with
a unified window that supports both goal-attribution and xG-weighted creation.

Output: feature_store/cxa/shot_action_windows.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import Event, Shot, PassEvent, CarryEvent
from opponent_adjusted.db.session import get_session

logger = logging.getLogger(__name__)

# Window parameters
MAX_ACTIONS = 8
MAX_SECONDS_BEFORE_SHOT = 15.0
ACTION_TYPES = {"Pass", "Carry", "Dribble"}


def distance_to_goal(x: float, y: float) -> float:
    """Distance from (x, y) to goal center (120, 40)."""
    return float(np.sqrt((120.0 - x) ** 2 + (40.0 - y) ** 2))


def angle_to_goal(x: float, y: float) -> float:
    """Angle to goal in degrees (0 = straight on, 90 = sideline)."""
    dx = 120.0 - x
    dy = abs(40.0 - y)
    if dx <= 0:
        return 90.0
    return float(np.degrees(np.arctan(dy / dx)))


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _event_timestamp_seconds(minute: int, second: int, period: int) -> float:
    """Convert event time to cumulative seconds (for ordering within match)."""
    period_offset = (period - 1) * 45 * 60 if period <= 2 else 90 * 60 + (period - 3) * 15 * 60
    return period_offset + minute * 60 + second


def build_shot_action_windows(
    session: Optional[Session] = None,
    match_ids: Optional[List[int]] = None,
    max_actions: int = MAX_ACTIONS,
    max_seconds: float = MAX_SECONDS_BEFORE_SHOT,
) -> pd.DataFrame:
    """Build pre-shot action windows for all shots.

    For each shot, extracts the last N actions (Pass/Carry/Dribble) in the
    same possession, up to K seconds before the shot.

    Args:
        session: SQLAlchemy session (creates one if not provided)
        match_ids: Optional filter to specific matches
        max_actions: Maximum actions in window (default 8)
        max_seconds: Maximum seconds before shot to include (default 15)

    Returns:
        DataFrame with one row per shot, wide-format action columns
    """
    logger.info("Building shot action windows...")
    logger.info(f"  Max actions: {max_actions}, Max seconds: {max_seconds}")

    own_session = session is None
    if own_session:
        session = get_session()

    try:
        # Load shots
        shots_df = _load_shots(session, match_ids)
        logger.info(f"  Loaded {len(shots_df):,} shots")

        if shots_df.empty:
            return pd.DataFrame()

        # Load all events for relevant matches
        relevant_matches = shots_df["match_id"].unique().tolist()
        events_df = _load_events(session, relevant_matches)
        logger.info(f"  Loaded {len(events_df):,} events from {len(relevant_matches)} matches")

        # Group events by match_id for faster filtering
        events_by_match = {mid: grp for mid, grp in events_df.groupby("match_id")}

        # Build windows with progress logging
        windows = []
        total = len(shots_df)
        for i, (_, shot) in enumerate(shots_df.iterrows()):
            match_events = events_by_match.get(shot["match_id"], pd.DataFrame())
            window = _build_single_window(shot, match_events, max_actions, max_seconds)
            windows.append(window)

            if (i + 1) % 1000 == 0:
                logger.info(f"    Processed {i+1:,}/{total:,} shots...")

        result = pd.DataFrame(windows)

        # Summary stats
        shots_with_actions = (result["num_actions"] > 0).sum()
        logger.info(
            f"  Shots with ≥1 action: {shots_with_actions:,} ({100*shots_with_actions/len(result):.1f}%)"
        )
        logger.info(f"  Mean actions per shot: {result['num_actions'].mean():.2f}")

        return result

    finally:
        if own_session:
            session.close()


def _load_shots(session: Session, match_ids: Optional[List[int]]) -> pd.DataFrame:
    """Load shots with key columns."""
    stmt = (
        select(
            Shot.id.label("shot_id"),
            Shot.match_id,
            Shot.team_id,
            Shot.player_id,
            Shot.statsbomb_xg,
            Shot.outcome,
            Event.period,
            Event.minute,
            Event.second,
            Event.possession,
            Event.location_x.label("shot_x"),
            Event.location_y.label("shot_y"),
        )
        .select_from(Shot)
        .join(Event, Event.id == Shot.event_id)
        .where(Shot.shot_type != "Penalty")
    )

    if match_ids:
        stmt = stmt.where(Shot.match_id.in_(match_ids))

    stmt = stmt.order_by(Shot.match_id, Event.period, Event.minute, Event.second)

    rows = session.execute(stmt).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])

    if not df.empty:
        df["is_goal"] = df["outcome"].str.lower() == "goal"
        df["shot_timestamp"] = df.apply(
            lambda r: _event_timestamp_seconds(r["minute"], r["second"], r["period"]), axis=1
        )

    return df


def _load_events(session: Session, match_ids: List[int]) -> pd.DataFrame:
    """Load Pass/Carry/Dribble events for window building.

    Note: End locations come from join tables (passes, carries).
    For Dribble events, we use start location as end location (no movement).
    """
    # Load passes with end locations
    pass_stmt = (
        select(
            Event.id.label("event_id"),
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.type.label("event_type"),
            Event.period,
            Event.minute,
            Event.second,
            Event.possession,
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            PassEvent.end_x,
            PassEvent.end_y,
            Event.under_pressure,
        )
        .select_from(Event)
        .join(PassEvent, PassEvent.event_id == Event.id)
        .where(
            and_(
                Event.match_id.in_(match_ids),
                Event.type == "Pass",
            )
        )
    )

    # Load carries with end locations
    carry_stmt = (
        select(
            Event.id.label("event_id"),
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.type.label("event_type"),
            Event.period,
            Event.minute,
            Event.second,
            Event.possession,
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            CarryEvent.end_x,
            CarryEvent.end_y,
            Event.under_pressure,
        )
        .select_from(Event)
        .join(CarryEvent, CarryEvent.event_id == Event.id)
        .where(
            and_(
                Event.match_id.in_(match_ids),
                Event.type == "Carry",
            )
        )
    )

    # Load dribbles (use start location as end, since they don't have end_location)
    dribble_stmt = (
        select(
            Event.id.label("event_id"),
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.type.label("event_type"),
            Event.period,
            Event.minute,
            Event.second,
            Event.possession,
            Event.location_x.label("start_x"),
            Event.location_y.label("start_y"),
            Event.location_x.label("end_x"),  # Use start as end for dribbles
            Event.location_y.label("end_y"),
            Event.under_pressure,
        )
        .select_from(Event)
        .where(
            and_(
                Event.match_id.in_(match_ids),
                Event.type == "Dribble",
            )
        )
    )

    # Execute all queries and combine
    passes = session.execute(pass_stmt).mappings().all()
    carries = session.execute(carry_stmt).mappings().all()
    dribbles = session.execute(dribble_stmt).mappings().all()

    all_events = (
        [dict(r) for r in passes] + [dict(r) for r in carries] + [dict(r) for r in dribbles]
    )
    df = pd.DataFrame(all_events)

    if not df.empty:
        df["event_timestamp"] = df.apply(
            lambda r: _event_timestamp_seconds(r["minute"], r["second"], r["period"]), axis=1
        )
        # Fill missing end locations with start (for carries that don't move)
        df["end_x"] = df["end_x"].fillna(df["start_x"])
        df["end_y"] = df["end_y"].fillna(df["start_y"])

    return df


def _build_single_window(
    shot: pd.Series,
    events_df: pd.DataFrame,
    max_actions: int,
    max_seconds: float,
) -> Dict[str, Any]:
    """Build action window for a single shot."""
    shot_id = shot["shot_id"]
    match_id = shot["match_id"]
    team_id = shot["team_id"]
    possession = shot["possession"]
    shot_ts = shot["shot_timestamp"]

    # Filter to same match, team, possession, before the shot
    mask = (
        (events_df["match_id"] == match_id)
        & (events_df["team_id"] == team_id)
        & (events_df["possession"] == possession)
        & (events_df["event_timestamp"] < shot_ts)
        & (events_df["event_timestamp"] >= shot_ts - max_seconds)
    )

    window_events = events_df.loc[mask]

    # Sort by time descending (most recent first) and take last N
    if len(window_events) > 0:
        window_events = window_events.nlargest(max_actions, "event_timestamp")

    num_actions = len(window_events)

    # Build output row
    row: Dict[str, Any] = {
        "shot_id": shot_id,
        "match_id": match_id,
        "team_id": team_id,
        "possession": possession,
        "shot_minute": shot["minute"],
        "shot_second": shot["second"],
        "shot_x": shot["shot_x"],
        "shot_y": shot["shot_y"],
        "statsbomb_xg": shot["statsbomb_xg"],
        "is_goal": shot["is_goal"],
        "num_actions": num_actions,
    }

    # Add per-action columns (action1 = closest to shot)
    if num_actions > 0:
        evt_values = window_events.to_dict("records")
        for i, evt in enumerate(evt_values, start=1):
            prefix = f"action{i}_"

            end_x = float(evt["end_x"]) if pd.notna(evt["end_x"]) else 60.0
            end_y = float(evt["end_y"]) if pd.notna(evt["end_y"]) else 40.0
            start_x = float(evt["start_x"]) if pd.notna(evt["start_x"]) else 60.0
            start_y = float(evt["start_y"]) if pd.notna(evt["start_y"]) else 40.0

            row[f"{prefix}type"] = evt["event_type"]
            row[f"{prefix}player_id"] = evt["player_id"]
            row[f"{prefix}start_x"] = start_x
            row[f"{prefix}start_y"] = start_y
            row[f"{prefix}end_x"] = end_x
            row[f"{prefix}end_y"] = end_y
            row[f"{prefix}distance_to_goal"] = distance_to_goal(end_x, end_y)
            row[f"{prefix}angle_to_goal"] = angle_to_goal(end_x, end_y)
            row[f"{prefix}under_pressure"] = (
                bool(evt["under_pressure"]) if pd.notna(evt["under_pressure"]) else False
            )
            row[f"{prefix}seconds_to_shot"] = shot_ts - evt["event_timestamp"]

            # Type flags
            row[f"{prefix}is_pass"] = evt["event_type"] == "Pass"
            row[f"{prefix}is_carry"] = evt["event_type"] == "Carry"
            row[f"{prefix}is_dribble"] = evt["event_type"] == "Dribble"

            # Box flag
            row[f"{prefix}is_into_box"] = (end_x >= 102) and (18 <= end_y <= 62)

    # Fill missing action slots with None
    for i in range(num_actions + 1, max_actions + 1):
        prefix = f"action{i}_"
        for col in [
            "type",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "distance_to_goal",
            "angle_to_goal",
            "under_pressure",
            "seconds_to_shot",
            "is_pass",
            "is_carry",
            "is_dribble",
            "is_into_box",
        ]:
            row[f"{prefix}{col}"] = None

    return row


def run_window_builder() -> Path:
    """Run the window builder and save to feature store."""
    repo_root = _get_repo_root()
    output_path = repo_root / "feature_store" / "cxa" / "shot_action_windows.parquet"

    windows_df = build_shot_action_windows()

    if windows_df.empty:
        logger.warning("No windows built!")
        return output_path

    windows_df.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(windows_df):,} shot windows to {output_path}")

    return output_path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    run_window_builder()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
