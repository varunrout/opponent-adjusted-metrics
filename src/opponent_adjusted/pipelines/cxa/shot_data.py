"""Shot data extraction for CxA analysis.

Builds shot dataset with key_pass_id linking for xA attribution.
Extracts key_pass_id from raw StatsBomb JSON to link shots to assists.
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
    Shot,
    RawEvent,
    Player,
    Team,
    Match,
    ShotPrediction,
    ModelRegistry,
)

logger = logging.getLogger(__name__)


def build_shot_dataset(
    session: Session,
    *,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
    include_penalties: bool = False,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> pd.DataFrame:
    """Build shot dataset with key_pass linking.

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter
        include_penalties: Whether to include penalty kicks
        model_name: Optional model name to include CxG predictions
        model_version: Optional model version for predictions

    Returns:
        DataFrame with shot data including key_pass_id
    """
    logger.info("Building shot dataset...")

    # Base query
    stmt = (
        select(
            Shot.id.label("shot_id"),
            Shot.event_id.label("shot_event_id"),
            Shot.match_id,
            Shot.team_id,
            Shot.player_id,
            Shot.opponent_team_id,
            Shot.statsbomb_xg,
            Shot.outcome,
            Shot.body_part,
            Shot.technique,
            Shot.shot_type,
            Shot.first_time,
            Shot.is_blocked,
            # Event details
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            Event.location_x.label("shot_x"),
            Event.location_y.label("shot_y"),
            Event.under_pressure,
            Event.raw_event_id,
        )
        .select_from(Shot)
        .join(Event, Event.id == Shot.event_id)
    )

    # Apply filters
    if match_ids:
        stmt = stmt.where(Shot.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == Shot.match_id).where(
            Match.competition_id == competition_id
        )

    if not include_penalties:
        stmt = stmt.where(Shot.shot_type != "Penalty")

    stmt = stmt.order_by(Shot.match_id, Event.period, Event.minute, Event.second)

    # Execute query
    logger.info("Executing shot query...")
    rows = session.execute(stmt).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        logger.warning("No shots found with given filters")
        return df

    logger.info(f"Retrieved {len(df):,} shots")

    # Extract key_pass_id from raw events
    df = _extract_key_pass_ids(session, df)

    # Add CxG predictions if model specified
    if model_name:
        df = _add_predictions(session, df, model_name, model_version)

    # Add entity names
    df = _add_entity_names(session, df)

    # Derive fields
    df = _derive_fields(df)

    logger.info(f"Final shot dataset: {len(df):,} rows, {len(df.columns)} columns")
    return df


def _extract_key_pass_ids(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Extract key_pass_id from raw event JSON.

    StatsBomb stores key_pass_id in shot.key_pass_id field of raw JSON.
    This links directly to the assist pass.
    """
    if df.empty:
        return df

    raw_event_ids = df["raw_event_id"].unique().tolist()
    total_ids = len(raw_event_ids)
    logger.info(f"Extracting key_pass_ids from {total_ids:,} shot events...")

    # Query raw events in small batches (JSON parsing is expensive)
    batch_size = 200
    key_pass_map = {}

    for i in range(0, len(raw_event_ids), batch_size):
        batch_ids = raw_event_ids[i : i + batch_size]
        stmt = select(RawEvent.id, RawEvent.raw_json).where(RawEvent.id.in_(batch_ids))
        rows = session.execute(stmt).all()

        for raw_id, raw_json in rows:
            shot_data = raw_json.get("shot", {})
            key_pass_id = shot_data.get("key_pass_id")
            play_pattern = raw_json.get("play_pattern", {}).get("name", "Unknown")
            position = raw_json.get("position", {}).get("name", "Unknown")

            key_pass_map[raw_id] = {
                "key_pass_id": key_pass_id,  # StatsBomb event UUID
                "play_pattern": play_pattern,
                "shooter_position": position,
            }

        if (i + batch_size) % 1000 == 0 or (i + batch_size) >= total_ids:
            logger.info(f"  Processed {min(i + batch_size, total_ids):,}/{total_ids:,}")

    df["key_pass_id"] = df["raw_event_id"].map(lambda x: key_pass_map.get(x, {}).get("key_pass_id"))
    df["play_pattern"] = df["raw_event_id"].map(
        lambda x: key_pass_map.get(x, {}).get("play_pattern", "Unknown")
    )
    df["shooter_position"] = df["raw_event_id"].map(
        lambda x: key_pass_map.get(x, {}).get("shooter_position", "Unknown")
    )

    # Count shots with key passes
    shots_with_assist = df["key_pass_id"].notna().sum()
    logger.info(
        f"Shots with key_pass_id: {shots_with_assist:,} ({100*shots_with_assist/len(df):.1f}%)"
    )

    return df


def _add_predictions(
    session: Session,
    df: pd.DataFrame,
    model_name: str,
    model_version: Optional[str],
) -> pd.DataFrame:
    """Add CxG predictions from model."""
    if df.empty:
        return df

    # Find model
    stmt = select(ModelRegistry.id).where(ModelRegistry.model_name == model_name)
    if model_version:
        stmt = stmt.where(ModelRegistry.version == model_version)
    stmt = stmt.order_by(ModelRegistry.id.desc()).limit(1)

    result = session.execute(stmt).scalar()
    if result is None:
        logger.warning(f"Model {model_name} (version={model_version}) not found")
        return df

    model_id = result

    # Get predictions
    shot_ids = df["shot_id"].tolist()
    stmt = (
        select(
            ShotPrediction.shot_id,
            ShotPrediction.raw_probability,
            ShotPrediction.neutral_probability,
        )
        .where(ShotPrediction.model_id == model_id)
        .where(ShotPrediction.shot_id.in_(shot_ids))
    )

    pred_map = {}
    for row in session.execute(stmt).mappings().all():
        pred_map[row["shot_id"]] = {
            "cxg_raw": row["raw_probability"],
            "cxg_neutral": row["neutral_probability"],
        }

    df["cxg_raw"] = df["shot_id"].map(lambda x: pred_map.get(x, {}).get("cxg_raw"))
    df["cxg_neutral"] = df["shot_id"].map(lambda x: pred_map.get(x, {}).get("cxg_neutral"))

    return df


def _add_entity_names(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Add player and team names."""
    if df.empty:
        return df

    # Player names
    player_ids = df["player_id"].dropna().unique().tolist()
    player_map = {}
    if player_ids:
        stmt = select(Player.id, Player.name).where(Player.id.in_(player_ids))
        for player_id, name in session.execute(stmt).all():
            player_map[player_id] = name

    df["shooter_name"] = df["player_id"].map(player_map)

    # Team names
    team_ids = list(set(df["team_id"].tolist() + df["opponent_team_id"].tolist()))
    team_map = {}
    if team_ids:
        stmt = select(Team.id, Team.name).where(Team.id.in_(team_ids))
        for team_id, name in session.execute(stmt).all():
            team_map[team_id] = name

    df["team_name"] = df["team_id"].map(team_map)
    df["opponent_name"] = df["opponent_team_id"].map(team_map)

    return df


def _derive_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional fields."""
    if df.empty:
        return df

    # Goal flag
    df["is_goal"] = df["outcome"] == "Goal"

    # Shot value for xA (prefer CxG neutral > CxG raw > StatsBomb xG)
    if "cxg_neutral" in df.columns:
        df["shot_value"] = df["cxg_neutral"].fillna(
            df.get("cxg_raw", df["statsbomb_xg"]).fillna(df["statsbomb_xg"])
        )
    elif "cxg_raw" in df.columns:
        df["shot_value"] = df["cxg_raw"].fillna(df["statsbomb_xg"])
    else:
        df["shot_value"] = df["statsbomb_xg"]

    # Has assist flag
    df["has_assist"] = df["key_pass_id"].notna()

    # Open play vs set piece
    df["is_open_play"] = df["play_pattern"] == "Regular Play"

    return df


def link_shots_to_passes(
    shot_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    session: Session,
) -> pd.DataFrame:
    """Link shots to their key passes using StatsBomb key_pass_id.

    The key_pass_id in shots is a StatsBomb event UUID that matches
    the statsbomb_event_id in raw_events.

    Args:
        shot_df: Shot dataset with key_pass_id column
        pass_df: Pass dataset with pass_id column
        session: SQLAlchemy session for UUID lookup

    Returns:
        Pass dataset with shot linkage columns added
    """
    if shot_df.empty or pass_df.empty:
        return pass_df

    # Get key_pass UUIDs that exist
    key_pass_uuids = shot_df["key_pass_id"].dropna().unique().tolist()
    if not key_pass_uuids:
        logger.info("No key passes to link")
        pass_df["is_assist"] = False
        pass_df["assisted_shot_id"] = None
        pass_df["shot_value"] = None
        return pass_df

    # Map UUID to raw_event_id
    stmt = select(RawEvent.id, RawEvent.statsbomb_event_id).where(
        RawEvent.statsbomb_event_id.in_(key_pass_uuids)
    )
    uuid_to_raw_id = {}
    for raw_id, uuid in session.execute(stmt).all():
        uuid_to_raw_id[uuid] = raw_id

    # Create shot lookup by key_pass raw_event_id
    shot_df["key_pass_raw_id"] = shot_df["key_pass_id"].map(uuid_to_raw_id)

    shot_lookup = {}
    for _, row in shot_df[shot_df["key_pass_raw_id"].notna()].iterrows():
        raw_id = int(row["key_pass_raw_id"])
        shot_lookup[raw_id] = {
            "shot_id": row["shot_id"],
            "shot_value": row["shot_value"],
            "is_goal": row["is_goal"],
        }

    # Link passes to shots
    pass_df["is_assist"] = pass_df["raw_event_id"].isin(shot_lookup.keys())
    pass_df["assisted_shot_id"] = pass_df["raw_event_id"].map(
        lambda x: shot_lookup.get(x, {}).get("shot_id")
    )
    pass_df["shot_value"] = pass_df["raw_event_id"].map(
        lambda x: shot_lookup.get(x, {}).get("shot_value")
    )
    pass_df["assisted_goal"] = pass_df["raw_event_id"].map(
        lambda x: shot_lookup.get(x, {}).get("is_goal", False)
    )

    assist_count = pass_df["is_assist"].sum()
    logger.info(f"Linked {assist_count:,} passes as assists")

    return pass_df


def save_shot_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "shot_dataset.csv",
) -> Path:
    """Save shot dataset to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved shot dataset to {output_path}")

    return output_path
