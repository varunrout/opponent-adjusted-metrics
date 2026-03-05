"""Lineup data extraction for tactical position analysis.

Extracts tactical positions and formations from Starting XI events.
This provides the assigned/tactical position for each player,
which can differ from their on-pitch position during the game.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import (
    RawEvent,
    Match,
    Team,
    Player,
)

logger = logging.getLogger(__name__)


def build_lineup_dataset(
    session: Session,
    *,
    match_ids: Optional[list[int]] = None,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Build lineup dataset from Starting XI events.

    Extracts:
    - Player tactical positions (assigned role)
    - Team formations
    - Jersey numbers

    Args:
        session: SQLAlchemy session
        match_ids: Optional list of match IDs to filter
        competition_id: Optional competition ID to filter

    Returns:
        DataFrame with lineup data per player per match
    """
    logger.info("Building lineup dataset...")

    # Query Starting XI events
    stmt = (
        select(
            RawEvent.id,
            RawEvent.match_id,
            RawEvent.raw_json,
        )
        .where(RawEvent.type == "Starting XI")
    )

    if match_ids:
        stmt = stmt.where(RawEvent.match_id.in_(match_ids))

    if competition_id:
        stmt = stmt.join(Match, Match.id == RawEvent.match_id).where(
            Match.competition_id == competition_id
        )

    rows = session.execute(stmt).all()
    logger.info(f"Found {len(rows)} Starting XI events")

    # Parse lineup data
    lineup_rows = []
    for raw_id, match_id, raw_json in rows:
        team_data = raw_json.get("team", {})
        team_sb_id = team_data.get("id")
        team_name = team_data.get("name")

        tactics = raw_json.get("tactics", {})
        formation = tactics.get("formation")
        lineup = tactics.get("lineup", [])

        for player_entry in lineup:
            player_data = player_entry.get("player", {})
            position_data = player_entry.get("position", {})

            lineup_rows.append({
                "match_id": match_id,
                "team_statsbomb_id": team_sb_id,
                "team_name": team_name,
                "player_statsbomb_id": player_data.get("id"),
                "player_name": player_data.get("name"),
                "tactical_position": position_data.get("name"),
                "tactical_position_id": position_data.get("id"),
                "formation": formation,
                "jersey_number": player_entry.get("jersey_number"),
            })

    df = pd.DataFrame(lineup_rows)

    if df.empty:
        logger.warning("No lineup data found")
        return df

    logger.info(f"Extracted {len(df):,} player-match lineup entries")

    # Map to internal IDs
    df = _map_to_internal_ids(session, df)

    # Add position groups
    df = _add_position_groups(df)

    return df


def _map_to_internal_ids(session: Session, df: pd.DataFrame) -> pd.DataFrame:
    """Map StatsBomb IDs to internal database IDs."""
    if df.empty:
        return df

    # Map team IDs
    team_sb_ids = df["team_statsbomb_id"].dropna().unique().tolist()
    if team_sb_ids:
        stmt = select(Team.id, Team.statsbomb_team_id).where(
            Team.statsbomb_team_id.in_(team_sb_ids)
        )
        team_map = {sb_id: db_id for db_id, sb_id in session.execute(stmt).all()}
        df["team_id"] = df["team_statsbomb_id"].map(team_map)
    else:
        df["team_id"] = None

    # Map player IDs
    player_sb_ids = df["player_statsbomb_id"].dropna().unique().tolist()
    if player_sb_ids:
        stmt = select(Player.id, Player.statsbomb_player_id).where(
            Player.statsbomb_player_id.in_(player_sb_ids)
        )
        player_map = {sb_id: db_id for db_id, sb_id in session.execute(stmt).all()}
        df["player_id"] = df["player_statsbomb_id"].map(player_map)
    else:
        df["player_id"] = None

    return df


def _add_position_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add broad position group classification."""
    if df.empty:
        return df

    # StatsBomb position mapping
    position_groups = {
        "Goalkeeper": "Goalkeeper",
        "Right Back": "Defender",
        "Right Center Back": "Defender",
        "Center Back": "Defender",
        "Left Center Back": "Defender",
        "Left Back": "Defender",
        "Right Wing Back": "Defender",
        "Left Wing Back": "Defender",
        "Right Defensive Midfield": "Midfielder",
        "Center Defensive Midfield": "Midfielder",
        "Left Defensive Midfield": "Midfielder",
        "Right Midfield": "Midfielder",
        "Right Center Midfield": "Midfielder",
        "Center Midfield": "Midfielder",
        "Left Center Midfield": "Midfielder",
        "Left Midfield": "Midfielder",
        "Right Attacking Midfield": "Midfielder",
        "Center Attacking Midfield": "Midfielder",
        "Left Attacking Midfield": "Midfielder",
        "Right Wing": "Forward",
        "Left Wing": "Forward",
        "Right Center Forward": "Forward",
        "Center Forward": "Forward",
        "Left Center Forward": "Forward",
        "Striker": "Forward",
        "Secondary Striker": "Forward",
    }

    df["position_group"] = df["tactical_position"].map(position_groups).fillna("Unknown")

    # More granular grouping
    detailed_groups = {
        "Goalkeeper": "GK",
        "Right Back": "FB",
        "Left Back": "FB",
        "Right Wing Back": "WB",
        "Left Wing Back": "WB",
        "Right Center Back": "CB",
        "Center Back": "CB",
        "Left Center Back": "CB",
        "Right Defensive Midfield": "DM",
        "Center Defensive Midfield": "DM",
        "Left Defensive Midfield": "DM",
        "Right Midfield": "WM",
        "Left Midfield": "WM",
        "Right Center Midfield": "CM",
        "Center Midfield": "CM",
        "Left Center Midfield": "CM",
        "Right Attacking Midfield": "AM",
        "Center Attacking Midfield": "AM",
        "Left Attacking Midfield": "AM",
        "Right Wing": "W",
        "Left Wing": "W",
        "Right Center Forward": "CF",
        "Center Forward": "CF",
        "Left Center Forward": "CF",
        "Striker": "ST",
        "Secondary Striker": "SS",
    }

    df["position_code"] = df["tactical_position"].map(detailed_groups).fillna("UNK")

    return df


def enrich_passes_with_lineup(
    pass_df: pd.DataFrame,
    lineup_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich pass dataset with tactical position from lineup.

    Args:
        pass_df: Pass dataset
        lineup_df: Lineup dataset

    Returns:
        Pass dataset with tactical position columns added
    """
    if pass_df.empty or lineup_df.empty:
        return pass_df

    # Create lookup key: (match_id, player_id) -> tactical_position
    lineup_lookup = {}
    for _, row in lineup_df.iterrows():
        key = (row["match_id"], row["player_id"])
        lineup_lookup[key] = {
            "tactical_position": row["tactical_position"],
            "position_group": row["position_group"],
            "position_code": row["position_code"],
            "formation": row["formation"],
        }

    # Map to passes
    def get_lineup_info(row, field):
        key = (row["match_id"], row["player_id"])
        return lineup_lookup.get(key, {}).get(field)

    pass_df["tactical_position"] = pass_df.apply(
        lambda r: get_lineup_info(r, "tactical_position"), axis=1
    )
    pass_df["tactical_group"] = pass_df.apply(
        lambda r: get_lineup_info(r, "position_group"), axis=1
    )
    pass_df["tactical_code"] = pass_df.apply(
        lambda r: get_lineup_info(r, "position_code"), axis=1
    )
    pass_df["formation"] = pass_df.apply(
        lambda r: get_lineup_info(r, "formation"), axis=1
    )

    # Fill unknowns
    pass_df["tactical_position"] = pass_df["tactical_position"].fillna("Unknown")
    pass_df["tactical_group"] = pass_df["tactical_group"].fillna("Unknown")
    pass_df["tactical_code"] = pass_df["tactical_code"].fillna("UNK")

    matched = (pass_df["tactical_position"] != "Unknown").sum()
    logger.info(
        f"Matched {matched:,}/{len(pass_df):,} passes "
        f"({100*matched/len(pass_df):.1f}%) to tactical positions"
    )

    return pass_df


def save_lineup_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "lineup_dataset.csv",
) -> Path:
    """Save lineup dataset to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved lineup dataset to {output_path}")

    return output_path
