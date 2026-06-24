"""
Action Sequence Builder for True xA Methodology.

Builds complete action sequences leading to shots, including:
- Passes (direct assists)
- Carries (ball progression by same player)
- Dribbles (take-ons)

This enables credit distribution across all contributors,
not just the final pass (traditional xA).

Key concepts:
- Action sequence: All ball-progression events leading to a shot
- Pre-assist: Second-to-last action that enables the final assist
- Credit distribution: Value shared across sequence based on contribution
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import uuid

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# Event types that can be part of an assist sequence
SEQUENCE_ACTION_TYPES = {"Pass", "Carry", "Dribble"}


def build_action_sequences(
    session: Session,
    competition_id: Optional[int] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Build complete action sequences leading to shots.

    Unlike pass-only sequences, this includes carries and dribbles
    that contribute to chance creation.

    Args:
        session: Database session
        competition_id: Optional filter
        k: Maximum actions to trace back (default 5)

    Returns:
        DataFrame with sequence-level data
    """
    logger.info(f"Building action sequences (k={k})...")

    # Get shots with assist info
    shots_df = _get_shots_with_assists(session, competition_id)
    logger.info(f"Found {len(shots_df):,} shots with assists")

    # Get all ball-progression events
    actions_df = _get_ball_progression_events(session, competition_id)
    logger.info(f"Found {len(actions_df):,} ball progression events")

    # Build sequences
    sequences = []

    for _, shot in shots_df.iterrows():
        seq = _trace_action_sequence(
            shot=shot,
            actions_df=actions_df,
            k=k,
        )
        if seq:
            sequences.append(seq)

    if not sequences:
        logger.warning("No sequences built!")
        return pd.DataFrame()

    sequence_df = pd.DataFrame(sequences)

    # Add derived features
    sequence_df = _add_sequence_features(sequence_df, k)

    logger.info(f"Built {len(sequence_df):,} action sequences")
    _log_sequence_stats(sequence_df)

    return sequence_df


def _get_shots_with_assists(
    session: Session,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Get shots that have assist information."""
    query = """
    SELECT 
        e.id as shot_id,
        e.raw_event_id,
        e.match_id,
        e.team_id,
        e.player_id as shooter_id,
        p.name as shooter_name,
        e.minute,
        e.second,
        e.location_x as shot_x,
        e.location_y as shot_y,
        e.possession,
        s.statsbomb_xg,
        s.outcome,
        CASE WHEN s.outcome = 'Goal' THEN 1 ELSE 0 END as is_goal
    FROM events e
    JOIN shots s ON e.id = s.event_id
    JOIN matches m ON e.match_id = m.id
    LEFT JOIN players p ON e.player_id = p.id
    WHERE e.type = 'Shot'
    """

    if competition_id:
        query += f" AND m.competition_id = {competition_id}"

    query += " ORDER BY e.match_id, e.minute, e.second"

    return pd.read_sql(query, session.bind)


def _get_ball_progression_events(
    session: Session,
    competition_id: Optional[int] = None,
) -> pd.DataFrame:
    """Get all ball progression events (passes, carries, dribbles)."""
    query = """
    SELECT 
        e.id as event_id,
        e.raw_event_id,
        e.match_id,
        e.team_id,
        e.player_id,
        pl.name as player_name,
        e.type as event_type,
        e.minute,
        e.second,
        e.location_x as start_x,
        e.location_y as start_y,
        e.possession,
        e.under_pressure,
        -- Pass specific
        p.end_x as pass_end_x,
        p.end_y as pass_end_y,
        p.outcome as pass_outcome,
        p.pass_height,
        p.is_cross,
        p.is_through_ball,
        p.recipient_player_id,
        rp.name as recipient_player_name,
        -- Carry specific
        c.end_x as carry_end_x,
        c.end_y as carry_end_y,
        c.length as carry_length,
        -- Dribble specific
        d.outcome as dribble_outcome,
        d.nutmeg as dribble_nutmeg
    FROM events e
    JOIN matches m ON e.match_id = m.id
    LEFT JOIN players pl ON e.player_id = pl.id
    LEFT JOIN passes p ON e.id = p.event_id
    LEFT JOIN players rp ON p.recipient_player_id = rp.id
    LEFT JOIN carries c ON e.id = c.event_id
    LEFT JOIN dribbles d ON e.id = d.event_id
    WHERE e.type IN ('Pass', 'Carry', 'Dribble')
    """

    if competition_id:
        query += f" AND m.competition_id = {competition_id}"

    query += " ORDER BY e.match_id, e.possession, e.minute, e.second"

    df = pd.read_sql(query, session.bind)

    # Consolidate end locations
    df["end_x"] = df["pass_end_x"].fillna(df["carry_end_x"]).fillna(df["start_x"])
    df["end_y"] = df["pass_end_y"].fillna(df["carry_end_y"]).fillna(df["start_y"])

    return df


def _trace_action_sequence(
    shot: pd.Series,
    actions_df: pd.DataFrame,
    k: int,
) -> Optional[Dict[str, Any]]:
    """
    Trace back from shot to build action sequence.

    Unlike pass-only tracing, this includes carries and dribbles.
    """
    match_id = shot["match_id"]
    team_id = shot["team_id"]
    possession = shot["possession"]
    shot_x = shot["shot_x"]
    shot_y = shot["shot_y"]
    shot_minute = shot["minute"]
    shot_second = shot["second"]

    # Filter to same possession
    poss_actions = actions_df[
        (actions_df["match_id"] == match_id)
        & (actions_df["team_id"] == team_id)
        & (actions_df["possession"] == possession)
    ].copy()

    if poss_actions.empty:
        return None

    # Sort by time (most recent first for tracing back)
    poss_actions = poss_actions.sort_values(["minute", "second"], ascending=[False, False])

    # Find key pass/action - the one closest to shot location before shot time
    before_shot = poss_actions[
        (poss_actions["minute"] < shot_minute)
        | ((poss_actions["minute"] == shot_minute) & (poss_actions["second"] < shot_second))
    ].copy()

    if before_shot.empty:
        return None

    # Find action with end location closest to shot location
    before_shot["dist_to_shot"] = np.sqrt(
        (before_shot["end_x"] - shot_x) ** 2 + (before_shot["end_y"] - shot_y) ** 2
    )

    # Key action is the one closest to shot (within 10m and within 10 seconds)
    recent = before_shot[
        (before_shot["dist_to_shot"] < 10)
        & ((shot_minute - before_shot["minute"]) * 60 + (shot_second - before_shot["second"]) < 10)
    ]

    if recent.empty:
        return None

    key_pass = recent.nsmallest(1, "dist_to_shot").iloc[0]

    # Get all actions before the shot, after or at key pass time
    before_shot = poss_actions[
        (poss_actions["minute"] < shot_minute)
        | ((poss_actions["minute"] == shot_minute) & (poss_actions["second"] < shot_second))
    ]

    # Trace chain backwards from key pass
    chain = [key_pass.to_dict()]
    current_player = key_pass["player_id"]
    current_start = (key_pass["start_x"], key_pass["start_y"])

    # Look for preceding actions
    candidates = before_shot[
        (before_shot["minute"] < key_pass["minute"])
        | (
            (before_shot["minute"] == key_pass["minute"])
            & (before_shot["second"] < key_pass["second"])
        )
    ]

    for _ in range(k - 1):  # Already have key pass
        if candidates.empty:
            break

        # Find action that leads to current action
        # Either: pass to current player, or same player's carry/dribble
        found_prev = False

        for _, candidate in candidates.iterrows():
            end_loc = (candidate["end_x"], candidate["end_y"])

            # Check if this action connects to current
            # Pass: recipient is current player
            if candidate["event_type"] == "Pass":
                if candidate.get("recipient_player_id") == current_player:
                    # Location check (within 5m)
                    dist = np.sqrt(
                        (end_loc[0] - current_start[0]) ** 2 + (end_loc[1] - current_start[1]) ** 2
                    )
                    if dist < 10:  # Reasonable tolerance
                        chain.append(candidate.to_dict())
                        current_player = candidate["player_id"]
                        current_start = (candidate["start_x"], candidate["start_y"])
                        found_prev = True
                        break

            # Carry/Dribble: same player
            elif candidate["event_type"] in ("Carry", "Dribble"):
                if candidate["player_id"] == current_player:
                    dist = np.sqrt(
                        (end_loc[0] - current_start[0]) ** 2 + (end_loc[1] - current_start[1]) ** 2
                    )
                    if dist < 10:
                        chain.append(candidate.to_dict())
                        current_start = (candidate["start_x"], candidate["start_y"])
                        found_prev = True
                        break

        if not found_prev:
            break

        # Update candidates (before current action)
        current_action = chain[-1]
        candidates = candidates[
            (candidates["minute"] < current_action["minute"])
            | (
                (candidates["minute"] == current_action["minute"])
                & (candidates["second"] < current_action["second"])
            )
        ]

    # Build sequence data
    seq_id = str(uuid.uuid4())
    seq_data = {
        "sequence_id": seq_id,
        "match_id": match_id,
        "team_id": team_id,
        "possession": possession,
        "shot_id": shot["shot_id"],
        "shooter_id": shot["shooter_id"],
        "shooter_name": shot["shooter_name"],
        "shot_x": shot["shot_x"],
        "shot_y": shot["shot_y"],
        "shot_xg": shot["statsbomb_xg"],
        "shot_minute": shot["minute"],
        "shot_second": shot["second"],
        "is_goal": shot["is_goal"],
        "num_actions": len(chain),
    }

    # Add per-action features (1 = key pass, 2 = pre-assist, etc.)
    for i, action in enumerate(chain):
        pos = i + 1
        prefix = f"action{pos}_"

        seq_data[f"{prefix}type"] = action["event_type"]
        seq_data[f"{prefix}player_id"] = action["player_id"]
        seq_data[f"{prefix}player_name"] = action["player_name"]
        seq_data[f"{prefix}start_x"] = action["start_x"]
        seq_data[f"{prefix}start_y"] = action["start_y"]
        seq_data[f"{prefix}end_x"] = action["end_x"]
        seq_data[f"{prefix}end_y"] = action["end_y"]
        seq_data[f"{prefix}under_pressure"] = action["under_pressure"]

        if action["event_type"] == "Pass":
            seq_data[f"{prefix}is_cross"] = action.get("is_cross", False)
            seq_data[f"{prefix}is_through_ball"] = action.get("is_through_ball", False)
            seq_data[f"{prefix}pass_height"] = action.get("pass_height")

    return seq_data


def _add_sequence_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add derived features to sequence data."""
    df = df.copy()

    # Count action types in sequence
    df["num_passes"] = 0
    df["num_carries"] = 0
    df["num_dribbles"] = 0

    for i in range(1, k + 1):
        type_col = f"action{i}_type"
        if type_col in df.columns:
            df["num_passes"] += (df[type_col] == "Pass").fillna(False).astype(int)
            df["num_carries"] += (df[type_col] == "Carry").fillna(False).astype(int)
            df["num_dribbles"] += (df[type_col] == "Dribble").fillna(False).astype(int)

    # Sequence progression (total x gained)
    first_start_x = None
    for i in range(k, 0, -1):
        col = f"action{i}_start_x"
        if col in df.columns and first_start_x is None:
            first_start_x = df[col]
        elif col in df.columns:
            first_start_x = df[col].fillna(first_start_x)

    if first_start_x is not None:
        df["sequence_x_progression"] = df["shot_x"] - first_start_x
    else:
        df["sequence_x_progression"] = 0

    # Key pass features (action1)
    if "action1_end_x" in df.columns:
        df["key_pass_distance_to_goal"] = np.sqrt(
            (120 - df["action1_end_x"]) ** 2 + (40 - df["action1_end_y"]) ** 2
        )
        df["key_pass_enters_box"] = (
            (df["action1_end_x"] >= 102) & (df["action1_end_y"] >= 18) & (df["action1_end_y"] <= 62)
        ).astype(int)

    # Pre-assist features (action2 if exists)
    if "action2_type" in df.columns:
        df["has_pre_assist"] = df["action2_type"].notna().astype(int)
        df["pre_assist_is_pass"] = (df["action2_type"] == "Pass").astype(int)
        df["pre_assist_is_carry"] = (df["action2_type"] == "Carry").astype(int)
    else:
        df["has_pre_assist"] = 0
        df["pre_assist_is_pass"] = 0
        df["pre_assist_is_carry"] = 0

    # Unique contributors
    player_cols = [
        f"action{i}_player_id" for i in range(1, k + 1) if f"action{i}_player_id" in df.columns
    ]
    if player_cols:
        df["num_contributors"] = df[player_cols].apply(lambda row: row.dropna().nunique(), axis=1)

    return df


def _log_sequence_stats(df: pd.DataFrame) -> None:
    """Log summary statistics."""
    logger.info(f"  Goals: {df['is_goal'].sum():,}")
    logger.info(f"  Mean shot xG: {df['shot_xg'].mean():.4f}")
    logger.info(f"  Mean actions per sequence: {df['num_actions'].mean():.2f}")
    logger.info(f"  Sequences with pre-assist: {(df['has_pre_assist'] == 1).sum():,}")
    logger.info(
        f"  Action types: Passes={df['num_passes'].sum()}, "
        f"Carries={df['num_carries'].sum()}, "
        f"Dribbles={df['num_dribbles'].sum()}"
    )


def build_action_level_dataset(
    sequence_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    Flatten sequences to action-level for per-action modeling.

    Each row represents one action in a sequence, with:
    - Action features
    - Sequence context
    - Target: is_goal (binary)

    Args:
        sequence_df: Sequence-level data
        k: Max actions per sequence

    Returns:
        Action-level DataFrame
    """
    actions = []

    for _, seq in sequence_df.iterrows():
        seq_id = seq["sequence_id"]
        is_goal = seq["is_goal"]
        shot_xg = seq["shot_xg"]
        num_actions = seq["num_actions"]

        for pos in range(1, k + 1):
            type_col = f"action{pos}_type"
            if type_col not in seq or pd.isna(seq[type_col]):
                continue

            action = {
                "sequence_id": seq_id,
                "action_position": pos,
                "actions_to_shot": pos,
                "is_key_action": pos == 1,
                "is_pre_assist": pos == 2,
                # Action features
                "action_type": seq[type_col],
                "player_id": seq.get(f"action{pos}_player_id"),
                "player_name": seq.get(f"action{pos}_player_name"),
                "start_x": seq.get(f"action{pos}_start_x"),
                "start_y": seq.get(f"action{pos}_start_y"),
                "end_x": seq.get(f"action{pos}_end_x"),
                "end_y": seq.get(f"action{pos}_end_y"),
                "under_pressure": seq.get(f"action{pos}_under_pressure", False),
                "is_cross": seq.get(f"action{pos}_is_cross", False),
                "is_through_ball": seq.get(f"action{pos}_is_through_ball", False),
                # Sequence context
                "match_id": seq["match_id"],
                "team_id": seq["team_id"],
                "num_actions_in_sequence": num_actions,
                "shot_x": seq["shot_x"],
                "shot_y": seq["shot_y"],
                # Targets
                "is_goal": is_goal,
                "shot_xg": shot_xg,
            }

            # Action-specific features
            action["distance"] = (
                np.sqrt(
                    (action["end_x"] - action["start_x"]) ** 2
                    + (action["end_y"] - action["start_y"]) ** 2
                )
                if action["end_x"] and action["start_x"]
                else 0
            )

            action["x_progress"] = (
                (action["end_x"] - action["start_x"])
                if action["end_x"] and action["start_x"]
                else 0
            )

            action["distance_to_goal"] = (
                np.sqrt((120 - action["end_x"]) ** 2 + (40 - action["end_y"]) ** 2)
                if action["end_x"]
                else None
            )

            actions.append(action)

    return pd.DataFrame(actions)
