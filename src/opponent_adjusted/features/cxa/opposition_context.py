"""
Opposition Context Features for Action Sequences.

Enriches action sequences with opposition defensive metrics:
- Opponent defensive profile (global + zone ratings)
- Sequence pressure metrics
- Game state context
- Match-level opponent stats
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def assign_shot_zone(shot_x: float, shot_y: float) -> str:
    """
    Assign shot to defensive zone (A-F) based on location.
    
    Zones based on distance and centrality:
    - A: Close central (< 12m, central)
    - B: Close wide (< 12m, wide)
    - C: Mid central (12-20m, central)
    - D: Mid wide (12-20m, wide)
    - E: Far central (> 20m, central)
    - F: Far wide (> 20m, wide)
    """
    # Distance to goal (goal at x=120)
    goal_x, goal_y = 120.0, 40.0
    distance = np.sqrt((shot_x - goal_x)**2 + (shot_y - goal_y)**2)
    
    # Centrality (distance from center line at y=40)
    centrality = abs(shot_y - 40.0)
    is_central = centrality < 12  # Within 12m of center
    
    if distance < 12:
        return "A" if is_central else "B"
    elif distance < 20:
        return "C" if is_central else "D"
    else:
        return "E" if is_central else "F"


def get_match_opponent_mapping(session: Session) -> pd.DataFrame:
    """Get mapping of match_id + team_id -> opponent_team_id."""
    query = """
    SELECT 
        m.id as match_id,
        m.home_team_id,
        m.away_team_id
    FROM matches m
    """
    df = pd.read_sql(query, session.bind)
    
    # Create two rows per match (one for each team's perspective)
    home = df[['match_id', 'home_team_id', 'away_team_id']].copy()
    home.columns = ['match_id', 'team_id', 'opponent_team_id']
    
    away = df[['match_id', 'away_team_id', 'home_team_id']].copy()
    away.columns = ['match_id', 'team_id', 'opponent_team_id']
    
    return pd.concat([home, away], ignore_index=True)


def get_match_goals(session: Session) -> pd.DataFrame:
    """Get goal times to calculate score differential at any minute."""
    query = """
    SELECT 
        e.match_id,
        e.team_id,
        e.minute,
        e.second
    FROM events e
    JOIN shots s ON e.id = s.event_id
    WHERE s.outcome = 'Goal'
    ORDER BY e.match_id, e.minute, e.second
    """
    return pd.read_sql(query, session.bind)


def build_score_lookup(goals_df: pd.DataFrame) -> dict:
    """Build a lookup dict for score differential at each minute per match/team."""
    lookup = {}
    
    for match_id in goals_df['match_id'].unique():
        match_goals = goals_df[goals_df['match_id'] == match_id]
        teams = match_goals['team_id'].unique().tolist()
        
        if len(teams) < 2:
            continue
            
        # Build cumulative score by minute
        for minute in range(0, 150):
            before = match_goals[match_goals['minute'] < minute]
            for team_id in teams:
                team_goals = len(before[before['team_id'] == team_id])
                opp_goals = len(before[before['team_id'] != team_id])
                lookup[(match_id, team_id, minute)] = team_goals - opp_goals
    
    return lookup


def get_match_defensive_stats(session: Session) -> pd.DataFrame:
    """Get match-level defensive statistics per opponent."""
    query = """
    SELECT 
        e.match_id,
        m.home_team_id,
        m.away_team_id,
        e.team_id as defending_team_id,
        COUNT(CASE WHEN e.type = 'Block' THEN 1 END) as blocks,
        COUNT(CASE WHEN e.type = 'Interception' THEN 1 END) as interceptions,
        COUNT(CASE WHEN e.type = 'Clearance' THEN 1 END) as clearances,
        COUNT(CASE WHEN e.type = 'Pressure' THEN 1 END) as pressures
    FROM events e
    JOIN matches m ON e.match_id = m.id
    WHERE e.type IN ('Block', 'Interception', 'Clearance', 'Pressure')
    GROUP BY e.match_id, e.team_id, m.home_team_id, m.away_team_id
    """
    return pd.read_sql(query, session.bind)


def add_opponent_profiles(
    df: pd.DataFrame, 
    profiles: pd.DataFrame,
    opponent_mapping: pd.DataFrame
) -> pd.DataFrame:
    """Add opponent defensive profile metrics to sequences."""
    logger.info("Adding opponent defensive profiles...")
    
    if profiles.empty:
        logger.warning("No opponent profiles available, skipping")
        return df
    
    # Join to get opponent_team_id
    df = df.merge(
        opponent_mapping[['match_id', 'team_id', 'opponent_team_id']],
        on=['match_id', 'team_id'],
        how='left'
    )
    
    # Assign shot zone
    df['shot_zone'] = df.apply(
        lambda r: assign_shot_zone(r['shot_x'], r['shot_y']) 
        if pd.notna(r['shot_x']) else None,
        axis=1
    )
    
    # Ensure expected columns exist
    required_global = ["team_id", "global_rating", "block_rate", "shots_conceded", "goals_conceded"]
    for col in required_global:
        if col not in profiles.columns:
            profiles[col] = np.nan

    if "zone_id" not in profiles.columns:
        profiles["zone_id"] = np.nan

    if "zone_rating" not in profiles.columns:
        profiles["zone_rating"] = np.nan

    # Get global profiles
    global_profiles = profiles[profiles['zone_id'].isna()][
        ['team_id', 'global_rating', 'block_rate', 'shots_conceded', 'goals_conceded']
    ].copy()
    global_profiles.columns = [
        'opponent_team_id', 'opponent_global_rating', 'opponent_block_rate',
        'opponent_shots_conceded', 'opponent_goals_conceded'
    ]
    
    df = df.merge(global_profiles, on='opponent_team_id', how='left')
    
    # Get zone profiles
    zone_profiles = profiles[profiles['zone_id'].notna()][
        ['team_id', 'zone_id', 'zone_rating']
    ].copy()
    zone_profiles.columns = ['opponent_team_id', 'zone_id', 'opponent_zone_rating']
    
    df = df.merge(
        zone_profiles,
        left_on=['opponent_team_id', 'shot_zone'],
        right_on=['opponent_team_id', 'zone_id'],
        how='left'
    )
    
    # Drop redundant zone_id column
    if 'zone_id' in df.columns:
        df = df.drop(columns=['zone_id'])
    
    logger.info(f"  Added profiles for {df['opponent_global_rating'].notna().sum():,} sequences")
    
    return df


def add_pressure_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sequence-level pressure metrics."""
    logger.info("Calculating pressure metrics...")
    
    # Count pressured actions
    pressure_cols = [f'action{i}_under_pressure' for i in range(1, 6)]
    available_pressure_cols = [c for c in pressure_cols if c in df.columns]
    
    if not available_pressure_cols:
        logger.warning("No pressure columns found")
        df['sequence_pressure_count'] = 0
        df['sequence_pressure_rate'] = 0.0
        df['key_action_under_pressure'] = False
        return df
    
    # Convert to numeric (handle boolean/None)
    for col in available_pressure_cols:
        df[col] = df[col].fillna(False).astype(bool).astype(int)
    
    # Count pressured actions
    df['sequence_pressure_count'] = df[available_pressure_cols].sum(axis=1)
    
    # Pressure rate (pressured / total actions)
    df['sequence_pressure_rate'] = df['sequence_pressure_count'] / df['num_actions'].clip(lower=1)
    
    # Key action (action1) under pressure
    if 'action1_under_pressure' in df.columns:
        df['key_action_under_pressure'] = df['action1_under_pressure'].fillna(False).astype(bool)
    else:
        df['key_action_under_pressure'] = False
    
    logger.info(f"  Pressured sequences: {(df['sequence_pressure_count'] > 0).sum():,}")
    logger.info(f"  Mean pressure rate: {df['sequence_pressure_rate'].mean():.1%}")
    
    return df


def add_game_state(df: pd.DataFrame, goals_df: pd.DataFrame) -> pd.DataFrame:
    """Add game state context (score differential, minute bucket)."""
    logger.info("Adding game state context...")
    
    # Get sequence start minute (use shot minute as proxy if no action minute)
    if 'shot_minute' in df.columns:
        df['sequence_minute'] = df['shot_minute']
    elif 'action1_minute' in df.columns:
        df['sequence_minute'] = df['action1_minute']
    else:
        df['sequence_minute'] = 45  # Default to middle
    
    # Fill NaN minutes
    df['sequence_minute'] = df['sequence_minute'].fillna(45).astype(int)
    
    # Build score lookup (vectorized approach)
    logger.info("  Building score lookup...")
    score_lookup = build_score_lookup(goals_df)
    
    # Vectorized score differential lookup
    logger.info("  Calculating score differentials...")
    df['score_differential'] = df.apply(
        lambda r: score_lookup.get(
            (r['match_id'], r['team_id'], int(r['sequence_minute'])), 
            0
        ),
        axis=1
    )
    
    # Game state categories
    df['team_winning'] = df['score_differential'] > 0
    df['team_losing'] = df['score_differential'] < 0
    df['team_drawing'] = df['score_differential'] == 0
    
    # Opponent state (inverse)
    df['opponent_chasing'] = df['team_winning']  # Opponent trailing, may be open
    df['opponent_protecting'] = df['team_losing']  # Opponent leading, may be defensive
    
    # Minute buckets
    df['minute_bucket'] = pd.cut(
        df['sequence_minute'].fillna(45),
        bins=[0, 30, 60, 90, 150],
        labels=['early', 'mid', 'late', 'extra_time'],
        include_lowest=True
    )
    
    logger.info(f"  Winning: {df['team_winning'].sum():,}, "
                f"Drawing: {df['team_drawing'].sum():,}, "
                f"Losing: {df['team_losing'].sum():,}")
    
    return df


def add_match_defensive_stats(
    df: pd.DataFrame, 
    defensive_stats: pd.DataFrame,
    opponent_mapping: pd.DataFrame
) -> pd.DataFrame:
    """Add match-level opponent defensive statistics."""
    logger.info("Adding match-level defensive stats...")
    
    if defensive_stats.empty:
        logger.warning("No defensive stats available")
        return df
    
    # Ensure opponent_team_id is available
    if 'opponent_team_id' not in df.columns:
        df = df.merge(
            opponent_mapping[['match_id', 'team_id', 'opponent_team_id']],
            on=['match_id', 'team_id'],
            how='left'
        )
    
    # Get opponent's defensive actions in this match
    opp_stats = defensive_stats[['match_id', 'defending_team_id', 
                                  'blocks', 'interceptions', 'clearances', 'pressures']].copy()
    opp_stats.columns = ['match_id', 'opponent_team_id',
                         'opponent_match_blocks', 'opponent_match_interceptions',
                         'opponent_match_clearances', 'opponent_match_pressures']
    
    df = df.merge(opp_stats, on=['match_id', 'opponent_team_id'], how='left')
    
    # Total defensive actions
    def_cols = ['opponent_match_blocks', 'opponent_match_interceptions', 
                'opponent_match_clearances', 'opponent_match_pressures']
    for col in def_cols:
        df[col] = df[col].fillna(0)
    
    df['opponent_match_defensive_actions'] = df[def_cols].sum(axis=1)
    
    logger.info(f"  Mean opponent defensive actions: {df['opponent_match_defensive_actions'].mean():.1f}")
    
    return df


def build_opposition_context(
    action_sequences_df: pd.DataFrame,
    opponent_profiles_df: pd.DataFrame,
    session: Session,
) -> pd.DataFrame:
    """
    Build opposition context features for action sequences.
    
    Args:
        action_sequences_df: Action sequences DataFrame
        opponent_profiles_df: Opponent profiles DataFrame from CxG
        session: Database session for fetching additional data
        
    Returns:
        DataFrame with opposition context features added
    """
    logger.info("Building opposition context features...")
    
    df = action_sequences_df.copy()
    
    # Fetch supporting data
    opponent_mapping = get_match_opponent_mapping(session)
    goals_df = get_match_goals(session)
    defensive_stats = get_match_defensive_stats(session)
    
    logger.info(f"  Matches: {opponent_mapping['match_id'].nunique()}")
    logger.info(f"  Goals: {len(goals_df)}")
    
    # Add opposition context features
    df = add_opponent_profiles(df, opponent_profiles_df, opponent_mapping)
    df = add_pressure_metrics(df)
    df = add_game_state(df, goals_df)
    df = add_match_defensive_stats(df, defensive_stats, opponent_mapping)
    
    logger.info(f"Built opposition context: {len(df):,} sequences, {df.shape[1]} columns")
    
    return df
