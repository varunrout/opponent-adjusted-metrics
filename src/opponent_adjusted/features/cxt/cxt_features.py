"""CxT Feature Engineering Module.

Adds contextual features to ball progression data for CxT modeling:
- Opponent context (defensive quality, zone ratings)
- Game state features (minute, period)
- Pressure indicators
- Zone characteristics
"""

from __future__ import annotations

import logging

import pandas as pd

from opponent_adjusted.db.session import get_session
from opponent_adjusted.db.models import OpponentDefProfile

logger = logging.getLogger(__name__)


def load_opponent_profiles(version_tag: str = "cxt_v1") -> pd.DataFrame:
    """Load opponent xT profiles from database.

    Args:
        version_tag: Profile version to load

    Returns:
        DataFrame with opponent profiles
    """
    logger.info(f"Loading opponent profiles for version: {version_tag}")

    with get_session() as session:
        profiles = (
            session.query(OpponentDefProfile)
            .filter(OpponentDefProfile.version_tag == version_tag)
            .all()
        )

        if not profiles:
            logger.warning(f"No profiles found for version {version_tag}")
            return pd.DataFrame()

        data = []
        for p in profiles:
            data.append(
                {
                    "team_id": p.team_id,
                    "zone_id": p.zone_id,
                    "global_rating": p.global_rating,
                    "zone_rating": p.zone_rating,
                    "block_rate": p.block_rate,
                    "sample_size": p.shots_sample,
                }
            )

        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} profile rows")
        return df


def add_opponent_features(
    df: pd.DataFrame,
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add opponent defensive quality features.

    Args:
        df: Progressions DataFrame with opponent_id column
        profiles_df: Opponent profiles DataFrame

    Returns:
        DataFrame with opponent features added
    """
    logger.info("Adding opponent features...")

    df = df.copy()

    if profiles_df.empty:
        logger.warning("No opponent profiles available, adding placeholders")
        df["opponent_global_rating"] = 50.0
        df["opponent_zone_rating"] = 50.0
        df["opponent_block_rate"] = 0.5
        return df

    # Extract global profiles (zone_id is None)
    global_profiles = profiles_df[profiles_df["zone_id"].isna()].copy()
    global_map = global_profiles.set_index("team_id")["global_rating"].to_dict()
    global_block_map = global_profiles.set_index("team_id")["block_rate"].to_dict()

    # Map global rating
    df["opponent_global_rating"] = df["opponent_id"].map(global_map).fillna(50.0)
    df["opponent_global_block_rate"] = df["opponent_id"].map(global_block_map).fillna(0.5)

    # Map zone-specific ratings
    # Need to map based on opponent_id AND zone_letter
    zone_profiles = profiles_df[profiles_df["zone_id"].notna()].copy()

    # Create zone letter from macro_zone_start
    zone_letter_map = {
        7: "A",
        8: "B",
        9: "B",
        4: "C",
        5: "D",
        6: "D",
        1: "E",
        2: "F",
        3: "F",
    }
    df["zone_letter"] = df["macro_zone_start"].map(zone_letter_map)

    # Create lookup key
    zone_profiles["lookup_key"] = (
        zone_profiles["team_id"].astype(str) + "_" + zone_profiles["zone_id"]
    )
    zone_rating_map = zone_profiles.set_index("lookup_key")["zone_rating"].to_dict()
    zone_block_map = zone_profiles.set_index("lookup_key")["block_rate"].to_dict()

    df["lookup_key"] = df["opponent_id"].astype(str) + "_" + df["zone_letter"]
    df["opponent_zone_rating"] = (
        df["lookup_key"].map(zone_rating_map).fillna(df["opponent_global_rating"])
    )
    df["opponent_zone_block_rate"] = (
        df["lookup_key"].map(zone_block_map).fillna(df["opponent_global_block_rate"])
    )

    # Clean up
    df = df.drop(columns=["lookup_key"])

    # Derived features
    df["opponent_rating_diff"] = df["opponent_zone_rating"] - df["opponent_global_rating"]
    df["opponent_is_strong"] = df["opponent_global_rating"] < 48  # Below average = strong
    df["opponent_is_weak"] = df["opponent_global_rating"] > 52  # Above average = weak

    logger.info("  Added opponent features")
    return df


def add_game_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add game context features.

    Args:
        df: Progressions DataFrame

    Returns:
        DataFrame with game context features added
    """
    logger.info("Adding game context features...")

    df = df.copy()

    # Time features
    df["minute_normalized"] = df["minute"] / 90.0
    df["minute_bucket"] = pd.cut(
        df["minute"],
        bins=[0, 15, 30, 45, 60, 75, 90, 120],
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
        right=False,
    )

    # Period features
    df["is_first_half"] = df["period"] == 1
    df["is_second_half"] = df["period"] == 2
    df["is_extra_time"] = df["period"] > 2

    # Late game indicator
    df["is_late_game"] = df["minute"] >= 75
    df["is_very_late"] = df["minute"] >= 85

    # Early game
    df["is_early_game"] = df["minute"] <= 15

    logger.info("  Added game context features")
    return df


def add_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add zone-based features.

    Args:
        df: Progressions DataFrame

    Returns:
        DataFrame with zone features added
    """
    logger.info("Adding zone features...")

    df = df.copy()

    # Zone characteristics
    zone_names = {
        1: "DEF_CENTRAL",
        2: "DEF_WIDE_L",
        3: "DEF_WIDE_R",
        4: "MID_CENTRAL",
        5: "MID_WIDE_L",
        6: "MID_WIDE_R",
        7: "ATT_CENTRAL",
        8: "ATT_WIDE_L",
        9: "ATT_WIDE_R",
    }

    df["start_zone_name"] = df["macro_zone_start"].map(zone_names)
    df["end_zone_name"] = df["macro_zone_end"].map(zone_names)

    # Pitch third
    df["start_third"] = df["macro_zone_start"].apply(
        lambda z: "DEF" if z <= 3 else ("MID" if z <= 6 else "ATT")
    )
    df["end_third"] = df["macro_zone_end"].apply(
        lambda z: "DEF" if z <= 3 else ("MID" if z <= 6 else "ATT")
    )

    # Central vs Wide
    df["start_is_central"] = df["macro_zone_start"].isin([1, 4, 7])
    df["end_is_central"] = df["macro_zone_end"].isin([1, 4, 7])

    # Zone transitions
    df["zone_changed"] = df["macro_zone_start"] != df["macro_zone_end"]
    df["moved_to_att_third"] = (df["start_third"] != "ATT") & (df["end_third"] == "ATT")
    df["moved_central_to_central"] = df["start_is_central"] & df["end_is_central"]
    df["moved_wide_to_central"] = ~df["start_is_central"] & df["end_is_central"]

    logger.info("  Added zone features")
    return df


def add_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pressure-related features.

    Args:
        df: Progressions DataFrame

    Returns:
        DataFrame with pressure features added
    """
    logger.info("Adding pressure features...")

    df = df.copy()

    # Basic pressure indicator
    df["under_pressure"] = df["under_pressure"].fillna(False)

    # Pressure binary
    df["pressure_flag"] = df["under_pressure"].astype(int)

    logger.info("  Added pressure features")
    return df


def add_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add action-type features.

    Args:
        df: Progressions DataFrame

    Returns:
        DataFrame with action features added
    """
    logger.info("Adding action features...")

    df = df.copy()

    # One-hot encode action type
    df["is_pass"] = (df["action_type"] == "pass").astype(int)
    df["is_carry"] = (df["action_type"] == "carry").astype(int)
    df["is_dribble"] = (df["action_type"] == "dribble").astype(int)

    # Action outcome
    df["action_success"] = ~df["action_outcome"].isin(["Incomplete", "Out", "Pass Offside"])
    df["action_success"] = df["action_success"].fillna(True)  # Carries are successful by default

    logger.info("  Added action features")
    return df


def engineer_cxt_features(
    df: pd.DataFrame,
    opponent_profiles_version: str = "cxt_v1",
) -> pd.DataFrame:
    """Apply full feature engineering pipeline.

    Args:
        df: Raw progressions DataFrame
        opponent_profiles_version: Version tag for opponent profiles

    Returns:
        DataFrame with all CxT features
    """
    logger.info("=" * 60)
    logger.info("CxT Feature Engineering")
    logger.info("=" * 60)

    # Load opponent profiles
    profiles_df = load_opponent_profiles(opponent_profiles_version)

    # Apply feature engineering
    df = add_opponent_features(df, profiles_df)
    df = add_game_context_features(df)
    df = add_zone_features(df)
    df = add_pressure_features(df)
    df = add_action_features(df)

    logger.info("=" * 60)
    logger.info(f"Feature engineering complete: {len(df.columns)} columns")
    logger.info("=" * 60)

    return df


def get_feature_columns() -> dict:
    """Get categorized feature columns for modeling.

    Returns:
        Dictionary with feature categories
    """
    return {
        "target": ["xt_delta"],
        "numeric": [
            "start_xt",
            "end_xt",
            "minute_normalized",
            "opponent_global_rating",
            "opponent_zone_rating",
            "opponent_rating_diff",
        ],
        "binary": [
            "under_pressure",
            "is_progressive",
            "is_into_final_third",
            "is_into_penalty_area",
            "is_pass",
            "is_carry",
            "is_dribble",
            "is_home",
            "is_late_game",
            "is_early_game",
            "is_first_half",
            "is_second_half",
            "start_is_central",
            "end_is_central",
            "zone_changed",
            "moved_to_att_third",
            "opponent_is_strong",
            "opponent_is_weak",
            "action_success",
        ],
        "categorical": [
            "macro_zone_start",
            "macro_zone_end",
            "start_third",
            "end_third",
            "action_type",
        ],
        "identifiers": [
            "event_id",
            "match_id",
            "team_id",
            "player_id",
            "opponent_id",
        ],
    }
