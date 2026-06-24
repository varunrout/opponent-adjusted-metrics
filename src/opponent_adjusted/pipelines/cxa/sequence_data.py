"""Assist sequence data aggregation for CxA analysis.

Creates sequence-level data where each row represents one complete
assist sequence (k passes leading to a shot).

Integrates xT (Expected Threat) features to measure value creation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_sequence_dataset(
    passes_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """Build sequence-level dataset from passes.

    Each row represents one assist sequence ending in a shot.
    Includes all k passes with their individual and aggregated features.

    Args:
        passes_df: Pass-level data with xT features and sequence info
        shots_df: Shot-level data
        k: Maximum passes to trace back (sequence length)

    Returns:
        DataFrame with one row per assist sequence
    """
    logger.info(f"Building assist sequence dataset (k={k})...")

    df = passes_df.copy()

    # Filter to only passes in sequences
    if "sequence_id" not in df.columns:
        raise ValueError("passes_df must have sequence_id column. Run build_pass_sequences first.")

    sequence_passes = df[df["sequence_id"].notna()].copy()

    if sequence_passes.empty:
        logger.warning("No passes found in sequences!")
        return pd.DataFrame()

    logger.info(f"Processing {sequence_passes['sequence_id'].nunique():,} sequences...")

    # Get the shot info for each sequence
    shot_info = _get_shot_info(shots_df)

    # Build wide-format sequence data
    sequences = []

    for seq_id, group in sequence_passes.groupby("sequence_id"):
        seq_data = _build_single_sequence(group, k, seq_id)
        if seq_data:
            sequences.append(seq_data)

    if not sequences:
        logger.warning("No valid sequences built!")
        return pd.DataFrame()

    sequence_df = pd.DataFrame(sequences)

    # Merge with shot info using shot_id
    sequence_df = sequence_df.merge(
        shot_info,
        on="shot_id",
        how="left",
        suffixes=("", "_shot"),
    )

    # Add derived features
    sequence_df = _add_derived_features(sequence_df, k)

    logger.info(f"Built sequence dataset: {len(sequence_df):,} sequences")
    _log_summary(sequence_df)

    return sequence_df


def _get_shot_info(shots_df: pd.DataFrame) -> pd.DataFrame:
    """Extract key shot info for sequence linking."""
    columns = [
        "shot_id",
        "match_id",
        "team_id",
        "player_id",
        "player_name",
        "minute",
        "second",
        "start_x",
        "start_y",
        "statsbomb_xg",
        "outcome",
        "body_part",
        "technique",
        "first_time",
        "is_goal",
    ]

    available = [c for c in columns if c in shots_df.columns]

    return (
        shots_df[available]
        .copy()
        .rename(
            columns={
                "player_id": "shooter_id",
                "player_name": "shooter_name",
                "minute": "shot_minute",
                "second": "shot_second",
                "start_x": "shot_x",
                "start_y": "shot_y",
                "statsbomb_xg": "xg",
                "outcome": "shot_outcome",
                "body_part": "shot_body_part",
                "technique": "shot_technique",
                "first_time": "shot_first_time",
            }
        )
    )


def _build_single_sequence(
    group: pd.DataFrame,
    k: int,
    seq_id: str,
) -> Optional[dict]:
    """Build a single sequence row from grouped passes."""
    # Sort by passes_to_shot (1 = key pass, 2 = second assist, etc.)
    group = group.sort_values("passes_to_shot")

    seq_data = {
        "sequence_id": seq_id,
        "match_id": group["match_id"].iloc[0],
        "team_id": group["team_id"].iloc[0] if "team_id" in group.columns else None,
        "competition_id": (
            group["competition_id"].iloc[0] if "competition_id" in group.columns else None
        ),
        "possession": group["possession"].iloc[0] if "possession" in group.columns else None,
        "num_passes_in_sequence": len(group),
        # Add shot_id for linking - stored in sequence_shot_id column
        "shot_id": (
            group["sequence_shot_id"].iloc[0] if "sequence_shot_id" in group.columns else None
        ),
    }

    # Add per-pass features
    for _, row in group.iterrows():
        pos = int(row["passes_to_shot"])
        prefix = f"pass{pos}_"

        # Basic info
        seq_data[f"{prefix}id"] = row.get("pass_id")
        seq_data[f"{prefix}player_id"] = row.get("player_id")
        # Handle both player_name and passer_name column names
        seq_data[f"{prefix}player_name"] = row.get("player_name") or row.get("passer_name")
        seq_data[f"{prefix}recipient_id"] = row.get("recipient_id")
        seq_data[f"{prefix}recipient_name"] = row.get("recipient_name")

        # Location
        seq_data[f"{prefix}start_x"] = row.get("start_x")
        seq_data[f"{prefix}start_y"] = row.get("start_y")
        seq_data[f"{prefix}end_x"] = row.get("end_x")
        seq_data[f"{prefix}end_y"] = row.get("end_y")

        # xT features
        seq_data[f"{prefix}start_xt"] = row.get("start_xt", 0)
        seq_data[f"{prefix}end_xt"] = row.get("end_xt", 0)
        seq_data[f"{prefix}xt_delta"] = row.get("xt_delta", 0)

        # Pass type features
        seq_data[f"{prefix}is_progressive"] = row.get("is_progressive", False)
        seq_data[f"{prefix}is_into_box"] = row.get("is_into_box", False)
        seq_data[f"{prefix}is_cross"] = row.get("is_cross", False)
        seq_data[f"{prefix}is_through_ball"] = row.get("is_through_ball", False)

        # xA from sequence
        seq_data[f"{prefix}sequence_xA"] = row.get("sequence_xA", 0)

        # Timing
        seq_data[f"{prefix}minute"] = row.get("minute")
        seq_data[f"{prefix}second"] = row.get("second")

    # Fill missing positions with None
    for pos in range(1, k + 1):
        prefix = f"pass{pos}_"
        if f"{prefix}id" not in seq_data:
            for col in [
                "id",
                "player_id",
                "player_name",
                "recipient_id",
                "recipient_name",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "start_xt",
                "end_xt",
                "xt_delta",
                "is_progressive",
                "is_into_box",
                "is_cross",
                "is_through_ball",
                "sequence_xA",
                "minute",
                "second",
            ]:
                seq_data[f"{prefix}{col}"] = None

    return seq_data


def _add_derived_features(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Add derived/aggregated features to sequence data."""
    # Total xT gained across sequence
    xt_cols = [f"pass{i}_xt_delta" for i in range(1, k + 1)]
    available_xt = [c for c in xt_cols if c in df.columns]

    if available_xt:
        df["total_xt_delta"] = df[available_xt].sum(axis=1, skipna=True)
        df["mean_xt_delta"] = df[available_xt].mean(axis=1, skipna=True)
        df["max_xt_delta"] = df[available_xt].max(axis=1, skipna=True)

    # Total sequence xA
    xa_cols = [f"pass{i}_sequence_xA" for i in range(1, k + 1)]
    available_xa = [c for c in xa_cols if c in df.columns]

    if available_xa:
        df["total_sequence_xA"] = df[available_xa].sum(axis=1, skipna=True)

    # Count special pass types
    for pass_type in ["progressive", "into_box", "cross", "through_ball"]:
        type_cols = [f"pass{i}_is_{pass_type}" for i in range(1, k + 1)]
        available_type = [c for c in type_cols if c in df.columns]
        if available_type:
            df[f"num_{pass_type}"] = df[available_type].sum(axis=1, skipna=True)

    # Spatial progression (from first pass start to shot)
    if "pass1_end_x" in df.columns and "shot_x" in df.columns:
        # Get the starting position of the earliest pass in sequence
        df["sequence_start_x"] = None
        df["sequence_start_y"] = None

        for i in range(k, 0, -1):
            x_col = f"pass{i}_start_x"
            y_col = f"pass{i}_start_y"
            if x_col in df.columns:
                mask = df[x_col].notna() & df["sequence_start_x"].isna()
                df.loc[mask, "sequence_start_x"] = df.loc[mask, x_col]
                df.loc[mask, "sequence_start_y"] = df.loc[mask, y_col]

        # Distance from sequence start to shot
        df["sequence_total_x_progression"] = df["shot_x"] - df["sequence_start_x"]
        df["sequence_total_distance"] = np.sqrt(
            (df["shot_x"] - df["sequence_start_x"]) ** 2
            + (df["shot_y"] - df["sequence_start_y"]) ** 2
        )

    # Duration (from first pass to shot)
    if "pass1_minute" in df.columns and "shot_minute" in df.columns:
        # Find earliest pass time
        df["sequence_start_minute"] = None
        df["sequence_start_second"] = None

        for i in range(k, 0, -1):
            min_col = f"pass{i}_minute"
            sec_col = f"pass{i}_second"
            if min_col in df.columns:
                mask = df[min_col].notna() & df["sequence_start_minute"].isna()
                df.loc[mask, "sequence_start_minute"] = df.loc[mask, min_col]
                df.loc[mask, "sequence_start_second"] = df.loc[mask, sec_col]

        df["sequence_duration_seconds"] = (df["shot_minute"] - df["sequence_start_minute"]) * 60 + (
            df["shot_second"] - df["sequence_start_second"]
        )

    # Zone classification
    if "sequence_start_x" in df.columns:
        df["sequence_start_zone"] = df["sequence_start_x"].apply(_get_third)

    if "shot_x" in df.columns:
        df["shot_zone"] = df["shot_x"].apply(_get_third)

    # Outcome category
    if "shot_outcome" in df.columns:
        df["is_goal"] = df["shot_outcome"] == "Goal"
        df["is_on_target"] = df["shot_outcome"].isin(["Goal", "Saved", "Saved Off Target"])

    return df


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


def _log_summary(df: pd.DataFrame) -> None:
    """Log summary statistics."""
    if "total_xt_delta" in df.columns:
        logger.info(f"  Mean total xT delta: {df['total_xt_delta'].mean():.4f}")

    if "total_sequence_xA" in df.columns:
        logger.info(f"  Mean total sequence xA: {df['total_sequence_xA'].mean():.4f}")

    if "is_goal" in df.columns:
        goals = df["is_goal"].sum()
        logger.info(f"  Sequences ending in goal: {goals:,} ({100*goals/len(df):.1f}%)")

    if "xg" in df.columns:
        logger.info(f"  Mean xG of shots: {df['xg'].mean():.4f}")


def save_sequence_dataset(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "assist_sequences.csv",
) -> None:
    """Save sequence dataset to CSV."""
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved sequence dataset to {output_path}")


# === Player-level aggregation ===


def aggregate_player_sequences(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Aggregate sequence data to player level.

    Creates stats for each player's involvement in sequences:
    - As key passer (position 1)
    - As second assist (position 2)
    - As third assist (position 3)
    """
    logger.info("Aggregating sequences by player...")

    player_stats = []

    for pos in range(1, k + 1):
        player_col = f"pass{pos}_player_id"
        name_col = f"pass{pos}_player_name"
        xt_col = f"pass{pos}_xt_delta"
        xa_col = f"pass{pos}_sequence_xA"

        if player_col not in df.columns:
            continue

        subset = df[df[player_col].notna()].copy()

        if subset.empty:
            continue

        agg = (
            subset.groupby([player_col, name_col])
            .agg(
                count=(player_col, "count"),
                total_xt=(
                    pd.NamedAgg(column=xt_col, aggfunc="sum")
                    if xt_col in df.columns
                    else (player_col, "count")
                ),
                total_xA=(
                    pd.NamedAgg(column=xa_col, aggfunc="sum")
                    if xa_col in df.columns
                    else (player_col, "count")
                ),
                goals_assisted=(
                    ("is_goal", lambda x: x.sum() if pos == 1 else 0)
                    if "is_goal" in df.columns
                    else (player_col, "count")
                ),
            )
            .reset_index()
        )

        agg["position_in_sequence"] = pos
        agg = agg.rename(columns={player_col: "player_id", name_col: "player_name"})

        player_stats.append(agg)

    if not player_stats:
        return pd.DataFrame()

    result = pd.concat(player_stats, ignore_index=True)

    # Pivot to wide format
    result_wide = result.pivot_table(
        index=["player_id", "player_name"],
        columns="position_in_sequence",
        values=["count", "total_xt", "total_xA", "goals_assisted"],
        fill_value=0,
    ).reset_index()

    # Flatten column names
    result_wide.columns = [
        f"{col[0]}_pos{col[1]}" if isinstance(col, tuple) and col[1] != "" else col[0]
        for col in result_wide.columns
    ]

    logger.info(f"Aggregated stats for {len(result_wide):,} players")

    return result_wide
