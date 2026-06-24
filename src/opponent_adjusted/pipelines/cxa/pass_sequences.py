"""Pass sequence extraction for CxA analysis.

Builds chain-based pass sequences by linking passes where:
- Pass A's recipient becomes Pass B's passer
- Within same match, team, and possession
- Temporal ordering preserved

Supports k=3 attribution (key pass + 2 preceding passes).
"""

from __future__ import annotations

import logging
import uuid

import pandas as pd

logger = logging.getLogger(__name__)


def build_pass_sequences(
    passes_df: pd.DataFrame,
    shots_df: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """Build pass sequences and enrich passes with sequence information.

    For each shot with a key_pass_id, traces back k passes in the chain:
    - k=1: Key pass (direct assist)
    - k=2: Second assist (pass to key passer)
    - k=3: Third assist (pass to second assister)

    Args:
        passes_df: Pass dataset from build_pass_dataset()
        shots_df: Shot dataset with key_pass_id column
        k: Number of passes to trace back (default 3)

    Returns:
        Enriched passes DataFrame with sequence columns
    """
    logger.info(f"Building pass sequences with k={k}...")

    # Initialize new columns
    df = passes_df.copy()
    df["sequence_id"] = None
    df["passes_to_shot"] = None  # 1=key pass, 2=second assist, 3=third assist
    df["is_key_pass"] = False
    df["is_second_assist"] = False
    df["is_third_assist"] = False
    df["is_in_shot_sequence"] = False
    df["sequence_shot_id"] = None
    df["sequence_shot_xg"] = None
    df["sequence_shot_outcome"] = None
    df["sequence_resulted_goal"] = False

    # Get shots with key passes
    shots_with_assist = shots_df[shots_df["key_pass_id"].notna()].copy()
    logger.info(f"Processing {len(shots_with_assist)} shots with key passes...")

    # Create lookup indices for fast chain building
    # Index passes by (match_id, team_id, possession, recipient_id) -> list of passes
    logger.info("Building pass chain index...")
    df["_idx"] = df.index  # Store original index

    # Sort by match, possession, timestamp for temporal ordering
    df = df.sort_values(["match_id", "possession", "minute", "second"])

    # Build index: for each (match, team, possession, player), find passes they made
    # This lets us find "who passed to this player" by looking at passes where recipient = player
    pass_index = _build_pass_index(df)

    sequences_found = 0
    total_passes_attributed = 0

    for _, shot in shots_with_assist.iterrows():
        key_pass_id = shot["key_pass_id"]  # This is a StatsBomb UUID
        shot_id = shot["shot_id"]
        shot_xg = shot.get("statsbomb_xg", None)
        shot_outcome = shot.get("outcome", None)
        is_goal = shot_outcome == "Goal" if shot_outcome else False

        # Find the key pass by statsbomb_event_id (UUID)
        key_pass_mask = df["statsbomb_event_id"] == key_pass_id
        if not key_pass_mask.any():
            # Key pass not found in passes (might be different event type)
            continue

        key_pass_idx = df[key_pass_mask].index[0]
        key_pass = df.loc[key_pass_idx]

        # Generate sequence ID
        seq_id = str(uuid.uuid4())

        # Build the chain backwards from key pass
        chain = _trace_pass_chain(
            df=df,
            pass_index=pass_index,
            start_pass_idx=key_pass_idx,
            match_id=key_pass["match_id"],
            team_id=key_pass["team_id"],
            possession=key_pass["possession"],
            k=k,
        )

        # Attribute sequence info to passes in chain
        for i, pass_idx in enumerate(chain):
            passes_to_shot = i + 1  # 1=key pass, 2=second assist, etc.

            df.loc[pass_idx, "sequence_id"] = seq_id
            df.loc[pass_idx, "passes_to_shot"] = passes_to_shot
            df.loc[pass_idx, "is_in_shot_sequence"] = True
            df.loc[pass_idx, "sequence_shot_id"] = shot_id
            df.loc[pass_idx, "sequence_shot_xg"] = shot_xg
            df.loc[pass_idx, "sequence_shot_outcome"] = shot_outcome
            df.loc[pass_idx, "sequence_resulted_goal"] = is_goal

            if passes_to_shot == 1:
                df.loc[pass_idx, "is_key_pass"] = True
            elif passes_to_shot == 2:
                df.loc[pass_idx, "is_second_assist"] = True
            elif passes_to_shot == 3:
                df.loc[pass_idx, "is_third_assist"] = True

        sequences_found += 1
        total_passes_attributed += len(chain)

    # Clean up temporary column
    df = df.drop(columns=["_idx"])

    # Convert passes_to_shot to int where not null
    df["passes_to_shot"] = df["passes_to_shot"].astype("Int64")  # Nullable int

    logger.info(f"Found {sequences_found} shot sequences")
    logger.info(f"Attributed {total_passes_attributed} passes to sequences")
    logger.info(f"  Key passes (k=1): {df['is_key_pass'].sum()}")
    logger.info(f"  Second assists (k=2): {df['is_second_assist'].sum()}")
    logger.info(f"  Third assists (k=3): {df['is_third_assist'].sum()}")

    return df


def _build_pass_index(df: pd.DataFrame) -> dict:
    """Build index for fast pass chain lookups.

    Creates a dictionary mapping:
        (match_id, team_id, possession, recipient_id) -> list of (pass_idx, passer_id, minute, second)

    This allows us to quickly find "passes that were received by player X"
    so we can then look for the previous pass in the chain.
    """
    index = {}

    for idx, row in df.iterrows():
        recipient_id = row["recipient_id"]
        if pd.isna(recipient_id):
            continue

        key = (row["match_id"], row["team_id"], row["possession"], int(recipient_id))

        if key not in index:
            index[key] = []

        index[key].append(
            {
                "pass_idx": idx,
                "passer_id": row["player_id"],
                "minute": row["minute"],
                "second": row["second"],
            }
        )

    # Sort each list by time (for temporal ordering)
    for key in index:
        index[key].sort(key=lambda x: (x["minute"], x["second"]))

    return index


def _trace_pass_chain(
    df: pd.DataFrame,
    pass_index: dict,
    start_pass_idx: int,
    match_id: int,
    team_id: int,
    possession: int,
    k: int,
) -> list[int]:
    """Trace back the pass chain from a key pass.

    Args:
        df: Passes DataFrame
        pass_index: Pre-built index from _build_pass_index
        start_pass_idx: Index of the key pass
        match_id: Match ID for filtering
        team_id: Team ID for filtering
        possession: Possession number for filtering
        k: Maximum passes to trace back

    Returns:
        List of pass indices in order [key_pass, second_assist, third_assist, ...]
    """
    chain = [start_pass_idx]
    current_pass = df.loc[start_pass_idx]

    for _ in range(k - 1):  # Already have key pass, find k-1 more
        # Who made this pass?
        passer_id = current_pass["player_id"]

        if pd.isna(passer_id):
            break

        # Find passes that were received by this passer (in same possession)
        key = (match_id, team_id, possession, int(passer_id))

        if key not in pass_index:
            break

        candidates = pass_index[key]

        # Find the most recent pass to this player BEFORE the current pass
        current_time = (current_pass["minute"], current_pass["second"])

        prev_pass = None
        for candidate in reversed(candidates):  # Iterate from latest to earliest
            candidate_time = (candidate["minute"], candidate["second"])
            if candidate_time < current_time:
                prev_pass = candidate
                break

        if prev_pass is None:
            break

        # Add to chain
        chain.append(prev_pass["pass_idx"])
        current_pass = df.loc[prev_pass["pass_idx"]]

    return chain


def compute_sequence_xA(
    df: pd.DataFrame,
    decay_factor: float = 0.5,
) -> pd.DataFrame:
    """Compute distributed xA for passes in sequences.

    Distributes the shot xG across passes in the sequence with decay:
    - Key pass (k=1): xG * 1.0
    - Second assist (k=2): xG * decay_factor
    - Third assist (k=3): xG * decay_factor^2

    Args:
        df: Enriched passes DataFrame from build_pass_sequences()
        decay_factor: Decay rate for earlier passes (default 0.5)

    Returns:
        DataFrame with sequence_xA column added
    """
    logger.info(f"Computing sequence xA with decay={decay_factor}...")

    df = df.copy()
    df["sequence_xA"] = 0.0

    # Only process passes in sequences
    in_sequence = df["is_in_shot_sequence"].fillna(False).astype(bool)

    for idx in df[in_sequence].index:
        passes_to_shot = df.loc[idx, "passes_to_shot"]
        shot_xg = df.loc[idx, "sequence_shot_xg"]

        if pd.isna(passes_to_shot) or pd.isna(shot_xg):
            continue

        # Apply decay based on position in sequence
        # k=1 (key pass): decay^0 = 1.0
        # k=2 (second assist): decay^1 = 0.5
        # k=3 (third assist): decay^2 = 0.25
        decay = decay_factor ** (passes_to_shot - 1)
        df.loc[idx, "sequence_xA"] = shot_xg * decay

    # Summary stats
    total_sequence_xa = df["sequence_xA"].sum()
    key_pass_xa = df[df["is_key_pass"]]["sequence_xA"].sum()
    second_assist_xa = df[df["is_second_assist"]]["sequence_xA"].sum()
    third_assist_xa = df[df["is_third_assist"]]["sequence_xA"].sum()

    logger.info(f"Total sequence xA: {total_sequence_xa:.2f}")
    logger.info(f"  Key pass xA: {key_pass_xa:.2f}")
    logger.info(f"  Second assist xA: {second_assist_xa:.2f}")
    logger.info(f"  Third assist xA: {third_assist_xa:.2f}")

    return df


def get_sequence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for each shot sequence.

    Args:
        df: Enriched passes DataFrame from build_pass_sequences()

    Returns:
        DataFrame with one row per sequence
    """
    sequences = (
        df[df["sequence_id"].notna()]
        .groupby("sequence_id")
        .agg(
            {
                "pass_id": "count",  # Number of passes in sequence
                "sequence_shot_id": "first",
                "sequence_shot_xg": "first",
                "sequence_shot_outcome": "first",
                "sequence_resulted_goal": "first",
                "match_id": "first",
                "team_id": "first",
                "possession": "first",
                "sequence_xA": "sum",  # Total xA distributed
            }
        )
        .rename(
            columns={
                "pass_id": "sequence_length",
            }
        )
    )

    return sequences.reset_index()
