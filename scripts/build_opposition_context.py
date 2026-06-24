#!/usr/bin/env python
"""
Build Opposition Context Features for Action Sequences.

Enriches action sequences with opposition defensive metrics.

Usage:
    python scripts/build_opposition_context.py

Outputs (feature_store/cxa/):
    - action_sequences_opposition.parquet: Action sequences with opposition context
"""

from __future__ import annotations

import logging
import sys
import pandas as pd
from pathlib import Path

from opponent_adjusted.config import settings, ensure_directories
from opponent_adjusted.db.session import get_session
from opponent_adjusted.features.cxa.opposition_context import build_opposition_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_cxa_path() -> Path:
    """Get the cxA feature store path."""
    return settings.feature_store_path / "cxa"


def get_cxg_path() -> Path:
    """Get the cxG feature store path."""
    return settings.feature_store_path / "cxg"


def main():
    """Run opposition context feature builder."""
    logger.info("=" * 70)
    logger.info("BUILDING OPPOSITION CONTEXT FEATURES")
    logger.info("=" * 70)

    ensure_directories()
    cxa_path = get_cxa_path()
    cxg_path = get_cxg_path()

    # Load action sequences
    action_seq_path = cxa_path / "action_sequences.parquet"
    if not action_seq_path.exists():
        raise FileNotFoundError(f"Action sequences not found at {action_seq_path}")

    logger.info(f"\nLoading action sequences from {action_seq_path}...")
    action_sequences_df = pd.read_parquet(action_seq_path)
    logger.info(f"Loaded {len(action_sequences_df):,} action sequences")

    # Load opponent profiles
    profiles_path = cxg_path / "opponent_profiles.parquet"
    logger.info(f"\nLoading opponent profiles from {profiles_path}...")
    if profiles_path.exists():
        opponent_profiles_df = pd.read_parquet(profiles_path)
        logger.info(f"Loaded {len(opponent_profiles_df):,} opponent profiles")
    else:
        logger.warning("Opponent profiles not found, using empty DataFrame")
        opponent_profiles_df = pd.DataFrame()

    # Build opposition context
    logger.info("\n" + "-" * 50)
    with get_session() as session:
        df = build_opposition_context(
            action_sequences_df=action_sequences_df,
            opponent_profiles_df=opponent_profiles_df,
            session=session,
        )

    # Save
    output_path = cxa_path / "action_sequences_opposition.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"\nSaved: {output_path}")
    logger.info(f"Shape: {df.shape}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("OPPOSITION CONTEXT SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total sequences:           {len(df):,}")
    logger.info(f"With opponent profile:     {df['opponent_global_rating'].notna().sum():,}")
    logger.info(f"With zone rating:          {df['opponent_zone_rating'].notna().sum():,}")
    logger.info(f"Under pressure (any):      {(df['sequence_pressure_count'] > 0).sum():,}")
    logger.info(f"Key action pressured:      {df['key_action_under_pressure'].sum():,}")
    logger.info(f"Opponent chasing (open):   {df['opponent_chasing'].sum():,}")
    logger.info(f"Opponent protecting:       {df['opponent_protecting'].sum():,}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
