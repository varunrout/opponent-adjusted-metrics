#!/usr/bin/env python
"""
CxG Pipeline Runner.

Runs cxG data pipelines and saves features to the feature store as parquet files.

Usage:
    python scripts/run_cxg_pipeline.py [--competition-id 3]

Outputs (feature_store/cxg/):
    - shots.parquet: Shot-level data with geometric features
    - shot_features.parquet: Enriched shot features (context, game state)
    - opponent_profiles.parquet: Opponent defensive profiles (global + zone)
"""

from __future__ import annotations

import argparse
import logging
import sys

from opponent_adjusted.pipelines.cxg import run_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run cxG pipeline")
    parser.add_argument(
        "--competition-id", "-c",
        type=int,
        default=None,
        help="Competition ID to process (default: None = all competitions)"
    )

    args = parser.parse_args()

    try:
        run_pipeline(competition_id=args.competition_id)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
