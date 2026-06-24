#!/usr/bin/env python
"""
CxT Feature Engineering Runner.

Applies contextual feature engineering to the progressions dataset,
adding opponent context, game state, and other modeling features.

Usage:
    python scripts/run_cxt_features.py [--opponent-version cxt_v1]

Outputs (feature_store/cxt/):
    - progressions_featured.parquet: Progressions with all CxT features
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

from opponent_adjusted.config import settings
from opponent_adjusted.features.cxt.cxt_features import (
    engineer_cxt_features,
    get_feature_columns,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_feature_store_path() -> Path:
    """Get the CxT feature store path."""
    return settings.feature_store_path / "cxt"


def run_feature_engineering(
    opponent_version: str = "cxt_v1",
    force: bool = False,
) -> dict:
    """
    Run feature engineering on progressions data.

    Args:
        opponent_version: Version tag for opponent profiles
        force: Overwrite existing output

    Returns:
        Dictionary with statistics
    """
    output_dir = get_feature_store_path()

    # Check input exists
    input_path = output_dir / "progressions.parquet"
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        logger.error("Run run_cxt_pipeline.py first")
        return {"error": "missing input"}

    # Check output
    output_path = output_dir / "progressions_featured.parquet"
    if output_path.exists() and not force:
        logger.warning(f"Output exists: {output_path}")
        logger.warning("Use --force to overwrite")
        return {"skipped": True}

    logger.info("=" * 70)
    logger.info("CxT FEATURE ENGINEERING")
    logger.info("=" * 70)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Opponent profiles version: {opponent_version}")
    logger.info("=" * 70)

    start_time = datetime.now()

    # Load progressions
    logger.info("\nLoading progressions...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Apply feature engineering
    logger.info("\nApplying feature engineering...")
    df_featured = engineer_cxt_features(df, opponent_version)

    # Save output
    logger.info("\nSaving featured dataset...")
    df_featured.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved: {len(df_featured):,} rows, {len(df_featured.columns)} columns")

    # Collect statistics
    feature_cols = get_feature_columns()
    stats = {
        "total_rows": len(df_featured),
        "total_columns": len(df_featured.columns),
        "numeric_features": len(feature_cols["numeric"]),
        "binary_features": len(feature_cols["binary"]),
        "categorical_features": len(feature_cols["categorical"]),
        "opponent_profiles_version": opponent_version,
        "mean_opponent_rating": float(df_featured["opponent_global_rating"].mean()),
        "mean_xt_delta": float(df_featured["xt_delta"].mean()),
    }

    elapsed = datetime.now() - start_time

    # Save metadata
    metadata = {
        "step": "feature_engineering",
        "version": "1.0.0",
        "run_timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "input": str(input_path),
        "output": str(output_path),
        "statistics": stats,
        "feature_categories": feature_cols,
    }

    metadata_path = output_dir / "features_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total columns: {stats['total_columns']}")
    logger.info(f"  - Numeric: {stats['numeric_features']}")
    logger.info(f"  - Binary: {stats['binary_features']}")
    logger.info(f"  - Categorical: {stats['categorical_features']}")
    logger.info(f"Mean opponent rating: {stats['mean_opponent_rating']:.2f}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 70)

    return {
        "output": str(output_path),
        "statistics": stats,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CxT feature engineering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--opponent-version",
        type=str,
        default="cxt_v1",
        help="Opponent profile version tag (default: cxt_v1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output",
    )

    args = parser.parse_args()

    try:
        result = run_feature_engineering(
            opponent_version=args.opponent_version,
            force=args.force,
        )

        if result.get("error"):
            sys.exit(1)
        if result.get("skipped"):
            sys.exit(0)

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
