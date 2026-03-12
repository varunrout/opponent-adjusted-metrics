#!/usr/bin/env python
"""
CxT Pipeline Runner.

Runs the CxT data pipeline to extract ball progressions (passes, carries, dribbles)
and saves features to the feature store as parquet files.

Usage:
    python scripts/run_cxt_pipeline.py [--competition-id 3] [--force]

Outputs (feature_store/cxt/):
    - progressions.parquet: All ball progression actions with xT values
    - pipeline_metadata.json: Pipeline run metadata
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from opponent_adjusted.config import settings, ensure_directories
from opponent_adjusted.db.session import get_session
from opponent_adjusted.pipelines.cxt.extract_progressions import build_progressions_dataset

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


def save_parquet(df, name: str, output_dir: Path) -> Path:
    """Save DataFrame as parquet with metadata."""
    filepath = output_dir / f"{name}.parquet"
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info(f"Saved {name}: {len(df):,} rows -> {filepath}")
    return filepath


def save_metadata(metadata: dict, output_dir: Path) -> Path:
    """Save pipeline metadata as JSON."""
    filepath = output_dir / "pipeline_metadata.json"
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata -> {filepath}")
    return filepath


def run_pipeline(
    competition_id: int = None,
    force: bool = False,
    include_dribbles: bool = True,
) -> dict:
    """
    Run the CxT data extraction pipeline.
    
    Args:
        competition_id: Filter to specific competition (None = all competitions)
        force: Overwrite existing files
        include_dribbles: Whether to include dribbles (default True)
        
    Returns:
        Dictionary of output file paths and statistics
    """
    ensure_directories()
    output_dir = get_feature_store_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing files
    progressions_file = output_dir / "progressions.parquet"
    if progressions_file.exists() and not force:
        logger.warning(f"Output file exists: {progressions_file}")
        logger.warning("Use --force to overwrite")
        return {"skipped": True, "reason": "file exists"}
    
    logger.info("=" * 70)
    logger.info("CxT PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Competition ID: {competition_id or 'ALL'}")
    logger.info(f"Include Dribbles: {include_dribbles}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Database: {settings.database_url}")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    outputs = {}
    stats = {}
    
    with get_session() as session:
        # Build progressions dataset
        logger.info("\n[1/1] Building progressions dataset...")
        progressions_df = build_progressions_dataset(
            session,
            competition_id=competition_id,
            include_dribbles=include_dribbles,
        )
        
        # Collect statistics
        stats = {
            "total_actions": len(progressions_df),
            "passes": int((progressions_df["action_type"] == "pass").sum()),
            "carries": int((progressions_df["action_type"] == "carry").sum()),
            "dribbles": int((progressions_df["action_type"] == "dribble").sum()) if include_dribbles else 0,
            "matches": int(progressions_df["match_id"].nunique()),
            "teams": int(progressions_df["team_id"].nunique()),
            "players": int(progressions_df["player_id"].nunique()),
            "positive_xt_actions": int((progressions_df["xt_delta"] > 0).sum()),
            "mean_xt_delta": float(progressions_df["xt_delta"].mean()),
            "progressive_actions_pct": float((progressions_df["is_progressive"]).mean() * 100),
        }
        
        # Save output
        outputs["progressions"] = save_parquet(progressions_df, "progressions", output_dir)
    
    # Calculate elapsed time
    elapsed = datetime.now() - start_time
    
    # Save metadata
    metadata = {
        "pipeline": "cxt",
        "version": "1.0.0",
        "run_timestamp": start_time.isoformat(),
        "elapsed_seconds": elapsed.total_seconds(),
        "parameters": {
            "competition_id": competition_id,
            "include_dribbles": include_dribbles,
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "statistics": stats,
    }
    save_metadata(metadata, output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total actions: {stats['total_actions']:,}")
    logger.info(f"  - Passes: {stats['passes']:,}")
    logger.info(f"  - Carries: {stats['carries']:,}")
    logger.info(f"  - Dribbles: {stats['dribbles']:,}")
    logger.info(f"Matches: {stats['matches']:,}")
    logger.info(f"Players: {stats['players']:,}")
    logger.info(f"Progressive actions: {stats['progressive_actions_pct']:.1f}%")
    logger.info(f"Mean xT delta: {stats['mean_xt_delta']:.6f}")
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 70)
    
    return {
        "outputs": outputs,
        "statistics": stats,
        "metadata": metadata,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CxT data extraction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="Filter to specific competition ID (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--no-dribbles",
        action="store_true",
        help="Exclude dribbles from the dataset",
    )
    
    args = parser.parse_args()
    
    try:
        result = run_pipeline(
            competition_id=args.competition_id,
            force=args.force,
            include_dribbles=not args.no_dribbles,
        )
        
        if result.get("skipped"):
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
