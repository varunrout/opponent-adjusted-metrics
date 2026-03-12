#!/usr/bin/env python
"""
CxA Pipeline Runner.

Runs all cxA data pipelines and saves features to the feature store as parquet files.

Usage:
    python scripts/run_cxa_pipeline.py [--competition-id 3] [--force]

Outputs (feature_store/cxa/):
    - lineups.parquet: Tactical positions from Starting XI
    - passes.parquet: Pass-level data with enrichments
    - shots.parquet: Shot data with key_pass linking
    - pass_sequences.parquet: Passes enriched with sequence attribution
    - possessions.parquet: Possession-level aggregates
    - sequences.parquet: Sequence-level (one row per assist sequence)
    - action_sequences.parquet: Full action chains including carries/dribbles
    - action_sequences_opposition.parquet: Action sequences with opposition context
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from opponent_adjusted.config import settings, ensure_directories
from opponent_adjusted.db.session import get_session
from opponent_adjusted.pipelines.cxa.lineup_data import build_lineup_dataset
from opponent_adjusted.pipelines.cxa.pass_data import build_pass_dataset
from opponent_adjusted.pipelines.cxa.shot_data import build_shot_dataset
from opponent_adjusted.pipelines.cxa.pass_sequences import build_pass_sequences
from opponent_adjusted.pipelines.cxa.possession_data import build_possession_dataset
from opponent_adjusted.pipelines.cxa.sequence_data import build_sequence_dataset
from opponent_adjusted.features.cxa.sequence_builder import build_action_sequences
from opponent_adjusted.features.cxa.opposition_context import build_opposition_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_feature_store_path() -> Path:
    """Get the cxA feature store path."""
    return settings.feature_store_path / "cxa"


def save_parquet(df, name: str, output_dir: Path) -> Path:
    """Save DataFrame as parquet with metadata."""
    filepath = output_dir / f"{name}.parquet"
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info(f"Saved {name}: {len(df):,} rows -> {filepath}")
    return filepath


def run_pipeline(competition_id: int = None, force: bool = False) -> dict:
    """
    Run the full cxA pipeline.
    
    Args:
        competition_id: Filter to specific competition (None = all competitions)
        force: Overwrite existing files
        
    Returns:
        Dictionary of output file paths
    """
    ensure_directories()
    output_dir = get_feature_store_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("CxA PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Competition ID: {competition_id or 'ALL'}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Database: {settings.database_url}")
    logger.info("=" * 70)
    
    outputs = {}
    
    with get_session() as session:
        # 1. Lineups
        logger.info("\n[1/8] Building lineup dataset...")
        lineups_df = build_lineup_dataset(session, competition_id=competition_id)
        outputs["lineups"] = save_parquet(lineups_df, "lineups", output_dir)
        
        # 2. Passes
        logger.info("\n[2/8] Building pass dataset...")
        passes_df = build_pass_dataset(session, competition_id=competition_id)
        
        # Add xT features if available
        try:
            from opponent_adjusted.features.cxt.xt_model import add_xt_features
            passes_df = add_xt_features(passes_df)
            logger.info("Added xT features to passes")
        except ImportError:
            logger.warning("xT features not available")
        
        outputs["passes"] = save_parquet(passes_df, "passes", output_dir)
        
        # 3. Shots
        logger.info("\n[3/8] Building shot dataset...")
        shots_df = build_shot_dataset(session, competition_id=competition_id)
        outputs["shots"] = save_parquet(shots_df, "shots", output_dir)
        
        # 4. Pass sequences (attribute passes to shots)
        logger.info("\n[4/8] Building pass sequences...")
        pass_sequences_df = build_pass_sequences(passes_df, shots_df, k=3)
        outputs["pass_sequences"] = save_parquet(pass_sequences_df, "pass_sequences", output_dir)
        
        # 5. Possessions
        logger.info("\n[5/8] Building possession dataset...")
        possessions_df = build_possession_dataset(pass_sequences_df, shots_df)
        outputs["possessions"] = save_parquet(possessions_df, "possessions", output_dir)
        
        # 6. Sequences (sequence-level, one row per assist sequence)
        logger.info("\n[6/8] Building sequence dataset...")
        sequences_df = build_sequence_dataset(pass_sequences_df, shots_df, k=3)
        outputs["sequences"] = save_parquet(sequences_df, "sequences", output_dir)
        
        # 7. Action sequences (full chains including carries/dribbles)
        logger.info("\n[7/8] Building action sequences (passes + carries + dribbles)...")
        action_sequences_df = build_action_sequences(session, competition_id=competition_id, k=5)
        outputs["action_sequences"] = save_parquet(action_sequences_df, "action_sequences", output_dir)
        
        # 8. Opposition context (enrich action sequences with opponent metrics)
        logger.info("\n[8/8] Building opposition context features...")
        # Load opponent profiles from CxG feature store
        cxg_profiles_path = settings.feature_store_path / "cxg" / "opponent_profiles.parquet"
        if cxg_profiles_path.exists():
            import pandas as pd
            opponent_profiles_df = pd.read_parquet(cxg_profiles_path)
            logger.info(f"Loaded {len(opponent_profiles_df):,} opponent profiles")
        else:
            import pandas as pd
            opponent_profiles_df = pd.DataFrame()
            logger.warning("Opponent profiles not found, run CxG pipeline first")
        
        opposition_df = build_opposition_context(
            action_sequences_df=action_sequences_df,
            opponent_profiles_df=opponent_profiles_df,
            session=session,
        )
        outputs["action_sequences_opposition"] = save_parquet(
            opposition_df, "action_sequences_opposition", output_dir
        )
    
    # Save metadata
    metadata = {
        "pipeline": "cxa",
        "competition_id": competition_id,
        "created_at": datetime.now().isoformat(),
        "files": {k: str(v) for k, v in outputs.items()},
        "row_counts": {
            "lineups": len(lineups_df),
            "passes": len(passes_df),
            "shots": len(shots_df),
            "pass_sequences": len(pass_sequences_df),
            "possessions": len(possessions_df),
            "sequences": len(sequences_df),
            "action_sequences": len(action_sequences_df) if not action_sequences_df.empty else 0,
            "action_sequences_opposition": len(opposition_df) if not opposition_df.empty else 0,
        },
    }
    
    import json
    metadata_path = output_dir / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Output files:")
    for name, path in outputs.items():
        logger.info(f"  {name}: {path}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run cxA pipeline")
    parser.add_argument(
        "--competition-id", "-c",
        type=int,
        default=None,
        help="Competition ID to process (default: None = all competitions)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing files"
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            competition_id=args.competition_id,
            force=args.force,
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
