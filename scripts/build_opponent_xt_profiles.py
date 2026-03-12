"""Build opponent xT profiles for CxT modeling.

For each defending team, compute:
- Global xT deficit: average xT conceded per possession against them
- Zone xT ratings: xT conceded per macro-zone, with shrinkage

Uses progressions data (passes, carries) to calculate how much xT 
opponents allow through ball progression.

Teams that allow MORE xT are WEAKER defensively.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from opponent_adjusted.db.session import session_scope
from opponent_adjusted.db.models import OpponentDefProfile
from opponent_adjusted.config import settings
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ZoneAgg:
    """Zone-level aggregation for xT concession."""
    total_xt: float = 0.0
    n_actions: int = 0
    positive_xt_count: int = 0  # Number of progressive actions
    
    @property
    def mean_xt(self) -> float:
        return self.total_xt / self.n_actions if self.n_actions > 0 else 0.0
    
    @property
    def progressive_rate(self) -> float:
        return self.positive_xt_count / self.n_actions if self.n_actions > 0 else 0.0


def _shrink(zone_mean: float, n_zone: int, global_mean: float, prior: float = 50.0) -> float:
    """Apply Bayesian shrinkage to stabilize zone estimates."""
    if n_zone <= 0:
        return global_mean
    return (n_zone * zone_mean + prior * global_mean) / (n_zone + prior)


def get_feature_store_path() -> Path:
    """Get the CxT feature store path."""
    return settings.feature_store_path / "cxt"


def build_xt_profiles(version: str = "cxt_v1", force: bool = False) -> None:
    """Build opponent xT profiles from progressions data.
    
    Args:
        version: Version tag for the profiles
        force: Overwrite existing profiles
    """
    # Load progressions data
    progressions_path = get_feature_store_path() / "progressions.parquet"
    if not progressions_path.exists():
        logger.error("Progressions data not found at %s", progressions_path)
        logger.error("Run run_cxt_pipeline.py first")
        return
    
    logger.info("Loading progressions data...")
    df = pd.read_parquet(progressions_path)
    logger.info(f"Loaded {len(df):,} progressions")
    
    # Filter to successful actions (for xT, we care about what got through)
    # Include: completed passes, carries, successful dribbles
    # Exclude: incomplete passes (already penalized in xT delta)
    action_mask = (
        (df["action_type"] == "carry") |  # All carries count
        (df["action_type"] == "dribble") |  # All dribbles count
        ((df["action_type"] == "pass") & ~df["action_outcome"].isin(["Incomplete", "Out", "Pass Offside"]))
    )
    df_successful = df[action_mask].copy()
    logger.info(f"Using {len(df_successful):,} successful actions for xT profiles")
    
    # Map macro-zones to letters (like existing profiles use A-F)
    # Zones 1-9 -> map to approximate A-F
    # ATT zones (7,8,9) -> A, B (close)
    # MID zones (4,5,6) -> C, D (mid)
    # DEF zones (1,2,3) -> E, F (far)
    zone_letter_map = {
        7: "A", 8: "B", 9: "B",  # ATT central/wide
        4: "C", 5: "D", 6: "D",  # MID central/wide
        1: "E", 2: "F", 3: "F",  # DEF central/wide
    }
    df_successful["zone_letter"] = df_successful["macro_zone_start"].map(zone_letter_map)
    
    # Aggregate by opponent (defending team) and zone
    # NOTE: "opponent_id" in progressions is who the acting team is PLAYING AGAINST
    # So xT conceded BY opponent_id = xT gained BY team_id when facing opponent_id
    
    global_agg: Dict[int, ZoneAgg] = defaultdict(ZoneAgg)
    zone_agg: Dict[Tuple[int, str], ZoneAgg] = defaultdict(ZoneAgg)
    
    for _, row in df_successful.iterrows():
        opponent_id = row["opponent_id"]
        if pd.isna(opponent_id):
            continue
        opponent_id = int(opponent_id)
        
        xt_delta = row["xt_delta"]
        is_progressive = row["is_progressive"]
        zone = row["zone_letter"]
        
        # Global aggregation
        g = global_agg[opponent_id]
        g.total_xt += float(xt_delta)
        g.n_actions += 1
        if is_progressive:
            g.positive_xt_count += 1
        
        # Zone aggregation
        if zone:
            zg = zone_agg[(opponent_id, zone)]
            zg.total_xt += float(xt_delta)
            zg.n_actions += 1
            if is_progressive:
                zg.positive_xt_count += 1
    
    logger.info(f"Computed profiles for {len(global_agg)} opponents")
    
    # Write to database
    with session_scope() as session:
        # Check existing profiles
        existing = session.query(OpponentDefProfile).filter_by(version_tag=version).count()
        if existing > 0 and not force:
            logger.warning(f"Found {existing} existing profiles for version {version}")
            logger.warning("Use --force to overwrite")
            return
        
        # Delete existing if forcing
        if existing > 0 and force:
            session.query(OpponentDefProfile).filter_by(version_tag=version).delete()
            logger.info(f"Deleted {existing} existing profiles")
        
        inserted = 0
        
        for team_id, g in global_agg.items():
            # Global profile
            # Rating: Higher = more xT conceded = WORSE defense
            # Scale to 0-100 range (50 = average)
            global_rating = 50 + (g.mean_xt * 10000)  # Scale factor for visibility
            global_rating = np.clip(global_rating, 0, 100)
            
            # Insert global profile
            global_profile = OpponentDefProfile(
                team_id=team_id,
                version_tag=version,
                zone_id=None,  # NULL = global
                global_rating=float(global_rating),
                block_rate=float(1 - g.progressive_rate),  # Inverse of progressive rate
                zone_rating=None,
                shots_sample=g.n_actions,  # Using shots_sample for action count
            )
            session.add(global_profile)
            inserted += 1
            
            # Zone profiles with shrinkage
            for zone in "ABCDEF":
                zg = zone_agg.get((team_id, zone))
                n_zone = zg.n_actions if zg else 0
                
                if zg:
                    zone_mean_xt = zg.mean_xt
                else:
                    zone_mean_xt = g.mean_xt
                
                # Shrink toward global
                shrunk_xt = _shrink(zone_mean_xt, n_zone, g.mean_xt, prior=50.0)
                
                # Convert to rating (0-100)
                zone_rating = 50 + (shrunk_xt * 10000)
                zone_rating = np.clip(zone_rating, 0, 100)
                
                zone_progressive = (zg.progressive_rate if zg else g.progressive_rate)
                
                zone_profile = OpponentDefProfile(
                    team_id=team_id,
                    version_tag=version,
                    zone_id=zone,
                    global_rating=None,
                    block_rate=float(1 - zone_progressive),
                    zone_rating=float(zone_rating),
                    shots_sample=n_zone,
                )
                session.add(zone_profile)
                inserted += 1
        
        session.commit()
        logger.info(f"Inserted {inserted} opponent xT profiles for version {version}")
    
    # Save summary to feature store
    summary_path = get_feature_store_path() / "opponent_xt_profiles_summary.parquet"
    summary_data = []
    for team_id, g in global_agg.items():
        summary_data.append({
            "opponent_id": team_id,
            "n_actions_faced": g.n_actions,
            "total_xt_conceded": g.total_xt,
            "mean_xt_conceded": g.mean_xt,
            "progressive_rate_conceded": g.progressive_rate,
            "version": version,
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_parquet(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Print top/bottom opponents
    summary_df_sorted = summary_df.sort_values("mean_xt_conceded", ascending=False)
    logger.info("\nTop 5 WEAKEST defenses (concede most xT):")
    for _, row in summary_df_sorted.head(5).iterrows():
        logger.info(f"  Team {row['opponent_id']}: {row['mean_xt_conceded']:.6f} xT/action")
    
    logger.info("\nTop 5 STRONGEST defenses (concede least xT):")
    for _, row in summary_df_sorted.tail(5).iterrows():
        logger.info(f"  Team {row['opponent_id']}: {row['mean_xt_conceded']:.6f} xT/action")


def main():
    parser = argparse.ArgumentParser(description="Build opponent xT profiles for CxT")
    parser.add_argument("--version", default="cxt_v1", help="Version tag for profiles")
    parser.add_argument("--force", action="store_true", help="Overwrite existing profiles")
    args = parser.parse_args()
    
    build_xt_profiles(version=args.version, force=args.force)


if __name__ == "__main__":
    main()
