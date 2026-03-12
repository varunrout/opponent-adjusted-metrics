"""Phase 0: Data Overview for CxT EDA.

Provides dataset summary, null analysis, and schema validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def run_phase0_eda(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """Run Phase 0: Data Overview analysis.
    
    Args:
        df: Progressions DataFrame
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("=" * 60)
    logger.info("Phase 0: Data Overview")
    logger.info("=" * 60)
    
    results = {}
    
    # Basic stats
    results["total_rows"] = len(df)
    results["total_columns"] = len(df.columns)
    results["columns"] = list(df.columns)
    
    # Action type distribution
    action_counts = df["action_type"].value_counts().to_dict()
    results["action_types"] = action_counts
    logger.info(f"Total actions: {len(df):,}")
    for action, count in action_counts.items():
        logger.info(f"  - {action}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Null analysis
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    null_df = pd.DataFrame({
        "column": null_counts.index,
        "null_count": null_counts.values,
        "null_pct": null_pct.values
    }).sort_values("null_pct", ascending=False)
    
    # Filter to columns with nulls
    null_df_with_nulls = null_df[null_df["null_count"] > 0]
    results["null_analysis"] = null_df_with_nulls.to_dict("records")
    
    logger.info(f"\nNull Analysis ({len(null_df_with_nulls)} columns with nulls):")
    for _, row in null_df_with_nulls.head(10).iterrows():
        logger.info(f"  - {row['column']}: {row['null_count']:,} ({row['null_pct']:.1f}%)")
    
    # Save null analysis
    null_csv = output_dir / "csv" / "null_analysis.csv"
    null_csv.parent.mkdir(parents=True, exist_ok=True)
    null_df.to_csv(null_csv, index=False)
    
    # Unique values for categorical columns
    categorical_cols = ["action_type", "action_outcome", "under_pressure"]
    results["categorical_uniques"] = {}
    for col in categorical_cols:
        if col in df.columns:
            results["categorical_uniques"][col] = df[col].nunique()
    
    # Match/Team/Player counts
    results["unique_matches"] = df["match_id"].nunique()
    results["unique_teams"] = df["team_id"].nunique()
    results["unique_players"] = df["player_id"].nunique()
    
    logger.info(f"\nUnique entities:")
    logger.info(f"  - Matches: {results['unique_matches']:,}")
    logger.info(f"  - Teams: {results['unique_teams']:,}")
    logger.info(f"  - Players: {results['unique_players']:,}")
    
    # xT value ranges
    xt_stats = {
        "start_xt": {
            "min": float(df["start_xt"].min()),
            "max": float(df["start_xt"].max()),
            "mean": float(df["start_xt"].mean()),
            "std": float(df["start_xt"].std()),
        },
        "end_xt": {
            "min": float(df["end_xt"].min()),
            "max": float(df["end_xt"].max()),
            "mean": float(df["end_xt"].mean()),
            "std": float(df["end_xt"].std()),
        },
        "xt_delta": {
            "min": float(df["xt_delta"].min()),
            "max": float(df["xt_delta"].max()),
            "mean": float(df["xt_delta"].mean()),
            "std": float(df["xt_delta"].std()),
        },
    }
    results["xt_stats"] = xt_stats
    
    logger.info(f"\nxT Statistics:")
    logger.info(f"  - start_xt: mean={xt_stats['start_xt']['mean']:.4f}, std={xt_stats['start_xt']['std']:.4f}")
    logger.info(f"  - end_xt: mean={xt_stats['end_xt']['mean']:.4f}, std={xt_stats['end_xt']['std']:.4f}")
    logger.info(f"  - xt_delta: mean={xt_stats['xt_delta']['mean']:.6f}, std={xt_stats['xt_delta']['std']:.4f}")
    
    # Zone distribution
    zone_counts = df["macro_zone_start"].value_counts().sort_index()
    results["zone_distribution"] = zone_counts.to_dict()
    
    logger.info(f"\nMacro-Zone Distribution (start):")
    for zone, count in zone_counts.items():
        logger.info(f"  - Zone {zone}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Save zone distribution
    zone_df = pd.DataFrame({
        "zone": zone_counts.index,
        "count": zone_counts.values,
        "pct": (zone_counts / len(df) * 100).values
    })
    zone_csv = output_dir / "csv" / "zone_distribution.csv"
    zone_df.to_csv(zone_csv, index=False)
    
    logger.info(f"\n✓ Phase 0 complete")
    return results
