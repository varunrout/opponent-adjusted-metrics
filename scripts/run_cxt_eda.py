#!/usr/bin/env python
"""
CxT EDA Runner.

Runs comprehensive Exploratory Data Analysis on the CxT feature store.

Usage:
    python scripts/run_cxt_eda.py [--input PATH] [--output PATH]

Outputs (outputs/analysis/cxt/eda/):
    - eda_report.md: Summary report
    - csv/: Tabular analysis outputs
    - plots/: Visualizations
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from opponent_adjusted.analysis.cxt.eda import run_full_eda

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CxT EDA analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to progressions parquet (default: feature_store/cxt/progressions.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: outputs/analysis/cxt/eda/)",
    )
    
    args = parser.parse_args()
    
    try:
        results = run_full_eda(
            progressions_path=args.input,
            output_dir=args.output,
        )
        
        # Print key metrics
        logger.info("\n" + "=" * 70)
        logger.info("KEY METRICS SUMMARY")
        logger.info("=" * 70)
        
        if "phase0" in results:
            p0 = results["phase0"]
            logger.info(f"Total Actions: {p0.get('total_rows', 0):,}")
            logger.info(f"xT Delta Mean: {p0.get('xt_stats', {}).get('xt_delta', {}).get('mean', 0):.6f}")
        
        if "phase1" in results:
            p1 = results["phase1"]
            logger.info(f"Pressure %%: {p1.get('pressure_pct', 0):.1f}%%")
            logger.info(f"Pressure xT Effect: {p1.get('pressure_xt_diff', 0):.6f}")
        
        if "phase2" in results:
            p2 = results["phase2"]
            logger.info(f"Pass Completion: {p2.get('pass_completion_rate', 0)*100:.1f}%%")
        
        if "phase3" in results:
            p3 = results["phase3"]
            if "pressure_tier_xt_signal" in p3:
                signal = p3["pressure_tier_xt_signal"]
                logger.info(f"Opponent Pressure Signal: {signal:.6f}")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"EDA failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
