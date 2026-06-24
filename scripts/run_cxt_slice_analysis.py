#!/usr/bin/env python
"""
Run CxT Pre-Model Slice Analysis

Validates feature signals before modeling through lift analysis
on success rates across key slice dimensions.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from opponent_adjusted.config import settings
from opponent_adjusted.analysis.cxt.slice_analysis import (
    run_slice_analysis,
    validate_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CxT pre-model slice analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=settings.feature_store_path / "cxt" / "progressions_featured.parquet",
        help="Input featured parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cxt/slices"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--min-signal",
        type=float,
        default=1.05,
        help="Minimum lift ratio to consider signal meaningful",
    )
    args = parser.parse_args()

    start_time = datetime.now()

    logger.info("=" * 70)
    logger.info("CxT PRE-MODEL SLICE ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 70)

    # Load data
    logger.info("\nLoading featured data...")
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return 1

    df = pd.read_parquet(args.input)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Run slice analysis
    logger.info("\nRunning slice analysis...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_slice_analysis(df, output_dir=args.output_dir)

    # Validate signals
    if "signal_summary" in results:
        passed, weak_features = validate_signals(
            results["signal_summary"],
            min_ratio=args.min_signal,
        )

        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 70)

        if passed:
            logger.info("✓ Signal validation PASSED")
            logger.info("  Features show meaningful discriminative signal")
        else:
            logger.warning("⚠ Signal validation concerns")
            logger.warning(f"  Weak signal features: {weak_features}")

        # Show summary statistics
        signal_df = results["signal_summary"]
        logger.info("\nSignal Summary:")
        logger.info(f"  Total features analyzed: {len(signal_df)}")
        logger.info(f"  Avg lift ratio: {signal_df['lift_ratio'].mean():.2f}x")
        logger.info(f"  Max lift ratio: {signal_df['lift_ratio'].max():.2f}x")
        logger.info(f"  Min lift ratio: {signal_df['lift_ratio'].min():.2f}x")

        # Top signals
        logger.info("\nTop 5 discriminative features:")
        top5 = signal_df.nlargest(5, "lift_ratio")
        for _, row in top5.iterrows():
            logger.info(
                f"  {row['feature']}: {row['lift_ratio']:.2f}x lift ratio, "
                f"range={row['lift_range']:.3f}"
            )

    # Generate report
    report_path = args.output_dir / "slice_analysis_report.md"
    _generate_report(results, report_path)
    logger.info(f"\nReport saved: {report_path}")

    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("SLICE ANALYSIS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Elapsed time: {elapsed}")
    logger.info("=" * 70)

    return 0


def _generate_report(results: dict, output_path: Path) -> None:
    """Generate markdown report."""

    lines = [
        "# CxT Pre-Model Slice Analysis Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This analysis validates feature signals before model training by examining ",
        "how success rates vary across different slice dimensions.",
        "",
    ]

    # Signal summary
    if "signal_summary" in results:
        signal_df = results["signal_summary"]
        lines.extend(
            [
                "## Signal Strength Summary",
                "",
                "| Feature | Lift Ratio | Lift Range |",
                "|---------|------------|------------|",
            ]
        )

        for _, row in signal_df.sort_values("lift_ratio", ascending=False).iterrows():
            lines.append(
                f"| {row['feature']} | {row['lift_ratio']:.2f}x | {row['lift_range']:.3f} |"
            )
        lines.append("")

    # Slice details
    if "all_lifts" in results:
        lifts_df = results["all_lifts"]

        lines.extend(
            [
                "## Detailed Slice Analysis",
                "",
            ]
        )

        for feature in lifts_df["feature"].unique():
            feat_df = lifts_df[lifts_df["feature"] == feature]
            lines.extend(
                [
                    f"### {feature}",
                    "",
                    "| Bin | Count | Success Rate | Lift |",
                    "|-----|-------|--------------|------|",
                ]
            )

            for _, row in feat_df.iterrows():
                lines.append(
                    f"| {row['bin']} | {row['count']:,} | {row['mean_outcome']:.1%} | {row['lift']:.3f} |"
                )
            lines.append("")

    # Visualizations
    lines.extend(
        [
            "## Visualizations",
            "",
            "- [Signal Strength Chart](signal_strength.png)",
            "- [Key Slice Lifts](key_slice_lifts.png)",
            "- [Zone Lift Heatmap](zone_lift_heatmap.png)",
            "",
            "## Interpretation",
            "",
            "**Lift > 1.0**: Higher success rate than overall average",
            "",
            "**Lift < 1.0**: Lower success rate than overall average",
            "",
            "**Lift Ratio > 1.1**: Meaningful discriminative signal for modeling",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
