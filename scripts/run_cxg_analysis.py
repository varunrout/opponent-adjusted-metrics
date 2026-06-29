"""Run the pre-model CxG target and feature study.

This analysis reads engineered `shot_features` joined to base `shots`, writes
analysis artifacts under `outputs/analysis/cxg/`, and intentionally avoids
post-model prediction, registry, aggregate, and leaderboard outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from opponent_adjusted.analysis import run_pre_model_cxg_analysis
from opponent_adjusted.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-model CxG analysis")
    parser.add_argument(
        "--database-url",
        type=str,
        default=settings.database_url,
        help="SQLAlchemy database URL. Defaults to configured DATABASE_URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cxg"),
        help="Directory for generated analysis artifacts.",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Optional shot_features version_tag filter.",
    )
    parser.add_argument(
        "--min-slice-size",
        type=int,
        default=30,
        help="Minimum rows required before reporting a slice.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(args.database_url)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        result = run_pre_model_cxg_analysis(
            session,
            output_dir=args.output_dir,
            version_tag=args.version_tag,
            min_slice_size=args.min_slice_size,
        )

    print(f"Wrote pre-model CxG analysis report: {result.report_path}")
    print(f"Rows: {result.row_count} | Goal rate: {result.goal_rate:.3f}")
    print(f"Candidate features: {result.feature_count}")
    print(f"Leakage/reference columns flagged: {result.leakage_risk_count}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
