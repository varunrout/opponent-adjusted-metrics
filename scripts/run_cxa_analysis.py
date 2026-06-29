"""Run the pre-model CxA target and action-feature study."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from opponent_adjusted.analysis import run_pre_model_cxa_analysis
from opponent_adjusted.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-model CxA analysis")
    parser.add_argument(
        "--database-url",
        type=str,
        default=settings.database_url,
        help="SQLAlchemy database URL. Defaults to configured DATABASE_URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cxa"),
        help="Directory for generated CxA analysis artifacts.",
    )
    parser.add_argument(
        "--feature-family",
        type=str,
        default="cxa",
        help="Optional action_features feature_family filter.",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default=None,
        help="Optional action_features version_tag filter.",
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
        result = run_pre_model_cxa_analysis(
            session,
            output_dir=args.output_dir,
            feature_family=args.feature_family,
            version_tag=args.version_tag,
            min_slice_size=args.min_slice_size,
        )

    print(f"Wrote pre-model CxA analysis report: {result.report_path}")
    print(f"Rows: {result.row_count} | Target rate: {result.target_rate:.4f}")
    print(f"Target column: {result.target_column}")
    print(f"Candidate features: {result.candidate_feature_count}")
    print(f"Leakage/reference columns flagged: {result.leakage_risk_count}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
