"""Run the pre-model CxT-style ball progression feature study."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from opponent_adjusted.analysis.cxt import run_pre_model_cxt_analysis
from opponent_adjusted.config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pre-model CxT/progression analysis")
    parser.add_argument(
        "--database-url",
        type=str,
        default=settings.database_url,
        help="SQLAlchemy database URL. Defaults to configured DATABASE_URL.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Optional explicit pre-model progression/action parquet fallback.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/cxt"),
        help="Directory for generated CxT analysis artifacts.",
    )
    parser.add_argument(
        "--min-sample-size",
        type=int,
        default=30,
        help="Minimum rows before flagging sparse zones, transitions, or slices.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = create_engine(args.database_url)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        result = run_pre_model_cxt_analysis(
            session,
            output_dir=args.output_dir,
            parquet_path=args.parquet_path,
            min_sample_size=args.min_sample_size,
        )

    print(f"Wrote pre-model CxT analysis report: {result.report_path}")
    print(f"Rows: {result.row_count} | Source: {result.data_source}")
    print(f"Target/proxy: {result.target_proxy_column or 'missing'}")
    print(f"Candidate features: {result.candidate_feature_count}")
    print(f"Leakage/reference columns flagged: {result.leakage_risk_count}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
