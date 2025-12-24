"""Run exploratory analysis on the cxA baseline CSV.

This reads the baseline attribution CSV created by `scripts/build_cxa_baselines.py`
and exports aggregated tables and plots under `outputs/analysis/cxa/`.

Example:
    python scripts/analyze_cxa_baselines.py \
      --baseline-csv outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv

Optionally attach player/team names (requires DB):
    python scripts/analyze_cxa_baselines.py \
      --baseline-csv outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv \
      --database-url sqlite:///data/opponent_adjusted.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Allow running directly without Poetry
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from opponent_adjusted.analysis.cxa import run_cxa_baseline_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze cxA baseline CSV")
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_baselines_pass_level.csv",
    )
    p.add_argument("--database-url", type=str, default=None)
    p.add_argument("--top-n", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.database_url:
        engine = create_engine(args.database_url)
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as session:
            run_cxa_baseline_analysis(args.baseline_csv, session=session, top_n=args.top_n)
    else:
        run_cxa_baseline_analysis(args.baseline_csv, session=None, top_n=args.top_n)


if __name__ == "__main__":
    main()
