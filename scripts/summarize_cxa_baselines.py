"""Summarize cxA baseline outputs.

This is a lightweight validation helper for the CSV emitted by
`scripts/build_cxa_baselines.py`.

Examples:
    python scripts/summarize_cxa_baselines.py \
      --path outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly without Poetry
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from opponent_adjusted.analysis.cxa import summarize_cxa_baselines_from_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize cxA baseline CSV")
    p.add_argument(
        "--path",
        default="outputs/analysis/cxa/csv/cxa_baselines_pass_level.csv",
        help="Path to baseline CSV",
    )
    p.add_argument("--top", default=15, type=int, help="Top N players/teams to show")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summarize_cxa_baselines_from_csv(args.path, top_n=args.top, verbose=True)


if __name__ == "__main__":
    main()
