"""Build a modeling-ready cxA training dataset (pass-level).

This script emits pass rows with regression targets based on the sequence-based
xA+ definition (decay-weighted allocation of shot value across the last k
actions before a shot).

Unlike `build_cxa_baselines.py` (which emits only pass-shot attribution rows),
this script also emits negative examples (passes that are not attributed to any
shot in the chosen window), enabling supervised learning.

Targets:
- y_xa_plus: decay-weighted share of shot_value (0 for negatives)
- y_keypass: shot_value for the final pass before a shot else 0 (0 for negatives)

Example:
    python scripts/build_cxa_training_dataset.py \
      --database-url sqlite:///data/opponent_adjusted.db \
      --model-name cxg --k-actions 3 --decay 0.6 \
      --negatives-ratio 2.0 \
      --out outputs/analysis/cxa/csv/cxa_training_pass_level.csv
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

from opponent_adjusted.pipelines.cxa import build_cxa_training_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cxA training dataset")
    p.add_argument("--database-url", required=True, type=str)
    p.add_argument("--model-name", default="cxg", type=str)
    p.add_argument("--version", default=None, type=str)
    p.add_argument("--k-actions", default=3, type=int)
    p.add_argument("--decay", default=0.6, type=float)
    p.add_argument(
        "--negatives-ratio",
        default=2.0,
        type=float,
        help="Number of negative pass rows per positive row",
    )
    p.add_argument("--seed", default=7, type=int)
    p.add_argument("--limit-matches", default=None, type=int)
    p.add_argument(
        "--out",
        default="outputs/analysis/cxa/csv/cxa_training_pass_level.csv",
        type=str,
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine = create_engine(args.database_url)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        df = build_cxa_training_dataset(
            session,
            model_name=args.model_name,
            version=args.version,
            k_actions=args.k_actions,
            decay=args.decay,
            negatives_ratio=args.negatives_ratio,
            seed=args.seed,
            limit_matches=args.limit_matches,
            progress_every_matches=10,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
