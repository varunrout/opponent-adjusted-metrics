"""Build cxA baselines (baseline xA and sequence-based xA+).

Outputs pass-level attribution tables that can be used for analysis and as
starting targets/features for full cxA modelling.

Baselines implemented:
- Key-pass xA (industry): xA(pass) = shot_value if pass is the final pass before the shot else 0
- Sequence-based xA+ (project baseline): distribute shot NeutralCxG across last k passes before shot

Shot value preference order:
1) ShotPrediction.neutral_probability for selected model/version (when available)
2) ShotPrediction.raw_probability when record is_neutralized=True (fallback)
3) Shot.statsbomb_xg (fallback)

Example:
    poetry run python scripts/build_cxa_baselines.py \
      --database-url sqlite:///data/opponent_adjusted.db \
      --model-name cxg --version v1 \
      --k-actions 3 --decay 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Allow running this script directly (without `poetry run`) by ensuring `src/` is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _REPO_ROOT / "src"
if _SRC_PATH.exists() and str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from opponent_adjusted.pipelines.cxa import build_cxa_baselines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cxA baselines (xA and xA+)")
    p.add_argument("--database-url", required=True, type=str)
    p.add_argument("--model-name", default="cxg", type=str)
    p.add_argument("--version", default=None, type=str)
    p.add_argument("--k-actions", default=3, type=int)
    p.add_argument("--decay", default=0.6, type=float)
    p.add_argument("--limit-shots", default=None, type=int)
    p.add_argument("--out", default=None, type=str, help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    engine = create_engine(args.database_url)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        df = build_cxa_baselines(
            session,
            model_name=args.model_name,
            version=args.version,
            k_actions=args.k_actions,
            decay=args.decay,
            limit_shots=args.limit_shots,
            progress_every=200,
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path("outputs") / "analysis" / "cxa" / "csv" / "cxa_baselines_pass_level.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df):,} pass-attribution rows to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
