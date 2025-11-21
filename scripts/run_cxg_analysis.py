"""Entry-point script with simple CxG analysis examples.

This is intentionally lightweight and primarily intended for local,
manual use while iterating on the CxG model and neutralization
pipelines. It currently demonstrates how to:

- fetch a few shot-level CxG records
- fetch top-N player and team summaries

Example usage (from project root):

    poetry run python -m scripts.run_cxg_analysis \
        --database-url sqlite:///data/opponent_adjusted.db \
        --model-name cxg --limit 20
"""

from __future__ import annotations

import argparse
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from opponent_adjusted.analysis import (
    compute_shot_level_cxg,
    summarize_player_cxg,
    summarize_team_cxg,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run basic CxG analyses")
    parser.add_argument(
        "--database-url",
        type=str,
        required=True,
        help="SQLAlchemy database URL (e.g. sqlite:///path/to.db)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cxg",
        help="Model name in model_registry (default: cxg)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Optional exact model version (default: latest for model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit for number of rows to print in summaries",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    engine = create_engine(args.database_url)
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:  # type: Session
        print("=== Sample shot-level CxG records ===")
        shots = compute_shot_level_cxg(
            session,
            model_name=args.model_name,
            version=args.version,
            limit=args.limit,
        )
        for s in shots:
            print(s)

        print("\n=== Top player CxG summaries ===")
        players = summarize_player_cxg(
            session,
            model_name=args.model_name,
            version=args.version,
            limit=args.limit,
        )
        for p in players:
            print(p)

        print("\n=== Top team CxG summaries ===")
        teams = summarize_team_cxg(
            session,
            model_name=args.model_name,
            version=args.version,
            limit=args.limit,
        )
        for t in teams:
            print(t)


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
