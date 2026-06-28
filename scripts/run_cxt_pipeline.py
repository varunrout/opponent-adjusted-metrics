#!/usr/bin/env python
"""Run the leakage-safe baseline CxT pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from opponent_adjusted.config import ensure_directories, settings
from opponent_adjusted.features.cxt.baseline import run_baseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_CANDIDATES = (
    settings.feature_store_path / "cxt" / "progressions_featured.parquet",
    settings.feature_store_path / "cxt" / "progressions.parquet",
    settings.feature_store_path / "cxa" / "action_features.parquet",
)


def resolve_input_path(input_path: Path | None) -> Path:
    """Resolve the production CxT action input.

    The reusable baseline helper still has a tiny synthetic fixture for unit
    tests, but the production script must use real generated action outputs.
    """

    if input_path is not None:
        return input_path
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    candidates = ", ".join(str(path) for path in DEFAULT_INPUT_CANDIDATES)
    raise FileNotFoundError(
        "No real CxT action input found. Build CxA/CxT action features first or pass "
        f"--input explicitly. Checked: {candidates}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline CxT pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional CSV, JSON, JSONL, or parquet action table. "
            "If omitted, the script uses generated local CxT/CxA action features."
        ),
    )
    parser.add_argument(
        "--feature-store-dir",
        type=Path,
        default=settings.feature_store_path / "cxt",
        help="Feature-store output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/modeling/cxt"),
        help="Modeling output directory.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also write CSV mirrors for predictions and aggregates.",
    )
    parser.add_argument(
        "--no-db-persist",
        action="store_true",
        help="Write CxT files only and skip DB persistence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    input_path = resolve_input_path(args.input)
    logger.info("Baseline CxT input path: %s", input_path)

    outputs = run_baseline(
        input_path=input_path,
        feature_store_dir=args.feature_store_dir,
        output_dir=args.output_dir,
        write_csv=args.write_csv,
        persist_db=not args.no_db_persist,
    )

    logger.info("Baseline CxT feature path: %s", outputs.feature_path)
    logger.info("Baseline CxT threat grid path: %s", outputs.threat_grid_path)
    logger.info("Baseline CxT predictions path: %s", outputs.predictions_path)
    logger.info("Baseline CxT player aggregates path: %s", outputs.player_aggregates_path)
    logger.info("Baseline CxT team aggregates path: %s", outputs.team_aggregates_path)
    logger.info("Baseline CxT sequence aggregates path: %s", outputs.sequence_aggregates_path)
    logger.info("Baseline CxT metrics path: %s", outputs.metrics_path)
    logger.info(
        "Baseline CxT zone transition summary path: %s",
        outputs.zone_transition_summary_path,
    )
    logger.info("Baseline CxT top actions path: %s", outputs.top_actions_path)
    logger.info(
        "Baseline CxT interpretation summary path: %s",
        outputs.interpretation_summary_path,
    )


if __name__ == "__main__":
    main()
