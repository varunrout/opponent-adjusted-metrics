"""CLI wrapper for fetching a configured StatsBomb Open Data subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.config import settings  # noqa: E402
from opponent_adjusted.ingestion.contracts import load_subset_config  # noqa: E402
from opponent_adjusted.ingestion.statsbomb_source import (  # noqa: E402
    StatsBombSource,
    fetch_json_with_retries,
)
from opponent_adjusted.ingestion.subset_fetch import run_subset_fetch  # noqa: E402
from opponent_adjusted.storage.local import LocalRawStatsBombStore  # noqa: E402
from opponent_adjusted.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "statsbomb_subset.json"

# Compatibility seam for existing tests and callers that monkeypatch this name.
_fetch_with_retries = fetch_json_with_retries


def fetch_subset(config_path: Path, output_dir: Path, *, include_events: bool, force: bool) -> dict:
    """Fetch and locally store the configured subset."""
    config = load_subset_config(config_path)
    source = StatsBombSource(base_url=RAW_BASE_URL, fetch_json=_fetch_with_retries)
    store = LocalRawStatsBombStore(output_dir)
    return run_subset_fetch(
        config,
        source=source,
        store=store,
        include_events=include_events,
        force=force,
        config_label=str(config_path),
        output_label=str(output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch configured StatsBomb Open Data subset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Subset config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.statsbomb_data_path,
        help="StatsBomb output directory",
    )
    parser.add_argument(
        "--events",
        "--with-events",
        dest="with_events",
        action="store_true",
        help="Download match event files as well as competitions and matches",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = fetch_subset(
        config_path=args.config,
        output_dir=args.output_dir,
        include_events=args.with_events,
        force=args.force,
    )
    report_dir = PROJECT_ROOT / "outputs" / "reports" / "ingestion"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "fetch_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Fetch summary written to %s", report_path)
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
