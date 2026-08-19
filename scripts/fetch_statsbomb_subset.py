"""CLI wrapper for fetching a configured StatsBomb Open Data subset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.ingestion.contracts import load_subset_config  # noqa: E402
from opponent_adjusted.ingestion.statsbomb_source import (  # noqa: E402
    StatsBombSource,
    build_raw_base_url_for_ref,
    fetch_json_with_retries,
    is_valid_source_ref,
)
from opponent_adjusted.ingestion.subset_fetch import run_subset_fetch  # noqa: E402
from opponent_adjusted.storage.gcs import GCSRawStatsBombStore  # noqa: E402
from opponent_adjusted.storage.local import LocalRawStatsBombStore  # noqa: E402

logger = logging.getLogger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "statsbomb_subset.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "statsbomb"

# Compatibility seam for existing tests and callers that monkeypatch this name.
_fetch_with_retries = fetch_json_with_retries


def fetch_subset(
    config_path: Path,
    output_dir: Path,
    *,
    include_events: bool,
    include_three_sixty: bool = False,
    force: bool = False,
) -> dict:
    """Fetch and locally store the configured subset."""
    config = load_subset_config(config_path)
    source = StatsBombSource(base_url=RAW_BASE_URL, fetch_json=_fetch_with_retries)
    store = LocalRawStatsBombStore(output_dir)
    return run_subset_fetch(
        config,
        source=source,
        store=store,
        include_events=include_events,
        include_three_sixty=include_three_sixty,
        force=force,
        config_label=str(config_path),
        output_label=str(output_dir),
    )


def fetch_subset_to_gcs(
    config_path: Path,
    *,
    include_events: bool,
    include_three_sixty: bool = False,
    force: bool = False,
    gcs_bucket: str,
    data_version: str,
    source_ref: str,
) -> dict:
    """Fetch and store the configured subset into versioned GCS raw landing."""
    if not is_valid_source_ref(source_ref):
        raise ValueError(f"Invalid --source-ref (expected 40-char hex SHA): {source_ref}")
    if not is_valid_source_ref(data_version):
        raise ValueError(f"Invalid --data-version (expected 40-char hex SHA): {data_version}")
    if force:
        raise ValueError("--force is not supported for immutable GCS raw landing")

    config = load_subset_config(config_path)
    source = StatsBombSource(
        base_url=build_raw_base_url_for_ref(source_ref),
        fetch_json=_fetch_with_retries,
    )
    store = GCSRawStatsBombStore(gcs_bucket, data_version)
    return run_subset_fetch(
        config,
        source=source,
        store=store,
        include_events=include_events,
        include_three_sixty=include_three_sixty,
        force=force,
        config_label=str(config_path),
        output_label=f"gs://{gcs_bucket}/raw/statsbomb/{data_version}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch configured StatsBomb Open Data subset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Subset config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="StatsBomb output directory",
    )
    parser.add_argument(
        "--events",
        "--with-events",
        dest="with_events",
        action="store_true",
        help="Download match event files as well as competitions and matches",
    )
    parser.add_argument(
        "--with-360",
        dest="with_three_sixty",
        action="store_true",
        help="Download separate StatsBomb 360 files for supported competitions",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--gcs-bucket", type=str, default=None, help="Target GCS bucket")
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Immutable raw data version (40-char source commit SHA)",
    )
    parser.add_argument(
        "--source-ref",
        type=str,
        default=None,
        help="Pinned StatsBomb source commit SHA used to build raw URLs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.gcs_bucket:
        if not args.data_version or not args.source_ref:
            raise ValueError("--gcs-bucket requires both --data-version and --source-ref")
        summary = fetch_subset_to_gcs(
            config_path=args.config,
            include_events=args.with_events,
            include_three_sixty=args.with_three_sixty,
            force=args.force,
            gcs_bucket=args.gcs_bucket,
            data_version=args.data_version,
            source_ref=args.source_ref,
        )
    else:
        summary = fetch_subset(
            config_path=args.config,
            output_dir=args.output_dir,
            include_events=args.with_events,
            include_three_sixty=args.with_three_sixty,
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
