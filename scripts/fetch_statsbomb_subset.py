"""Fetch a configured StatsBomb Open Data subset.

The script downloads only the configured competitions, matches, and optionally
match event files from StatsBomb's public open-data repository. It is designed
for deterministic project setup rather than exploratory data discovery.

Examples:
    poetry run python scripts/fetch_statsbomb_subset.py
    poetry run python scripts/fetch_statsbomb_subset.py --events
    poetry run python scripts/fetch_statsbomb_subset.py --config configs/statsbomb_subset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opponent_adjusted.config import settings  # noqa: E402
from opponent_adjusted.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "statsbomb_subset.json"


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "opponent-adjusted-fetch/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - public StatsBomb URL only
        return response.read()


def _load_json_url(url: str) -> list | dict:
    return json.loads(_fetch(url).decode("utf-8"))


def _write_json(path: Path, payload: list | dict, *, force: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return False
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def _load_subset_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Subset config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_competitions(all_competitions: list[dict], configured: list[dict]) -> list[dict]:
    wanted = {(int(item["competition_id"]), int(item["season_id"])) for item in configured}
    return [
        comp
        for comp in all_competitions
        if (int(comp["competition_id"]), int(comp["season_id"])) in wanted
    ]


def _fetch_with_retries(
    url: str, *, retries: int = 3, backoff_seconds: float = 0.8
) -> list | dict | None:
    for attempt in range(1, retries + 1):
        try:
            return _load_json_url(url)
        except (HTTPError, URLError) as exc:
            logger.warning("Fetch failed %s/%s for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                return None
            time.sleep(backoff_seconds * attempt)
    return None


def fetch_subset(config_path: Path, output_dir: Path, *, include_events: bool, force: bool) -> dict:
    config = _load_subset_config(config_path)
    competitions_config = config.get("competitions", [])
    if not competitions_config:
        raise ValueError(f"No competitions configured in {config_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "competitions_selected": 0,
        "competitions_written": 0,
        "matches_written": 0,
        "matches_skipped_existing": 0,
        "events_written": 0,
        "events_skipped_existing": 0,
        "missing": [],
    }

    logger.info("Fetching competitions index")
    all_competitions = _fetch_with_retries(f"{RAW_BASE_URL}/competitions.json")
    if not isinstance(all_competitions, list):
        raise RuntimeError("Could not fetch StatsBomb competitions.json")

    selected_competitions = _filter_competitions(all_competitions, competitions_config)
    summary["competitions_selected"] = len(selected_competitions)

    if not selected_competitions:
        raise RuntimeError("Configured competitions were not found in StatsBomb competitions.json")

    if _write_json(output_dir / "competitions.json", selected_competitions, force=force):
        summary["competitions_written"] = 1

    for comp in selected_competitions:
        competition_id = int(comp["competition_id"])
        season_id = int(comp["season_id"])
        label = f"competition_id={competition_id} season_id={season_id}"
        logger.info("Fetching matches for %s", label)

        matches_url = f"{RAW_BASE_URL}/matches/{competition_id}/{season_id}.json"
        matches = _fetch_with_retries(matches_url)
        if not isinstance(matches, list):
            logger.warning("Could not fetch matches for %s", label)
            summary["missing"].append(
                {"scope": "matches", "competition_id": competition_id, "season_id": season_id}
            )
            continue

        matches_path = output_dir / "matches" / str(competition_id) / f"{season_id}.json"
        if matches_path.exists() and not force:
            summary["matches_skipped_existing"] += 1
        elif _write_json(matches_path, matches, force=force):
            summary["matches_written"] += 1

        if not include_events:
            continue

        should_fetch_events = next(
            (
                bool(item.get("include_events", True))
                for item in competitions_config
                if int(item["competition_id"]) == competition_id
                and int(item["season_id"]) == season_id
            ),
            True,
        )
        if not should_fetch_events:
            continue

        for match in matches:
            match_id = int(match["match_id"])
            event_path = output_dir / "events" / f"{match_id}.json"
            if event_path.exists() and not force:
                summary["events_skipped_existing"] += 1
                continue

            events_url = f"{RAW_BASE_URL}/events/{match_id}.json"
            events = _fetch_with_retries(events_url)
            if not isinstance(events, list):
                logger.warning("Could not fetch events for match_id=%s", match_id)
                summary["missing"].append({"scope": "events", "match_id": match_id})
                continue

            if _write_json(event_path, events, force=force):
                summary["events_written"] += 1
            time.sleep(0.05)

    report_dir = PROJECT_ROOT / "outputs" / "reports" / "ingestion"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "fetch_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Fetch summary written to %s", report_path)
    return summary


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
    logger.info("Done: %s", summary)


if __name__ == "__main__":
    main()
