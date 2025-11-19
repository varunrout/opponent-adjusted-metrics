"""Fetch a minimal subset of StatsBomb Open Data needed for ingestion.

Downloads:
- competitions.json
- matches/<competition_id>/<season_id>.json for configured filters
- events/<match_id>.json for all matches in those files

Sources raw files from GitHub without cloning the whole repo.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.config import settings
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def _choose_branch() -> str:
    # StatsBomb open-data historically uses 'master'. Keep fallback to 'main'.
    return "master"


def _fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "opponent-adjusted-fetch/1.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _fetch_to_file(url: str, dest: Path, retries: int = 3, backoff: float = 0.8) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            data = _fetch(url)
            dest.write_bytes(data)
            return True
        except (HTTPError, URLError) as e:
            logger.warning("Fetch failed (%s/%s): %s", attempt, retries, url)
            if attempt == retries:
                logger.error("Giving up on %s: %s", url, e)
                return False
            time.sleep(backoff * attempt)
    return False


def fetch_competitions(base_raw: str, out_root: Path) -> Path:
    url = f"{base_raw}/competitions.json"
    dest = out_root / "competitions.json"
    ok = _fetch_to_file(url, dest)
    if ok:
        logger.info("Downloaded competitions.json -> %s", dest)
    return dest


def fetch_matches_for_filters(base_raw: str, out_root: Path, filters: list[dict]) -> list[Path]:
    downloaded = []
    for flt in filters:
        if "competition_id" not in flt or "season_id" not in flt:
            continue
        comp_id = int(flt["competition_id"])
        season_id = int(flt["season_id"]) 
        url = f"{base_raw}/matches/{comp_id}/{season_id}.json"
        dest = out_root / "matches" / str(comp_id) / f"{season_id}.json"
        if _fetch_to_file(url, dest):
            logger.info("Downloaded matches for %s/%s -> %s", comp_id, season_id, dest)
            downloaded.append(dest)
    return downloaded


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fetch_events_from_matches(base_raw: str, out_root: Path, match_files: list[Path]) -> int:
    count = 0
    for mf in match_files:
        matches = _load_json(mf) or []
        for m in matches:
            match_id = m.get("match_id")
            if match_id is None:
                continue
            url = f"{base_raw}/events/{match_id}.json"
            dest = out_root / "events" / f"{match_id}.json"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            if _fetch_to_file(url, dest):
                count += 1
                # light pacing
                time.sleep(0.05)
    return count


def main():
    parser = argparse.ArgumentParser(description="Fetch StatsBomb subset")
    parser.add_argument(
        "--with-events",
        dest="with_events",
        action="store_true",
        help="Also download events for matches in the configured filters",
    )
    parser.add_argument(
        "--events",
        dest="with_events",
        action="store_true",
        help="Alias for --with-events",
    )
    args = parser.parse_args()

    branch = _choose_branch()
    base_raw = f"https://raw.githubusercontent.com/statsbomb/open-data/{branch}/data"

    out_root = settings.statsbomb_data_path
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching competitions.json ...")
    comp_path = fetch_competitions(base_raw, out_root)
    if not comp_path.exists():
        logger.error("Failed to download competitions.json; aborting.")
        sys.exit(1)

    filters = settings.competitions
    logger.info("Fetching matches for %d filters ...", len(filters))
    match_files = fetch_matches_for_filters(base_raw, out_root, filters)
    if not match_files:
        logger.warning("No match files downloaded. Check filters or connectivity.")

    if args.with_events and match_files:
        logger.info("Fetching events for downloaded matches ...")
        ev_count = fetch_events_from_matches(base_raw, out_root, match_files)
        logger.info("Downloaded %d event files", ev_count)

    logger.info("Fetch complete. Output root: %s", out_root)


if __name__ == "__main__":
    main()
