"""HTTP source client for StatsBomb Open Data."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
JsonPayload = list | dict
FetchBytes = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "opponent-adjusted-fetch/1.0"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - public StatsBomb URL only
        return response.read()


def _load_json_url(url: str, fetch_bytes: FetchBytes | None = None) -> JsonPayload:
    loader = fetch_bytes or _fetch
    return json.loads(loader(url).decode("utf-8"))


def fetch_json_with_retries(
    url: str,
    *,
    retries: int = 3,
    backoff_seconds: float = 0.8,
    fetch_json: Callable[[str], JsonPayload] | None = None,
) -> JsonPayload | None:
    """Fetch a JSON payload with the existing retry and failure semantics."""
    loader = fetch_json or _load_json_url
    for attempt in range(1, retries + 1):
        try:
            return loader(url)
        except (HTTPError, URLError) as exc:
            logger.warning("Fetch failed %s/%s for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                return None
            time.sleep(backoff_seconds * attempt)
    return None


class StatsBombSource:
    """Read-only StatsBomb Open Data source client."""

    def __init__(
        self,
        *,
        base_url: str = RAW_BASE_URL,
        fetch_json: Callable[[str], JsonPayload | None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._fetch_json = fetch_json or fetch_json_with_retries

    def get_competitions(self) -> JsonPayload | None:
        return self._fetch_json(f"{self.base_url}/competitions.json")

    def get_matches(self, competition_id: int, season_id: int) -> JsonPayload | None:
        return self._fetch_json(f"{self.base_url}/matches/{competition_id}/{season_id}.json")

    def get_events(self, match_id: int) -> JsonPayload | None:
        return self._fetch_json(f"{self.base_url}/events/{match_id}.json")

    def pace_after_event_fetch(self) -> None:
        """Preserve legacy per-event pacing after successful event fetch+write."""
        time.sleep(0.05)
