"""Local raw StatsBomb JSON storage."""

from __future__ import annotations

import json
from pathlib import Path

from opponent_adjusted.storage.interfaces import JsonPayload


class LocalRawStatsBombStore:
    """Persist raw StatsBomb payloads using the existing local layout."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def has_events(self, match_id: int) -> bool:
        return (self.root / "events" / f"{match_id}.json").exists()

    def has_three_sixty(self, match_id: int) -> bool:
        return (self.root / "three-sixty" / f"{match_id}.json").exists()

    def _write(self, path: Path, payload: JsonPayload, *, force: bool) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            return False
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return True

    def write_competitions(self, payload: JsonPayload, *, force: bool = False) -> bool:
        return self._write(self.root / "competitions.json", payload, force=force)

    def write_matches(
        self,
        competition_id: int,
        season_id: int,
        payload: JsonPayload,
        *,
        force: bool = False,
    ) -> bool:
        path = self.root / "matches" / str(competition_id) / f"{season_id}.json"
        return self._write(path, payload, force=force)

    def write_events(self, match_id: int, payload: JsonPayload, *, force: bool = False) -> bool:
        return self._write(self.root / "events" / f"{match_id}.json", payload, force=force)

    def write_three_sixty(
        self,
        match_id: int,
        payload: JsonPayload,
        *,
        force: bool = False,
    ) -> bool:
        return self._write(self.root / "three-sixty" / f"{match_id}.json", payload, force=force)
