"""Storage boundary for raw StatsBomb payloads."""

from __future__ import annotations

from typing import Protocol

JsonPayload = list | dict


class RawStatsBombStore(Protocol):
    """Minimal storage contract required by subset ingestion."""

    def has_events(self, match_id: int) -> bool:
        """Return True when the raw events payload already exists for this match."""

    def write_competitions(self, payload: JsonPayload, *, force: bool = False) -> bool:
        """Persist the filtered competitions payload; return whether written."""

    def write_matches(
        self,
        competition_id: int,
        season_id: int,
        payload: JsonPayload,
        *,
        force: bool = False,
    ) -> bool:
        """Persist a competition-season matches payload; return whether written."""

    def write_events(self, match_id: int, payload: JsonPayload, *, force: bool = False) -> bool:
        """Persist a match events payload; return whether written."""
