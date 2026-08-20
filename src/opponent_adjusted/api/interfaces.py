"""Serving boundary for the dashboard API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class CompetitionRecord:
    """One competition-season row from the serving store."""

    competition_id: int
    season_id: int
    competition_name: str | None
    competition_gender: str | None
    country_name: str | None
    season_name: str | None
    match_updated: str | None
    match_available: str | None
    match_updated_360: str | None
    match_available_360: str | None


@dataclass(frozen=True)
class MatchRecord:
    """One match row from the serving store."""

    match_id: int
    competition_id: int
    season_id: int
    match_date: str | None
    kick_off: str | None
    home_team_id: int | None
    home_team_name: str | None
    away_team_id: int | None
    away_team_name: str | None
    home_score: int | None
    away_score: int | None
    competition_stage: str | None
    stadium: str | None
    referee: str | None
    match_status: str | None
    match_status_360: str | None
    last_updated: str | None
    last_updated_360: str | None


class ServingStore(Protocol):
    """Minimal read-only contract required by the dashboard API."""

    def list_competitions(self) -> list[CompetitionRecord]:
        """Return one row per (competition_id, season_id)."""

    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[MatchRecord]:
        """Return matches, optionally filtered by competition_id and/or season_id."""
