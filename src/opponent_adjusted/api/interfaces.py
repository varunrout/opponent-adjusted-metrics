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


@dataclass(frozen=True)
class LineupPlayerRecord:
    """One starting XI player row for a match, from the serving store."""

    match_id: int
    team_id: int | None
    team_name: str | None
    formation: int | None
    player_id: int
    player_name: str | None
    position_name: str | None
    jersey_number: int | None


@dataclass(frozen=True)
class ShotRecord:
    """One shot row for a match, from the serving store."""

    event_id: str
    match_id: int
    team_id: int | None
    player_id: int | None
    player_name: str | None
    minute: int | None
    period: int | None
    location_x: float | None
    location_y: float | None
    end_x: float | None
    end_y: float | None
    statsbomb_xg: float | None
    outcome_name: str | None
    body_part_name: str | None
    is_goal: bool


@dataclass(frozen=True)
class PlayerSeasonRecord:
    """One player's shot aggregates row from the serving store."""

    player_id: int
    player_name: str | None
    team_name: str | None
    shots: int
    goals: int
    total_xg: float


@dataclass(frozen=True)
class TeamSeasonRecord:
    """One team's shot aggregates row from the serving store."""

    team_id: int
    team_name: str | None
    shots: int
    goals: int
    total_xg: float


class ServingStore(Protocol):
    """Minimal read-only contract required by the dashboard API."""

    def list_competitions(self) -> list[CompetitionRecord]:
        """Return one row per (competition_id, season_id)."""

    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
        team_id: int | None = None,
    ) -> list[MatchRecord]:
        """Return matches, optionally filtered by competition_id, season_id, and/or team_id (home or away)."""

    def get_match(self, match_id: int) -> MatchRecord | None:
        """Return the match row for match_id, or None if no such match exists."""

    def list_lineups(self, match_id: int) -> list[LineupPlayerRecord]:
        """Return starting XI rows for both teams in match_id, via starting_xi_players_join_events_matches."""

    def list_shots(self, match_id: int) -> list[ShotRecord]:
        """Return shots for match_id, joined to events for player_name/minute/period, via shots_join_events_matches."""

    def list_player_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[PlayerSeasonRecord]:
        """Return per-player shot aggregates (shots/goals/total_xg), optionally filtered, sorted by total_xg descending."""

    def list_player_shots(
        self,
        player_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        """Return a player's shots joined to events for player_name/minute/period, optionally filtered by competition_id/season_id."""

    def list_team_seasons(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[TeamSeasonRecord]:
        """Return per-team shot aggregates (shots/goals/total_xg), optionally filtered, sorted by total_xg descending."""

    def list_team_shots(
        self,
        team_id: int,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[ShotRecord]:
        """Return a team's shots joined to events for player_name/minute/period, optionally filtered by competition_id/season_id."""
