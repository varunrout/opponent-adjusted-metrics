"""Pydantic response models for the dashboard API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CompetitionResponse(BaseModel):
    """API response shape for a competition-season."""

    model_config = ConfigDict(from_attributes=True)

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


class MatchResponse(BaseModel):
    """API response shape for a match."""

    model_config = ConfigDict(from_attributes=True)

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


class LineupPlayerResponse(BaseModel):
    """API response shape for a starting XI player."""

    model_config = ConfigDict(from_attributes=True)

    team_id: int | None
    team_name: str | None
    formation: int | None
    player_id: int
    player_name: str | None
    position_name: str | None
    jersey_number: int | None


class MatchDetailResponse(MatchResponse):
    """API response shape for a match with its starting lineups."""

    lineups: list[LineupPlayerResponse]


class ShotResponse(BaseModel):
    """API response shape for a shot."""

    model_config = ConfigDict(from_attributes=True)

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
