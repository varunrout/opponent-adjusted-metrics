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
