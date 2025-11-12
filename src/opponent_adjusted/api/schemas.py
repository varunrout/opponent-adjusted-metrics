"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class ModelVersionResponse(BaseModel):
    """Model version information."""

    model_name: str
    version: str
    algorithm: str
    trained_on_version_tag: str
    created_at: datetime


class ShotPredictionRequest(BaseModel):
    """Request for shot prediction."""

    # Geometric features
    location_x: float = Field(..., ge=0, le=120, description="X coordinate (0-120)")
    location_y: float = Field(..., ge=0, le=80, description="Y coordinate (0-80)")

    # Shot characteristics
    body_part: str = Field(..., description="Body part (e.g., 'Right Foot')")
    technique: str = Field(..., description="Technique (e.g., 'Normal', 'Volley')")
    shot_type: str = Field(..., description="Shot type (e.g., 'Open Play')")
    first_time: bool = Field(False, description="First-time shot")

    # Context
    minute: int = Field(..., ge=0, le=120, description="Match minute")
    score_diff: int = Field(..., description="Team goals - opponent goals")
    under_pressure: bool = Field(False, description="Under defensive pressure")

    # Opponent
    opponent_team_id: int = Field(..., description="Opponent team ID")

    # Optional possession context
    possession_duration: Optional[float] = Field(None, ge=0, description="Possession duration")
    possession_length: Optional[int] = Field(None, ge=0, description="Events in possession")


class ShotPredictionResponse(BaseModel):
    """Response for shot prediction."""

    raw_probability: float = Field(..., description="Raw CxG probability")
    neutral_probability: float = Field(..., description="Neutralized CxG probability")
    opponent_adjusted_diff: float = Field(..., description="Raw - neutral difference")
    opponent_adjusted_ratio: float = Field(..., description="Raw / neutral ratio")

    # Feature contributions (optional)
    geometry_score: Optional[float] = None
    context_score: Optional[float] = None
    opponent_effect: Optional[float] = None


class PlayerAggregateResponse(BaseModel):
    """Player aggregate metrics."""

    player_id: int
    player_name: str
    shots_count: int
    summed_cxg: float
    summed_neutral_cxg: float
    summed_oppadj_diff: float
    avg_oppadj_diff: Optional[float]


class TeamAggregateResponse(BaseModel):
    """Team aggregate metrics."""

    team_id: int
    team_name: str
    shots_count: int
    summed_cxg: float
    summed_neutral_cxg: float
    summed_oppadj_diff: float
    avg_oppadj_diff: Optional[float]
