"""SQLAlchemy database models for opponent-adjusted metrics."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from opponent_adjusted.db.base import Base


class Competition(Base):
    """Competition reference table."""

    __tablename__ = "competitions"

    statsbomb_competition_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    season: Mapped[str] = mapped_column(String(50), nullable=False)

    # Relationships
    matches: Mapped[list["Match"]] = relationship("Match", back_populates="competition")

    __table_args__ = (
        UniqueConstraint("statsbomb_competition_id", "season", name="uq_competition_season"),
        Index("ix_competitions_name_season", "name", "season"),
    )


class Team(Base):
    """Team reference table."""

    __tablename__ = "teams"

    statsbomb_team_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    home_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.home_team_id", back_populates="home_team"
    )
    away_matches: Mapped[list["Match"]] = relationship(
        "Match", foreign_keys="Match.away_team_id", back_populates="away_team"
    )
    opponent_profiles: Mapped[list["OpponentDefProfile"]] = relationship(
        "OpponentDefProfile", back_populates="team"
    )


class Player(Base):
    """Player reference table."""

    __tablename__ = "players"

    statsbomb_player_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    position: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)


class Match(Base):
    """Match table."""

    __tablename__ = "matches"

    statsbomb_match_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False)
    competition_id: Mapped[int] = mapped_column(ForeignKey("competitions.id"), nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    kickoff_time: Mapped[Optional[DateTime]] = mapped_column(DateTime, nullable=True)
    match_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    season: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    competition: Mapped["Competition"] = relationship("Competition", back_populates="matches")
    home_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[home_team_id], back_populates="home_matches"
    )
    away_team: Mapped["Team"] = relationship(
        "Team", foreign_keys=[away_team_id], back_populates="away_matches"
    )
    raw_events: Mapped[list["RawEvent"]] = relationship("RawEvent", back_populates="match")
    events: Mapped[list["Event"]] = relationship("Event", back_populates="match")
    shots: Mapped[list["Shot"]] = relationship("Shot", back_populates="match")

    __table_args__ = (Index("ix_matches_competition_id", "competition_id"),)


class RawEvent(Base):
    """Raw event data from StatsBomb."""

    __tablename__ = "raw_events"

    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    statsbomb_event_id: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    period: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    second: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="raw_events")

    __table_args__ = (
        UniqueConstraint("match_id", "statsbomb_event_id", name="uq_raw_event"),
        Index("ix_raw_events_match_id", "match_id"),
        Index("ix_raw_events_type", "type"),
    )


class Possession(Base):
    """Possession sequence table."""

    __tablename__ = "possessions"

    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    possession_number: Mapped[int] = mapped_column(Integer, nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    start_event_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("events.id"), nullable=True
    )
    end_event_id: Mapped[Optional[int]] = mapped_column(ForeignKey("events.id"), nullable=True)
    start_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_minute: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    event_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        UniqueConstraint("match_id", "possession_number", name="uq_possession"),
        Index("ix_possessions_match_id", "match_id"),
        Index("ix_possessions_team_id", "team_id"),
    )


class Event(Base):
    """Normalized event table."""

    __tablename__ = "events"

    raw_event_id: Mapped[int] = mapped_column(ForeignKey("raw_events.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    player_id: Mapped[Optional[int]] = mapped_column(ForeignKey("players.id"), nullable=True)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    period: Mapped[int] = mapped_column(Integer, nullable=False)
    minute: Mapped[int] = mapped_column(Integer, nullable=False)
    second: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    possession: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    location_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    location_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    under_pressure: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="events")

    __table_args__ = (
        Index("ix_events_match_id", "match_id"),
        Index("ix_events_match_possession", "match_id", "possession"),
        Index("ix_events_team_id", "team_id"),
        Index("ix_events_type", "type"),
    )


class Shot(Base):
    """Shot table with base information."""

    __tablename__ = "shots"

    event_id: Mapped[int] = mapped_column(
        ForeignKey("events.id"), unique=True, nullable=False
    )
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    player_id: Mapped[Optional[int]] = mapped_column(ForeignKey("players.id"), nullable=True)
    opponent_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    statsbomb_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    body_part: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    technique: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    shot_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    outcome: Mapped[str] = mapped_column(String(50), nullable=False)
    first_time: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    match: Mapped["Match"] = relationship("Match", back_populates="shots")
    shot_features: Mapped[Optional["ShotFeature"]] = relationship(
        "ShotFeature", back_populates="shot", uselist=False
    )
    predictions: Mapped[list["ShotPrediction"]] = relationship(
        "ShotPrediction", back_populates="shot"
    )

    __table_args__ = (
        Index("ix_shots_match_id", "match_id"),
        Index("ix_shots_team_id", "team_id"),
        Index("ix_shots_opponent_team_id", "opponent_team_id"),
    )


class ShotFeature(Base):
    """Shot features table."""

    __tablename__ = "shot_features"

    shot_id: Mapped[int] = mapped_column(ForeignKey("shots.id"), unique=True, nullable=False)
    version_tag: Mapped[str] = mapped_column(String(20), nullable=False)

    # Geometry features
    shot_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shot_angle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    centrality: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    distance_to_goal_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Context features
    score_diff_at_shot: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_leading: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_trailing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_drawing: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    minute_bucket: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Possession features
    possession_sequence_length: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    possession_duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    previous_action_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Pressure features
    recent_def_actions_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pressure_proxy_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    shot: Mapped["Shot"] = relationship("Shot", back_populates="shot_features")

    __table_args__ = (Index("ix_shot_features_version_tag", "version_tag"),)


class OpponentDefProfile(Base):
    """Opponent defensive profile table."""

    __tablename__ = "opponent_def_profile"

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(20), nullable=False)
    zone_id: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)  # NULL for global
    global_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    block_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    zone_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    shots_sample: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    team: Mapped["Team"] = relationship("Team", back_populates="opponent_profiles")

    __table_args__ = (
        UniqueConstraint("team_id", "version_tag", "zone_id", name="uq_opponent_profile"),
        Index("ix_opponent_profile_team_version", "team_id", "version_tag"),
    )


class ModelRegistry(Base):
    """Model registry table."""

    __tablename__ = "model_registry"

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(100), nullable=False)
    hyperparams: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    trained_on_version_tag: Mapped[str] = mapped_column(String(20), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    calibration_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationships
    predictions: Mapped[list["ShotPrediction"]] = relationship(
        "ShotPrediction", back_populates="model"
    )

    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_version"),
        Index("ix_model_registry_name_version", "model_name", "version"),
    )


class ShotPrediction(Base):
    """Shot predictions table."""

    __tablename__ = "shot_predictions"

    shot_id: Mapped[int] = mapped_column(ForeignKey("shots.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(20), nullable=False)
    is_neutralized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_probability: Mapped[float] = mapped_column(Float, nullable=False)
    neutral_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opponent_adjusted_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    opponent_adjusted_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Relationships
    shot: Mapped["Shot"] = relationship("Shot", back_populates="predictions")
    model: Mapped["ModelRegistry"] = relationship("ModelRegistry", back_populates="predictions")

    __table_args__ = (
        UniqueConstraint(
            "shot_id", "model_id", "is_neutralized", name="uq_shot_prediction"
        ),
        Index("ix_shot_predictions_model_id", "model_id"),
        Index("ix_shot_predictions_shot_id", "shot_id"),
    )


class AggregatesPlayer(Base):
    """Player-level aggregates table."""

    __tablename__ = "aggregates_player"

    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(20), nullable=False)
    shots_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summed_cxg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    summed_neutral_cxg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    summed_oppadj_diff: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_oppadj_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("player_id", "model_id", "version_tag", name="uq_player_aggregate"),
        Index("ix_aggregates_player_model_id", "model_id"),
    )


class AggregatesTeam(Base):
    """Team-level aggregates table."""

    __tablename__ = "aggregates_team"

    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), nullable=False)
    version_tag: Mapped[str] = mapped_column(String(20), nullable=False)
    shots_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summed_cxg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    summed_neutral_cxg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    summed_oppadj_diff: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    avg_oppadj_diff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("team_id", "model_id", "version_tag", name="uq_team_aggregate"),
        Index("ix_aggregates_team_model_id", "model_id"),
    )


class EvaluationMetric(Base):
    """Evaluation metrics table."""

    __tablename__ = "evaluation_metrics"

    model_id: Mapped[int] = mapped_column(ForeignKey("model_registry.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    slice_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    slice_filter: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_evaluation_metrics_model_id", "model_id"),
        Index("ix_evaluation_metrics_metric_name", "metric_name"),
    )
