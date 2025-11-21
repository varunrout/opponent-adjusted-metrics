"""Convenience analysis helpers for CxG and opponent-adjusted metrics.

These helpers sit on top of the core relational schema (see
`opponent_adjusted.db.models`) and provide simple access patterns for
common analyses:

- Shot-level CxG vs neutralized CxG diagnostics
- Player-level aggregates (summed CxG, neutral CxG, opponent-adjusted diffs)
- Team-level aggregates

The intent is that these functions stay thin: they should express
*what* we want to compute in terms of the public ORM models, and leave
heavier modeling logic to the `modeling` / `pipelines` layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import (
    AggregatesPlayer,
    AggregatesTeam,
    ModelRegistry,
    Shot,
    ShotPrediction,
)


@dataclass
class ShotLevelCxGRecord:
    """Lightweight view of shot-level CxG information."""

    shot_id: int
    match_id: int
    team_id: int
    player_id: Optional[int]
    opponent_team_id: int
    outcome: str
    statsbomb_xg: Optional[float]
    raw_probability: float
    neutral_probability: Optional[float]
    opponent_adjusted_diff: Optional[float]
    opponent_adjusted_ratio: Optional[float]


@dataclass
class PlayerCxGSummary:
    """Player-level CxG summary for a particular model and version_tag."""

    player_id: int
    model_id: int
    version_tag: str
    shots_count: int
    summed_cxg: float
    summed_neutral_cxg: float
    summed_oppadj_diff: float
    avg_oppadj_diff: Optional[float]


@dataclass
class TeamCxGSummary:
    """Team-level CxG summary for a particular model and version_tag."""

    team_id: int
    model_id: int
    version_tag: str
    shots_count: int
    summed_cxg: float
    summed_neutral_cxg: float
    summed_oppadj_diff: float
    avg_oppadj_diff: Optional[float]


def _resolve_model_id(
    session: Session,
    model_name: str | None = None,
    version: str | None = None,
) -> int:
    """Resolve a model_id from the registry."""

    name = model_name or "cxg"

    stmt = select(ModelRegistry).where(ModelRegistry.model_name == name)
    if version is not None:
        stmt = stmt.where(ModelRegistry.version == version)

    stmt = stmt.order_by(ModelRegistry.version.desc())

    model = session.execute(stmt).scalars().first()
    if model is None:
        raise ValueError(f"No model found in registry for name={name!r}, version={version!r}")

    return model.id


def compute_shot_level_cxg(
    session: Session,
    *,
    model_name: str | None = None,
    version: str | None = None,
    only_neutralized: bool | None = None,
    limit: Optional[int] = None,
) -> List[ShotLevelCxGRecord]:
    """Return shot-level CxG diagnostics for a given model."""

    model_id = _resolve_model_id(session, model_name=model_name, version=version)

    stmt = (
        select(
            Shot.id.label("shot_id"),
            Shot.match_id,
            Shot.team_id,
            Shot.player_id,
            Shot.opponent_team_id,
            Shot.outcome,
            Shot.statsbomb_xg,
            ShotPrediction.raw_probability,
            ShotPrediction.neutral_probability,
            ShotPrediction.opponent_adjusted_diff,
            ShotPrediction.opponent_adjusted_ratio,
        )
        .join(ShotPrediction, ShotPrediction.shot_id == Shot.id)
        .where(ShotPrediction.model_id == model_id)
    )

    if only_neutralized is True:
        stmt = stmt.where(ShotPrediction.is_neutralized.is_(True))
    elif only_neutralized is False:
        stmt = stmt.where(ShotPrediction.is_neutralized.is_(False))

    if limit is not None:
        stmt = stmt.limit(limit)

    rows = session.execute(stmt).all()

    return [
        ShotLevelCxGRecord(
            shot_id=row.shot_id,
            match_id=row.match_id,
            team_id=row.team_id,
            player_id=row.player_id,
            opponent_team_id=row.opponent_team_id,
            outcome=row.outcome,
            statsbomb_xg=row.statsbomb_xg,
            raw_probability=row.raw_probability,
            neutral_probability=row.neutral_probability,
            opponent_adjusted_diff=row.opponent_adjusted_diff,
            opponent_adjusted_ratio=row.opponent_adjusted_ratio,
        )
        for row in rows
    ]


def summarize_player_cxg(
    session: Session,
    *,
    model_name: str | None = None,
    version: str | None = None,
    limit: Optional[int] = None,
) -> List[PlayerCxGSummary]:
    """Return player-level CxG summaries from `AggregatesPlayer`."""

    model_id = _resolve_model_id(session, model_name=model_name, version=version)

    stmt = select(AggregatesPlayer).where(AggregatesPlayer.model_id == model_id)

    if limit is not None:
        stmt = stmt.order_by(AggregatesPlayer.summed_cxg.desc()).limit(limit)

    rows: Sequence[AggregatesPlayer] = session.execute(stmt).scalars().all()

    return [
        PlayerCxGSummary(
            player_id=row.player_id,
            model_id=row.model_id,
            version_tag=row.version_tag,
            shots_count=row.shots_count,
            summed_cxg=row.summed_cxg,
            summed_neutral_cxg=row.summed_neutral_cxg,
            summed_oppadj_diff=row.summed_oppadj_diff,
            avg_oppadj_diff=row.avg_oppadj_diff,
        )
        for row in rows
    ]


def summarize_team_cxg(
    session: Session,
    *,
    model_name: str | None = None,
    version: str | None = None,
    limit: Optional[int] = None,
) -> List[TeamCxGSummary]:
    """Return team-level CxG summaries from `AggregatesTeam`."""

    model_id = _resolve_model_id(session, model_name=model_name, version=version)

    stmt = select(AggregatesTeam).where(AggregatesTeam.model_id == model_id)

    if limit is not None:
        stmt = stmt.order_by(AggregatesTeam.summed_cxg.desc()).limit(limit)

    rows: Sequence[AggregatesTeam] = session.execute(stmt).scalars().all()

    return [
        TeamCxGSummary(
            team_id=row.team_id,
            model_id=row.model_id,
            version_tag=row.version_tag,
            shots_count=row.shots_count,
            summed_cxg=row.summed_cxg,
            summed_neutral_cxg=row.summed_neutral_cxg,
            summed_oppadj_diff=row.summed_oppadj_diff,
            avg_oppadj_diff=row.avg_oppadj_diff,
        )
        for row in rows
    ]
