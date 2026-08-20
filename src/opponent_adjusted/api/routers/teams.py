"""Teams endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import ShotResponse, TeamSeasonResponse

router = APIRouter(prefix="/v1/teams", tags=["teams"])


@router.get("", response_model=list[TeamSeasonResponse])
def list_teams(
    competition_id: int | None = None,
    season_id: int | None = None,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[TeamSeasonResponse]:
    records = store.list_team_seasons(competition_id=competition_id, season_id=season_id)
    return [TeamSeasonResponse.model_validate(record) for record in records]


@router.get("/{team_id}/shots", response_model=list[ShotResponse])
def list_team_shots(
    team_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[ShotResponse]:
    records = store.list_team_shots(team_id, competition_id=competition_id, season_id=season_id)
    return [ShotResponse.model_validate(record) for record in records]
