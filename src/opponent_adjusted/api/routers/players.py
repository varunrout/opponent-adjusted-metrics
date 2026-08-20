"""Players endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import PlayerSeasonResponse, ShotResponse

router = APIRouter(prefix="/v1/players", tags=["players"])


@router.get("", response_model=list[PlayerSeasonResponse])
def list_players(
    competition_id: int | None = None,
    season_id: int | None = None,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[PlayerSeasonResponse]:
    records = store.list_player_seasons(competition_id=competition_id, season_id=season_id)
    return [PlayerSeasonResponse.model_validate(record) for record in records]


@router.get("/{player_id}/shots", response_model=list[ShotResponse])
def list_player_shots(
    player_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[ShotResponse]:
    records = store.list_player_shots(player_id, competition_id=competition_id, season_id=season_id)
    return [ShotResponse.model_validate(record) for record in records]
