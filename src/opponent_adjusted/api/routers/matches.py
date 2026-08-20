"""Matches endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import MatchResponse

router = APIRouter(prefix="/v1/matches", tags=["matches"])


@router.get("", response_model=list[MatchResponse])
def list_matches(
    competition_id: int | None = None,
    season_id: int | None = None,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[MatchResponse]:
    records = store.list_matches(competition_id=competition_id, season_id=season_id)
    return [MatchResponse.model_validate(record) for record in records]
