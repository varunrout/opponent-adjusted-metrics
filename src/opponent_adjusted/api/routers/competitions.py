"""Competitions endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import CompetitionResponse

router = APIRouter(prefix="/v1/competitions", tags=["competitions"])


@router.get("", response_model=list[CompetitionResponse])
def list_competitions(
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[CompetitionResponse]:
    return [CompetitionResponse.model_validate(record) for record in store.list_competitions()]
