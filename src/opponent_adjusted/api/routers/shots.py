"""Shots endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import ShotResponse

router = APIRouter(prefix="/v1/matches", tags=["shots"])


@router.get("/{match_id}/shots", response_model=list[ShotResponse])
def list_shots(
    match_id: int,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> list[ShotResponse]:
    return [ShotResponse.model_validate(record) for record in store.list_shots(match_id)]
