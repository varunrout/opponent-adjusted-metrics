"""Matches endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.dependencies import Role, get_role, get_store
from opponent_adjusted.api.interfaces import ServingStore
from opponent_adjusted.api.models import LineupPlayerResponse, MatchDetailResponse, MatchResponse

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


@router.get("/{match_id}", response_model=MatchDetailResponse)
def get_match(
    match_id: int,
    store: ServingStore = Depends(get_store),
    role: Role = Depends(get_role),
) -> MatchDetailResponse:
    match = store.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    lineups = store.list_lineups(match_id)
    return MatchDetailResponse(
        **MatchResponse.model_validate(match).model_dump(),
        lineups=[LineupPlayerResponse.model_validate(row) for row in lineups],
    )
