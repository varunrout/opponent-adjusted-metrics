"""Shots endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.dependencies import Role, get_freeze_frame_store, get_role, get_store
from opponent_adjusted.api.freeze_frame import FreezeFrameStore, ShotFreezeFrameResponse
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


@router.get("/{match_id}/shots/{event_id}/freeze-frame", response_model=ShotFreezeFrameResponse)
def get_shot_freeze_frame(
    match_id: int,
    event_id: str,
    store: FreezeFrameStore = Depends(get_freeze_frame_store),
    role: Role = Depends(get_role),
) -> ShotFreezeFrameResponse:
    """Return the shot's real StatsBomb 360 freeze frame (teammate/opponent/GK
    positions), from oam_core.three_sixty_frames/three_sixty_players.

    Not admin-gated — same Explore-zone reasoning as /v1/cxg/coverage. 404
    (not an empty 200) when the shot has no 360 frame, matching the existing
    getMatch/ApiError 404 pattern the frontend already handles.
    """
    frame = store.get_freeze_frame(match_id, event_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="No 360 freeze frame for this shot")
    return frame
