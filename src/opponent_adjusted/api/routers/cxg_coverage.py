"""CxG v3 coverage endpoint (Explore-zone: Matches/Players/Teams, guest-visible)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.cxg_coverage import (
    TRACK_TABLE_PREFIXES,
    CxgCoverageResponse,
    CxgCoverageStore,
    CxgMatchScopeRow,
    CxgMatchScopeStore,
)
from opponent_adjusted.api.dependencies import (
    Role,
    get_cxg_coverage_store,
    get_cxg_match_scope_store,
    get_role,
)

router = APIRouter(prefix="/v1/cxg", tags=["cxg"])


@router.get("/coverage", response_model=CxgCoverageResponse)
def get_cxg_coverage(
    track: str,
    event_ids: str,
    store: CxgCoverageStore = Depends(get_cxg_coverage_store),
    role: Role = Depends(get_role),
) -> CxgCoverageResponse:
    """Return CxG v3 test-set values for the given comma-separated event_ids.

    Not admin-gated — this is Explore-zone content, guest-visible per
    design_spec_v2.md §4a, unlike the Analysis tab's admin-only endpoints.
    event_ids outside v3 test-set coverage are simply absent from the
    response's `values` dict, never a placeholder — the frontend shows
    xG-only for those, per §4a.
    """
    if track not in TRACK_TABLE_PREFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"track must be one of {sorted(TRACK_TABLE_PREFIXES)}",
        )
    ids = [event_id.strip() for event_id in event_ids.split(",") if event_id.strip()]
    values = store.get_cxg_for_events(ids, track=track)
    return CxgCoverageResponse(track=track, values=values)


@router.get("/matches", response_model=list[CxgMatchScopeRow])
def get_cxg_matches(
    track: str,
    store: CxgMatchScopeStore = Depends(get_cxg_match_scope_store),
    role: Role = Depends(get_role),
) -> list[CxgMatchScopeRow]:
    """Return the matches that carry CxG v3 test-split predictions for `track`.

    Per content_spec_v3.md §9.3/§2.2a: this is how the Explore-zone "CxG
    matches only" scope toggle knows which of the 610 matches to show.
    Guest-accessible, same as /coverage — this is Explore-zone content, not
    the admin-only Analysis tab.
    """
    if track not in TRACK_TABLE_PREFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"track must be one of {sorted(TRACK_TABLE_PREFIXES)}",
        )
    return store.list_covered_matches(track=track)
