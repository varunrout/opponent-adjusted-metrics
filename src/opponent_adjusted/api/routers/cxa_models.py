"""CxA combined-scorer public routes: Models-page comparison
(`GET /v1/models/cxa-models`) and per-pass coverage (`GET /v1/cxa/coverage`).

Mirrors `routers/models.py` (public, no admin gate) and `routers/cxg_coverage.py`
(guest-accessible coverage lookup) exactly, per this task's instructions. No new
patterns invented.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.cxa_models import (
    TRACKS,
    CxaCoverageResponse,
    CxaExplainability,
    CxaModelStore,
    CxaModelSummary,
)
from opponent_adjusted.api.dependencies import Role, get_cxa_model_store, get_role

router = APIRouter(tags=["cxa"])


@router.get("/v1/models/cxa-models", response_model=list[CxaModelSummary])
def list_cxa_model_summaries(
    store: CxaModelStore = Depends(get_cxa_model_store),
) -> list[CxaModelSummary]:
    """Public, no admin gate -- mirrors GET /v1/models/cxg-models. Returns one
    summary per track (event, plus), each with P_create's and P_convert's own
    test-split metrics kept separate and clearly stage-labelled (never unioned
    into one fake "CxA model" number, per decision 5), plus coverage stats and
    the standing ~2% chance-creating-coverage caveat (decision 1)."""
    return store.list_model_summaries()


@router.get("/v1/models/cxa-models/{track}/explainability", response_model=CxaExplainability)
def get_cxa_explainability(
    track: str,
    store: CxaModelStore = Depends(get_cxa_model_store),
) -> CxaExplainability:
    """Public, no admin gate -- mirrors GET /v1/models/cxg-models/{model_key}/
    coefficients' nesting style. Feature importances for whichever of this track's
    two stages are tree-family (P_create always; P_convert on event-only), and
    coefficients for whichever are logistic-family (P_convert on CxA+ only) --
    never both for the same stage, and never a fabricated coefficients row for a
    tree-family stage."""
    if track not in TRACKS:
        raise HTTPException(status_code=400, detail=f"track must be one of {sorted(TRACKS)}")
    try:
        return store.get_explainability(track)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/v1/cxa/coverage", response_model=CxaCoverageResponse)
def get_cxa_coverage(
    track: str,
    pass_event_ids: str,
    store: CxaModelStore = Depends(get_cxa_model_store),
    role: Role = Depends(get_role),
) -> CxaCoverageResponse:
    """Guest-accessible, no admin gate -- mirrors GET /v1/cxg/coverage. Returns
    CxA test-split values for the given comma-separated pass_event_ids.
    pass_event_ids outside test-split coverage are simply absent from the
    response's `values` dict, never a placeholder; for a covered pass that never
    created a chance, `p_convert_predicted_prob`/`cxa_combined_score` are `null`
    in the response, never 0 or a placeholder."""
    if track not in TRACKS:
        raise HTTPException(status_code=400, detail=f"track must be one of {sorted(TRACKS)}")
    ids = [pid.strip() for pid in pass_event_ids.split(",") if pid.strip()]
    values = store.get_cxa_for_passes(ids, track=track)
    return CxaCoverageResponse(track=track, values=values)
