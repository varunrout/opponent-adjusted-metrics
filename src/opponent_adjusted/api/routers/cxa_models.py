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
    CxaShotCoverageResponse,
)
from opponent_adjusted.api.dependencies import (
    Role,
    get_cxa_model_store,
    get_quadrant_scatter_store,
    get_role,
)
from opponent_adjusted.api.quadrant_scatter import (
    BigQueryQuadrantScatterStore,
    PlayerCxaResponse,
    TeamCxaResponse,
)

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


@router.get("/v1/cxa/coverage-by-shot", response_model=CxaShotCoverageResponse)
def get_cxa_coverage_by_shot(
    track: str,
    shot_event_ids: str,
    store: CxaModelStore = Depends(get_cxa_model_store),
    role: Role = Depends(get_role),
) -> CxaShotCoverageResponse:
    """Guest-accessible, no admin gate -- the per-shot mirror of GET
    /v1/cxa/coverage, for the per-pass display (docs/analysis/cxa_pass_detail_v1.md):
    a shot detail view already has the shot's event_id on screen (ShotResponse),
    not the pass that created it, so this is keyed by shot_event_id. A
    shot_event_id outside test-split chance-creating-pass coverage is simply
    absent from the response's `values` dict, never a placeholder."""
    if track not in TRACKS:
        raise HTTPException(status_code=400, detail=f"track must be one of {sorted(TRACKS)}")
    ids = [sid.strip() for sid in shot_event_ids.split(",") if sid.strip()]
    values = store.get_cxa_for_shots(ids, track=track)
    return CxaShotCoverageResponse(track=track, values=values)


@router.get("/v1/cxa/player-season", response_model=PlayerCxaResponse)
def get_player_cxa(
    player_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
    store: BigQueryQuadrantScatterStore = Depends(get_quadrant_scatter_store),
    role: Role = Depends(get_role),
) -> PlayerCxaResponse:
    """Guest-accessible, no admin gate -- reuses `/v1/analysis/quadrant-scatter`'s
    own store and table (`oam_serving.player_season_cxg_cxa_v1`). That endpoint
    stays admin-gated (owned by the Analysis tab's own contract), but the
    underlying rows carry nothing sensitive -- everything in them is already
    public elsewhere (Models pages, per-shot coverage) -- so a guest-accessible
    per-player read of the same cached table is a convenience endpoint, not a
    new access boundary. See docs/analysis/cxa_players_teams_v1.md."""
    return store.get_player_cxa(player_id, competition_id=competition_id, season_id=season_id)


@router.get("/v1/cxa/team-season", response_model=TeamCxaResponse)
def get_team_cxa(
    team_id: int,
    competition_id: int | None = None,
    season_id: int | None = None,
    store: BigQueryQuadrantScatterStore = Depends(get_quadrant_scatter_store),
    role: Role = Depends(get_role),
) -> TeamCxaResponse:
    """Guest-accessible, no admin gate -- team-grain rollup over the same
    player-grain table (sum of every matched player-season row's own totals/
    counts, confirmed live to never double-count a player who appears under
    more than one team_id for the same (competition_id, season_id, split))."""
    return store.get_team_cxa(team_id, competition_id=competition_id, season_id=season_id)
