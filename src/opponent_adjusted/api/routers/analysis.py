"""Analysis endpoints (oam_analysis dataset; admin-only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.analysis_interfaces import AnalysisStore
from opponent_adjusted.api.analysis_models import (
    BivariateCandidateResponse,
    BivariateInteractionResponse,
    BivariateResponse,
    BivariateStratifiedResponse,
    ChartsResponse,
    FeatureCorrelationResponse,
    FeatureInventoryResponse,
    PcaComponentResponse,
    PcaLoadingResponse,
    PcaResponse,
    RenderedChartResponse,
    UnivariateTargetResponse,
)
from opponent_adjusted.api.dependencies import Role, get_analysis_store, require_admin

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])


@router.get("/features", response_model=list[FeatureInventoryResponse])
def list_features(
    family: str | None = None,
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> list[FeatureInventoryResponse]:
    records = store.list_features(family=family)
    return [FeatureInventoryResponse.model_validate(record) for record in records]


@router.get("/correlation", response_model=list[FeatureCorrelationResponse])
def list_correlation(
    family: str | None = None,
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> list[FeatureCorrelationResponse]:
    records = store.list_correlations(family=family)
    return [FeatureCorrelationResponse.model_validate(record) for record in records]


@router.get("/univariate", response_model=list[UnivariateTargetResponse])
def list_univariate(
    family: str | None = None,
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> list[UnivariateTargetResponse]:
    records = store.list_univariate(family=family)
    return [UnivariateTargetResponse.model_validate(record) for record in records]


@router.get("/bivariate", response_model=BivariateResponse)
def get_bivariate(
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> BivariateResponse:
    return BivariateResponse(
        candidates=[
            BivariateCandidateResponse.model_validate(record)
            for record in store.list_bivariate_candidates()
        ],
        interactions=[
            BivariateInteractionResponse.model_validate(record)
            for record in store.list_bivariate_interactions()
        ],
        stratified=[
            BivariateStratifiedResponse.model_validate(record)
            for record in store.list_bivariate_stratified()
        ],
    )


@router.get("/pca", response_model=PcaResponse)
def get_pca(
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> PcaResponse:
    return PcaResponse(
        components=[
            PcaComponentResponse.model_validate(record) for record in store.list_pca_components()
        ],
        loadings=[
            PcaLoadingResponse.model_validate(record) for record in store.list_pca_loadings()
        ],
    )


@router.get("/charts", response_model=ChartsResponse)
def list_charts(
    run_id: str | None = None,
    store: AnalysisStore = Depends(get_analysis_store),
    role: Role = Depends(require_admin),
) -> ChartsResponse:
    resolved_run_id = run_id or store.get_latest_chart_run_id()
    if resolved_run_id is None:
        raise HTTPException(status_code=404, detail="No rendered chart runs found")
    return ChartsResponse(
        run_id=resolved_run_id,
        charts=[
            RenderedChartResponse.model_validate(record)
            for record in store.list_charts(resolved_run_id)
        ],
    )
