"""Public mirror of the two /v1/analysis/cxg-models* routes, for the public
Models page.

Deliberate divergence from the rest of /v1/analysis (routers/analysis.py,
fully admin-gated, unchanged by this file): this is a portfolio site, and
the CxG results table + per-model coefficients are the modelling-rigor
content the Models page exists to show any visitor, not admin-only
diagnostic surface. Everything else under /v1/analysis (features,
correlation, univariate, bivariate, pca, charts) stays admin-only — this
router only re-exposes the two routes the Models page actually needs,
via the same AnalysisStore methods routers/analysis.py itself calls, with
no admin gate.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from opponent_adjusted.api.analysis_interfaces import AnalysisStore
from opponent_adjusted.api.analysis_models import CxgCoefficientResponse, CxgModelResultResponse
from opponent_adjusted.api.dependencies import get_analysis_store

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("/cxg-models", response_model=list[CxgModelResultResponse])
def list_cxg_model_results(
    store: AnalysisStore = Depends(get_analysis_store),
) -> list[CxgModelResultResponse]:
    """Public mirror of GET /v1/analysis/cxg-models — same store method, no
    admin gate. Powers the public Models page's results table + version
    history."""
    return [CxgModelResultResponse.model_validate(r) for r in store.list_cxg_model_results()]


@router.get("/cxg-models/{model_key}/coefficients", response_model=list[CxgCoefficientResponse])
def list_cxg_model_coefficients(
    model_key: str,
    store: AnalysisStore = Depends(get_analysis_store),
) -> list[CxgCoefficientResponse]:
    """Public mirror of GET /v1/analysis/cxg-models/{model_key}/coefficients
    — same store method, no admin gate. Powers the public Models page's
    coefficients/forest-plot panel."""
    try:
        records = store.list_cxg_coefficients(model_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [CxgCoefficientResponse.model_validate(r) for r in records]
