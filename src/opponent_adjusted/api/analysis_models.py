"""Pydantic response models for the oam_analysis dashboard endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FeatureInventoryResponse(BaseModel):
    """API response shape for a feature inventory row."""

    model_config = ConfigDict(from_attributes=True)

    feature_family: str
    source_table: str | None
    column_name: str
    data_type: str | None
    column_role: str | None
    is_numeric: bool | None
    is_categorical: bool | None


class FeatureCorrelationResponse(BaseModel):
    """API response shape for an inter-feature correlation row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    feature_a: str
    feature_b: str
    r_train: float | None
    n_train: int
    is_redundant: bool
    resolution: str
    resolution_reason: str | None


class UnivariateTargetResponse(BaseModel):
    """API response shape for a univariate target-lift row."""

    model_config = ConfigDict(from_attributes=True)

    feature_family: str
    column_name: str
    data_type: str | None
    row_count: int | None
    non_null_count: int | None
    goal_rate: float | None
    mean_when_available: float | None
    mean_for_goals: float | None
    mean_for_non_goals: float | None
    point_biserial_corr: float | None


class BivariateCandidateResponse(BaseModel):
    """API response shape for a bivariate candidate pool row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    feature_family: str
    column_name: str
    data_type: str
    qualification_reason: str


class BivariateInteractionResponse(BaseModel):
    """API response shape for a fitted bivariate interaction row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    tier: int
    feature_a: str
    feature_b: str
    n_train: int
    interaction_coef: float | None
    interaction_se: float | None
    interaction_p_raw: float | None
    interaction_p_fdr: float | None
    lr_stat: float | None
    main_effect_a_coef: float | None
    main_effect_b_coef: float | None
    validated_on_val_split: bool | None
    fit_status: str


class BivariateStratifiedResponse(BaseModel):
    """API response shape for a stratified bivariate goal-rate row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    tier: int
    feature_a: str
    feature_b: str
    stratum_a: str
    stratum_b: str
    n: int
    goal_count: int
    goal_rate: float | None


class PcaComponentResponse(BaseModel):
    """API response shape for a PCA component variance-explained row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    component_number: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float


class PcaLoadingResponse(BaseModel):
    """API response shape for a PCA per-feature loading row."""

    model_config = ConfigDict(from_attributes=True)

    track: str
    component_number: int
    feature_name: str
    loading: float


class RenderedChartResponse(BaseModel):
    """API response shape for a rendered chart registry row."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    chart_name: str
    html_uri: str | None
    png_uri: str | None
    rendered_at: str | None
    # Short-lived (15 min) signed HTTPS URLs for html_uri/png_uri, populated
    # by the router only when signing succeeds — kept alongside the raw
    # gs:// URIs above (not replacing them) for reference/debugging. None
    # when signing is unavailable or fails for this specific chart; the
    # frontend falls back to showing the raw URI text in that case.
    signed_html_url: str | None = None
    signed_png_url: str | None = None


class BivariateResponse(BaseModel):
    """API response shape for the composite bivariate endpoint."""

    model_config = ConfigDict(from_attributes=True)

    candidates: list[BivariateCandidateResponse]
    interactions: list[BivariateInteractionResponse]
    stratified: list[BivariateStratifiedResponse]


class PcaResponse(BaseModel):
    """API response shape for the composite PCA endpoint."""

    model_config = ConfigDict(from_attributes=True)

    components: list[PcaComponentResponse]
    loadings: list[PcaLoadingResponse]


class CxgModelResultResponse(BaseModel):
    """API response shape for one metrics row from an oam_ml model table."""

    model_config = ConfigDict(from_attributes=True)

    model_key: str
    track: str
    split: str
    model: str
    n: int
    log_loss: float | None
    brier_score: float | None
    roc_auc: float | None
    is_frozen: bool
    is_current: bool


class CxgCoefficientResponse(BaseModel):
    """API response shape for one per-feature coefficient row from an oam_ml model table."""

    model_config = ConfigDict(from_attributes=True)

    model_key: str
    track: str
    feature: str
    coefficient: float | None
    std_error: float | None
    p_value: float | None


class ChartsResponse(BaseModel):
    """API response shape for the charts endpoint."""

    model_config = ConfigDict(from_attributes=True)

    run_id: str
    charts: list[RenderedChartResponse]
