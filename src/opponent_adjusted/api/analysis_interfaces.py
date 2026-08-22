"""Serving boundary for the oam_analysis dashboard endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FeatureInventoryRecord:
    """One feature inventory row from the oam_analysis serving store."""

    feature_family: str
    source_table: str | None
    column_name: str
    data_type: str | None
    column_role: str | None
    is_numeric: bool | None
    is_categorical: bool | None


@dataclass(frozen=True)
class FeatureCorrelationRecord:
    """One inter-feature correlation row from the oam_analysis serving store."""

    track: str
    feature_a: str
    feature_b: str
    r_train: float | None
    n_train: int
    is_redundant: bool
    resolution: str
    resolution_reason: str | None


@dataclass(frozen=True)
class UnivariateTargetRecord:
    """One univariate target-lift row from the oam_analysis serving store."""

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


@dataclass(frozen=True)
class BivariateCandidateRecord:
    """One bivariate candidate pool row from the oam_analysis serving store."""

    track: str
    feature_family: str
    column_name: str
    data_type: str
    qualification_reason: str


@dataclass(frozen=True)
class BivariateInteractionRecord:
    """One fitted bivariate interaction row from the oam_analysis serving store."""

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


@dataclass(frozen=True)
class BivariateStratifiedRecord:
    """One stratified bivariate goal-rate row from the oam_analysis serving store."""

    track: str
    tier: int
    feature_a: str
    feature_b: str
    stratum_a: str
    stratum_b: str
    n: int
    goal_count: int
    goal_rate: float | None


@dataclass(frozen=True)
class PcaComponentRecord:
    """One PCA component variance-explained row from the oam_analysis serving store."""

    track: str
    component_number: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float


@dataclass(frozen=True)
class PcaLoadingRecord:
    """One PCA per-feature loading row from the oam_analysis serving store."""

    track: str
    component_number: int
    feature_name: str
    loading: float


@dataclass(frozen=True)
class RenderedChartRecord:
    """One rendered chart registry row from the oam_analysis serving store."""

    run_id: str
    chart_name: str
    html_uri: str | None
    png_uri: str | None
    rendered_at: str | None


class AnalysisStore(Protocol):
    """Read-only contract for the oam_analysis dashboard endpoints."""

    def list_features(self, *, family: str | None = None) -> list[FeatureInventoryRecord]:
        """Return feature inventory rows, optionally filtered by family."""

    def list_correlations(self, *, family: str | None = None) -> list[FeatureCorrelationRecord]:
        """Return inter-feature correlation rows; if family given, rows where feature_a or feature_b belongs to that family."""

    def list_univariate(self, *, family: str | None = None) -> list[UnivariateTargetRecord]:
        """Return univariate target-lift rows, optionally filtered by family."""

    def list_bivariate_candidates(self) -> list[BivariateCandidateRecord]:
        """Return the bivariate candidate pool."""

    def list_bivariate_interactions(self) -> list[BivariateInteractionRecord]:
        """Return fitted bivariate interaction results."""

    def list_bivariate_stratified(self) -> list[BivariateStratifiedRecord]:
        """Return stratified bivariate goal-rate breakdowns."""

    def list_pca_components(self) -> list[PcaComponentRecord]:
        """Return PCA component variance-explained rows."""

    def list_pca_loadings(self) -> list[PcaLoadingRecord]:
        """Return PCA per-feature loading rows."""

    def get_latest_chart_run_id(self) -> str | None:
        """Return the most recently rendered run_id, or None if the registry is empty."""

    def list_charts(self, run_id: str) -> list[RenderedChartRecord]:
        """Return rendered chart rows for a specific run_id."""
