"""Typed contracts for the correlation/redundancy screen + PCA analysis tables.

Follows the `ColumnContract`/`TableContract` pattern used for Silver tables
(`opponent_adjusted.pipelines.silver.contracts`) -- reused here read-only, matching
`defprofile/contracts.py`'s precedent for Analysis-layer (`oam_analysis`) tables.
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

CXG_FEATURE_CORRELATION_V1 = TableContract(
    name="cxg_feature_correlation_v1",
    key=["track", "feature_a", "feature_b"],
    columns=[
        _col("track", "string", nullable=False),  # cxg_event | cxg_plus
        _col("feature_a", "string", nullable=False),
        _col("feature_b", "string", nullable=False),
        _col("r_train", "float64"),
        _col("n_train", "int64", nullable=False),
        _col("is_redundant", "bool", nullable=False),  # r_train >= 0.85 (abs)
        # kept_both_moderate | dropped_a | dropped_b | dropped_both
        _col("resolution", "string", nullable=False),
        _col("resolution_reason", "string"),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_BIVARIATE_CANDIDATE_POOL_V1 = TableContract(
    name="cxg_bivariate_candidate_pool_v1",
    key=["track", "column_name"],
    columns=[
        _col("track", "string", nullable=False),
        _col("feature_family", "string", nullable=False),
        _col("column_name", "string", nullable=False),
        _col("data_type", "string", nullable=False),
        # univariate_stable | deliberately_included_despite_weak_signal
        # | categorical_proven_stable | inherited_from_cxg_reverified
        _col("qualification_reason", "string", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_PCA_COMPONENTS_V1 = TableContract(
    name="cxg_pca_components_v1",
    key=["track", "component_number"],
    columns=[
        _col("track", "string", nullable=False),
        _col("component_number", "int64", nullable=False),
        _col("explained_variance_ratio", "float64", nullable=False),
        _col("cumulative_variance_ratio", "float64", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_PCA_LOADINGS_V1 = TableContract(
    name="cxg_pca_loadings_v1",
    key=["track", "component_number", "feature_name"],
    columns=[
        _col("track", "string", nullable=False),
        _col("component_number", "int64", nullable=False),
        _col("feature_name", "string", nullable=False),
        _col("loading", "float64", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)
