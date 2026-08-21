"""Typed contracts for bivariate interaction-testing analysis tables.

Follows the `ColumnContract`/`TableContract` pattern used elsewhere in the analysis layer
(see `corrpca/contracts.py` / `defprofile/contracts.py` for precedent).
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

CXG_BIVARIATE_INTERACTION_V1 = TableContract(
    name="cxg_bivariate_interaction_v1",
    key=["track", "tier", "feature_a", "feature_b"],
    columns=[
        _col("track", "string", nullable=False),  # cxg_event | cxg_plus
        _col("tier", "int64", nullable=False),  # 1-4
        _col("feature_a", "string", nullable=False),
        _col("feature_b", "string", nullable=False),
        _col("n_train", "int64", nullable=False),
        # NULL when the interaction term is multi-df (categorical feature involved) --
        # see interaction_p_raw/lr_stat for the joint test in that case.
        _col("interaction_coef", "float64"),
        _col("interaction_se", "float64"),
        _col("interaction_p_raw", "float64"),
        _col("interaction_p_fdr", "float64"),
        _col("lr_stat", "float64"),
        # NULL when the corresponding feature is categorical (multi-level main effect,
        # no single coefficient to report).
        _col("main_effect_a_coef", "float64"),
        _col("main_effect_b_coef", "float64"),
        # NULL = never tested on validation (didn't clear the promotion bar for this tier).
        # TRUE = tested and effect direction/significance held. FALSE = tested and did not hold.
        _col("validated_on_val_split", "bool"),
        _col("fit_status", "string", nullable=False),  # ok | insufficient_data | fit_failed
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_BIVARIATE_STRATIFIED_V1 = TableContract(
    name="cxg_bivariate_stratified_v1",
    key=["track", "tier", "feature_a", "feature_b", "stratum_a", "stratum_b"],
    columns=[
        _col("track", "string", nullable=False),
        _col("tier", "int64", nullable=False),
        _col("feature_a", "string", nullable=False),
        _col("feature_b", "string", nullable=False),
        _col("stratum_a", "string", nullable=False),
        _col("stratum_b", "string", nullable=False),
        _col("n", "int64", nullable=False),
        _col("goal_count", "int64", nullable=False),
        _col("goal_rate", "float64"),
        _col("materialized_at", "string", nullable=False),
    ],
)
