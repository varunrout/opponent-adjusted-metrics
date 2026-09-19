"""Typed contracts for the dumb-baseline / v1 kitchen-sink modeling tables.

Follows the `ColumnContract`/`TableContract` pattern used elsewhere in the analysis layer
(see `bivariate/contracts.py` for the most recent precedent). Written to `oam_ml`, not
`oam_analysis` -- these are model outputs/evaluation results, not feature-analysis tables.
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

CXG_BASELINE_V1_PREDICTIONS = TableContract(
    name="cxg_baseline_v1_predictions",
    key=["track", "event_id"],
    columns=[
        _col("track", "string", nullable=False),  # cxg_event | cxg_plus
        _col("event_id", "string", nullable=False),
        _col("split", "string", nullable=False),  # train | validation | test
        _col("dumb_baseline_prob", "float64", nullable=False),
        _col("v1_predicted_prob", "float64", nullable=False),
        # NULL for shots with no StatsBomb xG value (rare but not assumed impossible).
        _col("statsbomb_xg", "float64"),
        _col("is_goal", "bool", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_BASELINE_V1_METRICS = TableContract(
    name="cxg_baseline_v1_metrics",
    key=["track", "split", "model"],
    columns=[
        _col("track", "string", nullable=False),
        _col("split", "string", nullable=False),  # validation | test
        _col("model", "string", nullable=False),  # dumb_baseline | v1 | statsbomb_xg
        _col("n", "int64", nullable=False),
        _col("log_loss", "float64"),
        _col("brier_score", "float64"),
        # NULL for the dumb baseline (a constant predictor has no ROC-AUC -- undefined,
        # not 0.5; see report for why reporting 0.5 there would be misleading).
        _col("roc_auc", "float64"),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_BASELINE_V1_COEFFICIENTS = TableContract(
    name="cxg_baseline_v1_coefficients",
    key=["track", "feature"],
    columns=[
        _col("track", "string", nullable=False),
        _col("feature", "string", nullable=False),  # includes "const" and one-hot dummy names
        _col("coefficient", "float64", nullable=False),
        _col("std_error", "float64", nullable=False),
        _col("p_value", "float64", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

CXG_BASELINE_V1_DIVERGENCE = TableContract(
    name="cxg_baseline_v1_divergence",
    key=["stratum_type", "stratum_value"],
    columns=[
        # CxG+ only (CxG has no opponent-adjustment features for this comparison to test).
        _col("stratum_type", "string", nullable=False),  # defensive_profile_cluster | odi_tercile_<feature>
        _col("stratum_value", "string", nullable=False),
        _col("n", "int64", nullable=False),
        _col("mean_divergence", "float64", nullable=False),  # mean(v1_predicted_prob - statsbomb_xg)
        _col("mean_v1_prob", "float64", nullable=False),
        _col("mean_statsbomb_xg", "float64", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)
