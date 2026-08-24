"""Typed contract for `oam_analysis.cxg_defensive_profile_clusters_v1`.

Follows the `ColumnContract`/`TableContract` pattern used for Silver tables
(`opponent_adjusted.pipelines.silver.contracts`) -- reused here read-only,
not modified, per project governance.
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

CXG_DEFENSIVE_PROFILE_CLUSTERS_V1 = TableContract(
    name="cxg_defensive_profile_clusters_v1",
    key=["event_id", "cluster_model_version"],
    columns=[
        _col("event_id", "string", nullable=False),
        _col("match_id", "int64", nullable=False),
        _col("split", "string", nullable=False),  # train | validation | test, for traceability
        # Null only when the shared visible-area geometry eligibility gate failed for
        # every clustering input column (see defprofile/features.py) -- see
        # `cluster_label_null_reason` for why.
        _col("cluster_label", "int64"),
        _col("cluster_label_null_reason", "string"),
        _col("cluster_model_version", "string", nullable=False),
        _col("k", "int64", nullable=False),
        _col("feature_set_version", "string", nullable=False),
        _col("data_version", "string", nullable=False),
        _col("silver_schema_version", "string", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)
