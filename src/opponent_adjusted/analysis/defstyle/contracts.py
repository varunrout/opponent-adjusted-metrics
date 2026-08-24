"""Typed contracts for the CxG+ Phase B defender-style clustering tables.

Follows the `ColumnContract`/`TableContract` pattern used for Silver tables
(`opponent_adjusted.pipelines.silver.contracts`) -- reused here read-only,
not modified, per project governance and matching Phase 2's
`analysis/defprofile/contracts.py`.

EXPOSURE RULE
-------------
`player_id` appears in `cxg_defender_style_clusters_v1` and ONLY there. That
table is an internal lookup keyed by player: the join from a shot to its
nearest defender's archetype needs it, so hiding it would just mean
recomputing it somewhere less auditable. It is never published into a
shot-level, model-facing table.

`GOLD_STYLE_ARCHETYPE_COLUMN` -- the single column merged into
`oam_features.cxg_defensive_360_features` -- carries the archetype STRING
alone. No `player_id`, no `team_id`, no cluster index that could be joined
back to an identity. `cxg_defensive_360_features` itself has no `player_id`
or `team_id` column today (verified against the live schema: its only
non-feature columns are `event_id`, `match_id`, `data_version`,
`feature_version`, `materialized_at`), and this phase does not add one.
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

# One row per clustered player. INTERNAL LOOKUP TABLE -- never exposed downstream.
CXG_DEFENDER_STYLE_CLUSTERS_V1 = TableContract(
    name="cxg_defender_style_clusters_v1",
    key=["player_id", "cluster_model_version"],
    columns=[
        # Internal-only identity: required to resolve a shot's nearest defender.
        _col("player_id", "int64", nullable=False),
        # Action counts and shares over the player's FULL event history (the
        # vector actually scored). `*_train` counts are the train-split-only
        # figures that gated and drove the fit.
        _col("total_actions", "int64", nullable=False),
        _col("total_actions_train", "int64", nullable=False),
        _col("pressure_share", "float64"),
        _col("duel_share", "float64"),
        _col("interception_share", "float64"),
        _col("clearance_share", "float64"),
        _col("block_share", "float64"),
        _col("foul_committed_share", "float64"),
        _col("fifty_fifty_share", "float64"),
        # NULL when the player fell below the sample-size threshold; never a
        # default or fallback assignment. See `style_archetype_null_reason`.
        _col("cluster_label", "int64"),
        _col("style_archetype", "string"),
        _col("style_archetype_null_reason", "string"),
        # True when this player's train-split vector was part of the K-Means fit.
        _col("used_in_fit", "bool", nullable=False),
        _col("cluster_model_version", "string", nullable=False),
        _col("k", "int64", nullable=False),
        _col("feature_set_version", "string", nullable=False),
        _col("data_version", "string", nullable=False),
        _col("silver_schema_version", "string", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

# One row per cluster: centroid per action-rate feature, size, archetype label.
CXG_DEFENDER_STYLE_CLUSTER_PROFILE_V1 = TableContract(
    name="cxg_defender_style_cluster_profile_v1",
    key=["cluster_label", "cluster_model_version"],
    columns=[
        _col("cluster_label", "int64", nullable=False),
        _col("style_archetype", "string", nullable=False),
        # True when the centroid carries no action distinctive enough to name,
        # or when its defining action is the annotation-density-driven 50/50
        # share. Recorded so a muddy cluster cannot be quietly consumed as if
        # it were an interpreted archetype.
        _col("is_muddy", "bool", nullable=False),
        _col("cluster_size", "int64", nullable=False),
        _col("cluster_size_train", "int64", nullable=False),
        _col("cluster_fraction", "float64", nullable=False),
        # Centroid in raw share space (interpretable) ...
        _col("pressure_share_centroid", "float64"),
        _col("duel_share_centroid", "float64"),
        _col("interception_share_centroid", "float64"),
        _col("clearance_share_centroid", "float64"),
        _col("block_share_centroid", "float64"),
        _col("foul_committed_share_centroid", "float64"),
        _col("fifty_fifty_share_centroid", "float64"),
        # ... and in scaled space (what the label was actually derived from).
        _col("pressure_share_z", "float64"),
        _col("duel_share_z", "float64"),
        _col("interception_share_z", "float64"),
        _col("clearance_share_z", "float64"),
        _col("block_share_z", "float64"),
        _col("foul_committed_share_z", "float64"),
        _col("fifty_fifty_share_z", "float64"),
        _col("dominant_feature", "string", nullable=False),
        _col("dominant_feature_z", "float64", nullable=False),
        _col("cluster_model_version", "string", nullable=False),
        _col("k", "int64", nullable=False),
        _col("feature_set_version", "string", nullable=False),
        _col("data_version", "string", nullable=False),
        _col("silver_schema_version", "string", nullable=False),
        _col("materialized_at", "string", nullable=False),
    ],
)

# The single additive column merged into the CxG+ Gold defensive feature family.
GOLD_STYLE_ARCHETYPE_COLUMN = ColumnContract(
    name="nearest_defender_style_archetype", arrow_type="string", nullable=True
)

GOLD_TARGET_TABLE = "cxg_defensive_360_features"
GOLD_TARGET_DATASET = "oam_features"

# Columns this phase must never write into a shot-level/model-facing table.
FORBIDDEN_GOLD_COLUMNS: frozenset[str] = frozenset({"player_id", "team_id"})
