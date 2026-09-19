"""Typed contracts for the ODI (On-pitch Defensive Index) Analysis-layer tables.

Follows the `ColumnContract`/`TableContract` pattern used for Silver tables
(`opponent_adjusted.pipelines.silver.contracts`) -- reused here read-only, not
modified, per project governance. These two tables live in `oam_analysis`
(BigQuery dataset `oam-varun-260819.oam_analysis`), not `oam_core`.
"""

from __future__ import annotations

from opponent_adjusted.pipelines.silver.contracts import ColumnContract, TableContract

_col = ColumnContract

ANALYSIS_LINEAGE = [
    _col("data_version", "string", nullable=False),
    _col("silver_schema_version", "string", nullable=False),
    _col("feature_version", "string", nullable=False),
    _col("materialized_at", "string", nullable=False),
]

# Position names (StatsBomb taxonomy) treated as "back line" for mean back-line ODI.
# Any position_name containing one of these substrings (case-sensitive, StatsBomb's
# own naming) is classified as a back-line defender. Excludes the goalkeeper, who is
# reported separately as `gk_odi`.
BACKLINE_POSITION_MARKERS: tuple[str, ...] = ("Back",)

# Phase A geometric/categorical defender features (`nearest_defender_role` /
# `second_nearest_defender_role`, cxg_defensive_360_features): a finer 5-bucket role
# taxonomy, extending the same case-sensitive substring-matching convention as
# BACKLINE_POSITION_MARKERS above rather than inventing a new classification scheme. Checked
# in the order below -- "Center Back" must be excluded before the generic "Back" check, or
# every center-back would misclassify as Fullback_WingBack (both contain "Back").
GK_POSITION_NAME = "Goalkeeper"
CENTER_BACK_MARKERS: tuple[str, ...] = ("Center Back",)
FULLBACK_WINGBACK_MARKERS: tuple[str, ...] = ("Back",)
MIDFIELD_MARKERS: tuple[str, ...] = ("Midfield",)


def bucket_defender_role(position_name: str | None) -> str | None:
    """GK / CB / Fullback_WingBack / Midfield / Attack. Returns None for a missing
    `position_name` -- never guessed. Verified against every distinct `position_name`
    value observed in `oam_core.shot_freeze_frame_players` (25 values, StatsBomb taxonomy):
    every one falls cleanly into exactly one bucket with this ordering."""
    if position_name is None:
        return None
    if position_name == GK_POSITION_NAME:
        return "GK"
    if any(m in position_name for m in CENTER_BACK_MARKERS):
        return "CB"
    if any(m in position_name for m in FULLBACK_WINGBACK_MARKERS):
        return "Fullback_WingBack"
    if any(m in position_name for m in MIDFIELD_MARKERS):
        return "Midfield"
    return "Attack"


CXG_DEFENSIVE_INVOLVEMENT_V1 = TableContract(
    name="cxg_defensive_involvement_v1",
    key=["shot_event_id", "data_version", "silver_schema_version"],
    columns=[
        _col("shot_event_id", "string", nullable=False),
        _col("match_id", "int64", nullable=False),
        _col("competition_id", "int64", nullable=False),
        _col("season_id", "int64", nullable=False),
        _col("period", "int64"),
        _col("shot_timestamp_seconds", "float64"),
        # Nearest defender identity (the "involvement" -- this shot counts against
        # this defender's rolling ODI as time passes).
        _col("player_id", "int64", nullable=False),
        _col("team_id", "int64"),  # defending team_id (NOT the shooting team)
        _col("position_id", "int64"),
        _col("position_name", "string"),
        _col("nearest_defender_distance_native", "float64"),
        # The shot outcome this defender is charged with.
        _col("statsbomb_xg", "float64"),
        _col("is_goal", "bool"),
        *ANALYSIS_LINEAGE,
    ],
)

CXG_ODI_FEATURES_V1 = TableContract(
    name="cxg_odi_features_v1",
    key=["event_id", "data_version", "silver_schema_version"],
    columns=[
        _col("event_id", "string", nullable=False),
        _col("match_id", "int64", nullable=False),
        _col("team_id", "int64"),  # shooting team, for join convenience
        _col("period", "int64"),
        _col("shot_timestamp_seconds", "float64"),
        _col("has_freeze_frame", "bool", nullable=False),
        # Nearest-defender ODI.
        _col("nearest_defender_player_id", "int64"),
        _col("nearest_defender_odi", "float64"),
        _col("nearest_defender_odi_eligible", "bool", nullable=False),
        _col("nearest_defender_odi_involvement_count", "int64"),
        _col("nearest_defender_odi_null_reason", "string"),
        # Mean back-line ODI (defenders whose position_name matches BACKLINE_POSITION_MARKERS).
        _col("backline_defender_count", "int64", nullable=False),
        _col("backline_odi_eligible_count", "int64", nullable=False),
        _col("mean_backline_odi", "float64"),
        _col("mean_backline_odi_null_reason", "string"),
        # GK ODI (position_name == "Goalkeeper" among defending freeze-frame players).
        _col("gk_player_id", "int64"),
        _col("gk_odi", "float64"),
        _col("gk_odi_eligible", "bool", nullable=False),
        _col("gk_odi_null_reason", "string"),
        *ANALYSIS_LINEAGE,
    ],
)
