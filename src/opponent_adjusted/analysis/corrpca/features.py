"""Candidate pools and redundancy resolution for the correlation/redundancy screen.

Pipeline stage order (locked, supersedes `docs/cxg_split_policy_and_parallel_plan.md`'s
original ordering): Summary Statistics -> EDA -> Univariate -> **Correlation/Redundancy
Screen (this module) -> PCA if applicable (this module)** -> Bivariate (separate, later
task). This module never drops a feature for weak/no univariate signal -- only true
near-duplicate redundancy (train r >= 0.85) is grounds for trimming, and even then only
one member of the pair unless there is precedent to drop both (see DROP_BOTH_PRECEDENT).

All correlations below are real numbers computed against `cxg_event_model_matrix_v1` /
`cxg_plus_360_model_matrix_v1`, train split only, and verified against the numbers already
computed earlier in this session before this module was written -- see
`audit_outputs/cxg_analysis/correlation_screen_pca/correlation_screen_pca_report.md` for the
full verification trail and reasoning per pair.
"""

from __future__ import annotations

REDUNDANCY_THRESHOLD = 0.85

# --- Step 0: univariate-qualified pools, re-verified this run ---

# CxG track (cxg_event, 15,737-shot full population). Unchanged by this module -- this is
# the pool CxG's own bivariate work will use, untouched by any 360-cohort recheck.
CXG_EVENT_QUALIFIED = (
    "shot_x_sb",
    "last_action_interval_s",
    "previous_action_time_gap_s",
    "last_box_entry_to_shot_s",
    "regain_height_speed_interaction",
    "first_box_entry_to_shot_s",
)

# Of the 6 CxG-origin candidates, re-verified for sign-stability + min(|r|)>=0.05 SPECIFICALLY
# within the 3,960-shot 360-only cohort (cxg_plus_360_model_matrix_v1's own splits) -- the
# 360-tracked matches are not a random sample of all matches. Only 3 of 6 hold up:
#   - shot_x_sb: train=0.193, val=0.208, test=0.154 -- holds (min|r|=0.154)
#   - last_action_interval_s: train=-0.108, val=-0.090, test=-0.088 -- holds (min|r|=0.088)
#   - previous_action_time_gap_s: train=-0.094, val=-0.083, test=-0.071 -- holds (min|r|=0.071)
#   - last_box_entry_to_shot_s: train=-0.089, val=-0.033, test=-0.105 -- FAILS (val min|r|=0.033 < 0.05)
#   - regain_height_speed_interaction: train=+0.035, val=-0.0002, test=+0.168 -- FAILS (sign flips)
#   - first_box_entry_to_shot_s: train=-0.069, val=-0.029, test=-0.106 -- FAILS (val min|r|=0.029 < 0.05)
CXG_ORIGIN_REVERIFIED_FOR_PLUS = (
    "shot_x_sb",
    "last_action_interval_s",
    "previous_action_time_gap_s",
)

CXG_PLUS_NATIVE_NUMERIC_QUALIFIED = (
    "visible_goal_angle_proxy",
    "gk_distance_to_shooter",
    "defensive_reset_index",
    "rest_defence_reset_fraction",
    "goal_mouth_defender_count",
    "estimated_goalface_occlusion",
    "defenders_between_ball_and_goal",
    "defensive_line_depth",
    "shot_corridor_occlusion",
    "shooter_space_previous_linked_event",
    "defensive_hull_area",
    "defensive_centroid_x",
    "min_defensive_compactness_sequence",
    "pre_shot_receiver_space",
    "nearest_defender_distance_delta",
    "defensive_compactness",
    "defensive_width",
    "max_goal_exposure",
)

# Deliberately included despite weak/unstable univariate signal (0/3 sign-stable across
# splits) -- per the non-negotiable principle, ODI is kept in the pool specifically to test
# whether its weak univariate signal is context-conditional in bivariate.
CXG_PLUS_ODI_TRIO = ("nearest_defender_odi", "mean_backline_odi", "gk_odi")

# Categorical, proven stable via Phase 2's own descriptive analysis (train-only goal-rate
# separation replicated across splits) -- not eligible for the numeric point-biserial
# univariate table by construction, but a qualified CxG+ candidate regardless.
CXG_PLUS_CATEGORICAL = ("defensive_profile_cluster",)

# Full pre-redundancy-trim CxG+ pool: 3 (reverified inherited) + 18 (native numeric)
# + 1 (categorical) + 3 (ODI) = 25.
CXG_PLUS_PRE_TRIM_NUMERIC = CXG_ORIGIN_REVERIFIED_FOR_PLUS + CXG_PLUS_NATIVE_NUMERIC_QUALIFIED + CXG_PLUS_ODI_TRIO

# --- Step 2: redundancy resolution (train r >= 0.85 pairs only) ---
# Each entry: (feature_a, feature_b, r_train, track, drop, reason)
# `drop` is a tuple of the feature(s) removed from the final pool for THIS pair (empty if
# neither -- not used here since every listed pair is >=0.85 by construction).

# Phase 2 precedent, applied for consistency (same features, same call):
DROP_BOTH_PRECEDENT = {
    frozenset({"defensive_compactness", "defensive_hull_area"}): (
        "defensive_compactness",
        "defensive_hull_area",
    )
}

REDUNDANT_PAIRS = [
    # (track, feature_a, feature_b, r_train, dropped, reason)
    (
        "cxg_event",
        "last_action_interval_s",
        "previous_action_time_gap_s",
        0.8639,
        ("previous_action_time_gap_s",),
        "Tie-break (no precedent): last_action_interval_s has higher min(|r_train|,|r_val|,|r_test|) "
        "in the CxG track (0.0922 vs 0.0842). Also different governed families (E5 Possession Tempo "
        "vs E10 Immediate Pre-Shot Sequence) despite similar naming -- not a naming coincidence, a "
        "real redundant pair.",
    ),
    (
        "cxg_plus",
        "defensive_reset_index",
        "rest_defence_reset_fraction",
        0.9959,
        ("rest_defence_reset_fraction",),
        "Tie-break (no precedent, highest r in the matrix): defensive_reset_index has higher "
        "min(|r_train|,|r_val|,|r_test|) (0.1706 vs 0.1678).",
    ),
    (
        "cxg_plus",
        "defensive_compactness",
        "defensive_hull_area",
        0.9167,
        ("defensive_compactness", "defensive_hull_area"),
        "Phase 2 precedent (defprofile/features.py): defensive_compactness = defensive_width * "
        "defensive_length (literal product of a primitive already in a related pool), "
        "defensive_hull_area a correlated composite of the same footprint -- drop both, keep the "
        "primitive defensive_width (defensive_length is not in this pool's qualified candidates).",
    ),
    (
        "cxg_plus",
        "gk_distance_to_shooter",
        "shot_x_sb",
        -0.8998,
        ("shot_x_sb",),
        "Tie-break (no precedent, newly surfaced by the Step 0 reverification -- not in the original "
        "given baseline, which predates shot_x_sb being confirmed as a reverified CxG+ candidate): "
        "gk_distance_to_shooter has higher min(|r_train|,|r_val|,|r_test|) within the 360-cohort "
        "(0.2132 vs 0.1544). shot_x_sb is NOT dropped from the CxG track's own pool -- this redundancy "
        "is specific to the 360-cohort, where GK positioning and shot x-position are mechanically linked.",
    ),
    (
        "cxg_plus",
        "pre_shot_receiver_space",
        "shooter_space_previous_linked_event",
        0.8624,
        ("shooter_space_previous_linked_event",),
        "Tie-break (no precedent): near-tied on min(|r_train|,|r_val|,|r_test|) (0.0822 vs 0.0829, "
        "within noise) -- resolved via governed-taxonomy primacy: both are F10 'Shooter / Receiver "
        "Space Evolution', but pre_shot_receiver_space is a direct state measurement while "
        "shooter_space_previous_linked_event is explicitly a derived/relational lookback "
        "(references a prior linked event) -- kept the more primary one.",
    ),
    (
        "cxg_plus",
        "defensive_centroid_x",
        "defensive_line_depth",
        0.8524,
        ("defensive_centroid_x",),
        "Phase 2 precedent (defprofile/features.py): kept defensive_line_depth (governed F2 primary "
        "depth name), dropped defensive_centroid_x.",
    ),
    (
        "cxg_plus",
        "last_action_interval_s",
        "previous_action_time_gap_s",
        0.8516,
        ("previous_action_time_gap_s",),
        "Same underlying feature pair as the cxg_event resolution above, recomputed within the "
        "360-cohort (r differs slightly: 0.8516 vs 0.8639) -- same call applied for consistency "
        "(last_action_interval_s has higher min|r| in the CxG+-population-specific univariate check too: "
        "0.088 vs 0.071).",
    ),
]

# Below-threshold but worth documenting explicitly (three-way relationship, per task): none of
# these are grounds for trimming (all r_train < 0.85), kept as-is. shooter_space_previous_linked_event
# is still removed from the final pool, but because of ITS OWN >=0.85 pair above, not because of
# these two moderate correlations.
NOTED_MODERATE_TRIOS = [
    ("cxg_plus", "nearest_defender_distance_delta", "shooter_space_previous_linked_event", -0.7919),
    ("cxg_plus", "nearest_defender_distance_delta", "pre_shot_receiver_space", -0.6936),
]


def dropped_features(track: str) -> set[str]:
    return {f for t, _, _, _, dropped, _ in REDUNDANT_PAIRS if t == track for f in dropped}


def final_cxg_event_pool() -> tuple[str, ...]:
    dropped = dropped_features("cxg_event")
    return tuple(f for f in CXG_EVENT_QUALIFIED if f not in dropped)


def final_cxg_plus_numeric_pool() -> tuple[str, ...]:
    dropped = dropped_features("cxg_plus")
    return tuple(f for f in CXG_PLUS_PRE_TRIM_NUMERIC if f not in dropped)
