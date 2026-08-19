"""Governed CxG/CxG+ contextual candidate taxonomy.

This contract records candidate membership only. It intentionally does not
define physical shot-state features, derivation formulas, or a feature version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceType = Literal["event", "360"]

CXG_CONTEXT_TAXONOMY_ID = "cxg_context_taxonomy_v3"


@dataclass(frozen=True)
class FeatureFamily:
    """A governed contextual candidate family."""

    family_id: str
    name: str
    source_type: SourceType
    candidates: tuple[str, ...]
    methodology_flags: tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


EVENT_FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        "E1",
        "Match State Context",
        "event",
        (
            "score_diff",
            "game_state",
            "match_minute",
            "regulation_time_remaining",
            "manpower_diff",
            "late_game_trailing",
            "late_game_leading",
        ),
    ),
    FeatureFamily(
        "E2",
        "Possession Origin Context",
        "event",
        (
            "possession_start_x",
            "possession_start_y",
            "possession_start_goal_distance",
            "possession_start_zone",
            "possession_start_type",
            "high_regain",
            "deep_regain",
            "restart_vs_live_regain",
        ),
    ),
    FeatureFamily(
        "E3",
        "Possession Age / Transition-Time Context",
        "event",
        (
            "possession_age_s",
            "shot_within_5s_regain",
            "shot_within_10s_regain",
            "shot_within_15s_regain",
        ),
    ),
    FeatureFamily(
        "E4",
        "Possession Complexity Context",
        "event",
        (
            "possession_action_count",
            "possession_pass_count",
            "possession_carry_count",
            "possession_dribble_count",
            "unique_attackers_involved",
            "recorded_receipt_count",
        ),
    ),
    FeatureFamily(
        "E5",
        "Possession Tempo Context",
        "event",
        (
            "avg_action_interval_s",
            "last_action_interval_s",
            "actions_per_second",
        ),
    ),
    FeatureFamily(
        "E6",
        "Progression & Directness Context",
        "event",
        (
            "goalward_progress_sb",
            "recorded_event_path_length_sb",
            "possession_directness_proxy",
            "pace_to_goal",
            "progressive_actions_n",
            "max_single_action_progress_sb",
        ),
    ),
    FeatureFamily(
        "E7",
        "Territory Entry & Pre-Shot Occupation",
        "event",
        (
            "final_third_entry_to_shot_s",
            "first_box_entry_to_shot_s",
            "last_box_entry_to_shot_s",
            "final_third_actions_before_shot",
            "box_actions_before_shot",
            "box_exit_reentry",
        ),
    ),
    FeatureFamily(
        "E8",
        "Turnover & Transition Context",
        "event",
        (
            "high_turnover_shot",
            "transition_proxy",
            "counterpress_regain_proxy",
            "regain_height_speed_interaction",
            "regain_to_box_entry_s",
        ),
    ),
    FeatureFamily(
        "E9",
        "Pressure History Context",
        "event",
        (
            "pressures_faced_possession",
            "pressures_last_5s",
            "pressures_last_10s",
            "time_since_last_pressure_s",
            "under_pressure_actions_n",
            "under_pressure_action_share",
            "successful_under_pressure_actions_n",
        ),
    ),
    FeatureFamily(
        "E10",
        "Immediate Pre-Shot Sequence Context",
        "event",
        (
            "previous_action_type",
            "previous_action_time_gap_s",
            "previous_action_distance_to_shot",
            "previous_action_progress_sb",
            "previous_action_under_pressure",
            "same_player_previous_action",
        ),
    ),
    FeatureFamily(
        "E11",
        "Set-Piece / Restart Context",
        "event",
        (
            "set_piece_category",
            "set_piece_phase",
            "seconds_since_restart",
            "actions_since_restart",
            "direct_vs_second_phase",
        ),
    ),
    FeatureFamily(
        "E12",
        "Recent Match Momentum Context",
        "event",
        (
            "team_shots_last_5m",
            "opp_shots_last_5m",
            "team_attacking_actions_last_5m",
            "opp_attacking_actions_last_5m",
            "event_possession_share_last_5m",
            "territorial_dominance_last_5m",
        ),
    ),
    FeatureFamily(
        "E13",
        "Event-Derived Phase-of-Play Context",
        "event",
        (
            "phase_location",
            "possession_directness_score",
            "phase_directness_bucket",
            "phase_control_score",
            "phase_control_state",
            "time_to_control",
        ),
        ("derivation_parameters_not_yet_locked",),
    ),
)

THREE_SIXTY_FAMILIES: tuple[FeatureFamily, ...] = (
    FeatureFamily(
        "F1",
        "Local Shot Pressure State",
        "360",
        (
            "nearest_defender_distance",
            "defenders_within_3m",
            "defenders_within_5m",
            "defenders_within_8m",
            "local_defensive_density",
            "defenders_between_ball_and_goal",
            "actor_space",
        ),
    ),
    FeatureFamily(
        "F2",
        "Defensive Block Geometry",
        "360",
        (
            "defensive_line_depth",
            "defensive_centroid_x",
            "defensive_centroid_y",
            "defensive_width",
            "defensive_length",
            "defensive_compactness",
            "defensive_hull_area",
        ),
    ),
    FeatureFamily(
        "F3",
        "Box Occupation & Numerical State",
        "360",
        (
            "defenders_in_box",
            "attackers_in_box",
            "box_numerical_balance",
            "defenders_goal_side",
            "central_defenders_between_ball_and_goal",
        ),
    ),
    FeatureFamily(
        "F4",
        "Goalkeeper Spatial State",
        "360",
        (
            "gk_x",
            "gk_y",
            "gk_depth",
            "gk_lateral_offset",
            "gk_distance_to_shooter",
            "gk_distance_to_goal_centre",
        ),
    ),
    FeatureFamily(
        "F5",
        "Goal Exposure / Shooting Corridor",
        "360",
        (
            "estimated_goalface_occlusion",
            "shot_corridor_occlusion",
            "visible_goal_angle_proxy",
            "goal_mouth_defender_count",
        ),
    ),
    FeatureFamily(
        "F6",
        "Defensive Shape Evolution",
        "360",
        (
            "defensive_line_depth_delta",
            "defensive_centroid_delta",
            "defensive_width_delta",
            "defensive_length_delta",
            "defensive_compactness_delta",
            "defensive_hull_area_delta",
            "defensive_line_state_change_rate",
        ),
    ),
    FeatureFamily(
        "F7",
        "Defensive Recovery / Box Reset",
        "360",
        (
            "defenders_goal_side_delta",
            "defenders_in_box_delta",
            "attackers_in_box_delta",
            "box_numerical_balance_delta",
            "local_density_delta",
        ),
    ),
    FeatureFamily(
        "F8",
        "Nearest-Pressure Evolution",
        "360",
        (
            "nearest_defender_distance_delta",
            "nearest_defender_distance_state_change_rate",
            "local_density_state_change_rate",
        ),
    ),
    FeatureFamily(
        "F9",
        "Goalkeeper Movement State",
        "360",
        (
            "gk_lateral_displacement",
            "gk_depth_displacement",
            "gk_total_displacement",
            "gk_lateral_state_change_rate",
            "gk_depth_state_change_rate",
        ),
    ),
    FeatureFamily(
        "F10",
        "Shooter / Receiver Space Evolution",
        "360",
        (
            "pre_shot_receiver_space",
            "shooter_space_previous_linked_event",
            "shooter_space_change",
            "time_since_shooter_max_linkable_space",
        ),
        ("requires_explicit_actor_receiver_shooter_linkage",),
    ),
    FeatureFamily(
        "F11",
        "Defensive Bypass / Line-Break State",
        "360",
        (
            "defensive_layer_bypass_proxy_last_action",
            "defensive_layer_bypass_proxy_last_3_actions",
            "line_break_proxy_last_action",
            "line_break_proxy_possession_count",
        ),
        ("governed_spatial_proxies_not_native_subscription_metrics",),
    ),
    FeatureFamily(
        "F12",
        "Rest-Defence / Transition Structure",
        "360",
        (
            "defenders_behind_ball_at_regain",
            "rest_defence_count_at_regain",
            "rest_defence_count_at_shot",
            "rest_defence_recovery_delta",
            "rest_defence_reset_fraction",
        ),
    ),
    FeatureFamily(
        "F13",
        "Goal Exposure Evolution",
        "360",
        (
            "estimated_goalface_occlusion_delta",
            "shot_corridor_occlusion_delta",
            "visible_goal_angle_delta",
        ),
    ),
    FeatureFamily(
        "F14",
        "Sequence-Level Spatial Dynamics",
        "360",
        (
            "mean_defensive_compactness_last_3",
            "min_defensive_compactness_sequence",
            "max_box_numerical_advantage",
            "time_since_max_box_advantage",
            "max_goal_exposure",
            "goal_exposure_decay",
            "defensive_recovery_slope",
        ),
        ("requires_explicit_sequence_depth_eligibility_and_coverage_metadata",),
    ),
    FeatureFamily(
        "F15",
        "Composite Spatial-State Features",
        "360",
        (
            "transition_space_decay",
            "defensive_reset_index",
            "gk_setness_proxy",
        ),
        ("later_stage_not_first_pass_screening",),
    ),
)

EVENT_FAMILY_CANDIDATE_COUNTS = {
    "E1": 7,
    "E2": 8,
    "E3": 4,
    "E4": 6,
    "E5": 3,
    "E6": 6,
    "E7": 6,
    "E8": 5,
    "E9": 7,
    "E10": 6,
    "E11": 5,
    "E12": 6,
    "E13": 6,
}

THREE_SIXTY_FAMILY_CANDIDATE_COUNTS = {
    "F1": 7,
    "F2": 7,
    "F3": 5,
    "F4": 6,
    "F5": 4,
    "F6": 7,
    "F7": 5,
    "F8": 3,
    "F9": 5,
    "F10": 4,
    "F11": 4,
    "F12": 5,
    "F13": 3,
    "F14": 7,
    "F15": 3,
}


def event_families() -> tuple[FeatureFamily, ...]:
    """Return the event-derived CxG contextual candidate families."""
    return EVENT_FAMILIES


def three_sixty_families() -> tuple[FeatureFamily, ...]:
    """Return the 360-derived CxG+ contextual candidate families."""
    return THREE_SIXTY_FAMILIES


def event_candidate_names() -> tuple[str, ...]:
    """Return all event-derived contextual candidates in taxonomy order."""
    return tuple(candidate for family in EVENT_FAMILIES for candidate in family.candidates)


def event_candidate_names_for_families(family_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return event candidates for governed family IDs in taxonomy order."""
    selected = set(family_ids)
    return tuple(
        candidate
        for family in EVENT_FAMILIES
        if family.family_id in selected
        for candidate in family.candidates
    )


def three_sixty_candidate_names() -> tuple[str, ...]:
    """Return all 360-derived contextual candidates in taxonomy order."""
    return tuple(candidate for family in THREE_SIXTY_FAMILIES for candidate in family.candidates)


def contextual_candidate_names() -> tuple[str, ...]:
    """Return all contextual candidates in taxonomy order."""
    return event_candidate_names() + three_sixty_candidate_names()


def cxg_event_contextual_allowlist() -> tuple[str, ...]:
    """Return the event-only contextual candidates permitted for CxG."""
    return event_candidate_names()
