"""Backward-compatible geometry feature imports."""

from opponent_adjusted.features.cxg.geometry import (
    assign_zone,
    calculate_all_geometry_features,
    calculate_centrality,
    calculate_distance,
    calculate_distance_to_goal_line,
    calculate_shot_angle,
)

__all__ = [
    "calculate_distance",
    "calculate_shot_angle",
    "calculate_centrality",
    "calculate_distance_to_goal_line",
    "assign_zone",
    "calculate_all_geometry_features",
]
