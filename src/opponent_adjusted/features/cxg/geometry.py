"""Geometric feature calculations for shots."""

import numpy as np
from typing import Optional, Tuple

from opponent_adjusted.config import settings


def calculate_distance(x: float, y: float, target_x: float = None, target_y: float = None) -> float:
    """Calculate Euclidean distance from (x, y) to target point.

    Args:
        x: X coordinate
        y: Y coordinate
        target_x: Target X coordinate (defaults to goal center)
        target_y: Target Y coordinate (defaults to goal center)

    Returns:
        Euclidean distance
    """
    if target_x is None:
        target_x = settings.goal_center_x
    if target_y is None:
        target_y = settings.goal_center_y

    return np.sqrt((target_x - x) ** 2 + (target_y - y) ** 2)


def calculate_shot_angle(x: float, y: float) -> float:
    """Calculate shot angle (angle between vectors to goalposts).

    Args:
        x: X coordinate of shot
        y: Y coordinate of shot

    Returns:
        Angle in radians (clamped to [0, π])
    """
    # Vectors from shot position to goalposts
    v1_x = settings.goal_center_x - x
    v1_y = settings.goal_post_left_y - y
    v2_x = settings.goal_center_x - x
    v2_y = settings.goal_post_right_y - y

    # Calculate magnitudes
    mag1 = np.sqrt(v1_x**2 + v1_y**2)
    mag2 = np.sqrt(v2_x**2 + v2_y**2)

    # Handle edge case of being exactly at goal
    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Calculate dot product
    dot_product = v1_x * v2_x + v1_y * v2_y

    # Calculate angle using arccos, with numerical stability
    cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return float(angle)


def calculate_centrality(y: float) -> float:
    """Calculate how far the shot is from the center line.

    Args:
        y: Y coordinate of shot

    Returns:
        Absolute distance from center line
    """
    return abs(y - settings.goal_center_y)


def calculate_distance_to_goal_line(x: float) -> float:
    """Calculate distance to goal line.

    Args:
        x: X coordinate of shot

    Returns:
        Distance to goal line
    """
    return settings.goal_center_x - x


def assign_zone(distance: float, centrality: float) -> str:
    """Assign a zone based on distance and centrality.

    Zones:
    - A: Close central (dist ≤ 12, central ≤ 10)
    - B: Close wide (dist ≤ 12, central > 10)
    - C: Mid central (12 < dist ≤ 20, central ≤ 10)
    - D: Mid wide (12 < dist ≤ 20, central > 10)
    - E: Far central (dist > 20, central ≤ 10)
    - F: Far wide (dist > 20, central > 10)

    Args:
        distance: Shot distance from goal
        centrality: Distance from center line

    Returns:
        Zone identifier (A-F)
    """
    zones = settings.zone_definitions

    if distance <= zones["A"]["max_distance"]:
        if centrality <= zones["A"]["max_centrality"]:
            return "A"
        else:
            return "B"
    elif distance <= zones["C"]["max_distance"]:
        if centrality <= zones["C"]["max_centrality"]:
            return "C"
        else:
            return "D"
    else:
        if centrality <= zones["E"]["max_centrality"]:
            return "E"
        else:
            return "F"


def calculate_all_geometry_features(
    x: float, y: float
) -> dict:
    """Calculate all geometric features for a shot.

    Args:
        x: X coordinate of shot
        y: Y coordinate of shot

    Returns:
        Dictionary with all geometric features
    """
    distance = calculate_distance(x, y)
    angle = calculate_shot_angle(x, y)
    centrality = calculate_centrality(y)
    dist_to_goal_line = calculate_distance_to_goal_line(x)
    zone = assign_zone(distance, centrality)

    return {
        "shot_distance": distance,
        "shot_angle": angle,
        "centrality": centrality,
        "distance_to_goal_line": dist_to_goal_line,
        "zone_id": zone,
    }
