"""Shared native StatsBomb geometry for CxG feature contracts."""

from __future__ import annotations

from dataclasses import dataclass
import math

PITCH_X_MIN = 0.0
PITCH_X_MAX = 120.0
PITCH_Y_MIN = 0.0
PITCH_Y_MAX = 80.0
GOAL_X = 120.0
GOAL_CENTRE_Y = 40.0
GOAL_POST_LOW_Y = 36.0
GOAL_POST_HIGH_Y = 44.0
# Native penalty-box boundary shared by new CxG+ (F-family) code. Matches the inline literals
# already used by the frozen E7-E12 box predicate; declared here only for NEW code to reuse
# without duplicating the raw constants inconsistently. The frozen E7-E12 module is unchanged.
BOX_X_MIN = 102.0
BOX_Y_MIN = 18.0
BOX_Y_MAX = 62.0


@dataclass(frozen=True)
class ShotGeometry:
    """Native-frame derived geometry for one raw shot location."""

    geometry_valid: bool
    goal_line_distance_sb: float | None
    lateral_goal_offset_sb: float | None
    shot_distance_sb: float | None
    shot_angle_rad: float | None


def _is_valid_coordinate(value: float | None, lower: float, upper: float) -> bool:
    return value is not None and math.isfinite(value) and lower <= value <= upper


def shot_geometry(x: float | None, y: float | None) -> ShotGeometry:
    """Derive governed native StatsBomb shot geometry without coordinate clipping."""
    if not (
        _is_valid_coordinate(x, PITCH_X_MIN, PITCH_X_MAX)
        and _is_valid_coordinate(y, PITCH_Y_MIN, PITCH_Y_MAX)
    ):
        return ShotGeometry(False, None, None, None, None)

    goal_line_distance_sb = GOAL_X - x
    lateral_goal_offset_sb = y - GOAL_CENTRE_Y
    shot_distance_sb = math.hypot(goal_line_distance_sb, lateral_goal_offset_sb)
    v1_x, v1_y = GOAL_X - x, GOAL_POST_LOW_Y - y
    v2_x, v2_y = GOAL_X - x, GOAL_POST_HIGH_Y - y
    shot_angle_rad = math.atan2(abs(v1_x * v2_y - v1_y * v2_x), v1_x * v2_x + v1_y * v2_y)
    return ShotGeometry(
        True,
        goal_line_distance_sb,
        lateral_goal_offset_sb,
        shot_distance_sb,
        shot_angle_rad,
    )
