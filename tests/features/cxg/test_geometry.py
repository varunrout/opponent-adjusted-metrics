import math

import pytest

from opponent_adjusted.features.cxg.geometry import (
    GOAL_POST_HIGH_Y,
    GOAL_POST_LOW_Y,
    GOAL_X,
    shot_geometry,
)


def test_native_goal_centre_and_origin_are_valid_without_conversion():
    goal_centre = shot_geometry(120.0, 40.0)
    origin = shot_geometry(0.0, 0.0)

    assert goal_centre.geometry_valid
    assert goal_centre.goal_line_distance_sb == 0.0
    assert goal_centre.lateral_goal_offset_sb == 0.0
    assert origin.geometry_valid


def test_native_distance_helpers_preserve_centre_axis_geometry():
    geometry = shot_geometry(100.0, 40.0)

    assert geometry.goal_line_distance_sb == 20.0
    assert geometry.lateral_goal_offset_sb == 0.0
    assert geometry.shot_distance_sb == 20.0


def test_off_axis_distance_and_exact_two_post_angle_formula():
    x, y = 100.0, 30.0
    geometry = shot_geometry(x, y)
    v1_x, v1_y = GOAL_X - x, GOAL_POST_LOW_Y - y
    v2_x, v2_y = GOAL_X - x, GOAL_POST_HIGH_Y - y
    expected_distance = math.sqrt((120.0 - x) ** 2 + (40.0 - y) ** 2)
    expected_angle = math.atan2(abs(v1_x * v2_y - v1_y * v2_x), v1_x * v2_x + v1_y * v2_y)

    assert geometry.shot_distance_sb == pytest.approx(expected_distance)
    assert geometry.shot_angle_rad == pytest.approx(expected_angle)


@pytest.mark.parametrize(
    "x,y", [(120.0, 40.0), (100.0, 36.0), (100.0, 44.0), (60.0, 40.0), (0.0, 0.0), (0.0, 80.0)]
)
def test_angle_is_bounded_for_valid_native_geometry(x, y):
    geometry = shot_geometry(x, y)

    assert geometry.geometry_valid
    assert 0.0 <= geometry.shot_angle_rad <= math.pi


@pytest.mark.parametrize(
    "x,y",
    [
        (None, 40.0),
        (100.0, None),
        (120.1, 40.0),
        (100.0, 80.1),
        (-0.1, 40.0),
        (100.0, -0.1),
        (math.nan, 40.0),
        (math.inf, 40.0),
        (-math.inf, 40.0),
    ],
)
def test_invalid_coordinates_are_not_clipped_and_emit_no_geometry(x, y):
    geometry = shot_geometry(x, y)

    assert not geometry.geometry_valid
    assert geometry.goal_line_distance_sb is None
    assert geometry.lateral_goal_offset_sb is None
    assert geometry.shot_distance_sb is None
    assert geometry.shot_angle_rad is None
