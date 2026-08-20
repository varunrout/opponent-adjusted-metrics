import math

from opponent_adjusted.features.cxg.three_sixty_frame import FramePlayer
from opponent_adjusted.features.cxg import three_sixty_geometry as geo


def player(ordinal, teammate=None, actor=None, keeper=None, x=60.0, y=40.0):
    return FramePlayer(ordinal, teammate, actor, keeper, x, y)


def test_outfield_attacker_defender_and_opposition_partitioning():
    players = (
        player(0, teammate=True, x=60, y=40),
        player(1, teammate=False, keeper=False, x=100, y=40),
        player(2, teammate=False, keeper=True, x=5, y=40),
        player(3, teammate=None, x=50, y=40),
    )
    attackers = geo.outfield_attackers(players)
    defenders = geo.outfield_defenders(players)
    opposition = geo.all_opposition(players)
    assert [p.ordinal for p in attackers] == [0]
    assert [p.ordinal for p in defenders] == [1]
    assert {p.ordinal for p in opposition} == {1, 2}


def test_cone_lane_bounds_and_contains():
    cone = geo.Cone(ball_x=100.0, ball_y=40.0)
    assert cone.lane_bounds(90.0) is None
    bounds_at_goal = cone.lane_bounds(120.0)
    assert bounds_at_goal == (36.0, 44.0)
    assert cone.contains(110.0, 40.0)
    assert not cone.contains(110.0, 70.0)


def test_nearest_defender_distance_and_radius_counts():
    defenders = (player(0, x=100.0, y=40.0), player(1, x=110.0, y=40.0))
    distance = geo.nearest_defender_distance_m(100.0, 40.0, defenders)
    assert math.isclose(distance, 0.0)
    assert geo.nearest_defender_distance_m(100.0, 40.0, ()) is None
    assert geo.count_within_radius_m(90.0, 40.0, defenders, 15.0) >= 1


def test_defensive_block_geometry_min_counts():
    assert geo.defensive_line_depth(()) is None
    one = (player(0, x=20.0, y=30.0),)
    assert geo.defensive_line_depth(one) == 20.0
    assert geo.defensive_compactness(one) is None  # needs >= 2
    two = one + (player(1, x=25.0, y=50.0),)
    assert geo.defensive_compactness(two) == (50.0 - 30.0) * (25.0 - 20.0)
    assert geo.defensive_hull_area(two) is None  # needs >= 3
    three = two + (player(2, x=30.0, y=40.0),)
    area = geo.defensive_hull_area(three)
    assert area is not None and area > 0


def test_gk_geometry_helpers():
    gk = player(0, x=115.0, y=44.0)
    assert geo.gk_depth(gk) == 5.0
    assert geo.gk_lateral_offset(gk) == 4.0
    assert math.isclose(geo.gk_distance_to_goal_centre(gk), math.hypot(5.0, 4.0))


def test_estimated_goalface_occlusion_bounded_and_zero_with_no_blockers():
    assert geo.estimated_goalface_occlusion(100.0, 40.0, ()) == 0.0
    occlusion = geo.estimated_goalface_occlusion(100.0, 40.0, (player(0, x=115.0, y=40.0),))
    assert 0.0 <= occlusion <= 1.0


def test_shot_corridor_occlusion_capped_at_one():
    cone = geo.Cone(100.0, 40.0)
    many_blockers = tuple(player(i, x=115.0, y=40.0 + i * 0.01) for i in range(50))
    occlusion = geo.shot_corridor_occlusion(100.0, 40.0, many_blockers, cone)
    assert occlusion <= 1.0


def test_visible_goal_angle_proxy_no_blockers_returns_raw_angle():
    cone = geo.Cone(100.0, 40.0)
    assert geo.visible_goal_angle_proxy(100.0, 40.0, 0.5, (), cone) == 0.5


def test_visible_goal_angle_proxy_reduced_by_close_blocker():
    cone = geo.Cone(100.0, 40.0)
    blocker = (player(0, x=110.0, y=40.0),)
    reduced = geo.visible_goal_angle_proxy(100.0, 40.0, 0.5, blocker, cone)
    assert reduced < 0.5
    assert reduced >= 0.0
