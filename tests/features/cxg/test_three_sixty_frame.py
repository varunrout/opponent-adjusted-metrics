import math

from opponent_adjusted.features.cxg.three_sixty_frame import (
    DISTANCE_UNIT_CONTRACT,
    Frame,
    FramePlayer,
    NATIVE_TO_METRE_X,
    NATIVE_TO_METRE_Y,
    defending_keeper,
    find_actor,
    metre_distance,
    orient_players,
    point_observable,
    region_observable,
)

VISIBLE_AREA_FULL = (0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0)


def player(ordinal, teammate=None, actor=None, keeper=None, x=60.0, y=40.0):
    return FramePlayer(ordinal, teammate, actor, keeper, x, y)


def test_frame_player_coordinate_validity():
    assert player(0, x=60.0, y=40.0).coordinate_valid
    assert not player(1, x=None, y=40.0).coordinate_valid
    assert not player(2, x=-1.0, y=40.0).coordinate_valid
    assert not player(3, x=60.0, y=81.0).coordinate_valid
    assert not player(4, x=math.nan, y=40.0).coordinate_valid


def test_frame_has_visible_area_requires_min_points():
    frame_ok = Frame("e1", 1, VISIBLE_AREA_FULL, ())
    assert frame_ok.has_visible_area
    frame_empty = Frame("e2", 1, (), ())
    assert not frame_empty.has_visible_area
    frame_odd = Frame("e3", 1, (0.0, 0.0, 1.0), ())
    assert not frame_odd.has_visible_area


def test_orient_players_same_team_is_identity():
    players = (player(0, teammate=True),)
    oriented = orient_players(players, frame_event_team_id=1, shot_team_id=1)
    assert oriented == players


def test_orient_players_opponent_frame_inverts_teammate():
    players = (player(0, teammate=True), player(1, teammate=False), player(2, teammate=None))
    oriented = orient_players(players, frame_event_team_id=2, shot_team_id=1)
    assert [p.teammate for p in oriented] == [False, True, None]


def test_orient_players_unknown_team_returns_none():
    players = (player(0, teammate=True),)
    assert orient_players(players, frame_event_team_id=None, shot_team_id=1) is None
    assert orient_players(players, frame_event_team_id=1, shot_team_id=None) is None


def test_defending_keeper_unambiguous_absent_and_ambiguous():
    none_case = (player(0, teammate=True, keeper=False),)
    assert defending_keeper(none_case) == (None, False)

    one_case = (player(0, teammate=False, keeper=True, x=5.0, y=40.0),)
    keeper, ambiguous = defending_keeper(one_case)
    assert keeper is not None and keeper.ordinal == 0
    assert ambiguous is False

    two_case = (
        player(0, teammate=False, keeper=True, x=5.0, y=40.0),
        player(1, teammate=False, keeper=True, x=6.0, y=41.0),
    )
    assert defending_keeper(two_case) == (None, True)


def test_find_actor_requires_unique_valid_actor():
    assert find_actor((player(0, actor=True),)) is not None
    assert find_actor((player(0, actor=False),)) is None
    assert find_actor((player(0, actor=True), player(1, actor=True))) is None
    assert find_actor((player(0, actor=True, x=None),)) is None


def test_metre_distance_uses_governed_bridge():
    distance = metre_distance(0.0, 0.0, 120.0, 0.0)
    assert math.isclose(distance, 105.0)


def test_distance_unit_contract_is_explicit_approximate_metres_not_measured():
    assert DISTANCE_UNIT_CONTRACT == (
        "approximate_metres_assumed_105x68_standard_pitch_not_measured_stadium_distance"
    )
    assert math.isclose(NATIVE_TO_METRE_X, 105.0 / 120.0)
    assert math.isclose(NATIVE_TO_METRE_Y, 68.0 / 80.0)


def test_point_observable_requires_visible_area_and_containment():
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, ())
    assert point_observable(frame, 60.0, 40.0)
    assert not point_observable(frame, 130.0, 40.0)
    assert not point_observable(None, 60.0, 40.0)
    assert not point_observable(Frame("e2", 1, (), ()), 60.0, 40.0)


def test_region_observable_requires_all_corners_inside():
    frame = Frame("e1", 1, (0.0, 0.0, 60.0, 0.0, 60.0, 80.0, 0.0, 80.0), ())
    assert region_observable(frame, ((10.0, 10.0), (50.0, 10.0), (50.0, 70.0)))
    assert not region_observable(frame, ((10.0, 10.0), (100.0, 10.0)))
    assert not region_observable(None, ((10.0, 10.0),))
