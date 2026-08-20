from opponent_adjusted.features.cxg.three_sixty_frame import Frame, FramePlayer
from opponent_adjusted.features.cxg.three_sixty_static import (
    F1_F5_FEATURES,
    derive_static_360_context,
)

VISIBLE_AREA_FULL = (0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0)


def player(ordinal, teammate=None, actor=None, keeper=None, x=60.0, y=40.0):
    return FramePlayer(ordinal, teammate, actor, keeper, x, y)


def test_exact_governed_f1_f5_membership():
    assert len(F1_F5_FEATURES) == 29


def test_no_frame_is_all_null():
    values = derive_static_360_context(None, None, 100.0, 40.0)
    assert set(values) == set(F1_F5_FEATURES)
    assert all(v is None for v in values.values())


def test_empty_frame_players_still_null_no_defenders_observed():
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, ())
    values = derive_static_360_context(frame, (), 100.0, 40.0)
    assert values["nearest_defender_distance"] is None
    assert values["defensive_line_depth"] is None
    assert values["gk_x"] is None


def test_one_defender_within_radius_bands():
    players = (player(0, teammate=False, keeper=False, x=101.0, y=40.0),)
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["defenders_within_3m"] == 1
    assert values["defenders_within_5m"] == 1
    assert values["defenders_within_8m"] == 1
    assert values["nearest_defender_distance"] is not None


def test_ball_referenced_features_do_not_require_visible_area():
    # visible_area is empty for 100% of the currently published corpus (see module
    # docstring METHODOLOGY_BUG note in three_sixty_static.py); the frame's own ball/actor
    # point is presumed observable regardless of visible_area population.
    empty_area = ()
    players = (player(0, teammate=False, keeper=False, x=101.0, y=40.0),)
    frame = Frame("e1", 1, empty_area, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["nearest_defender_distance"] is not None
    assert values["defenders_within_3m"] == 1


def test_no_keeper_visible_is_null_not_synthetic():
    players = (player(0, teammate=True, x=60.0, y=40.0),)
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["gk_x"] is None
    assert values["gk_distance_to_shooter"] is None


def test_multiple_keepers_is_null_and_audit_ambiguous():
    players = (
        player(0, teammate=False, keeper=True, x=5.0, y=40.0),
        player(1, teammate=False, keeper=True, x=6.0, y=41.0),
    )
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["gk_x"] is None


def test_single_keeper_geometry_computed():
    players = (player(0, teammate=False, keeper=True, x=115.0, y=44.0),)
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["gk_x"] == 115.0
    assert values["gk_depth"] == 5.0
    assert values["gk_lateral_offset"] == 4.0
    assert values["gk_distance_to_shooter"] is not None


def test_box_occupation_counts_and_balance_sign():
    players = (
        player(0, teammate=True, x=110.0, y=40.0),
        player(1, teammate=False, keeper=False, x=110.0, y=45.0),
        player(2, teammate=False, keeper=False, x=110.0, y=50.0),
    )
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 108.0, 40.0)
    assert values["attackers_in_box"] == 1
    assert values["defenders_in_box"] == 2
    assert values["box_numerical_balance"] == -1


def test_actor_space_requires_identifiable_actor():
    players = (player(0, teammate=True, actor=True, x=100.0, y=40.0),)
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, 100.0, 40.0)
    assert values["actor_space"] is None  # no opposition present -> nearest defender undefined

    players_with_opp = players + (player(1, teammate=False, keeper=False, x=105.0, y=40.0),)
    values2 = derive_static_360_context(frame, players_with_opp, 100.0, 40.0)
    assert values2["actor_space"] is not None

    no_actor_players = (player(0, teammate=False, keeper=False, x=105.0, y=40.0),)
    values3 = derive_static_360_context(frame, no_actor_players, 100.0, 40.0)
    assert values3["actor_space"] is None


def test_invalid_ball_location_nulls_ball_referenced_features():
    players = (player(0, teammate=False, keeper=False, x=101.0, y=40.0),)
    frame = Frame("e1", 1, VISIBLE_AREA_FULL, players)
    values = derive_static_360_context(frame, players, None, None)
    assert values["nearest_defender_distance"] is None
    assert values["estimated_goalface_occlusion"] is None
