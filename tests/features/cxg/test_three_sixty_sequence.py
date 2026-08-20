from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.three_sixty_frame import Frame, FramePlayer
from opponent_adjusted.features.cxg.three_sixty_sequence import (
    F6_F14_FEATURES,
    derive_dynamic_360_context,
)

VISIBLE_AREA_FULL = (0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0)


def event(event_id, index, **changes):
    values = dict(
        event_id=event_id,
        match_id=1,
        event_index=index,
        period=1,
        minute=0,
        second=index,
        timestamp=f"00:00:{index:02d}.000",
        event_type_name="Pass",
        outcome_name=None,
        team_id=1,
        possession_id=1,
        possession_team_id=1,
        location_x=60.0,
        location_y=40.0,
        player_id=1,
        play_pattern_name="Regular Play",
    )
    values.update(changes)
    return EventRecord(**values)


def shot(event_id="shot", index=9, **changes):
    changes.setdefault("event_type_name", "Shot")
    changes.setdefault("location_x", 110.0)
    return event(event_id, index, **changes)


def player(ordinal, teammate=None, actor=None, keeper=None, x=60.0, y=40.0):
    return FramePlayer(ordinal, teammate, actor, keeper, x, y)


def frame_for(event_id, players, match_id=1):
    return Frame(event_id, match_id, VISIBLE_AREA_FULL, players)


def values(events, frames, event_id="shot"):
    return derive_dynamic_360_context(events, frames)[event_id]


def test_exact_governed_f6_f14_membership():
    assert len(F6_F14_FEATURES) == 43


def test_no_frames_is_all_null():
    rows = [event("a", 1, second=1), shot(index=2, second=2)]
    v = values(rows, {})
    assert set(v) == set(F6_F14_FEATURES)
    assert all(x is None for x in v.values())


def test_deltas_computed_between_shot_and_latest_prior_frame():
    prior_players = (player(0, teammate=False, keeper=False, x=90.0, y=30.0),)
    shot_players = (player(0, teammate=False, keeper=False, x=95.0, y=30.0),)
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        shot(index=2, second=6, location_x=110, location_y=40),
    ]
    frames = {"a": frame_for("a", prior_players), "shot": frame_for("shot", shot_players)}
    v = values(rows, frames)
    assert v["defensive_line_depth_delta"] == 5.0
    assert v["defensive_line_state_change_rate"] == 1.0  # 5.0 / 5.0 seconds elapsed


def test_negative_or_zero_elapsed_excludes_prior_state():
    same_time_players = (player(0, teammate=False, keeper=False, x=90.0, y=30.0),)
    rows = [
        event("a", 1, second=5, location_x=60, location_y=40),
        shot(index=2, second=5, location_x=110, location_y=40),
    ]
    frames = {"a": frame_for("a", same_time_players)}
    v = values(rows, frames)
    assert v["defensive_line_depth_delta"] is None


def test_cross_possession_prior_event_is_excluded():
    other_possession = event("a", 1, second=1, possession_id=999, location_x=60, location_y=40)
    rows = [other_possession, shot(index=2, second=6, location_x=110, location_y=40)]
    frames = {"a": frame_for("a", (player(0, teammate=False, keeper=False, x=90.0, y=30.0),))}
    v = values(rows, frames)
    assert v["defensive_line_depth_delta"] is None


def test_f10_shooter_receipt_and_previous_linked_event():
    receipt_players = (
        player(0, teammate=True, actor=True, x=90.0, y=30.0),
        player(1, teammate=False, keeper=False, x=93.0, y=30.0),
    )
    carry_players = (
        player(0, teammate=True, actor=True, x=95.0, y=30.0),
        player(1, teammate=False, keeper=False, x=97.0, y=30.0),
    )
    rows = [
        event(
            "receipt",
            1,
            second=1,
            event_type_name="Ball Receipt*",
            player_id=1,
            location_x=90,
            location_y=30,
        ),
        event(
            "carry",
            2,
            second=3,
            event_type_name="Carry",
            player_id=1,
            location_x=95,
            location_y=30,
            end_x=100,
            end_y=30,
        ),
        shot(index=3, second=6, player_id=1, location_x=110, location_y=40),
    ]
    frames = {
        "receipt": frame_for("receipt", receipt_players),
        "carry": frame_for("carry", carry_players),
    }
    v = values(rows, frames)
    assert v["pre_shot_receiver_space"] is not None
    assert v["shooter_space_previous_linked_event"] is not None
    assert v["shooter_space_change"] is not None


def test_f10_null_when_no_shooter_linkage():
    rows = [event("a", 1, second=1, player_id=2), shot(index=2, second=6, player_id=1)]
    frames = {"a": frame_for("a", (player(0, teammate=True, actor=True, x=90.0, y=30.0),))}
    v = values(rows, frames)
    assert v["pre_shot_receiver_space"] is None
    assert v["shooter_space_previous_linked_event"] is None


def test_f11_null_when_last_action_has_no_linked_frame_of_its_own():
    # Only the shot has a frame; the shot's own (later) snapshot must NOT be used as a stand-in
    # for the defensive layer the earlier action actually faced.
    shot_players = (player(0, teammate=False, keeper=False, x=90.0, y=40.0),)
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        shot(index=2, second=6, location_x=110, location_y=40),
    ]
    frames = {"shot": frame_for("shot", shot_players)}
    v = values(rows, frames)
    assert v["defensive_layer_bypass_proxy_last_action"] is None
    assert v["line_break_proxy_last_action"] is None
    assert v["line_break_proxy_possession_count"] is None


def test_f11_uses_last_action_own_linked_frame_not_shot_frame():
    action_players = (player(0, teammate=False, keeper=False, x=90.0, y=40.0),)
    shot_players = (player(0, teammate=False, keeper=False, x=200.0, y=200.0),)  # decoy, unused
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        shot(index=2, second=6, location_x=110, location_y=40),
    ]
    frames = {"a": frame_for("a", action_players), "shot": frame_for("shot", shot_players)}
    v = values(rows, frames)
    assert v["defensive_layer_bypass_proxy_last_action"] == 1
    assert v["line_break_proxy_last_action"] is True


def test_f11_zero_prior_actions_returns_null_possession_count():
    shot_players = (player(0, teammate=False, keeper=False, x=90.0, y=40.0),)
    rows = [shot(index=0, second=0, location_x=110, location_y=40)]
    frames = {"shot": frame_for("shot", shot_players)}
    v = values(rows, frames)
    assert v["line_break_proxy_possession_count"] is None
    assert v["defensive_layer_bypass_proxy_last_action"] is None


def test_f11_excludes_actions_without_their_own_frame_from_possession_count():
    # "a" has no frame (excluded), "b" does -> exactly one eligible transition (b -> shot).
    b_players = (player(0, teammate=False, keeper=False, x=95.0, y=40.0),)
    rows = [
        event("a", 1, second=1, location_x=50, location_y=40),
        event("b", 2, second=3, location_x=90, location_y=40),
        shot(index=3, second=6, location_x=110, location_y=40),
    ]
    frames = {"b": frame_for("b", b_players)}
    v = values(rows, frames)
    assert v["line_break_proxy_possession_count"] == 1


def test_f12_regain_features_null_for_restart_possession():
    rows = [shot(index=0, second=0, play_pattern_name="From Corner", location_x=110, location_y=40)]
    frames = {"shot": frame_for("shot", (player(0, teammate=True, x=60.0, y=40.0),))}
    v = values(rows, frames)
    assert v["defenders_behind_ball_at_regain"] is None
    assert v["rest_defence_count_at_regain"] is None


def test_f12_regain_features_computed_for_live_regain_with_frame():
    shot_players = (player(0, teammate=True, x=100.0, y=40.0),)
    rows = [
        shot(index=0, second=0, play_pattern_name="Regular Play", location_x=110, location_y=40)
    ]
    frames = {
        "shot": frame_for("shot", shot_players),
    }
    # zero prior events -> regain event == shot itself
    v = values(rows, frames)
    assert v["rest_defence_count_at_shot"] is not None


def test_f14_sequence_extrema_and_time_since_max():
    early_players = (
        player(0, teammate=True, x=100.0, y=40.0),
        player(1, teammate=False, keeper=False, x=101.0, y=40.0),
    )
    late_players = (player(0, teammate=True, x=100.0, y=40.0),)
    rows = [
        event("a", 1, second=1, location_x=60, location_y=40),
        event("b", 2, second=2, location_x=60, location_y=40),
        shot(index=3, second=6, location_x=108, location_y=40),
    ]
    frames = {"a": frame_for("a", early_players), "shot": frame_for("shot", late_players)}
    v = values(rows, frames)
    assert v["max_box_numerical_advantage"] is not None
    assert v["time_since_max_box_advantage"] is not None
