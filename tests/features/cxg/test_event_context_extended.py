from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.event_context_extended import (
    E7_E12_FEATURES,
    derive_extended_event_contexts,
)


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


def values(events, event_id="shot"):
    return derive_extended_event_contexts(events)[event_id].values


def test_exact_governed_e7_e12_membership():
    assert len(E7_E12_FEATURES) == 35
    assert set(E7_E12_FEATURES) == {
        "final_third_entry_to_shot_s",
        "first_box_entry_to_shot_s",
        "last_box_entry_to_shot_s",
        "final_third_actions_before_shot",
        "box_actions_before_shot",
        "box_exit_reentry",
        "high_turnover_shot",
        "transition_proxy",
        "counterpress_regain_proxy",
        "regain_height_speed_interaction",
        "regain_to_box_entry_s",
        "pressures_faced_possession",
        "pressures_last_5s",
        "pressures_last_10s",
        "time_since_last_pressure_s",
        "under_pressure_actions_n",
        "under_pressure_action_share",
        "successful_under_pressure_actions_n",
        "previous_action_type",
        "previous_action_time_gap_s",
        "previous_action_distance_to_shot",
        "previous_action_progress_sb",
        "previous_action_under_pressure",
        "same_player_previous_action",
        "set_piece_category",
        "set_piece_phase",
        "seconds_since_restart",
        "actions_since_restart",
        "direct_vs_second_phase",
        "team_shots_last_5m",
        "opp_shots_last_5m",
        "team_attacking_actions_last_5m",
        "opp_attacking_actions_last_5m",
        "event_possession_share_last_5m",
        "territorial_dominance_last_5m",
    }


def test_e7_entry_end_observations_and_reentry():
    rows = [
        event("pass", 1, second=0, location_x=70, end_x=85, end_y=40, duration=2),
        event("inside", 2, second=3, location_x=105),
        event("outside", 3, second=4, location_x=90),
        shot(index=4, second=5, location_x=110),
    ]
    v = values(rows)
    assert v["final_third_entry_to_shot_s"] == 3
    assert v["first_box_entry_to_shot_s"] == 2
    assert v["last_box_entry_to_shot_s"] == 0
    assert v["last_box_entry_to_shot_s"] <= v["first_box_entry_to_shot_s"]
    assert v["box_exit_reentry"] is True
    assert v["final_third_actions_before_shot"] == 2
    assert v["box_actions_before_shot"] == 1


def test_e8_live_regain_counterpress_and_restart_nulls():
    rows = [
        event("cp", 1, second=0, counterpress=True, possession_team_id=2),
        event("start", 2, second=3, location_x=100, play_pattern_name="From Counter"),
        shot(index=3, second=15, location_x=110),
    ]
    v = values(rows)
    assert v["transition_proxy"] is True
    assert v["high_turnover_shot"] is True
    assert v["counterpress_regain_proxy"] is True
    assert v["regain_height_speed_interaction"] is not None
    restart = values([shot(play_pattern_name="From Corner")])
    assert restart["transition_proxy"] is None
    assert restart["counterpress_regain_proxy"] is None


def test_e9_pressure_history_and_source_backed_success():
    rows = [
        event("pressure", 1, second=1, event_type_name="Pressure", team_id=2, possession_team_id=1),
        event("pass", 2, second=5, under_pressure=True, action_outcome_name=None),
        event(
            "dribble",
            3,
            second=6,
            event_type_name="Dribble",
            under_pressure=True,
            action_outcome_name="Complete",
        ),
        shot(index=4, second=11),
    ]
    v = values(rows)
    assert v["pressures_faced_possession"] == 1
    assert v["pressures_last_10s"] == 1
    assert v["time_since_last_pressure_s"] == 10
    assert v["under_pressure_actions_n"] == 2
    assert v["under_pressure_action_share"] == 2 / 2
    assert v["successful_under_pressure_actions_n"] == 2


def test_e10_previous_event_including_opponent_and_end_anchor():
    rows = [
        event("pass", 1, second=1, end_x=100, end_y=40, under_pressure=None, player_id=2),
        event(
            "pressure",
            2,
            second=3,
            event_type_name="Pressure",
            team_id=2,
            possession_team_id=1,
            player_id=3,
            location_x=90,
        ),
        shot(index=3, second=5, player_id=1),
    ]
    v = values(rows)
    assert v["previous_action_type"] == "Pressure"
    assert v["previous_action_time_gap_s"] == 2
    assert v["previous_action_distance_to_shot"] == 20
    assert v["previous_action_progress_sb"] is None
    assert v["same_player_previous_action"] is False


def test_e11_restart_phases_and_e12_strict_historical_window():
    rows = [
        event("corner", 1, second=0, play_pattern_name="From Corner", location_x=100),
        event("team-shot", 2, second=4, event_type_name="Shot", location_x=100),
        event(
            "opp-shot",
            3,
            second=8,
            event_type_name="Shot",
            team_id=2,
            possession_team_id=2,
            location_x=60,
        ),
        shot(index=4, second=10, location_x=110),
    ]
    v = values(rows)
    assert v["set_piece_category"] == "corner"
    assert v["set_piece_phase"] == "first_phase"
    assert v["direct_vs_second_phase"] == "direct"
    assert v["team_shots_last_5m"] == 1
    assert v["opp_shots_last_5m"] == 1
    assert v["team_attacking_actions_last_5m"] == 2
    assert v["opp_attacking_actions_last_5m"] == 1
    assert 0 <= v["event_possession_share_last_5m"] <= 1
    assert -1 <= v["territorial_dominance_last_5m"] <= 1


def test_extended_context_is_match_isolated_and_has_no_360_dependency():
    a = [
        event("a-start", 1, match_id=1, possession_id=1, location_x=20),
        shot("a-shot", 2, match_id=1, location_x=100),
    ]
    b = [
        event("b-start", 1, match_id=2, possession_id=1, location_x=80),
        shot("b-shot", 2, match_id=2, location_x=110),
    ]
    grouped = derive_extended_event_contexts([*a, *b])
    interleaved = derive_extended_event_contexts([a[0], b[0], a[1], b[1]])
    assert grouped["a-shot"].values == interleaved["a-shot"].values
    assert grouped["b-shot"].values == interleaved["b-shot"].values
    source = __import__(
        "opponent_adjusted.features.cxg.event_context_extended", fromlist=["x"]
    ).__loader__.get_source("opponent_adjusted.features.cxg.event_context_extended")
    assert "three_sixty" not in source
    assert "google.cloud" not in source
    assert "statsbomb_xg" not in source


def test_box_reentry_is_state_based_and_skips_invalid_observations():
    rows = [
        event("inside-a", 1, location_x=105),
        event("inside-b", 2, location_x=106),
        event("missing", 3, location_x=None, location_y=None),
        event("outside-a", 4, location_x=90),
        event("outside-b", 5, location_x=80),
        shot(index=6, location_x=110),
    ]
    assert values(rows)["box_exit_reentry"] is True
    assert (
        values(
            [
                event("inside", 1, location_x=105),
                event("outside", 2, location_x=90),
                shot(index=3, location_x=90),
            ]
        )["box_exit_reentry"]
        is False
    )


def test_counterpress_requires_strict_pre_start_order_and_evaluable_clock():
    before = values(
        [
            event("cp", 1, second=0, counterpress=True, possession_team_id=2),
            event("start", 2, second=3, play_pattern_name="From Counter"),
            shot(index=3, second=5),
        ]
    )
    after_same_clock = values(
        [
            event("start", 1, second=3, play_pattern_name="From Counter"),
            event("cp", 2, second=3, counterpress=True, possession_team_id=2),
            shot(index=3, second=5),
        ]
    )
    unevaluable = values(
        [
            event("start", 1, timestamp="invalid", play_pattern_name="From Counter"),
            shot(index=2, second=5),
        ]
    )
    assert before["counterpress_regain_proxy"] is True
    assert after_same_clock["counterpress_regain_proxy"] is False
    assert unevaluable["counterpress_regain_proxy"] is None


def test_e12_unknown_team_events_are_neither_team_nor_opposition():
    rows = [
        event("unknown-shot", 1, team_id=None, event_type_name="Shot", location_x=120),
        event("unknown-pass", 2, team_id=None, location_x=120),
        shot(index=3, second=10, location_x=110),
    ]
    v = values(rows)
    assert v["opp_shots_last_5m"] == 0
    assert v["opp_attacking_actions_last_5m"] == 0
    assert v["territorial_dominance_last_5m"] is None
