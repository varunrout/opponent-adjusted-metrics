from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.event_context_e13 import E13_FEATURES, derive_e13_contexts


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
    return derive_e13_contexts(events)[event_id].values


def test_exact_governed_e13_membership():
    assert len(E13_FEATURES) == 6
    assert set(E13_FEATURES) == {
        "phase_location",
        "possession_directness_score",
        "phase_directness_bucket",
        "phase_control_score",
        "phase_control_state",
        "time_to_control",
    }


def test_zero_prior_action_direct_shot_is_all_null_except_nothing_leaks():
    rows = [shot(index=0, second=0, location_x=110)]
    v = values(rows)
    assert v["phase_location"] is None
    assert v["possession_directness_score"] is None
    assert v["phase_directness_bucket"] is None
    assert v["phase_control_score"] is None
    assert v["phase_control_state"] is None
    assert v["time_to_control"] is None


def test_phase_location_uses_mean_prior_location_not_shot_location():
    rows = [
        event("a", 1, second=1, location_x=20, end_x=30, end_y=40),
        event("b", 2, second=2, location_x=30, end_x=40, end_y=40),
        shot(index=3, second=3, location_x=110, location_y=40),
    ]
    v = values(rows)
    assert v["phase_location"] == "defensive_third"


def test_direct_possession_yields_high_directness_and_direct_bucket():
    rows = [
        event("a", 1, second=1, location_x=40, location_y=40, end_x=100, end_y=40, duration=1),
        shot(index=2, second=2, location_x=110, location_y=40),
    ]
    v = values(rows)
    assert v["possession_directness_score"] == 1.0
    assert v["phase_directness_bucket"] == "direct"


def test_regressive_possession_yields_regressive_bucket():
    rows = [
        event("a", 1, second=1, location_x=100, location_y=40, end_x=40, end_y=40, duration=1),
        shot(index=2, second=2, location_x=110, location_y=40),
    ]
    v = values(rows)
    assert v["possession_directness_score"] == -1.0
    assert v["phase_directness_bucket"] == "regressive"


def test_single_successful_first_action_does_not_imply_settled():
    rows = [
        event("a", 1, second=1, event_type_name="Carry", end_x=61, end_y=40),
        shot(index=2, second=2),
    ]
    v = values(rows)
    assert v["phase_control_state"] != "settled"
    assert v["time_to_control"] is None  # never reaches the minimum action-count bar


def test_mature_multi_action_possession_reaches_settled_after_transition_window():
    rows = [
        event("a", 1, second=0, event_type_name="Carry", end_x=61, end_y=40),
        event("b", 2, second=5, event_type_name="Carry", end_x=65, end_y=40),
        event("c", 3, second=10, event_type_name="Carry", end_x=70, end_y=40),
        event("d", 4, second=16, event_type_name="Carry", end_x=75, end_y=40),
        shot(index=5, second=20),
    ]
    v = values(rows)
    assert v["phase_control_state"] == "settled"
    # settledness is only recognised once the live-regain transition window (15s) has passed,
    # i.e. at action "d" (age 16s), not earlier even though "c" already met the other criteria.
    assert v["time_to_control"] == 16.0


def test_rapid_transition_possession_stays_transition_despite_early_success():
    rows = [
        event("a", 1, second=0, event_type_name="Carry", end_x=61, end_y=40),
        event("b", 2, second=3, event_type_name="Carry", end_x=65, end_y=40),
        event("c", 3, second=6, event_type_name="Carry", end_x=70, end_y=40),
        shot(index=4, second=8),
    ]
    v = values(rows)
    assert v["phase_control_state"] == "transition"
    assert v["time_to_control"] is None


def test_phase_control_score_is_bounded_and_null_distinct_from_transition():
    rows_null = [shot(index=0, second=0)]
    assert values(rows_null)["phase_control_state"] is None
    assert values(rows_null)["phase_control_score"] is None

    rows_low = [
        event("a", 1, second=1, event_type_name="Miscontrol"),
        shot(index=2, second=2),
    ]
    v = values(rows_low)
    assert v["phase_control_state"] != "settled"
    assert 0.0 <= v["phase_control_score"] <= 1.0


def test_time_to_control_replays_causally_not_backfilled_from_shot_time():
    # A possession that only becomes settled at action "d"; time_to_control must equal that
    # action's own elapsed time, not the (larger) elapsed time of a later still-successful action.
    rows = [
        event("a", 1, second=0, event_type_name="Carry", end_x=61, end_y=40),
        event("b", 2, second=5, event_type_name="Carry", end_x=65, end_y=40),
        event("c", 3, second=10, event_type_name="Carry", end_x=70, end_y=40),
        event("d", 4, second=16, event_type_name="Carry", end_x=75, end_y=40),
        event("e", 5, second=25, event_type_name="Carry", end_x=80, end_y=40),
        shot(index=6, second=30),
    ]
    v = values(rows)
    assert v["time_to_control"] == 16.0


def test_e13_does_not_use_future_or_target_fields():
    rows = [
        event("a", 1, second=1, event_type_name="Carry", end_x=61, end_y=40),
        shot(index=2, second=2, outcome_name="Goal"),
    ]
    v = values(rows)
    assert set(v) == set(E13_FEATURES)
