import pytest

from opponent_adjusted.features.cxg.event_context import EventRecord
from opponent_adjusted.features.cxg.phase_c_rolling_window import (
    NULL_REASON_DEFENDING_TEAM_UNRESOLVED,
    NULL_REASON_FIRST_MATCH,
    NULL_REASON_NO_SHOT_CLOCK,
    NULL_REASON_ZERO_ELAPSED,
    cross_match_rolling_rate,
    decay_weight,
    defensive_action_rates,
    period_start_clock,
    territorial_dominance_extended,
)

ACTION_TYPES = frozenset({"Pressure", "Duel", "Interception", "Clearance", "Block", "Foul Committed", "50/50"})


def _event(
    event_id="e",
    match_id=1,
    event_index=0,
    period=1,
    minute=0,
    second=0,
    event_type_name="Pass",
    team_id=1,
    location_x=60.0,
    location_y=40.0,
    possession_id=1,
    possession_team_id=1,
) -> EventRecord:
    return EventRecord(
        event_id=event_id,
        match_id=match_id,
        event_index=event_index,
        period=period,
        minute=minute,
        second=second,
        timestamp=None,
        event_type_name=event_type_name,
        outcome_name=None,
        team_id=team_id,
        possession_id=possession_id,
        possession_team_id=possession_team_id,
        location_x=location_x,
        location_y=location_y,
    )


# --- decay_weight / cross_match_rolling_rate --------------------------------------------


def test_decay_weight_gap_1_full_half_life():
    assert decay_weight(3, half_life=3.0) == pytest.approx(0.5)


def test_decay_weight_decreases_with_gap():
    w1 = decay_weight(1, half_life=3.0)
    w2 = decay_weight(2, half_life=3.0)
    w3 = decay_weight(3, half_life=3.0)
    assert w1 > w2 > w3


def test_decay_weight_rejects_gap_below_1():
    with pytest.raises(ValueError):
        decay_weight(0)


def test_cross_match_rolling_rate_first_match_is_explicit_null():
    value, reason = cross_match_rolling_rate([])
    assert value is None
    assert reason == NULL_REASON_FIRST_MATCH


def test_cross_match_rolling_rate_single_prior_match_dominated_by_it():
    # A team with exactly one prior match: its rolling value must equal that match's rate
    # exactly (weight cancels out in the normalized weighted average).
    value, reason = cross_match_rolling_rate([4.2])
    assert reason is None
    assert value == pytest.approx(4.2)


def test_cross_match_rolling_rate_weights_recent_matches_more():
    # nearest-first ordering: index 0 = gap 1 (most recent)
    recent_high = cross_match_rolling_rate([10.0, 0.0, 0.0])[0]
    recent_low = cross_match_rolling_rate([0.0, 0.0, 10.0])[0]
    assert recent_high > recent_low


def test_cross_match_rolling_rate_matches_manual_weighted_average():
    rates = [2.0, 4.0, 6.0]
    half_life = 3.0
    weights = [0.5 ** (g / half_life) for g in (1, 2, 3)]
    expected = sum(w * r for w, r in zip(weights, rates)) / sum(weights)
    value, reason = cross_match_rolling_rate(rates, half_life=half_life)
    assert reason is None
    assert value == pytest.approx(expected)


# --- period_start_clock -------------------------------------------------------------------


def test_period_start_clock_is_min_observed_clock_in_period():
    events = [
        _event(event_id="a", period=1, minute=0, second=5),
        _event(event_id="b", period=1, minute=1, second=0),
        _event(event_id="c", period=2, minute=45, second=0),
    ]
    assert period_start_clock(events, 1) == pytest.approx(5.0)


def test_period_start_clock_none_when_no_events_in_period():
    events = [_event(event_id="a", period=2, minute=45, second=0)]
    assert period_start_clock(events, 1) is None


# --- defensive_action_rates -----------------------------------------------------------------


def test_defensive_action_rates_zero_elapsed_is_explicit_null():
    # Shot is the very first event of the period: no prior events, so no period-start
    # reference exists -- must null with the explicit cold-start reason, not a silent 0.
    columns = defensive_action_rates([], defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=0.0)
    assert columns["defensive_action_rate_null_reason"] == NULL_REASON_ZERO_ELAPSED
    assert all(columns[f"defensive_action_rate_{m}m"] is None for m in (15, 30, 45, 60))


def test_defensive_action_rates_no_shot_clock_is_explicit_null():
    columns = defensive_action_rates([], defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=None)
    assert columns["defensive_action_rate_null_reason"] == NULL_REASON_NO_SHOT_CLOCK


def test_defensive_action_rates_no_defending_team_is_explicit_null():
    columns = defensive_action_rates([], defending_team_id=None, action_types=ACTION_TYPES, shot_period=1, shot_clock=100.0)
    assert columns["defensive_action_rate_null_reason"] == NULL_REASON_DEFENDING_TEAM_UNRESOLVED


def test_defensive_action_rates_full_window_rate():
    # period starts at clock=0 (first event), 4 defending-team Pressure events all within
    # the last 15 minutes, shot at clock = 20 minutes (1200s) so the 15m window is a FULL
    # (non-partial) window -- rate should be exactly count / 15.
    events = [_event(event_id="start", minute=0, second=0, event_type_name="Pass", team_id=1)]
    events += [
        _event(event_id=f"d{i}", minute=10 + i, second=0, event_type_name="Pressure", team_id=2)
        for i in range(4)
    ]
    shot_clock = 20 * 60.0
    columns = defensive_action_rates(events, defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=shot_clock)
    assert columns["defensive_action_rate_null_reason"] is None
    assert columns["defensive_action_rate_15m"] == pytest.approx(4 / 15.0)


def test_defensive_action_rates_partial_window_uses_actual_elapsed_not_nominal():
    # Only 5 minutes have elapsed since period start (shot_clock=300s), well under the 15m
    # nominal window. The denominator must be the actual 5 elapsed minutes, not 15 --
    # otherwise the rate would be silently deflated (the anti-pattern the task forbids).
    events = [_event(event_id="start", minute=0, second=0, event_type_name="Pass", team_id=1)]
    events += [_event(event_id="d0", minute=2, second=0, event_type_name="Pressure", team_id=2)]
    shot_clock = 5 * 60.0
    columns = defensive_action_rates(events, defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=shot_clock)
    assert columns["defensive_action_rate_null_reason"] is None
    assert columns["defensive_action_rate_15m"] == pytest.approx(1 / 5.0)
    assert columns["defensive_action_rate_60m"] == pytest.approx(1 / 5.0)


def test_defensive_action_rates_ignores_non_defending_team_and_wrong_type():
    events = [
        _event(event_id="start", minute=0, second=0, event_type_name="Pass", team_id=1),
        _event(event_id="own", minute=1, second=0, event_type_name="Pressure", team_id=1),  # attacking team, ignored
        _event(event_id="wrong_type", minute=1, second=0, event_type_name="Pass", team_id=2),  # not defensive action
        _event(event_id="d0", minute=1, second=0, event_type_name="Pressure", team_id=2),
    ]
    shot_clock = 5 * 60.0
    columns = defensive_action_rates(events, defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=shot_clock)
    assert columns["defensive_action_rate_15m"] == pytest.approx(1 / 5.0)


def test_defensive_action_rates_ignores_other_period():
    events = [
        _event(event_id="start", period=1, minute=0, second=0, event_type_name="Pass", team_id=1),
        _event(event_id="d0", period=2, minute=1, second=0, event_type_name="Pressure", team_id=2),
    ]
    shot_clock = 5 * 60.0
    columns = defensive_action_rates(events, defending_team_id=2, action_types=ACTION_TYPES, shot_period=1, shot_clock=shot_clock)
    assert columns["defensive_action_rate_15m"] == pytest.approx(0.0)


# --- territorial_dominance_extended ----------------------------------------------------------


def test_territorial_dominance_extended_none_without_shot_clock():
    assert territorial_dominance_extended([], shot_team_id=1, shot_period=1, shot_clock=None) is None


def test_territorial_dominance_extended_positive_when_team_dominates():
    events = [
        _event(event_id=f"t{i}", minute=1, second=0, event_type_name="Pass", team_id=1, location_x=100.0)
        for i in range(3)
    ]
    value = territorial_dominance_extended(events, shot_team_id=1, shot_period=1, shot_clock=600.0)
    assert value == pytest.approx(1.0)


def test_territorial_dominance_extended_window_excludes_old_events():
    # An event far outside the 15-minute window must not affect the ratio.
    events = [
        _event(event_id="old", minute=0, second=0, event_type_name="Pass", team_id=2, location_x=100.0),
        _event(event_id="recent", minute=19, second=0, event_type_name="Pass", team_id=1, location_x=100.0),
    ]
    shot_clock = 20 * 60.0
    value = territorial_dominance_extended(events, shot_team_id=1, shot_period=1, shot_clock=shot_clock)
    assert value == pytest.approx(1.0)
