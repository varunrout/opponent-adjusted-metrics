from opponent_adjusted.analysis.odi.aggregator import WINDOW_S, InvolvementRow, compute_odi


def _row(player_id, shot_id, clock_s, xg, is_goal, period=1, match_id=1):
    return InvolvementRow(player_id, match_id, shot_id, period, clock_s, xg, is_goal)


def test_cold_start_below_window_yields_null_with_reason():
    result = compute_odi(101, 1, 1, 500.0, "shotA", [], period_tenure_s=400.0)
    assert result.odi is None
    assert not result.eligible
    assert result.null_reason == "cold_start_lt_15min"


def test_not_on_pitch_yields_null_with_reason():
    result = compute_odi(101, 1, 1, 500.0, "shotA", [], period_tenure_s=None)
    assert result.odi is None
    assert result.null_reason == "not_on_pitch"


def test_eligible_sums_xg_minus_goals_in_window():
    rows = [
        _row(101, "shot1", clock_s=100.0, xg=0.1, is_goal=False),
        _row(101, "shot2", clock_s=200.0, xg=0.3, is_goal=True),
        _row(101, "shot3", clock_s=800.0, xg=0.2, is_goal=False),
    ]
    result = compute_odi(101, 1, 1, 1000.0, "shotX", rows, period_tenure_s=1000.0)
    assert result.eligible
    assert result.involvement_count == 3
    assert abs(result.odi - (0.1 + 0.3 + 0.2 - 1)) < 1e-9


def test_window_excludes_events_outside_trailing_15_minutes():
    rows = [
        _row(101, "shot1", clock_s=1000.0 - WINDOW_S - 1, xg=0.5, is_goal=False),  # just outside
        _row(101, "shot2", clock_s=1000.0 - WINDOW_S + 1, xg=0.2, is_goal=False),  # just inside
    ]
    result = compute_odi(101, 1, 1, 1000.0, "shotX", rows, period_tenure_s=1000.0)
    assert result.involvement_count == 1
    assert abs(result.odi - 0.2) < 1e-9


def test_self_shot_excluded_by_id_even_at_zero_diff():
    rows = [_row(101, "shotX", clock_s=1000.0, xg=0.9, is_goal=False)]
    result = compute_odi(101, 1, 1, 1000.0, "shotX", rows, period_tenure_s=1000.0)
    assert result.involvement_count == 0
    assert result.odi == 0.0


def test_window_never_bridges_periods():
    rows = [_row(101, "shot1", clock_s=100.0, xg=0.9, is_goal=False, period=1)]
    result = compute_odi(101, 1, 2, 200.0, "shotX", rows, period_tenure_s=1000.0)
    assert result.involvement_count == 0


def test_other_players_and_matches_excluded():
    rows = [
        _row(999, "shot1", clock_s=500.0, xg=0.9, is_goal=False, match_id=1),
        _row(101, "shot2", clock_s=500.0, xg=0.7, is_goal=False, match_id=2),
    ]
    result = compute_odi(101, 1, 1, 1000.0, "shotX", rows, period_tenure_s=1000.0)
    assert result.involvement_count == 0
