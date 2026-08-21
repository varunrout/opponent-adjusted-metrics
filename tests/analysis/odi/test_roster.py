from opponent_adjusted.analysis.odi.roster import MatchRoster, StarterRecord, SubstitutionRecord


def _roster(subs=(), period_starts=None):
    starters = [
        StarterRecord(team_id=1, player_id=101, position_name="Center Back"),
        StarterRecord(team_id=1, player_id=102, position_name="Goalkeeper"),
        StarterRecord(team_id=2, player_id=201, position_name="Center Back"),
    ]
    return MatchRoster(
        starters=starters,
        subs=list(subs),
        period_start_clock_s=period_starts or {1: 0.0, 2: 2700.0},
    )


def test_starter_on_pitch_from_kickoff():
    roster = _roster()
    on_pitch = roster.on_pitch(team_id=1, period=1, clock_s=100.0)
    ids = {p.player_id for p in on_pitch}
    assert ids == {101, 102}


def test_starter_removed_after_sub_off():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = _roster(subs=subs)
    before = {p.player_id for p in roster.on_pitch(team_id=1, period=1, clock_s=1100.0)}
    after = {p.player_id for p in roster.on_pitch(team_id=1, period=1, clock_s=1300.0)}
    assert 101 in before and 301 not in before
    assert 101 not in after and 301 in after


def test_substitute_position_lookup_used():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = MatchRoster(
        starters=[StarterRecord(team_id=1, player_id=101, position_name="Center Back")],
        subs=subs,
        period_start_clock_s={1: 0.0},
        substitute_position_lookup={301: "Left Back"},
    )
    on_pitch = roster.on_pitch(team_id=1, period=1, clock_s=1300.0)
    assert on_pitch[0].position_name == "Left Back"


def test_period_tenure_starts_at_kickoff_for_continuous_starter():
    roster = _roster()
    tenure = roster.period_tenure_s(player_id=101, team_id=1, period=1, clock_s=1000.0)
    assert tenure == 1000.0


def test_period_tenure_resets_at_new_period_for_continuous_player():
    # Player 101 never subbed, but period 2 tenure measures from period-2 kickoff.
    roster = _roster(period_starts={1: 0.0, 2: 2700.0})
    tenure = roster.period_tenure_s(player_id=101, team_id=1, period=2, clock_s=2750.0)
    assert tenure == 50.0


def test_period_tenure_starts_at_sub_on_time_within_same_period():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = _roster(subs=subs)
    tenure = roster.period_tenure_s(player_id=301, team_id=1, period=1, clock_s=1500.0)
    assert tenure == 300.0


def test_substitute_carried_forward_gets_period_start_tenure_in_later_period():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = _roster(subs=subs, period_starts={1: 0.0, 2: 2700.0})
    tenure = roster.period_tenure_s(player_id=301, team_id=1, period=2, clock_s=2800.0)
    assert tenure == 100.0


def test_player_not_on_pitch_returns_none_tenure():
    roster = _roster()
    assert roster.period_tenure_s(player_id=999, team_id=1, period=1, clock_s=1000.0) is None


def test_subbed_off_player_not_on_pitch_after_sub():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = _roster(subs=subs)
    assert roster.period_tenure_s(player_id=101, team_id=1, period=1, clock_s=1300.0) is None


def test_other_team_players_unaffected_by_sub():
    subs = [SubstitutionRecord(team_id=1, period=1, clock_s=1200.0, player_off_id=101, player_on_id=301)]
    roster = _roster(subs=subs)
    on_pitch = {p.player_id for p in roster.on_pitch(team_id=2, period=1, clock_s=1300.0)}
    assert on_pitch == {201}
