from opponent_adjusted.features.cxg import event_context
from opponent_adjusted.features.cxg.physical import S_FEATURES


def _event(event_id, index, **overrides):
    values = {
        "event_id": event_id,
        "match_id": 1,
        "event_index": index,
        "period": 1,
        "minute": 0,
        "second": index,
        "timestamp": f"00:00:{index:02d}.000",
        "event_type_name": "Pass",
        "outcome_name": None,
        "team_id": 1,
        "possession_id": 1,
        "possession_team_id": 1,
        "location_x": 40.0,
        "location_y": 40.0,
    }
    values.update(overrides)
    return event_context.EventRecord(**values)


def _shot(event_id, index, **overrides):
    overrides.setdefault("event_type_name", "Shot")
    overrides.setdefault("location_x", 100.0)
    return _event(event_id, index, **overrides)


def test_e1_pre_shot_score_is_strict_and_preserves_goal_differences():
    events = [
        _shot("first-goal", 1, outcome_name="Goal"),
        _shot("lead-one", 2),
        _shot("second-goal", 3, outcome_name="Goal"),
        _shot("lead-two", 4),
        _shot("opponent-goal", 5, team_id=2, possession_team_id=2, outcome_name="Goal"),
        _shot("lead-one-again", 6),
        _shot("opponent-shot", 7, team_id=2, possession_team_id=2),
    ]
    contexts = event_context.derive_event_contexts(events)

    assert contexts["first-goal"].score_difference_pre_shot == 0
    assert contexts["lead-one"].score_difference_pre_shot == 1
    assert contexts["lead-two"].score_difference_pre_shot == 2
    assert contexts["lead-one-again"].score_difference_pre_shot == 1
    assert contexts["opponent-shot"].score_difference_pre_shot == -1


def test_own_goal_for_counts_once_for_benefiting_team():
    events = [
        _event("own-for", 1, event_type_name="Own Goal For", team_id=1),
        _event("own-against", 2, event_type_name="Own Goal Against", team_id=2),
        _shot("after-own", 3),
    ]

    context = event_context.derive_event_contexts(events)["after-own"]

    assert context.score_difference_pre_shot == 1
    assert event_context.goal_beneficiary_team_id(events[1]) is None


def test_period_five_score_is_null_and_final_score_excludes_shootout():
    events = [
        _shot("normal-goal", 1, outcome_name="Goal"),
        _shot("shootout", 2, period=5, minute=120, second=0, outcome_name="Goal"),
    ]
    contexts = event_context.derive_event_contexts(events)
    validation = event_context.validate_final_score(
        events, home_team_id=1, away_team_id=2, home_score=1, away_score=0
    )

    assert contexts["shootout"].score_difference_pre_shot is None
    assert validation.matches_metadata is True


def test_final_score_validation_reports_mismatch_without_becoming_a_feature():
    validation = event_context.validate_final_score(
        [_shot("goal", 1, outcome_name="Goal")],
        home_team_id=1,
        away_team_id=2,
        home_score=0,
        away_score=0,
    )

    assert validation.matches_metadata is False


def test_match_clock_retains_fraction_stoppage_and_extra_time():
    events = [
        _shot("fraction", 1, minute=10, second=2, timestamp="00:10:02.500"),
        _shot("stoppage", 2, minute=48, second=30, timestamp="00:03:30.125"),
        _shot("extra", 3, period=3, minute=105, second=1, timestamp="00:00:01.250"),
        _shot("invalid", 4, timestamp="invalid"),
    ]
    contexts = event_context.derive_event_contexts(events)

    assert contexts["fraction"].match_elapsed_s == 602.5
    assert contexts["stoppage"].match_elapsed_s == 2910.125
    assert contexts["extra"].match_elapsed_s == 6301.25
    assert contexts["invalid"].match_elapsed_s is None


def test_possession_age_count_progression_and_speed_use_only_prior_same_team_events():
    events = [
        _event("prior-possession", 1, possession_id=1, location_x=10.0),
        _event("opponent", 2, team_id=2, possession_team_id=1, location_x=90.0),
        _event("start", 3, possession_id=2, location_x=40.0, second=0),
        _event("same-team", 4, possession_id=2, location_x=60.0, second=5),
        _shot("shot", 5, possession_id=2, location_x=100.0, second=10),
    ]
    context = event_context.derive_event_contexts(events)["shot"]

    assert context.possession_age_s == 10.0
    assert context.possession_team_events_pre_shot == 2
    assert context.possession_net_progression_sb == 60.0
    assert context.possession_progression_speed_sb_s == 6.0


def test_first_event_shot_has_zero_age_and_null_speed():
    context = event_context.derive_event_contexts([_shot("first", 1)])["first"]

    assert context.possession_age_s == 0.0
    assert context.possession_team_events_pre_shot == 0
    assert context.possession_net_progression_sb == 0.0
    assert context.possession_progression_speed_sb_s is None


def test_negative_age_invalid_coordinates_and_invalid_possession_are_not_fabricated():
    events = [
        _event("start", 1, second=10, location_x=200.0),
        _shot("negative", 2, second=5),
        _shot("mismatch", 3, possession_id=2, possession_team_id=2),
    ]
    contexts = event_context.derive_event_contexts(events)

    assert contexts["negative"].possession_age_s is None
    assert contexts["negative"].possession_net_progression_sb == 0.0
    assert contexts["negative"].possession_progression_speed_sb_s is None
    assert contexts["mismatch"].possession_context_valid is False
    assert contexts["mismatch"].possession_team_events_pre_shot is None


def test_invalid_shot_location_keeps_progression_and_speed_missing():
    context = event_context.derive_event_contexts(
        [_event("start", 1, location_x=40.0), _shot("invalid", 2, location_x=120.1)]
    )["invalid"]

    assert context.possession_net_progression_sb is None
    assert context.possession_progression_speed_sb_s is None


def test_signed_progression_and_speed_are_not_absolute_or_epsilon_adjusted():
    events = [
        _event("start", 1, location_x=100.0, second=0),
        _shot("retreat", 2, location_x=90.0, second=5),
    ]
    context = event_context.derive_event_contexts(events)["retreat"]

    assert context.possession_net_progression_sb == -10.0
    assert context.possession_progression_speed_sb_s == -2.0


def test_contract_is_exact_and_does_not_broaden_phase_3b1():
    assert event_context.CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID == "cxg_event_context_e1_e6_v1"
    assert event_context.E1_E6_FEATURES == (
        "score_difference_pre_shot",
        "match_elapsed_s",
        "possession_age_s",
        "possession_team_events_pre_shot",
        "possession_net_progression_sb",
        "possession_progression_speed_sb_s",
    )
    assert all(not feature.startswith("previous_") for feature in event_context.E1_E6_FEATURES)
    assert "statsbomb_xg" not in event_context.E1_E6_FEATURES
    assert "is_goal" not in event_context.E1_E6_FEATURES
    assert "end_x" not in event_context.E1_E6_FEATURES
    assert S_FEATURES == (
        "shot_distance_sb",
        "shot_angle_rad",
        "body_part_name",
        "technique_name",
        "shot_type_name",
        "first_time",
        "open_goal",
        "one_on_one",
        "follows_dribble",
        "under_pressure",
    )


def test_event_context_module_has_no_360_or_post_shot_dependency():
    source = event_context.__loader__.get_source(event_context.__name__)

    assert "three_sixty_frames" not in source
    assert "three_sixty_players" not in source
    assert "shot_freeze_frame_players" not in source
    assert "statsbomb_xg" not in source
    assert "end_x" not in source
