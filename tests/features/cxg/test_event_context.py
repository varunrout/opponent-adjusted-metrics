from opponent_adjusted.features.cxg import event_context
from opponent_adjusted.features.cxg.contracts import event_candidate_names_for_families
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
        "player_id": 10,
        "play_pattern_name": "Regular Play",
    }
    values.update(overrides)
    return event_context.EventRecord(**values)


def _shot(event_id, index, **overrides):
    overrides.setdefault("event_type_name", "Shot")
    overrides.setdefault("location_x", 100.0)
    return _event(event_id, index, **overrides)


def _values(events, event_id):
    return event_context.derive_event_contexts(events)[event_id].values


def test_taxonomy_membership_is_exact_governed_e1_e6_bank():
    expected = event_candidate_names_for_families(("E1", "E2", "E3", "E4", "E5", "E6"))
    assert event_context.CXG_EVENT_CONTEXT_E1_E6_CONTRACT_ID == "cxg_event_context_e1_e6_v1"
    assert event_context.E1_E6_FEATURES == expected
    assert len(expected) == 34
    assert "score_difference_pre_shot" not in expected
    assert "match_elapsed_s" not in expected
    assert "goalward_progress_m" not in expected
    assert "statsbomb_xg" not in expected
    assert "is_goal" not in expected
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


def test_score_game_state_clock_and_late_flags_are_strictly_pre_shot():
    events = [
        _shot("goal", 1, minute=74, second=59, outcome_name="Goal"),
        _shot("lead", 2, minute=75, second=0),
        _shot("opponent", 3, minute=76, second=0, team_id=2, possession_team_id=2),
    ]
    contexts = event_context.derive_event_contexts(events)
    assert contexts["goal"].value("score_diff") == 0
    assert contexts["lead"].value("score_diff") == 1
    assert contexts["lead"].value("game_state") == "leading"
    assert contexts["lead"].value("match_minute") == 75.0
    assert contexts["lead"].value("regulation_time_remaining") == 900.0
    assert contexts["lead"].value("late_game_leading") is True
    assert contexts["opponent"].value("score_diff") == -1


def test_own_goal_cards_and_period_five_governance():
    events = [
        _event("own-for", 1, event_type_name="Own Goal For"),
        _event("own-against", 2, event_type_name="Own Goal Against", team_id=2),
        _event("red", 3, card_name="Red Card", player_id=99),
        _event("red-duplicate", 4, card_name="Second Yellow", player_id=99),
        _shot("after", 5),
        _shot("shootout", 6, period=5, minute=120, second=0),
    ]
    contexts = event_context.derive_event_contexts(events)
    assert contexts["after"].value("score_diff") == 1
    assert contexts["after"].value("manpower_diff") == -1
    assert contexts["shootout"].value("score_diff") is None
    assert contexts["shootout"].value("game_state") is None
    assert contexts["shootout"].value("late_game_leading") is None


def test_origin_regain_windows_complexity_and_tempo():
    events = [
        _event(
            "origin",
            1,
            second=0,
            location_x=100,
            location_y=40,
            play_pattern_name="From Counter",
            player_id=1,
        ),
        _event(
            "receipt", 2, second=4, event_type_name="Ball Receipt*", location_x=101, player_id=2
        ),
        _event(
            "carry",
            3,
            second=7,
            event_type_name="Carry",
            location_x=102,
            end_x=110,
            end_y=40,
            player_id=1,
        ),
        _event("dribble", 4, second=8, event_type_name="Dribble", location_x=110, player_id=3),
        _shot("shot", 5, second=10, location_x=112),
    ]
    values = _values(events, "shot")
    assert values["possession_start_x"] == 100
    assert values["possession_start_zone"] == "final_third"
    assert values["possession_start_type"] == "counter"
    assert values["restart_vs_live_regain"] == "live_regain"
    assert values["high_regain"] is True
    assert values["shot_within_5s_regain"] is False
    assert values["shot_within_10s_regain"] is True
    assert values["shot_within_15s_regain"] is True
    assert values["possession_action_count"] == 4
    assert values["possession_pass_count"] == 1
    assert values["possession_carry_count"] == 1
    assert values["possession_dribble_count"] == 1
    assert values["unique_attackers_involved"] == 3
    assert values["recorded_receipt_count"] == 1
    assert values["avg_action_interval_s"] == 2.5
    assert values["last_action_interval_s"] == 2.0
    assert values["actions_per_second"] == 0.4


def test_restart_and_invalid_possession_context_are_null_or_false_as_governed():
    restart = _values([_shot("restart", 1, play_pattern_name="From Corner")], "restart")
    mismatch = _values([_shot("mismatch", 1, possession_team_id=2)], "mismatch")
    assert restart["restart_vs_live_regain"] == "restart"
    assert restart["high_regain"] is False
    assert restart["shot_within_5s_regain"] is None
    assert mismatch["possession_start_x"] is None
    assert mismatch["possession_action_count"] is None


def test_e6_native_goal_progress_path_directness_and_actions():
    events = [
        _event("start", 1, second=0, location_x=40, location_y=40, player_id=1),
        _event("pass", 2, second=2, location_x=40, location_y=40, end_x=60, end_y=40, player_id=2),
        _event("missing", 3, second=3, location_x=None, location_y=None),
        _event(
            "carry",
            4,
            second=4,
            location_x=60,
            location_y=40,
            end_x=90,
            end_y=40,
            event_type_name="Carry",
            player_id=3,
        ),
        _shot("shot", 5, second=10, location_x=100, location_y=40),
    ]
    values = _values(events, "shot")
    assert values["goalward_progress_sb"] == 60.0
    assert values["recorded_event_path_length_sb"] == 60.0
    assert values["possession_directness_proxy"] == 1.0
    assert values["pace_to_goal"] == 6.0
    assert values["progressive_actions_n"] == 2
    assert values["max_single_action_progress_sb"] == 30.0


def test_e6_signed_and_zero_age_behaviour():
    retreat = _values(
        [_event("start", 1, second=0, location_x=100), _shot("shot", 2, second=5, location_x=90)],
        "shot",
    )
    first = _values([_shot("first", 1)], "first")
    assert retreat["goalward_progress_sb"] < 0
    assert retreat["pace_to_goal"] < 0
    assert first["possession_age_s"] == 0
    assert first["avg_action_interval_s"] is None
    assert first["actions_per_second"] is None
    assert first["pace_to_goal"] is None


def test_no_360_cloud_or_post_shot_dependency():
    source = event_context.__loader__.get_source(event_context.__name__)
    for forbidden in (
        "three_sixty_frames",
        "three_sixty_players",
        "shot_freeze_frame_players",
        "google.cloud",
        "statsbomb_xg",
        "end_z",
    ):
        assert forbidden not in source


def test_match_score_and_dismissal_state_are_isolated_with_shared_ids():
    match_a = [
        _shot("a-goal", 1, match_id=1, outcome_name="Goal"),
        _event("a-red", 2, match_id=1, card_name="Red Card", player_id=10),
    ]
    match_b = [_shot("b-shot", 1, match_id=2, player_id=10, second=1)]

    context = event_context.derive_event_contexts([*match_a, *match_b])["b-shot"]

    assert context.value("score_diff") == 0
    assert context.value("game_state") == "drawing"
    assert context.value("manpower_diff") == 0


def test_possession_identity_and_batch_order_are_match_isolated():
    match_a = [
        _event("a-start", 1, match_id=1, possession_id=1, second=0, location_x=20),
        _shot("a-shot", 2, match_id=1, possession_id=1, second=5, location_x=60),
    ]
    match_b = [
        _event("b-start", 1, match_id=2, possession_id=1, second=0, location_x=80),
        _shot("b-shot", 2, match_id=2, possession_id=1, second=5, location_x=100),
    ]
    grouped = event_context.derive_event_contexts([*match_a, *match_b])
    interleaved = event_context.derive_event_contexts(
        [match_a[0], match_b[0], match_a[1], match_b[1]]
    )
    reversed_blocks = event_context.derive_event_contexts([*match_b, *match_a])

    for event_id in ("a-shot", "b-shot"):
        assert (
            grouped[event_id].values
            == interleaved[event_id].values
            == reversed_blocks[event_id].values
        )
    assert grouped["a-shot"].value("possession_start_x") == 20
    assert grouped["b-shot"].value("possession_start_x") == 80


def test_e1_card_regulation_and_late_threshold_boundaries():
    events = [
        _event("yellow", 1, card_name="Yellow Card"),
        _shot("at-60", 2, minute=60, second=0),
        _shot("before-75", 3, minute=74, second=59, team_id=2, possession_team_id=2),
        _event("second-yellow", 4, card_name="Second Yellow", player_id=55),
        _shot("after-card", 5, minute=91, second=0),
    ]
    contexts = event_context.derive_event_contexts(events)

    assert contexts["at-60"].value("manpower_diff") == 0
    assert contexts["at-60"].value("regulation_time_remaining") == 1800.0
    assert contexts["before-75"].value("late_game_trailing") is False
    assert contexts["after-card"].value("regulation_time_remaining") == 0.0
    assert contexts["after-card"].value("manpower_diff") == -1


def test_origin_zone_boundaries_invalid_origin_and_live_regain_thresholds():
    defensive = _values([_shot("defensive", 1, location_x=20)], "defensive")
    middle = _values([_shot("middle", 1, location_x=60)], "middle")
    final = _values([_shot("final", 1, location_x=100)], "final")
    invalid_then_valid = _values(
        [_event("invalid", 1, location_x=121), _shot("valid", 2, location_x=100)], "valid"
    )

    assert defensive["possession_start_zone"] == "defensive_third"
    assert middle["possession_start_zone"] == "middle_third"
    assert final["possession_start_zone"] == "final_third"
    assert invalid_then_valid["possession_start_x"] == 100
    assert final["possession_start_goal_distance"] == 20.0
    assert final["deep_regain"] is False


def test_live_regain_window_nesting_and_restart_nulls():
    def flags(seconds):
        timestamp = f"00:00:{seconds:02d}.000" if isinstance(seconds, int) else "00:00:04.900"
        values = _values(
            [
                _event("start", 1, second=0, play_pattern_name="From Counter"),
                _shot("shot", 2, second=int(seconds), timestamp=timestamp),
            ],
            "shot",
        )
        return (
            values["shot_within_5s_regain"],
            values["shot_within_10s_regain"],
            values["shot_within_15s_regain"],
        )

    assert flags(4.9) == (True, True, True)
    assert flags(7) == (False, True, True)
    assert flags(12) == (False, False, True)
    assert flags(20) == (False, False, False)
    restart = _values([_shot("corner", 1, play_pattern_name="From Corner")], "corner")
    assert restart["shot_within_5s_regain"] is None


def test_e4_excludes_opponents_adjacent_possessions_and_current_shot():
    values = _values(
        [
            _event("prior", 1, possession_id=1, event_type_name="Pass"),
            _event("opponent", 2, team_id=2, possession_team_id=1, possession_id=2),
            _event("start", 3, possession_id=2, event_type_name="Carry"),
            _shot("shot", 4, possession_id=2),
        ],
        "shot",
    )
    assert values["possession_action_count"] == 1
    assert values["possession_pass_count"] == 0
    assert values["possession_carry_count"] == 1


def test_e6_lateral_progress_threshold_and_regressive_action_cases():
    lateral = _values(
        [
            _event("start", 1, location_x=100, location_y=0),
            _shot("shot", 2, second=5, location_x=100, location_y=40),
        ],
        "shot",
    )
    actions = _values(
        [
            _event("under", 1, event_type_name="Pass", location_x=40, end_x=47, end_y=40),
            _event("boundary", 2, event_type_name="Pass", location_x=40, end_x=48, end_y=40),
            _event("carry", 3, event_type_name="Carry", location_x=60, end_x=66, end_y=40),
            _event("non-action", 4, event_type_name="Dribble", location_x=80, end_x=120, end_y=40),
            _shot("shot", 5, second=10, location_x=90),
        ],
        "shot",
    )
    regressive = _values(
        [
            _event("pass", 1, event_type_name="Pass", location_x=80, end_x=60, end_y=40),
            _shot("shot", 2, second=5, location_x=70),
        ],
        "shot",
    )

    assert lateral["goalward_progress_sb"] > 0
    assert actions["progressive_actions_n"] == 2
    assert regressive["max_single_action_progress_sb"] < 0
