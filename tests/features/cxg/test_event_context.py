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
