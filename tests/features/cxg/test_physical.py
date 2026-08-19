from opponent_adjusted.features.cxg import physical


def _state(**overrides):
    values = {
        "event_id": "event",
        "match_id": 1,
        "competition_id": 43,
        "season_id": 106,
        "player_id": 10,
        "team_id": 20,
        "raw_shot_x": 100.0,
        "raw_shot_y": 40.0,
        "period": 1,
        "data_version": "data",
        "silver_schema_version": "statsbomb_silver_v1_2",
        "body_part_name": "Right Foot",
        "technique_name": "Normal",
        "shot_type_name": "Open Play",
        "first_time": None,
        "open_goal": None,
        "one_on_one": None,
        "follows_dribble": None,
        "under_pressure": None,
        "is_goal": False,
        "statsbomb_xg": 0.1,
        "end_x": 120.0,
        "end_y": 40.0,
        "end_z": 1.0,
        "outcome_name": "Saved",
        "saved_off_target": False,
        "saved_to_post": False,
        "key_pass_id": "pass",
        "deflected": False,
        "aerial_won": False,
    }
    values.update(overrides)
    return physical.build_physical_shot_state(**values)


def test_exact_s_contract_and_governance_roles():
    assert physical.CXG_PHYSICAL_STATE_CONTRACT_ID == "cxg_physical_state_v1"
    assert set(physical.S_FEATURES) == {
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
    }
    assert "shot_distance_m" not in physical.S_FEATURES
    assert physical.TARGET_FIELDS == ("is_goal",)
    assert physical.BENCHMARK_FIELDS == ("statsbomb_xg",)
    assert set(physical.POST_SHOT_FIELDS).isdisjoint(physical.S_FEATURES)
    assert physical.LINKAGE_FIELDS == ("key_pass_id",)
    assert set(physical.EXCLUDED_PHYSICAL_FIELDS).isdisjoint(physical.S_FEATURES)


def test_model_eligibility_excludes_only_period_five_or_invalid_geometry():
    assert _state(period=5).physical_model_eligible is False
    assert _state(period=4).physical_model_eligible is True
    assert _state(period=1, raw_shot_x=120.1).physical_model_eligible is False


def test_nullable_booleans_and_raw_audit_fields_are_preserved():
    state = _state()

    assert state.first_time is None
    assert state.open_goal is None
    assert state.one_on_one is None
    assert state.follows_dribble is None
    assert state.under_pressure is None
    assert state.raw_shot_x == 100.0
    assert state.raw_shot_y == 40.0
    assert state.geometry.shot_distance_sb == 20.0


def test_physical_module_has_no_360_dependency():
    source = physical.__loader__.get_source(physical.__name__)

    assert "three_sixty_frames" not in source
    assert "three_sixty_players" not in source
    assert "shot_freeze_frame_players" not in source
