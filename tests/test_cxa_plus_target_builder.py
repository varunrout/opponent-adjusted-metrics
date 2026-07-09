import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_cxa_plus_targets import (
    PRIMARY_TARGET,
    TARGET_COLUMNS,
    build_cxa_plus_target_dataset,
    build_cxa_plus_targets,
    build_leakage_exclusions,
    compute_quality_checks,
    validate_required_columns,
)


def _action_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": "a1",
                "event_id": "e1",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 1,
                "team_id": 100,
                "player_id": 1000,
                "start_x": 30.0,
                "start_y": 40.0,
                "end_x": 45.0,
                "end_y": 42.0,
                "action_type": "Pass",
                "period": 1,
                "minute": 1,
                "second": 3,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
            {
                "action_id": "a2",
                "event_id": "e2",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 2,
                "team_id": 100,
                "player_id": 1001,
                "start_x": 45.0,
                "start_y": 42.0,
                "end_x": 65.0,
                "end_y": 42.0,
                "action_type": "Carry",
                "period": 1,
                "minute": 1,
                "second": 7,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
            {
                "action_id": "a3",
                "event_id": "e3",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 3,
                "team_id": 100,
                "player_id": 1002,
                "start_x": 65.0,
                "start_y": 42.0,
                "end_x": 88.0,
                "end_y": 40.0,
                "action_type": "Shot assist",
                "period": 1,
                "minute": 1,
                "second": 9,
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.30,
            },
            {
                "action_id": "a4",
                "event_id": "e4",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 4,
                "team_id": 100,
                "player_id": 1003,
                "start_x": 20.0,
                "start_y": 50.0,
                "end_x": 21.0,
                "end_y": 51.0,
                "action_type": "Recovery",
                "period": 1,
                "minute": 1,
                "second": 15,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
            {
                "action_id": "b1",
                "event_id": "e5",
                "match_id": 1,
                "possession": 11,
                "sequence_id": "s2",
                "action_position": 1,
                "team_id": 200,
                "player_id": 2000,
                "start_x": 35.0,
                "start_y": 30.0,
                "end_x": 55.0,
                "end_y": 35.0,
                "action_type": "Pass",
                "period": 1,
                "minute": 2,
                "second": 1,
                "shot_created": 1,
                "created_shot_id": "shot-2",
                "created_shot_cxg": 0.10,
            },
            {
                "action_id": "c1",
                "event_id": "e6",
                "match_id": 2,
                "possession": 10,
                "sequence_id": "s3",
                "action_position": 1,
                "team_id": 300,
                "player_id": 3000,
                "start_x": 40.0,
                "start_y": 20.0,
                "end_x": 50.0,
                "end_y": 20.0,
                "action_type": "Pass",
                "period": 1,
                "minute": 3,
                "second": 1,
                "shot_created": 1,
                "created_shot_id": "shot-3",
                "created_shot_cxg": None,
            },
        ]
    )


def _write_feature_table(root: Path, frame: pd.DataFrame | None = None) -> Path:
    feature_path = root / "feature_store" / "cxa" / "action_features.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    (frame if frame is not None else _action_features()).to_parquet(feature_path, index=False)
    return feature_path


def test_script_writes_all_required_outputs(tmp_path: Path):
    feature_path = _write_feature_table(tmp_path)
    outputs = build_cxa_plus_target_dataset(
        feature_path=feature_path,
        output_dir=tmp_path / "feature_store" / "cxa_plus",
        design_summary_path=tmp_path / "missing_design_summary.json",
    )

    assert set(outputs) == {"targets", "summary", "quality", "leakage_exclusions", "report"}
    for path in outputs.values():
        assert path.exists()

    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["primary_target"] == "shot_within_next_5_actions"
    assert summary["row_count"] == len(_action_features())


def test_next_5_target_only_looks_at_future_actions_not_current_action():
    targets = build_cxa_plus_targets(_action_features())
    by_id = targets.set_index("action_id")

    assert by_id.loc["a1", PRIMARY_TARGET] == 1
    assert by_id.loc["a2", PRIMARY_TARGET] == 1
    assert by_id.loc["a3", PRIMARY_TARGET] == 0
    assert by_id.loc["a3", "shot_later_in_possession"] == 0


def test_target_does_not_cross_possession_boundaries():
    targets = build_cxa_plus_targets(_action_features())
    by_id = targets.set_index("action_id")

    assert by_id.loc["a4", PRIMARY_TARGET] == 0
    assert by_id.loc["b1", PRIMARY_TARGET] == 0


def test_target_does_not_cross_match_boundaries():
    targets = build_cxa_plus_targets(_action_features())
    by_id = targets.set_index("action_id")

    assert by_id.loc["b1", "match_id"] != by_id.loc["c1", "match_id"]
    assert by_id.loc["b1", PRIMARY_TARGET] == 0


def test_same_created_shot_id_is_not_counted_as_downstream_for_current_shot():
    frame = pd.DataFrame(
        [
            {
                "action_id": "a1",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 1,
                "team_id": 100,
                "player_id": 1,
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.25,
            },
            {
                "action_id": "a2",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 2,
                "team_id": 100,
                "player_id": 2,
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.25,
            },
            {
                "action_id": "a3",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 3,
                "team_id": 100,
                "player_id": 3,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
        ]
    )

    targets = build_cxa_plus_targets(frame).set_index("action_id")

    assert targets.loc["a1", "shot_within_next_1_action"] == 0
    assert targets.loc["a1", PRIMARY_TARGET] == 0
    assert targets.loc["a1", "shot_later_in_possession"] == 0
    assert targets.loc["a1", "max_created_shot_cxg_within_next_5_actions"] == 0.0
    assert targets.loc["a1", "sum_created_shot_cxg_rest_of_possession"] == 0.0
    assert targets.loc["a1", "discounted_downstream_shot_value"] == 0.0


def test_value_targets_deduplicate_repeated_future_created_shot_id():
    frame = pd.DataFrame(
        [
            {
                "action_id": "a1",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 1,
                "team_id": 100,
                "player_id": 1,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
            {
                "action_id": "a2",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 2,
                "team_id": 100,
                "player_id": 2,
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.40,
            },
            {
                "action_id": "a3",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 3,
                "team_id": 100,
                "player_id": 3,
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.40,
            },
            {
                "action_id": "a4",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 4,
                "team_id": 100,
                "player_id": 4,
                "shot_created": 1,
                "created_shot_id": "shot-2",
                "created_shot_cxg": 0.10,
            },
        ]
    )

    targets = build_cxa_plus_targets(frame).set_index("action_id")

    assert targets.loc["a1", PRIMARY_TARGET] == 1
    assert targets.loc["a1", "max_created_shot_cxg_within_next_5_actions"] == 0.40
    assert targets.loc["a1", "sum_created_shot_cxg_rest_of_possession"] == pytest.approx(0.50)
    assert targets.loc["a1", "discounted_downstream_shot_value"] == pytest.approx(0.40 + (0.10 / 3))


def test_missing_future_created_shot_ids_are_not_collapsed_together():
    frame = pd.DataFrame(
        [
            {
                "action_id": "a1",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 1,
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
            },
            {
                "action_id": "a2",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 2,
                "shot_created": 1,
                "created_shot_id": None,
                "created_shot_cxg": 0.20,
            },
            {
                "action_id": "a3",
                "match_id": 1,
                "possession": 10,
                "sequence_id": "s1",
                "action_position": 3,
                "shot_created": 1,
                "created_shot_id": None,
                "created_shot_cxg": 0.30,
            },
        ]
    )

    targets = build_cxa_plus_targets(frame).set_index("action_id")

    assert targets.loc["a1", "sum_created_shot_cxg_rest_of_possession"] == pytest.approx(0.50)


def test_quality_check_catches_same_created_shot_id_current_label_leakage():
    bad_targets = build_cxa_plus_targets(_action_features())
    bad_targets = bad_targets.loc[bad_targets["action_id"].isin(["a3", "a4"])].copy()
    bad_targets.loc[bad_targets["action_id"] == "a4", "shot_created"] = 1
    bad_targets.loc[bad_targets["action_id"] == "a4", "created_shot_id"] = "shot-1"
    bad_targets.loc[bad_targets["action_id"] == "a3", PRIMARY_TARGET] = 1

    quality = compute_quality_checks(bad_targets, missing_order_fields=[])
    current_leakage = quality.loc[
        quality["check_name"] == "no_current_action_label_leakage_check"
    ].iloc[0]

    assert current_leakage["status"] == "failed"
    assert int(current_leakage["value"]) == 1


def test_deterministic_ordering_is_used():
    shuffled = _action_features().sample(frac=1.0, random_state=7)
    targets = build_cxa_plus_targets(shuffled)

    assert targets["action_id"].tolist() == ["a1", "a2", "a3", "a4", "b1", "c1"]
    assert targets.loc[targets["action_id"] == "a1", PRIMARY_TARGET].iloc[0] == 1


def test_missing_action_position_fails_clearly():
    frame = _action_features().drop(columns=["action_position"])

    with pytest.raises(ValueError, match="missing_required_order_fields"):
        validate_required_columns(frame)
    with pytest.raises(ValueError, match="action_position"):
        build_cxa_plus_targets(frame)


def test_leakage_exclusions_include_target_and_reference_fields():
    targets = build_cxa_plus_targets(_action_features())
    exclusions = build_leakage_exclusions(targets)
    excluded = set(exclusions["column"])

    assert set(TARGET_COLUMNS).issubset(excluded)
    assert {"shot_created", "created_shot_id", "created_shot_cxg"}.issubset(excluded)
    assert "discounted_downstream_shot_value" in excluded


def test_no_existing_cxa_artifacts_are_modified(tmp_path: Path):
    feature_path = _write_feature_table(tmp_path)
    original = pd.read_parquet(feature_path)

    build_cxa_plus_target_dataset(
        feature_path=feature_path,
        output_dir=tmp_path / "feature_store" / "cxa_plus",
        design_summary_path=tmp_path / "missing_design_summary.json",
    )

    after = pd.read_parquet(feature_path)
    pd.testing.assert_frame_equal(original, after)


def test_no_model_training_is_imported_or_called():
    source = Path("scripts/build_cxa_plus_targets.py").read_text(encoding="utf-8")

    assert "run_cxa_diagnostic_training" not in source
    assert ".fit(" not in source
    assert "joblib" not in source
