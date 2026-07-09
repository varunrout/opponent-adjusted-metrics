from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_cxa_plus_feature_matrix import (
    PRIMARY_TARGET,
    build_cxa_plus_feature_matrix,
)


def _action_features_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": "a1",
                "event_id": "e1",
                "match_id": 1,
                "possession": 11,
                "sequence_id": "s1",
                "action_position": 1,
                "team_id": 10,
                "player_id": 100,
                "length": 10.2,
                "angle": 0.2,
                "x_progression": 9.0,
                "y_progression": 1.4,
                "distance_to_goal_before": 55.0,
                "distance_to_goal_after": 46.0,
                "angle_to_goal_before": 0.4,
                "angle_to_goal_after": 0.55,
                "is_pass": 1,
                "is_carry": 0,
                "is_dribble": 0,
                "is_cross": 0,
                "is_cutback": 0,
                "is_through_ball": 0,
                "is_progressive": 1,
                "enters_final_third": 1,
                "enters_penalty_area": 0,
                "enters_zone14": 1,
                "switches_play": 0,
                "body_part": "right_foot",
                "pass_height": "ground",
                "start_zone": "left_half",
                "end_zone": "zone14",
                "start_third": "middle",
                "end_third": "final",
                "under_pressure": 1,
                "prior_action_type": "carry",
                "prior_action_success": 1,
                "seconds_since_possession_start": 4.0,
                "sequence_length_so_far": 1,
                "set_piece_phase": "open_play",
                "shot_created": 0,
                "created_shot_id": None,
                "created_shot_cxg": 0.0,
                "created_shot_distance": 0.0,
                "predicted_shot_created_probability": 0.1,
            },
            {
                "action_id": "a2",
                "event_id": "e2",
                "match_id": 1,
                "possession": 11,
                "sequence_id": "s1",
                "action_position": 2,
                "team_id": 10,
                "player_id": 101,
                "length": 22.0,
                "angle": 0.35,
                "x_progression": 16.0,
                "y_progression": 2.0,
                "distance_to_goal_before": 46.0,
                "distance_to_goal_after": 30.0,
                "angle_to_goal_before": 0.55,
                "angle_to_goal_after": 0.9,
                "is_pass": 0,
                "is_carry": 1,
                "is_dribble": 0,
                "is_cross": 0,
                "is_cutback": 0,
                "is_through_ball": 1,
                "is_progressive": 1,
                "enters_final_third": 1,
                "enters_penalty_area": 1,
                "enters_zone14": 0,
                "switches_play": 0,
                "body_part": "right_foot",
                "pass_height": "high",
                "start_zone": "zone14",
                "end_zone": "box",
                "start_third": "final",
                "end_third": "final",
                "under_pressure": 0,
                "prior_action_type": "pass",
                "prior_action_success": 1,
                "seconds_since_possession_start": 8.0,
                "sequence_length_so_far": 2,
                "set_piece_phase": "open_play",
                "shot_created": 1,
                "created_shot_id": "shot-1",
                "created_shot_cxg": 0.3,
                "created_shot_angle": 0.8,
                "model_score": 0.2,
            },
        ]
    )


def _targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": "a1",
                "event_id": "e1",
                "match_id": 1,
                "possession": 11,
                "sequence_id": "s1",
                "action_position": 1,
                "team_id": 10,
                "player_id": 100,
                "shot_within_next_1_action": 1,
                "shot_within_next_3_actions": 1,
                "shot_within_next_5_actions": 1,
                "shot_later_in_possession": 1,
                "max_created_shot_cxg_within_next_5_actions": 0.3,
                "sum_created_shot_cxg_rest_of_possession": 0.3,
                "discounted_downstream_shot_value": 0.3,
            },
            {
                "action_id": "a2",
                "event_id": "e2",
                "match_id": 1,
                "possession": 11,
                "sequence_id": "s1",
                "action_position": 2,
                "team_id": 10,
                "player_id": 101,
                "shot_within_next_1_action": 0,
                "shot_within_next_3_actions": 0,
                "shot_within_next_5_actions": 0,
                "shot_later_in_possession": 0,
                "max_created_shot_cxg_within_next_5_actions": 0.0,
                "sum_created_shot_cxg_rest_of_possession": 0.0,
                "discounted_downstream_shot_value": 0.0,
            },
        ]
    )


def _contract_payload() -> dict[str, list[str] | str]:
    return {
        "model_version": "diagnostic_v1",
        "primary_target": PRIMARY_TARGET,
        "identifier_columns": [
            "action_id",
            "event_id",
            "match_id",
            "possession",
            "sequence_id",
            "action_position",
            "team_id",
            "player_id",
        ],
        "target_columns": [
            "shot_within_next_1_action",
            "shot_within_next_3_actions",
            "shot_within_next_5_actions",
            "shot_later_in_possession",
        ],
        "reference_only_columns": ["shot_created", "created_shot_id", "created_shot_cxg"],
        "leakage_excluded_columns": [
            "max_created_shot_cxg_within_next_5_actions",
            "sum_created_shot_cxg_rest_of_possession",
            "discounted_downstream_shot_value",
        ],
        "model_output_columns": ["predicted_shot_created_probability", "model_score"],
        "requires_review_columns": [],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    action_path = tmp_path / "feature_store" / "cxa" / "action_features.parquet"
    target_path = tmp_path / "feature_store" / "cxa_plus" / "cxa_plus_action_targets.parquet"
    contract_path = (
        tmp_path
        / "outputs"
        / "modeling"
        / "cxa_plus"
        / "diagnostic_v1"
        / "contracts"
        / "feature_contract.json"
    )
    action_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    _action_features_frame().to_parquet(action_path, index=False)
    _targets_frame().to_parquet(target_path, index=False)
    contract_path.write_text(json.dumps(_contract_payload(), indent=2), encoding="utf-8")
    return action_path, target_path, contract_path


def test_feature_matrix_row_count_matches_target_and_primary_target_present(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = build_cxa_plus_feature_matrix(
        action_features_path=action_path,
        targets_path=target_path,
        contract_path=contract_path,
        output_dir=output_dir,
    )

    matrix = pd.read_parquet(outputs["feature_matrix"])
    targets = pd.read_parquet(target_path)
    assert len(matrix) == len(targets)
    assert PRIMARY_TARGET in matrix.columns


def test_richer_safe_features_are_retained_when_present(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = build_cxa_plus_feature_matrix(
        action_features_path=action_path,
        targets_path=target_path,
        contract_path=contract_path,
        output_dir=output_dir,
    )

    matrix = pd.read_parquet(outputs["feature_matrix"])
    summary = json.loads(outputs["feature_matrix_summary"].read_text(encoding="utf-8"))
    expected = {
        "length",
        "angle",
        "x_progression",
        "y_progression",
        "distance_to_goal_before",
        "distance_to_goal_after",
        "angle_to_goal_before",
        "angle_to_goal_after",
        "is_pass",
        "is_carry",
        "is_through_ball",
        "is_progressive",
        "enters_final_third",
        "enters_penalty_area",
        "body_part",
        "pass_height",
        "start_zone",
        "end_zone",
        "under_pressure",
        "prior_action_type",
        "prior_action_success",
        "seconds_since_possession_start",
        "sequence_length_so_far",
        "set_piece_phase",
    }
    assert expected.issubset(set(matrix.columns))
    assert expected.issubset(set(summary["eligible_model_features"]))


def test_leakage_columns_and_created_shot_prefix_are_not_eligible(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = build_cxa_plus_feature_matrix(
        action_features_path=action_path,
        targets_path=target_path,
        contract_path=contract_path,
        output_dir=output_dir,
    )

    summary = json.loads(outputs["feature_matrix_summary"].read_text(encoding="utf-8"))
    eligible = set(summary["eligible_model_features"])
    forbidden = {
        "shot_created",
        "created_shot_id",
        "created_shot_cxg",
        "created_shot_distance",
        "created_shot_angle",
        "shot_within_next_1_action",
        "shot_within_next_3_actions",
        "shot_within_next_5_actions",
        "shot_later_in_possession",
        "max_created_shot_cxg_within_next_5_actions",
        "sum_created_shot_cxg_rest_of_possession",
        "discounted_downstream_shot_value",
        "predicted_shot_created_probability",
        "model_score",
    }
    assert forbidden.isdisjoint(eligible)
    assert not any(column.startswith("created_shot_") for column in eligible)


def test_duplicate_join_keys_fail_clearly(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    duplicated_actions = pd.concat(
        [_action_features_frame(), _action_features_frame().iloc[[0]]], ignore_index=True
    )
    duplicated_actions.to_parquet(action_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        build_cxa_plus_feature_matrix(
            action_features_path=action_path,
            targets_path=target_path,
            contract_path=contract_path,
            output_dir=output_dir,
        )


def test_unmatched_target_rows_fail_clearly(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    bad_targets = _targets_frame().copy()
    bad_targets.loc[0, "action_id"] = "missing-action"
    bad_targets.to_parquet(target_path, index=False)

    with pytest.raises(ValueError, match="did not match action features"):
        build_cxa_plus_feature_matrix(
            action_features_path=action_path,
            targets_path=target_path,
            contract_path=contract_path,
            output_dir=output_dir,
        )


def test_script_writes_required_outputs(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = build_cxa_plus_feature_matrix(
        action_features_path=action_path,
        targets_path=target_path,
        contract_path=contract_path,
        output_dir=output_dir,
    )

    for path in outputs.values():
        assert path.exists()


def test_existing_cxg_cxa_and_target_artifacts_are_not_modified(tmp_path: Path) -> None:
    action_path, target_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    cxg_artifact = tmp_path / "outputs" / "modeling" / "cxg" / "diagnostic_v1" / "sentinel.json"
    cxa_artifact = tmp_path / "outputs" / "results" / "cxa" / "diagnostic_v1" / "sentinel.csv"
    cxa_plus_target_artifact = (
        tmp_path / "feature_store" / "cxa_plus" / "cxa_plus_target_summary.json"
    )
    for path in (cxg_artifact, cxa_artifact, cxa_plus_target_artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel", encoding="utf-8")

    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (cxg_artifact, cxa_artifact, cxa_plus_target_artifact, action_path, target_path)
    }

    build_cxa_plus_feature_matrix(
        action_features_path=action_path,
        targets_path=target_path,
        contract_path=contract_path,
        output_dir=output_dir,
    )

    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (cxg_artifact, cxa_artifact, cxa_plus_target_artifact, action_path, target_path)
    }
    assert before == after


def test_no_model_training_is_imported_or_executed() -> None:
    source = Path("scripts/build_cxa_plus_feature_matrix.py").read_text(encoding="utf-8").lower()
    assert "sklearn" not in source
    assert ".fit(" not in source
    assert "joblib" not in source
    assert "run_cxa_diagnostic_training" not in source
