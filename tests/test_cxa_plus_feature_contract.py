from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.prepare_cxa_plus_feature_contract import (
    PRIMARY_TARGET,
    prepare_cxa_plus_feature_contract,
)


def _build_targets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": ["a1", "a2", "a3", "a4"],
            "event_id": ["e1", "e2", "e3", "e4"],
            "match_id": ["m1", "m1", "m2", "m2"],
            "possession": [1, 1, 2, 2],
            "sequence_id": ["s1", "s1", "s2", "s2"],
            "team_id": ["t1", "t1", "t2", "t2"],
            "player_id": ["p1", "p2", "p3", "p4"],
            "action_type": ["carry", "pass", "pass", "dribble"],
            "action_position": [1, 2, 3, 4],
            "start_x": [30.0, 42.0, 58.0, 71.0],
            "start_y": [48.0, 41.0, 55.0, 62.0],
            "end_x": [42.0, 58.0, 71.0, 88.0],
            "end_y": [41.0, 55.0, 62.0, 50.0],
            "minute": [5, 17, 61, 80],
            "second": [12, 31, 10, 45],
            "shot_created": [0, 1, 0, 0],
            "created_shot_id": ["", "shot-9", "", ""],
            "created_shot_cxg": [0.0, 0.32, 0.0, 0.0],
            "shot_within_next_1_action": [0, 1, 0, 0],
            "shot_within_next_3_actions": [1, 1, 0, 1],
            "shot_within_next_5_actions": [1, 1, 0, 1],
            "shot_later_in_possession": [1, 1, 0, 1],
            "max_created_shot_cxg_within_next_5_actions": [0.14, 0.32, 0.0, 0.25],
            "sum_created_shot_cxg_rest_of_possession": [0.14, 0.32, 0.0, 0.25],
            "discounted_downstream_shot_value": [0.12, 0.29, 0.0, 0.2],
        }
    )


def test_primary_target_present_and_not_eligible_feature(tmp_path: Path) -> None:
    frame = _build_targets_frame()
    input_path = tmp_path / "targets.parquet"
    output_dir = tmp_path / "outputs"
    frame.to_parquet(input_path, index=False)

    prepare_cxa_plus_feature_contract(input_path=input_path, output_dir=output_dir)
    contract = json.loads(
        (output_dir / "contracts" / "feature_contract.json").read_text(encoding="utf-8")
    )

    eligible = set().union(
        *[set(values) for values in contract["eligible_feature_candidates"].values()]
    )
    assert contract["primary_target"] == PRIMARY_TARGET
    assert PRIMARY_TARGET in contract["target_columns"]
    assert PRIMARY_TARGET not in eligible


def test_required_leakage_and_reference_columns_excluded(tmp_path: Path) -> None:
    frame = _build_targets_frame()
    input_path = tmp_path / "targets.parquet"
    output_dir = tmp_path / "outputs"
    frame.to_parquet(input_path, index=False)

    prepare_cxa_plus_feature_contract(input_path=input_path, output_dir=output_dir)
    contract = json.loads(
        (output_dir / "contracts" / "feature_contract.json").read_text(encoding="utf-8")
    )
    excluded = set(
        contract["target_columns"]
        + contract["reference_only_columns"]
        + contract["leakage_excluded_columns"]
    )
    assert "shot_created" in excluded
    assert "created_shot_id" in excluded
    assert "created_shot_cxg" in excluded
    assert "max_created_shot_cxg_within_next_5_actions" in excluded
    assert "sum_created_shot_cxg_rest_of_possession" in excluded
    assert "discounted_downstream_shot_value" in excluded


def test_identifier_columns_are_not_eligible_features(tmp_path: Path) -> None:
    frame = _build_targets_frame()
    input_path = tmp_path / "targets.parquet"
    output_dir = tmp_path / "outputs"
    frame.to_parquet(input_path, index=False)

    prepare_cxa_plus_feature_contract(input_path=input_path, output_dir=output_dir)
    contract = json.loads(
        (output_dir / "contracts" / "feature_contract.json").read_text(encoding="utf-8")
    )
    eligible = set().union(
        *[set(values) for values in contract["eligible_feature_candidates"].values()]
    )

    for column in [
        "action_id",
        "event_id",
        "match_id",
        "possession",
        "sequence_id",
        "team_id",
        "player_id",
    ]:
        assert column in contract["identifier_columns"]
        assert column not in eligible


def test_at_least_one_valid_feature_remains(tmp_path: Path) -> None:
    frame = _build_targets_frame()
    input_path = tmp_path / "targets.parquet"
    output_dir = tmp_path / "outputs"
    frame.to_parquet(input_path, index=False)

    prepare_cxa_plus_feature_contract(input_path=input_path, output_dir=output_dir)
    contract = json.loads(
        (output_dir / "contracts" / "feature_contract.json").read_text(encoding="utf-8")
    )
    eligible = set().union(
        *[set(values) for values in contract["eligible_feature_candidates"].values()]
    )

    assert eligible
    assert {"start_x", "start_y", "end_x", "end_y", "action_type"}.intersection(eligible)


def test_expected_outputs_are_written(tmp_path: Path) -> None:
    frame = _build_targets_frame()
    input_path = tmp_path / "targets.parquet"
    output_dir = tmp_path / "outputs"
    frame.to_parquet(input_path, index=False)

    outputs = prepare_cxa_plus_feature_contract(input_path=input_path, output_dir=output_dir)

    for path in outputs.values():
        assert path.exists()
    assert (output_dir / "contracts" / "feature_contract.json").exists()
    assert (output_dir / "diagnostics" / "resolved_features.json").exists()
    assert (output_dir / "diagnostics" / "excluded_columns.csv").exists()
    assert (output_dir / "diagnostics" / "feature_group_summary.csv").exists()
    assert (output_dir / "reports" / "feature_contract_report.md").exists()


def test_script_does_not_import_or_execute_training() -> None:
    source = (
        Path("scripts/prepare_cxa_plus_feature_contract.py").read_text(encoding="utf-8").lower()
    )
    assert "sklearn" not in source
    assert ".fit(" not in source
    assert "joblib" not in source
    assert "train_" not in source
