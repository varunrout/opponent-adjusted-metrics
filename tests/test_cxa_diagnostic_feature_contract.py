import json
from pathlib import Path

import pandas as pd

from scripts.prepare_cxa_diagnostic_feature_contract import (
    prepare_cxa_diagnostic_feature_contract,
)


def _synthetic_cxa_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": ["a1", "a2", "a3"],
            "event_id": ["e1", "e2", "e3"],
            "match_id": [1, 1, 2],
            "team_id": [10, 10, 20],
            "player_id": [100, 101, 102],
            "possession_id": ["p1", "p1", "p2"],
            "sequence_id": ["s1", "s1", "s2"],
            "created_shot_id": ["shot1", None, None],
            "shot_created": [1, 0, 0],
            "created_shot_cxg": [0.25, 0.0, 0.0],
            "created_shot_distance": [12.0, None, None],
            "created_shot_angle": [0.4, None, None],
            "cxa_value": [0.2, 0.0, 0.0],
            "predicted_cxa": [0.2, 0.1, 0.05],
            "predicted_shot_created_probability": [0.3, 0.05, 0.02],
            "mean_predicted_probability": [0.2, 0.1, 0.05],
            "sequence_cxa": [0.2, 0.2, 0.0],
            "possession_cxa": [0.2, 0.2, 0.0],
            "cxa_share": [1.0, 0.0, 0.0],
            "led_to_shot": [1, 0, 0],
            "positive_mean_created_shot_cxg": [0.25, 0.25, 0.25],
            "baseline_probability": [0.1, 0.1, 0.1],
            "start_x": [40.0, 50.0, 55.0],
            "start_y": [30.0, 35.0, 40.0],
            "end_x": [55.0, 60.0, 65.0],
            "end_y": [35.0, 38.0, 42.0],
            "length": [15.0, 10.0, 11.0],
            "is_pass": [1, 0, 1],
            "under_pressure": [0, 1, 0],
            "enters_final_third": [1, 0, 1],
            "action_type": ["Pass", "Carry", "Pass"],
            "play_pattern": ["Open Play", "Open Play", "From Throw In"],
            "distance_to_goal_before": [60.0, 50.0, 45.0],
            "distance_to_goal_after": [45.0, 40.0, 35.0],
            "angle_to_goal_before": [0.2, 0.3, 0.4],
            "angle_to_goal_after": [0.3, 0.4, 0.5],
            "final_shot_xg": [0.25, 0.0, 0.0],
            "future_shot_xg": [0.25, 0.0, 0.0],
            "shot_outcome": ["Goal", None, None],
            "actions_until_shot": [1, None, None],
            "next_action_is_shot": [1, 0, 0],
            "post_action_result": ["complete", "complete", "turnover"],
        }
    )


def _write_features(tmp_path: Path, frame: pd.DataFrame | None = None) -> Path:
    path = tmp_path / "feature_store" / "cxa" / "action_features.parquet"
    path.parent.mkdir(parents=True)
    (frame if frame is not None else _synthetic_cxa_features()).to_parquet(path, index=False)
    return path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_excludes_targets_references_outputs_and_identifiers(tmp_path: Path):
    input_path = _write_features(tmp_path)
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=input_path,
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    contract = _load_json(outputs["feature_contract"])
    selected = set().union(
        *[set(values) for values in contract["selected_feature_candidates"].values()]
    )
    excluded = contract["excluded_columns"]

    assert contract["primary_target"] == "shot_created"
    assert contract["attribution_reference"] == "created_shot_cxg"
    assert contract["value_output"] == "cxa_value"
    assert "shot_created" in excluded["target_columns"]
    assert "created_shot_cxg" in excluded["reference_only_columns"]
    assert "cxa_value" in excluded["output_prediction_columns"]
    assert "action_id" in excluded["identifier_columns"]
    assert "player_id" in excluded["identifier_columns"]
    assert "created_shot_id" in excluded["identifier_columns"]
    assert not selected & {
        "shot_created",
        "created_shot_cxg",
        "created_shot_id",
        "cxa_value",
        "predicted_cxa",
        "action_id",
        "match_id",
        "team_id",
        "player_id",
    }


def test_predicted_and_probability_columns_are_output_prediction_exclusions(tmp_path: Path):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    excluded = pd.read_csv(outputs["excluded_columns"])
    mapped = dict(zip(excluded["column"], excluded["exclusion_type"]))

    assert mapped["predicted_cxa"] == "output_prediction"
    assert mapped["predicted_shot_created_probability"] == "output_prediction"
    assert mapped["mean_predicted_probability"] == "output_prediction"
    assert mapped["baseline_probability"] == "output_prediction"


def test_created_shot_id_sparsity_is_not_a_hard_failure(tmp_path: Path):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    resolved = _load_json(outputs["resolved_features"])

    assert resolved["missing_required_columns"] == []
    assert resolved["identifier_summary"]["created_shot_id"]["missing"] == 2
    assert "created_shot_id sparsity is expected" in "\n".join(resolved["review_notes"])


def test_before_after_spatial_columns_require_review_not_allowed_normally(tmp_path: Path):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    contract = _load_json(outputs["feature_contract"])
    group_summary = pd.read_csv(outputs["feature_group_summary"])

    for column in (
        "distance_to_goal_before",
        "distance_to_goal_after",
        "angle_to_goal_before",
        "angle_to_goal_after",
    ):
        assert column in contract["excluded_columns"]["requires_review_columns"]
        row = group_summary[group_summary["column"] == column].iloc[0]
        assert row["classification"] == "requires_review"
        assert row["feature_group"] == "reviewed_spatial_context"


def test_allowed_candidates_keep_safe_numeric_binary_and_categorical_features(tmp_path: Path):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    contract = _load_json(outputs["feature_contract"])

    assert {"start_x", "start_y", "end_x", "end_y", "length"}.issubset(
        set(contract["selected_feature_candidates"]["numeric"])
    )
    assert {"is_pass", "under_pressure"}.issubset(
        set(contract["selected_feature_candidates"]["binary"])
    )
    assert "enters_final_third" in contract["selected_feature_candidates"]["binary"]
    assert {"action_type", "play_pattern"}.issubset(
        set(contract["selected_feature_candidates"]["categorical"])
    )


def test_final_third_context_is_allowed_but_future_shot_patterns_are_leakage(
    tmp_path: Path,
):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    contract = _load_json(outputs["feature_contract"])
    group_summary = pd.read_csv(outputs["feature_group_summary"])

    assert "enters_final_third" in contract["selected_feature_candidates"]["binary"]
    final_third = group_summary[group_summary["column"] == "enters_final_third"].iloc[0]
    assert final_third["classification"] == "allowed_binary"

    leakage_columns = set(contract["excluded_columns"]["leakage_excluded_columns"])
    assert {
        "final_shot_xg",
        "future_shot_xg",
        "shot_outcome",
        "actions_until_shot",
        "next_action_is_shot",
        "post_action_result",
    }.issubset(leakage_columns)


def test_contract_artifacts_are_written_with_expected_shapes(tmp_path: Path):
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    for path in outputs.values():
        assert path.exists()

    resolved = _load_json(outputs["resolved_features"])
    excluded = pd.read_csv(outputs["excluded_columns"])
    group_summary = pd.read_csv(outputs["feature_group_summary"])
    report = outputs["feature_contract_report"].read_text(encoding="utf-8")

    assert resolved["target_summary"]["positive_count"] == 1
    assert resolved["identifier_summary"]["action_id"]["missing"] == 0
    assert {"column", "exclusion_type", "reason", "severity", "can_appear_in_outputs"}.issubset(
        set(excluded.columns)
    )
    assert {
        "feature_group",
        "column",
        "classification",
        "dtype",
        "missing_count",
        "missing_pct",
        "distinct_count",
        "notes",
    }.issubset(set(group_summary.columns))
    assert "# CxA Diagnostic Feature Contract Report" in report
    assert "This PR does not train a model." in report


def test_missing_optional_columns_do_not_crash(tmp_path: Path):
    minimal = pd.DataFrame(
        {
            "action_id": ["a1", "a2"],
            "match_id": [1, 1],
            "team_id": [10, 10],
            "shot_created": [0, 1],
            "start_x": [40.0, 50.0],
            "action_type": ["Pass", "Carry"],
        }
    )
    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=_write_features(tmp_path, minimal),
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )
    contract = _load_json(outputs["feature_contract"])

    assert contract["row_count"] == 2
    assert "start_x" in contract["selected_feature_candidates"]["numeric"]
    assert "action_type" in contract["selected_feature_candidates"]["categorical"]


def test_makefile_contains_prepare_contract_target():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    assert "prepare-cxa-diagnostic-contract:" in makefile
    assert "poetry run python scripts/prepare_cxa_diagnostic_feature_contract.py" in makefile
