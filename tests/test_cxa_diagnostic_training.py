import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cxa_diagnostic_training import (
    assert_leakage_guard,
    resolve_selected_features,
    run_cxa_diagnostic_training,
)


def _synthetic_actions(row_count: int = 60, include_optional_ids: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(row_count):
        row = {
            "action_id": f"a{idx}",
            "event_id": f"e{idx}",
            "match_id": idx % 10,
            "team_id": 10 + (idx % 3),
            "player_id": 100 + (idx % 8),
            "shot_created": 1 if idx % 8 == 0 else 0,
            "created_shot_cxg": 0.2 if idx % 8 == 0 else 0.0,
            "cxa_value": 0.18 if idx % 8 == 0 else 0.0,
            "created_shot_id": f"s{idx}" if idx % 8 == 0 else None,
            "predicted_cxa": 0.1,
            "start_x": 30.0 + (idx % 20),
            "start_y": 20.0 + (idx % 10),
            "end_x": 40.0 + (idx % 25),
            "end_y": 25.0 + (idx % 12),
            "length": 8.0 + (idx % 6),
            "is_pass": 1 if idx % 2 == 0 else 0,
            "under_pressure": None if idx % 11 == 0 else idx % 5 == 0,
            "enters_final_third": 1 if idx % 4 == 0 else 0,
            "action_type": "Pass" if idx % 2 == 0 else "Carry",
            "play_pattern": "Open Play" if idx % 3 else "From Throw In",
            "distance_to_goal_before": 60.0 - (idx % 15),
        }
        if include_optional_ids:
            row["sequence_id"] = f"seq{idx % 6}"
            row["possession"] = idx % 15
        rows.append(row)
    df = pd.DataFrame(rows)
    df["under_pressure"] = df["under_pressure"].astype("boolean")
    return df


def _contract() -> dict:
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "primary_target": "shot_created",
        "attribution_reference": "created_shot_cxg",
        "value_output": "cxa_value",
        "selected_feature_candidates": {
            "numeric": ["start_x", "start_y", "end_x", "end_y", "length"],
            "binary": ["is_pass", "under_pressure", "enters_final_third"],
            "categorical": ["action_type", "play_pattern"],
        },
        "excluded_columns": {
            "target_columns": ["shot_created"],
            "reference_only_columns": ["created_shot_cxg", "created_shot_id"],
            "output_prediction_columns": ["cxa_value", "predicted_cxa"],
            "leakage_excluded_columns": [],
            "identifier_columns": [
                "action_id",
                "event_id",
                "match_id",
                "team_id",
                "player_id",
                "sequence_id",
                "possession",
            ],
            "requires_review_columns": ["distance_to_goal_before"],
            "excluded_unknown_columns": [],
        },
    }


def _write_inputs(tmp_path: Path, *, include_optional_ids: bool = True) -> tuple[Path, Path]:
    feature_path = tmp_path / "feature_store" / "cxa" / "action_features.parquet"
    contract_path = (
        tmp_path
        / "outputs"
        / "modeling"
        / "cxa"
        / "diagnostic_v1"
        / "contracts"
        / "feature_contract.json"
    )
    feature_path.parent.mkdir(parents=True)
    contract_path.parent.mkdir(parents=True)
    _synthetic_actions(include_optional_ids=include_optional_ids).to_parquet(
        feature_path, index=False
    )
    contract_path.write_text(json.dumps(_contract(), indent=2), encoding="utf-8")
    return feature_path, contract_path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_training_script_reads_selected_features_from_contract(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path)
    frame = pd.read_parquet(feature_path)
    contract = _load_json(contract_path)

    resolved = resolve_selected_features(frame, contract)

    assert resolved == contract["selected_feature_candidates"]


def test_leakage_guard_fails_if_forbidden_column_is_selected():
    contract = _contract()
    feature_groups = {
        "numeric": ["start_x", "created_shot_cxg"],
        "binary": [],
        "categorical": [],
    }

    with pytest.raises(ValueError, match="leakage guard failed"):
        assert_leakage_guard(feature_groups, contract)


def test_selected_features_exclude_targets_references_outputs_and_identifiers(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path)
    frame = pd.read_parquet(feature_path)
    resolved = resolve_selected_features(frame, _load_json(contract_path))
    selected = set().union(*[set(values) for values in resolved.values()])

    assert not selected & {
        "shot_created",
        "created_shot_cxg",
        "cxa_value",
        "created_shot_id",
        "predicted_cxa",
        "action_id",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "sequence_id",
        "possession",
        "distance_to_goal_before",
    }


def test_tiny_diagnostic_training_run_writes_outputs(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1"

    outputs = run_cxa_diagnostic_training(
        input_path=feature_path,
        contract_path=contract_path,
        output_dir=output_dir,
        random_state=7,
    )

    assert outputs.model_candidates.exists()
    assert outputs.selected_model.exists()
    assert outputs.selected_model_metadata.exists()
    assert outputs.cross_validated_predictions.exists()
    assert outputs.model_comparison.exists()
    assert outputs.training_report.exists()
    assert outputs.training_summary.exists()

    comparison = pd.read_csv(outputs.model_comparison)
    assert {
        "candidate_name",
        "log_loss",
        "brier",
        "roc_auc",
        "average_precision",
        "positive_rate",
        "mean_predicted_probability",
        "calibration_error",
        "selected",
        "notes",
    }.issubset(set(comparison.columns))
    assert set(comparison["candidate_name"]) == {
        "logistic_regression",
        "calibrated_logistic_regression",
        "gradient_boosting",
        "calibrated_gradient_boosting_sigmoid",
    }
    assert comparison["selected"].sum() == 1


def test_cross_validated_predictions_have_required_columns_and_probabilities(
    tmp_path: Path,
):
    feature_path, contract_path = _write_inputs(tmp_path)
    outputs = run_cxa_diagnostic_training(
        input_path=feature_path,
        contract_path=contract_path,
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    predictions = pd.read_parquet(outputs.cross_validated_predictions)

    assert {
        "action_id",
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "sequence_id",
        "possession",
        "shot_created",
        "predicted_shot_created_probability",
        "model_candidate",
        "fold",
        "split",
    }.issubset(set(predictions.columns))
    assert predictions["predicted_shot_created_probability"].between(0, 1).all()
    assert "created_shot_cxg" not in predictions.columns
    assert "cxa_value" not in predictions.columns


def test_selected_model_metadata_records_counts_and_positive_rate(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path)
    outputs = run_cxa_diagnostic_training(
        input_path=feature_path,
        contract_path=contract_path,
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    metadata = _load_json(outputs.selected_model_metadata)

    assert metadata["metric"] == "cxa"
    assert metadata["model_version"] == "diagnostic_v1"
    assert metadata["primary_target"] == "shot_created"
    assert metadata["attribution_reference"] == "created_shot_cxg"
    assert metadata["value_output"] == "cxa_value"
    assert metadata["selected_feature_count"] == 10
    assert metadata["numeric_feature_count"] == 5
    assert metadata["binary_feature_count"] == 3
    assert metadata["categorical_feature_count"] == 2
    assert metadata["leakage_guard_passed"] is True
    assert metadata["positive_count"] > 0
    assert 0 < metadata["positive_rate"] < 1


def test_training_report_mentions_target_reference_output_separation(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path)
    outputs = run_cxa_diagnostic_training(
        input_path=feature_path,
        contract_path=contract_path,
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    report = outputs.training_report.read_text(encoding="utf-8")

    assert "# Diagnostic CxA Training Report" in report
    assert "predicts `shot_created`" in report
    assert "excludes `created_shot_cxg`, `cxa_value`, identifiers" in report
    assert "Validation and promotion" not in report
    assert "does not validate or promote" in report


def test_missing_optional_id_columns_do_not_crash_prediction_export(tmp_path: Path):
    feature_path, contract_path = _write_inputs(tmp_path, include_optional_ids=False)
    outputs = run_cxa_diagnostic_training(
        input_path=feature_path,
        contract_path=contract_path,
        output_dir=tmp_path / "outputs" / "modeling" / "cxa" / "diagnostic_v1",
    )

    predictions = pd.read_parquet(outputs.cross_validated_predictions)

    assert "sequence_id" not in predictions.columns
    assert "possession" not in predictions.columns
    assert "action_id" in predictions.columns
