import json
from pathlib import Path

import pandas as pd

from scripts.validate_feature_contract import validate


CONTRACT_PATH = Path("configs/feature_contracts/cxa_v1.json")
DESIGN_DOC_PATH = Path("docs/modeling/cxa/design_contract.md")


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_cxa_contract_declares_design_stage_and_target_contract():
    contract = _load_contract()

    assert contract["metric"] == "cxa"
    assert contract["status"] == "design_contract"
    assert contract["grain"] == (
        "one row per eligible attacking action with a fixed lookahead window for shot creation"
    )
    assert contract["definition"]["event_data_only"] is True
    assert contract["definition"]["tracking_data_required"] is False
    assert contract["target"]["target_column"] == "created_shot_cxg"
    assert contract["target"]["shot_creation_indicator"] == "shot_created"
    assert contract["target"]["zero_when_no_shot"] is True
    assert set(contract["target_columns"]) == {"shot_created", "created_shot_cxg"}


def test_cxa_contract_defines_required_features_and_leakage_guardrails():
    contract = _load_contract()

    for key in (
        "identity_columns",
        "required_numeric_features",
        "required_binary_features",
        "required_categorical_features",
        "optional_context_features",
        "prohibited_leakage_columns",
        "forbidden_training_features",
    ):
        assert contract[key], f"{key} should not be empty"

    assert "match_id" in contract["identity_columns"]
    assert contract["split_group_column"] == "match_id"
    assert "created_shot_outcome" in contract["prohibited_leakage_columns"]
    assert "post_shot_xg" in contract["forbidden_training_features"]
    assert "created_shot_cxg" not in contract["required_numeric_features"]


def test_cxa_contract_defines_validation_and_output_contracts():
    contract = _load_contract()
    validation = contract["validation_contract"]
    output = contract["output_contract"]

    assert "grouped_validation_by_match" in validation["required_checks"]
    assert "no_forbidden_training_features" in validation["required_checks"]
    assert validation["target_bounds"] == {
        "created_shot_cxg_min": 0.0,
        "created_shot_cxg_max": 1.0,
    }
    assert "traditional_assist_indicator" in validation["baseline_comparisons"]

    assert output["feature_store_dir"] == "feature_store/cxa/"
    assert output["modeling_dir"] == "outputs/modeling/cxa/"
    expected_paths = output["expected_future_paths"]
    assert expected_paths["features"] == "feature_store/cxa/action_features.parquet"
    assert expected_paths["predictions"].endswith("predictions/action_predictions.parquet")
    assert expected_paths["player_aggregates"].endswith("aggregates/player_cxa.parquet")
    assert expected_paths["team_aggregates"].endswith("aggregates/team_cxa.parquet")


def test_cxa_design_document_contains_required_sections():
    text = DESIGN_DOC_PATH.read_text(encoding="utf-8")

    for heading in (
        "## Definition",
        "## Target",
        "## Eligible Actions",
        "## Attribution Logic",
        "## Feature Families",
        "## Leakage Risks",
        "## Baseline Model Plan",
        "## Validation Plan",
        "## Output Contract",
        "## Limitations",
    ):
        assert heading in text

    assert "not evidence that a final CxA model has been trained" in text
    assert "must not require tracking data" in text


def test_cxa_contract_validator_accepts_minimal_synthetic_action_table(tmp_path: Path):
    contract = _load_contract()
    row = {}
    for key in (
        "identity_columns",
        "target_columns",
        "required_numeric_features",
        "required_binary_features",
        "required_categorical_features",
    ):
        for column in contract[key]:
            row[column] = 0

    row.update(
        {
            "action_id": "action-1",
            "event_id": "event-1",
            "sequence_id": "sequence-1",
            "match_id": 1,
            "possession": 1,
            "team_id": 10,
            "player_id": 100,
            "shot_created": 1,
            "created_shot_cxg": 0.18,
            "action_type": "Pass",
            "play_pattern": "Open Play",
            "body_part": "Right Foot",
            "pass_height": "Ground Pass",
            "start_zone": "middle_left",
            "end_zone": "final_central",
            "start_third": "middle",
            "end_third": "final",
            "score_state": "drawing",
            "under_pressure": False,
            "is_pass": True,
        }
    )
    data_path = tmp_path / "cxa_actions.csv"
    pd.DataFrame([row]).to_csv(data_path, index=False)

    report = validate(CONTRACT_PATH, data_path)

    assert report["valid"] is True
    assert report["missing_required"] == []
    assert report["missing_split_group"] is False


def test_cxa_contract_validator_rejects_leakage_columns(tmp_path: Path):
    contract = _load_contract()
    row = {}
    for key in (
        "identity_columns",
        "target_columns",
        "required_numeric_features",
        "required_binary_features",
        "required_categorical_features",
    ):
        for column in contract[key]:
            row[column] = 0
    row["match_id"] = 1
    row["created_shot_outcome"] = "Goal"
    data_path = tmp_path / "cxa_actions_with_leakage.csv"
    pd.DataFrame([row]).to_csv(data_path, index=False)

    report = validate(CONTRACT_PATH, data_path)

    assert report["valid"] is False
    assert report["present_forbidden"] == ["created_shot_outcome"]
