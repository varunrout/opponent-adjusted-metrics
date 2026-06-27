import json
from pathlib import Path

import pandas as pd

from scripts.check_cxg_outputs import assert_git_ignored
from scripts.validate_feature_contract import validate


CONTRACT_PATH = Path("configs/feature_contracts/cxt_v1.json")
DESIGN_DOC_PATH = Path("docs/modeling/cxt/design_contract.md")


def _load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_cxt_contract_declares_design_stage_and_variants():
    contract = _load_contract()

    assert contract["metric"] == "cxt"
    assert contract["status"] == "design_contract"
    assert contract["definition"]["baseline_formula"] == (
        "threat_value(end_location) - threat_value(start_location)"
    )
    assert contract["definition"]["event_data_only"] is True
    assert contract["definition"]["tracking_data_required"] is False

    assert set(contract["variants"]) >= {
        "baseline_cxt",
        "cxt_plus",
        "contextual_cxt",
        "advanced_cxt",
        "od_cxt",
        "od_cxt_plus",
    }
    assert all(
        variant["implemented_in_this_pr"] is False for variant in contract["variants"].values()
    )


def test_cxt_contract_defines_eligibility_identifiers_and_locations():
    contract = _load_contract()

    for action_type in ("pass", "carry", "dribble", "cross"):
        assert action_type in contract["eligibility"]["eligible_action_types"]

    for column in ("match_id", "team_id", "player_id", "possession_id", "action_id"):
        assert column in contract["identity_columns"]

    assert contract["required_location_fields"] == [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
    ]
    assert set(contract["baseline_value_fields"]) == {
        "start_zone",
        "end_zone",
        "start_threat",
        "end_threat",
        "cxt_value",
    }
    assert set(contract["future_enhancement_fields"]) >= {
        "cxt_plus",
        "state_value_before",
        "state_value_after",
        "advanced_cxt",
        "od_cxt",
        "od_cxt_plus",
    }


def test_cxt_contract_defines_leakage_guardrails_and_outputs():
    contract = _load_contract()

    prohibited = set(contract["prohibited_leakage_columns"])
    assert {
        "future_shot_xg",
        "future_shot_location",
        "future_goal",
        "future_shot_outcome",
        "next_action_is_shot",
        "actions_until_shot",
        "total_future_possession_length",
        "goal_outcome",
        "shot_outcome",
    } <= prohibited
    assert prohibited <= set(contract["forbidden_training_features"])
    assert "Future outcomes may estimate zone/state values" in (
        contract["validation_contract"]["leakage_rule"]
    )

    expected_paths = contract["output_contract"]["expected_future_paths"]
    assert expected_paths == {
        "features": "feature_store/cxt/action_features.parquet",
        "threat_grid": "outputs/modeling/cxt/threat_grid.parquet",
        "predictions": "outputs/modeling/cxt/predictions/action_threat.parquet",
        "player_aggregates": "outputs/modeling/cxt/aggregates/player_cxt.parquet",
        "team_aggregates": "outputs/modeling/cxt/aggregates/team_cxt.parquet",
        "sequence_aggregates": "outputs/modeling/cxt/aggregates/sequence_cxt.parquet",
        "metrics": "outputs/modeling/cxt/reports/metrics.json",
        "interpretation_summary": ("outputs/modeling/cxt/reports/interpretation_summary.json"),
    }


def test_cxt_design_document_contains_required_sections_and_scope():
    text = DESIGN_DOC_PATH.read_text(encoding="utf-8")

    for heading in (
        "## Definition",
        "## Baseline CxT",
        "## CxT+",
        "## Contextual CxT",
        "## Advanced CxT",
        "## OD-CxT",
        "## Eligible Actions",
        "## Feature Families",
        "## Leakage Guardrails",
        "## Validation Plan",
        "## Output Contract",
        "## Limitations",
    ):
        assert heading in text

    assert "CxG = shot quality" in text
    assert "CxA = chance creation actions" in text
    assert "CxT = territorial and threat progression" in text
    assert "does not implement the CxT model" in text
    assert "Baseline CxT is upcoming PR24" in text


def test_cxt_contract_validator_accepts_minimal_synthetic_action_table(
    tmp_path: Path,
):
    contract = _load_contract()
    row = {}
    for key in ("identity_columns", "required_location_fields", "baseline_value_fields"):
        for column in contract[key]:
            row[column] = 0

    row.update(
        {
            "action_id": "action-1",
            "event_id": "event-1",
            "match_id": 1,
            "possession_id": 100,
            "team_id": 10,
            "team_name": "Home",
            "player_id": 1000,
            "player_name": "Player One",
            "start_x": 40.0,
            "start_y": 35.0,
            "end_x": 65.0,
            "end_y": 38.0,
            "start_zone": "middle_left",
            "end_zone": "final_central",
            "start_threat": 0.03,
            "end_threat": 0.09,
            "cxt_value": 0.06,
        }
    )
    data_path = tmp_path / "cxt_actions.csv"
    pd.DataFrame([row]).to_csv(data_path, index=False)

    report = validate(CONTRACT_PATH, data_path)

    assert report["valid"] is True
    assert report["missing_required"] == []
    assert report["present_forbidden"] == []
    assert report["missing_split_group"] is False


def test_cxt_contract_validator_rejects_leakage_columns(tmp_path: Path):
    contract = _load_contract()
    row = {}
    for key in ("identity_columns", "required_location_fields", "baseline_value_fields"):
        for column in contract[key]:
            row[column] = 0
    row["match_id"] = 1
    row["future_shot_xg"] = 0.42

    data_path = tmp_path / "cxt_actions_with_leakage.csv"
    pd.DataFrame([row]).to_csv(data_path, index=False)

    report = validate(CONTRACT_PATH, data_path)

    assert report["valid"] is False
    assert report["present_forbidden"] == ["future_shot_xg"]


def test_cxt_declared_generated_paths_are_git_ignored():
    contract = _load_contract()
    expected_paths = contract["output_contract"]["expected_future_paths"]

    assert_git_ignored((Path(path) for path in expected_paths.values()), Path.cwd())
