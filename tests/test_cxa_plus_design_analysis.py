import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.analyze_cxa_plus_design import (
    TARGET_COLUMNS,
    add_downstream_window_targets,
    analyze_cxa_plus_design,
    audit_required_fields,
    leakage_risk_register,
    resolve_order_columns,
)


def _action_features(include_optional: bool = True) -> pd.DataFrame:
    rows = [
        {
            "action_id": "a1",
            "event_id": "e1",
            "match_id": 1,
            "possession": 10,
            "sequence_id": "s1",
            "team_id": 100,
            "player_id": 1000,
            "action_position": 1,
            "minute": 1,
            "second": 5,
            "action_type": "Pass",
            "shot_created": 0,
            "created_shot_id": None,
            "created_shot_cxg": 0.0,
            "start_x": 35.0,
            "start_y": 40.0,
            "end_x": 55.0,
            "end_y": 41.0,
        },
        {
            "action_id": "a2",
            "event_id": "e2",
            "match_id": 1,
            "possession": 10,
            "sequence_id": "s1",
            "team_id": 100,
            "player_id": 1001,
            "action_position": 2,
            "minute": 1,
            "second": 7,
            "action_type": "Carry",
            "shot_created": 0,
            "created_shot_id": None,
            "created_shot_cxg": 0.0,
            "start_x": 55.0,
            "start_y": 41.0,
            "end_x": 70.0,
            "end_y": 42.0,
        },
        {
            "action_id": "a3",
            "event_id": "e3",
            "match_id": 1,
            "possession": 10,
            "sequence_id": "s1",
            "team_id": 100,
            "player_id": 1002,
            "action_position": 3,
            "minute": 1,
            "second": 9,
            "action_type": "Shot Assist",
            "shot_created": 1,
            "created_shot_id": "shot-1",
            "created_shot_cxg": 0.28,
            "start_x": 70.0,
            "start_y": 42.0,
            "end_x": 88.0,
            "end_y": 40.0,
        },
        {
            "action_id": "b1",
            "event_id": "e4",
            "match_id": 1,
            "possession": 11,
            "sequence_id": "s2",
            "team_id": 200,
            "player_id": 2000,
            "action_position": 1,
            "minute": 2,
            "second": 1,
            "action_type": "Pass",
            "shot_created": 0,
            "created_shot_id": None,
            "created_shot_cxg": 0.0,
            "start_x": 20.0,
            "start_y": 30.0,
            "end_x": 25.0,
            "end_y": 35.0,
        },
    ]
    frame = pd.DataFrame(rows)
    if include_optional:
        frame["diagnostic_cxa"] = [0.03, 0.08, 0.45, 0.02]
        frame["period"] = 1
        frame["start_zone"] = ["middle", "final", "box", "defensive"]
        frame["end_zone"] = ["middle", "final", "box", "middle"]
    return frame


def _write_inputs(root: Path, *, include_optional: bool = True) -> tuple[Path, Path]:
    feature_path = root / "feature_store" / "cxa" / "action_features.parquet"
    results_path = (
        root / "outputs" / "results" / "cxa" / "diagnostic_v1" / "action_predictions.parquet"
    )
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    features = _action_features(include_optional=include_optional)
    features.drop(columns=["diagnostic_cxa"], errors="ignore").to_parquet(feature_path, index=False)
    pd.DataFrame(
        {
            "action_id": features["action_id"],
            "diagnostic_cxa": [0.03, 0.08, 0.45, 0.02],
            "predicted_shot_created_probability": [0.03, 0.08, 0.45, 0.02],
        }
    ).to_parquet(results_path, index=False)
    return feature_path, results_path


def test_script_writes_all_required_outputs(tmp_path: Path):
    feature_path, results_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "analysis" / "cxa_plus" / "design"

    outputs = analyze_cxa_plus_design(
        feature_path=feature_path,
        diagnostic_results_path=results_path,
        output_dir=output_dir,
    )

    assert set(outputs) == {
        "cxa_plus_design_report",
        "cxa_plus_design_summary",
        "possession_window_coverage",
        "downstream_shot_window_rates",
        "candidate_targets",
        "leakage_risk_register",
        "required_fields_audit",
        "sequence_window_examples",
    }
    for path in outputs.values():
        assert path.exists()

    summary = json.loads(outputs["cxa_plus_design_summary"].read_text(encoding="utf-8"))
    assert summary["recommended_first_target"] == "shot_within_next_5_actions"


def test_candidate_targets_include_expected_target_names(tmp_path: Path):
    feature_path, results_path = _write_inputs(tmp_path)
    outputs = analyze_cxa_plus_design(
        feature_path=feature_path,
        diagnostic_results_path=results_path,
        output_dir=tmp_path / "analysis",
    )

    candidates = pd.read_csv(outputs["candidate_targets"])

    assert set(TARGET_COLUMNS).issubset(set(candidates["target_name"]))
    recommended = candidates[candidates["recommended_for_first_model"]]
    assert recommended.iloc[0]["target_name"] == "shot_within_next_5_actions"


def test_downstream_window_targets_use_future_actions_only():
    frame = add_downstream_window_targets(_action_features())
    first = frame.loc[frame["action_id"] == "a1"].iloc[0]
    shot = frame.loc[frame["action_id"] == "a3"].iloc[0]

    assert first["shot_within_next_1_action"] == 0
    assert first["shot_within_next_3_actions"] == 1
    assert first["shot_within_next_5_actions"] == 1
    assert first["sum_created_shot_cxg_rest_of_possession"] == pytest.approx(0.28)
    assert shot["shot_later_in_possession"] == 0


def test_leakage_risk_register_flags_future_and_outcome_fields():
    frame = add_downstream_window_targets(_action_features())
    leakage = leakage_risk_register(frame)
    risks = dict(zip(leakage["field"], leakage["model_feature_allowed"], strict=False))

    assert risks["created_shot_id"] is False
    assert risks["created_shot_cxg"] is False
    assert risks["discounted_downstream_shot_value"] is False
    assert risks["shot_within_next_5_actions"] is False


def test_report_distinguishes_diagnostic_cxa_cxa_plus_and_advanced_cxa(tmp_path: Path):
    feature_path, results_path = _write_inputs(tmp_path)
    outputs = analyze_cxa_plus_design(
        feature_path=feature_path,
        diagnostic_results_path=results_path,
        output_dir=tmp_path / "analysis",
    )

    report = outputs["cxa_plus_design_report"].read_text(encoding="utf-8")

    assert "Diagnostic CxA predicts `shot_created`" in report
    assert "CxA+ should predict downstream chance creation" in report
    assert "Advanced CxA should later estimate state-value delta" in report


def test_missing_optional_fields_produce_warnings_not_crashes(tmp_path: Path):
    feature_path, results_path = _write_inputs(tmp_path, include_optional=False)
    outputs = analyze_cxa_plus_design(
        feature_path=feature_path,
        diagnostic_results_path=results_path,
        output_dir=tmp_path / "analysis",
    )

    fields = pd.read_csv(outputs["required_fields_audit"])

    assert outputs["cxa_plus_design_report"].exists()
    assert fields.loc[fields["field"] == "period", "status"].iloc[0] == "missing"


def test_invalid_ordering_fields_fail_clearly():
    frame = _action_features().drop(columns=["action_position", "minute", "second", "period"])

    with pytest.raises(ValueError, match="within-possession ordering field"):
        resolve_order_columns(frame)


def test_no_model_training_is_imported_or_called():
    source = Path("scripts/analyze_cxa_plus_design.py").read_text(encoding="utf-8")

    assert "run_cxa_diagnostic_training" not in source
    assert "fit(" not in source
    assert "joblib.dump" not in source


def test_existing_cxa_result_files_are_not_modified(tmp_path: Path):
    feature_path, results_path = _write_inputs(tmp_path)
    original = pd.read_parquet(results_path)

    analyze_cxa_plus_design(
        feature_path=feature_path,
        diagnostic_results_path=results_path,
        output_dir=tmp_path / "analysis",
    )

    after = pd.read_parquet(results_path)
    pd.testing.assert_frame_equal(original, after)


def test_required_field_audit_records_reference_fields_as_not_features():
    audit = audit_required_fields(_action_features())
    notes = dict(zip(audit["field"], audit["notes"], strict=False))

    assert "not a model feature" in notes["created_shot_cxg"]
    assert "not a model feature" in notes["diagnostic_cxa"]
