import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cxg_diagnostic_training import (
    DEFAULT_CONTRACT_PATH,
    _comparison_table,
    _select_candidate,
    load_raw_and_modeling_features,
    load_contract,
    resolve_features,
    run_diagnostic_training,
    train_diagnostic_candidates,
    validate_no_forbidden_features,
)
from scripts.run_cxg_end_to_end import run_end_to_end


def _synthetic_diagnostic_frame(row_count: int = 48) -> pd.DataFrame:
    rows = []
    for i in range(row_count):
        close_shot = i % 6 in {0, 1}
        goal = int(close_shot and i % 4 == 0)
        rows.append(
            {
                "shot_id": i,
                "event_id": f"event-{i}",
                "match_id": i // 6,
                "team_id": 100 + (i % 3),
                "player_id": 200 + (i % 7),
                "is_goal": goal,
                "shot_distance": 7.0 + (i % 18),
                "shot_angle": 0.55 if close_shot else 0.18 + ((i % 5) * 0.03),
                "centrality": 0.8 if close_shot else 0.25,
                "location_x": 102.0 - (i % 12),
                "location_y": 40.0 + ((i % 9) - 4),
                "score_diff_at_shot": (i % 3) - 1,
                "minute": 4 + i,
                "time_gap_seconds": float(i % 10),
                "possession_sequence_length": 3 + (i % 6),
                "possession_duration": 6.0 + (i % 11),
                "previous_action_gap": float(i % 5),
                "recent_def_actions_count": i % 4,
                "pressure_proxy_score": 0.15 * (i % 5),
                "opponent_def_rating_global": 0.9 + ((i % 4) * 0.05),
                "first_time": i % 5 == 0,
                "under_pressure": i % 3 == 0,
                "is_leading": i % 3 == 1,
                "is_trailing": i % 3 == 2,
                "is_drawing": i % 3 == 0,
                "body_part": "Head" if i % 8 == 0 else "Right Foot",
                "technique": "Volley" if i % 9 == 0 else "Normal",
                "shot_type": "Free Kick" if i % 10 == 0 else "Open Play",
                "play_pattern": "From Corner" if i % 7 == 0 else "Regular Play",
                "set_piece_category": "corner" if i % 7 == 0 else "open_play",
                "set_piece_phase": "delivery" if i % 7 == 0 else "none",
                "minute_bucket_label": "46-60" if i >= row_count // 2 else "0-15",
                "score_state": "leading" if i % 3 == 1 else "level",
                "chain_label": "fast" if i % 2 else "settled",
                "assist_category": "cutback" if i % 6 == 0 else "none",
                "pressure_state": "under_pressure" if i % 3 == 0 else "no_pressure",
                "def_label": "compact" if i % 4 == 0 else "average",
                "statsbomb_xg": 0.04 + (goal * 0.3),
                "outcome": "Goal" if goal else "Saved",
                "is_blocked": i % 11 == 0,
                "cxg_raw": 0.1,
                "model_registry": "do-not-use",
                "shot_predictions": 0.2,
            }
        )
    return pd.DataFrame(rows)


def test_feature_contract_loads_with_expected_shape():
    contract = load_contract(DEFAULT_CONTRACT_PATH)

    assert contract["version"] == "cxg_diagnostic_v1"
    assert contract["target_column"] == "is_goal"
    assert contract["group_column"] == "match_id"
    assert "shot_distance" in contract["eligible_numeric_features"]
    assert "statsbomb_xg" in contract["reference_only_columns"]
    assert "is_blocked" in contract["excluded_leakage_columns"]
    assert {candidate["name"] for candidate in contract["model_candidates"]} == {
        "geometry_logistic",
        "diagnostic_logistic",
        "diagnostic_baseline_parity_logistic",
        "calibrated_diagnostic_logistic_sigmoid",
        "gradient_boosting",
        "calibrated_gradient_boosting_sigmoid",
        "extra_trees",
        "calibrated_extra_trees_sigmoid",
    }


def test_unavailable_optional_features_are_ignored_safely():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    frame = _synthetic_diagnostic_frame()[
        ["shot_id", "match_id", "is_goal", "shot_distance", "shot_angle", "body_part"]
    ]

    resolved = resolve_features(frame, contract)

    assert resolved.numeric == ["shot_distance", "shot_angle"]
    assert resolved.binary == []
    assert resolved.categorical == ["body_part"]
    assert "centrality" in resolved.unavailable["numeric"]


def test_baseline_coordinate_aliases_are_source_available():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    frame = _synthetic_diagnostic_frame()[["match_id", "is_goal", "shot_distance", "shot_angle"]]
    frame["location_x"] = 100.0
    frame["location_y"] = 40.0

    resolved = resolve_features(
        frame,
        contract,
        original_input_columns={
            "match_id",
            "is_goal",
            "shot_distance",
            "shot_angle",
            "shot_x",
            "shot_y",
        },
    )

    assert "location_x" in resolved.source_available["numeric"]
    assert "location_y" in resolved.source_available["numeric"]
    assert "location_x" not in resolved.synthetic_default_features["numeric"]


def test_leakage_and_reference_columns_are_excluded_from_features():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    frame = _synthetic_diagnostic_frame()

    resolved = resolve_features(frame, contract)

    assert "statsbomb_xg" not in resolved.all_features
    assert "cxg_raw" not in resolved.all_features
    assert "is_blocked" not in resolved.all_features
    assert {"statsbomb_xg"}.issubset(resolved.reference_present)
    assert {"cxg_raw", "outcome", "is_blocked"}.issubset(resolved.excluded_present)
    with pytest.raises(ValueError, match="Forbidden leakage/reference columns"):
        validate_no_forbidden_features(["shot_distance", "statsbomb_xg"], contract)


def test_tiny_synthetic_training_run_writes_required_outputs(tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    output_dir = tmp_path / "diagnostic_v1"
    _synthetic_diagnostic_frame().to_parquet(input_path, index=False)

    outputs = run_diagnostic_training(
        input_path=input_path,
        output_dir=output_dir,
        min_category_count=2,
        random_state=7,
    )

    expected = {
        "feature_contract",
        "excluded_columns",
        "resolved_features",
        "feature_group_summary",
        "model_candidates",
        "model_comparison",
        "fold_metrics",
        "candidate_calibration_summary",
        "candidate_probability_summary",
        "selected_model_metadata",
        "selected_model",
        "cross_validated_predictions",
        "training_report",
        "training_summary",
    }
    assert set(outputs) == expected
    for path in outputs.values():
        assert path.exists()

    comparison = pd.read_csv(outputs["model_comparison"])
    folds = pd.read_csv(outputs["fold_metrics"])
    predictions = pd.read_parquet(outputs["cross_validated_predictions"])
    metadata = json.loads(outputs["selected_model_metadata"].read_text(encoding="utf-8"))
    calibration = pd.read_csv(outputs["candidate_calibration_summary"])

    assert outputs["feature_contract"] == output_dir / "contracts" / "feature_contract.json"
    assert outputs["excluded_columns"] == output_dir / "diagnostics" / "excluded_columns.csv"
    assert outputs["resolved_features"] == output_dir / "diagnostics" / "resolved_features.json"
    assert outputs["feature_group_summary"] == (
        output_dir / "diagnostics" / "feature_group_summary.csv"
    )
    assert outputs["selected_model"] == output_dir / "models" / "selected_model.joblib"
    assert outputs["model_candidates"] == output_dir / "models" / "model_candidates.json"
    assert outputs["cross_validated_predictions"] == (
        output_dir / "predictions" / "cross_validated_predictions.parquet"
    )
    assert outputs["training_report"] == output_dir / "reports" / "training_report.md"
    assert outputs["model_comparison"] == output_dir / "reports" / "model_comparison.csv"
    assert outputs["fold_metrics"] == output_dir / "reports" / "fold_metrics.csv"
    assert outputs["candidate_calibration_summary"] == (
        output_dir / "reports" / "candidate_calibration_summary.csv"
    )
    assert outputs["candidate_probability_summary"] == (
        output_dir / "reports" / "candidate_probability_summary.csv"
    )
    assert outputs["training_summary"] == output_dir / "reports" / "training_summary.json"
    assert set(comparison["model_candidate"]) == {
        "geometry_logistic",
        "diagnostic_logistic",
        "diagnostic_baseline_parity_logistic",
        "calibrated_diagnostic_logistic_sigmoid",
        "gradient_boosting",
        "calibrated_gradient_boosting_sigmoid",
        "extra_trees",
        "calibrated_extra_trees_sigmoid",
    }
    assert comparison["selected_model"].sum() == 1
    assert {
        "calibration_proxy_error",
        "absolute_calibration_proxy_error",
        "selection_rank",
        "selected_model",
    }.issubset(comparison.columns)
    assert {
        "model_candidate",
        "fold",
        "rows",
        "goals",
        "goal_rate",
        "mean_predicted_probability",
        "calibration_error",
        "absolute_calibration_error",
    }.issubset(calibration.columns)
    assert set(predictions["prediction_source"]) == {"cross_validated"}
    assert {"shot_id", "event_id", "match_id", "team_id", "player_id", "is_goal"}.issubset(
        predictions.columns
    )
    assert folds["excluded_leakage_reference_column_count"].min() >= 1
    assert metadata["selected_model"] in set(comparison["model_candidate"])
    assert metadata["selection_metric"] == "log_loss_mean"
    assert metadata["selection_reason"] == metadata["selected_reason"]
    assert metadata["selected_features"]
    assert metadata["selected_feature_count"] == len(metadata["selected_features"])
    assert metadata["candidate_feature_counts"]
    assert "source_available_features" in metadata
    assert "synthetic_default_features" in metadata
    assert metadata["forbidden_features_used"] == []
    assert "statsbomb_xg" not in metadata["resolved_features"]["numeric"]
    assert "is_blocked" not in metadata["resolved_features"]["binary"]
    resolved_payload = json.loads(outputs["resolved_features"].read_text(encoding="utf-8"))
    assert {
        "source_available",
        "model_available",
        "synthetic_default_features",
        "unavailable",
    }.issubset(resolved_payload)


def test_single_class_folds_skip_roc_auc_safely():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    frame = _synthetic_diagnostic_frame(row_count=24)
    frame["is_goal"] = [1] * 6 + [0] * 18

    _, _, _, fold_metrics, _, _ = train_diagnostic_candidates(
        frame,
        contract,
        min_category_count=2,
        random_state=3,
    )

    assert "skipped_single_class_fold" in set(fold_metrics["roc_auc_status"])


def test_selected_model_uses_log_loss_as_primary_rule():
    fold_metrics = pd.DataFrame(
        [
            {
                "model_candidate": "low_brier",
                "fold": 1,
                "brier": 0.05,
                "log_loss": 0.30,
                "roc_auc": 0.9,
                "row_count": 10,
                "goal_count": 1,
                "goal_rate": 0.1,
                "mean_predicted_probability": 0.1,
                "calibration_proxy_error": 0.0,
                "absolute_calibration_proxy_error": 0.0,
                "feature_count": 3,
                "excluded_leakage_reference_column_count": 0,
            },
            {
                "model_candidate": "low_log_loss",
                "fold": 1,
                "brier": 0.06,
                "log_loss": 0.20,
                "roc_auc": 0.7,
                "row_count": 10,
                "goal_count": 1,
                "goal_rate": 0.1,
                "mean_predicted_probability": 0.12,
                "calibration_proxy_error": 0.02,
                "absolute_calibration_proxy_error": 0.02,
                "feature_count": 3,
                "excluded_leakage_reference_column_count": 0,
            },
        ]
    )

    comparison = _comparison_table(fold_metrics)

    assert _select_candidate(comparison) == "low_log_loss"


def test_selected_model_uses_brier_as_first_tie_breaker():
    fold_metrics = pd.DataFrame(
        [
            {
                "model_candidate": "worse_brier",
                "fold": 1,
                "brier": 0.07,
                "log_loss": 0.20,
                "roc_auc": 0.9,
                "row_count": 10,
                "goal_count": 1,
                "goal_rate": 0.1,
                "mean_predicted_probability": 0.1,
                "calibration_proxy_error": 0.0,
                "absolute_calibration_proxy_error": 0.0,
                "feature_count": 3,
                "excluded_leakage_reference_column_count": 0,
            },
            {
                "model_candidate": "better_brier",
                "fold": 1,
                "brier": 0.05,
                "log_loss": 0.20,
                "roc_auc": 0.7,
                "row_count": 10,
                "goal_count": 1,
                "goal_rate": 0.1,
                "mean_predicted_probability": 0.13,
                "calibration_proxy_error": 0.03,
                "absolute_calibration_proxy_error": 0.03,
                "feature_count": 3,
                "excluded_leakage_reference_column_count": 0,
            },
        ]
    )

    comparison = _comparison_table(fold_metrics)

    assert _select_candidate(comparison) == "better_brier"


def test_synthetic_default_columns_are_recorded_and_excluded(tmp_path: Path):
    input_path = tmp_path / "minimal_shot_features.parquet"
    output_dir = tmp_path / "diagnostic_v1"
    source = _synthetic_diagnostic_frame().drop(
        columns=[
            "assist_category",
            "chain_label",
            "def_label",
            "opponent_def_rating_global",
            "pressure_state",
            "set_piece_category",
            "set_piece_phase",
        ]
    )
    source.to_parquet(input_path, index=False)

    frame, _, original_columns = load_raw_and_modeling_features(input_path)
    resolved = resolve_features(
        frame,
        load_contract(DEFAULT_CONTRACT_PATH),
        original_input_columns=original_columns,
    )

    assert "opponent_def_rating_global" not in resolved.source_available["numeric"]
    assert "opponent_def_rating_global" in resolved.synthetic_default_features["numeric"]
    assert "opponent_def_rating_global" in resolved.synthetic_default_excluded["numeric"]
    assert "def_label" not in resolved.source_available["categorical"]
    assert "def_label" in resolved.synthetic_default_features["categorical"]
    assert "def_label" in resolved.synthetic_default_excluded["categorical"]
    assert "def_label" not in resolved.categorical

    outputs = run_diagnostic_training(
        input_path=input_path,
        output_dir=output_dir,
        min_category_count=2,
        random_state=11,
    )
    resolved_payload = json.loads(outputs["resolved_features"].read_text(encoding="utf-8"))
    summary = pd.read_csv(outputs["feature_group_summary"])
    metadata = json.loads(outputs["selected_model_metadata"].read_text(encoding="utf-8"))
    candidate_features = {
        feature
        for candidate in metadata["model_candidates"]
        for group in candidate["features"].values()
        for feature in group
    }

    assert "def_label" in resolved_payload["synthetic_default_features"]["categorical"]
    assert "def_label" in resolved_payload["synthetic_default_excluded"]["categorical"]
    assert "def_label" not in candidate_features
    assert "opponent_def_rating_global" not in candidate_features
    def_label_row = summary.loc[summary["feature"] == "def_label"].iloc[0]
    assert def_label_row["availability_source"] == "synthetic_default"
    assert def_label_row["used_in_training"] is False or not def_label_row["used_in_training"]
    assert def_label_row["excluded_reason"] == "constant_synthetic_default"


def test_existing_cxg_baseline_import_still_works(tmp_path: Path):
    input_path = tmp_path / "baseline_shot_features.parquet"
    _synthetic_diagnostic_frame().drop(
        columns=[
            "event_id",
            "outcome",
            "is_blocked",
            "cxg_raw",
            "model_registry",
            "shot_predictions",
        ]
    ).to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "baseline")

    assert outputs.model_path.exists()
    assert outputs.scored_predictions_path.exists()
