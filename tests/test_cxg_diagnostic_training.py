import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cxg_diagnostic_training import (
    DEFAULT_CONTRACT_PATH,
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
        "gradient_boosting",
        "extra_trees",
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
        "model_candidates",
        "model_comparison",
        "fold_metrics",
        "selected_model_metadata",
        "selected_model",
        "cross_validated_predictions",
        "training_report",
    }
    assert set(outputs) == expected
    for path in outputs.values():
        assert path.exists()

    comparison = pd.read_csv(outputs["model_comparison"])
    folds = pd.read_csv(outputs["fold_metrics"])
    predictions = pd.read_parquet(outputs["cross_validated_predictions"])
    metadata = json.loads(outputs["selected_model_metadata"].read_text(encoding="utf-8"))

    assert set(comparison["model_candidate"]) == {
        "geometry_logistic",
        "diagnostic_logistic",
        "gradient_boosting",
        "extra_trees",
    }
    assert set(predictions["prediction_source"]) == {"cross_validated"}
    assert {"shot_id", "event_id", "match_id", "team_id", "player_id", "is_goal"}.issubset(
        predictions.columns
    )
    assert folds["excluded_leakage_reference_column_count"].min() >= 1
    assert metadata["selected_model"] in set(comparison["model_candidate"])
    assert "statsbomb_xg" not in metadata["resolved_features"]["numeric"]
    assert "is_blocked" not in metadata["resolved_features"]["binary"]


def test_single_class_folds_skip_roc_auc_safely():
    contract = load_contract(DEFAULT_CONTRACT_PATH)
    frame = _synthetic_diagnostic_frame(row_count=24)
    frame["is_goal"] = [1] * 6 + [0] * 18

    _, _, _, fold_metrics, _ = train_diagnostic_candidates(
        frame,
        contract,
        min_category_count=2,
        random_state=3,
    )

    assert "skipped_single_class_fold" in set(fold_metrics["roc_auc_status"])


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
