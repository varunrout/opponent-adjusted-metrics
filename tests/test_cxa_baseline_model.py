import json
from pathlib import Path

import joblib
import pandas as pd

from scripts.check_cxg_outputs import assert_git_ignored
from scripts.run_cxa_end_to_end import run_end_to_end
from scripts.run_cxa_pipeline import build_action_features, save_action_features


def _synthetic_events(
    one_class: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    pass_details = []
    shots = []
    event_id = 1
    for match_id in range(1, 5):
        for sequence in range(3):
            team_id = 10 + (sequence % 2)
            possession = (match_id * 10) + sequence
            for pos in range(4):
                action_type = "Pass" if pos % 2 == 0 else "Carry"
                rows.append(
                    {
                        "event_id": event_id,
                        "raw_event_id": event_id,
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_id": 100 + pos,
                        "action_type": action_type,
                        "period": 1,
                        "minute": sequence * 5,
                        "second": pos * 3,
                        "possession": possession,
                        "start_x": 42.0 + (pos * 10),
                        "start_y": 25.0 + (pos * 4),
                        "under_pressure": pos == 1,
                        "event_outcome": None,
                    }
                )
                if action_type == "Pass":
                    pass_details.append(
                        {
                            "event_id": event_id,
                            "end_x": 52.0 + (pos * 10),
                            "end_y": 28.0 + (pos * 4),
                            "length": 11.0,
                            "angle": 0.2,
                            "pass_height": "Ground Pass",
                            "pass_type": None,
                            "body_part": "Right Foot",
                            "is_cross": pos == 2,
                            "is_through_ball": pos == 0,
                        }
                    )
                event_id += 1

            if not one_class and sequence < 2:
                rows.append(
                    {
                        "event_id": event_id,
                        "raw_event_id": event_id,
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_id": 900,
                        "action_type": "Shot",
                        "period": 1,
                        "minute": sequence * 5,
                        "second": 13,
                        "possession": possession,
                        "start_x": 104.0,
                        "start_y": 40.0,
                        "under_pressure": False,
                        "event_outcome": "Goal" if sequence == 0 else "Saved",
                    }
                )
                shots.append(
                    {
                        "shot_id": 5000 + event_id,
                        "event_id": event_id,
                        "statsbomb_xg": 0.25 + (sequence * 0.03),
                    }
                )
                event_id += 1

    return (
        pd.DataFrame(rows),
        pd.DataFrame(shots),
        {"Pass": pd.DataFrame(pass_details), "Carry": pd.DataFrame()},
    )


def test_cxa_feature_table_is_generated_under_contract_path(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    output_path = save_action_features(features, tmp_path / "feature_store" / "cxa")

    assert output_path == tmp_path / "feature_store" / "cxa" / "action_features.parquet"
    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert not written.empty
    assert {"action_id", "shot_created", "created_shot_cxg"}.issubset(written.columns)
    assert written["shot_created"].isin([0, 1]).all()
    assert written["created_shot_cxg"].between(0, 1).all()


def test_cxa_end_to_end_emits_model_predictions_and_aggregates(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxa", model_version="cxa-test-v1"
    )

    assert outputs.model_path.exists()
    assert outputs.metadata_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.predictions_path.exists()
    assert outputs.player_aggregates_path.exists()
    assert outputs.team_aggregates_path.exists()
    assert hasattr(joblib.load(outputs.model_path), "predict_proba")

    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(outputs.predictions_path)
    assert metadata["model_version"] == "cxa-test-v1"
    assert metadata["target"] == "shot_created"
    assert metadata["value_column"] == "created_shot_cxg"
    assert metadata["leakage_guardrails"]["forbidden_training_features_excluded"] is True
    assert metrics["row_count"] == len(features)
    assert metrics["log_loss_status"] == "computed"
    assert {"predicted_cxa", "baseline_cxa", "cxa_above_baseline"}.issubset(predictions.columns)
    assert not pd.read_parquet(outputs.player_aggregates_path).empty
    assert not pd.read_parquet(outputs.team_aggregates_path).empty


def test_cxa_baseline_excludes_forbidden_leakage_features(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    features["created_shot_outcome"] = "Goal"
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "cxa")
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    model_features = set().union(*[set(cols) for cols in metadata["features"].values()])

    assert "created_shot_outcome" not in model_features
    assert "post_shot_xg" not in model_features


def test_cxa_baseline_handles_one_class_data_safely(tmp_path: Path):
    events, shots, details = _synthetic_events(one_class=True)
    features = build_action_features(events, shots.iloc[0:0], details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "cxa")
    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(outputs.predictions_path)

    assert metrics["positive_count"] == 0
    assert metrics["log_loss_status"] == "skipped_single_class"
    assert metrics["roc_auc_status"] == "skipped_single_class"
    assert predictions["predicted_cxa"].eq(0.0).all()


def test_cxa_generated_paths_are_git_ignored():
    assert_git_ignored(
        (
            Path("feature_store/cxa/action_features.parquet"),
            Path("outputs/modeling/cxa/models/baseline_model.joblib"),
            Path("outputs/modeling/cxa/models/baseline_model.json"),
            Path("outputs/modeling/cxa/reports/metrics.json"),
            Path("outputs/modeling/cxa/predictions/action_predictions.parquet"),
            Path("outputs/modeling/cxa/aggregates/player_cxa.parquet"),
            Path("outputs/modeling/cxa/aggregates/team_cxa.parquet"),
        ),
        Path.cwd(),
    )
