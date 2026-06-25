import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from opponent_adjusted.api import cxg_inference
from opponent_adjusted.api.service import app
from scripts.run_cxg_end_to_end import run_end_to_end


def _synthetic_cxg_frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        is_goal = int(i % 5 == 0 or (i % 7 == 0 and i % 2 == 0))
        rows.append(
            {
                "shot_id": i,
                "match_id": i // 10,
                "team_id": 10 + (i % 3),
                "team_name": f"Team {i % 3}",
                "player_id": 100 + (i % 6),
                "player_name": f"Player {i % 6}",
                "opponent_team_id": 20 + (i % 4),
                "is_goal": is_goal,
                "shot_distance": 7.0 + (i % 20),
                "shot_angle": 0.15 + ((i % 8) * 0.06),
                "statsbomb_xg": 0.05 + (is_goal * 0.25) + ((i % 4) * 0.01),
                "score_diff_at_shot": (i % 3) - 1,
                "minute": 5 + i,
                "time_gap_seconds": float(i % 12),
                "is_leading": (i % 3) == 2,
                "is_trailing": (i % 3) == 0,
                "is_drawing": (i % 3) == 1,
                "possession_match": float(i % 2),
                "chain_label": "fast" if i % 2 else "slow",
                "pass_style": "cutback" if i % 4 == 0 else "none",
                "score_state": "leading" if (i % 3) == 2 else "drawing",
                "simple_state": "leading" if (i % 3) == 2 else "drawing",
                "minute_bucket_label": "46-60" if i >= 45 else "31-45",
                "assist_category": "pass" if i % 4 == 0 else "none",
                "pressure_state": "under_pressure" if i % 3 == 0 else "no_pressure",
                "set_piece_category": "open_play",
                "set_piece_phase": "none",
                "def_label": "average",
            }
        )
    return pd.DataFrame(rows)


def test_cxg_end_to_end_runner_emits_artifact_metadata_and_outputs(tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)

    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version="test-v1"
    )

    assert outputs.model_path.exists()
    assert outputs.metadata_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.scored_predictions_path.exists()
    assert outputs.player_aggregates_path.exists()
    assert outputs.team_aggregates_path.exists()
    assert outputs.model_card_path.exists()
    assert hasattr(joblib.load(outputs.model_path), "predict_proba")

    scored = pd.read_parquet(outputs.scored_predictions_path)
    assert {"cxg_raw", "cxg_neutral", "cxg_opp_adjusted_diff"}.issubset(scored.columns)
    assert not pd.read_parquet(outputs.player_aggregates_path).empty
    assert not pd.read_parquet(outputs.team_aggregates_path).empty


def test_cxg_metadata_schema_contains_api_loader_fields(tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)

    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version="test-v2"
    )
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))

    assert metadata["artifact_path"] == str(outputs.model_path)
    assert metadata["model_version"] == "test-v2"
    assert metadata["version"] == "test-v2"
    assert metadata["created_at"]
    assert metadata["features"]["numeric"]
    assert set(metadata["features"]).issuperset({"numeric", "binary", "categorical"})


def test_cxg_api_positive_path_with_fixture_model(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)
    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version="api-v1"
    )

    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )

    client = TestClient(app)
    response = client.post(
        "/predict/cxg",
        json={
            "location_x": 102.0,
            "location_y": 40.0,
            "body_part": "Right Foot",
            "technique": "Normal",
            "shot_type": "Open Play",
            "first_time": False,
            "minute": 55,
            "score_diff": 0,
            "under_pressure": False,
            "opponent_team_id": 1,
            "possession_duration": 8.5,
            "possession_length": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["raw_probability"] <= 1.0
    assert 0.0 <= body["neutral_probability"] <= 1.0
    assert body["model_version"] == "api-v1"
    assert body["model_path"] == str(outputs.model_path)
