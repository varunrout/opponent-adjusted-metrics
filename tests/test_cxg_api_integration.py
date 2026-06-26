import json
from pathlib import Path

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


def _prediction_payload(**overrides):
    payload = {
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
    }
    payload.update(overrides)
    return payload


def _fixture_model(tmp_path: Path, model_version: str = "api-v1"):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)
    return run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version=model_version
    )


def test_predict_cxg_returns_501_when_artifact_missing(monkeypatch):
    monkeypatch.setattr(cxg_inference, "_candidate_model_paths", lambda registry_path=None: [])

    response = TestClient(app).post("/predict/cxg", json=_prediction_payload())

    assert response.status_code == 501
    assert "requires a trained model artefact" in response.json()["detail"]


def test_predict_cxg_returns_controlled_error_when_metadata_missing(tmp_path: Path, monkeypatch):
    outputs = _fixture_model(tmp_path)
    outputs.metadata_path.unlink()
    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )

    response = TestClient(app).post("/predict/cxg", json=_prediction_payload())

    assert response.status_code == 501
    assert "metadata" in response.json()["detail"].lower()
    assert "missing" in response.json()["detail"].lower()


def test_predict_cxg_returns_controlled_error_when_metadata_incomplete(tmp_path: Path, monkeypatch):
    outputs = _fixture_model(tmp_path)
    outputs.metadata_path.write_text(json.dumps({"model_version": "broken"}), encoding="utf-8")
    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )

    response = TestClient(app).post("/predict/cxg", json=_prediction_payload())

    assert response.status_code == 501
    assert "metadata" in response.json()["detail"].lower()
    assert "target" in response.json()["detail"].lower()


def test_generated_cxg_artifact_metadata_is_api_compatible(tmp_path: Path, monkeypatch):
    outputs = _fixture_model(tmp_path, model_version="api-compatible-v1")
    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )

    response = TestClient(app).post(
        "/predict/cxg",
        json=_prediction_payload(possession_duration=8.5, possession_length=5),
    )

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["raw_probability"] <= 1.0
    assert 0.0 <= body["neutral_probability"] <= 1.0
    assert body["model_version"] == "api-compatible-v1"
    assert body["model_path"] == str(outputs.model_path)
    assert body["model_metadata"]["model_version"] == "api-compatible-v1"
    assert body["model_metadata"]["target"] == "is_goal"
    assert body["model_metadata"]["feature_count"] > 0
    assert "cxg_raw" in body["model_metadata"]["prediction_columns"]


def test_predict_cxg_optional_context_defaults_are_deterministic(tmp_path: Path, monkeypatch):
    outputs = _fixture_model(tmp_path, model_version="defaults-v1")
    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )
    client = TestClient(app)

    first = client.post("/predict/cxg", json=_prediction_payload())
    second = client.post(
        "/predict/cxg",
        json=_prediction_payload(possession_duration=None, possession_length=None),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["raw_probability"] == second.json()["raw_probability"]
    assert first.json()["neutral_probability"] == second.json()["neutral_probability"]


def test_predict_cxg_rejects_invalid_numeric_request():
    response = TestClient(app).post(
        "/predict/cxg",
        json=_prediction_payload(location_x=130.0, opponent_team_id=0),
    )

    assert response.status_code == 422
