from pathlib import Path

import joblib
import pytest

from opponent_adjusted.api import inference
from opponent_adjusted.api.inference import build_cxg_feature_frame, load_cxg_artifact
from opponent_adjusted.api.schemas import ShotPredictionRequest


class DummyModel:
    def predict_proba(self, features):
        return [[0.72, 0.28]]


@pytest.fixture(autouse=True)
def clear_artifact_cache():
    load_cxg_artifact.cache_clear()
    yield
    load_cxg_artifact.cache_clear()


def _sample_request() -> ShotPredictionRequest:
    return ShotPredictionRequest(
        location_x=105.0,
        location_y=40.0,
        body_part="Right Foot",
        technique="Normal",
        shot_type="Open Play",
        first_time=False,
        minute=55,
        score_diff=0,
        under_pressure=False,
        opponent_team_id=1,
        possession_duration=8.2,
        possession_length=6,
    )


def test_build_feature_frame_uses_metadata_order():
    metadata = {
        "features": {
            "numeric": ["distance_to_goal", "minute"],
            "binary": ["under_pressure"],
            "categorical": ["body_part"],
        }
    }

    frame = build_cxg_feature_frame(_sample_request(), metadata)

    assert list(frame.columns) == ["distance_to_goal", "minute", "under_pressure", "body_part"]
    assert frame.loc[0, "minute"] == 55
    assert frame.loc[0, "body_part"] == "Right Foot"


def test_load_cxg_artifact_from_env(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    joblib.dump(DummyModel(), model_path)
    monkeypatch.setenv("CXG_MODEL_ARTIFACT", str(model_path))

    model, metadata, loaded_path = inference.load_cxg_artifact()

    assert isinstance(model, DummyModel)
    assert metadata == {}
    assert loaded_path == Path(model_path)


def test_load_cxg_artifact_reads_metadata(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "model.json"
    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text('{"neutral_probability_reference": 0.2}', encoding="utf-8")
    monkeypatch.setenv("CXG_MODEL_ARTIFACT", str(model_path))
    monkeypatch.setenv("CXG_MODEL_METADATA", str(metadata_path))

    _, metadata, _ = inference.load_cxg_artifact()

    assert metadata["neutral_probability_reference"] == 0.2
