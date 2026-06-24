"""End-to-end test for features pipeline and API wiring.

This test assumes that the ingestion E2E test or an equivalent ingestion step
has already populated the SQLite test database with matches and raw events. It
then runs event normalization and shot feature building, and finally exercises
the FastAPI application using TestClient against the same database.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Ensure scripts and src are importable when running via pytest
ROOT = Path(__file__).parent.parent
SCRIPTS_ROOT = ROOT / "scripts"
SRC_ROOT = ROOT / "src"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts import build_shot_features as build_shot_features_script
from scripts import normalize_events as normalize_events_script

from opponent_adjusted.api.service import app
from opponent_adjusted.db.models import Event, Shot, ShotFeature
from opponent_adjusted.db.session import SessionLocal


@pytest.mark.e2e
def test_features_and_api_wiring(e2e_test_env):
    """Run normalization + feature build and hit basic API endpoints.

    The ingestion E2E test creates competitions, matches, and raw events in the
    SQLite-backed database. Here we:

    1. Normalize raw events into the events and detail tables.
    2. Build shot features for version v1.
    3. Assert that core tables have data populated.
    4. Use FastAPI's TestClient to exercise basic API behaviour.
    """

    normalize_events_script.main([])
    build_shot_features_script.main([])

    with SessionLocal() as session:
        assert session.query(Event).count() > 0, "No normalized events created"
        assert session.query(Shot).count() > 0, "No shots created from events"
        assert session.query(ShotFeature).count() > 0, "No shot features created"

    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"

    resp = client.get("/models/cxg/version")
    assert resp.status_code == 404

    # /predict/cxg now has model-backed behaviour. With no trained model
    # artefact available in the fixture environment, it should fail cleanly
    # with 501 rather than returning a hardcoded placeholder response.
    resp = client.post(
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
            "possession_length": 6,
        },
    )
    assert resp.status_code == 501
    assert "trained model artefact" in resp.json()["detail"]

    resp = client.get("/aggregates/player", params={"model": "cxg_v1"})
    assert resp.status_code == 404

    resp = client.get("/aggregates/team", params={"model": "cxg_v1"})
    assert resp.status_code == 404
