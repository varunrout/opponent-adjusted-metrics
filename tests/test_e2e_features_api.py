"""End-to-end test for features pipeline and API wiring.

This test seeds its own SQLite test database with competitions,
matches, and raw events. It then runs event normalization and shot
feature building, and finally exercises the FastAPI application using
TestClient against the same database.
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

from scripts import normalize_events as normalize_events_script  # noqa: E402
from scripts import build_shot_features as build_shot_features_script  # noqa: E402
from scripts.ingest_competitions import ingest_competitions  # noqa: E402
from scripts.ingest_matches import ingest_matches  # noqa: E402
from scripts.ingest_events import ingest_events  # noqa: E402

from opponent_adjusted.api.service import app  # noqa: E402
from opponent_adjusted.db.session import SessionLocal  # noqa: E402
from opponent_adjusted.db.models import Event, Shot, ShotFeature  # noqa: E402


@pytest.mark.e2e
def test_features_and_api_wiring(e2e_test_env):
    """Run normalization + feature build and hit basic API endpoints.

    This test creates competitions, matches, and raw events in the
    SQLite-backed database. Here we:

    1. Seed competitions, matches, and raw events into the test database.
    2. Normalize raw events into the `events` and detail tables.
    3. Build shot features for version `v1`.
    4. Assert that core tables have data populated.
    5. Use FastAPI's TestClient to exercise the health and model
       endpoints, verifying current placeholder / 404 behaviour.
    """

    # Step 1: seed this test's database independently so it does not rely
    # on another E2E test running first.
    ingest_competitions()
    ingest_matches()
    ingest_events(limit=1)

    # Step 2: normalize events into canonical tables by calling the
    # underlying functions without going through argparse-based CLIs.
    normalize_events_script.main([])

    # Step 3: build shot features for default version v1 in the same way.
    build_shot_features_script.main([])

    # Step 4: basic invariants on feature tables.
    with SessionLocal() as session:
        assert session.query(Event).count() > 0, "No normalized events created"
        assert session.query(Shot).count() > 0, "No shots created from events"
        assert session.query(ShotFeature).count() > 0, "No shot features created"

    # Step 5: exercise API endpoints using TestClient. These currently
    # run against the same SQLite DB via session_scope.
    client = TestClient(app)

    # /health should be 200 and return expected payload
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "healthy"

    # /models/cxg/version should 404 when no model is registered
    resp = client.get("/models/cxg/version")
    assert resp.status_code == 404

    # /predict/cxg should reflect the 501 placeholder behaviour
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
            "possession_length": 5,
        },
    )
    assert resp.status_code == 501
    assert "trained model artefact" in resp.json()["detail"]

    # Aggregates endpoints should 404 for an unknown / missing model
    resp = client.get("/aggregates/player", params={"model": "cxg_v1"})
    assert resp.status_code == 404

    resp = client.get("/aggregates/team", params={"model": "cxg_v1"})
    assert resp.status_code == 404
