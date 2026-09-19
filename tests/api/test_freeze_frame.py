"""Tests for the per-shot 360 freeze-frame store and router
(oam_core.three_sixty_frames/three_sixty_players).

Mocks the BigQuery client per the established pattern (see
test_cxg_coverage.py) — no real credentials or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api import freeze_frame
from opponent_adjusted.api.dependencies import get_freeze_frame_store, get_role
from opponent_adjusted.api.main import app


def _frame_row(visible_area: list[float], frame_player_count: int):
    row = MagicMock()
    data = {"visible_area": visible_area, "frame_player_count": frame_player_count}
    row.__getitem__.side_effect = data.__getitem__
    return row


def _player_row(ordinal: int, teammate: bool, actor: bool, keeper: bool, x: float, y: float):
    row = MagicMock()
    data = {
        "frame_player_ordinal": ordinal,
        "teammate": teammate,
        "actor": actor,
        "keeper": keeper,
        "x": x,
        "y": y,
    }
    row.__getitem__.side_effect = data.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client():
    from opponent_adjusted.api import bigquery_store

    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client


def test_get_freeze_frame_returns_none_when_no_frame_row(isolated_client):
    mock_client = isolated_client
    mock_client.query.return_value.result.return_value = []

    store = freeze_frame.BigQueryFreezeFrameStore()
    result = store.get_freeze_frame(3794686, "not-covered-event")

    assert result is None
    # Only the frame query should run — no point querying players for a
    # shot that has no frame at all.
    assert mock_client.query.call_count == 1


def test_get_freeze_frame_returns_frame_and_players(isolated_client):
    mock_client = isolated_client

    def side_effect(query, job_config=None):  # noqa: ARG001
        result_mock = MagicMock()
        if "three_sixty_frames" in query:
            result_mock.result.return_value = [_frame_row([0.0, 0.0, 120.0, 80.0], 4)]
        else:
            result_mock.result.return_value = [
                _player_row(0, True, False, False, 82.1, 24.6),
                _player_row(1, True, True, False, 93.6, 51.6),
                _player_row(2, False, False, False, 94.0, 50.5),
                _player_row(3, False, False, True, 117.3, 41.6),
            ]
        return result_mock

    mock_client.query.side_effect = side_effect

    store = freeze_frame.BigQueryFreezeFrameStore()
    result = store.get_freeze_frame(3794686, "4f0aba85-af15-453b-8052-cbc71c47b93c")

    assert result is not None
    assert result.event_id == "4f0aba85-af15-453b-8052-cbc71c47b93c"
    assert result.match_id == 3794686
    assert result.visible_area == [0.0, 0.0, 120.0, 80.0]
    assert len(result.players) == 4
    actor = next(p for p in result.players if p.actor)
    assert actor.teammate is True
    keeper = next(p for p in result.players if p.keeper)
    assert keeper.teammate is False


def test_get_freeze_frame_filters_by_silver_schema_version(isolated_client):
    mock_client = isolated_client
    mock_client.query.return_value.result.return_value = []

    store = freeze_frame.BigQueryFreezeFrameStore()
    store.get_freeze_frame(1, "e1")

    call = mock_client.query.call_args_list[0]
    job_config = call.kwargs.get("job_config")
    assert job_config is not None
    version_params = [p for p in job_config.query_parameters if p.name == "silver_schema_version"]
    assert len(version_params) == 1
    from opponent_adjusted.api.bigquery_store import SILVER_SCHEMA_VERSION

    assert version_params[0].value == SILVER_SCHEMA_VERSION


class _FakeFreezeFrameStore:
    def __init__(self, frames: dict[tuple[int, str], freeze_frame.ShotFreezeFrameResponse]):
        self._frames = frames

    def get_freeze_frame(self, match_id: int, event_id: str):
        return self._frames.get((match_id, event_id))


@pytest.fixture
def client():
    fake_frame = freeze_frame.ShotFreezeFrameResponse(
        event_id="covered-event",
        match_id=3794686,
        visible_area=[0.0, 0.0, 120.0, 80.0],
        players=[
            freeze_frame.FreezeFramePlayerResponse(
                ordinal=0, teammate=True, actor=True, keeper=False, x=93.6, y=51.6
            ),
        ],
    )
    app.dependency_overrides[get_freeze_frame_store] = lambda: _FakeFreezeFrameStore(
        {(3794686, "covered-event"): fake_frame}
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_freeze_frame_store, None)
    app.dependency_overrides.pop(get_role, None)


def test_freeze_frame_endpoint_returns_payload_for_covered_shot(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/matches/3794686/shots/covered-event/freeze-frame")
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == "covered-event"
    assert len(body["players"]) == 1


def test_freeze_frame_endpoint_404s_for_uncovered_shot(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/matches/3794686/shots/not-covered-event/freeze-frame")
    assert response.status_code == 404


def test_freeze_frame_endpoint_is_guest_accessible_not_admin_gated(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/matches/3794686/shots/covered-event/freeze-frame")
    assert response.status_code == 200
