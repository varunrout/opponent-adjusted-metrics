"""Tests for the opponent-adjusted per-shot context store and router
(oam_analysis.cxg_analysis_opponent_adjusted_v1, content_spec_v3.md §9.2).

Mocks the BigQuery client per the established pattern (see test_cxg_coverage.py) —
no real credentials or network access.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api import cxg_coverage
from opponent_adjusted.api.dependencies import get_opponent_context_store, get_role
from opponent_adjusted.api.main import app


def _make_context_row(event_id: str, **overrides):
    defaults = {
        "event_id": event_id,
        "match_id": 7,
        "player_id": 3010,
        "team_id": 771,
        "nearest_defender_odi": 1.2,
        "mean_backline_odi": 2.3,
        "gk_odi": 3.4,
        "defensive_profile_cluster": 2,
        "nearest_defender_role": "Center Back",
        "nearest_defender_zone_displacement": 0.5,
        "nearest_defender_gap": 1.1,
        "nearest_defender_style_archetype": "Interceptor",
        "has_360_frame": True,
    }
    defaults.update(overrides)
    row = MagicMock()
    row.__getitem__.side_effect = defaults.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    from opponent_adjusted.api import bigquery_store

    cxg_coverage._opponent_context_cache.clear()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    cxg_coverage._opponent_context_cache.clear()


def test_get_opponent_context_returns_only_requested_covered_ids(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _make_context_row("shot-1"),
        _make_context_row("shot-2", nearest_defender_style_archetype="Pressurer"),
    ]

    store = cxg_coverage.BigQueryOpponentContextStore()
    result = store.get_opponent_context(["shot-1", "shot-3-not-covered"])

    assert [row.event_id for row in result] == ["shot-1"]
    assert result[0].nearest_defender_role == "Center Back"


def test_get_opponent_context_bulk_lookup(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _make_context_row("shot-1"),
        _make_context_row("shot-2", nearest_defender_style_archetype="Pressurer"),
    ]

    store = cxg_coverage.BigQueryOpponentContextStore()
    result = store.get_opponent_context(["shot-1", "shot-2"])

    assert {row.event_id for row in result} == {"shot-1", "shot-2"}


def test_full_table_is_cached_across_calls(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_make_context_row("shot-1")]

    store = cxg_coverage.BigQueryOpponentContextStore()
    store.get_opponent_context(["shot-1"])
    store.get_opponent_context(["shot-1"])
    store.get_opponent_context([])

    assert mock_client.query.call_count == 1


class _FakeOpponentContextStore:
    def __init__(self, rows: dict[str, cxg_coverage.OpponentContextResponse]):
        self._rows = rows

    def get_opponent_context(
        self, event_ids: list[str]
    ) -> list[cxg_coverage.OpponentContextResponse]:
        return [self._rows[eid] for eid in event_ids if eid in self._rows]


@pytest.fixture
def client():
    fake_row = cxg_coverage.OpponentContextResponse(
        event_id="covered-1",
        match_id=7,
        player_id=3010,
        team_id=771,
        nearest_defender_odi=1.2,
        mean_backline_odi=2.3,
        gk_odi=3.4,
        defensive_profile_cluster=2,
        nearest_defender_role="Center Back",
        nearest_defender_zone_displacement=0.5,
        nearest_defender_gap=1.1,
        nearest_defender_style_archetype="Interceptor",
        has_360_frame=True,
    )
    app.dependency_overrides[get_opponent_context_store] = lambda: _FakeOpponentContextStore(
        {"covered-1": fake_row}
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_opponent_context_store, None)
    app.dependency_overrides.pop(get_role, None)


def test_opponent_context_endpoint_is_guest_accessible_not_admin_gated(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get(
        "/v1/cxg/opponent-context", params={"event_ids": "covered-1,not-covered-2"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["event_id"] == "covered-1"
    assert body[0]["nearest_defender_style_archetype"] == "Interceptor"


def test_opponent_context_endpoint_handles_empty_event_ids(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/cxg/opponent-context", params={"event_ids": ""})
    assert response.status_code == 200
    assert response.json() == []
