"""Tests for the CxG v3 Explore-zone coverage store and router.

Mocks the BigQuery client per the established pattern
(test_bigquery_client_singleton.py / test_bigquery_store_caching.py) —
no real credentials or network access. This is the rule most likely to
get silently violated by a careless join, per the prompt: a shot with
coverage must show its real CxG value; a shot without coverage must be
absent from the response, never a placeholder.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api import cxg_coverage
from opponent_adjusted.api.dependencies import get_cxg_coverage_store, get_role
from opponent_adjusted.api.main import app


def _make_pred_row(event_id: str, prob: float):
    row = MagicMock()
    row.__getitem__.side_effect = {"event_id": event_id, "v3_predicted_prob": prob}.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    # cxg_coverage.py imports _client from bigquery_store rather than
    # defining its own, so the shared singleton lives there.
    from opponent_adjusted.api import bigquery_store

    cxg_coverage._coverage_cache.clear()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    cxg_coverage._coverage_cache.clear()


def test_get_cxg_for_events_returns_only_covered_ids(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _make_pred_row("covered-1", 0.42),
        _make_pred_row("covered-2", 0.15),
    ]

    store = cxg_coverage.BigQueryCxgCoverageStore()
    result = store.get_cxg_for_events(
        ["covered-1", "covered-2", "not-covered-3"], track="cxg_event"
    )

    assert result == {"covered-1": 0.42, "covered-2": 0.15}
    assert "not-covered-3" not in result


def test_get_cxg_for_events_unknown_track_raises_without_querying(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    store = cxg_coverage.BigQueryCxgCoverageStore()

    with pytest.raises(ValueError):
        store.get_cxg_for_events(["a"], track="not_a_real_track")

    mock_client.query.assert_not_called()


def test_track_coverage_is_cached_across_calls(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_make_pred_row("e1", 0.5)]

    store = cxg_coverage.BigQueryCxgCoverageStore()
    store.get_cxg_for_events(["e1"], track="cxg_event")
    store.get_cxg_for_events(["e1"], track="cxg_event")
    store.get_cxg_for_events(["e1", "e2"], track="cxg_event")

    assert mock_client.query.call_count == 1


def test_different_tracks_are_separate_cache_entries(isolated_client_and_cache):
    mock_client = isolated_client_and_cache

    def side_effect(query, job_config=None):  # noqa: ARG001
        result_mock = MagicMock()
        if "cxg_plus_v3" in query:
            result_mock.result.return_value = [_make_pred_row("plus-1", 0.9)]
        else:
            result_mock.result.return_value = [_make_pred_row("event-1", 0.1)]
        return result_mock

    mock_client.query.side_effect = side_effect

    store = cxg_coverage.BigQueryCxgCoverageStore()
    event_result = store.get_cxg_for_events(["event-1"], track="cxg_event")
    plus_result = store.get_cxg_for_events(["plus-1"], track="cxg_plus")

    assert event_result == {"event-1": 0.1}
    assert plus_result == {"plus-1": 0.9}
    assert mock_client.query.call_count == 2


class _FakeCoverageStore:
    def __init__(self, coverage: dict[str, float]):
        self._coverage = coverage

    def get_cxg_for_events(
        self, event_ids: list[str], *, track: str
    ) -> dict[str, float]:  # noqa: ARG002
        return {eid: val for eid, val in self._coverage.items() if eid in event_ids}


@pytest.fixture
def client():
    app.dependency_overrides[get_cxg_coverage_store] = lambda: _FakeCoverageStore(
        {"covered-1": 0.3, "covered-2": 0.6}
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_cxg_coverage_store, None)
    app.dependency_overrides.pop(get_role, None)


def test_coverage_endpoint_is_guest_accessible_not_admin_gated(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get(
        "/v1/cxg/coverage", params={"track": "cxg_event", "event_ids": "covered-1,not-covered-3"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["track"] == "cxg_event"
    assert body["values"] == {"covered-1": 0.3}
    assert "not-covered-3" not in body["values"]


def test_coverage_endpoint_rejects_unknown_track(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/cxg/coverage", params={"track": "bogus", "event_ids": "covered-1"})
    assert response.status_code == 400


def test_coverage_endpoint_handles_empty_event_ids(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/cxg/coverage", params={"track": "cxg_event", "event_ids": ""})
    assert response.status_code == 200
    assert response.json()["values"] == {}
