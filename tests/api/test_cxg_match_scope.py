"""Tests for the CxG match-scope store and router (GET /v1/cxg/matches).

Mirrors test_cxg_coverage.py's mocking pattern — no real credentials or
network access. The rule most likely to get silently violated: cxg_plus
must filter to has_360_match=TRUE on top of the shared test-split filter,
and the two "what counts as covered" rules (cxg_coverage.py's per-shot
lookup and this per-match lookup) must never disagree.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api import cxg_coverage
from opponent_adjusted.api.dependencies import get_cxg_match_scope_store, get_role
from opponent_adjusted.api.main import app


def _make_match_row(match_id: int, *, has_360: bool = False):
    data = {
        "match_id": match_id,
        "split": "test",
        "has_360_match": has_360,
        "event_shot_count": 25,
        "plus_shot_count": 25 if has_360 else 0,
        "event_goal_count": 3,
        "plus_goal_count": 3 if has_360 else 0,
    }
    row = MagicMock()
    row.__getitem__.side_effect = data.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    from opponent_adjusted.api import bigquery_store

    cxg_coverage._match_scope_cache.clear()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    cxg_coverage._match_scope_cache.clear()


def test_list_covered_matches_returns_92_for_cxg_event_and_23_for_cxg_plus(
    isolated_client_and_cache,
):
    mock_client = isolated_client_and_cache
    # 92 test-split matches total, 23 of which have 360 data — matches the
    # live counts confirmed against oam_analysis.cxg_match_splits_v1.
    event_rows = [_make_match_row(i, has_360=(i < 23)) for i in range(92)]
    plus_rows = [row for row in event_rows if row["has_360_match"]]

    def side_effect(query, job_config=None):  # noqa: ARG001
        result_mock = MagicMock()
        if "has_360_match = TRUE" in query:
            result_mock.result.return_value = plus_rows
        else:
            result_mock.result.return_value = event_rows
        return result_mock

    mock_client.query.side_effect = side_effect

    store = cxg_coverage.BigQueryCxgMatchScopeStore()
    event_matches = store.list_covered_matches(track="cxg_event")
    plus_matches = store.list_covered_matches(track="cxg_plus")

    assert len(event_matches) == 92
    assert len(plus_matches) == 23
    assert all(m.has_360_match for m in plus_matches)


def test_query_filters_on_the_shared_coverage_split_constant(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = []

    store = cxg_coverage.BigQueryCxgMatchScopeStore()
    store.list_covered_matches(track="cxg_event")

    query = mock_client.query.call_args[0][0]
    assert f"split = '{cxg_coverage.COVERAGE_SPLIT}'" in query
    assert cxg_coverage.COVERAGE_SPLIT == "test"


def test_cxg_plus_adds_has_360_filter_but_cxg_event_does_not(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = []

    store = cxg_coverage.BigQueryCxgMatchScopeStore()
    store.list_covered_matches(track="cxg_event")
    store.list_covered_matches(track="cxg_plus")

    event_query = mock_client.query.call_args_list[0][0][0]
    plus_query = mock_client.query.call_args_list[1][0][0]
    assert "has_360_match = TRUE" not in event_query
    assert "has_360_match = TRUE" in plus_query


def test_unknown_track_raises_without_querying(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    store = cxg_coverage.BigQueryCxgMatchScopeStore()

    with pytest.raises(ValueError):
        store.list_covered_matches(track="not_a_real_track")

    mock_client.query.assert_not_called()


def test_track_matches_are_cached_across_calls(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_make_match_row(1)]

    store = cxg_coverage.BigQueryCxgMatchScopeStore()
    store.list_covered_matches(track="cxg_event")
    store.list_covered_matches(track="cxg_event")

    assert mock_client.query.call_count == 1


class _FakeMatchScopeStore:
    def __init__(self, matches_by_track: dict[str, list[cxg_coverage.CxgMatchScopeRow]]):
        self._matches_by_track = matches_by_track

    def list_covered_matches(self, *, track: str) -> list[cxg_coverage.CxgMatchScopeRow]:
        return self._matches_by_track.get(track, [])


@pytest.fixture
def client():
    fake_row = cxg_coverage.CxgMatchScopeRow(
        match_id=1,
        split="test",
        has_360_match=True,
        event_shot_count=20,
        plus_shot_count=20,
        event_goal_count=2,
        plus_goal_count=2,
    )
    app.dependency_overrides[get_cxg_match_scope_store] = lambda: _FakeMatchScopeStore(
        {"cxg_event": [fake_row], "cxg_plus": [fake_row]}
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_cxg_match_scope_store, None)
    app.dependency_overrides.pop(get_role, None)


def test_matches_endpoint_is_guest_accessible_not_admin_gated(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/cxg/matches", params={"track": "cxg_event"})
    assert response.status_code == 200
    body = response.json()
    assert body == [
        {
            "match_id": 1,
            "split": "test",
            "has_360_match": True,
            "event_shot_count": 20,
            "plus_shot_count": 20,
            "event_goal_count": 2,
            "plus_goal_count": 2,
        }
    ]


def test_matches_endpoint_rejects_unknown_track(client):
    app.dependency_overrides[get_role] = lambda: "guest"
    response = client.get("/v1/cxg/matches", params={"track": "bogus"})
    assert response.status_code == 400


def test_matches_endpoint_reachable_with_no_authorization_header_at_all(client):
    # No app.dependency_overrides[get_role] set here — exercises the real
    # get_role -> get_auth_context path with zero Authorization header,
    # which must resolve to guest rather than reject.
    response = client.get("/v1/cxg/matches", params={"track": "cxg_event"})
    assert response.status_code == 200
