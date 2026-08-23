"""Tests for the TTL cache layer wrapping BigQueryServingStore's public methods.

Mirrors test_bigquery_client_singleton.py's mocking pattern — no real
BigQuery credentials or network access required. Uses get_match as the
representative method: simple single-parameter query, easy to distinguish
cache hits (query never re-called) from misses (query re-called with the
new parameter value).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opponent_adjusted.api import bigquery_store
from opponent_adjusted.api.bigquery_store import BigQueryServingStore

_ALL_CACHES = [
    "_competitions_cache",
    "_matches_cache",
    "_match_cache",
    "_lineups_cache",
    "_shots_cache",
    "_player_seasons_cache",
    "_player_shots_cache",
    "_team_seasons_cache",
    "_team_shots_cache",
]


def _make_match_row(match_id: int):
    data = {
        "match_id": match_id,
        "competition_id": 1,
        "season_id": 1,
        "match_date": "2024-01-01",
        "kick_off": "12:00:00",
        "home_team_id": 10,
        "home_team_name": "Home",
        "away_team_id": 20,
        "away_team_name": "Away",
        "home_score": 1,
        "away_score": 0,
        "competition_stage": "Group",
        "stadium": "Stadium",
        "referee": "Ref",
        "match_status": "available",
        "match_status_360": None,
        "last_updated": "2024-01-01",
        "last_updated_360": None,
    }
    row = MagicMock()
    row.__getitem__.side_effect = data.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_caches():
    """Swap in a fresh mock BigQuery client and clear every cache before/after each test.

    Caches are module-level singletons, shared across the whole test
    process — without this, a hit populated by one test could silently
    satisfy another test's assertion about call counts.
    """
    for name in _ALL_CACHES:
        getattr(bigquery_store, name).clear()

    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client

    yield mock_client

    bigquery_store._client_instance = original_client
    for name in _ALL_CACHES:
        getattr(bigquery_store, name).clear()


def test_repeated_call_with_same_args_hits_bigquery_only_once(isolated_client_and_caches):
    mock_client = isolated_client_and_caches
    mock_client.query.return_value.result.return_value = [_make_match_row(7)]

    store = BigQueryServingStore()
    first = store.get_match(7)
    second = store.get_match(7)

    assert first is not None
    assert first == second
    assert mock_client.query.call_count == 1


def test_different_args_produce_separate_cache_entries_no_false_hit(isolated_client_and_caches):
    mock_client = isolated_client_and_caches

    def query_side_effect(query, job_config=None):  # noqa: ARG001
        requested_match_id = job_config.query_parameters[0].value
        result_mock = MagicMock()
        result_mock.result.return_value = [_make_match_row(requested_match_id)]
        return result_mock

    mock_client.query.side_effect = query_side_effect

    store = BigQueryServingStore()
    first = store.get_match(7)
    second = store.get_match(8)

    assert first is not None and second is not None
    assert first.match_id == 7
    assert second.match_id == 8
    assert first != second
    assert mock_client.query.call_count == 2

    # A subsequent call with either previously-seen match_id is still a
    # cache hit — the earlier round of two distinct-arg calls shouldn't
    # have collided with each other's cache slot.
    store.get_match(7)
    store.get_match(8)
    assert mock_client.query.call_count == 2
