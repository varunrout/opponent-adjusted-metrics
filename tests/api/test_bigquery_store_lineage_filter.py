"""Regression tests: every BigQueryServingStore query must filter to one
silver_schema_version.

oam_core's silver tables each carry multiple full lineage-versioned
copies of every row under one data_version (confirmed live against
BigQuery while investigating a join for the CxG Explore-zone badge):
matches 1830 raw rows / 610 distinct, competitions 15/5, and a real
match's shots (joined to events/matches, exactly the join list_shots
uses) 147 raw rows for 49 distinct shots — all exactly 3x, one copy per
silver_schema_version (statsbomb_silver_v1, _v1_1, _v1_2). None of these
nine queries filtered on it before, so every one of them was returning/
aggregating 3x-inflated results in production. This file guards against
that regressing silently.

Mocks the BigQuery client per the established pattern
(test_bigquery_client_singleton.py / test_bigquery_store_caching.py) —
no real credentials or network access.
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


@pytest.fixture(autouse=True)
def isolated_client_and_caches():
    for name in _ALL_CACHES:
        getattr(bigquery_store, name).clear()

    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    mock_client.query.return_value.result.return_value = []
    bigquery_store._client_instance = mock_client

    yield mock_client

    bigquery_store._client_instance = original_client
    for name in _ALL_CACHES:
        getattr(bigquery_store, name).clear()


def _assert_filters_on_silver_schema_version(mock_client) -> None:
    """Every call this test made must have passed a job_config whose
    query_parameters include exactly one silver_schema_version parameter,
    set to the canonical current version."""
    assert mock_client.query.call_count >= 1
    for call in mock_client.query.call_args_list:
        job_config = call.kwargs.get("job_config")
        assert job_config is not None, "query() was called with no job_config at all"
        version_params = [
            p for p in job_config.query_parameters if p.name == "silver_schema_version"
        ]
        assert (
            len(version_params) == 1
        ), f"expected exactly one silver_schema_version parameter, got {version_params}"
        assert version_params[0].value == bigquery_store.SILVER_SCHEMA_VERSION


def test_list_competitions_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_competitions()
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_matches_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_matches()
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_matches_with_filters_still_includes_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_matches(competition_id=1, season_id=2, team_id=3)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_get_match_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().get_match(7)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_lineups_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_lineups(7)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_shots_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_shots(7)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_player_seasons_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_player_seasons()
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_player_shots_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_player_shots(42)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_team_seasons_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_team_seasons()
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)


def test_list_team_shots_filters_by_silver_schema_version(isolated_client_and_caches):
    BigQueryServingStore().list_team_shots(99)
    _assert_filters_on_silver_schema_version(isolated_client_and_caches)
