"""Tests for the player-season CxG/CxA quadrant-scatter store.

Mocks the BigQuery client per the established pattern (test_cxa_models.py). The
rule most likely to get silently violated: a player-season with zero coverage for
a metric must show `None` for that metric's mean/total (never 0), while its own
`_n` count is a real 0 -- and the store must always filter to `split='test'`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opponent_adjusted.api import quadrant_scatter


def _row(values: dict):
    row = MagicMock()
    row.items.return_value = list(values.items())
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    from opponent_adjusted.api import bigquery_store

    quadrant_scatter._scatter_cache.clear()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    quadrant_scatter._scatter_cache.clear()


def _row_values(**overrides):
    base = dict(
        player_id=1,
        player_name="Test Player",
        team_id=10,
        team_name="Test Team",
        competition_id=2,
        season_id=2026,
        split="test",
        cxg_event_n_shots=5,
        cxg_event_mean=0.12,
        cxg_event_total=0.6,
        cxg_event_total_xg=0.5,
        cxg_event_goals=1,
        cxg_plus_n_shots=0,
        cxg_plus_mean=None,
        cxg_plus_total=None,
        cxg_plus_total_xg=None,
        cxg_plus_goals=0,
        cxa_event_n_passes_created=2,
        cxa_event_mean=0.03,
        cxa_event_total=0.06,
        cxa_plus_n_passes_created=0,
        cxa_plus_mean=None,
        cxa_plus_total=None,
    )
    base.update(overrides)
    return base


def test_list_player_season_rows_filters_test_split_and_preserves_null_vs_zero(
    isolated_client_and_cache,
):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_row(_row_values())]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    rows = store.list_player_season_rows()

    assert len(rows) == 1
    row = rows[0]
    assert row.split == "test"
    # Zero-coverage metric: count is a real 0, mean/total are None, never 0.
    assert row.cxg_plus_n_shots == 0
    assert row.cxg_plus_mean is None
    assert row.cxg_plus_total is None
    assert row.cxa_plus_n_passes_created == 0
    assert row.cxa_plus_mean is None
    # Covered metric: real values pass through untouched.
    assert row.cxg_event_n_shots == 5
    assert row.cxg_event_mean == 0.12
    assert row.cxa_event_n_passes_created == 2

    query_text = mock_client.query.call_args[0][0]
    assert "split = @split" in query_text
    job_config = mock_client.query.call_args[1]["job_config"]
    split_param = next(p for p in job_config.query_parameters if p.name == "split")
    assert split_param.value == "test"


def test_list_player_season_rows_applies_competition_and_season_filters(
    isolated_client_and_cache,
):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_row(_row_values())]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    store.list_player_season_rows(competition_id=2, season_id=2026)

    job_config = mock_client.query.call_args[1]["job_config"]
    param_names = {p.name for p in job_config.query_parameters}
    assert param_names == {"split", "competition_id", "season_id"}


def test_list_player_season_rows_is_cached_per_scope(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_row(_row_values())]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    store.list_player_season_rows()
    store.list_player_season_rows()
    store.list_player_season_rows(competition_id=2, season_id=2026)

    assert mock_client.query.call_count == 2
