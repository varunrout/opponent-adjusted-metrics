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


def test_get_player_cxa_sums_across_seasons_and_preserves_null_vs_zero(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row(_row_values(player_id=1, season_id=2025, cxa_event_n_passes_created=2, cxa_event_mean=0.03, cxa_event_total=0.06)),
        _row(_row_values(player_id=1, season_id=2026, cxa_event_n_passes_created=3, cxa_event_mean=0.02, cxa_event_total=0.06)),
        _row(_row_values(player_id=2, season_id=2026)),  # a different player, must not leak in
    ]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    result = store.get_player_cxa(1)

    assert result.player_id == 1
    # Summed across both of player 1's seasons: n=5, total=0.12 -> mean=0.024,
    # never a re-average of the two per-season means (which would wrongly give 0.025).
    assert result.event.n == 5
    assert result.event.total == pytest.approx(0.12)
    assert result.event.mean == pytest.approx(0.024)
    # cxa_plus was never set for either row (n=0 in both) -> null, not 0.0.
    assert result.plus.n == 0
    assert result.plus.mean is None
    assert result.plus.total is None


def test_get_player_cxa_with_zero_matching_rows_is_all_null(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [_row(_row_values(player_id=999))]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    result = store.get_player_cxa(1)  # no row matches player_id=1

    assert result.event.n == 0
    assert result.event.mean is None
    assert result.plus.n == 0
    assert result.plus.mean is None


def test_get_team_cxa_sums_across_every_matching_player_no_double_count(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row(_row_values(player_id=1, team_id=10, cxa_event_n_passes_created=2, cxa_event_mean=0.03, cxa_event_total=0.06)),
        _row(_row_values(player_id=2, team_id=10, cxa_event_n_passes_created=1, cxa_event_mean=0.05, cxa_event_total=0.05)),
        _row(_row_values(player_id=3, team_id=20, cxa_event_n_passes_created=9, cxa_event_mean=0.9, cxa_event_total=8.1)),
    ]

    store = quadrant_scatter.BigQueryQuadrantScatterStore()
    result = store.get_team_cxa(10)

    assert result.team_id == 10
    # Team 10's two players combined: n=3, total=0.11 -- team 20's player (n=9)
    # must not leak into team 10's rollup.
    assert result.event.n == 3
    assert result.event.total == pytest.approx(0.11)
    assert result.event.mean == pytest.approx(0.11 / 3)
