"""Tests for the CxA combined-scorer store and router.

Mocks the BigQuery client per the established pattern (test_cxg_coverage.py).
The rule most likely to get silently violated by a careless join: a pass that
created a chance must show its real p_convert/cxa_combined_score; a pass that
never created one must show `None` (never a placeholder like 0) for those two
fields; a pass_event_id with no row at all must be absent from the response.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opponent_adjusted.api import cxa_models


def _row(values: dict):
    row = MagicMock()
    row.__getitem__.side_effect = values.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    from opponent_adjusted.api import bigquery_store

    cxa_models._coverage_cache.clear()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    cxa_models._coverage_cache.clear()


def test_get_cxa_for_passes_returns_only_covered_ids_and_no_placeholder(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row({
            "pass_event_id": "chance-creating-1",
            "p_create_predicted_prob": 0.12,
            "p_convert_predicted_prob": 0.34,
            "cxa_combined_score": 0.0408,
        }),
        _row({
            "pass_event_id": "not-chance-creating-2",
            "p_create_predicted_prob": 0.05,
            "p_convert_predicted_prob": None,
            "cxa_combined_score": None,
        }),
    ]

    store = cxa_models.BigQueryCxaModelStore()
    result = store.get_cxa_for_passes(
        ["chance-creating-1", "not-chance-creating-2", "never-scored-3"], track="event"
    )

    assert set(result.keys()) == {"chance-creating-1", "not-chance-creating-2"}
    assert result["chance-creating-1"].p_convert_predicted_prob == 0.34
    assert result["chance-creating-1"].cxa_combined_score == 0.0408
    # Never-chance-creating: p_convert/combined are None, not 0 or a placeholder.
    assert result["not-chance-creating-2"].p_create_predicted_prob == 0.05
    assert result["not-chance-creating-2"].p_convert_predicted_prob is None
    assert result["not-chance-creating-2"].cxa_combined_score is None
    # Never scored at all: simply absent.
    assert "never-scored-3" not in result


def test_get_cxa_for_passes_unknown_track_raises_without_querying(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    store = cxa_models.BigQueryCxaModelStore()

    with pytest.raises(ValueError):
        store.get_cxa_for_passes(["a"], track="not_a_real_track")

    mock_client.query.assert_not_called()


def test_track_coverage_is_cached_across_calls(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row({
            "pass_event_id": "e1",
            "p_create_predicted_prob": 0.1,
            "p_convert_predicted_prob": 0.2,
            "cxa_combined_score": 0.02,
        })
    ]

    store = cxa_models.BigQueryCxaModelStore()
    store.get_cxa_for_passes(["e1"], track="event")
    store.get_cxa_for_passes(["e1"], track="event")
    store.get_cxa_for_passes(["e1", "e2"], track="event")

    assert mock_client.query.call_count == 1


def test_list_model_summaries_keeps_stages_separate_and_includes_coverage(isolated_client_and_cache):
    mock_client = isolated_client_and_cache

    p_create_rows = [
        _row({"model": "dumb_baseline", "split": "test", "n": 100, "log_loss": 0.3, "brier_score": None, "roc_auc": None}),
        _row({"model": "frozen_tree", "split": "test", "n": 100, "log_loss": 0.06, "brier_score": 0.02, "roc_auc": 0.91}),
    ]
    p_convert_rows = [
        _row({"model": "frozen_candidate", "split": "test", "n": 10, "log_loss": 0.25, "brier_score": 0.07, "roc_auc": 0.76}),
    ]
    coverage_row = _row({"population_n": 100, "chance_creating_n": 10})

    # 2 tracks x (p_create query + p_convert query + coverage query) = 6 calls,
    # cycling through the 3 canned results per track.
    mock_client.query.return_value.result.side_effect = [
        p_create_rows, p_convert_rows, [coverage_row],
        p_create_rows, p_convert_rows, [coverage_row],
    ]

    store = cxa_models.BigQueryCxaModelStore()
    summaries = store.list_model_summaries()

    assert [s.track for s in summaries] == ["event", "plus"]
    for summary in summaries:
        stages = {m.stage for m in summary.stage_metrics}
        assert stages == {"p_create", "p_convert"}
        # Never a single unioned "combined" row masquerading as a stage.
        assert all(m.stage in ("p_create", "p_convert") for m in summary.stage_metrics)
        frozen = [m for m in summary.stage_metrics if m.is_frozen]
        assert {m.model for m in frozen} == {"frozen_tree", "frozen_candidate"}
        assert summary.coverage.population_n == 100
        assert summary.coverage.chance_creating_n == 10
        assert summary.coverage.coverage_pct == 10.0
        assert summary.combined_score_caveat == cxa_models.COMBINED_SCORE_CAVEAT
        assert "~2%" in summary.combined_score_caveat
