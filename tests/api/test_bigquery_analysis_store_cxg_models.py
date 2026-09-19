"""Tests for BigQueryAnalysisStore's list_cxg_model_results/list_cxg_coefficients.

Mirrors the mocking pattern in test_bigquery_client_singleton.py and
test_gcs_signing.py — no real BigQuery credentials or network access
required. Swaps in a mock client via bigquery_store._client_instance
directly, same as test_bigquery_store_caching.py does for the oam_core store.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from opponent_adjusted.api import bigquery_store
from opponent_adjusted.api.bigquery_analysis_store import BigQueryAnalysisStore


def _make_row(data: dict):
    row = MagicMock()
    row.__getitem__.side_effect = data.__getitem__
    return row


@pytest.fixture(autouse=True)
def isolated_client():
    """Swap in a fresh mock BigQuery client for each test; restore after."""
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client


def test_list_cxg_model_results_maps_rows_and_flags_current_models(isolated_client):
    fake_rows = [
        _make_row(
            {
                "model_key": "event_v3",
                "track": "cxg_event",
                "split": "test",
                "model": "v3",
                "n": 2427,
                "log_loss": 0.3003,
                "brier_score": 0.0852,
                "roc_auc": 0.7148,
            }
        ),
        _make_row(
            {
                "model_key": "baseline_v1",
                "track": "cxg_event",
                "split": "test",
                "model": "v1",
                "n": 2427,
                "log_loss": 0.3058,
                "brier_score": 0.0872,
                "roc_auc": 0.6939,
            }
        ),
    ]
    isolated_client.query.return_value.result.return_value = fake_rows

    store = BigQueryAnalysisStore()
    results = store.list_cxg_model_results()

    assert len(results) == 2
    by_key = {r.model_key: r for r in results}

    assert by_key["event_v3"].is_frozen is True
    assert by_key["event_v3"].is_current is True
    assert by_key["event_v3"].track == "cxg_event"
    assert by_key["event_v3"].log_loss == 0.3003

    assert by_key["baseline_v1"].is_frozen is True
    assert by_key["baseline_v1"].is_current is False

    # A single UNION ALL query, not one query per model table.
    assert isolated_client.query.call_count == 1
    query_text = isolated_client.query.call_args[0][0]
    assert "UNION ALL" in query_text
    assert "cxg_baseline_v1_metrics" in query_text
    assert "cxg_event_v3_metrics" in query_text
    assert "cxg_plus_v2_metrics" in query_text
    assert "cxg_plus_v3_metrics" in query_text


def test_list_cxg_coefficients_maps_rows_for_a_known_model_key(isolated_client):
    fake_rows = [
        _make_row(
            {
                "track": "cxg_event",
                "feature": "const",
                "coefficient": -2.4634,
                "std_error": None,
                "p_value": None,
            }
        ),
        _make_row(
            {
                "track": "cxg_event",
                "feature": "shot_x_sb",
                "coefficient": 0.7701,
                "std_error": None,
                "p_value": None,
            }
        ),
    ]
    isolated_client.query.return_value.result.return_value = fake_rows

    store = BigQueryAnalysisStore()
    results = store.list_cxg_coefficients("event_v3")

    assert len(results) == 2
    assert all(r.model_key == "event_v3" for r in results)
    assert {r.feature for r in results} == {"const", "shot_x_sb"}

    query_text = isolated_client.query.call_args[0][0]
    assert "cxg_event_v3_coefficients" in query_text


def test_list_cxg_coefficients_unknown_model_key_raises_without_querying(isolated_client):
    store = BigQueryAnalysisStore()

    with pytest.raises(ValueError, match="not-a-real-model"):
        store.list_cxg_coefficients("not-a-real-model")

    isolated_client.query.assert_not_called()
