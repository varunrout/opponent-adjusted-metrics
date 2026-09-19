"""Tests for the shared BigQuery client singleton in bigquery_store.py."""

from __future__ import annotations

from unittest.mock import patch

from opponent_adjusted.api import bigquery_store


def test_client_returns_the_same_instance_on_repeated_calls():
    """_client() must not construct a fresh bigquery.Client() on every call."""
    bigquery_store._client_instance = None
    try:
        with patch.object(bigquery_store.bigquery, "Client") as mock_client_cls:
            mock_client_cls.return_value = object()

            first = bigquery_store._client()
            second = bigquery_store._client()

            assert first is second
            mock_client_cls.assert_called_once_with(project=bigquery_store.PROJECT)
    finally:
        bigquery_store._client_instance = None
