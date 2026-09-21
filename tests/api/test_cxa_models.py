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


def _clear_all_caches():
    cxa_models._coverage_cache.clear()
    cxa_models._shot_coverage_cache.clear()
    cxa_models._summaries_cache.clear()
    cxa_models._explainability_cache.clear()


@pytest.fixture(autouse=True)
def isolated_client_and_cache():
    from opponent_adjusted.api import bigquery_store

    _clear_all_caches()
    original_client = bigquery_store._client_instance
    mock_client = MagicMock()
    bigquery_store._client_instance = mock_client
    yield mock_client
    bigquery_store._client_instance = original_client
    _clear_all_caches()


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


def test_get_cxa_for_shots_returns_only_shots_with_a_chance_creating_pass(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row({
            "shot_event_id": "shot-1",
            "pass_event_id": "pass-1",
            "p_create_predicted_prob": 0.62,
            "p_convert_predicted_prob": 0.34,
            "cxa_combined_score": 0.2108,
        }),
    ]

    store = cxa_models.BigQueryCxaModelStore()
    result = store.get_cxa_for_shots(["shot-1", "shot-2-no-pass"], track="event")

    # Query filters shot_event_id IS NOT NULL server-side, so only real
    # chance-creating-pass shots ever come back -- shot-2 (no such pass) is
    # simply absent, never a placeholder.
    assert set(result.keys()) == {"shot-1"}
    assert result["shot-1"].pass_event_id == "pass-1"
    assert result["shot-1"].p_create_predicted_prob == 0.62
    assert result["shot-1"].p_convert_predicted_prob == 0.34
    assert result["shot-1"].cxa_combined_score == 0.2108
    query_text = mock_client.query.call_args[0][0]
    assert "shot_event_id IS NOT NULL" in query_text


def test_get_cxa_for_shots_unknown_track_raises_without_querying(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    store = cxa_models.BigQueryCxaModelStore()

    with pytest.raises(ValueError):
        store.get_cxa_for_shots(["a"], track="not_a_real_track")

    mock_client.query.assert_not_called()


def test_shot_coverage_is_cached_across_calls_and_independent_of_pass_coverage_cache(
    isolated_client_and_cache,
):
    mock_client = isolated_client_and_cache
    mock_client.query.return_value.result.return_value = [
        _row({
            "shot_event_id": "shot-1",
            "pass_event_id": "pass-1",
            "p_create_predicted_prob": 0.5,
            "p_convert_predicted_prob": 0.4,
            "cxa_combined_score": 0.2,
        })
    ]

    store = cxa_models.BigQueryCxaModelStore()
    store.get_cxa_for_shots(["shot-1"], track="event")
    store.get_cxa_for_shots(["shot-1"], track="event")
    store.get_cxa_for_shots(["shot-1", "shot-2"], track="event")

    assert mock_client.query.call_count == 1


def test_list_model_summaries_keeps_stages_separate_and_includes_coverage(isolated_client_and_cache):
    mock_client = isolated_client_and_cache

    p_create_frozen_row = [_row({"model_family": "lightgbm_tree", "feature_list": ["f1", "f2"]})]
    p_convert_frozen_row = [_row({"model_family": "logistic_mle", "feature_list": ["g1"]})]
    p_create_rows = [
        _row({"model": "dumb_baseline", "split": "test", "n": 100, "log_loss": 0.3, "brier_score": None, "roc_auc": None}),
        _row({"model": "frozen_tree", "split": "test", "n": 100, "log_loss": 0.06, "brier_score": 0.02, "roc_auc": 0.91}),
    ]
    p_convert_rows = [
        _row({"model": "frozen_candidate", "split": "test", "n": 10, "log_loss": 0.25, "brier_score": 0.07, "roc_auc": 0.76}),
    ]
    coverage_row = [_row({"population_n": 100, "chance_creating_n": 10})]

    # 2 tracks x (p_create frozen config + p_convert frozen config + p_create
    # metrics + p_convert metrics + coverage) = 10 calls, cycling through the
    # same 5 canned results per track.
    mock_client.query.return_value.result.side_effect = [
        p_create_frozen_row, p_convert_frozen_row, p_create_rows, p_convert_rows, coverage_row,
        p_create_frozen_row, p_convert_frozen_row, p_create_rows, p_convert_rows, coverage_row,
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
        assert summary.p_create_model_family == "lightgbm_tree"
        assert summary.p_create_feature_list == ["f1", "f2"]
        assert summary.p_convert_model_family == "logistic_mle"
        assert summary.p_convert_feature_list == ["g1"]
        assert summary.coverage.population_n == 100
        assert summary.coverage.chance_creating_n == 10
        assert summary.coverage.coverage_pct == 10.0
        assert summary.combined_score_caveat == cxa_models.COMBINED_SCORE_CAVEAT
        assert "~2%" in summary.combined_score_caveat


def test_get_explainability_splits_tree_importances_from_logistic_coefficients(isolated_client_and_cache):
    mock_client = isolated_client_and_cache

    p_create_frozen_row = [_row({"model_family": "lightgbm_tree", "feature_list": ["f1"]})]
    p_convert_frozen_row = [_row({"model_family": "logistic_mle", "feature_list": ["g1"]})]
    p_create_importance_rows = [
        _row({"feature": "f1", "importance_split": 42, "importance_gain": 12.5}),
    ]
    p_convert_coefficient_rows = [
        _row({"feature": "g1", "coefficient": 0.5, "std_error": 0.1, "p_value": 0.01}),
    ]

    mock_client.query.return_value.result.side_effect = [
        p_create_frozen_row, p_convert_frozen_row, p_create_importance_rows, p_convert_coefficient_rows,
    ]

    store = cxa_models.BigQueryCxaModelStore()
    result = store.get_explainability("event")

    assert result.track == "event"
    # Tree-family stage (p_create) -> feature_importances only, never a fake
    # coefficients row for a model that has no coefficients.
    assert len(result.feature_importances) == 1
    assert result.feature_importances[0].stage == "p_create"
    assert result.feature_importances[0].feature == "f1"
    # Logistic-family stage (p_convert here, in this canned example) -> coefficients only.
    assert len(result.coefficients) == 1
    assert result.coefficients[0].stage == "p_convert"
    assert result.coefficients[0].feature == "g1"


def test_get_explainability_unknown_track_raises_without_querying(isolated_client_and_cache):
    mock_client = isolated_client_and_cache
    store = cxa_models.BigQueryCxaModelStore()

    with pytest.raises(ValueError):
        store.get_explainability("not_a_real_track")

    mock_client.query.assert_not_called()
