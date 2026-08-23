"""Tests for the analysis router (oam_analysis dataset; admin-only)."""

from __future__ import annotations

import pytest

import opponent_adjusted.api.routers.analysis as analysis_router
from opponent_adjusted.api.dependencies import get_role
from opponent_adjusted.api.main import app

ROUTES = [
    "/v1/analysis/features",
    "/v1/analysis/correlation",
    "/v1/analysis/univariate",
    "/v1/analysis/bivariate",
    "/v1/analysis/pca",
    "/v1/analysis/charts",
]


def _fake_sign(uri: str | None) -> str | None:
    return f"https://signed.example/{uri}" if uri else None


@pytest.fixture(autouse=True)
def _no_real_gcs_signing(monkeypatch):
    """No real GCS/IAM calls in this test suite — deterministic fake by default.

    Individual tests (e.g. the signing-raises test) can further override
    this via their own monkeypatch.setattr call.
    """
    monkeypatch.setattr(analysis_router, "sign_gcs_uri", _fake_sign)


@pytest.fixture(autouse=True)
def _clear_role_override():
    yield
    app.dependency_overrides.pop(get_role, None)


def _as_role(role: str) -> None:
    app.dependency_overrides[get_role] = lambda: role


@pytest.mark.parametrize("route", ROUTES)
def test_guest_gets_403(client, route):
    _as_role("guest")
    response = client.get(route)
    assert response.status_code == 403


@pytest.mark.parametrize("route", ROUTES)
def test_viewer_gets_403(client, route):
    _as_role("viewer")
    response = client.get(route)
    assert response.status_code == 403


@pytest.mark.parametrize("route", ROUTES)
def test_admin_gets_200(client, route):
    _as_role("admin")
    response = client.get(route)
    assert response.status_code == 200


def test_features_filters_by_family(client):
    _as_role("admin")
    response = client.get("/v1/analysis/features", params={"family": "shot_geometry"})
    assert response.status_code == 200
    body = response.json()
    assert {row["column_name"] for row in body} == {"shot_distance", "shot_angle"}
    assert all(row["feature_family"] == "shot_geometry" for row in body)


def test_features_unfiltered_returns_all_families(client):
    _as_role("admin")
    response = client.get("/v1/analysis/features")
    assert response.status_code == 200
    body = response.json()
    assert {row["feature_family"] for row in body} == {"shot_geometry", "opponent_adjusted"}


def test_correlation_filters_by_family_via_feature_a_or_b(client):
    _as_role("admin")
    response = client.get("/v1/analysis/correlation", params={"family": "opponent_adjusted"})
    assert response.status_code == 200
    body = response.json()
    pairs = {(row["feature_a"], row["feature_b"]) for row in body}
    # Rows involving nearest_defender_odi/pressure_odi (opponent_adjusted columns) should be
    # included, whether they appear as feature_a or feature_b.
    assert ("nearest_defender_odi", "pressure_odi") in pairs
    assert ("shot_distance", "nearest_defender_odi") in pairs
    # The row between two unrelated columns must be excluded.
    assert ("unrelated_column_a", "unrelated_column_b") not in pairs
    # The row entirely within shot_geometry must be excluded.
    assert ("shot_distance", "shot_angle") not in pairs


def test_correlation_unfiltered_returns_all_rows(client):
    _as_role("admin")
    response = client.get("/v1/analysis/correlation")
    assert response.status_code == 200
    assert len(response.json()) == 4


def test_univariate_filters_by_family(client):
    _as_role("admin")
    response = client.get("/v1/analysis/univariate", params={"family": "shot_geometry"})
    assert response.status_code == 200
    body = response.json()
    assert {row["column_name"] for row in body} == {"shot_distance", "shot_angle"}


def test_bivariate_returns_three_nested_lists(client):
    _as_role("admin")
    response = client.get("/v1/analysis/bivariate")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"candidates", "interactions", "stratified"}
    assert len(body["candidates"]) == 2
    assert len(body["interactions"]) == 1
    assert len(body["stratified"]) == 1


def test_pca_returns_two_nested_lists(client):
    _as_role("admin")
    response = client.get("/v1/analysis/pca")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"components", "loadings"}
    assert len(body["components"]) == 2
    assert len(body["loadings"]) == 2


def test_charts_with_no_run_id_returns_latest(client):
    _as_role("admin")
    response = client.get("/v1/analysis/charts")
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "cxg-analysis-20260822T014705Z"
    assert len(body["charts"]) == 2
    assert all(c["html_uri"].startswith("gs://bucket/new/") for c in body["charts"])
    # Router wires signed_html_url/signed_png_url from sign_gcs_uri's return
    # value end to end — raw URIs stay present alongside the signed ones.
    for chart in body["charts"]:
        assert chart["signed_html_url"] == f"https://signed.example/{chart['html_uri']}"
        assert chart["signed_png_url"] == f"https://signed.example/{chart['png_uri']}"


def test_charts_with_explicit_older_run_id(client):
    _as_role("admin")
    response = client.get("/v1/analysis/charts", params={"run_id": "cxg-analysis-20260810T010000Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "cxg-analysis-20260810T010000Z"
    assert len(body["charts"]) == 2
    assert all(c["html_uri"].startswith("gs://bucket/old/") for c in body["charts"])


def test_charts_returns_200_and_null_signed_urls_when_signing_raises(client, monkeypatch):
    """Hard gate 3: one chart's signing failure must not 500 the whole endpoint."""

    def _boom(_uri):
        raise RuntimeError("signing exploded")

    monkeypatch.setattr(analysis_router, "sign_gcs_uri", _boom)

    _as_role("admin")
    response = client.get("/v1/analysis/charts")

    assert response.status_code == 200
    body = response.json()
    assert len(body["charts"]) == 2
    for chart in body["charts"]:
        assert chart["signed_html_url"] is None
        assert chart["signed_png_url"] is None
        # The raw URIs are still present — signing failure doesn't hide them.
        assert chart["html_uri"]
