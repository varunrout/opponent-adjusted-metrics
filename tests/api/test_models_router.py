"""Tests for the public /v1/models router — the Models-page mirror of
/v1/analysis/cxg-models* (routers/analysis.py stays admin-only, unchanged).
"""

from __future__ import annotations

import pytest

from opponent_adjusted.api.dependencies import get_role
from opponent_adjusted.api.main import app


@pytest.fixture(autouse=True)
def _clear_role_override():
    yield
    app.dependency_overrides.pop(get_role, None)


def _as_role(role: str) -> None:
    app.dependency_overrides[get_role] = lambda: role


@pytest.mark.parametrize("role", ["guest", "viewer", "admin"])
def test_cxg_models_is_accessible_to_every_role(client, role):
    _as_role(role)
    response = client.get("/v1/models/cxg-models")
    assert response.status_code == 200


def test_cxg_models_no_auth_at_all_still_succeeds(client):
    # No dependency override at all — exercises the real get_role default
    # (guest), matching an unauthenticated curl with no Authorization header.
    response = client.get("/v1/models/cxg-models")
    assert response.status_code == 200


def test_cxg_models_returns_all_frozen_model_rows(client):
    response = client.get("/v1/models/cxg-models")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 6
    model_keys = {row["model_key"] for row in body}
    assert model_keys == {"event_v3", "plus_v3", "baseline_v1", "plus_v2"}


@pytest.mark.parametrize("role", ["guest", "viewer", "admin"])
def test_cxg_model_coefficients_is_accessible_to_every_role(client, role):
    _as_role(role)
    response = client.get("/v1/models/cxg-models/event_v3/coefficients")
    assert response.status_code == 200


def test_cxg_model_coefficients_returns_expected_rows(client):
    response = client.get("/v1/models/cxg-models/event_v3/coefficients")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {row["feature"] for row in body} == {"const", "shot_x_sb"}
    assert all(row["model_key"] == "event_v3" for row in body)


def test_cxg_model_coefficients_for_different_model_key_returns_different_rows(client):
    response = client.get("/v1/models/cxg-models/plus_v3/coefficients")
    assert response.status_code == 200
    body = response.json()
    assert {row["feature"] for row in body} == {"const", "nearest_defender_zone_displacement"}
    assert all(row["model_key"] == "plus_v3" for row in body)


def test_cxg_model_coefficients_unknown_model_key_returns_404(client):
    response = client.get("/v1/models/cxg-models/not-a-real-model/coefficients")
    assert response.status_code == 404


def test_admin_only_analysis_route_is_unaffected_still_403_for_guest(client):
    """Guardrail: this new router must not have loosened routers/analysis.py's
    own admin gate on the same underlying data."""
    _as_role("guest")
    response = client.get("/v1/analysis/cxg-models")
    assert response.status_code == 403
