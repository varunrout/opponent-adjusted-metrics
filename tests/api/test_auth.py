"""Tests for auth resolution: get_auth_context / get_role / GET /v1/me."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api.dependencies import DecodedToken, get_token_verifier
from opponent_adjusted.api.main import app


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.pop(get_token_verifier, None)


def test_me_no_authorization_header_returns_guest(client: TestClient) -> None:
    response = client.get("/v1/me")
    assert response.status_code == 200
    assert response.json() == {"role": "guest", "uid": None, "email": None}


def test_me_valid_token_returns_admin(client: TestClient) -> None:
    def fake_verifier(token: str) -> DecodedToken:
        assert token == "good-token"
        return DecodedToken(uid="u1", email="a@b.com", role="admin")

    app.dependency_overrides[get_token_verifier] = lambda: fake_verifier

    response = client.get("/v1/me", headers={"Authorization": "Bearer good-token"})
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "admin"
    assert body["uid"] == "u1"
    assert body["email"] == "a@b.com"


def test_me_invalid_token_returns_401(client: TestClient) -> None:
    def fake_verifier(token: str) -> DecodedToken:
        raise ValueError("bad token")

    app.dependency_overrides[get_token_verifier] = lambda: fake_verifier

    response = client.get("/v1/me", headers={"Authorization": "Bearer bad-token"})
    assert response.status_code == 401


def test_me_malformed_header_returns_401_without_calling_verifier(client: TestClient) -> None:
    def fake_verifier(token: str) -> DecodedToken:
        raise AssertionError("verifier should not be called for a malformed header")

    app.dependency_overrides[get_token_verifier] = lambda: fake_verifier

    response = client.get("/v1/me", headers={"Authorization": "NotBearer xyz"})
    assert response.status_code == 401


def test_me_unrecognized_role_falls_back_to_guest(client: TestClient) -> None:
    def fake_verifier(token: str) -> DecodedToken:
        return DecodedToken(uid="u2", email="c@d.com", role="superuser")

    app.dependency_overrides[get_token_verifier] = lambda: fake_verifier

    response = client.get("/v1/me", headers={"Authorization": "Bearer some-token"})
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "guest"
    assert body["uid"] == "u2"
    assert body["email"] == "c@d.com"


def test_public_route_still_accessible_as_guest_without_token(client: TestClient) -> None:
    response = client.get("/v1/matches")
    assert response.status_code == 200
