"""Tests for the matches router."""

from tests.api.conftest import FAKE_MATCHES


def test_list_matches_returns_all_fake_matches(client):
    response = client.get("/v1/matches")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(FAKE_MATCHES)
    assert {m["match_id"] for m in body} == {m.match_id for m in FAKE_MATCHES}


def test_list_matches_filters_by_competition_id(client):
    response = client.get("/v1/matches", params={"competition_id": 43})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["match_id"] == 7
    assert body[0]["competition_id"] == 43


def test_list_matches_filters_by_team_id_home(client):
    response = client.get("/v1/matches", params={"team_id": 771})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["match_id"] == 7


def test_list_matches_filters_by_team_id_away(client):
    response = client.get("/v1/matches", params={"team_id": 220})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["match_id"] == 8


def test_list_matches_filters_by_team_id_no_match(client):
    response = client.get("/v1/matches", params={"team_id": 999999})

    assert response.status_code == 200
    assert response.json() == []
