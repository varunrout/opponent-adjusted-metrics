"""Tests for the teams router."""

import pytest


def test_list_teams_returns_expected_shape(client):
    response = client.get("/v1/teams")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 4

    france = next(t for t in body if t["team_id"] == 771)
    assert france["team_name"] == "France"
    assert france["shots"] == 3
    assert france["goals"] == 1
    assert france["total_xg"] == pytest.approx(0.45)
    assert set(france.keys()) == {
        "team_id",
        "team_name",
        "shots",
        "goals",
        "total_xg",
    }


def test_list_teams_sorted_by_total_xg_descending(client):
    response = client.get("/v1/teams")

    assert response.status_code == 200
    body = response.json()
    xg_values = [t["total_xg"] for t in body]
    assert xg_values == sorted(xg_values, reverse=True)
    assert body[0]["team_id"] == 217


def test_list_teams_filters_by_competition_id(client):
    response = client.get("/v1/teams", params={"competition_id": 43})

    assert response.status_code == 200
    body = response.json()
    assert {t["team_id"] for t in body} == {771, 772}


def test_list_teams_filters_by_season_id(client):
    response = client.get("/v1/teams", params={"season_id": 1})

    assert response.status_code == 200
    body = response.json()
    assert {t["team_id"] for t in body} == {217}
