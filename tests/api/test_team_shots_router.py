"""Tests for the team shots router."""

from tests.api.conftest import FAKE_SHOTS


def test_list_team_shots_returns_only_that_teams_shots(client):
    response = client.get("/v1/teams/771/shots")

    assert response.status_code == 200
    body = response.json()
    expected = [s for s in FAKE_SHOTS if s.team_id == 771]
    assert len(body) == len(expected) == 3
    assert {s["event_id"] for s in body} == {"shot-1", "shot-3", "shot-4"}

    goal = next(s for s in body if s["event_id"] == "shot-1")
    assert goal["is_goal"] is True
    assert goal["statsbomb_xg"] == 0.32
    assert set(goal.keys()) == {
        "event_id",
        "match_id",
        "team_id",
        "player_id",
        "player_name",
        "minute",
        "period",
        "location_x",
        "location_y",
        "end_x",
        "end_y",
        "statsbomb_xg",
        "outcome_name",
        "body_part_name",
        "is_goal",
    }


def test_list_team_shots_filters_by_competition_id(client):
    response = client.get("/v1/teams/217/shots", params={"competition_id": 11})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["event_id"] == "shot-5"


def test_list_team_shots_filters_by_competition_id_no_match(client):
    response = client.get("/v1/teams/217/shots", params={"competition_id": 43})

    assert response.status_code == 200
    assert response.json() == []


def test_list_team_shots_returns_empty_for_unknown_team(client):
    response = client.get("/v1/teams/999999/shots")

    assert response.status_code == 200
    assert response.json() == []
