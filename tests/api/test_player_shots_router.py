"""Tests for the player shots router."""

from tests.api.conftest import FAKE_SHOTS


def test_list_player_shots_returns_only_that_players_shots(client):
    response = client.get("/v1/players/3010/shots")

    assert response.status_code == 200
    body = response.json()
    expected = [s for s in FAKE_SHOTS if s.player_id == 3010]
    assert len(body) == len(expected) == 2
    assert {s["event_id"] for s in body} == {"shot-1", "shot-4"}

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


def test_list_player_shots_filters_by_competition_id(client):
    response = client.get("/v1/players/4001/shots", params={"competition_id": 11})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["event_id"] == "shot-5"


def test_list_player_shots_filters_by_competition_id_no_match(client):
    response = client.get("/v1/players/4001/shots", params={"competition_id": 43})

    assert response.status_code == 200
    assert response.json() == []


def test_list_player_shots_returns_empty_for_unknown_player(client):
    response = client.get("/v1/players/999999/shots")

    assert response.status_code == 200
    assert response.json() == []
