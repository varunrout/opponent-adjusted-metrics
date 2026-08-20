"""Tests for the shots router."""

from tests.api.conftest import FAKE_SHOTS


def test_list_shots_returns_expected_shots(client):
    response = client.get("/v1/matches/7/shots")

    assert response.status_code == 200
    body = response.json()
    expected = [s for s in FAKE_SHOTS if s.match_id == 7]
    assert len(body) == len(expected)
    assert {s["event_id"] for s in body} == {s.event_id for s in expected}

    first = next(s for s in body if s["event_id"] == "shot-1")
    assert first["is_goal"] is True
    assert first["statsbomb_xg"] == 0.32
    assert first["player_name"] == "Kylian Mbappé"
    assert set(first.keys()) == {
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

    third = next(s for s in body if s["event_id"] == "shot-3")
    assert third["is_goal"] is False
    assert third["player_name"] is None
    assert third["end_x"] is None


def test_list_shots_returns_empty_for_match_with_no_shots(client):
    response = client.get("/v1/matches/999/shots")

    assert response.status_code == 200
    assert response.json() == []
