"""Tests for the team shots-faced router (the defensive counterpart to /shots)."""

from tests.api.conftest import FAKE_SHOTS


def test_list_team_shots_faced_returns_only_opponent_shots(client):
    response = client.get("/v1/teams/771/shots-faced")

    assert response.status_code == 200
    body = response.json()
    # France (771) played only match 7, against Croatia (772) — shot-2 is
    # Croatia's only shot in that match.
    assert {s["event_id"] for s in body} == {"shot-2"}
    assert all(s["team_id"] == 772 for s in body)


def test_list_team_shots_faced_never_returns_the_teams_own_shots(client):
    response = client.get("/v1/teams/772/shots-faced")

    assert response.status_code == 200
    body = response.json()
    expected = {s.event_id for s in FAKE_SHOTS if s.match_id == 7 and s.team_id == 771}
    assert {s["event_id"] for s in body} == expected
    assert all(s["team_id"] == 771 for s in body)


def test_list_team_shots_faced_returns_empty_for_team_not_in_any_match(client):
    response = client.get("/v1/teams/999999/shots-faced")

    assert response.status_code == 200
    assert response.json() == []


def test_list_team_shots_faced_filters_by_competition_id_no_match(client):
    response = client.get("/v1/teams/771/shots-faced", params={"competition_id": 11})

    assert response.status_code == 200
    assert response.json() == []
