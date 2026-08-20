"""Tests for the match detail router."""

from tests.api.conftest import FAKE_LINEUPS, FAKE_MATCHES


def test_get_match_returns_match_with_lineups(client):
    response = client.get("/v1/matches/7")

    assert response.status_code == 200
    body = response.json()
    assert body["match_id"] == 7
    assert body["home_team_name"] == "France"
    assert len(body["lineups"]) == len([row for row in FAKE_LINEUPS if row.match_id == 7])

    first = body["lineups"][0]
    assert set(first.keys()) == {
        "team_id",
        "team_name",
        "formation",
        "player_id",
        "player_name",
        "position_name",
        "jersey_number",
    }


def test_get_match_returns_404_for_nonexistent_match(client):
    response = client.get("/v1/matches/99999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Match not found"


def test_get_match_matches_list_matches_fields(client):
    response = client.get("/v1/matches/7")

    body = response.json()
    expected = next(m for m in FAKE_MATCHES if m.match_id == 7)
    assert body["competition_id"] == expected.competition_id
    assert body["away_team_name"] == expected.away_team_name
