"""Tests for the players router."""


def test_list_players_returns_expected_shape(client):
    response = client.get("/v1/players")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 5

    mbappe = next(p for p in body if p["player_id"] == 3010)
    assert mbappe["player_name"] == "Kylian Mbappé"
    assert mbappe["team_id"] == 771
    assert mbappe["team_name"] == "France"
    assert mbappe["shots"] == 2
    assert mbappe["goals"] == 1
    assert mbappe["total_xg"] == 0.4
    assert set(mbappe.keys()) == {
        "player_id",
        "player_name",
        "team_id",
        "team_name",
        "shots",
        "goals",
        "total_xg",
    }


def test_list_players_sorted_by_total_xg_descending(client):
    response = client.get("/v1/players")

    assert response.status_code == 200
    body = response.json()
    xg_values = [p["total_xg"] for p in body]
    assert xg_values == sorted(xg_values, reverse=True)
    assert body[0]["player_id"] == 4001


def test_list_players_filters_by_competition_id(client):
    response = client.get("/v1/players", params={"competition_id": 43})

    assert response.status_code == 200
    body = response.json()
    assert {p["player_id"] for p in body} == {3009, 3010, 3011}


def test_list_players_filters_by_season_id(client):
    response = client.get("/v1/players", params={"season_id": 1})

    assert response.status_code == 200
    body = response.json()
    assert {p["player_id"] for p in body} == {4001}
