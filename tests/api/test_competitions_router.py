"""Tests for the competitions router."""

from tests.api.conftest import FAKE_COMPETITIONS


def test_list_competitions_returns_expected_rows(client):
    response = client.get("/v1/competitions")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(FAKE_COMPETITIONS)

    first = body[0]
    assert first["competition_id"] == FAKE_COMPETITIONS[0].competition_id
    assert first["season_id"] == FAKE_COMPETITIONS[0].season_id
    assert first["competition_name"] == FAKE_COMPETITIONS[0].competition_name
    assert set(first.keys()) == {
        "competition_id",
        "season_id",
        "competition_name",
        "competition_gender",
        "country_name",
        "season_name",
        "match_updated",
        "match_available",
        "match_updated_360",
        "match_available_360",
    }
