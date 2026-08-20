"""Fixtures for API router tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api.dependencies import get_store
from opponent_adjusted.api.interfaces import CompetitionRecord, MatchRecord
from opponent_adjusted.api.main import app


class FakeServingStore:
    """In-memory ServingStore fake for tests."""

    def __init__(
        self,
        competitions: list[CompetitionRecord],
        matches: list[MatchRecord],
    ) -> None:
        self._competitions = competitions
        self._matches = matches

    def list_competitions(self) -> list[CompetitionRecord]:
        return list(self._competitions)

    def list_matches(
        self,
        *,
        competition_id: int | None = None,
        season_id: int | None = None,
    ) -> list[MatchRecord]:
        matches = self._matches
        if competition_id is not None:
            matches = [m for m in matches if m.competition_id == competition_id]
        if season_id is not None:
            matches = [m for m in matches if m.season_id == season_id]
        return list(matches)


FAKE_COMPETITIONS = [
    CompetitionRecord(
        competition_id=43,
        season_id=3,
        competition_name="FIFA World Cup",
        competition_gender="male",
        country_name="International",
        season_name="2018",
        match_updated="2021-01-01T00:00:00",
        match_available="2021-01-01T00:00:00",
        match_updated_360=None,
        match_available_360=None,
    ),
    CompetitionRecord(
        competition_id=11,
        season_id=1,
        competition_name="La Liga",
        competition_gender="male",
        country_name="Spain",
        season_name="2020/2021",
        match_updated="2021-02-01T00:00:00",
        match_available="2021-02-01T00:00:00",
        match_updated_360="2021-02-02T00:00:00",
        match_available_360="2021-02-02T00:00:00",
    ),
    CompetitionRecord(
        competition_id=72,
        season_id=30,
        competition_name="Women's World Cup",
        competition_gender="female",
        country_name="International",
        season_name="2023",
        match_updated="2023-08-01T00:00:00",
        match_available="2023-08-01T00:00:00",
        match_updated_360=None,
        match_available_360=None,
    ),
]

FAKE_MATCHES = [
    MatchRecord(
        match_id=7,
        competition_id=43,
        season_id=3,
        match_date="2018-07-15",
        kick_off="17:00:00.000",
        home_team_id=771,
        home_team_name="France",
        away_team_id=772,
        away_team_name="Croatia",
        home_score=4,
        away_score=2,
        competition_stage="Final",
        stadium="Luzhniki Stadium",
        referee="Nestor Pitana",
        match_status="available",
        match_status_360=None,
        last_updated="2021-01-01T00:00:00",
        last_updated_360=None,
    ),
    MatchRecord(
        match_id=8,
        competition_id=11,
        season_id=1,
        match_date="2021-01-10",
        kick_off="20:00:00.000",
        home_team_id=217,
        home_team_name="Barcelona",
        away_team_id=220,
        away_team_name="Real Madrid",
        home_score=1,
        away_score=3,
        competition_stage="Regular Season",
        stadium="Camp Nou",
        referee="Antonio Mateu Lahoz",
        match_status="available",
        match_status_360="available",
        last_updated="2021-02-01T00:00:00",
        last_updated_360="2021-02-02T00:00:00",
    ),
    MatchRecord(
        match_id=9,
        competition_id=72,
        season_id=30,
        match_date="2023-08-20",
        kick_off="12:00:00.000",
        home_team_id=901,
        home_team_name="Spain",
        away_team_id=902,
        away_team_name="England",
        home_score=1,
        away_score=0,
        competition_stage="Final",
        stadium="Stadium Australia",
        referee="Tori Penso",
        match_status="available",
        match_status_360=None,
        last_updated="2023-08-21T00:00:00",
        last_updated_360=None,
    ),
]


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_store] = lambda: FakeServingStore(
        competitions=FAKE_COMPETITIONS, matches=FAKE_MATCHES
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
