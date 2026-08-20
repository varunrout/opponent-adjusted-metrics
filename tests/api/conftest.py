"""Fixtures for API router tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from opponent_adjusted.api.dependencies import get_store
from opponent_adjusted.api.interfaces import (
    CompetitionRecord,
    LineupPlayerRecord,
    MatchRecord,
    ShotRecord,
)
from opponent_adjusted.api.main import app


class FakeServingStore:
    """In-memory ServingStore fake for tests."""

    def __init__(
        self,
        competitions: list[CompetitionRecord],
        matches: list[MatchRecord],
        lineups: list[LineupPlayerRecord] | None = None,
        shots: list[ShotRecord] | None = None,
    ) -> None:
        self._competitions = competitions
        self._matches = matches
        self._lineups = lineups or []
        self._shots = shots or []

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

    def get_match(self, match_id: int) -> MatchRecord | None:
        for match in self._matches:
            if match.match_id == match_id:
                return match
        return None

    def list_lineups(self, match_id: int) -> list[LineupPlayerRecord]:
        return [row for row in self._lineups if row.match_id == match_id]

    def list_shots(self, match_id: int) -> list[ShotRecord]:
        return [row for row in self._shots if row.match_id == match_id]


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


FAKE_LINEUPS = [
    LineupPlayerRecord(
        match_id=7,
        team_id=771,
        team_name="France",
        formation=4231,
        player_id=3009,
        player_name="Hugo Lloris",
        position_name="Goalkeeper",
        jersey_number=1,
    ),
    LineupPlayerRecord(
        match_id=7,
        team_id=771,
        team_name="France",
        formation=4231,
        player_id=3010,
        player_name="Kylian Mbappé",
        position_name="Right Wing",
        jersey_number=10,
    ),
    LineupPlayerRecord(
        match_id=7,
        team_id=772,
        team_name="Croatia",
        formation=4141,
        player_id=3011,
        player_name="Luka Modrić",
        position_name="Center Attacking Midfield",
        jersey_number=10,
    ),
]

FAKE_SHOTS = [
    ShotRecord(
        event_id="shot-1",
        match_id=7,
        team_id=771,
        player_id=3010,
        player_name="Kylian Mbappé",
        minute=38,
        period=1,
        location_x=110.0,
        location_y=40.0,
        end_x=120.0,
        end_y=39.0,
        statsbomb_xg=0.32,
        outcome_name="Goal",
        body_part_name="Right Foot",
        is_goal=True,
    ),
    ShotRecord(
        event_id="shot-2",
        match_id=7,
        team_id=772,
        player_id=3011,
        player_name="Luka Modrić",
        minute=27,
        period=1,
        location_x=105.0,
        location_y=38.0,
        end_x=118.0,
        end_y=41.0,
        statsbomb_xg=0.11,
        outcome_name="Saved",
        body_part_name="Left Foot",
        is_goal=False,
    ),
    ShotRecord(
        event_id="shot-3",
        match_id=7,
        team_id=771,
        player_id=3009,
        player_name=None,
        minute=52,
        period=2,
        location_x=100.0,
        location_y=42.0,
        end_x=None,
        end_y=None,
        statsbomb_xg=0.05,
        outcome_name="Off T",
        body_part_name="Head",
        is_goal=False,
    ),
]


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_store] = lambda: FakeServingStore(
        competitions=FAKE_COMPETITIONS,
        matches=FAKE_MATCHES,
        lineups=FAKE_LINEUPS,
        shots=FAKE_SHOTS,
    )
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()
