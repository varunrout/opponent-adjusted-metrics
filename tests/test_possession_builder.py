from datetime import datetime
from pathlib import Path

from opponent_adjusted.db.models import Competition, Event, Match, Possession, RawEvent, Team
from opponent_adjusted.db.session import session_scope
from scripts.build_possessions import build_possessions
from scripts.report_ingestion_status import build_report


MAKEFILE_PATH = Path("Makefile")


def _seed_normalized_events() -> tuple[int, int]:
    with session_scope() as session:
        competition = Competition(
            statsbomb_competition_id=1,
            name="Fixture League",
            season="2026",
        )
        home = Team(statsbomb_team_id=10, name="Home")
        away = Team(statsbomb_team_id=20, name="Away")
        session.add_all([competition, home, away])
        session.flush()

        match = Match(
            statsbomb_match_id=100,
            competition_id=competition.id,
            home_team_id=home.id,
            away_team_id=away.id,
            match_date=datetime(2026, 1, 1),
            season="2026",
        )
        session.add(match)
        session.flush()

        raw_events = [
            RawEvent(
                match_id=match.id,
                statsbomb_event_id="event-1",
                raw_json={
                    "possession": 7,
                    "possession_team": {"id": 10, "name": "Home"},
                    "team": {"id": 10, "name": "Home"},
                },
                type="Pass",
                period=1,
                minute=1,
                second=1,
            ),
            RawEvent(
                match_id=match.id,
                statsbomb_event_id="event-2",
                raw_json={
                    "possession": 7,
                    "possession_team": {"id": 10, "name": "Home"},
                    "team": {"id": 20, "name": "Away"},
                },
                type="Pressure",
                period=1,
                minute=1,
                second=5,
            ),
            RawEvent(
                match_id=match.id,
                statsbomb_event_id="event-3",
                raw_json={
                    "possession": 8,
                    "possession_team": {"id": 20, "name": "Away"},
                    "team": {"id": 20, "name": "Away"},
                },
                type="Carry",
                period=1,
                minute=2,
                second=0,
            ),
        ]
        session.add_all(raw_events)
        session.flush()

        normalized_events = [
            Event(
                raw_event_id=raw_events[0].id,
                match_id=match.id,
                team_id=home.id,
                type="Pass",
                period=1,
                minute=1,
                second=1,
                possession=7,
            ),
            Event(
                raw_event_id=raw_events[1].id,
                match_id=match.id,
                team_id=away.id,
                type="Pressure",
                period=1,
                minute=1,
                second=5,
                possession=7,
            ),
            Event(
                raw_event_id=raw_events[2].id,
                match_id=match.id,
                team_id=away.id,
                type="Carry",
                period=1,
                minute=2,
                second=0,
                possession=8,
            ),
        ]
        session.add_all(normalized_events)
        session.flush()

        return home.id, away.id


def test_build_possessions_creates_rows_from_normalized_events(e2e_test_env):
    home_id, away_id = _seed_normalized_events()

    with session_scope() as session:
        summary = build_possessions(session)
        possessions = session.query(Possession).order_by(Possession.possession_number).all()

        assert summary.events_loaded == 3
        assert summary.possessions_built == 2
        assert summary.rows_inserted == 2
        assert summary.final_possession_count == 2
        assert [row.possession_number for row in possessions] == [7, 8]
        assert possessions[0].team_id == home_id
        assert possessions[0].event_count == 2
        assert possessions[0].duration_seconds == 4.0
        assert possessions[1].team_id == away_id


def test_build_possessions_is_idempotent(e2e_test_env):
    _seed_normalized_events()

    with session_scope() as session:
        first = build_possessions(session)
        second = build_possessions(session)

        assert first.final_possession_count == 2
        assert second.rows_deleted == 2
        assert second.rows_inserted == 2
        assert second.final_possession_count == 2
        assert session.query(Possession).count() == 2


def test_ingestion_report_marks_possessions_ready(e2e_test_env):
    _seed_normalized_events()

    with session_scope() as session:
        build_possessions(session)

    report = build_report()

    assert report["table_counts"]["possessions"] == 2
    assert report["readiness"]["has_possessions"] is True


def test_makefile_exposes_build_possessions_target():
    text = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "build-possessions:" in text
    assert "poetry run python scripts/build_possessions.py" in text
