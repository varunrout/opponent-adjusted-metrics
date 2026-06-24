"""Quality checks for deterministic fixture-backed StatsBomb ingestion."""

from scripts.ingest_competitions import ingest_competitions
from scripts.ingest_events import ingest_events
from scripts.ingest_matches import ingest_matches
from scripts.report_ingestion_status import build_report

from opponent_adjusted.db.models import Competition, Match, RawEvent
from opponent_adjusted.db.session import SessionLocal


def _run_fixture_ingestion() -> None:
    ingest_competitions()
    ingest_matches()
    ingest_events(limit=1)


def _ingestion_counts() -> tuple[int, int, int]:
    with SessionLocal() as session:
        return (
            session.query(Competition).count(),
            session.query(Match).count(),
            session.query(RawEvent).count(),
        )


def test_ingestion_is_idempotent(e2e_test_env):
    """Running fixture ingestion twice should not duplicate DB rows."""
    _run_fixture_ingestion()
    first_counts = _ingestion_counts()

    _run_fixture_ingestion()
    second_counts = _ingestion_counts()

    assert first_counts == (1, 1, 3)
    assert second_counts == first_counts


def test_ingestion_status_report_readiness_and_event_types(e2e_test_env):
    """The status report should reflect fixture ingestion readiness."""
    _run_fixture_ingestion()

    report = build_report()

    assert report["table_counts"]["competitions"] == 1
    assert report["table_counts"]["matches"] == 1
    assert report["table_counts"]["raw_events"] == 3
    assert report["readiness"]["has_competitions"] is True
    assert report["readiness"]["has_matches"] is True
    assert report["readiness"]["has_raw_events"] is True
    assert report["readiness"]["has_normalized_events"] is False
    assert {row["event_type"] for row in report["event_type_counts"]} >= {"Pass", "Shot"}
