"""End-to-end tests for the fixture-backed StatsBomb ingestion pipeline."""

import pytest

from scripts.ingest_competitions import ingest_competitions
from scripts.ingest_events import ingest_events
from scripts.ingest_matches import ingest_matches

from opponent_adjusted.db.models import Competition, Match, RawEvent
from opponent_adjusted.db.session import SessionLocal


@pytest.mark.e2e
def test_ingestion_end_to_end(e2e_test_env):
    """Run competitions -> matches -> events ingestion on tiny committed fixtures."""

    statsbomb_dir = e2e_test_env / "statsbomb"
    assert statsbomb_dir.exists(), "Expected fixture StatsBomb subset under the temp data root"

    ingest_competitions()
    ingest_matches()
    ingest_events(limit=1)

    with SessionLocal() as session:
        num_competitions = session.query(Competition).count()
        num_matches = session.query(Match).count()
        num_events = session.query(RawEvent).count()

    assert num_competitions == 1
    assert num_matches == 1
    assert num_events == 3
