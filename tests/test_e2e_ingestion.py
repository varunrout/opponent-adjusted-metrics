"""End-to-end tests for StatsBomb ingestion pipeline.

These tests exercise the ingestion scripts against the bundled
`data/statsbomb` subset using a lightweight SQLite database. The goal
is to validate that our wiring from config -> DB -> ingestion scripts
works on a minimal realistic dataset.
"""

from pathlib import Path

import pytest

from scripts.ingest_competitions import ingest_competitions
from scripts.ingest_matches import ingest_matches
from scripts.ingest_events import ingest_events

from opponent_adjusted.db.models import Competition, Match, RawEvent
from opponent_adjusted.db.session import SessionLocal


@pytest.mark.e2e
def test_ingestion_end_to_end(e2e_test_env):
	"""Run competitions -> matches -> events ingestion on a tiny subset.

	Assumes that `data/statsbomb` contains at least one competition,
	one match, and one events file from the StatsBomb open-data
	subset already checked into the repository.
	"""

	# Sanity check: statsbomb data directory should exist under the
	# temporary data root that the fixture configured.
	data_root = e2e_test_env
	statsbomb_dir = Path("data") / "statsbomb"
	assert statsbomb_dir.exists(), "Expected bundled StatsBomb subset under data/statsbomb"

	# Run the three ingestion stages
	ingest_competitions()
	ingest_matches()
	# Limit to a single match to keep runtime bounded in CI
	ingest_events(limit=1)

	# Verify that core tables have been populated
	with SessionLocal() as session:
		num_competitions = session.query(Competition).count()
		num_matches = session.query(Match).count()
		num_events = session.query(RawEvent).count()

	# We expect at least one row in each table when the bundled
	# StatsBomb data is present.
	assert num_competitions > 0, "No competitions ingested"
	assert num_matches > 0, "No matches ingested"
	assert num_events > 0, "No raw events ingested"