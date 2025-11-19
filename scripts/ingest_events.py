"""Script to ingest raw events for all ingested matches.

Optional flags:
    --limit N   Only process first N matches (useful for quick tests)
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.db.models import Match, RawEvent
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.ingestion.statsbomb_io import StatsBombLoader
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_events(limit: int | None = None) -> None:
    logger.info("Starting events ingestion%s...", f" (limit={limit})" if limit else "")
    loader = StatsBombLoader()

    inserted = 0
    skipped = 0

    with session_scope() as session:
        q = session.query(Match)
        if limit:
            q = q.limit(limit)
        matches = q.all()
        if not matches:
            logger.warning("No matches found in DB. Run ingest_matches first.")
            return

        for match in matches:
            events = loader.load_events(match.statsbomb_match_id)
            if not events:
                logger.warning(
                    "No events for match_id=%s (file missing?)", match.statsbomb_match_id
                )
                continue

            for i, ev in enumerate(events, start=1):
                ev_id = ev.get("id")
                ev_type = ev.get("type", {}).get("name", "Unknown")
                period = ev.get("period", 0)
                minute = ev.get("minute", 0)
                second = ev.get("second", 0)

                # Deduplicate raw event
                existing = (
                    session.query(RawEvent)
                    .filter_by(match_id=match.id, statsbomb_event_id=ev_id)
                    .first()
                )
                if existing:
                    skipped += 1
                    continue

                raw_row = RawEvent(
                    match_id=match.id,
                    statsbomb_event_id=str(ev_id),
                    raw_json=ev,
                    type=ev_type,
                    period=period,
                    minute=minute,
                    second=second,
                )
                session.add(raw_row)
                inserted += 1
                if inserted % 1000 == 0:
                    session.flush()
                    logger.info("Inserted %d raw events so far...", inserted)

    logger.info("Events ingestion complete. Inserted: %s, Skipped: %s", inserted, skipped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit number of matches")
    args = parser.parse_args()
    ingest_events(limit=args.limit)
