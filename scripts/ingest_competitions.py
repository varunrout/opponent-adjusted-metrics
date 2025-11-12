"""Script to ingest competitions from StatsBomb Open Data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Competition
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.ingestion.statsbomb_io import StatsBombLoader
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_competitions():
    """Ingest competitions from StatsBomb Open Data."""
    logger.info("Starting competition ingestion...")

    # Initialize loader
    loader = StatsBombLoader()

    # Discover competitions matching our filters
    competitions = loader.discover_competitions()

    if not competitions:
        logger.warning(
            "No competitions found. Make sure StatsBomb data is downloaded to: "
            f"{settings.statsbomb_data_path}"
        )
        logger.info(
            "To get StatsBomb data, clone: https://github.com/statsbomb/open-data"
        )
        return

    # Insert competitions
    inserted = 0
    skipped = 0

    with session_scope() as session:
        for comp in competitions:
            comp_id = comp["competition_id"]
            season_id = comp["season_id"]
            comp_name = comp.get("competition_name", "")
            season_name = comp.get("season_name", "")

            # Check if already exists
            existing = (
                session.query(Competition)
                .filter_by(statsbomb_competition_id=comp_id, season=season_name)
                .first()
            )

            if existing:
                skipped += 1
                logger.debug(f"Competition already exists: {comp_name} {season_name}")
                continue

            # Create new competition
            new_comp = Competition(
                statsbomb_competition_id=comp_id,
                name=comp_name,
                season=season_name,
            )
            session.add(new_comp)
            inserted += 1
            logger.info(f"Inserted: {comp_name} {season_name}")

    logger.info(f"Competition ingestion complete. Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    ingest_competitions()
