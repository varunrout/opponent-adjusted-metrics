"""Script to ingest matches for filtered competitions from StatsBomb Open Data."""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Competition, Match, Team
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.ingestion.statsbomb_io import StatsBombLoader
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def _get_or_create_team(session, team_dict: dict, role: str) -> Team:
    """Get or create a Team from a StatsBomb match team block.

    Handles home_team / away_team naming differences.
    """
    # StatsBomb match JSON uses home_team_id / away_team_id, not nested id
    team_id = (
        team_dict.get(f"{role}_team_id")
        or team_dict.get("team_id")
        or team_dict.get("id")
    )
    name = (
        team_dict.get(f"{role}_team_name")
        or team_dict.get("team_name")
        or team_dict.get("name")
    )

    if team_id is None or name is None:
        raise ValueError(f"Missing team identifiers for role={role}: {team_dict.keys()}")

    team = session.query(Team).filter_by(statsbomb_team_id=team_id).first()
    if team:
        # Optionally update name if changed
        if name and team.name != name:
            team.name = name
        return team

    team = Team(statsbomb_team_id=team_id, name=name, country=None)
    session.add(team)
    session.flush()
    return team


def ingest_matches() -> None:
    logger.info("Starting matches ingestion...")

    loader = StatsBombLoader()
    competitions = loader.discover_competitions()

    if not competitions:
        logger.warning(
            "No competitions discovered. Make sure StatsBomb data is available at: %s",
            settings.statsbomb_data_path,
        )
        return

    inserted = 0
    skipped = 0

    with session_scope() as session:
        for comp in competitions:
            comp_id = comp["competition_id"]
            season_id = comp["season_id"]
            season_name = comp.get("season_name", "")
            comp_row = (
                session.query(Competition)
                .filter_by(statsbomb_competition_id=comp_id, season=season_name)
                .first()
            )

            if not comp_row:
                logger.warning(
                    "Competition row not found yet for %s %s. Run ingest_competitions first.",
                    comp.get("competition_name"),
                    season_name,
                )
                # Continue anyway; we can still stage matches if desired by creating the comp
                comp_row = Competition(
                    statsbomb_competition_id=comp_id,
                    name=comp.get("competition_name", str(comp_id)),
                    season=season_name,
                )
                session.add(comp_row)
                session.flush()

            matches = loader.load_matches(comp_id, season_id)
            if not matches:
                logger.warning(
                    "No matches file found for competition_id=%s season_id=%s", comp_id, season_id
                )
                continue

            for m in matches:
                sb_match_id = m.get("match_id")
                # Deduplicate
                existing = (
                    session.query(Match).filter_by(statsbomb_match_id=sb_match_id).first()
                )
                if existing:
                    skipped += 1
                    continue

                try:
                    home_team = _get_or_create_team(session, m.get("home_team", {}), "home")
                    away_team = _get_or_create_team(session, m.get("away_team", {}), "away")
                except ValueError as e:
                    logger.warning(f"Skipping match {sb_match_id} due to team parse error: {e}")
                    skipped += 1
                    continue

                # Parse optional dates
                match_date = None
                kickoff = None
                try:
                    # StatsBomb stores YYYY-MM-DD
                    if m.get("match_date"):
                        match_date = datetime.fromisoformat(m["match_date"])  # type: ignore[arg-type]
                except Exception:
                    pass
                try:
                    # Some datasets have 'kick_off' as HH:MM:SS
                    if m.get("kick_off"):
                        # Use today's date just to keep it as a datetime if time only
                        kickoff = datetime.fromisoformat(
                            f"1970-01-01T{m['kick_off']}"
                        )
                except Exception:
                    pass

                row = Match(
                    statsbomb_match_id=sb_match_id,
                    competition_id=comp_row.id,
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    kickoff_time=kickoff,
                    match_date=match_date,
                    season=season_name,
                )
                session.add(row)
                inserted += 1

    logger.info("Matches ingestion complete. Inserted: %s, Skipped: %s", inserted, skipped)


if __name__ == "__main__":
    ingest_matches()
