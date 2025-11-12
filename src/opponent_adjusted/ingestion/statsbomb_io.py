"""StatsBomb data I/O utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from opponent_adjusted.config import settings
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


class StatsBombLoader:
    """Loader for StatsBomb Open Data."""

    def __init__(self, data_path: Optional[Path] = None):
        """Initialize the StatsBomb loader.

        Args:
            data_path: Path to StatsBomb data directory
        """
        self.data_path = data_path or settings.statsbomb_data_path
        self.competitions_file = self.data_path / "competitions.json"
        self.matches_dir = self.data_path / "matches"
        self.events_dir = self.data_path / "events"

    def load_competitions(self) -> List[Dict[str, Any]]:
        """Load competitions from StatsBomb data.

        Returns:
            List of competition dictionaries
        """
        if not self.competitions_file.exists():
            logger.warning(f"Competitions file not found: {self.competitions_file}")
            return []

        with open(self.competitions_file, "r", encoding="utf-8") as f:
            competitions = json.load(f)

        logger.info(f"Loaded {len(competitions)} competitions")
        return competitions

    def load_matches(self, competition_id: int, season_id: int) -> List[Dict[str, Any]]:
        """Load matches for a specific competition and season.

        Args:
            competition_id: StatsBomb competition ID
            season_id: StatsBomb season ID

        Returns:
            List of match dictionaries
        """
        matches_file = self.matches_dir / f"{competition_id}" / f"{season_id}.json"

        if not matches_file.exists():
            logger.warning(f"Matches file not found: {matches_file}")
            return []

        with open(matches_file, "r", encoding="utf-8") as f:
            matches = json.load(f)

        logger.info(
            f"Loaded {len(matches)} matches for competition {competition_id}, season {season_id}"
        )
        return matches

    def load_events(self, match_id: int) -> List[Dict[str, Any]]:
        """Load events for a specific match.

        Args:
            match_id: StatsBomb match ID

        Returns:
            List of event dictionaries
        """
        events_file = self.events_dir / f"{match_id}.json"

        if not events_file.exists():
            logger.warning(f"Events file not found: {events_file}")
            return []

        with open(events_file, "r", encoding="utf-8") as f:
            events = json.load(f)

        logger.debug(f"Loaded {len(events)} events for match {match_id}")
        return events

    def discover_competitions(
        self, competition_filters: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Discover competitions matching filters.

        Args:
            competition_filters: List of dicts with 'name' and 'season' keys

        Returns:
            List of filtered competition dictionaries
        """
        all_competitions = self.load_competitions()

        if not competition_filters:
            competition_filters = settings.competitions

        filtered = []
        for comp in all_competitions:
            comp_name = comp.get("competition_name", "")
            season_name = comp.get("season_name", "")

            for filter_comp in competition_filters:
                if (
                    filter_comp["name"] in comp_name
                    and filter_comp["season"] in season_name
                ):
                    filtered.append(comp)
                    break

        logger.info(
            f"Discovered {len(filtered)} competitions matching filters: {competition_filters}"
        )
        return filtered

    def discover_all_matches(
        self, competition_filters: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Discover all matches for filtered competitions.

        Args:
            competition_filters: List of dicts with 'name' and 'season' keys

        Returns:
            List of all match dictionaries
        """
        competitions = self.discover_competitions(competition_filters)
        all_matches = []

        for comp in competitions:
            comp_id = comp["competition_id"]
            season_id = comp["season_id"]
            matches = self.load_matches(comp_id, season_id)

            # Add competition info to each match
            for match in matches:
                match["_competition_id"] = comp_id
                match["_season_id"] = season_id
                match["_competition_name"] = comp.get("competition_name")
                match["_season_name"] = comp.get("season_name")

            all_matches.extend(matches)

        logger.info(f"Discovered {len(all_matches)} total matches")
        return all_matches


def extract_shot_info(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract shot information from an event.

    Args:
        event: Event dictionary

    Returns:
        Shot information dictionary or None if not a shot
    """
    if event.get("type", {}).get("name") != "Shot":
        return None

    shot_data = event.get("shot", {})
    location = event.get("location", [None, None])

    return {
        "statsbomb_xg": shot_data.get("statsbomb_xg"),
        "body_part": shot_data.get("body_part", {}).get("name"),
        "technique": shot_data.get("technique", {}).get("name"),
        "shot_type": shot_data.get("type", {}).get("name"),
        "outcome": shot_data.get("outcome", {}).get("name"),
        "first_time": shot_data.get("first_time", False),
        "location_x": location[0] if len(location) > 0 else None,
        "location_y": location[1] if len(location) > 1 else None,
    }


def extract_event_location(event: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Extract location from event.

    Args:
        event: Event dictionary

    Returns:
        Tuple of (x, y) coordinates or (None, None)
    """
    location = event.get("location", [None, None])
    if len(location) >= 2:
        return location[0], location[1]
    return None, None


def is_defensive_action(event_type: str) -> bool:
    """Check if event type is a defensive action.

    Args:
        event_type: Event type name

    Returns:
        True if defensive action
    """
    defensive_types = {
        "Pressure",
        "Block",
        "Interception",
        "Clearance",
        "Duel",
        "Tackle",
        "Foul Committed",
        "Shield",
    }
    return event_type in defensive_types
