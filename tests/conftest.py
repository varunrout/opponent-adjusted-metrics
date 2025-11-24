"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_shot_event():
    """Sample shot event for testing."""
    return {
        "id": "test-event-1",
        "type": {"name": "Shot"},
        "location": [108.0, 40.0],
        "minute": 45,
        "second": 30,
        "period": 1,
        "possession": 15,
        "team": {"id": 1, "name": "Team A"},
        "player": {"id": 100, "name": "Player 1"},
        "under_pressure": True,
        "shot": {
            "statsbomb_xg": 0.15,
            "outcome": {"name": "Goal"},
            "body_part": {"name": "Right Foot"},
            "technique": {"name": "Normal"},
            "type": {"name": "Open Play"},
            "first_time": False,
        },
    }


@pytest.fixture
def sample_possession_events():
    """Sample possession events for testing."""
    return [
        {"type": "Pass", "timestamp_seconds": 0.0},
        {"type": "Carry", "timestamp_seconds": 2.0},
        {"type": "Pressure", "timestamp_seconds": 3.0},
        {"type": "Pass", "timestamp_seconds": 5.0},
        {"type": "Shot", "timestamp_seconds": 7.0},
    ]


@pytest.fixture
def sample_match_data():
    """Sample match data for testing."""
    return {
        "match_id": 123456,
        "competition": {"id": 43, "name": "FIFA World Cup"},
        "season": {"id": 3, "name": "2018"},
        "home_team": {"id": 1, "name": "Team A", "country": "Country A"},
        "away_team": {"id": 2, "name": "Team B", "country": "Country B"},
        "match_date": "2018-06-14",
        "kick_off": "17:00:00",
    }
