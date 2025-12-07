"""Pytest configuration and fixtures."""

import os
import sys
from pathlib import Path

import pytest


# Ensure the src directory is on the path so that the
# `opponent_adjusted` package can be imported in tests.
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opponent_adjusted.config import settings, ensure_directories


@pytest.fixture(scope="session")
def e2e_test_env(tmp_path_factory):
    """Configure a lightweight SQLite-backed environment for end-to-end tests.

    This fixture:
    - Points DATA_ROOT to a temporary directory under pytest's control
    - Forces DATABASE_BACKEND=sqlite so the project uses the local dev DB file
    - Ensures all required data / reports directories exist

    Returns the resolved data root path for further use in tests.
    """

    data_root = tmp_path_factory.mktemp("e2e_data")

    # Point config to the temporary data root and SQLite backend
    os.environ["DATA_ROOT"] = str(data_root)
    os.environ["DATABASE_BACKEND"] = "sqlite"

    # Re-run directory setup for the new data root
    settings.data_root = data_root
    ensure_directories()

    return data_root


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
