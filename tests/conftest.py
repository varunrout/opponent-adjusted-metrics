"""Pytest configuration and fixtures."""

import os
import shutil
import sys
from pathlib import Path

import pytest

# Ensure the src directory is on the path so that the
# `opponent_adjusted` package can be imported in tests.
SRC_ROOT = Path(__file__).parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opponent_adjusted.config import settings, ensure_directories  # noqa: E402
from opponent_adjusted.db.base import Base  # noqa: E402
from opponent_adjusted.db import session as db_session  # noqa: E402


@pytest.fixture
def e2e_test_env(tmp_path):
    """Configure a fixture-backed SQLite environment for end-to-end tests.

    This fixture:
    - Copies the tiny committed StatsBomb fixture subset into pytest temp data
    - Points DATA_ROOT and STATSBOMB_DATA_PATH to that temporary copy
    - Forces DATABASE_BACKEND=sqlite and binds the DB session to a temp DB file
    - Ensures all required data / reports directories exist

    Returns the resolved data root path for further use in tests.
    """

    data_root = tmp_path / "e2e_data"
    statsbomb_path = data_root / "statsbomb"
    db_path = data_root / "opponent_adjusted.db"
    fixture_statsbomb_path = Path(__file__).parent / "fixtures" / "statsbomb"

    data_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_statsbomb_path, statsbomb_path)

    os.environ["DATA_ROOT"] = str(data_root)
    os.environ["STATSBOMB_DATA_PATH"] = str(statsbomb_path)
    os.environ["DATABASE_BACKEND"] = "sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    settings.data_root = data_root
    settings.statsbomb_data_path = statsbomb_path
    settings.database_backend = "sqlite"
    settings.database_url = f"sqlite:///{db_path}"
    ensure_directories()

    db_session.engine.dispose()
    db_session.engine = db_session.create_engine(settings.database_url, echo=False)
    db_session.SessionLocal.configure(bind=db_session.engine)
    Base.metadata.drop_all(bind=db_session.engine)
    Base.metadata.create_all(bind=db_session.engine)

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
