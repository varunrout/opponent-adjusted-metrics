import sqlite3
from pathlib import Path

import pandas as pd

from opponent_adjusted.analysis.shared.loaders import (
    load_model_registry,
    load_player_aggregates,
    load_shots_with_predictions,
    load_table,
    load_team_aggregates,
    resolve_latest_model_version,
)


def _create_cxg_fixture_db(path: Path, include_optional: bool = True) -> Path:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE model_registry (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            version TEXT,
            algorithm TEXT,
            artifact_path TEXT,
            created_at TEXT
        );
        CREATE TABLE shots (
            id INTEGER PRIMARY KEY,
            event_id INTEGER,
            match_id INTEGER,
            team_id INTEGER,
            player_id INTEGER,
            opponent_team_id INTEGER,
            statsbomb_xg REAL,
            body_part TEXT,
            shot_type TEXT,
            outcome TEXT
        );
        CREATE TABLE shot_predictions (
            id INTEGER PRIMARY KEY,
            shot_id INTEGER,
            model_id INTEGER,
            raw_probability REAL,
            neutral_probability REAL,
            opponent_adjusted_diff REAL,
            opponent_adjusted_ratio REAL
        );
        CREATE TABLE aggregates_player (
            id INTEGER PRIMARY KEY,
            player_id INTEGER,
            model_id INTEGER,
            version_tag TEXT,
            shots_count INTEGER,
            summed_cxg REAL,
            summed_neutral_cxg REAL,
            summed_oppadj_diff REAL,
            avg_oppadj_diff REAL
        );
        CREATE TABLE aggregates_team (
            id INTEGER PRIMARY KEY,
            team_id INTEGER,
            model_id INTEGER,
            version_tag TEXT,
            shots_count INTEGER,
            summed_cxg REAL,
            summed_neutral_cxg REAL,
            summed_oppadj_diff REAL,
            avg_oppadj_diff REAL
        );
        CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT);
        """
    )
    if include_optional:
        cur.executescript(
            """
            CREATE TABLE shot_features (
                id INTEGER PRIMARY KEY,
                shot_id INTEGER,
                minute_bucket TEXT,
                pressure_proxy_score REAL,
                recent_def_actions_count INTEGER
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY,
                minute INTEGER,
                second INTEGER,
                under_pressure INTEGER
            );
            """
        )
    cur.executemany(
        "INSERT INTO model_registry VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "cxg", "cxg_old", "fixture", "old.joblib", "2026-01-01"),
            (2, "cxg", "cxg_latest", "fixture", "latest.joblib", "2026-01-02"),
            (3, "cxa", "cxa_latest", "fixture", "cxa.joblib", "2026-01-03"),
        ],
    )
    cur.executemany(
        "INSERT INTO shots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (10, 100, 1, 20, 200, 30, 0.20, "Right Foot", "Open Play", "Goal"),
            (11, 101, 1, 20, 201, 31, 0.05, "Head", "Open Play", "Saved"),
        ],
    )
    cur.executemany(
        "INSERT INTO shot_predictions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, 10, 2, 0.25, 0.22, 0.03, 1.13),
            (2, 11, 2, 0.04, 0.05, -0.01, 0.80),
        ],
    )
    if include_optional:
        cur.executemany(
            "INSERT INTO shot_features VALUES (?, ?, ?, ?, ?)",
            [(1, 10, "0-15", 0.5, 2), (2, 11, "0-15", 0.1, 0)],
        )
        cur.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?)", [(100, 3, 10, 1), (101, 4, 20, 0)]
        )
    cur.executemany("INSERT INTO players VALUES (?, ?)", [(200, "Player A"), (201, "Player B")])
    cur.executemany("INSERT INTO teams VALUES (?, ?)", [(20, "Team A"), (30, "Opponent A")])
    cur.execute(
        "INSERT INTO aggregates_player VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, 200, 2, "cxg_latest", 1, 0.25, 0.22, 0.03, 0.03),
    )
    cur.execute(
        "INSERT INTO aggregates_team VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (1, 20, 2, "cxg_latest", 2, 0.29, 0.27, 0.02, 0.01),
    )
    conn.commit()
    conn.close()
    return path


def test_loaders_read_tables_and_resolve_latest_model(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db")

    registry = load_model_registry(db_path=db_path)
    assert len(registry) == 3
    assert resolve_latest_model_version("cxg", db_path=db_path) == "cxg_latest"
    assert load_table("shots", db_path=db_path).shape[0] == 2
    assert load_table("missing", db_path=db_path).empty


def test_load_shots_with_predictions_uses_db_join_and_optional_columns(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db")

    shots = load_shots_with_predictions(db_path=db_path)

    assert len(shots) == 2
    assert {"shot_id", "is_goal", "cxg_raw", "cxg_neutral", "cxg_opp_adjusted_diff"}.issubset(
        shots.columns
    )
    assert int(shots["is_goal"].sum()) == 1
    assert "under_pressure" in shots.columns


def test_load_shots_with_predictions_handles_missing_optional_context(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db", include_optional=False)

    shots = load_shots_with_predictions(db_path=db_path)

    assert len(shots) == 2
    assert "minute_bucket" in shots.columns
    assert shots["minute_bucket"].isna().all()


def test_load_aggregates_join_readable_names(tmp_path: Path):
    db_path = _create_cxg_fixture_db(tmp_path / "fixture.db")

    players = load_player_aggregates("cxg", db_path=db_path)
    teams = load_team_aggregates("cxg", db_path=db_path)

    assert players.loc[0, "player_name"] == "Player A"
    assert teams.loc[0, "team_name"] == "Team A"
    assert isinstance(players, pd.DataFrame)
