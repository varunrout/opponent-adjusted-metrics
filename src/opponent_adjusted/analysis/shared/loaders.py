"""DB-backed loaders for v1 analysis reports."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from opponent_adjusted.config import settings

DEFAULT_DB_PATH = Path("data") / "opponent_adjusted.db"


def _resolve_db_path(db_path: Path | None = None) -> Path:
    if db_path is not None:
        return Path(db_path)
    if settings.database_backend.lower() == "sqlite" and settings.database_url.startswith(
        "sqlite:///"
    ):
        return Path(settings.database_url.replace("sqlite:///", "", 1))
    return DEFAULT_DB_PATH


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = _resolve_db_path(db_path)
    return sqlite3.connect(path)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    if not _table_exists(conn, table_name):
        return set()
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()}


def _select_expr(
    columns: set[str],
    table_alias: str,
    source_name: str,
    output_name: str,
    default_sql: str = "NULL",
) -> str:
    if source_name in columns:
        return f"{table_alias}.{source_name} AS {output_name}"
    return f"{default_sql} AS {output_name}"


def _has_all(conn: sqlite3.Connection, tables: Iterable[str]) -> bool:
    return all(_table_exists(conn, table_name) for table_name in tables)


def load_table(table_name: str, db_path: Path | None = None) -> pd.DataFrame:
    """Load a full SQLite table into a DataFrame.

    The table name is validated against sqlite metadata before interpolation.
    Missing tables return an empty DataFrame.
    """

    with _connect(db_path) as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame()
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def load_model_registry(db_path: Path | None = None) -> pd.DataFrame:
    """Load model registry rows."""

    return load_table("model_registry", db_path=db_path)


def resolve_latest_model_version(
    model_family: str,
    db_path: Path | None = None,
) -> str | None:
    """Resolve the newest registered model version for a model family."""

    registry = load_model_registry(db_path=db_path)
    if registry.empty or "model_name" not in registry or "version" not in registry:
        return None
    matches = registry[registry["model_name"].astype(str).str.lower() == model_family.lower()]
    if matches.empty:
        return None
    sort_cols = [col for col in ("created_at", "id") if col in matches.columns]
    if sort_cols:
        matches = matches.sort_values(sort_cols)
    return str(matches.iloc[-1]["version"])


def load_shots_with_predictions(
    model_version: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load CxG shot rows joined to prediction and model metadata."""

    with _connect(db_path) as conn:
        required = ("shots", "shot_predictions", "model_registry")
        if not _has_all(conn, required):
            return pd.DataFrame()

        if model_version is None:
            model_version = resolve_latest_model_version("cxg", db_path=db_path)
        if model_version is None:
            return pd.DataFrame()

        shot_cols = _columns(conn, "shots")
        prediction_cols = _columns(conn, "shot_predictions")
        feature_cols = _columns(conn, "shot_features")
        event_cols = _columns(conn, "events")
        has_features = _table_exists(conn, "shot_features")
        has_events = _table_exists(conn, "events")

        select_parts = [
            "s.id AS shot_id",
            _select_expr(shot_cols, "s", "event_id", "event_id"),
            _select_expr(shot_cols, "s", "match_id", "match_id"),
            _select_expr(shot_cols, "s", "team_id", "team_id"),
            _select_expr(shot_cols, "s", "player_id", "player_id"),
            _select_expr(shot_cols, "s", "opponent_team_id", "opponent_team_id"),
            _select_expr(shot_cols, "s", "outcome", "outcome"),
            "CASE WHEN lower(COALESCE(s.outcome, '')) = 'goal' THEN 1 ELSE 0 END AS is_goal",
            _select_expr(shot_cols, "s", "statsbomb_xg", "statsbomb_xg"),
            _select_expr(shot_cols, "s", "statsbomb_xg", "provider_xg"),
            _select_expr(shot_cols, "s", "body_part", "body_part"),
            _select_expr(shot_cols, "s", "shot_type", "shot_type"),
            _select_expr(prediction_cols, "sp", "raw_probability", "cxg_raw"),
            _select_expr(prediction_cols, "sp", "raw_probability", "raw_probability"),
            _select_expr(prediction_cols, "sp", "neutral_probability", "cxg_neutral"),
            _select_expr(prediction_cols, "sp", "neutral_probability", "neutral_probability"),
            _select_expr(
                prediction_cols,
                "sp",
                "opponent_adjusted_diff",
                "cxg_opp_adjusted_diff",
            ),
            _select_expr(
                prediction_cols,
                "sp",
                "opponent_adjusted_ratio",
                "cxg_opp_adjusted_ratio",
            ),
            "mr.version AS model_version",
            "mr.model_name AS model_family",
        ]
        if has_features:
            select_parts.extend(
                [
                    _select_expr(feature_cols, "sf", "minute_bucket", "minute_bucket"),
                    _select_expr(
                        feature_cols, "sf", "pressure_proxy_score", "pressure_proxy_score"
                    ),
                    _select_expr(
                        feature_cols,
                        "sf",
                        "recent_def_actions_count",
                        "recent_def_actions_count",
                    ),
                ]
            )
        else:
            select_parts.extend(
                [
                    "NULL AS minute_bucket",
                    "NULL AS pressure_proxy_score",
                    "NULL AS recent_def_actions_count",
                ]
            )
        if has_events:
            select_parts.extend(
                [
                    _select_expr(event_cols, "e", "minute", "minute"),
                    _select_expr(event_cols, "e", "second", "second"),
                    _select_expr(event_cols, "e", "under_pressure", "under_pressure"),
                ]
            )
        else:
            select_parts.extend(["NULL AS minute", "NULL AS second", "NULL AS under_pressure"])

        joins = [
            "FROM shots s",
            "JOIN shot_predictions sp ON sp.shot_id = s.id",
            "JOIN model_registry mr ON mr.id = sp.model_id",
        ]
        if has_features:
            joins.append("LEFT JOIN shot_features sf ON sf.shot_id = s.id")
        if has_events:
            joins.append("LEFT JOIN events e ON e.id = s.event_id")

        query = f"""
            SELECT {", ".join(select_parts)}
            {" ".join(joins)}
            WHERE lower(mr.model_name) = 'cxg'
              AND mr.version = ?
        """
        return pd.read_sql_query(query, conn, params=(model_version,))


def _load_aggregates(
    table_name: str,
    model_family: str,
    db_path: Path | None = None,
) -> pd.DataFrame:
    with _connect(db_path) as conn:
        if not _has_all(conn, (table_name, "model_registry")):
            return pd.DataFrame()
        query = f"""
            SELECT a.*, mr.model_name AS model_family, mr.version AS model_version
            FROM {table_name} a
            JOIN model_registry mr ON mr.id = a.model_id
            WHERE lower(mr.model_name) = ?
        """
        return pd.read_sql_query(query, conn, params=(model_family.lower(),))


def load_player_aggregates(
    model_family: str,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load player aggregates for a model family."""

    df = _load_aggregates("aggregates_player", model_family, db_path=db_path)
    with _connect(db_path) as conn:
        if df.empty or "player_id" not in df or not _table_exists(conn, "players"):
            return df
        players = pd.read_sql_query(
            "SELECT id AS player_id, name AS player_name FROM players", conn
        )
    return df.merge(players, on="player_id", how="left")


def load_team_aggregates(
    model_family: str,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load team aggregates for a model family."""

    df = _load_aggregates("aggregates_team", model_family, db_path=db_path)
    with _connect(db_path) as conn:
        if df.empty or "team_id" not in df or not _table_exists(conn, "teams"):
            return df
        teams = pd.read_sql_query("SELECT id AS team_id, name AS team_name FROM teams", conn)
    return df.merge(teams, on="team_id", how="left")
