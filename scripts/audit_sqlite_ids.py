"""Audit identifier completeness for SQLite-backed CxG tables."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from opponent_adjusted.config import settings

AUDIT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("shots", "shot_id"),
    ("shots", "event_id"),
    ("shots", "match_id"),
    ("shots", "team_id"),
    ("shots", "opponent_team_id"),
    ("shots", "player_id"),
    ("events", "player_id"),
)


def _sqlite_path_from_url(url: str) -> Path | None:
    if url == "sqlite:///:memory:":
        raise ValueError("In-memory SQLite databases cannot be audited from a file path.")
    if not url.startswith("sqlite:///"):
        return None
    return Path(url.removeprefix("sqlite:///"))


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}


def _audit_column(conn: sqlite3.Connection, table: str, column: str) -> dict[str, Any]:
    table_exists = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        is not None
    )
    if not table_exists:
        return {
            "table": table,
            "column": column,
            "rows": 0,
            "missing": 0,
            "missing_pct": 0.0,
            "status": "table_missing",
        }

    columns = _table_columns(conn, table)
    if column not in columns:
        return {
            "table": table,
            "column": column,
            "rows": 0,
            "missing": 0,
            "missing_pct": 0.0,
            "status": "column_missing",
        }

    total_rows, missing_rows = conn.execute(
        f'SELECT COUNT(*), SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END) FROM "{table}"'
    ).fetchone()
    total = int(total_rows or 0)
    missing = int(missing_rows or 0)
    missing_pct = (missing / total * 100.0) if total else 0.0
    if total == 0:
        status = "no_rows"
    elif missing == 0:
        status = "ok"
    elif missing == total:
        status = "all_missing"
    else:
        status = "partial_missing"
    return {
        "table": table,
        "column": column,
        "rows": total,
        "missing": missing,
        "missing_pct": round(missing_pct, 4),
        "status": status,
    }


def audit_sqlite_ids(sqlite_path: Path) -> pd.DataFrame:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {sqlite_path}")
    with sqlite3.connect(sqlite_path) as conn:
        rows = [_audit_column(conn, table, column) for table, column in AUDIT_COLUMNS]
    return pd.DataFrame(rows)


def _resolve_sqlite_path(cli_path: str | None) -> Path | None:
    if cli_path:
        return Path(cli_path)
    from_url = _sqlite_path_from_url(settings.database_url)
    if from_url is not None:
        return from_url
    fallback = Path(settings.data_root) / "opponent_adjusted.db"
    if fallback.exists():
        return fallback
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit SQLite identifier completeness.")
    parser.add_argument(
        "--sqlite-path",
        help="Path to the SQLite database file. If omitted, derives from DATABASE_URL.",
    )
    args = parser.parse_args(argv)

    sqlite_path = _resolve_sqlite_path(args.sqlite_path)
    if sqlite_path is None:
        print(
            "No SQLite database path could be resolved. Set DATABASE_URL to sqlite:///... "
            "or pass --sqlite-path."
        )
        return
    report = audit_sqlite_ids(sqlite_path)

    print(f"SQLite ID audit: {sqlite_path}")
    print(report.to_string(index=False))

    shot_player_row = report[(report["table"] == "shots") & (report["column"] == "player_id")].iloc[
        0
    ]
    if shot_player_row["status"] == "all_missing":
        print(
            "\nALERT: shots.player_id is all missing. Player-level CxG summaries are unreliable "
            "until shot rows preserve linked event player identifiers."
        )


if __name__ == "__main__":
    main()
