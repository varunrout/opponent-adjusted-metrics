"""Database bootstrap helpers."""

from pathlib import Path

from sqlalchemy.engine import make_url


def ensure_sqlite_database_parent(database_url: str) -> None:
    """Create the parent directory for file-backed SQLite database URLs.

    SQLite can create a missing database file, but it cannot create a missing
    parent directory. Alembic calls this before opening a connection so clean
    checkouts with `sqlite:///data/opponent_adjusted.db` can migrate directly.
    """

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return

    database_path = url.database
    if not database_path or database_path == ":memory:":
        return

    parent = Path(database_path).expanduser().parent
    if parent == Path("."):
        return

    parent.mkdir(parents=True, exist_ok=True)
