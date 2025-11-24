"""One-off script to copy data from the Postgres Docker DB
into the local SQLite dev database at data/opponent_adjusted.db.

Usage (from repo root, with Docker DB running):

    poetry run python scripts/migrate_postgres_to_sqlite.py

This will:
- Ensure SQLite schema exists (via SQLAlchemy models)
- Copy data for a selected set of tables in dependency-safe order.
"""

from __future__ import annotations

from typing import Type

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from opponent_adjusted.config import settings
from opponent_adjusted.db.base import Base
from opponent_adjusted.db import models
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)


def get_postgres_engine():
    """Engine for the primary Postgres DB (Docker)."""
    return create_engine(settings.database_url)


def get_sqlite_engine(sqlite_url: str = "sqlite:///data/opponent_adjusted.db"):
    """Engine for the local SQLite dev DB."""
    return create_engine(sqlite_url)


def create_sqlite_schema(sqlite_engine) -> None:
    """Create all tables in SQLite based on the shared models."""
    Base.metadata.create_all(bind=sqlite_engine)
    logger.info("SQLite schema created/verified at %s", sqlite_engine.url)


def copy_table(
    pg_sess: Session,
    sqlite_sess: Session,
    model: Type[Base],
    batch_size: int = 5000,
) -> int:
    """Copy all rows for a single ORM model from Postgres to SQLite in batches.

    This avoids loading large tables (e.g. events, raw_events) fully into memory.
    Returns number of rows copied.
    """

    # Truncate destination table first for idempotency
    sqlite_sess.execute(model.__table__.delete())
    sqlite_sess.commit()

    pk_col = list(model.__table__.primary_key.columns)[0]
    last_pk = None
    total = 0
    col_names = [c.name for c in model.__table__.columns]

    while True:
        stmt = select(model).order_by(pk_col).limit(batch_size)
        if last_pk is not None:
            stmt = stmt.where(pk_col > last_pk)

        batch = pg_sess.execute(stmt).scalars().all()
        if not batch:
            break

        for row in batch:
            data = {name: getattr(row, name) for name in col_names}
            sqlite_sess.add(model(**data))
            last_pk = getattr(row, pk_col.name)

        sqlite_sess.commit()
        total += len(batch)
        logger.info(
            "Table %s: copied %d rows so far",
            model.__tablename__,
            total,
        )

    logger.info("Table %s: copied %d rows (final)", model.__tablename__, total)
    return total


def main() -> None:
    pg_engine = get_postgres_engine()
    sqlite_engine = get_sqlite_engine()

    create_sqlite_schema(sqlite_engine)

    # Dependency-respecting order: refs → matches/events → shots/features → profiles → aggregates
    tables: list[Type[Base]] = [
        models.Competition,
        models.Team,
        models.Player,
        models.Match,
        models.RawEvent,
        models.Event,
        models.Possession,
        models.PassEvent,
        models.DribbleEvent,
        models.CarryEvent,
        models.ClearanceEvent,
        models.DuelEvent,
        models.BlockEvent,
        models.PressureEvent,
        models.BallReceiptEvent,
        models.Shot,
        models.ShotFeature,
        models.OpponentDefProfile,
        models.ModelRegistry,
        models.ShotPrediction,
        models.AggregatesPlayer,
        models.AggregatesTeam,
        models.EvaluationMetric,
    ]

    with Session(pg_engine) as pg_sess, Session(sqlite_engine) as sqlite_sess:
        total_rows = 0
        for model in tables:
            try:
                total_rows += copy_table(pg_sess, sqlite_sess, model)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed copying table %s: %s", model.__tablename__, exc)
                raise

    logger.info("Migration complete. Total rows copied: %d", total_rows)


if __name__ == "__main__":
    main()
