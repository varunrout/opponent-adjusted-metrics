"""Database session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from opponent_adjusted.config import settings
from opponent_adjusted.utils.logging import get_logger

logger = get_logger(__name__)

# Create engine
engine = create_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session() -> Session:
    """Get a new database session.

    Returns:
        SQLAlchemy session
    """
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope for database operations.

    Yields:
        SQLAlchemy session

    Example:
        with session_scope() as session:
            session.add(obj)
            # Will auto-commit on success or rollback on error
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables.

    Note: In production, use Alembic migrations instead.
    """
    from opponent_adjusted.db.base import Base

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def drop_db() -> None:
    """Drop all database tables.

    Warning: This will delete all data!
    """
    from opponent_adjusted.db.base import Base

    Base.metadata.drop_all(bind=engine)
    logger.warning("Database tables dropped")
