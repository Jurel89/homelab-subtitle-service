"""Base repository with shared engine and session management."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, Optional

try:
    from sqlalchemy import create_engine, Engine
    from sqlalchemy.orm import Session, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .settings import Settings, get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=4)
def get_engine(database_url: str, echo: bool = False) -> Engine:
    """
    Get or create a shared SQLAlchemy engine for the given database URL.

    Uses lru_cache so that repositories sharing the same URL and echo
    setting will reuse the same engine instance.

    Parameters
    ----------
    database_url : str
        Database connection URL.
    echo : bool
        Whether to echo SQL statements (for debugging).

    Returns
    -------
    Engine
        SQLAlchemy engine instance.
    """
    return create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
    )


class BaseRepository:
    """
    Base class for all repositories.

    Provides shared engine creation (via lru_cache) and a session
    context manager with automatic commit/rollback.

    Parameters
    ----------
    database_url : str, optional
        Database connection URL. If not provided, uses settings.
    settings : Settings, optional
        Application settings. If not provided, uses get_settings().
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy is required. "
                "Install with: pip install homelab-subtitle-service[server]"
            )

        self._settings = settings or get_settings()
        self._database_url = database_url or self._settings.database_url

        self._engine = get_engine(
            self._database_url,
            echo=self._settings.log_level == "DEBUG",
        )
        self._session_factory = sessionmaker(bind=self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.

        Handles commit/rollback automatically.

        Yields
        ------
        Session
            SQLAlchemy session for database operations.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
