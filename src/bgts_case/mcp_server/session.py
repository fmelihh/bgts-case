from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from bgts_case.secret import secrets

engine = create_engine(secrets.database_url, future=True, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
    class_=Session,
)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a `Session`, committing on success and rolling back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
