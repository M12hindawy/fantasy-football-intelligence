"""
db/repository.py - Repository pattern abstracting all DB operations.
Change DATABASE_URL in .env to switch between SQLite and PostgreSQL.
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Generator, Optional, Type, TypeVar
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from config.settings import settings
from db.models import Base

T = TypeVar("T")

_engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def init_db() -> None:
    """Create all tables. Safe to call multiple times."""
    Base.metadata.create_all(bind=_engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Repository:
    """Generic repository for CRUD operations."""

    def __init__(self, session: Session):
        self.session = session

    def get(self, model: Type[T], pk: object) -> Optional[T]:
        return self.session.get(model, pk)

    def add(self, obj: object) -> object:
        self.session.add(obj)
        self.session.flush()
        return obj

    def add_all(self, objs: list) -> None:
        self.session.add_all(objs)
        self.session.flush()

    def delete(self, obj: object) -> None:
        self.session.delete(obj)
        self.session.flush()

    def query(self, model: Type[T], **filters) -> list:
        stmt = select(model)
        for attr, val in filters.items():
            stmt = stmt.where(getattr(model, attr) == val)
        return list(self.session.scalars(stmt).all())

    def upsert(self, model: Type[T], pk_field: str, pk_value: object, **data) -> object:
        obj = self.get(model, pk_value)
        if obj is None:
            obj = model(**{pk_field: pk_value, **data})
            self.session.add(obj)
        else:
            for k, v in data.items():
                setattr(obj, k, v)
        self.session.flush()
        return obj
