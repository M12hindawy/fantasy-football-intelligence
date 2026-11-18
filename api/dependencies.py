"""
api/dependencies.py
===================
FastAPI dependencies, primarily for database injection.
"""
from typing import Generator
from sqlalchemy.orm import Session
from db.repository import get_session

def get_db() -> Generator[Session, None, None]:
    """Dependency that provides a SQLAlchemy session per request."""
    with get_session() as session:
        yield session
