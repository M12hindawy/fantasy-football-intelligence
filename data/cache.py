"""
data/cache.py
=============
SQLite-backed TTL cache for API responses.

Every record stored in the cache carries:
- retrieved_at  : when it was fetched
- valid_until   : derived from the data-type threshold
- data_type     : key into freshness.THRESHOLDS
- payload       : JSON-serialised response body

The cache is used by all fetchers to avoid redundant API calls and to
provide offline access to recently fetched data.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator, Optional

from config.freshness import THRESHOLDS


class CacheRecord:
    """A single cache entry."""

    def __init__(
        self,
        key: str,
        data_type: str,
        payload: Any,
        retrieved_at: datetime,
        valid_until: datetime,
    ) -> None:
        self.key = key
        self.data_type = data_type
        self.payload = payload
        self.retrieved_at = retrieved_at
        self.valid_until = valid_until

    @property
    def is_fresh(self) -> bool:
        return datetime.now(timezone.utc) <= self.valid_until

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.retrieved_at).total_seconds()


class FootballCache:
    """
    Lightweight SQLite cache with TTL support.

    Parameters
    ----------
    db_path : path to the SQLite file (created if it does not exist)
    enabled : set to False to bypass all cache reads/writes
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache (
        key         TEXT PRIMARY KEY,
        data_type   TEXT NOT NULL,
        payload     TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        valid_until  TEXT NOT NULL
    );
    """

    def __init__(self, db_path: str = "football_agent_cache.db", enabled: bool = True) -> None:
        self._path = Path(db_path)
        self._enabled = enabled
        if enabled:
            self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(self._SCHEMA)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[CacheRecord]:
        """Return a fresh cache record or None."""
        if not self._enabled:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            record = self._row_to_record(row)
            if not record.is_fresh:
                return None
            return record

    def set(self, key: str, data_type: str, payload: Any) -> None:
        """Store *payload* in the cache with TTL derived from *data_type*."""
        if not self._enabled:
            return
        now = datetime.now(timezone.utc)
        ttl_s = THRESHOLDS.get(data_type, 3_600)
        valid_until = now + timedelta(seconds=ttl_s)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache (key, data_type, payload, retrieved_at, valid_until)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    data_type   = excluded.data_type,
                    payload     = excluded.payload,
                    retrieved_at = excluded.retrieved_at,
                    valid_until  = excluded.valid_until
                """,
                (
                    key,
                    data_type,
                    json.dumps(payload, default=str),
                    now.isoformat(),
                    valid_until.isoformat(),
                ),
            )

    def invalidate(self, key: str) -> None:
        """Remove a specific key from the cache."""
        if not self._enabled:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))

    def invalidate_by_type(self, data_type: str) -> int:
        """Remove all records of a given data_type.  Returns rows deleted."""
        if not self._enabled:
            return 0
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE data_type = ?", (data_type,)
            )
            return cur.rowcount

    def purge_expired(self) -> int:
        """Delete all expired records.  Returns rows deleted."""
        if not self._enabled:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM cache WHERE valid_until < ?", (now,)
            )
            return cur.rowcount

    def stats(self) -> dict[str, Any]:
        """Return cache statistics for monitoring."""
        if not self._enabled:
            return {"enabled": False}
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            fresh = conn.execute(
                "SELECT COUNT(*) FROM cache WHERE valid_until >= ?", (now,)
            ).fetchone()[0]
        return {"enabled": True, "total_records": total, "fresh_records": fresh}

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> CacheRecord:
        return CacheRecord(
            key=row["key"],
            data_type=row["data_type"],
            payload=json.loads(row["payload"]),
            retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
            valid_until=datetime.fromisoformat(row["valid_until"]),
        )
