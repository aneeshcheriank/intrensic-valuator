"""
Application-level SQLite cache with TTL-driven expiration.

Provides the ``DataCache`` class — a persistent, thread-safe key-value
store backed by SQLite.  Each entry carries an absolute expiry timestamp;
expired entries are lazily evicted on read (no background thread needed).

TTL constants are provided for the data freshness requirements defined
in the architecture:

======= ============= ============================================
Constant   TTL          Use for
======= ============= ============================================
PRICE      1 day        Stock price quotes
FINANCIALS 7 days       Income statement, balance sheet, cash flow
RATES      7 days       Treasury yields, central bank rates
GDP       30 days       GDP growth, CPI, macro aggregates
ESTIMATES  7 days       Analyst consensus estimates
DEFAULT    1 day        Fallback for unclassified data
======= ============= ============================================

Cache key convention: ``"{source}:{identifier}:{data_type}"``

Examples::

    "yfinance:AAPL:cash_flow"
    "fred:GDP:quarterly"
    "worldbank:IN:gdp_growth"
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from src.utils.config import get_config

# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

TTL_PRICE: int = 86_400  # 1 day
TTL_FINANCIALS: int = 604_800  # 7 days
TTL_MACRO_RATES: int = 604_800  # 7 days
TTL_MACRO_GDP: int = 2_592_000  # 30 days
TTL_ESTIMATES: int = 604_800  # 7 days
TTL_DEFAULT: int = 86_400  # 1 day fallback

# ---------------------------------------------------------------------------
# JSON encoder / decoder that handles numpy & pandas
# ---------------------------------------------------------------------------


class _CacheEncoder(json.JSONEncoder):
    """Custom encoder that copes with numpy scalars, arrays, and pandas types."""

    def default(self, obj: Any) -> Any:
        # numpy scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # pandas
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="split")
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return super().default(obj)


def _encode_value(value: Any) -> str:
    """Serialize *value* to a JSON string suitable for cache storage."""
    # If the whole value is a DataFrame, store it via a sentinel so we can
    # round-trip perfectly.
    if isinstance(value, pd.DataFrame):
        return json.dumps(
            {"__pd_dataframe__": True, "data": value.to_dict(orient="split")}
        )
    if isinstance(value, pd.Series):
        return json.dumps(
            {"__pd_series__": True, "data": value.to_list(), "name": value.name}
        )
    return json.dumps(value, cls=_CacheEncoder)


def _decode_value(raw: str) -> Any:
    """Deserialize a JSON string back to the original Python object."""
    obj = json.loads(raw)
    if isinstance(obj, dict):
        if obj.get("__pd_dataframe__"):
            return pd.DataFrame(**obj["data"])
        if obj.get("__pd_series__"):
            s = pd.Series(obj["data"], name=obj.get("name"))
            return s
    return obj


# ---------------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    expires_at  REAL NOT NULL,
    created_at  REAL NOT NULL
)
"""

# Index on expires_at so ``expire()`` can run quickly even with many rows.
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries (expires_at)
"""


# ---------------------------------------------------------------------------
# DataCache
# ---------------------------------------------------------------------------


class DataCache:
    """Thread-safe, SQLite-backed cache with per-key TTL expiration.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it doesn't exist.
        Defaults to ``<project_root>/cache/valuator_cache.db``.
    """

    # Shared hit/miss/expire counters (across all instances if they share one
    # DB, but realistically one instance per process).
    _hits: ClassVar[int] = 0
    _misses: ClassVar[int] = 0
    _expirations: ClassVar[int] = 0

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            cfg = get_config()
            cfg.ensure_cache_dir()
            db_path = str(cfg.cache_db_path)

        self._db_path: str = db_path
        self._lock: threading.Lock = threading.Lock()

        # Initialise schema on first use (lazy, but we do it eagerly so the
        # file is ready).
        with self._get_conn() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.commit()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new (or reused) SQLite connection.

        ``check_same_thread=False`` is safe because we guard writes with a
        mutex and SQLite's WAL mode allows concurrent reads.
        """
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def _write_conn(self):
        """Context manager that acquires the write lock and yields a connection."""
        with self._lock:
            conn = self._get_conn()
            try:
                yield conn
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the cached value for *key*, or ``None`` if missing / expired.

        Expired entries are silently deleted on access (lazy eviction).
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE key = ?",
                (key,),
            ).fetchone()

            if row is None:
                DataCache._misses += 1
                return None

            value_json, expires_at = row
            if expires_at < self._now():
                # Expired — evict and count as miss.
                self.delete(key)
                DataCache._expirations += 1
                DataCache._misses += 1
                return None

            DataCache._hits += 1
            return _decode_value(value_json)
        finally:
            conn.close()

    def set(self, key: str, value: Any, ttl_seconds: int = TTL_DEFAULT) -> None:
        """Store *value* under *key* with the given TTL.

        If *key* already exists its value and expiry are overwritten.
        """
        expires_at = self._now() + ttl_seconds
        value_json = _encode_value(value)

        with self._write_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries(key, value, expires_at, created_at) "
                "VALUES (?, ?, ?, ?)",
                (key, value_json, expires_at, self._now()),
            )
            conn.commit()

    def delete(self, key: str) -> None:
        """Remove *key* from the cache (no-op if not present)."""
        with self._write_conn() as conn:
            conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> None:
        """Remove **all** entries from the cache."""
        with self._write_conn() as conn:
            conn.execute("DELETE FROM cache_entries")
            conn.commit()

    def expire(self) -> int:
        """Explicitly evict all expired entries.  Returns the count removed."""
        with self._write_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at < ?", (self._now(),)
            )
            conn.commit()
            removed = cursor.rowcount
            DataCache._expirations += removed
            return removed

    def stats(self) -> dict[str, Any]:
        """Return usage statistics.

        Returns a dict with keys ``total_entries``, ``expired_entries``,
        ``hits``, ``misses``, ``expirations``, ``db_size_bytes``,
        ``approximate_hit_rate``.
        """
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
            expired = conn.execute(
                "SELECT COUNT(*) FROM cache_entries WHERE expires_at < ?",
                (self._now(),),
            ).fetchone()[0]
        finally:
            conn.close()

        total_requests = DataCache._hits + DataCache._misses
        hit_rate = (
            DataCache._hits / total_requests if total_requests > 0 else float("nan")
        )

        import os

        db_size = os.path.getsize(self._db_path) if os.path.exists(self._db_path) else 0

        return {
            "total_entries": total,
            "expired_entries": expired,
            "hits": DataCache._hits,
            "misses": DataCache._misses,
            "expirations": DataCache._expirations,
            "db_size_bytes": db_size,
            "approximate_hit_rate": round(hit_rate, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> float:
        """Return the current time as a Unix timestamp.

        Extracted to a method so tests can monkey-patch it.
        """
        return time.time()
