"""SQLite-backed checkpoint cache for long backtest runs.

Ported from primerjava's LangGraph SqliteSaver idea (``graph/checkpointer.py``)
but specialised for the deterministic checkpoint-grid in
:mod:`backtesting.strategy_backtest`.

Use case
--------
A multi-year top-100 backtest evaluates O(tickers × checkpoints) point-in-time
classifications. Each is deterministic given the input bundle + date + mode,
yet a crash in the middle of a several-hour run currently throws all work
away.

This cache stores a JSON-serialised result keyed by
``(strategy_mode, ticker, checkpoint_date)``. A second invocation can read
cached rows instead of recomputing them. The cache is purely additive — the
caller is responsible for invalidating it (e.g. when feature schema or
classification logic changes; see ``schema_version`` argument).

The module deliberately depends on stdlib only (``sqlite3``, ``json``) so it
adds zero footprint to ``requirements.txt``.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoint_results (
    strategy_mode  TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    ticker         TEXT NOT NULL,
    checkpoint_iso TEXT NOT NULL,
    payload_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    PRIMARY KEY (strategy_mode, schema_version, ticker, checkpoint_iso)
);
CREATE INDEX IF NOT EXISTS idx_cp_ticker_date ON checkpoint_results (ticker, checkpoint_iso);
"""


def _iso(dt: datetime) -> str:
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


class BacktestCheckpointCache:
    """File-backed cache for (mode, ticker, date) -> result dict.

    Designed for additive use::

        cache = BacktestCheckpointCache(path, strategy_mode="ml", schema_version="v3")
        if (row := cache.get(ticker, cp_date)) is not None:
            use(row)
        else:
            row = expensive_compute(ticker, cp_date)
            cache.put(ticker, cp_date, row)

    The class is safe to construct against a non-existent file; the schema
    is created lazily on first connect.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        strategy_mode: str,
        schema_version: str = "v1",
    ):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = str(strategy_mode)
        self._schema = str(schema_version)
        self._ensure_schema()

    @property
    def path(self) -> Path:
        return self._path

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    def get(self, ticker: str, checkpoint: datetime) -> dict[str, Any] | None:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT payload_json FROM checkpoint_results "
                "WHERE strategy_mode = ? AND schema_version = ? "
                "AND ticker = ? AND checkpoint_iso = ?",
                (self._mode, self._schema, ticker.upper(), _iso(checkpoint)),
            )
            row = cur.fetchone()
            if row is None:
                return None
            try:
                return json.loads(row[0])
            except (TypeError, ValueError):
                return None

    def put(self, ticker: str, checkpoint: datetime, payload: dict[str, Any]) -> None:
        blob = json.dumps(payload, default=str)
        now = datetime.utcnow().isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoint_results "
                "(strategy_mode, schema_version, ticker, checkpoint_iso, payload_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self._mode,
                    self._schema,
                    ticker.upper(),
                    _iso(checkpoint),
                    blob,
                    now,
                ),
            )

    def count(self) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM checkpoint_results "
                "WHERE strategy_mode = ? AND schema_version = ?",
                (self._mode, self._schema),
            )
            return int(cur.fetchone()[0])

    def clear(self) -> int:
        """Drop all rows for this ``(strategy_mode, schema_version)`` pair."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM checkpoint_results "
                "WHERE strategy_mode = ? AND schema_version = ?",
                (self._mode, self._schema),
            )
            return int(cur.rowcount or 0)
