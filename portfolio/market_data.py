"""Cached OHLCV fetch for daily agent (reduces repeated yfinance calls)."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)

from portfolio.store import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "ohlcv"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = hashlib.sha256(f"{ticker.upper()}|{start}|{end}".encode()).hexdigest()[:16]
    return CACHE_DIR / f"{ticker.upper()}_{key}.feather"


def fetch_history(
    ticker: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
    max_age_days: int = 1,
) -> pd.DataFrame | None:
    """Daily OHLCV between ``start`` and ``end`` (yfinance), with optional feather cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker, start, end)
    if use_cache and path.is_file():
        age = date.today() - date.fromtimestamp(path.stat().st_mtime)
        if age.days <= max_age_days:
            try:
                df = pd.read_feather(path)
                if df is not None and not df.empty:
                    return df.sort_index()
            except (OSError, ValueError, KeyError) as exc:
                _log.debug("OHLCV cache read failed for %s: %s", ticker, exc)
    try:
        df = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=True,
        )
        if df is None or df.empty or "Close" not in df.columns:
            _log.warning("Empty OHLCV for %s (%s to %s)", ticker, start, end)
            return None
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        if use_cache:
            try:
                df.reset_index().to_feather(path)
            except (OSError, ValueError) as exc:
                _log.debug("OHLCV cache write failed for %s: %s", ticker, exc)
        return df
    except (OSError, ValueError, KeyError) as exc:
        _log.warning("yfinance fetch failed for %s: %s", ticker, exc)
        return None


def fetch_window(
    ticker: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Inclusive window; ``end`` is bumped by one day for yfinance exclusivity."""
    try:
        end_dt = date.fromisoformat(end[:10]) + timedelta(days=1)
        end_excl = end_dt.isoformat()
    except ValueError:
        end_excl = end
    return fetch_history(ticker, start, end_excl, use_cache=use_cache)
