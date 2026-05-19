"""Ticker lists for daily scan (legacy top-100 or PIT-filtered)."""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.yearly_top100_universe import (
    default_universe_cache_dir,
    load_universe_map_for_lag_years,
    read_ticker_lines,
)


def _apply_pit_filter(tickers: list[str], as_of: date) -> list[str]:
    from backtesting.sp500_pit_universe import members_as_of

    pit = members_as_of(as_of)
    return [t for t in tickers if t.upper() in pit]


def resolve_tickers(
    *,
    explicit: list[str] | None,
    universe: str,
    max_tickers: int | None,
    universe_source: str = "legacy",
    as_of: date | None = None,
) -> list[str]:
    as_of = as_of or datetime.today().date()
    src = (universe_source or "legacy").strip().lower()

    if explicit:
        out = [t.upper() for t in explicit]
        if src in ("pit", "pit_filter"):
            out = _apply_pit_filter(out, as_of)
        if max_tickers and max_tickers > 0:
            out = out[:max_tickers]
        return out

    u = universe.strip().lower()
    if u in ("top100", "top-100", "dollar_volume_top100"):
        lag_year = as_of.year - 1
        cache = default_universe_cache_dir(_ROOT)
        path = cache / f"{lag_year}.txt"
        if path.is_file():
            tickers = read_ticker_lines(path)
        else:
            uni = load_universe_map_for_lag_years([lag_year], cache, auto_build_missing=False)
            tickers = uni.get(lag_year, [])
        if src in ("pit", "pit_filter"):
            tickers = _apply_pit_filter(tickers, as_of)
        if max_tickers and max_tickers > 0:
            tickers = tickers[:max_tickers]
        return tickers

    raise ValueError(f"Unknown universe '{universe}' — pass tickers on CLI or use top100")
