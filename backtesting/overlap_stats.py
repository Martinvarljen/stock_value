"""
overlap_stats.py — Overlap-adjusted sample sizes for forward-return tier stats.

Checkpoints on the same ticker with overlapping forward windows (e.g. two
6M returns starting one quarter apart) are not independent. ``effective_n``
counts greedy non-overlapping windows per ticker, then sums across tickers.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


def _as_datetime(d: Any) -> datetime:
    if isinstance(d, datetime):
        return d.replace(tzinfo=None) if d.tzinfo else d
    return datetime.fromisoformat(str(d)[:10])


def horizon_days(months: int) -> float:
    return float(months) * 30.44


def effective_n_greedy(dates: list[datetime], months: int) -> int:
    """Non-overlapping checkpoint count for one ticker and one forward horizon."""
    if not dates:
        return 0
    gap = horizon_days(months)
    ordered = sorted(dates)
    count = 0
    last: datetime | None = None
    for d in ordered:
        if last is None or (d - last).days >= gap:
            count += 1
            last = d
    return count


def effective_n_for_signals(
    signals: list[dict],
    months: int,
    *,
    return_key: str | None = None,
) -> float:
    """
    Sum of per-ticker non-overlapping counts for signals with a valid forward return.

    ``return_key`` defaults to ``fwd_{months}m``.
    """
    fk = return_key or f"fwd_{months}m"
    by_ticker: dict[str, list[datetime]] = {}
    for s in signals:
        if s.get(fk) is None:
            continue
        tk = str(s.get("ticker", ""))
        if not tk:
            continue
        by_ticker.setdefault(tk, []).append(_as_datetime(s["date"]))
    return float(sum(effective_n_greedy(ds, months) for ds in by_ticker.values()))


def overlap_inflation_factor(n_raw: int, n_eff: float) -> float | None:
    """sqrt(n_raw / n_eff) — scales standard errors when n_eff < n_raw."""
    if n_raw <= 0 or n_eff <= 0:
        return None
    if n_eff >= n_raw:
        return 1.0
    return (n_raw / n_eff) ** 0.5
