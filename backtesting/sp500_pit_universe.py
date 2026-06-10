"""
sp500_pit_universe.py — Point-in-time S&P 500 membership.

The legacy ``yearly_top100_universe`` builds a top-100 dollar-volume
ranking from the *current* S&P 500 list (Wikipedia). That introduces
survivorship bias because names that were in the index during the
backtest window but later delisted (Lehman, Bear, GE, Sears, SVB,
Signature, First Republic, …) silently disappear from the candidate
pool.

This module replaces the candidate pool with a point-in-time membership
set built from a delisted-overlay snapshot:

    current_members ∪ ever-removed-members
    └─ minus members removed before ``as_of``
    └─ plus members added after their add-date

The seed snapshot ships at ``backtesting/sp500_changes.csv``. It covers
the largest / most-cited changes 2008-2026 — extend from S&P DJI press
releases when new rebalances occur.

Public API
----------
* ``members_as_of(date)``           → set of tickers in the index on that date.
* ``ever_members_in_window(start, end)`` → set of tickers that were members
                                            *at some point* during [start, end].
* ``pit_top_n(year, top_n=100)``    → drop-in replacement for the legacy
                                      ``build_dollar_volume_top_n`` that
                                      ranks against the PIT pool for that year.

The functions emit a one-time warning when the change-log doesn't cover
the requested date range, so the caller knows the result is partial-PIT
rather than full-PIT.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable


_DEFAULT_CHANGES_CSV = Path(__file__).resolve().parent / "sp500_changes.csv"
_COVERAGE_WARNED: set[tuple[str, str]] = set()


def reset_coverage_warnings() -> None:
    """Test helper: allow coverage warnings to fire again."""
    _COVERAGE_WARNED.clear()


@dataclass(frozen=True)
class Change:
    when: date
    action: str          # "add" | "remove"
    ticker: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.action not in ("add", "remove"):
            raise ValueError(f"Bad action {self.action!r}")


def load_changes(path: Path = _DEFAULT_CHANGES_CSV) -> list[Change]:
    """Parse the seed CSV. Comment lines (``#``) and blanks are skipped."""
    if not path.is_file():
        raise FileNotFoundError(f"S&P 500 change log not found: {path}")
    out: list[Change] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            cell0 = raw[0].strip()
            if not cell0 or cell0.startswith("#"):
                continue
            if len(raw) < 3:
                continue
            try:
                when = date.fromisoformat(raw[0].strip())
            except ValueError:
                continue
            action = raw[1].strip().lower()
            ticker = raw[2].strip().upper().replace(".", "-")
            note = (raw[3].strip() if len(raw) > 3 else "")
            try:
                out.append(Change(when=when, action=action, ticker=ticker, note=note))
            except ValueError:
                continue
    out.sort(key=lambda c: (c.when, c.ticker))
    return out


def _coverage_warning(changes: list[Change], start: date, end: date) -> None:
    """Warn at most once per query window (see module docstring)."""
    key = (start.isoformat(), end.isoformat())
    if key in _COVERAGE_WARNED:
        return
    if not changes:
        warnings.warn(
            "S&P 500 PIT change log is empty; falling back to current "
            "membership only (this is the survivorship-biased path).",
            stacklevel=3,
        )
        _COVERAGE_WARNED.add(key)
        return
    log_min = changes[0].when
    log_max = changes[-1].when
    if start < log_min or end > log_max:
        # Live daily runs after the CSV end date: membership uses current list + log;
        # no need to spam warnings on every trading day.
        if start >= log_min and end > log_max:
            return
        warnings.warn(
            f"S&P 500 PIT change log covers {log_min.isoformat()}..{log_max.isoformat()} "
            f"but query window is {start.isoformat()}..{end.isoformat()}. "
            f"Membership outside that window is approximate. Extend "
            f"backtesting/sp500_changes.csv for full PIT coverage.",
            stacklevel=3,
        )
        _COVERAGE_WARNED.add(key)


# ── public API ────────────────────────────────────────────────────────────────

def members_as_of(
    asof: date,
    *,
    current_pool: Iterable[str] | None = None,
    changes_path: Path = _DEFAULT_CHANGES_CSV,
) -> set[str]:
    """Return the ticker set that was in the S&P 500 on ``asof``.

    ``current_pool`` defaults to ``fetch_sp500_symbols()`` from the
    legacy module — i.e. today's index. We start from "today" and walk
    the change log *backwards* to ``asof``: every "add" after ``asof``
    is undone (removed from the set) and every "remove" after ``asof``
    is undone (re-added to the set).

    The walk is deterministic: it does not need network access if the
    user passes ``current_pool`` explicitly.
    """
    if current_pool is None:
        from backtesting.yearly_top100_universe import fetch_sp500_symbols
        current_pool = fetch_sp500_symbols()
    members = {t.upper().replace(".", "-") for t in current_pool}

    changes = load_changes(changes_path)
    _coverage_warning(changes, asof, asof)

    # Walk backwards through changes that occurred AFTER asof, undoing them.
    for ch in reversed(changes):
        if ch.when <= asof:
            break
        if ch.action == "add":
            members.discard(ch.ticker)
        else:  # remove
            members.add(ch.ticker)
    return members


def ever_members_in_window(
    start: date,
    end: date,
    *,
    current_pool: Iterable[str] | None = None,
    changes_path: Path = _DEFAULT_CHANGES_CSV,
) -> set[str]:
    """Tickers that were in the index AT ANY POINT during ``[start, end]``.

    Equivalent to the union of ``members_as_of(d)`` for every ``d`` in
    the window — but computed in one pass via the change log.
    """
    base = members_as_of(end, current_pool=current_pool, changes_path=changes_path)
    changes = load_changes(changes_path)
    _coverage_warning(changes, start, end)
    out = set(base)
    for ch in changes:
        if start <= ch.when <= end:
            out.add(ch.ticker)
    return out


def pit_top_n(
    year: int,
    *,
    top_n: int = 100,
    current_pool: Iterable[str] | None = None,
    sleep_s: float = 0.05,
    verbose: bool = True,
    changes_path: Path = _DEFAULT_CHANGES_CSV,
) -> list[str]:
    """Drop-in PIT replacement for ``build_dollar_volume_top_n``.

    Uses the universe of names that were members AT ANY POINT during
    calendar year ``year``, then ranks them by total dollar volume in
    that year (same ranking metric as the legacy module).
    """
    from backtesting.yearly_top100_universe import dollar_volume_sum
    import time

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    pool = ever_members_in_window(start, end, current_pool=current_pool,
                                   changes_path=changes_path)
    pool_list = sorted(pool)
    if verbose:
        print(f"  PIT pool for {year}: {len(pool_list)} tickers")
    scores: list[tuple[str, float]] = []
    for i, sym in enumerate(pool_list, 1):
        if verbose and (i == 1 or i % 50 == 0 or i == len(pool_list)):
            print(f"    [{i}/{len(pool_list)}] scoring {sym} …", flush=True)
        dv = dollar_volume_sum(sym, year)
        if dv is not None and dv > 0:
            scores.append((sym, dv))
        if sleep_s:
            time.sleep(sleep_s)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]
