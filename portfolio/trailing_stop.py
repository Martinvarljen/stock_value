"""Trailing stop: ratchet ``Position.stop_price`` from peak/trough since entry."""

from __future__ import annotations

from typing import Any

from portfolio.store import Position


def trailing_enabled(cfg: dict[str, Any]) -> bool:
    if not cfg.get("use_trailing_stop", False):
        return False
    return float(cfg.get("trailing_stop_pct", 0) or 0) > 0


def initial_fixed_stop(entry: float, side: str, cfg: dict[str, Any]) -> float:
    """Catastrophic stop from entry (used as floor/cap before trail engages)."""
    stop_pct = float(cfg.get("stop_loss_pct", 0.20))
    if side == "long":
        return entry * (1 - stop_pct)
    return entry * (1 + stop_pct)


def seed_trail_fields(pos: Position, cfg: dict[str, Any]) -> None:
    """Call once at entry so peak/trough and initial stop are set."""
    px = float(pos.entry_price)
    fixed = initial_fixed_stop(px, pos.side, cfg)
    pos.initial_stop_price = round(fixed, 4)
    pos.peak_price = px
    pos.trough_price = px
    if not trailing_enabled(cfg):
        pos.stop_price = round(fixed, 4)
        return
    pos.stop_price = round(fixed, 4)


def update_position_trail(
    pos: Position,
    bar_high: float,
    bar_low: float,
    cfg: dict[str, Any],
) -> bool:
    """
    Update peak/trough from today's range and ratchet ``stop_price``.

    Returns True if ``stop_price`` changed.
    """
    if not trailing_enabled(cfg):
        return False

    trail_pct = float(cfg["trailing_stop_pct"])
    activate = float(cfg.get("trail_activate_profit_pct", 0.0))
    entry = float(pos.entry_price)
    if entry <= 0:
        return False

    initial = float(pos.initial_stop_price or initial_fixed_stop(entry, pos.side, cfg))
    old_stop = float(pos.stop_price)

    if pos.side == "long":
        peak = max(float(pos.peak_price or entry), float(bar_high))
        pos.peak_price = peak
        fav = (peak - entry) / entry
        if fav < activate:
            return False
        trail = peak * (1.0 - trail_pct)
        pos.stop_price = round(max(initial, trail, old_stop), 4)
    else:
        trough = min(float(pos.trough_price or entry), float(bar_low))
        pos.trough_price = trough
        fav = (entry - trough) / entry
        if fav < activate:
            return False
        trail = trough * (1.0 + trail_pct)
        pos.stop_price = round(min(initial, trail, old_stop), 4)

    return pos.stop_price != old_stop


def update_open_trailing_stops(
    state,
    *,
    bar_fetcher,
    cfg: dict[str, Any],
) -> int:
    """
    Refresh trails for all open positions.

    ``bar_fetcher(ticker) -> (high, low) | None`` for the current session.
    Returns count of positions whose stop moved.
    """
    n = 0
    for pos in state.positions:
        bars = bar_fetcher(pos.ticker.upper())
        if bars is None:
            continue
        hi, lo = bars
        if update_position_trail(pos, hi, lo, cfg):
            n += 1
    return n


def stop_exit_label(pos: Position) -> str:
    """Human-readable exit tag for ledger / reports."""
    if pos.initial_stop_price is None:
        return f"Stop hit ({pos.side})"
    init = float(pos.initial_stop_price)
    cur = float(pos.stop_price)
    if pos.side == "long" and cur > init + 1e-8:
        return f"Trailing stop hit ({pos.side})"
    if pos.side == "short" and cur < init - 1e-8:
        return f"Trailing stop hit ({pos.side})"
    return f"Stop hit ({pos.side})"
