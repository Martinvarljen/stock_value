"""Paper portfolio: apply decisions to state and ledger."""

from __future__ import annotations

from datetime import date
from typing import Any

from portfolio.decisions import Action, TickerDecision
from portfolio.store import Position, PortfolioState, append_ledger


def mark_nav_prices(state: PortfolioState, prices: dict[str, float]) -> None:
    """Mark open positions to market using close prices."""
    mtm = state.cash
    for p in state.positions:
        px = prices.get(p.ticker.upper()) or p.entry_price
        if p.side == "long":
            mtm += p.notional * (1 + (px - p.entry_price) / p.entry_price)
        else:
            mtm += p.notional * (1 + (p.entry_price - px) / p.entry_price)
    state.nav = round(mtm, 6)


def apply_decisions(
    state: PortfolioState,
    decisions: list[TickerDecision],
    *,
    run_date: date,
    cfg: dict[str, Any],
    write_ledger: bool = True,
) -> list[dict[str, Any]]:
    """Mutate state; return ledger rows written today."""
    frac = float(cfg.get("position_frac", 0.1))
    max_pos = int(cfg.get("max_positions", 10))
    stop_pct = float(cfg.get("stop_loss_pct", 0.20))
    tp_pct = float(cfg.get("take_profit_pct", 0.25))
    max_hold = int(cfg.get("max_hold_days", 25))
    scale = float(cfg.get("_regime_scale", 1.0))
    eff_frac = frac * scale

    written: list[dict[str, Any]] = []
    run_s = run_date.isoformat()

    # Exits first
    for d in decisions:
        if d.action != Action.EXIT or d.price is None:
            continue
        pos = state.position_for(d.ticker)
        if not pos:
            continue
        px = d.price
        if pos.side == "long":
            pnl_pct = (px - pos.entry_price) / pos.entry_price
            state.cash += pos.notional * (1 + pnl_pct)
        else:
            pnl_pct = (pos.entry_price - px) / pos.entry_price
            state.cash += pos.notional * (1 + pnl_pct)
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": Action.EXIT.value,
            "side": pos.side,
            "price": px,
            "notional": pos.notional,
            "pnl_pct": round(pnl_pct, 4),
            "reason": d.reason,
        }
        if write_ledger:
            append_ledger(row)
        written.append(row)
        state.positions = [p for p in state.positions if p.ticker.upper() != d.ticker.upper()]

    # Entries
    for d in decisions:
        if d.action not in (Action.ENTER_LONG, Action.ENTER_SHORT) or d.price is None:
            continue
        if state.position_for(d.ticker):
            continue
        if len(state.positions) >= max_pos:
            continue
        if state.cash < 1e-6:
            continue
        notional = min(eff_frac * state.nav, state.cash * 0.999)
        if notional < 1e-6:
            continue
        px = d.price
        side = "long" if d.action == Action.ENTER_LONG else "short"
        if side == "long":
            stop = px * (1 - stop_pct)
            tp = px * (1 + tp_pct)
        else:
            stop = px * (1 + stop_pct)
            tp = px * (1 - tp_pct)
        pos = Position(
            ticker=d.ticker,
            side=side,
            entry_date=run_s,
            entry_price=px,
            notional=notional,
            stop_price=round(stop, 4),
            take_profit_price=round(tp, 4),
            max_hold_days=max_hold,
            p_up_20d_at_entry=d.p_up_20d,
            entry_reason=d.reason,
        )
        state.positions.append(pos)
        state.cash -= notional
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": d.action.value,
            "side": side,
            "price": px,
            "notional": round(notional, 6),
            "reason": d.reason,
            "p_up_20d": d.p_up_20d,
        }
        if write_ledger:
            append_ledger(row)
        written.append(row)

    state.last_run_date = run_s
    prices = {d.ticker.upper(): float(d.price) for d in decisions if d.price is not None}
    mark_nav_prices(state, prices)
    return written
