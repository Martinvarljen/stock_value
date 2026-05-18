"""Paper portfolio: apply decisions to state and ledger.

Cost model (T212-style CFD)
---------------------------
* **Margin** = cash reserved per slot (~``position_frac`` × NAV).
* **Exposure** = margin × ``cfd_leverage`` (default 5×) for **both** long and short.
* P&L scales on **exposure**; stops/TP are on the underlying price move.
* Per-leg ``commission_bps`` + ``slippage_bps``.
* **Overnight interest** on **long and short** exposure (``overnight_interest_bps_annual``),
  accrued once per calendar day after the entry day.

Config: ``cfd_leverage`` (preferred), or ``short_leverage`` / ``long_leverage`` (legacy aliases).
Set ``cfd_leverage`` to ``1`` to disable leverage on both sides.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from portfolio.decisions import Action, TickerDecision
from portfolio.exit_policy import placeholder_take_profit_price, take_profit_enabled
from portfolio.store import Position, PortfolioState, append_ledger
from portfolio.trailing_stop import seed_trail_fields, stop_exit_label


def cfd_leverage(cfg: dict[str, Any]) -> float:
    """Leverage for long and short CFD legs (``1`` = cash fully funded)."""
    for key in ("cfd_leverage", "short_leverage", "long_leverage"):
        if key in cfg and cfg[key] is not None:
            return max(1.0, float(cfg[key]))
    return 1.0


def _cost_params(cfg: dict[str, Any]) -> tuple[float, float]:
    """Return (cost_one_way, overnight_daily_on_exposure)."""
    commission_bps = float(cfg.get("commission_bps", 1.0))
    slippage_bps = float(cfg.get("slippage_bps", 2.0))
    overnight_annual = float(
        cfg.get(
            "overnight_interest_bps_annual",
            cfg.get("borrow_bps_annual", 50.0),
        )
    )
    cost_one_way = (commission_bps + slippage_bps) / 10_000.0
    overnight_daily = overnight_annual / 10_000.0 / 252.0
    return cost_one_way, overnight_daily


def position_exposure(pos: Position, cfg: dict[str, Any] | None = None) -> float:
    """Mark-to-market / risk notional (full CFD exposure)."""
    if pos.margin is not None:
        return pos.notional
    lev = cfd_leverage(cfg or {})
    return pos.notional * lev


def _leveraged_pnl_cash(
    margin: float,
    exposure: float,
    entry: float,
    exit_px: float,
    side: str,
) -> float:
    """Cash returned on close: margin + P&L on exposure."""
    if entry <= 0:
        return margin
    if side == "long":
        pnl = exposure * (exit_px - entry) / entry
    else:
        pnl = exposure * (entry - exit_px) / entry
    return margin + pnl


def accrue_overnight_interest(
    state: PortfolioState,
    *,
    run_date: date,
    cfg: dict[str, Any],
    write_ledger: bool = False,
) -> list[dict[str, Any]]:
    """Deduct one day's overnight funding per open CFD leg (long or short)."""
    _, overnight_daily = _cost_params(cfg)
    if overnight_daily <= 0:
        return []
    run_s = run_date.isoformat()
    rows: list[dict[str, Any]] = []
    for pos in state.positions:
        if pos.days_held(run_date) < 1:
            continue
        exposure = position_exposure(pos, cfg)
        charge = exposure * overnight_daily
        if charge <= 0:
            continue
        state.cash -= charge
        row = {
            "date": run_s,
            "ticker": pos.ticker,
            "action": "OVERNIGHT_INTEREST",
            "side": pos.side,
            "exposure": round(exposure, 6),
            "overnight_charge": round(charge, 6),
            "days_held": pos.days_held(run_date),
        }
        if write_ledger:
            append_ledger(row)
        rows.append(row)
    return rows


# Backward-compatible alias
accrue_short_overnight_interest = accrue_overnight_interest


def mark_nav_prices(
    state: PortfolioState,
    prices: dict[str, float],
    *,
    cfg: dict[str, Any] | None = None,
) -> None:
    """Mark open positions to market using close prices."""
    mtm = state.cash
    for p in state.positions:
        px = prices.get(p.ticker.upper()) or p.entry_price
        margin = p.position_margin()
        mtm += _leveraged_pnl_cash(margin, p.notional, p.entry_price, px, p.side)
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
    tp_pct = float(cfg.get("take_profit_pct", 0.40))
    max_hold = int(cfg.get("max_hold_days", 25))
    scale = float(cfg.get("_regime_scale", 1.0))
    eff_frac = frac * scale
    if scale >= float(cfg.get("bull_scale_threshold", 0.99)):
        eff_frac *= float(cfg.get("bull_position_frac_mult", 1.0))
    cost_one_way, _ = _cost_params(cfg)
    lev = cfd_leverage(cfg)

    atr_stop_mult = float(cfg.get("atr_stop_mult", 0.0))
    atr_tp_mult = float(cfg.get("atr_tp_mult", 0.0))
    atr_min_stop_pct = float(cfg.get("atr_min_stop_pct", 0.04))
    atr_max_stop_pct = float(cfg.get("atr_max_stop_pct", 0.30))

    vol_target = float(cfg.get("vol_target_annual_pct", 0.0))
    vol_size_floor = float(cfg.get("vol_size_floor", 0.25))
    vol_size_cap = float(cfg.get("vol_size_cap", 2.0))

    written: list[dict[str, Any]] = []
    run_s = run_date.isoformat()

    written.extend(
        accrue_overnight_interest(state, run_date=run_date, cfg=cfg, write_ledger=write_ledger)
    )

    for d in decisions:
        if d.action != Action.EXIT or d.price is None:
            continue
        pos = state.position_for(d.ticker)
        if not pos:
            continue

        fill_kind = "close"
        px = d.price
        lo = d.intraday_low
        hi = d.intraday_high
        op = d.open_price
        if pos.side == "long":
            if lo is not None and lo <= pos.stop_price:
                px = op if op is not None and op < pos.stop_price else pos.stop_price
                fill_kind = "stop_touched"
                d.reason = stop_exit_label(pos) if not d.reason else d.reason
            elif take_profit_enabled(cfg) and hi is not None and hi >= pos.take_profit_price:
                px = op if op is not None and op > pos.take_profit_price else pos.take_profit_price
                fill_kind = "tp_touched"
        else:
            if hi is not None and hi >= pos.stop_price:
                px = op if op is not None and op > pos.stop_price else pos.stop_price
                fill_kind = "stop_touched"
                d.reason = stop_exit_label(pos) if not d.reason else d.reason
            elif take_profit_enabled(cfg) and lo is not None and lo <= pos.take_profit_price:
                px = op if op is not None and op < pos.take_profit_price else pos.take_profit_price
                fill_kind = "tp_touched"

        held = pos.days_held(run_date)
        margin = pos.position_margin()
        exposure = pos.notional
        if pos.entry_price > 0:
            if pos.side == "long":
                pnl_pct = (px - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - px) / pos.entry_price
        else:
            pnl_pct = 0.0
        gross_proceeds = _leveraged_pnl_cash(margin, exposure, pos.entry_price, px, pos.side)
        exit_cost = gross_proceeds * cost_one_way
        net_proceeds = gross_proceeds - exit_cost
        state.cash += net_proceeds
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": Action.EXIT.value,
            "side": pos.side,
            "price": px,
            "fill_kind": fill_kind,
            "close_price": d.price,
            "margin": round(margin, 6),
            "exposure": round(exposure, 6),
            "notional": pos.notional,
            "pnl_pct": round(pnl_pct, 4),
            "exit_cost": round(exit_cost, 6),
            "net_proceeds": round(net_proceeds, 6),
            "days_held": held,
            "cfd_leverage": lev,
            "reason": d.reason,
        }
        if write_ledger:
            append_ledger(row)
        written.append(row)
        state.positions = [p for p in state.positions if p.ticker.upper() != d.ticker.upper()]

    for d in decisions:
        if d.action not in (Action.ENTER_LONG, Action.ENTER_SHORT) or d.price is None:
            continue
        if state.position_for(d.ticker):
            continue
        if len(state.positions) >= max_pos:
            continue
        side = "long" if d.action == Action.ENTER_LONG else "short"
        per_position_frac = frac * scale
        if scale >= float(cfg.get("bull_scale_threshold", 0.99)):
            per_position_frac *= float(cfg.get("bull_position_frac_mult", 1.0))
        if side == "short":
            per_position_frac *= float(cfg.get("short_position_frac_mult", 1.0))
        if vol_target > 0 and d.vol_60d_annual is not None and d.vol_60d_annual > 1e-6:
            scale_v = vol_target / float(d.vol_60d_annual)
            scale_v = max(vol_size_floor, min(vol_size_cap, scale_v))
            per_position_frac *= scale_v
        budget = min(per_position_frac * state.nav, state.cash * 0.999)
        if budget < 1e-6:
            continue
        entry_cost = budget * cost_one_way
        px = d.price

        margin = budget - entry_cost
        if margin <= 0:
            continue
        exposure = margin * lev
        cash_out = margin

        if atr_stop_mult > 0 and d.atr_pct is not None and d.atr_pct > 0:
            stop_distance_pct = max(
                atr_min_stop_pct,
                min(atr_max_stop_pct, atr_stop_mult * float(d.atr_pct)),
            )
        else:
            stop_distance_pct = stop_pct
        if take_profit_enabled(cfg):
            if atr_tp_mult > 0 and d.atr_pct is not None and d.atr_pct > 0:
                tp_distance_pct = max(0.02, atr_tp_mult * float(d.atr_pct))
            else:
                tp_distance_pct = tp_pct
            if side == "long":
                tp = px * (1 + tp_distance_pct)
            else:
                tp = px * (1 - tp_distance_pct)
        else:
            tp = placeholder_take_profit_price(px, side)
        if side == "long":
            stop = px * (1 - stop_distance_pct)
        else:
            stop = px * (1 + stop_distance_pct)
        pos = Position(
            ticker=d.ticker,
            side=side,
            entry_date=run_s,
            entry_price=px,
            notional=round(exposure, 6),
            stop_price=round(stop, 4),
            take_profit_price=round(tp, 4),
            max_hold_days=max_hold,
            p_up_20d_at_entry=d.p_up_20d,
            entry_reason=d.reason,
            margin=round(margin, 6),
        )
        seed_trail_fields(pos, cfg)
        state.positions.append(pos)
        state.cash -= cash_out
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": d.action.value,
            "side": side,
            "price": px,
            "margin": round(margin, 6),
            "exposure": round(exposure, 6),
            "notional": round(exposure, 6),
            "entry_cost": round(entry_cost, 6),
            "budget_pre_cost": round(budget, 6),
            "cfd_leverage": lev,
            "reason": d.reason,
            "p_up_20d": d.p_up_20d,
        }
        if write_ledger:
            append_ledger(row)
        written.append(row)

    state.last_run_date = run_s
    prices = {d.ticker.upper(): float(d.price) for d in decisions if d.price is not None}
    mark_nav_prices(state, prices, cfg=cfg)
    return written
