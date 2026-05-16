"""Paper portfolio: apply decisions to state and ledger.

Adds an explicit cost model: per-leg ``commission_bps`` + ``slippage_bps``
on every entry and exit, plus an annualised short ``borrow_bps_annual``
accrued over days held. Defaults are conservative for liquid US large-cap.

Cost rules
----------
* On entry: actual invested principal = max budget - one-way cost; cash
  reduces by the full max budget (so the cost line is real, not a
  rounding-error). The position's stored ``notional`` is the invested
  principal so MTM is correct.
* On exit: PnL is reduced by the one-way cost on the exit leg. Shorts
  also pay accumulated borrow over ``days_held``.

Earlier behaviour assumed zero costs and zero borrow, which on a
10-position quarterly book at 25-day holds silently overstated returns by
roughly 200-400 bps annualised.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from portfolio.decisions import Action, TickerDecision
from portfolio.store import Position, PortfolioState, append_ledger


# ── cost knobs ─────────────────────────────────────────────────────────────

def _cost_params(cfg: dict[str, Any]) -> tuple[float, float]:
    """Return (cost_one_way, borrow_daily) from config."""
    commission_bps = float(cfg.get("commission_bps", 1.0))
    slippage_bps = float(cfg.get("slippage_bps", 2.0))
    borrow_bps_annual = float(cfg.get("borrow_bps_annual", 50.0))
    cost_one_way = (commission_bps + slippage_bps) / 10_000.0
    borrow_daily = borrow_bps_annual / 10_000.0 / 252.0
    return cost_one_way, borrow_daily


# ── NAV mark ───────────────────────────────────────────────────────────────

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


# ── decisions -> ledger ────────────────────────────────────────────────────

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
    cost_one_way, borrow_daily = _cost_params(cfg)

    # ATR-anchored stop knobs (used when ``d.atr_pct`` is available; flat
    # percent stop is the fallback for back-compat). ``atr_stop_mult`` of
    # 2.5 places the stop at 2.5 * ATR(14) below entry — Wilder-style
    # stop placement, way more robust than a flat 20% to vol regime.
    atr_stop_mult = float(cfg.get("atr_stop_mult", 0.0))  # 0 disables
    atr_tp_mult = float(cfg.get("atr_tp_mult", 0.0))      # 0 disables
    atr_min_stop_pct = float(cfg.get("atr_min_stop_pct", 0.04))
    atr_max_stop_pct = float(cfg.get("atr_max_stop_pct", 0.30))

    # Vol-targeted sizing knobs. Setting ``vol_target_annual_pct`` switches
    # sizing from flat ``position_frac`` to risk-parity:
    #   size = clip( target_vol / realised_vol_60d, vol_size_floor,
    #                vol_size_cap ) * position_frac
    # so a 50% vol name gets a fraction of the size of a 12% vol name.
    vol_target = float(cfg.get("vol_target_annual_pct", 0.0))  # 0 disables
    vol_size_floor = float(cfg.get("vol_size_floor", 0.25))
    vol_size_cap = float(cfg.get("vol_size_cap", 2.0))

    written: list[dict[str, Any]] = []
    run_s = run_date.isoformat()

    # Exits first
    for d in decisions:
        if d.action != Action.EXIT or d.price is None:
            continue
        pos = state.position_for(d.ticker)
        if not pos:
            continue

        # Determine the realistic fill price.
        #
        # If the day's intraday range touched the stop or take-profit
        # level we should fill THERE (with one-way slippage), not at
        # the closing print — otherwise a -30% gap-down day registers
        # the stop only at close, overstating returns badly. When the
        # range opened *through* the stop (gap), the open is the worst
        # achievable price and we fill there.
        #
        # When intraday range isn't supplied (live trading or legacy
        # callers) we fall back to the close-price exit.
        fill_kind = "close"
        px = d.price
        lo = d.intraday_low
        hi = d.intraday_high
        op = d.open_price
        if pos.side == "long":
            if lo is not None and lo <= pos.stop_price:
                # Stop touched intraday. Fill at stop, but if the day
                # opened below the stop (gap-down), fill at open.
                if op is not None and op < pos.stop_price:
                    px = op
                else:
                    px = pos.stop_price
                fill_kind = "stop_touched"
            elif hi is not None and hi >= pos.take_profit_price:
                if op is not None and op > pos.take_profit_price:
                    px = op
                else:
                    px = pos.take_profit_price
                fill_kind = "tp_touched"
        else:
            if hi is not None and hi >= pos.stop_price:
                if op is not None and op > pos.stop_price:
                    px = op
                else:
                    px = pos.stop_price
                fill_kind = "stop_touched"
            elif lo is not None and lo <= pos.take_profit_price:
                if op is not None and op < pos.take_profit_price:
                    px = op
                else:
                    px = pos.take_profit_price
                fill_kind = "tp_touched"

        held = pos.days_held(run_date)
        if pos.side == "long":
            pnl_pct = (px - pos.entry_price) / pos.entry_price
            gross_proceeds = pos.notional * (1 + pnl_pct)
            exit_cost = gross_proceeds * cost_one_way
            net_proceeds = gross_proceeds - exit_cost
            borrow_charge = 0.0
        else:
            pnl_pct = (pos.entry_price - px) / pos.entry_price
            gross_proceeds = pos.notional * (1 + pnl_pct)
            exit_cost = gross_proceeds * cost_one_way
            borrow_charge = pos.notional * borrow_daily * max(held, 0)
            net_proceeds = gross_proceeds - exit_cost - borrow_charge
        state.cash += net_proceeds
        # The reported pnl_pct already reflects the price move; the
        # ledger separately records the cost / borrow so the audit trail
        # remains transparent.
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": Action.EXIT.value,
            "side": pos.side,
            "price": px,
            "fill_kind": fill_kind,
            "close_price": d.price,
            "notional": pos.notional,
            "pnl_pct": round(pnl_pct, 4),
            "exit_cost": round(exit_cost, 6),
            "borrow_charge": round(borrow_charge, 6),
            "net_proceeds": round(net_proceeds, 6),
            "days_held": held,
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
        # ── sizing ──
        per_position_frac = eff_frac
        if vol_target > 0 and d.vol_60d_annual is not None and d.vol_60d_annual > 1e-6:
            scale_v = vol_target / float(d.vol_60d_annual)
            scale_v = max(vol_size_floor, min(vol_size_cap, scale_v))
            per_position_frac = eff_frac * scale_v
        budget = min(per_position_frac * state.nav, state.cash * 0.999)
        if budget < 1e-6:
            continue
        entry_cost = budget * cost_one_way
        invested = budget - entry_cost
        if invested <= 0:
            continue
        px = d.price
        side = "long" if d.action == Action.ENTER_LONG else "short"
        # ── stop / take-profit placement ──
        if atr_stop_mult > 0 and d.atr_pct is not None and d.atr_pct > 0:
            stop_distance_pct = max(
                atr_min_stop_pct,
                min(atr_max_stop_pct, atr_stop_mult * float(d.atr_pct)),
            )
        else:
            stop_distance_pct = stop_pct
        if atr_tp_mult > 0 and d.atr_pct is not None and d.atr_pct > 0:
            tp_distance_pct = max(0.02, atr_tp_mult * float(d.atr_pct))
        else:
            tp_distance_pct = tp_pct
        if side == "long":
            stop = px * (1 - stop_distance_pct)
            tp = px * (1 + tp_distance_pct)
        else:
            stop = px * (1 + stop_distance_pct)
            tp = px * (1 - tp_distance_pct)
        pos = Position(
            ticker=d.ticker,
            side=side,
            entry_date=run_s,
            entry_price=px,
            notional=invested,
            stop_price=round(stop, 4),
            take_profit_price=round(tp, 4),
            max_hold_days=max_hold,
            p_up_20d_at_entry=d.p_up_20d,
            entry_reason=d.reason,
        )
        state.positions.append(pos)
        state.cash -= budget  # cash leaves at full budget; cost is real
        row = {
            "date": run_s,
            "ticker": d.ticker,
            "action": d.action.value,
            "side": side,
            "price": px,
            "notional": round(invested, 6),
            "entry_cost": round(entry_cost, 6),
            "budget_pre_cost": round(budget, 6),
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
