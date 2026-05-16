"""Trade actions for one day — deterministic rules on ML score and regime."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from portfolio.store import Position, PortfolioState


class Action(str, Enum):
    NO_TRADE = "NO_TRADE"
    HOLD = "HOLD"
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT = "EXIT"


@dataclass
class TickerDecision:
    ticker: str
    action: Action
    reason: str
    ml_score: float | None = None
    quintile: int | None = None
    p_up_20d: float | None = None
    price: float | None = None
    # Risk inputs surfaced for the broker (ATR-anchored stops, vol-targeted
    # sizing). ``None`` falls back to the legacy fixed-percent stop / flat
    # ``position_frac`` sizing.
    atr_pct: float | None = None
    vol_60d_annual: float | None = None
    # Intraday range for the bar this decision was made on. Used by the
    # broker to fill stops at the actual stop level (with slippage) when
    # the day's range touched the stop, instead of marking the entire
    # exit at the closing print. Without this, a 30%-gap-down day fills
    # the stop only at close — silently overstating returns. ``None``
    # falls back to close-price exits.
    intraday_low: float | None = None
    intraday_high: float | None = None
    open_price: float | None = None


# Minimum cohort size for a meaningful cross-sectional quintile assignment.
# With <50 scored names, Q5 is 1-9 names and the cohort drifts day-to-day —
# a quintile threshold of "Q≥4" then silently changes meaning. Below this
# floor we return an empty map so callers fall back to absolute thresholds
# (e.g. ``min_p_up_long``) instead of cross-sectional rank.
QUINTILE_MIN_COHORT = 50


def _quintile_map(analyses: list[dict[str, Any]]) -> dict[str, int]:
    scored = [(a["ticker"], a["ml_score"]) for a in analyses if a.get("ok") and a.get("ml_score") is not None]
    if len(scored) < QUINTILE_MIN_COHORT:
        return {}
    # Lazy import keeps ``portfolio.decisions`` numpy-free at import time;
    # numpy is only paid for when a real cross-sectional run happens.
    from backtesting.ml_quant import assign_quintile

    tickers, scores = zip(*scored)
    qs = assign_quintile(list(scores))
    return dict(zip(tickers, qs))


def _stop_hit(pos: Position, price: float, cfg: dict) -> bool:
    if pos.side == "long":
        return price <= pos.stop_price
    return price >= pos.stop_price


def _tp_hit(pos: Position, price: float) -> bool:
    if pos.side == "long":
        return price >= pos.take_profit_price
    return price <= pos.take_profit_price


def decide_ticker(
    analysis: dict[str, Any],
    position: Position | None,
    *,
    quintile: int | None,
    regime: dict[str, Any],
    cfg: dict[str, Any],
    as_of: date,
) -> TickerDecision:
    ticker = analysis["ticker"]
    price = analysis.get("price")
    p_up = analysis.get("p_up_20d")
    score = analysis.get("ml_score")
    atr_pct = analysis.get("atr_pct")
    vol_60d = analysis.get("vol_60d_annual")
    bar_low = analysis.get("bar_low")
    bar_high = analysis.get("bar_high")
    bar_open = analysis.get("bar_open")
    base = TickerDecision(
        ticker=ticker,
        action=Action.NO_TRADE,
        reason="No signal",
        ml_score=score,
        quintile=quintile,
        p_up_20d=p_up,
        price=float(price) if price is not None else None,
        atr_pct=float(atr_pct) if atr_pct is not None else None,
        vol_60d_annual=float(vol_60d) if vol_60d is not None else None,
        intraday_low=float(bar_low) if bar_low is not None else None,
        intraday_high=float(bar_high) if bar_high is not None else None,
        open_price=float(bar_open) if bar_open is not None else None,
    )

    if not analysis.get("ok"):
        base.reason = analysis.get("error") or "Analysis failed"
        return base

    if analysis.get("critical_flags"):
        if position:
            base.action = Action.EXIT
            base.reason = "Critical flags — exit open position"
            return base
        base.reason = "Critical flags — no new entry"
        return base

    if price is None or float(price) <= 0:
        base.reason = "No price"
        return base

    price_f = float(price)
    scale = float(regime.get("gross_exposure_scale", 1.0))
    est_hold = int(cfg.get("estimated_hold_days", 20))
    max_hold = int(cfg.get("max_hold_days", 25))

    if position:
        held = position.days_held(as_of)
        if _stop_hit(position, price_f, cfg):
            base.action = Action.EXIT
            base.reason = f"Stop hit ({position.side})"
            return base
        if _tp_hit(position, price_f):
            base.action = Action.EXIT
            base.reason = f"Take-profit hit ({position.side})"
            return base
        if held >= max_hold:
            base.action = Action.EXIT
            base.reason = f"Max hold {max_hold}d"
            return base

        if position.side == "long":
            exit_p = float(cfg.get("exit_p_up_long", 0.45))
            if p_up is not None and float(p_up) < exit_p:
                base.action = Action.EXIT
                base.reason = f"P(up) 20d {float(p_up):.0%} < exit {exit_p:.0%}"
                return base
        else:
            exit_p = float(cfg.get("exit_p_up_short", 0.55))
            if p_up is not None and float(p_up) > exit_p:
                base.action = Action.EXIT
                base.reason = f"P(up) 20d {float(p_up):.0%} > cover {exit_p:.0%}"
                return base

        rem = position.estimated_days_remaining(est_hold, as_of)
        base.action = Action.HOLD
        base.reason = f"Holding; ~{rem}d est. remaining"
        return base

    # Ticker-level ML output gate. ``quintile`` may legitimately be ``None``
    # when the day's scored cohort was too small (see ``QUINTILE_MIN_COHORT``);
    # we fall back to a stricter absolute-threshold gate in that case.
    if p_up is None or score is None:
        base.reason = "Missing ML score"
        return base

    min_long = float(cfg.get("min_p_up_long", 0.58))
    max_short = float(cfg.get("max_p_up_short", 0.42))
    q_long = int(cfg.get("long_quintile_min", 4))
    q_short = int(cfg.get("short_quintile_max", 2))
    # Absolute fallback thresholds add a buffer when the cross-sectional
    # cohort isn't large enough to trust quintile rank.
    abs_long_buffer = float(cfg.get("min_p_up_long_abs_buffer", 0.04))
    abs_short_buffer = float(cfg.get("max_p_up_short_abs_buffer", 0.04))

    if cfg.get("regime_filter", True) and scale < 0.2:
        base.reason = "Regime: exposure scaled to zero"
        return base

    if quintile is None:
        if float(p_up) >= min_long + abs_long_buffer and scale >= 0.5:
            base.action = Action.ENTER_LONG
            base.reason = f"P(up)20d={float(p_up):.0%} (small cohort, abs gate)"
            return base
        if cfg.get("enable_short", True) and float(p_up) <= max_short - abs_short_buffer:
            if not cfg.get("regime_filter", True) or scale < 1.0:
                base.action = Action.ENTER_SHORT
                base.reason = f"P(up)20d={float(p_up):.0%} (small cohort, abs gate, bearish)"
                return base
        base.reason = f"P(up)20d={float(p_up):.0%} — small cohort, no entry"
        return base

    if quintile >= q_long and float(p_up) >= min_long and scale >= 0.5:
        base.action = Action.ENTER_LONG
        base.reason = f"Q{quintile} P(up)20d={float(p_up):.0%}"
        return base

    if cfg.get("enable_short", True) and quintile <= q_short and float(p_up) <= max_short:
        if not cfg.get("regime_filter", True) or scale < 1.0:
            base.action = Action.ENTER_SHORT
            base.reason = f"Q{quintile} P(up)20d={float(p_up):.0%} (bearish)"
            return base

    base.reason = f"Q{quintile} P(up)20d={float(p_up):.0%} — no entry"
    return base


def decide_universe(
    analyses: list[dict[str, Any]],
    state: PortfolioState,
    regime: dict[str, Any],
    cfg: dict[str, Any],
    as_of: date,
) -> list[TickerDecision]:
    qmap = _quintile_map(analyses)
    decisions: list[TickerDecision] = []
    for a in analyses:
        pos = state.position_for(a["ticker"])
        d = decide_ticker(
            a,
            pos,
            quintile=qmap.get(a["ticker"]),
            regime=regime,
            cfg=cfg,
            as_of=as_of,
        )
        decisions.append(d)
    return decisions


def prioritize_entries(decisions: list[TickerDecision], cfg: dict[str, Any]) -> list[TickerDecision]:
    """Cap new entries per day; prefer higher ml_score."""
    max_new = int(cfg.get("max_new_entries_per_day", 3))
    entries = [d for d in decisions if d.action in (Action.ENTER_LONG, Action.ENTER_SHORT)]
    entries.sort(key=lambda x: (x.ml_score or 0), reverse=True)
    allowed = {d.ticker for d in entries[:max_new]}
    out: list[TickerDecision] = []
    for d in decisions:
        if d.action in (Action.ENTER_LONG, Action.ENTER_SHORT) and d.ticker not in allowed:
            out.append(
                TickerDecision(
                    ticker=d.ticker,
                    action=Action.NO_TRADE,
                    reason=f"Deferred (daily entry cap {max_new})",
                    ml_score=d.ml_score,
                    quintile=d.quintile,
                    p_up_20d=d.p_up_20d,
                    price=d.price,
                )
            )
        else:
            out.append(d)
    return out
