"""Exit policy — regime-aware score exits, hold gates, take-profit vs trail."""

from __future__ import annotations

from typing import Any

from portfolio.trailing_stop import trailing_enabled


def take_profit_enabled(cfg: dict[str, Any]) -> bool:
    """Fixed TP is off when explicitly disabled or when trailing owns exits."""
    if cfg.get("use_take_profit") is False:
        return False
    if trailing_enabled(cfg) and not cfg.get("use_take_profit_with_trail", False):
        return False
    return float(cfg.get("take_profit_pct", 0) or 0) > 0


def placeholder_take_profit_price(entry: float, side: str) -> float:
    """Unreachable level so intraday/close logic never triggers TP."""
    if side == "long":
        return entry * 1e9
    return max(entry * 1e-9, 1e-12)


def long_score_exit_threshold(regime: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    """
    P(up) floor for long score-exit, or ``None`` if disabled for this regime.

    When ``score_exit_long_only_bear_regime`` is true, no score exit while SPY is
    above its 200d MA (``spy_bull``). Otherwise use ``exit_p_up_long`` in bear and
    ``exit_p_up_long_bull`` in bull (default 0.43 if unset).
    """
    bull = bool(regime.get("spy_bull"))
    if cfg.get("score_exit_long_only_bear_regime", False):
        if bull:
            return None
        return float(cfg.get("exit_p_up_long", 0.36))
    if bull and cfg.get("exit_p_up_long_bull") is not None:
        return float(cfg["exit_p_up_long_bull"])
    return float(cfg.get("exit_p_up_long", 0.36))


def min_hold_before_score_exit(cfg: dict[str, Any], side: str) -> int:
    key = (
        "min_hold_days_before_score_exit_long"
        if side == "long"
        else "min_hold_days_before_score_exit_short"
    )
    if key in cfg:
        return max(0, int(cfg[key]))
    return max(0, int(cfg.get("min_hold_days_before_score_exit", 0)))


def score_exit_blocked_by_hold(held_days: int, side: str, cfg: dict[str, Any]) -> bool:
    return held_days < min_hold_before_score_exit(cfg, side)


def should_exit_long_on_regime(regime: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Flat longs when SPY is not in confirmed bull (bear or unknown)."""
    if not cfg.get("exit_long_when_regime_not_bull", True):
        return False
    if cfg.get("long_entry_requires_bull_regime", False):
        return regime.get("regime_signal") != "bull"
    return not bool(regime.get("spy_bull"))


def long_entry_allowed(regime: dict[str, Any], cfg: dict[str, Any], scale: float) -> bool:
    """Long only in confirmed bull (SPY above 200d MA) unless configured otherwise."""
    if cfg.get("long_entry_requires_bull_regime", False):
        return regime.get("regime_signal") == "bull"
    floor = float(cfg.get("long_entry_min_regime_scale", cfg.get("bear_scale", 0.35)))
    return scale >= floor


def short_entry_allowed(regime: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """
    Short only when macro filters pass (confirmed bear + optional stress gates).

    Stricter mode (``research_ls`` profile) requires full risk-off sizing,
    SPY sufficiently below the 200d MA, and negative SPY 20d momentum so
    shorts are hedges in real drawdowns, not early bear probes.
    """
    if not cfg.get("enable_short", True):
        return False
    if cfg.get("short_entry_requires_bear_regime", True):
        if regime.get("regime_signal") != "bear":
            return False
    elif cfg.get("regime_filter", True):
        scale = float(regime.get("gross_exposure_scale", 1.0))
        if scale >= float(cfg.get("short_entry_max_regime_scale", 0.99)):
            return False

    if cfg.get("short_requires_full_risk_off", False):
        bear_scale = float(cfg.get("bear_scale", 0.35))
        scale = float(regime.get("gross_exposure_scale", 1.0))
        if scale > bear_scale + 0.01:
            return False

    min_below = cfg.get("short_min_spy_below_ma_pct")
    if min_below is not None:
        below = regime.get("spy_pct_below_ma200")
        if below is None or float(below) < float(min_below):
            return False

    max_spy_ret = cfg.get("short_max_spy_return_20d")
    if max_spy_ret is not None:
        spy_ret = regime.get("spy_return_20d")
        if spy_ret is None or float(spy_ret) > float(max_spy_ret):
            return False

    return True


def short_ticker_entry_allowed(analysis: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Per-name short gate after macro ``short_entry_allowed`` passes."""
    p_up = analysis.get("p_up_20d")
    if p_up is None:
        return False
    ceiling = float(cfg.get("max_p_up_short", 0.38))
    if float(p_up) > ceiling:
        return False
    floor = cfg.get("short_min_p_up_floor")
    if floor is not None and float(p_up) < float(floor):
        return False
    return True


def should_cover_short_on_regime(regime: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Cover open shorts when regime is not confirmed bear (bull or unknown)."""
    if not cfg.get("cover_short_when_regime_not_bear", True):
        return False
    if cfg.get("short_entry_requires_bear_regime", True):
        return regime.get("regime_signal") != "bear"
    return bool(regime.get("spy_bull"))


def short_score_exit_threshold(position, cfg: dict[str, Any]) -> float:
    """
    Cover short when P(up) rises above this level.

    Defaults to 0.48 (not 0.55). Optionally tie to entry + ``short_exit_p_up_delta``.
    """
    cap = float(cfg.get("exit_p_up_short", 0.48))
    entry_p = getattr(position, "p_up_20d_at_entry", None)
    if cfg.get("short_exit_p_up_relative_to_entry", True) and entry_p is not None:
        delta = float(cfg.get("short_exit_p_up_delta", 0.10))
        return min(cap, float(entry_p) + delta)
    return cap
