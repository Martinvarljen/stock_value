"""Post-run checks for agent backtests (NAV, limits, ledger sanity)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from portfolio.broker import cfd_leverage, position_exposure
from portfolio.store import PortfolioState


def _gross_exposure_pct(state: PortfolioState, cfg: dict[str, Any]) -> float:
    if state.nav <= 0:
        return 0.0
    gross = sum(position_exposure(p, cfg) for p in state.positions)
    return gross / state.nav


def validate_backtest_run(
    *,
    curve: pd.DataFrame,
    ledger: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    cfg: dict[str, Any],
    stats: dict[str, Any],
) -> list[str]:
    """Return human-readable invariant violations (empty == pass)."""
    errors: list[str] = []

    if curve.empty:
        errors.append("Equity curve is empty.")
        return errors

    strat = curve["strategy"].astype(float)
    if strat.isna().any():
        errors.append("Strategy NAV series contains NaN.")
    if (strat <= 0).any():
        errors.append("Strategy NAV went non-positive.")

    if not strat.is_monotonic_increasing and strat.iloc[-1] < strat.iloc[0]:
        pass  # drawdowns are fine
    peak = strat.cummax()
    dd = (strat - peak) / peak.replace(0, float("nan"))
    reported_mdd = stats.get("strategy_max_dd")
    if reported_mdd is not None and math.isfinite(reported_mdd):
        computed = float(dd.min()) if len(dd) else 0.0
        if abs(computed - float(reported_mdd)) > 0.02:
            errors.append(
                f"Max drawdown mismatch: summary {reported_mdd:.4f} vs curve {computed:.4f}."
            )

    limits_cfg = cfg.get("risk_limits") or {}
    max_gross_cap = float(limits_cfg.get("max_gross_exposure_pct", 1.2))
    observed_max = float(stats.get("max_gross_exposure", 0.0))
    if limits_cfg.get("enabled", True) and observed_max > max_gross_cap + 0.05:
        errors.append(
            f"Gross exposure {observed_max:.2%} exceeded cap {max_gross_cap:.2%} "
            f"(+5% tolerance)."
        )

    for row in ledger:
        act = row.get("action", "")
        if act.startswith("ENTER") and row.get("price") in (None, 0):
            errors.append(f"Entry without price: {row.get('ticker')} on {row.get('date')}.")
            break

    enter_tickers = {
        (str(r.get("ticker", "")).upper(), r.get("side") or "long")
        for r in ledger
        if str(r.get("action", "")).startswith("ENTER")
    }
    exit_keys = {
        (str(r.get("ticker", "")).upper(), r.get("side") or "long")
        for r in ledger
        if r.get("action") == "EXIT"
    }
    orphan_exits = exit_keys - enter_tickers
    if orphan_exits:
        tk, side = next(iter(orphan_exits))
        errors.append(f"EXIT without matching ENTER: {tk} ({side}).")

    if snapshots:
        navs = [float(s.get("nav", 0)) for s in snapshots]
        if any(n <= 0 for n in navs):
            errors.append("Snapshot NAV non-positive.")

    return errors
