"""
performance_metrics.py — Summary stats from an equity curve or period returns.

Research / reporting only (implementation brief §9.2). Not financial advice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def _annualized_return(total_return: float, n_periods: int, periods_per_year: float = 252.0) -> float:
    if n_periods < 2 or total_return <= -1:
        return float("nan")
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def summarize_backtest(
    period_returns: np.ndarray,
    equity: np.ndarray,
    *,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """
    period_returns: net strategy returns per bar (already after costs), may contain NaN.
    equity: cumulative equity path (length = len(returns)+1 typical).
    """
    r = np.asarray(period_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {"n_periods": 0, "error": "no_returns"}

    total_return = float(np.prod(1.0 + r) - 1.0)
    vol = float(np.std(r, ddof=1)) if n > 1 else 0.0
    mu = float(np.mean(r))
    sharpe = (mu / vol) * math.sqrt(periods_per_year) if vol > 1e-12 else 0.0

    neg = r[r < 0]
    downside = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mu / downside) * math.sqrt(periods_per_year) if downside > 1e-12 else 0.0

    gains = float(np.sum(r[r > 0]))
    losses = float(np.sum(r[r < 0]))
    profit_factor = gains / abs(losses) if losses < -1e-12 else float("inf")

    wins = int(np.sum(r > 0))
    win_rate = wins / n if n else 0.0

    eq = np.asarray(equity, dtype=float)
    mdd = max_drawdown(eq)
    cagr = _annualized_return(total_return, n, periods_per_year)
    calmar = cagr / abs(mdd) if mdd < -1e-6 and math.isfinite(cagr) else float("nan")

    return {
        "n_periods": n,
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6) if math.isfinite(cagr) else None,
        "vol_annual": round(vol * math.sqrt(periods_per_year), 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 6),
        "calmar": round(calmar, 4) if math.isfinite(calmar) else None,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "avg_period_return": round(mu, 8),
    }
