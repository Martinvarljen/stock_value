"""
kaufman_tsm.py — Indicators aligned with Perry J. Kaufman, *Trading Systems and Methods*
(5th ed., Wiley): momentum as first difference, moving-average context, time-based linear
trend / regression, and Kaufman's efficiency ratio (direction vs. path volatility).

Descriptive analytics only — not trading advice.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _linreg_on_last_n(y: np.ndarray) -> tuple[float, float, float, float]:
    """
    OLS on y[0..n-1] with x = 0..n-1 (oldest → newest).
    Returns slope, intercept, r_squared, y_hat_at_next_x (x=n).
    """
    n = len(y)
    if n < 3:
        return 0.0, float(y[-1]) if n else 0.0, 0.0, float(y[-1]) if n else 0.0
    x = np.arange(n, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    xc, yc = x - xm, y - ym
    var = float((xc**2).sum())
    if var < 1e-18:
        return 0.0, float(ym), 0.0, float(ym)
    slope = float((xc * yc).sum() / var)
    intercept = float(ym - slope * xm)
    pred_next = float(intercept + slope * n)
    y_hat = intercept + slope * x
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else 0.0
    return slope, intercept, max(0.0, min(1.0, r2)), pred_next


def efficiency_ratio(close: pd.Series, period: int) -> float:
    """
    Kaufman efficiency ratio: |net change over n bars| / sum(|day-to-day changes|).
    ∈ [0, 1] — higher means smoother directional move.
    """
    if len(close) < period + 1:
        return 0.0
    tail = close.iloc[-(period + 1) :].astype(float).values
    net = abs(float(tail[-1] - tail[0]))
    path = float(np.abs(np.diff(tail)).sum())
    return float(net / path) if path > 1e-12 else 0.0


def momentum_difference(close: pd.Series, n: int) -> float:
    """M_t = p_t − p_{t−n} (Kaufman: first difference over n)."""
    if len(close) <= n:
        return 0.0
    return float(close.iloc[-1]) - float(close.iloc[-1 - n])


def compute_kaufman_tsm(close: pd.Series, high: pd.Series, low: pd.Series) -> dict[str, Any]:
    """
    Last-bar summary of TSM-style trend diagnostics from daily OHLC.
    """
    c = close.astype(float).reset_index(drop=True)
    h = high.astype(float).reindex(c.index).fillna(c)
    l = low.astype(float).reindex(c.index).fillna(c)

    if len(c) < 80:
        return {
            "available": False,
            "reason": "Need at least ~80 daily bars for Kaufman-style regression/ER metrics.",
        }

    px = float(c.iloc[-1])
    if not math.isfinite(px) or px <= 0:
        return {"available": False, "reason": "Invalid last close."}

    er10 = efficiency_ratio(c, 10)
    er20 = efficiency_ratio(c, 20)

    mom5 = momentum_difference(c, 5)
    mom10 = momentum_difference(c, 10)
    mom20 = momentum_difference(c, 20)

    y20 = c.iloc[-20:].values.astype(np.float64)
    slope20, _int20, r2_20, pred20 = _linreg_on_last_n(y20)
    slope_norm_20 = slope20 / px
    forecast_1d_ret = (pred20 - px) / px

    y60 = c.iloc[-60:].values.astype(np.float64)
    slope60, _i60, r2_60, pred60 = _linreg_on_last_n(y60)
    slope_norm_60 = slope60 / px
    forecast_1d_ret_60 = (pred60 - px) / px

    # Simple composite bias in [-1, 1] for UI / downstream heuristics
    bias = (
        math.tanh((er10 - 0.35) * 4.0) * 0.35
        + math.tanh(slope_norm_20 * 400.0) * 0.35
        + math.tanh((mom10 / px) * 25.0) * 0.30
    )
    bias = max(-1.0, min(1.0, bias))

    direction = "up" if slope20 > 0 and mom10 >= 0 else ("down" if slope20 < 0 and mom10 <= 0 else "mixed")

    return {
        "available": True,
        "disclaimer": "Inspired by Kaufman (TSaM): momentum, linear trend, efficiency ratio — descriptive only.",
        "efficiency_ratio_10": round(er10, 4),
        "efficiency_ratio_20": round(er20, 4),
        "momentum_5d": round(mom5, 4),
        "momentum_10d": round(mom10, 4),
        "momentum_20d": round(mom20, 4),
        "momentum_10d_rel": round(mom10 / px, 5),
        "linreg_20d": {
            "slope": round(slope20, 6),
            "slope_norm": round(slope_norm_20, 8),
            "r2": round(r2_20, 4),
            "forecast_close": round(pred20, 4),
            "forecast_1d_return": round(forecast_1d_ret, 5),
        },
        "linreg_60d": {
            "slope_norm": round(slope_norm_60, 8),
            "r2": round(r2_60, 4),
            "forecast_1d_return": round(forecast_1d_ret_60, 5),
        },
        "combined_bias": round(bias, 3),
        "direction_hint": direction,
    }
