"""
vector_engine.py — Simple vector backtest (implementation brief §9.1).

Rules:
  - Caller supplies a signal aligned to each bar (typically from prior closes, no lookahead).
  - signal[i] is the position held over the interval from open[i+1] to open[i+2]
    (decision after close[i], execute at open[i+1], capture until open[i+2]).
  - Costs: on each position change, charge (commission_bps + slippage_bps) / 10_000
    times |Δposition| (full-notional single-asset model, position in [-1, 1] or [0, 1]).

Research only — not live execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .performance_metrics import summarize_backtest


def run_vector_backtest(
    ohlcv: pd.DataFrame,
    signal: pd.Series | np.ndarray,
    *,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """
    ohlcv must contain 'open' column; index should be sorted ascending.

    signal: length len(ohlcv), float/int in [-1,1] for long/short/flat or {0,1} long-only.
    """
    df = ohlcv.copy()
    if "open" not in df.columns:
        raise ValueError("ohlcv must include 'open'")
    df = df.sort_index()
    o = df["open"].astype(float).to_numpy()
    n = len(o)
    if n < 3:
        return {"ok": False, "error": "need_at_least_3_bars"}

    sig = np.asarray(signal, dtype=float).reshape(-1)
    if len(sig) != n:
        return {"ok": False, "error": f"signal_len_{len(sig)}_!=_n_{n}"}

    bps = (float(commission_bps) + float(slippage_bps)) / 10000.0
    n_r = n - 2
    gross = np.zeros(n_r)
    cost = np.zeros(n_r)
    prev = 0.0

    for i in range(n_r):
        o1, o2 = o[i + 1], o[i + 2]
        if not (np.isfinite(o1) and np.isfinite(o2)) or o1 <= 0:
            gross[i] = 0.0
        else:
            gross[i] = float(sig[i]) * (o2 / o1 - 1.0)
        cur = float(sig[i])
        delta = abs(cur - prev)
        cost[i] = delta * bps
        prev = cur

    net = gross - cost
    equity = np.empty(n_r + 1)
    equity[0] = 1.0
    for j in range(n_r):
        equity[j + 1] = equity[j] * max(1e-12, 1.0 + net[j])

    metrics = summarize_backtest(
        net, equity, periods_per_year=periods_per_year, risk_free_rate_annual=0.04
    )
    eff = sig[:n_r]
    dpos = np.diff(np.concatenate([[0.0], eff]))
    changes = int(np.sum(np.abs(dpos) > 1e-9))

    return {
        "ok": True,
        "period_returns": net,
        "gross_period_returns": gross,
        "costs_period": cost,
        "equity": equity,
        "n_position_changes": changes,
        "metrics": metrics,
        "commission_bps": commission_bps,
        "slippage_bps": slippage_bps,
    }


def signal_sma_cross(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """1 when fast EMA > slow EMA else 0 (computed on close, no lookahead in shift)."""
    c = close.astype(float)
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    return (ef > es).astype(float)


def signal_donchian_high_break(high: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Long 1 when close exceeds prior window max of high (signal known at close)."""
    h = high.astype(float)
    c = close.astype(float)
    prior_max = h.rolling(window).max().shift(1)
    return (c > prior_max).astype(float)


def signal_bollinger_mean_reversion(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
    rsi: pd.Series | None = None,
    rsi_os: float = 30.0,
) -> pd.Series:
    """
    Long 1 when close below lower band AND optional RSI oversold; else 0.
    rsi must be pre-aligned (same index) if provided.
    """
    c = close.astype(float)
    mid = c.rolling(window).mean()
    sd = c.rolling(window).std()
    lower = mid - num_std * sd
    long_sig = (c < lower).astype(float)
    if rsi is not None:
        r = rsi.astype(float).reindex(c.index)
        long_sig = ((c < lower) & (r < rsi_os)).astype(float)
    return long_sig
