"""
technical_extended.py — Extra classical indicators from daily OHLCV.

Uses the same 1Y series as data_layer (close_1y + high_1y + low_1y when present).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from kaufman_tsm import compute_kaufman_tsm


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _wilder_smooth(x: pd.Series, period: int) -> pd.Series:
    """Wilder (RMA) smoothing — first value = SMA, then RMA."""
    return x.ewm(alpha=1.0 / period, adjust=False).mean()


def analyze_extended_technicals(data: dict) -> dict[str, Any]:
    """
    Return last-bar MACD, Bollinger, Stochastic, ADX, RSI, Donchian, volume context,
    plus Kaufman TSM block. All numbers are descriptive; not trading advice.
    """
    close = pd.Series(data.get("close_1y") or [], dtype=float)
    high = pd.Series(data.get("high_1y") or close, dtype=float)
    low = pd.Series(data.get("low_1y") or close, dtype=float)

    if len(close) < 60:
        return {
            "available": False,
            "reason": "Need at least ~60 daily closes for extended technicals.",
            "kaufman_tsm": {"available": False, "reason": "Insufficient history for Kaufman-style metrics."},
        }

    if len(high) != len(close):
        high = close.copy()
    if len(low) != len(close):
        low = close.copy()

    c = close.reset_index(drop=True)
    h = high.reset_index(drop=True).reindex(c.index).fillna(c)
    l = low.reset_index(drop=True).reindex(c.index).fillna(c)

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    hist = macd_line - signal_line

    # ── Bollinger (20, 2) ─────────────────────────────────────────────────────
    mid20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_upper = mid20 + 2 * std20
    bb_lower = mid20 - 2 * std20
    last = float(c.iloc[-1])
    bu, bl, bm = float(bb_upper.iloc[-1]), float(bb_lower.iloc[-1]), float(mid20.iloc[-1])
    bb_width = (bu - bl) / bm if bm and not math.isnan(bm) else 0.0
    bb_pct = (last - bl) / (bu - bl) if abs(bu - bl) > 1e-9 else 0.5

    # ── Stochastic (14, 3) ────────────────────────────────────────────────────
    lowest14 = l.rolling(14).min()
    highest14 = h.rolling(14).max()
    stoch_k = 100.0 * (c - lowest14) / (highest14 - lowest14 + 1e-12)
    stoch_d = stoch_k.rolling(3).mean()

    # ── Realised vol (annualised) — used by broker for vol-targeted sizing ───
    rets = c.pct_change().dropna()
    n_vol = min(60, len(rets))
    rv60 = float(rets.tail(n_vol).std(ddof=1) * math.sqrt(252)) if n_vol >= 20 else None

    # ── ADX (14) + DI+/DI- ────────────────────────────────────────────────────
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=c.index)
    minus_dm = pd.Series(minus_dm, index=c.index)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = _wilder_smooth(tr, 14)
    plus_di = 100.0 * _wilder_smooth(plus_dm, 14) / (atr14 + 1e-12)
    minus_di = 100.0 * _wilder_smooth(minus_dm, 14) / (atr14 + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = _wilder_smooth(dx, 14)

    # ── RSI (14, Wilder-style EWM) ────────────────────────────────────────────
    delta_r = c.diff()
    g_r = delta_r.clip(lower=0.0)
    l_r = (-delta_r).clip(lower=0.0)
    avg_g = g_r.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    avg_l = l_r.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    rs_i = avg_g / (avg_l.replace(0.0, np.nan) + 1e-12)
    rsi14 = 100.0 - (100.0 / (1.0 + rs_i))

    # ── Donchian (20) — channel vs prior 20-bar extremes ──────────────────────
    dc_high = h.rolling(20).max()
    dc_low = l.rolling(20).min()
    dc_mid = (dc_high + dc_low) / 2.0
    prev_hi = dc_high.shift(1)
    prev_lo = dc_low.shift(1)
    dch, dcl, dcm = float(dc_high.iloc[-1]), float(dc_low.iloc[-1]), float(dc_mid.iloc[-1])
    dc_break = "inside"
    if len(c) > 21 and not math.isnan(float(prev_hi.iloc[-1])):
        ph, pl = float(prev_hi.iloc[-1]), float(prev_lo.iloc[-1])
        if last > ph:
            dc_break = "above_upper_20"
        elif last < pl:
            dc_break = "below_lower_20"

    # ── Volume: MA20, relative volume, OBV (brief §4.4) ────────────────────────
    vol_list = data.get("volume_1y") or []
    if len(vol_list) == len(c):
        vol_s = pd.Series([float(x) for x in vol_list], dtype=float)
    else:
        vol_s = pd.Series(np.nan, index=c.index, dtype=float)
    vol_ma20 = vol_s.rolling(20).mean()
    rel_vol = None
    if bool(vol_s.notna().any()):
        vm = float(vol_ma20.iloc[-1])
        vv = float(vol_s.iloc[-1])
        if math.isfinite(vm) and math.isfinite(vv) and vm > 0:
            rel_vol = vv / vm
    obv_bias = None
    if len(vol_s) == len(c) and vol_s.notna().sum() > 30:
        chg = c.diff().fillna(0.0)
        direction = np.sign(chg.to_numpy(dtype=float))
        v_np = vol_s.fillna(0.0).to_numpy(dtype=float)
        obv = np.cumsum(direction * v_np)
        if len(obv) >= 60:
            slope = float(obv[-1] - obv[-20])
            obv_bias = "rising" if slope > 0 else ("falling" if slope < 0 else "flat")

    def _last(x: pd.Series) -> float:
        v = float(x.iloc[-1])
        return v if not math.isnan(v) else 0.0

    macd_sig = "bullish" if _last(hist) > 0 else "bearish"
    rsi_v = _last(rsi14)
    rsi_zone = "oversold" if rsi_v < 30 else ("overbought" if rsi_v > 70 else "mid")

    out = {
        "available": True,
        "macd": {
            "line": round(_last(macd_line), 4),
            "signal": round(_last(signal_line), 4),
            "histogram": round(_last(hist), 4),
            "bias": macd_sig,
        },
        "bollinger_20": {
            "middle": round(bm, 2),
            "upper": round(bu, 2),
            "lower": round(bl, 2),
            "band_width_pct": round(float(bb_width), 4),
            "percent_b": round(float(min(1.0, max(0.0, bb_pct))), 3),
        },
        "stochastic_14_3": {
            "k": round(_last(stoch_k), 1),
            "d": round(_last(stoch_d), 1),
            "zone": "oversold" if _last(stoch_k) < 20 else ("overbought" if _last(stoch_k) > 80 else "mid"),
        },
        "adx_14": {
            "adx": round(_last(adx), 1),
            "plus_di": round(_last(plus_di), 1),
            "minus_di": round(_last(minus_di), 1),
            "trend_strength": "strong" if _last(adx) > 25 else "weak",
        },
        # ATR(14) absolute and as a % of last price — consumed by the
        # broker for ATR-anchored stops and vol-aware sizing.
        "atr_14": {
            "value": round(_last(atr14), 4),
            "pct_of_price": round(_last(atr14) / last, 6) if last > 0 else None,
        },
        "realised_vol_60d_annual": round(rv60, 6) if rv60 is not None and math.isfinite(rv60) else None,
        # Last bar OHL — consumed by the broker for intraday-touch stop
        # fills (otherwise stops record at close even when the day's
        # range pierced through them, overstating returns on gap days).
        "last_bar": {
            # We don't have a separate open series threaded through
            # ``data`` today, so fall back to prior close as an
            # approximate open for the gap-detection branch in the
            # broker. ``high`` / ``low`` come straight from the input
            # series and are exact — that's what the stop-touch logic
            # actually needs to anchor the fill.
            "open": round(float(c.iloc[-2]), 4) if len(c) >= 2 else round(float(c.iloc[-1]), 4),
            "high": round(float(h.iloc[-1]), 4),
            "low": round(float(l.iloc[-1]), 4),
            "close": round(float(c.iloc[-1]), 4),
        },
        "rsi_14": {
            "value": round(rsi_v, 1),
            "zone": rsi_zone,
        },
        "donchian_20": {
            "upper": round(dch, 4),
            "lower": round(dcl, 4),
            "mid": round(dcm, 4),
            "close_vs_channel": dc_break,
        },
        "volume_context": {
            "relative_vs_ma20": round(rel_vol, 3) if rel_vol is not None and math.isfinite(rel_vol) else None,
            "obv_slope_hint": obv_bias,
        },
    }
    out["kaufman_tsm"] = compute_kaufman_tsm(c, h, l)
    return out
