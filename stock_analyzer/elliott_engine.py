"""
elliott_engine.py — Heuristic swing structure & Fibonacci context.

This is NOT automated Elliott Wave International-style wave counting. Human wave
analysis is subjective. We only:
  • detect coarse swing highs / lows (fractal pivots on OHLC),
  • infer dominant direction,
  • project common Fibonacci retracements from the last clear leg,
  • emit numeric features (``dominant_direction``, ``price_vs_nearest_fib``)
    consumed by the ML feature builder, plus a plain-language *structure_hint*
    that lands in the trade-setup memory log.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _fractal_pivots(
    high: pd.Series,
    low: pd.Series,
    order: int = 4,
) -> list[dict[str, Any]]:
    """Alternating-style pivots: local max on high, local min on low."""
    pivots: list[dict[str, Any]] = []
    n = len(high)
    if n < 2 * order + 3:
        return pivots

    for i in range(order, n - order):
        h_win = high.iloc[i - order : i + order + 1]
        l_win = low.iloc[i - order : i + order + 1]
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])
        if hi >= float(h_win.max()) and hi > float(high.iloc[i - 1]) and hi > float(high.iloc[i + 1]):
            pivots.append({"kind": "H", "idx": i, "price": hi})
        elif lo <= float(l_win.min()) and lo < float(low.iloc[i - 1]) and lo < float(low.iloc[i + 1]):
            pivots.append({"kind": "L", "idx": i, "price": lo})

    # De-duplicate consecutive same-kind pivots (keep more extreme)
    merged: list[dict[str, Any]] = []
    for p in pivots:
        if not merged:
            merged.append(p)
            continue
        last = merged[-1]
        if last["kind"] == p["kind"]:
            if p["kind"] == "H" and p["price"] >= last["price"]:
                merged[-1] = p
            elif p["kind"] == "L" and p["price"] <= last["price"]:
                merged[-1] = p
        else:
            merged.append(p)
    return merged[-12:]


def _fib_levels(a: float, b: float) -> dict[str, float]:
    """Retracement ratios from price a toward b (a = swing start, b = swing end)."""
    diff = b - a
    ratios = (0.236, 0.382, 0.5, 0.618, 0.786)
    return {f"fib_{r}": round(a + diff * r, 4) for r in ratios}


def analyze_elliott_context(data: dict) -> dict[str, Any]:
    close = pd.Series(data.get("close_1y") or [], dtype=float)
    high = pd.Series(data.get("high_1y") or close, dtype=float)
    low = pd.Series(data.get("low_1y") or close, dtype=float)

    if len(close) < 80:
        return {
            "available": False,
            "reason": "Need ~80+ daily bars for swing / Fib context.",
        }

    if len(high) != len(close):
        high = close.copy()
    if len(low) != len(close):
        low = close.copy()

    h = high.reset_index(drop=True)
    l = low.reset_index(drop=True)
    c = close.reset_index(drop=True)

    pivots = _fractal_pivots(h, l, order=4)
    if len(pivots) < 2:
        return {
            "available": False,
            "reason": "Could not isolate enough swing pivots on this history.",
        }

    last_px = float(c.iloc[-1])
    last_p = pivots[-1]
    prev_p = pivots[-2]

    # Last leg direction: previous pivot -> last pivot
    leg_up = last_p["price"] > prev_p["price"]
    swing_start = float(prev_p["price"])
    swing_end = float(last_p["price"])

    if leg_up:
        dominant = "up"
        fib = _fib_levels(swing_end, swing_start)  # retrace down from high
        structure_hint = (
            "Possible corrective pullback after a push higher — "
            "watch common Fib retracements vs. trend support."
        )
    else:
        dominant = "down"
        fib = _fib_levels(swing_end, swing_start)  # bounce up from low
        structure_hint = (
            "Possible corrective bounce after a decline — "
            "treat Fib levels as potential resistance unless reclaimed."
        )

    nearest = min(fib.values(), key=lambda x: abs(x - last_px))
    hi_leg = max(swing_start, swing_end)
    lo_leg = min(swing_start, swing_end)

    return {
        "available": True,
        "disclaimer": "Heuristic swings only — not a certified Elliott wave count.",
        "dominant_direction": dominant,
        "last_leg_high": round(hi_leg, 4),
        "last_leg_low": round(lo_leg, 4),
        "fib_retracement": fib,
        "price_vs_nearest_fib": round(last_px - nearest, 4),
        "structure_hint": structure_hint,
        "pivot_count_used": len(pivots),
    }
