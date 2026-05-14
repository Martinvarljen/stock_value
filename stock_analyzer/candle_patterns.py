"""
candle_patterns.py — Recent candlestick structure (rule-based, last 1–3 bars).

Uses open_1y / high_1y / low_1y / close_1y when aligned; falls back to close-only
heuristics when open is missing.
"""

from __future__ import annotations

from typing import Any


def _body(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float]:
    body = abs(c - o)
    rng = max(h - l, 1e-9)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return body, upper, lower, rng


def candle_anatomy_last(o: float, h: float, l: float, c: float) -> dict[str, float | bool]:
    """Brief §5.1 — single-bar geometry (safe for zero-range)."""
    body, upper, lower, rng = _body(o, h, l, c)
    inv = 1.0 / rng
    return {
        "body": round(body, 6),
        "range": round(rng, 6),
        "upper_wick": round(upper, 6),
        "lower_wick": round(lower, 6),
        "body_pct": round(body * inv, 4),
        "upper_wick_pct": round(upper * inv, 4),
        "lower_wick_pct": round(lower * inv, 4),
        "is_bullish": c > o,
        "is_bearish": c < o,
    }


def analyze_candle_patterns(data: dict) -> dict[str, Any]:
    o = data.get("open_1y") or []
    h = data.get("high_1y") or []
    l = data.get("low_1y") or []
    c = data.get("close_1y") or []
    n = len(c)
    if n < 3:
        return {"available": False, "reason": "Need at least 3 daily bars."}

    if len(h) != n or len(l) != n:
        h = l = c
    if len(o) != n:
        o = [c[i] if i == 0 else c[i - 1] for i in range(n)]

    patterns: list[str] = []

    o1, h1, l1, c1 = float(o[-1]), float(h[-1]), float(l[-1]), float(c[-1])
    o0, h0, l0, c0 = float(o[-2]), float(h[-2]), float(l[-2]), float(c[-2])
    o_1, h_1, l_1, c_1 = float(o[-3]), float(h[-3]), float(l[-3]), float(c[-3])

    b1, u1, lw1, r1 = _body(o1, h1, l1, c1)
    b0, u0, lw0, r0 = _body(o0, h0, l0, c0)

    # Single-bar
    if b1 / r1 < 0.1:
        patterns.append("Doji / very small body (indecision)")
    if lw1 > 2 * b1 and u1 < b1 and c1 > o1:
        patterns.append("Hammer-like lower wick (potential bullish reversal context)")
    if u1 > 2 * b1 and lw1 < b1 and c1 < o1:
        patterns.append("Shooting star-like upper wick (potential exhaustion)")
    if b1 > 1e-9 and lw1 > 1.2 * b1 and u1 > 1.2 * b1 and b1 / r1 <= 0.25:
        patterns.append("Spinning top / long-legged indecision (large wicks vs body)")
    if b1 / r1 > 0.85 and u1 < 0.15 * r1 and lw1 < 0.15 * r1:
        patterns.append("Marubozu-like (full body, tiny wicks)")
    if c1 > o1 and c0 < o0 and c1 > o0 and o1 < c0 and b1 > b0:
        patterns.append("Bullish engulfing (last bar engulfs prior body)")
    if c1 < o1 and c0 > o0 and c1 < o0 and o1 > c0 and b1 > b0:
        patterns.append("Bearish engulfing (last bar engulfs prior body)")
    # Piercing / dark cloud (classic two-bar vs midpoint)
    mid0 = (h0 + l0) / 2.0
    if c0 < o0 and c1 > o1 and o1 < l0 and c1 > mid0 and c1 < o0:
        patterns.append("Piercing line–like (bullish reclaim past midpoint)")
    if c0 > o0 and c1 < o1 and o1 > h0 and c1 < mid0 and c1 > o0:
        patterns.append("Dark cloud cover–like (bearish rejection past midpoint)")
    # Harami (small body inside prior body)
    if b1 < b0 * 0.9 and max(o1, c1) <= max(o0, c0) and min(o1, c1) >= min(o0, c0) and b0 > 1e-9:
        if c1 > o1:
            patterns.append("Bullish harami (small green inside prior red body)")
        elif c1 < o1:
            patterns.append("Bearish harami (small red inside prior green body)")
    # Tweezer (equal-ish extremes)
    if abs(l1 - l0) / r1 < 0.08 and min(c0, c1) <= min(l0, l1) + 0.15 * r1:
        patterns.append("Tweezer bottom–style lows (within tolerance)")
    if abs(h1 - h0) / r1 < 0.08 and max(o0, o1) >= max(h0, h1) - 0.15 * r1:
        patterns.append("Tweezer top–style highs (within tolerance)")

    # Three-bar morning/evening style (loose)
    mid_1 = (h_1 + l_1) / 2
    if c_1 < o_1 and c0 < o0 and c1 > o1 and c1 > mid_1:
        patterns.append("Possible morning-star structure (3-bar bounce shape)")
    if c_1 > o_1 and c0 > o0 and c1 < o1 and c1 < mid_1:
        patterns.append("Possible evening-star structure (3-bar rollover shape)")

    if not patterns:
        patterns.append("No strong classic pattern on the last bars — context neutral.")

    bias = "bullish" if c1 > o1 and c1 >= c0 else ("bearish" if c1 < o1 and c1 <= c0 else "neutral")

    return {
        "available": True,
        "last_close": round(c1, 4),
        "candle_bias": bias,
        "last_bar_anatomy": candle_anatomy_last(o1, h1, l1, c1),
        "patterns": patterns,
        "summary": patterns[0] if len(patterns) == 1 else "; ".join(patterns[:3]),
    }
