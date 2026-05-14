"""
market_structure.py — Confirmed swing pivots + HH/HL/LH/LL labels (brief §6.1–6.2).

Pivots are included only after the right window has elapsed (no look-ahead on
confirmation index). Descriptive only.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _fractal_candidates(high: pd.Series, low: pd.Series, order: int) -> list[dict[str, Any]]:
    pivots: list[dict[str, Any]] = []
    n = len(high)
    for i in range(order, n - order):
        h_win = high.iloc[i - order : i + order + 1]
        l_win = low.iloc[i - order : i + order + 1]
        hi = float(high.iloc[i])
        lo = float(low.iloc[i])
        if hi >= float(h_win.max()) and hi > float(high.iloc[i - 1]) and hi > float(high.iloc[i + 1]):
            pivots.append({"kind": "H", "idx": i, "price": hi})
        elif lo <= float(l_win.min()) and lo < float(low.iloc[i - 1]) and lo < float(low.iloc[i + 1]):
            pivots.append({"kind": "L", "idx": i, "price": lo})

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
    return merged


def _confirmed_only(pivots: list[dict[str, Any]], n_bars: int, order: int) -> list[dict[str, Any]]:
    """Keep pivots whose index is at most n_bars - 1 - order (fully confirmed at last bar)."""
    lim = n_bars - 1 - order
    return [p for p in pivots if p["idx"] <= lim]


def _structure_labels(pivots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_h = last_l = None
    out: list[dict[str, Any]] = []
    for p in pivots:
        if p["kind"] == "H":
            if last_h is None:
                tag = "H"
            else:
                tag = "HH" if p["price"] > last_h["price"] else "LH"
            last_h = p
        else:
            if last_l is None:
                tag = "L"
            else:
                tag = "HL" if p["price"] > last_l["price"] else "LL"
            last_l = p
        out.append({**p, "label": tag})
    return out


def _infer_regime(labeled: list[dict[str, Any]]) -> str:
    if len(labeled) < 3:
        return "insufficient_swings"
    tail = labeled[-4:]
    tags = [x["label"] for x in tail]
    if tags.count("HH") >= 1 and tags.count("HL") >= 1 and "LL" not in tags[-3:]:
        return "up_sequence"
    if tags.count("LL") >= 1 and tags.count("LH") >= 1 and "HH" not in tags[-3:]:
        return "down_sequence"
    return "mixed_or_range"


def analyze_market_structure(data: dict, order: int = 4) -> dict[str, Any]:
    close = pd.Series(data.get("close_1y") or [], dtype=float)
    high = pd.Series(data.get("high_1y") or close, dtype=float)
    low = pd.Series(data.get("low_1y") or close, dtype=float)

    if len(close) < 2 * order + 15:
        return {
            "available": False,
            "reason": "Need more daily bars for confirmed swing structure.",
        }

    if len(high) != len(close):
        high = close.copy()
    if len(low) != len(close):
        low = close.copy()

    h = high.reset_index(drop=True)
    l = low.reset_index(drop=True)
    n = len(h)

    raw = _fractal_candidates(h, l, order)
    conf = _confirmed_only(raw, n, order)
    labeled = _structure_labels(conf)
    regime = _infer_regime(labeled) if labeled else "insufficient_swings"

    last_px = float(close.iloc[-1])
    last_swing = labeled[-1] if labeled else None
    dist = None
    if last_swing and math.isfinite(last_px):
        dist = round(last_px - float(last_swing["price"]), 4)

    recent = [
        {"kind": x["kind"], "label": x["label"], "idx": x["idx"], "price": round(float(x["price"]), 4)}
        for x in labeled[-6:]
    ]

    return {
        "available": True,
        "disclaimer": "Confirmed swings only (no right-side lookahead). Not automated Elliott counting.",
        "swing_order": order,
        "regime_hint": regime,
        "last_close_vs_last_pivot": dist,
        "recent_swings": recent,
        "n_confirmed_pivots": len(labeled),
    }
