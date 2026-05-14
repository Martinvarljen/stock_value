"""
trade_setup_engine.py — One-page trade context from fundamentals + tech + Elliott + candles.

Not a signal service: combines items already in `record` for journaling / review.
"""

from __future__ import annotations

from typing import Any


def build_trade_setup(record: dict) -> dict[str, Any]:
    price = record.get("current_price")
    if price is None or price <= 0:
        return {"available": False, "reason": "No usable price."}

    trend = record.get("momentum_trend") or "UNKNOWN"
    rsi = record.get("rsi14")
    clf = record.get("classification") or "N/A"
    bb = record.get("buy_below_price")
    fv = record.get("fair_value_weighted")

    ms = record.get("market_structure") or {}
    ext = record.get("extended_technicals") or {}
    ell = record.get("elliott_context") or {}
    cp = record.get("candle_patterns") or {}

    watch: list[dict[str, Any]] = []
    if bb:
        watch.append({"name": "Buy-below (valuation)", "price": round(float(bb), 2)})
    if fv:
        watch.append({"name": "Fair value", "price": round(float(fv), 2)})
    if ell.get("available") and isinstance(ell.get("fib_retracement"), dict):
        for k, v in list(ell["fib_retracement"].items())[:4]:
            watch.append({"name": f"Fib {k}", "price": float(v)})

    macd_bias = (ext.get("macd") or {}).get("bias")
    adx = (ext.get("adx_14") or {}).get("trend_strength")
    st_zone = (ext.get("stochastic_14_3") or {}).get("zone")

    parts = []
    if trend == "UPTREND":
        parts.append("trend up")
    elif trend == "DOWNTREND":
        parts.append("trend down")
    if isinstance(rsi, (int, float)):
        if rsi < 32:
            parts.append("RSI stretched down")
        elif rsi > 68:
            parts.append("RSI stretched up")
    if macd_bias == "bullish":
        parts.append("MACD histogram bullish")
    elif macd_bias == "bearish":
        parts.append("MACD histogram bearish")
    if adx == "strong":
        parts.append("ADX suggests directional regime")
    if ms.get("available"):
        parts.append(f"structure hint: {ms.get('regime_hint', '?')}")
    if ell.get("available"):
        parts.append(f"swing context {ell.get('dominant_direction', '?')}")
        pv = ell.get("price_vs_nearest_fib")
        if isinstance(pv, (int, float)):
            parts.append(f"price vs nearest Fib delta {pv:+.3g}")
    if cp.get("available"):
        parts.append(f"candles: {cp.get('candle_bias', '?')}")

    thesis = "; ".join(parts) if parts else "Mixed / wait for clearer alignment."

    risk = (
        f"Classification {clf}. Stochastic: {st_zone or 'n/a'}. "
        "For rule-strategy permutation / walk-forward checks, run "
        "`python backtesting/strategy_stat_tests.py TICKER`. "
        "This module does not size positions or set stops."
    )

    candle_line = (cp.get("summary") or "") if cp.get("available") else ""

    return {
        "available": True,
        "bias_summary": thesis,
        "classification": clf,
        "watch_levels": watch[:10],
        "risk_notes": risk,
        "elliott_note": (ell.get("structure_hint") or "") if ell.get("available") else "",
        "candle_note": candle_line,
    }
