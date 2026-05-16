"""
regime_multifactor.py — Multi-factor risk-on / risk-off classifier.

The single SPY-200d-MA filter in ``regime.py`` is the documented MVP.
The audit flagged it as too coarse: it stays "bull" through 10% draw-
downs and gives no signal for vol-regime shifts, breadth deteriora-
tion, or yield-curve inversions that historically lead recessions by
3-12 months.

This module composes four independent risk-on votes and returns a
weighted score plus a discrete regime label:

    1. Trend  — SPY close vs 200d MA (legacy filter, kept as the
                base signal).
    2. Vol    — SPY 60d realised vol vs its 5y percentile. High-vol
                regimes (>80th percentile) score risk-off.
    3. Breadth — fraction of universe constituents above their own
                 200d MA. <40% scores risk-off.
    4. Term   — 10y-2y Treasury slope (or any spread the caller
                supplies). Inversion (<0) scores risk-off.

Outputs
-------
``RegimeView`` is a frozen dataclass with:
    * ``score`` ∈ [0, 1]: 1 = full risk-on; 0 = full risk-off
    * ``label``: "bull" | "bear" | "unknown"
    * per-component contributions for the explanation log
    * ``gross_exposure_scale`` derived from the score

Backwards compat: when only ``spy_close`` is supplied, only the trend
component contributes — equivalent to ``regime.gross_exposure_scale``
within rounding. Adding breadth/vol/term lifts the resolution.

Use
---
::

    from backtesting.regime_multifactor import classify_regime, RegimeView

    view = classify_regime(
        spy_close=spy,
        as_of=date,
        breadth_pct_above_200ma=0.55,        # optional
        ten_minus_two_slope_bps=18,          # optional
    )
    cfg["_regime_scale"] = view.gross_exposure_scale

The function is intentionally pure-Python where possible; pandas is
imported lazily so headless tests can still run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RegimeView:
    score: float
    label: str
    components: dict[str, dict[str, Any]] = field(default_factory=dict)
    bear_scale: float = 0.35
    unknown_scale: float = 0.35

    @property
    def gross_exposure_scale(self) -> float:
        if self.label == "unknown":
            return self.unknown_scale
        # Linear blend: score=1.0 -> 1.0; score=0.0 -> bear_scale.
        return self.bear_scale + (1.0 - self.bear_scale) * max(0.0, min(1.0, self.score))

    def to_json(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "label": self.label,
            "gross_exposure_scale": round(self.gross_exposure_scale, 4),
            "components": self.components,
        }


# ── component scorers — each returns (vote in [0,1], detail dict) ────────────

def _score_trend_component(spy_close, as_of, *, ma_days: int = 200) -> tuple[float | None, dict[str, Any]]:
    """SPY ≥ 200d MA -> 1.0 (risk-on); below -> 0.0 (risk-off).

    Returns ``None`` when history is insufficient.
    """
    try:
        import pandas as pd
    except ImportError:
        return None, {"available": False, "reason": "pandas missing"}

    if spy_close is None or len(spy_close) < ma_days:
        return None, {"available": False, "reason": "insufficient history"}
    sub = spy_close[spy_close.index <= pd.Timestamp(as_of)]
    if len(sub) < ma_days:
        return None, {"available": False, "reason": "insufficient history at as_of"}
    px = float(sub.iloc[-1])
    ma = float(sub.iloc[-ma_days:].mean())
    if not math.isfinite(ma) or ma <= 0:
        return None, {"available": False, "reason": "bad ma"}
    return (1.0 if px >= ma else 0.0), {
        "available": True,
        "spy_close": px,
        "spy_ma": ma,
        "above_ma": px >= ma,
    }


def _score_vol_component(spy_close, as_of, *, vol_window: int = 60,
                          lookback_years: int = 5) -> tuple[float | None, dict[str, Any]]:
    """High realised vol percentile -> risk-off.

    score = 1 - clip(vol_percentile, 0, 1). 80th-percentile vol -> 0.20
    risk-on vote; 95th-percentile -> 0.05.
    """
    try:
        import pandas as pd
    except ImportError:
        return None, {"available": False, "reason": "pandas missing"}

    if spy_close is None or len(spy_close) < vol_window + 252:
        return None, {"available": False, "reason": "insufficient history"}
    sub = spy_close[spy_close.index <= pd.Timestamp(as_of)]
    if len(sub) < vol_window + 252:
        return None, {"available": False, "reason": "insufficient history at as_of"}
    rets = sub.pct_change().dropna()
    rv = rets.rolling(vol_window).std() * (252 ** 0.5)
    rv = rv.dropna()
    cutoff = rv.index[-1]
    lookback = rv.loc[:cutoff].iloc[-min(252 * lookback_years, len(rv)):]
    if len(lookback) < 252:
        return None, {"available": False, "reason": "insufficient lookback"}
    cur = float(rv.iloc[-1])
    pct_rank = float((lookback < cur).sum()) / float(len(lookback))
    score = max(0.0, min(1.0, 1.0 - pct_rank))
    return score, {
        "available": True,
        "rv_60d_annual": cur,
        "percentile_in_lookback": pct_rank,
    }


def _score_breadth_component(breadth_pct_above_200ma: float | None) -> tuple[float | None, dict[str, Any]]:
    """Fraction of universe above 200d MA. >60% -> 1.0; <40% -> 0.0.

    Caller must pass it in. The dynamic backtest can compute it from
    the universe's price panel — see ``compute_breadth_pct_above_ma``.
    """
    if breadth_pct_above_200ma is None:
        return None, {"available": False, "reason": "not supplied"}
    p = float(breadth_pct_above_200ma)
    if not 0.0 <= p <= 1.0:
        return None, {"available": False, "reason": f"out_of_range:{p}"}
    if p >= 0.60:
        score = 1.0
    elif p <= 0.40:
        score = 0.0
    else:
        score = (p - 0.40) / 0.20
    return score, {"available": True, "pct_above_200ma": p}


def _score_term_component(ten_minus_two_slope_bps: float | None) -> tuple[float | None, dict[str, Any]]:
    """10y - 2y Treasury slope. Positive -> risk-on; inverted -> risk-off.

    Maps slope ∈ [-50, +100] bps -> [0, 1] linearly. Strong positive
    slope (>+100bps) caps at 1.0; deep inversion (<-50bps) floors at
    0.0. The middle linearly interpolates so a 0bps slope scores 0.33.
    """
    if ten_minus_two_slope_bps is None:
        return None, {"available": False, "reason": "not supplied"}
    s = float(ten_minus_two_slope_bps)
    score = (s + 50.0) / 150.0
    score = max(0.0, min(1.0, score))
    return score, {"available": True, "ten_minus_two_bps": s}


# ── composer ─────────────────────────────────────────────────────────────────

DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "trend": 0.40,
    "vol": 0.20,
    "breadth": 0.20,
    "term": 0.20,
}


def classify_regime(
    *,
    spy_close=None,
    as_of: datetime | None = None,
    breadth_pct_above_200ma: float | None = None,
    ten_minus_two_slope_bps: float | None = None,
    weights: dict[str, float] | None = None,
    bear_scale: float = 0.35,
    unknown_scale: float | None = None,
    bull_threshold: float = 0.55,
    bear_threshold: float = 0.40,
) -> RegimeView:
    """Compose the four sub-scores into a single regime view.

    Components missing data (e.g. ``breadth_pct_above_200ma=None``) are
    dropped and the remaining weights are renormalised — this keeps the
    classifier graceful when only the SPY series is available.

    The discrete label uses two thresholds (default 0.40 / 0.55) to
    create a "no-trade" middle zone, which avoids whipsaw at the
    boundary.
    """
    if as_of is None:
        as_of = datetime.today()
    weights = dict(weights or DEFAULT_COMPONENT_WEIGHTS)
    us = bear_scale if unknown_scale is None else float(unknown_scale)

    raw: dict[str, tuple[float | None, dict[str, Any]]] = {
        "trend": _score_trend_component(spy_close, as_of),
        "vol": _score_vol_component(spy_close, as_of),
        "breadth": _score_breadth_component(breadth_pct_above_200ma),
        "term": _score_term_component(ten_minus_two_slope_bps),
    }

    # Renormalise weights over available components.
    avail = {k: v for k, v in raw.items() if v[0] is not None}
    if not avail:
        return RegimeView(score=0.5, label="unknown", components={
            k: v[1] for k, v in raw.items()
        }, bear_scale=bear_scale, unknown_scale=us)
    total_w = sum(weights.get(k, 0.0) for k in avail) or 1.0
    score = sum(weights.get(k, 0.0) * v[0] for k, v in avail.items()) / total_w

    if score >= bull_threshold:
        label = "bull"
    elif score <= bear_threshold:
        label = "bear"
    else:
        label = "unknown"

    components = {k: v[1] | {"score": v[0]} for k, v in raw.items() if v[0] is not None}
    components.update({k: v[1] for k, v in raw.items() if v[0] is None})
    return RegimeView(score=score, label=label, components=components,
                      bear_scale=bear_scale, unknown_scale=us)


# ── breadth helper ───────────────────────────────────────────────────────────

def compute_breadth_pct_above_ma(
    price_panels: dict[str, Any],
    as_of: datetime,
    *,
    ma_days: int = 200,
) -> float | None:
    """Given a dict of ticker → DataFrame[Close], compute fraction
    above 200d MA on ``as_of``. Returns ``None`` if too few names have
    enough history (need at least 30 valid names to make breadth
    meaningful)."""
    try:
        import pandas as pd
    except ImportError:
        return None

    above = 0
    total = 0
    ts = pd.Timestamp(as_of)
    for _, df in price_panels.items():
        if df is None or "Close" not in df:
            continue
        sub = df.loc[df.index <= ts]
        if len(sub) < ma_days:
            continue
        px = float(sub["Close"].iloc[-1])
        ma = float(sub["Close"].iloc[-ma_days:].mean())
        if not math.isfinite(px) or not math.isfinite(ma) or ma <= 0:
            continue
        total += 1
        if px >= ma:
            above += 1
    if total < 30:
        return None
    return above / total
