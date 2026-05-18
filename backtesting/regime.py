"""SPY regime filters for gross exposure scaling.

Two callable layers:
    1. ``spy_bull_regime`` — bool: SPY ≥ 200d MA. Backwards-compatible with the
       old behaviour (returns ``default_when_unknown=True`` when there isn't
       enough history). Pass ``default_when_unknown=False`` to abstain.
    2. ``gross_exposure_scale`` — maps the regime to a sizing multiplier in
       {1.0, bear_scale, unknown_scale}. ``unknown_scale`` defaults to
       ``bear_scale`` so callers abstain (= treat as bear) when the regime is
       not yet identifiable. This is the safer default for live capital and
       was the source of an "optimistic-default into uncertainty" footgun.

A breadth/vol-aware regime classifier remains a Tier-2 upgrade — the
single-MA filter is the documented MVP.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def spy_close_series(spy_hist: pd.DataFrame) -> pd.Series:
    if spy_hist is None or spy_hist.empty:
        return pd.Series(dtype=float)
    s = spy_hist["Close"].astype(float).copy()
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _has_enough_history(spy_close: pd.Series, as_of: datetime, ma_days: int) -> bool:
    if spy_close is None or spy_close.empty or ma_days < 20:
        return False
    sub = spy_close[spy_close.index <= pd.Timestamp(as_of)]
    return len(sub) >= ma_days


def spy_bull_regime(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
    default_when_unknown: bool = True,
) -> bool:
    """True when SPY close is above its trailing MA (risk-on proxy).

    ``default_when_unknown`` controls the fallback when there isn't enough
    history; the legacy default is ``True`` for backwards compatibility, but
    ``gross_exposure_scale`` now defaults to *abstaining* on unknown.
    """
    if not _has_enough_history(spy_close, as_of, ma_days):
        return default_when_unknown
    ts = pd.Timestamp(as_of)
    sub = spy_close[spy_close.index <= ts]
    px = float(sub.iloc[-1])
    ma = float(sub.iloc[-ma_days:].mean())
    return px >= ma if ma > 0 else default_when_unknown


def regime_signal(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
) -> str:
    """Tri-state regime: 'bull' / 'bear' / 'unknown'.

    Use this when the caller wants to distinguish "we know it's risk-off"
    from "we don't have enough data yet to decide".
    """
    if not _has_enough_history(spy_close, as_of, ma_days):
        return "unknown"
    return "bull" if spy_bull_regime(spy_close, as_of, ma_days=ma_days) else "bear"


def spy_pct_below_ma(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
) -> float | None:
    """Fraction SPY is below its MA (0 = at MA, 0.05 = 5% below). None if unknown."""
    if not _has_enough_history(spy_close, as_of, ma_days):
        return None
    ts = pd.Timestamp(as_of)
    sub = spy_close[spy_close.index <= ts]
    px = float(sub.iloc[-1])
    ma = float(sub.iloc[-ma_days:].mean())
    if ma <= 0:
        return None
    return max(0.0, (ma - px) / ma)


def spy_trailing_return(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    days: int = 20,
) -> float | None:
    """Simple trailing return over ``days`` bars ending at ``as_of``."""
    if spy_close is None or spy_close.empty or days < 1:
        return None
    ts = pd.Timestamp(as_of)
    sub = spy_close[spy_close.index <= ts]
    if len(sub) < days + 1:
        return None
    p0 = float(sub.iloc[-(days + 1)])
    p1 = float(sub.iloc[-1])
    if p0 <= 0:
        return None
    return p1 / p0 - 1.0


def build_regime_snapshot(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
    bear_scale: float = 0.35,
    momentum_days: int = 20,
) -> dict[str, object]:
    """Regime dict for portfolio decisions (bull/bear + short-gate features)."""
    return {
        "spy_bull": spy_bull_regime(spy_close, as_of, ma_days=ma_days),
        "regime_signal": regime_signal(spy_close, as_of, ma_days=ma_days),
        "gross_exposure_scale": gross_exposure_scale(
            spy_close, as_of, ma_days=ma_days, bear_scale=bear_scale,
        ),
        "spy_pct_below_ma200": spy_pct_below_ma(spy_close, as_of, ma_days=ma_days),
        "spy_return_20d": spy_trailing_return(spy_close, as_of, days=momentum_days),
    }


def gross_exposure_scale(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
    bear_scale: float = 0.35,
    unknown_scale: float | None = None,
) -> float:
    """Map regime to a sizing multiplier.

    bull   -> 1.0
    bear   -> ``bear_scale``
    unknown-> ``unknown_scale`` (defaults to ``bear_scale`` = abstain)
    """
    us = bear_scale if unknown_scale is None else float(unknown_scale)
    sig = regime_signal(spy_close, as_of, ma_days=ma_days)
    if sig == "bull":
        return 1.0
    if sig == "bear":
        return float(bear_scale)
    return us
