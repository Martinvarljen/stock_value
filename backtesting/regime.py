"""SPY regime filters for gross exposure scaling."""

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


def spy_bull_regime(spy_close: pd.Series, as_of: datetime, *, ma_days: int = 200) -> bool:
    """True when SPY close is above its trailing MA (risk-on proxy)."""
    if spy_close.empty or ma_days < 20:
        return True
    ts = pd.Timestamp(as_of)
    sub = spy_close[spy_close.index <= ts]
    if len(sub) < ma_days:
        return True
    px = float(sub.iloc[-1])
    ma = float(sub.iloc[-ma_days:].mean())
    return px >= ma if ma > 0 else True


def gross_exposure_scale(
    spy_close: pd.Series,
    as_of: datetime,
    *,
    ma_days: int = 200,
    bear_scale: float = 0.35,
) -> float:
    """1.0 in bull regime; ``bear_scale`` when SPY below MA."""
    return 1.0 if spy_bull_regime(spy_close, as_of, ma_days=ma_days) else bear_scale
