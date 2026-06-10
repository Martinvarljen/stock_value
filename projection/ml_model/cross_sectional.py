"""Cross-sectional market features for ML training (panel-level).

Ported from StockMarketTool ``FeatureEngine`` diffusion / A-D / dispersion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _panel_log_return(panel: pd.DataFrame, *, lag: int) -> pd.Series:
    """Per-row log return vs ``lag`` trading days ago (by ticker)."""
    p = panel.sort_values(["act_symbol", "date"]).copy()
    p["date"] = pd.to_datetime(p["date"])
    prev = p.groupby("act_symbol")["close"].shift(lag)
    return np.log(p["close"] / prev)


def diffusion_index(panel: pd.DataFrame, *, timeperiod: int = 21) -> pd.Series:
    """Fraction of stocks with positive N-day log returns, by date."""
    ret = _panel_log_return(panel, lag=timeperiod)
    tmp = panel.assign(_ret=ret.values)
    return tmp.groupby("date")["_ret"].apply(lambda x: (x > 0).sum() / max(x.count(), 1))


def advance_decline_spread(panel: pd.DataFrame, *, smooth: int = 5) -> pd.Series:
    """Smoothed (advances - declines) / total, in [-1, 1]."""
    ret = _panel_log_return(panel, lag=1)
    tmp = panel.assign(_ret=ret.values)
    daily = tmp.groupby("date")["_ret"].agg(
        advances=lambda x: (x > 0).sum(),
        declines=lambda x: (x < 0).sum(),
        total=lambda x: x.count(),
    )
    spread = (daily["advances"] - daily["declines"]) / daily["total"].replace(0, np.nan)
    return spread.rolling(smooth, min_periods=1).mean()


def cross_sectional_dispersion(panel: pd.DataFrame, *, smooth: int = 21) -> pd.Series:
    """Rolling mean of daily cross-sectional return std."""
    ret = _panel_log_return(panel, lag=1)
    tmp = panel.assign(_ret=ret.values)
    daily_disp = tmp.groupby("date")["_ret"].std()
    return daily_disp.rolling(smooth, min_periods=1).mean()


def attach_cross_sectional_features(
    rows: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Join date-level features onto per-ticker training rows."""
    if rows.empty or panel.empty:
        return rows
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    di = diffusion_index(p).rename("cs_diffusion_21d")
    ad = advance_decline_spread(p).rename("cs_ad_spread_5d")
    disp = cross_sectional_dispersion(p).rename("cs_dispersion_21d")
    macro = pd.concat([di, ad, disp], axis=1).reset_index()
    macro.columns = ["date"] + list(macro.columns[1:])
    out = rows.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.merge(macro, on="date", how="left")
