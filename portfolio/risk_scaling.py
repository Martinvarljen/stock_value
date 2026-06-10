"""Portfolio-level risk scalars (ported from StockMarketTool LiveStrat3 / HRPBacktest).

Combines drawdown, Yang-Zhang vol, and cross-sectional spread into a single
multiplier applied on top of the regime ``gross_exposure_scale``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

import json

from portfolio.market_data import fetch_history
from portfolio.paper_oos import CURVE_PATH


def _risk_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("risk_scaling") or {})


def peak_nav_from_history(nav: float, *, extra_peaks: list[float] | None = None) -> float:
    """Rolling peak NAV from paper OOS curve plus any in-run peaks."""
    peak = float(nav)
    for p in extra_peaks or []:
        if isinstance(p, (int, float)) and p == p:
            peak = max(peak, float(p))
    if CURVE_PATH.is_file():
        for line in CURVE_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                v = row.get("nav")
                if isinstance(v, (int, float)) and v == v:
                    peak = max(peak, float(v))
            except json.JSONDecodeError:
                continue
    return peak


def yang_zhang_median_vol(ohlc_df: pd.DataFrame, *, window: int = 20) -> float | None:
    """Cross-sectional median annualized Yang-Zhang vol (last window days)."""
    if ohlc_df is None or ohlc_df.empty or window < 2:
        return None
    df = ohlc_df.copy()
    for col in ("date", "open", "high", "low", "close"):
        if col not in df.columns:
            return None
    df["date"] = pd.to_datetime(df["date"])
    sym_col = "act_symbol" if "act_symbol" in df.columns else "ticker"
    if sym_col not in df.columns:
        return None

    df_o = df.pivot(index="date", columns=sym_col, values="open").ffill()
    df_h = df.pivot(index="date", columns=sym_col, values="high").ffill()
    df_l = df.pivot(index="date", columns=sym_col, values="low").ffill()
    df_c = df.pivot(index="date", columns=sym_col, values="close").ffill()
    if df_c.shape[0] < window + 1:
        return None

    log_ho = np.log(df_h / df_o)
    log_lo = np.log(df_l / df_o)
    log_co = np.log(df_c / df_o)
    log_oc = np.log(df_o / df_c.shift(1))
    rs_var = (log_ho * (log_ho - log_co)) + (log_lo * (log_lo - log_co))

    overnight_var = log_oc.tail(window).var(ddof=1)
    open_close_var = log_co.tail(window).var(ddof=1)
    rs_var_mean = rs_var.tail(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz_var = overnight_var + (k * open_close_var) + ((1 - k) * rs_var_mean)
    yz_vol = np.sqrt(yz_var.clip(lower=0) * 252)
    med = float(yz_vol.median())
    return med if math.isfinite(med) and med > 0 else None


def build_ohlc_panel(
    tickers: list[str],
    *,
    end: date,
    lookback_days: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch recent daily OHLC for many tickers (for YZ vol)."""
    start = (end - timedelta(days=lookback_days + 30)).isoformat()
    end_excl = (end + timedelta(days=1)).isoformat()
    rows: list[pd.DataFrame] = []
    for tk in tickers:
        hist = fetch_history(tk, start, end_excl, use_cache=use_cache)
        if hist is None or hist.empty:
            continue
        sub = hist.reset_index()
        date_col = sub.columns[0]
        sub = sub.rename(columns={date_col: "date"})
        need = {"Open", "High", "Low", "Close"}
        if not need.issubset(sub.columns):
            continue
        rows.append(
            pd.DataFrame({
                "date": pd.to_datetime(sub["date"]),
                "ticker": tk.upper(),
                "open": sub["Open"].astype(float),
                "high": sub["High"].astype(float),
                "low": sub["Low"].astype(float),
                "close": sub["Close"].astype(float),
            })
        )
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close"])
    return pd.concat(rows, ignore_index=True)


def drawdown_scalar(
    nav: float,
    peak_nav: float,
    cfg: dict[str, Any],
) -> float:
    rc = _risk_cfg(cfg)
    if not rc.get("enabled", False) or not rc.get("use_drawdown_scaling", True):
        return 1.0
    if peak_nav <= 0:
        return 1.0
    dd = (nav / peak_nav) - 1.0
    kill = float(rc.get("dd_kill_threshold", -0.25))
    warn = float(rc.get("dd_warning_threshold", -0.15))
    penalty = float(rc.get("dd_penalty", 0.5))
    if dd <= kill:
        return 0.0
    if dd <= warn:
        return penalty
    return 1.0


def volatility_scalar(
    ohlc_panel: pd.DataFrame | None,
    cfg: dict[str, Any],
) -> float:
    rc = _risk_cfg(cfg)
    if not rc.get("enabled", False) or not rc.get("use_vol_scaling", True):
        return 1.0
    target = float(rc.get("target_vol", 0.10))
    lookback = int(rc.get("vol_lookback", 20))
    max_lev = float(rc.get("max_vol_leverage", 1.0))
    vol_type = str(rc.get("volatility_type", "yang_zhang")).lower()
    if vol_type == "yang_zhang" and ohlc_panel is not None and not ohlc_panel.empty:
        med = yang_zhang_median_vol(ohlc_panel, window=lookback)
        if med and med > 0:
            return min(max_lev, target / med)
    return 1.0


def spread_scalar_from_scores(scores: list[float], cfg: dict[str, Any]) -> float:
    """Scale down when today's cross-sectional score dispersion is very low."""
    rc = _risk_cfg(cfg)
    if not rc.get("enabled", False) or not rc.get("use_spread_scaling", True):
        return 1.0
    clean = [float(s) for s in scores if isinstance(s, (int, float)) and s == s]
    if len(clean) < 10:
        return 1.0
    today_spread = float(np.std(clean))
    floor_spread = float(rc.get("min_score_std", 0.04))
    spread_floor = float(rc.get("spread_floor", 0.35))
    if today_spread >= floor_spread:
        return 1.0
    ratio = today_spread / floor_spread if floor_spread > 0 else 0.0
    return spread_floor + (1.0 - spread_floor) * max(0.0, min(1.0, ratio))


def compute_risk_scalar(
    *,
    cfg: dict[str, Any],
    nav: float,
    peak_nav: float,
    analyses: list[dict[str, Any]],
    tickers: list[str],
    as_of: date,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Return combined scalar and component breakdown."""
    rc = _risk_cfg(cfg)
    if not rc.get("enabled", False):
        return {"scalar": 1.0, "components": {}}

    scores = [a.get("ml_score") for a in analyses if a.get("ok")]
    dd_s = drawdown_scalar(nav, peak_nav, cfg)
    spread_s = spread_scalar_from_scores(
        [float(s) for s in scores if isinstance(s, (int, float)) and s == s],
        cfg,
    )

    ohlc_panel = None
    if rc.get("use_vol_scaling", True) and str(rc.get("volatility_type", "yang_zhang")) == "yang_zhang":
        lookback = int(rc.get("vol_lookback", 20))
        ohlc_panel = build_ohlc_panel(
            tickers[: int(rc.get("vol_sample_tickers", 40))],
            end=as_of,
            lookback_days=lookback + 20,
            use_cache=use_cache,
        )
    vol_s = volatility_scalar(ohlc_panel, cfg)

    combined = min(dd_s, vol_s, spread_s)
    return {
        "scalar": combined,
        "components": {
            "drawdown": dd_s,
            "volatility": vol_s,
            "spread": spread_s,
            "nav": nav,
            "peak_nav": peak_nav,
        },
    }


def apply_risk_scalar_to_regime(
    regime: dict[str, Any],
    risk_report: dict[str, Any],
) -> dict[str, Any]:
    """Multiply regime gross scale by portfolio risk scalar (in-place copy)."""
    out = dict(regime)
    base = float(out.get("gross_exposure_scale", 1.0))
    mult = float(risk_report.get("scalar", 1.0))
    out["gross_exposure_scale"] = round(base * mult, 6)
    out["risk_scaling"] = risk_report
    return out
