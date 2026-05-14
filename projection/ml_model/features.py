"""
features.py — Feature extraction for ML projection model.

Two modes:
  1. extract_features(record)          — live inference from a stock record
  2. extract_historical_features(...)  — historical features for training
"""

import sys
import math
from pathlib import Path
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# Ensure stock_analyzer + projection are importable when run standalone
_root = Path(__file__).resolve().parents[2]
for _p in [str(_root / "stock_analyzer"), str(_root / "projection")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Minimum daily bars for extended technicals (MA200 + vol60 + buffers)
MIN_OHLCV_BARS = 220

# Technical features (OHLCV-derived). Order is fixed for saved models / metadata.
# v3 (2026): v2 + long-horizon returns, drawdown, vol stress, panic proxy, SPY-relative regime.
TECH_FEATURES = [
    "rsi14",
    "price_vs_ma50",
    "price_vs_ma200",
    "return_5d",
    "return_10d",
    "return_1m",
    "return_3m",
    "return_6m",
    "return_1y",
    "return_2y",
    "volatility_20d",
    "vol_ratio_20_60",
    "volume_ratio",
    "atr14_rel",
    "bb_pctb",
    "macd_norm",
    "ma_spread_50_200",
    "hl_range_20d",
    "dd_from_high_252d",
    "vol_stress_vs_median",
    "panic_day_ratio_20d",
    "rel_ret_63_vs_spy",
    "spy_dd_126d",
    # Kaufman *Trading Systems & Methods*
    "tsm_er10",
    "tsm_er20",
    "tsm_mom10_rel",
    "tsm_reg20_slope_norm",
    "tsm_reg20_fcst1d",
    "tsm_reg20_r2",
    "tsm_bias",
]

FEATURE_SCHEMA_VERSION = 3

try:
    from kaufman_tsm import compute_kaufman_tsm as _compute_kaufman_tsm
except ImportError:
    _compute_kaufman_tsm = None  # type: ignore[misc, assignment]

# Full feature set (used for live inference)
ALL_FEATURES = TECH_FEATURES + [
    "valuation_upside",
    "fcf_yield",
    "earnings_yield",       # 1 / PE
    "operating_margin",
    "roic_wacc_spread",
    "revenue_cagr_5y",
    "beta",
    "net_debt_ebitda",
    "n_red_flags",
    "n_critical_flags",
    "composite_score",
]


_SPY_LIVE_CACHE: pd.Series | None = None
_SPY_LIVE_CACHE_DATE: date | None = None


def _get_live_spy_close_series() -> pd.Series | None:
    """One SPY download per calendar day (live dashboard / inference)."""
    global _SPY_LIVE_CACHE, _SPY_LIVE_CACHE_DATE
    today = date.today()
    if _SPY_LIVE_CACHE is not None and _SPY_LIVE_CACHE_DATE == today:
        return _SPY_LIVE_CACHE
    try:
        import yfinance as yf

        end = datetime.now()
        start = end - timedelta(days=900)
        sh = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if sh.empty or "Close" not in sh.columns:
            return None
        if sh.index.tz is not None:
            sh = sh.copy()
            sh.index = sh.index.tz_localize(None)
        s = sh["Close"].astype(float)
        _SPY_LIVE_CACHE = s
        _SPY_LIVE_CACHE_DATE = today
        return s
    except Exception:
        return None


# ── live feature extraction ────────────────────────────────────────────────────

def extract_features(record: dict) -> dict:
    """Extract ML features from a live stock record dict."""
    price = record.get("current_price") or 0.0

    feat: dict[str, float] = {}

    close_l = record.get("close_1y") or []
    n = len(close_l)
    high_l = record.get("high_1y") or []
    low_l = record.get("low_1y") or []
    vol_l = record.get("volume_1y") or []

    tech: dict[str, float] | None = None
    if n >= MIN_OHLCV_BARS:
        h = high_l if len(high_l) == n else close_l
        l = low_l if len(low_l) == n else close_l
        v = vol_l if len(vol_l) == n else [1.0] * n
        hist = pd.DataFrame({"Close": close_l, "High": h, "Low": l, "Volume": v})
        tech = technical_features_from_ohlcv(hist)

    if tech is None:
        tech = _tech_fallback_from_record(record, price)

    feat.update(tech)

    if tech is not None and n >= MIN_OHLCV_BARS:
        idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n, freq="B")
        cser = pd.Series(close_l, index=idx, dtype=float)
        spy_live = _get_live_spy_close_series()
        if spy_live is not None:
            _apply_spy_regime_features(feat, cser, spy_live, idx[-1].to_pydatetime())

    # Valuation
    fv = record.get("fair_value_weighted")
    feat["valuation_upside"] = _ratio_diff(fv, price) if (fv and price > 0) else 0.0
    feat["fcf_yield"] = _safe(record.get("fcf_yield"), 0.0)
    pe = record.get("pe_ratio")
    feat["earnings_yield"] = (1.0 / pe) if (pe and pe > 0) else 0.0

    # Quality
    feat["operating_margin"] = _safe(record.get("operating_margin"), 0.0)
    roic = record.get("roic")
    wacc = (record.get("wacc_data") or {}).get("wacc")
    feat["roic_wacc_spread"] = float(roic - wacc) if (roic is not None and wacc is not None) else 0.0

    # Growth
    feat["revenue_cagr_5y"] = _safe(record.get("revenue_cagr_5y"), 0.0)

    # Risk
    feat["beta"] = _safe(record.get("beta"), 1.0)
    feat["net_debt_ebitda"] = _safe(record.get("net_debt_ebitda"), 0.0)
    feat["n_red_flags"] = float(len(record.get("red_flags") or []))
    feat["n_critical_flags"] = float(len(record.get("critical_flags") or []))

    # Composite rule-based score as a meta-feature
    try:
        from projection_engine import _composite_score
        score, _ = _composite_score(record)
        feat["composite_score"] = float(score)
    except Exception:
        feat["composite_score"] = 0.0

    return feat


def _tech_fallback_from_record(record: dict, price: float) -> dict[str, float]:
    """When full OHLCV history is short: use flat file fields + neutral extended defaults."""
    feat: dict[str, float] = {}
    feat["rsi14"] = _safe(record.get("rsi14"), 50.0) / 100.0
    ma50 = record.get("ma50")
    ma200 = record.get("ma200")
    feat["price_vs_ma50"] = _ratio_diff(price, ma50)
    feat["price_vs_ma200"] = _ratio_diff(price, ma200)
    mom = record.get("momentum_metrics") or {}
    feat["return_1m"] = _safe((mom.get("return_1m") or {}).get("value"), 0.0)
    feat["return_3m"] = _safe((mom.get("return_3m") or {}).get("value"), 0.0)
    feat["return_6m"] = _safe((mom.get("return_6m") or {}).get("value"), 0.0)
    feat["return_5d"] = 0.0
    feat["return_10d"] = 0.0
    feat["volatility_20d"] = _safe(record.get("volatility_20d"), 0.20)
    feat["vol_ratio_20_60"] = 1.0
    feat["volume_ratio"] = _safe(record.get("volume_ratio"), 1.0)
    feat["atr14_rel"] = feat["volatility_20d"] * 0.5
    feat["bb_pctb"] = 0.5
    feat["macd_norm"] = 0.0
    if price and ma50 and ma200:
        feat["ma_spread_50_200"] = float((ma50 - ma200) / price)
    else:
        feat["ma_spread_50_200"] = 0.0
    feat["hl_range_20d"] = feat["volatility_20d"] / math.sqrt(252) if feat["volatility_20d"] else 0.02
    feat["return_1y"] = feat["return_6m"]
    feat["return_2y"] = feat["return_1y"]
    feat["dd_from_high_252d"] = 0.0
    feat["vol_stress_vs_median"] = 1.0
    feat["panic_day_ratio_20d"] = 0.0
    feat["rel_ret_63_vs_spy"] = 0.0
    feat["spy_dd_126d"] = 0.0
    for _k in (
        "tsm_er10",
        "tsm_er20",
        "tsm_mom10_rel",
        "tsm_reg20_slope_norm",
        "tsm_reg20_fcst1d",
        "tsm_reg20_r2",
        "tsm_bias",
    ):
        feat[_k] = 0.0
    return feat


# ── historical feature extraction (for training) ──────────────────────────────

def extract_historical_features(
    history: pd.DataFrame,
    date: datetime,
    spy_close: pd.Series | None = None,
) -> dict | None:
    """
    Extract TECH_FEATURES from historical OHLCV at a given date.
    Optional spy_close (aligned daily Close) adds market-relative regime columns.
    Fundamental columns in ALL_FEATURES stay NaN for training rows (imputed if used).
    """
    hist = history[history.index <= pd.Timestamp(date)]
    if len(hist) < MIN_OHLCV_BARS:
        return None

    sub = hist.copy()
    if "High" not in sub.columns:
        sub["High"] = sub["Close"]
    if "Low" not in sub.columns:
        sub["Low"] = sub["Close"]
    if "Volume" not in sub.columns:
        sub["Volume"] = 1.0

    feat = technical_features_from_ohlcv(sub)
    if feat is None:
        return None

    _apply_spy_regime_features(feat, sub["Close"].astype(float), spy_close, date)

    for col in ALL_FEATURES:
        if col not in feat:
            feat[col] = float("nan")

    return feat


def technical_features_from_ohlcv(hist: pd.DataFrame) -> dict[str, float] | None:
    """
    Last-bar technical features from aligned OHLCV (chronological rows).
    Returns None if insufficient history or invalid prices.
    """
    if len(hist) < MIN_OHLCV_BARS:
        return None

    close = hist["Close"].astype(float)
    high = hist["High"].astype(float) if "High" in hist.columns else close
    low = hist["Low"].astype(float) if "Low" in hist.columns else close
    vol = hist["Volume"].astype(float) if "Volume" in hist.columns else pd.Series(1.0, index=hist.index)

    price = float(close.iloc[-1])
    if price <= 0 or math.isnan(price):
        return None

    feat: dict[str, float] = {}

    feat["rsi14"] = _compute_rsi(close, 14) / 100.0
    feat["price_vs_ma50"] = _ratio_diff(price, float(close.iloc[-50:].mean()))
    feat["price_vs_ma200"] = _ratio_diff(price, float(close.iloc[-200:].mean()))
    feat["return_5d"] = _safe_return(close, 5)
    feat["return_10d"] = _safe_return(close, 10)
    feat["return_1m"] = _safe_return(close, 21)
    feat["return_3m"] = _safe_return(close, 63)
    feat["return_6m"] = _safe_return(close, 126)

    rets = close.pct_change().dropna()
    if len(close) >= 253:
        feat["return_1y"] = _safe_return(close, 252)
    else:
        feat["return_1y"] = feat["return_6m"]
    if len(close) >= 505:
        feat["return_2y"] = _safe_return(close, 504)
    else:
        feat["return_2y"] = feat["return_1y"]

    if len(rets) >= 20:
        v20 = float(rets.iloc[-20:].std() * math.sqrt(252))
    else:
        v20 = 0.2
    if len(rets) >= 60:
        v60 = float(rets.iloc[-60:].std() * math.sqrt(252))
    else:
        v60 = max(v20, 1e-6)
    feat["volatility_20d"] = v20
    feat["vol_ratio_20_60"] = float(min(5.0, max(0.2, v20 / max(v60, 1e-6))))

    avg_vol = float(vol.iloc[-20:].mean())
    feat["volume_ratio"] = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    feat["atr14_rel"] = _atr14_ratio(high, low, close)
    feat["bb_pctb"] = _bb_pctb(close, 20)
    ema12 = _ema_last(close, 12)
    ema26 = _ema_last(close, 26)
    feat["macd_norm"] = float((ema12 - ema26) / price)
    ma50 = float(close.iloc[-50:].mean())
    ma200 = float(close.iloc[-200:].mean())
    feat["ma_spread_50_200"] = float((ma50 - ma200) / price)
    hl = ((high.iloc[-20:] - low.iloc[-20:]) / close.iloc[-20:]).mean()
    feat["hl_range_20d"] = float(hl) if not (isinstance(hl, float) and math.isnan(hl)) else 0.0

    _apply_long_horizon_stress(feat, close, high, low, rets)
    _apply_tsm_ml_features(feat, close, high, low)

    return feat


def _apply_long_horizon_stress(
    feat: dict[str, float],
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    rets: pd.Series,
) -> None:
    """Drawdown, vol stress, panic proxy — captures crisis-style stretches (price-only)."""
    lc = len(close)
    if lc >= 252:
        rm = float(close.rolling(252).max().iloc[-1])
        c0 = float(close.iloc[-1])
        feat["dd_from_high_252d"] = float(max(-1.0, min(0.0, c0 / rm - 1.0))) if rm > 0 else 0.0
    else:
        feat["dd_from_high_252d"] = 0.0

    if len(rets) >= 252:
        rv = rets.rolling(20).std()
        cur = float(rv.iloc[-1])
        med = float(rv.iloc[-252:].median())
        feat["vol_stress_vs_median"] = float(min(4.0, max(0.25, cur / (med + 1e-12))))
    else:
        feat["vol_stress_vs_median"] = 1.0

    if lc >= 22:
        rng = (high - low).iloc[-20:]
        red = close.iloc[-20:] < close.shift(1).iloc[-20:]
        med_r = float(rng.median())
        if med_r > 1e-12 and not math.isnan(med_r):
            panic = float(((rng > med_r * 1.5) & red.fillna(False)).sum()) / 20.0
            feat["panic_day_ratio_20d"] = min(1.0, panic)
        else:
            feat["panic_day_ratio_20d"] = 0.0
    else:
        feat["panic_day_ratio_20d"] = 0.0


def _apply_spy_regime_features(
    feat: dict[str, float],
    stock_close: pd.Series,
    spy_close: pd.Series | None,
    date: datetime,
) -> None:
    """Relative strength vs SPY + SPY drawdown (shared market stress / crisis proxy)."""
    if spy_close is None or spy_close.empty:
        feat["rel_ret_63_vs_spy"] = 0.0
        feat["spy_dd_126d"] = 0.0
        return
    try:
        ts = pd.Timestamp(date)
        spy = spy_close[spy_close.index <= ts].dropna().astype(float)
        stk = stock_close[stock_close.index <= ts].dropna().astype(float)
        if len(spy) < 70 or len(stk) < 70:
            feat["rel_ret_63_vs_spy"] = 0.0
            feat["spy_dd_126d"] = 0.0
            return
        sr = float(stk.iloc[-1] / stk.iloc[-64] - 1.0) if len(stk) >= 64 else 0.0
        pr = float(spy.iloc[-1] / spy.iloc[-64] - 1.0) if len(spy) >= 64 else 0.0
        feat["rel_ret_63_vs_spy"] = float(max(-0.6, min(0.6, sr - pr)))
        mx = float(spy.iloc[-126:].max())
        lc = float(spy.iloc[-1])
        feat["spy_dd_126d"] = float(max(-0.75, min(0.0, lc / mx - 1.0))) if mx > 0 else 0.0
    except Exception:
        feat["rel_ret_63_vs_spy"] = 0.0
        feat["spy_dd_126d"] = 0.0


def _apply_tsm_ml_features(
    feat: dict[str, float],
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> None:
    """Populate TSM / Kaufman columns into feat dict (mutates feat)."""
    defaults = {
        "tsm_er10": 0.0,
        "tsm_er20": 0.0,
        "tsm_mom10_rel": 0.0,
        "tsm_reg20_slope_norm": 0.0,
        "tsm_reg20_fcst1d": 0.0,
        "tsm_reg20_r2": 0.0,
        "tsm_bias": 0.0,
    }
    if _compute_kaufman_tsm is None:
        feat.update(defaults)
        return
    tsm = _compute_kaufman_tsm(close, high, low)
    if not tsm.get("available"):
        feat.update(defaults)
        return
    lr = tsm.get("linreg_20d") or {}
    feat["tsm_er10"] = float(tsm.get("efficiency_ratio_10") or 0.0)
    feat["tsm_er20"] = float(tsm.get("efficiency_ratio_20") or 0.0)
    feat["tsm_mom10_rel"] = float(tsm.get("momentum_10d_rel") or 0.0)
    feat["tsm_reg20_slope_norm"] = float(lr.get("slope_norm") or 0.0)
    feat["tsm_reg20_fcst1d"] = float(lr.get("forecast_1d_return") or 0.0)
    feat["tsm_reg20_r2"] = float(lr.get("r2") or 0.0)
    feat["tsm_bias"] = float(tsm.get("combined_bias") or 0.0)


def feature_vector(feat: dict, feature_names: list[str]) -> np.ndarray:
    """Convert feature dict to ordered numpy array for model inference."""
    vals = []
    for name in feature_names:
        v = feat.get(name, 0.0)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = 0.0
        vals.append(float(v))
    return np.array(vals, dtype=np.float32).reshape(1, -1)


# ── helpers ────────────────────────────────────────────────────────────────────

def _ema_last(s: pd.Series, span: int) -> float:
    e = s.ewm(span=span, adjust=False).mean().iloc[-1]
    v = float(e)
    return v if not math.isnan(v) else float(s.iloc[-1])


def _atr14_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    a = float(atr.iloc[-1])
    c = float(close.iloc[-1])
    if c <= 0 or math.isnan(a):
        return 0.0
    return a / c


def _bb_pctb(close: pd.Series, window: int = 20) -> float:
    w = close.iloc[-window:]
    mid = float(w.mean())
    std = float(w.std())
    c = float(close.iloc[-1])
    if std == 0 or math.isnan(std):
        return 0.5
    upper = mid + 2 * std
    lower = mid - 2 * std
    den = upper - lower
    if abs(den) < 1e-12:
        return 0.5
    return float(max(0.0, min(1.0, (c - lower) / den)))


def _safe(val, default: float) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return default if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return default


def _ratio_diff(numerator, denominator) -> float:
    if numerator is None or denominator is None or denominator == 0:
        return 0.0
    return float((numerator - denominator) / denominator)


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).iloc[-period:].mean()
    loss = (-delta.clip(upper=0)).iloc[-period:].mean()
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def _safe_return(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback + 1:
        return 0.0
    return float((close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback])
