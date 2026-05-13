"""
features.py — Feature extraction for ML projection model.

Two modes:
  1. extract_features(record)          — live inference from a stock record
  2. extract_historical_features(...)  — historical features for training
"""

import sys
import math
from pathlib import Path
from datetime import datetime

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
TECH_FEATURES = [
    "rsi14",
    "price_vs_ma50",
    "price_vs_ma200",
    "return_5d",
    "return_10d",
    "return_1m",
    "return_3m",
    "return_6m",
    "volatility_20d",
    "vol_ratio_20_60",
    "volume_ratio",
    "atr14_rel",
    "bb_pctb",
    "macd_norm",
    "ma_spread_50_200",
    "hl_range_20d",
]

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
    return feat


# ── historical feature extraction (for training) ──────────────────────────────

def extract_historical_features(history: pd.DataFrame, date: datetime) -> dict | None:
    """
    Extract TECH_FEATURES from historical OHLCV at a given date.
    Fundamental features are NaN (imputed as dataset medians if ever used).
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

    return feat


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
