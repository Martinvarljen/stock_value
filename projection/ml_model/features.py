"""
features.py — Feature extraction for ML projection model.

Two modes:
  1. extract_features(record)          — live inference from a stock record
  2. extract_historical_features(...)  — historical features for training
"""

import sys
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Ensure stock_analyzer + projection are importable when run standalone
_root = Path(__file__).resolve().parents[3]
for _p in [str(_root / "stock_analyzer"), str(_root / "projection")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Technical features available from historical price data only
TECH_FEATURES = [
    "rsi14",
    "price_vs_ma50",
    "price_vs_ma200",
    "return_1m",
    "return_3m",
    "return_6m",
    "volatility_20d",
    "volume_ratio",
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

    # Technical
    feat["rsi14"] = _safe(record.get("rsi14"), 50.0) / 100.0

    ma50  = record.get("ma50")
    ma200 = record.get("ma200")
    feat["price_vs_ma50"]  = _ratio_diff(price, ma50)
    feat["price_vs_ma200"] = _ratio_diff(price, ma200)

    mom = record.get("momentum_metrics") or {}
    feat["return_1m"] = _safe((mom.get("return_1m") or {}).get("value"), 0.0)
    feat["return_3m"] = _safe((mom.get("return_3m") or {}).get("value"), 0.0)
    feat["return_6m"] = _safe((mom.get("return_6m") or {}).get("value"), 0.0)

    feat["volatility_20d"] = _safe(record.get("volatility_20d"), 0.20)
    feat["volume_ratio"]   = _safe(record.get("volume_ratio"), 1.0)

    # Valuation
    fv = record.get("fair_value_weighted")
    feat["valuation_upside"] = _ratio_diff(fv, price) if (fv and price > 0) else 0.0
    feat["fcf_yield"]        = _safe(record.get("fcf_yield"), 0.0)
    pe = record.get("pe_ratio")
    feat["earnings_yield"]   = (1.0 / pe) if (pe and pe > 0) else 0.0

    # Quality
    feat["operating_margin"] = _safe(record.get("operating_margin"), 0.0)
    roic = record.get("roic")
    wacc = (record.get("wacc_data") or {}).get("wacc")
    feat["roic_wacc_spread"] = float(roic - wacc) if (roic is not None and wacc is not None) else 0.0

    # Growth
    feat["revenue_cagr_5y"] = _safe(record.get("revenue_cagr_5y"), 0.0)

    # Risk
    feat["beta"]             = _safe(record.get("beta"), 1.0)
    feat["net_debt_ebitda"]  = _safe(record.get("net_debt_ebitda"), 0.0)
    feat["n_red_flags"]      = float(len(record.get("red_flags") or []))
    feat["n_critical_flags"] = float(len(record.get("critical_flags") or []))

    # Composite rule-based score as a meta-feature
    try:
        from projection_engine import _composite_score
        score, _ = _composite_score(record)
        feat["composite_score"] = float(score)
    except Exception:
        feat["composite_score"] = 0.0

    return feat


# ── historical feature extraction (for training) ──────────────────────────────

def extract_historical_features(history: pd.DataFrame, date: datetime) -> dict | None:
    """
    Extract TECH_FEATURES only from historical OHLCV data at a given date.
    Fundamental features are set to NaN (imputed as dataset medians during training).
    Returns None if insufficient history.
    """
    hist = history[history.index <= pd.Timestamp(date)]
    if len(hist) < 210:
        return None

    close  = hist["Close"]
    volume = hist["Volume"]
    price  = float(close.iloc[-1])

    feat: dict[str, float] = {}

    feat["rsi14"]          = _compute_rsi(close, 14) / 100.0
    feat["price_vs_ma50"]  = _ratio_diff(price, float(close.iloc[-50:].mean()))
    feat["price_vs_ma200"] = _ratio_diff(price, float(close.iloc[-200:].mean()))
    feat["return_1m"]      = _safe_return(close, 21)
    feat["return_3m"]      = _safe_return(close, 63)
    feat["return_6m"]      = _safe_return(close, 126)

    rets = close.pct_change().dropna()
    feat["volatility_20d"] = float(rets.iloc[-20:].std() * math.sqrt(252)) if len(rets) >= 20 else 0.20

    avg_vol = float(volume.iloc[-20:].mean())
    feat["volume_ratio"] = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

    # Fundamentals unavailable historically — filled with NaN → imputed as median
    for col in ALL_FEATURES:
        if col not in feat:
            feat[col] = float("nan")

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
    gain  = delta.clip(lower=0).iloc[-period:].mean()
    loss  = (-delta.clip(upper=0)).iloc[-period:].mean()
    if loss == 0:
        return 100.0
    return float(100 - 100 / (1 + gain / loss))


def _safe_return(close: pd.Series, lookback: int) -> float:
    if len(close) < lookback + 1:
        return 0.0
    return float((close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback])
