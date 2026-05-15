"""Fast point-in-time ML score for backtests (no live yfinance per day)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "stock_analyzer", _ROOT / "projection"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from backtesting.strategy_backtest import (
    enrich_point_in_time_technicals,
    reconstruct_data_at,
    reconstruct_price_only_at,
)
from financial_strength import analyze_financials
from risk_engine import analyze_risk
from ml_model.features import extract_features
from ml_model.predictor import ml_predict


def score_ticker_at(
    raw: dict,
    as_of: datetime,
    *,
    spy_close: pd.Series,
) -> dict | None:
    """ML-centric analysis as of ``as_of`` using cached yfinance bundle."""
    ticker = raw.get("ticker", "").upper()
    data = reconstruct_data_at(raw, as_of)
    fundamentals_ok = data is not None
    if data is None:
        data = reconstruct_price_only_at(raw, as_of)
    if data is None:
        return None

    data["feature_as_of"] = as_of
    data["checkpoint_date"] = as_of
    data["spy_close_series"] = spy_close
    if fundamentals_ok:
        enrich_point_in_time_technicals(data, raw, as_of)

    if fundamentals_ok:
        fin = analyze_financials(data)
        risk = analyze_risk(data)
        critical = (fin.get("critical_flags") or []) + (risk.get("critical_flags") or [])
    else:
        critical = []

    feats = extract_features(data)
    ml = ml_predict(feats, horizons=[20])
    p_up = float(ml[20]) if ml and 20 in ml else None

    return {
        "ticker": ticker,
        "ok": p_up is not None,
        "price": float(data["current_price"]),
        "p_up_20d": p_up,
        "ml_score": p_up,
        "critical_flags": critical,
        "error": None if p_up is not None else "ML unavailable",
    }
