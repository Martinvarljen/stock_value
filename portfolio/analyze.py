"""Per-ticker analysis for one daily run (stateless; no memory between runs)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "stock_analyzer", _ROOT / "projection", str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import build_analysis_bundle
from projection_engine import generate_projections

from backtesting.ml_quant import ml_score_from_signal
from backtesting.regime import gross_exposure_scale, spy_bull_regime, spy_close_series

import pandas as pd
import yfinance as yf


def load_spy_close() -> pd.Series:
    hist = yf.Ticker("SPY").history(period="max", interval="1d")
    return spy_close_series(hist)


def market_regime(as_of: datetime | None = None) -> dict[str, Any]:
    as_of = as_of or datetime.today()
    spy = load_spy_close()
    bull = spy_bull_regime(spy, as_of)
    scale = gross_exposure_scale(spy, as_of)
    return {
        "as_of": as_of.date().isoformat(),
        "spy_bull": bull,
        "gross_exposure_scale": scale,
    }


def analyze_ticker(
    ticker: str,
    *,
    margin_of_safety: float = 0.3,
    include_news: bool = False,
    news_days: int = 3,
) -> dict[str, Any] | None:
    """Full stack + projections for one symbol."""
    bundle, err = build_analysis_bundle(ticker, margin_of_safety, include_explanation=False)
    if bundle is None:
        return {"ticker": ticker.upper(), "error": err, "ok": False}

    record = bundle.record
    news_result = None
    if include_news:
        try:
            from news_engine import analyze_news

            news_result = analyze_news(ticker, days=news_days)
        except Exception as e:
            news_result = {"error": str(e)}

    proj = generate_projections(record, horizon_days=120, news_result=news_result)
    if proj.get("error"):
        return {
            "ticker": ticker.upper(),
            "ok": False,
            "error": proj["error"],
            "classification": record.get("classification"),
        }

    sig = {
        "p_up_20d": proj.get("p_up_20d"),
        "p_up_60d": proj.get("p_up_60d"),
        "composite_score": proj.get("composite_score"),
    }
    score = ml_score_from_signal(sig)

    return {
        "ticker": ticker.upper(),
        "ok": True,
        "company_name": record.get("company_name"),
        "price": record.get("current_price"),
        "classification": record.get("classification"),
        "momentum_trend": record.get("momentum_trend"),
        "projection_signal": proj.get("signal"),
        "confidence": proj.get("confidence"),
        "ml_used": bool(proj.get("ml_used")),
        "p_up_5d": proj.get("p_up_5d"),
        "p_up_20d": proj.get("p_up_20d"),
        "p_up_60d": proj.get("p_up_60d"),
        "ml_score": score,
        "expected_return_20d": proj.get("expected_return_20d"),
        "trade_setup": record.get("trade_setup"),
        "critical_flags": record.get("critical_flags") or [],
    }
