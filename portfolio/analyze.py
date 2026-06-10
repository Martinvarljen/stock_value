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
from backtesting.regime import build_regime_snapshot, spy_close_series

import logging

import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)

_SPY_CLOSE_CACHE: pd.Series | None = None


def load_spy_close(*, refresh: bool = False) -> pd.Series:
    """Session-cached SPY close series (one yfinance pull per process)."""
    global _SPY_CLOSE_CACHE
    if _SPY_CLOSE_CACHE is None or refresh:
        try:
            hist = yf.Ticker("SPY").history(period="max", interval="1d")
            _SPY_CLOSE_CACHE = spy_close_series(hist)
        except (OSError, ValueError, KeyError) as exc:
            _log.warning("SPY history fetch failed: %s", exc)
            if _SPY_CLOSE_CACHE is None:
                raise
    return _SPY_CLOSE_CACHE


def market_regime(as_of: datetime | None = None) -> dict[str, Any]:
    as_of = as_of or datetime.today()
    spy = load_spy_close()
    snap = build_regime_snapshot(spy, as_of)
    return {"as_of": as_of.date().isoformat(), **snap}


def analyze_ticker(
    ticker: str,
    *,
    margin_of_safety: float = 0.3,
    include_news: bool = False,
    news_days: int = 3,
    include_explanation: bool = True,
) -> dict[str, Any] | None:
    """Full stack + projections for one symbol.

    ``include_explanation`` defaults to ``True`` because the strategy now embeds
    the prose explanation into ``DecisionReport.extras["explanation"]`` so the
    decision memory log captures the rationale alongside each rating.
    """
    bundle, err = build_analysis_bundle(
        ticker, margin_of_safety, include_explanation=include_explanation
    )
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

    explanation = record.get("explanation") or {}
    tx = record.get("extended_technicals") or record.get("technical_extended") or {}
    atr_block = tx.get("atr_14") or {}
    last_bar = tx.get("last_bar") or {}

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
        "ohlcv_quality": record.get("ohlcv_quality"),
        "critical_flags": record.get("critical_flags") or [],
        "sector": record.get("sector"),
        "beta": record.get("beta"),
        # Risk inputs consumed by the broker for ATR-anchored stops + vol
        # targeted sizing. None when ``technical_extended`` is unavailable.
        "atr_pct": atr_block.get("pct_of_price"),
        "vol_60d_annual": tx.get("realised_vol_60d_annual"),
        "bar_low": last_bar.get("low"),
        "bar_high": last_bar.get("high"),
        "bar_open": last_bar.get("open"),
        "explanation_one_liner": explanation.get("one_liner"),
        "explanation_paragraphs": explanation.get("paragraphs"),
    }
