"""
Shared fundamental analysis pipeline used by the CLI (main.py) and the dashboard.

Returns an AnalysisBundle with the merged record plus engine outputs needed for
console printing. Skips ETFs and failed / incomplete data fetches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_layer import collect_data
from quality_engine import analyze_quality
from financial_strength import analyze_financials
from valuation_engine import analyze_valuation
from growth_engine import analyze_growth
from risk_engine import analyze_risk
from red_flags import analyze_red_flags
from classification_engine import classify_stock
from sector_engine import apply_sector_context
from momentum_engine import analyze_momentum
from technical_extended import analyze_extended_technicals
from elliott_engine import analyze_elliott_context
from trade_setup_engine import build_trade_setup
from candle_patterns import analyze_candle_patterns
from ohlcv_validate import validate_ohlcv_from_data_dict
from market_structure import analyze_market_structure
from backtest_engine import analyze_price_history
from explanation_engine import generate_explanation


@dataclass
class AnalysisBundle:
    """One ticker: merged flat record plus intermediate engine dicts for CLI printing."""

    record: dict[str, Any]
    sector_result: dict[str, Any]
    quality_result: dict[str, Any]
    financial_result: dict[str, Any]
    valuation_result: dict[str, Any]
    growth_result: dict[str, Any]
    risk_result: dict[str, Any]
    red_flag_result: dict[str, Any]
    momentum_result: dict[str, Any]
    backtest_result: dict[str, Any]


def build_analysis_bundle(
    ticker: str,
    margin_of_safety: float,
    *,
    include_explanation: bool = True,
) -> tuple[AnalysisBundle | None, str | None]:
    """
    Run the full fundamental + technical stack for one ticker.

    Returns (bundle, None) on success, or (None, error_message) when the ticker
    should be skipped (ETF, data fetch error, or missing price).
    """
    data = collect_data(ticker)

    if data.get("quote_type") == "ETF":
        return None, "ETFs are not supported (no fundamental data)"

    if data.get("error"):
        return None, f"Data error: {data.get('error')}"

    if data.get("current_price") is None:
        return None, "No price data — cannot complete analysis"

    sector_result = apply_sector_context(data)

    quality_result = analyze_quality(data)
    financial_result = analyze_financials(data)

    valuation_result = analyze_valuation(
        {**data, "sector_result": sector_result},
        margin_of_safety=margin_of_safety,
        wacc_adjustment=sector_result["wacc_adjustment"],
        terminal_growth_range=sector_result.get("terminal_growth_range"),
    )

    growth_result = analyze_growth(data)
    risk_result = analyze_risk(data)

    wacc = valuation_result.get("wacc_data", {}).get("wacc")
    red_flag_result = analyze_red_flags(data, wacc=wacc)

    all_critical = (
        (financial_result.get("critical_flags") or [])
        + (risk_result.get("critical_flags") or [])
    )

    record: dict[str, Any] = {
        **data,
        "quality_metrics": quality_result["quality_metrics"],
        "quality_flags": quality_result["quality_flags"],
        "financial_metrics": financial_result["financial_metrics"],
        "financial_flags": financial_result["financial_flags"],
        "valuation_metrics": valuation_result["valuation_metrics"],
        "valuation_flags": valuation_result["valuation_flags"],
        "fair_value_weighted": valuation_result["fair_value_weighted"],
        "buy_below_price": valuation_result["buy_below_price"],
        "wacc_data": valuation_result["wacc_data"],
        "scenarios": valuation_result["scenarios"],
        "tv_sensitivity": valuation_result.get("tv_sensitivity"),
        "growth_metrics": growth_result["growth_metrics"],
        "growth_flags": growth_result["growth_flags"],
        "risk_metrics": risk_result["risk_metrics"],
        "risk_flags": risk_result["risk_flags"],
        "red_flags": red_flag_result["red_flags"],
        "red_flag_summary": red_flag_result["summary"],
        "critical_flags": all_critical,
    }

    momentum_result = analyze_momentum(data)
    backtest_result = analyze_price_history(data)

    record["momentum_metrics"] = momentum_result["momentum_metrics"]
    record["momentum_flags"] = momentum_result["momentum_flags"]
    record["momentum_trend"] = momentum_result["trend"]
    record["backtest_metrics"] = backtest_result["backtest_metrics"]
    record["backtest_flags"] = backtest_result["backtest_flags"]

    record["extended_technicals"] = analyze_extended_technicals(data)
    record["elliott_context"] = analyze_elliott_context(data)
    record["candle_patterns"] = analyze_candle_patterns(data)
    record["ohlcv_quality"] = validate_ohlcv_from_data_dict(data)
    record["market_structure"] = analyze_market_structure(data)
    record["sector_result"] = sector_result

    clf_result = classify_stock(record)
    record["classification_result"] = clf_result
    record["classification"] = clf_result["classification"]
    record["trade_setup"] = build_trade_setup(record)

    if include_explanation:
        record["explanation"] = generate_explanation(record)

    bundle = AnalysisBundle(
        record=record,
        sector_result=sector_result,
        quality_result=quality_result,
        financial_result=financial_result,
        valuation_result=valuation_result,
        growth_result=growth_result,
        risk_result=risk_result,
        red_flag_result=red_flag_result,
        momentum_result=momentum_result,
        backtest_result=backtest_result,
    )
    return bundle, None
