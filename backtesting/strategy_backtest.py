"""
strategy_backtest.py  —  Historical strategy backtesting engine

Tests whether the stock_analyzer classification signals predicted future returns.

Approach:
  1. For each ticker, pull all available financial data from yfinance (5Y history).
  2. For quarterly checkpoints going back ~2 years:
     - Filter financials to only what was publicly available at that date
       (fiscal year end + 90 day reporting lag).
     - Get the historical closing price at that date.
     - Reconstruct the data dict and run the full analysis pipeline.
     - Record the classification.
  3. Measure forward returns (3M, 6M, 12M) from each checkpoint.
  4. Aggregate: average return by classification tier, hit rate, etc.

Usage:
    python strategy_backtest.py                       # backtest default tickers
    python strategy_backtest.py AAPL MSFT SHEL        # specific tickers
    python strategy_backtest.py --lookback 3          # 3 years lookback
"""

import sys
import os
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add stock_analyzer to path so we can import the engines
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "stock_analyzer"))

from quality_engine import analyze_quality
from financial_strength import analyze_financials
from valuation_engine import analyze_valuation
from growth_engine import analyze_growth
from risk_engine import analyze_risk
from red_flags import analyze_red_flags
from classification_engine import classify_stock
from sector_engine import apply_sector_context
from utils import _pct

# ── console encoding fix (Windows) ─────────────────────────────────────────────

def _configure_console():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

_configure_console()


# ── constants ──────────────────────────────────────────────────────────────────

REPORTING_LAG_DAYS = 90       # assume financials available 90 days after fiscal year end
MARGIN_OF_SAFETY   = 0.30
DEFAULT_LOOKBACK   = 2        # years
CHECKPOINT_FREQ    = "Q"      # quarterly checkpoints
FORWARD_PERIODS    = [3, 6, 12]  # months

TIER_ORDER = ["STRONG BUY", "BUY", "WATCHLIST", "HOLD", "AVOID", "STRONG AVOID"]


# ── data collection ───────────────────────────────────────────────────────────

def _safe(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return value
    except Exception:
        return default


def _series_to_list(series, n=5):
    if series is None or series.empty:
        return [None] * n
    series = series.dropna().sort_index()
    values = [_safe(v) for v in series.values]
    if len(values) >= n:
        return values[-n:]
    return [None] * (n - len(values)) + values


def _cagr(start, end, years):
    try:
        if start is None or end is None or years <= 0 or start <= 0:
            return None
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None


def _pct_change_list(lst):
    changes = []
    for i in range(1, len(lst)):
        a, b = lst[i - 1], lst[i]
        if a is None or b is None or a == 0:
            changes.append(None)
        else:
            changes.append((b - a) / abs(a))
    return changes


def _list_cagr(lst, require_positive_start=False):
    indexed = [(i, v) for i, v in enumerate(lst) if v is not None]
    if len(indexed) < 2:
        return None
    _, v_start = indexed[0]
    _, v_end = indexed[-1]
    years = indexed[-1][0] - indexed[0][0]
    if years <= 0 or (require_positive_start and v_start <= 0):
        return None
    return _cagr(v_start, v_end, years)


def _row(df, *keys):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series(dtype=float)


def _last(lst):
    for v in reversed(lst):
        if v is not None:
            return v
    return None


# ── point-in-time data reconstruction ─────────────────────────────────────────

def _filter_before(df, cutoff_date):
    """Filter a financials DataFrame to only columns (fiscal periods) available before cutoff."""
    if df is None or df.empty:
        return None
    cutoff = pd.Timestamp(cutoff_date)
    valid_cols = []
    for c in df.columns:
        col_ts = pd.Timestamp(c)
        if col_ts.tz is not None:
            col_ts = col_ts.tz_localize(None)
        if col_ts + pd.Timedelta(days=REPORTING_LAG_DAYS) <= cutoff:
            valid_cols.append(c)
    if not valid_cols:
        return None
    return df[valid_cols].sort_index(axis=1)


def _get_price_at(price_history: pd.DataFrame, target_date: datetime) -> float | None:
    """Get the closing price closest to (but not after) target_date."""
    if price_history is None or price_history.empty:
        return None
    target = pd.Timestamp(target_date)
    idx = price_history.index.tz_localize(None) if price_history.index.tz else price_history.index
    valid = price_history[idx <= target]
    if valid.empty:
        return None
    return float(valid["Close"].iloc[-1])


def _get_forward_price(price_history: pd.DataFrame, from_date: datetime, months: int) -> float | None:
    """Get closing price approximately `months` months after from_date."""
    if price_history is None or price_history.empty:
        return None
    idx = price_history.index.tz_localize(None) if price_history.index.tz else price_history.index
    target = pd.Timestamp(from_date + timedelta(days=months * 30.44))
    from_ts = pd.Timestamp(from_date)
    future = price_history[idx >= target]
    if future.empty:
        latest = price_history[idx > from_ts]
        if latest.empty:
            return None
        naive_last = idx[-1] if not latest.empty else None
        if naive_last and (naive_last - from_ts).days >= months * 30.44 * 0.8:
            return float(latest["Close"].iloc[-1])
        return None
    return float(future["Close"].iloc[0])


def collect_raw_yfinance(ticker: str) -> dict:
    """Pull all raw yfinance data once per ticker (financials + full price history)."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    try:
        inc = tk.financials
        inc_a = inc.sort_index(axis=1) if inc is not None and not inc.empty else None
    except Exception:
        inc_a = None

    try:
        bal = tk.balance_sheet
        bal_a = bal.sort_index(axis=1) if bal is not None and not bal.empty else None
    except Exception:
        bal_a = None

    try:
        cf = tk.cashflow
        cf_a = cf.sort_index(axis=1) if cf is not None and not cf.empty else None
    except Exception:
        cf_a = None

    # Pull full price history (daily) for the lookback + forward period
    try:
        hist = tk.history(period="max", interval="1d")
    except Exception:
        hist = pd.DataFrame()

    return {
        "ticker": ticker,
        "info": info,
        "income_statement": inc_a,
        "balance_sheet": bal_a,
        "cash_flow": cf_a,
        "price_history": hist,
    }


def reconstruct_data_at(raw: dict, as_of_date: datetime) -> dict | None:
    """
    Reconstruct a data dict as it would have appeared on `as_of_date`.
    Only uses financials published before that date and the price on that date.
    Returns None if insufficient data.
    """
    info = raw["info"]
    ticker = raw["ticker"]

    # Filter financials to point-in-time
    inc_a = _filter_before(raw["income_statement"], as_of_date)
    bal_a = _filter_before(raw["balance_sheet"], as_of_date)
    cf_a = _filter_before(raw["cash_flow"], as_of_date)

    if inc_a is None or inc_a.shape[1] < 2:
        return None

    # Get price at checkpoint date
    price = _get_price_at(raw["price_history"], as_of_date)
    if price is None or price <= 0:
        return None

    # Handle GBp → GBP
    currency = _safe(info.get("currency"))
    if currency == "GBp":
        price = price / 100
        currency = "GBP"

    result = {
        "ticker": ticker,
        "error": None,
        "company_name": _safe(info.get("longName") or info.get("shortName")),
        "quote_type": _safe(info.get("quoteType")),
        "sector": _safe(info.get("sector")),
        "industry": _safe(info.get("industry")),
        "exchange": _safe(info.get("exchange")),
        "currency": currency,
        "current_price": price,
        "eur_rate": None,
        "price_eur": None,
    }

    # Market cap approximation (shares × price at that date)
    shares_outstanding = _safe(info.get("sharesOutstanding"))
    result["market_cap"] = shares_outstanding * price if shares_outstanding else None
    result["shares_outstanding"] = shares_outstanding
    result["beta"] = _safe(info.get("beta"))

    # Income statement rows
    rev_s = _row(inc_a, "Total Revenue", "Revenue")
    gp_s = _row(inc_a, "Gross Profit")
    ebit_s = _row(inc_a, "EBIT", "Operating Income")
    ni_s = _row(inc_a, "Net Income")
    eps_s = _row(inc_a, "Basic EPS", "Diluted EPS")
    int_s = _row(inc_a, "Interest Expense")
    dep_s = _row(inc_a, "Reconciled Depreciation", "Depreciation And Amortization")
    tax_s = _row(inc_a, "Tax Provision", "Income Tax Expense")
    pbt_s = _row(inc_a, "Pretax Income")

    # Balance sheet rows
    assets_s = _row(bal_a, "Total Assets")
    liab_s = _row(bal_a, "Total Liabilities Net Minority Interest", "Total Liabilities")
    ltd_s = _row(bal_a, "Long Term Debt")
    std_s = _row(bal_a, "Current Debt", "Short Term Debt", "Current Portion Of Long Term Debt And Capital Lease Obligation")
    cash_s = _row(bal_a, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    equity_s = _row(bal_a, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")
    shares_s = _row(bal_a, "Share Issued", "Ordinary Shares Number")
    cur_assets = _row(bal_a, "Current Assets")
    cur_liab = _row(bal_a, "Current Liabilities")

    # Cash flow rows
    ocf_s = _row(cf_a, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex_s = _row(cf_a, "Capital Expenditure")
    div_s = _row(cf_a, "Common Stock Dividend Paid", "Cash Dividends Paid")
    sbc_s = _row(cf_a, "Stock Based Compensation")

    # Store 5-year annual lists
    result["revenue_5y"] = _series_to_list(rev_s)
    result["gross_profit_5y"] = _series_to_list(gp_s)
    result["ebit_5y"] = _series_to_list(ebit_s)
    result["net_income_5y"] = _series_to_list(ni_s)
    result["eps_5y"] = _series_to_list(eps_s)
    result["interest_expense_5y"] = _series_to_list(int_s)
    result["depreciation_5y"] = _series_to_list(dep_s)
    result["tax_provision_5y"] = _series_to_list(tax_s)
    result["pretax_income_5y"] = _series_to_list(pbt_s)

    result["total_assets_5y"] = _series_to_list(assets_s)
    result["total_liabilities_5y"] = _series_to_list(liab_s)
    result["total_debt_5y"] = _series_to_list(
        (ltd_s.fillna(0) + std_s.fillna(0)) if (not ltd_s.empty or not std_s.empty) else pd.Series(dtype=float)
    )
    result["cash_5y"] = _series_to_list(cash_s)
    result["equity_5y"] = _series_to_list(equity_s)
    result["shares_5y"] = _series_to_list(shares_s)
    result["current_assets_5y"] = _series_to_list(cur_assets)
    result["current_liab_5y"] = _series_to_list(cur_liab)
    result["ocf_5y"] = _series_to_list(ocf_s)
    result["capex_5y"] = _series_to_list(capex_s)
    result["dividends_5y"] = _series_to_list(div_s)
    result["sbc_5y"] = _series_to_list(sbc_s)

    # FCF
    fcf_list = []
    for o, c in zip(result["ocf_5y"], result["capex_5y"]):
        if o is None:
            fcf_list.append(None)
        elif c is None:
            fcf_list.append(o)
        else:
            fcf_list.append(o - abs(c))
    result["fcf_5y"] = fcf_list

    # Latest-year snapshots
    revenue = _last(result["revenue_5y"])
    gross_profit = _last(result["gross_profit_5y"])
    ebit = _last(result["ebit_5y"])
    net_income = _last(result["net_income_5y"])
    ocf = _last(result["ocf_5y"])
    capex = _last(result["capex_5y"])
    fcf = _last(result["fcf_5y"])
    total_debt = _last(result["total_debt_5y"])
    cash = _last(result["cash_5y"])
    equity = _last(result["equity_5y"])
    interest_exp = _last(result["interest_expense_5y"])
    dep = _last(result["depreciation_5y"])
    cur_a = _last(result["current_assets_5y"])
    cur_l = _last(result["current_liab_5y"])
    tax_prov = _last(result["tax_provision_5y"])
    pretax = _last(result["pretax_income_5y"])

    result.update({
        "revenue": revenue, "gross_profit": gross_profit,
        "ebit": ebit, "net_income": net_income,
        "ocf": ocf, "capex": capex, "fcf": fcf,
        "total_debt": total_debt, "cash": cash,
        "equity": equity, "interest_expense": interest_exp,
        "depreciation": dep,
    })

    # Derived ratios
    mktcap = result["market_cap"]
    ev = (mktcap + (total_debt or 0) - (cash or 0)) if mktcap else None
    result["enterprise_value"] = ev

    result["gross_margin"] = _safe(gross_profit / revenue) if (revenue not in (None, 0) and gross_profit is not None) else None
    result["operating_margin"] = _safe(ebit / revenue) if (revenue not in (None, 0) and ebit is not None) else None
    result["net_margin"] = _safe(net_income / revenue) if (revenue not in (None, 0) and net_income is not None) else None

    ebitda = None
    if ebit is not None and dep is not None:
        ebitda = ebit + dep
    elif ebit is not None:
        ebitda = ebit
    result["ebitda"] = ebitda

    result["roe"] = _safe(net_income / equity) if (net_income is not None and equity not in (None, 0)) else None
    assets_last = _last(result["total_assets_5y"])
    result["roa"] = _safe(net_income / assets_last) if (net_income is not None and assets_last not in (None, 0)) else None

    eff_tax = None
    if tax_prov is not None and pretax is not None and pretax != 0:
        eff_tax = max(0.0, min(0.5, tax_prov / pretax))
    result["effective_tax_rate"] = _safe(eff_tax, 0.22)

    net_debt = (total_debt - cash) if (total_debt is not None and cash is not None) else total_debt
    result["net_debt"] = net_debt
    invested_capital = (equity + (net_debt or 0)) if equity else None
    if ebit is not None and invested_capital and invested_capital != 0:
        t = result["effective_tax_rate"] or 0.22
        result["roic"] = _safe(ebit * (1 - t) / invested_capital)
    else:
        result["roic"] = None

    result["debt_equity"] = _safe(total_debt / equity) if (total_debt is not None and equity not in (None, 0)) else None
    result["net_debt_ebitda"] = _safe(net_debt / ebitda) if (net_debt is not None and ebitda not in (None, 0)) else None
    result["interest_coverage"] = _safe(abs(ebit) / abs(interest_exp)) if (ebit is not None and interest_exp not in (None, 0)) else None
    result["current_ratio"] = _safe(cur_a / cur_l) if (cur_a is not None and cur_l not in (None, 0)) else None
    result["fcf_yield"] = _safe(fcf / mktcap) if (fcf is not None and mktcap not in (None, 0)) else None

    result["ev_ebit"] = _safe(ev / ebit) if (ev is not None and ebit not in (None, 0)) else None
    result["ev_ebitda"] = _safe(ev / ebitda) if (ev is not None and ebitda not in (None, 0)) else None
    result["ev_fcf"] = _safe(ev / fcf) if (ev is not None and fcf not in (None, 0)) else None

    # Approximate P/E using point-in-time price and trailing EPS
    trailing_eps = _last(result["eps_5y"])
    result["trailing_eps"] = trailing_eps
    result["trailing_pe"] = _safe(price / trailing_eps) if (trailing_eps and trailing_eps > 0) else None
    result["forward_pe"] = None
    result["pb_ratio"] = _safe(price * (shares_outstanding or 0) / equity) if (equity and equity > 0 and shares_outstanding) else None
    result["dividend_yield"] = _safe(info.get("dividendYield"))
    result["peg_ratio"] = None
    result["ev_revenue"] = _safe(ev / revenue) if (ev is not None and revenue not in (None, 0)) else None

    # Growth CAGRs
    result["revenue_cagr_5y"] = _list_cagr(result["revenue_5y"])
    result["eps_cagr_5y"] = _list_cagr(result["eps_5y"], require_positive_start=True)
    result["fcf_cagr_5y"] = _list_cagr(result["fcf_5y"], require_positive_start=True)
    result["revenue_yoy_changes"] = _pct_change_list(result["revenue_5y"])

    # Shares dilution
    sh_now = shares_outstanding or _last(result["shares_5y"])
    sh_old = next((v for v in result["shares_5y"] if v is not None), None)
    if sh_now and sh_old and sh_old != 0:
        result["shares_change_pct"] = (sh_now - sh_old) / sh_old
    else:
        result["shares_change_pct"] = None

    # Capex ratios
    capex_ratios = []
    for r, c in zip(result["revenue_5y"], result["capex_5y"]):
        if r and c and r != 0:
            capex_ratios.append(abs(c) / r)
    result["capex_pct_revenue"] = float(np.mean(capex_ratios)) if capex_ratios else None

    dep_5y = result.get("depreciation_5y") or []
    net_capex_ratios = []
    for r, c, d in zip(result["revenue_5y"], result["capex_5y"], dep_5y):
        if r and c and r != 0:
            gross_cx = abs(c)
            da = abs(d) if d else 0.0
            net_cx = max(gross_cx - da, 0.0)
            net_capex_ratios.append(net_cx / r)
    result["net_capex_pct_revenue"] = float(np.mean(net_capex_ratios)) if net_capex_ratios else None

    # Price history fields (approximated from what we have up to as_of_date)
    ph = raw["price_history"]
    if ph is not None and not ph.empty:
        cutoff = pd.Timestamp(as_of_date)
        ph_idx = ph.index.tz_localize(None) if ph.index.tz else ph.index
        hist_before = ph[ph_idx <= cutoff]
        if len(hist_before) >= 200:
            result["ma50"] = float(hist_before["Close"].iloc[-50:].mean())
            result["ma200"] = float(hist_before["Close"].iloc[-200:].mean())
        else:
            result["ma50"] = None
            result["ma200"] = None

        # 1Y return
        if len(hist_before) > 252:
            p_1y_ago = float(hist_before["Close"].iloc[-252])
            result["return_1y"] = (price - p_1y_ago) / p_1y_ago if p_1y_ago > 0 else None
        else:
            result["return_1y"] = None
        result["return_3y"] = None
    else:
        result["ma50"] = None
        result["ma200"] = None
        result["return_1y"] = None
        result["return_3y"] = None

    result["rsi14"] = None
    result["close_1y"] = []
    result["close_5y_monthly"] = []
    result["week52_high"] = None
    result["week52_low"] = None
    result["pe_5y_min"] = None
    result["pe_5y_max"] = None
    result["pe_5y_median"] = None
    result["data_quality_score"] = 60  # approximate since we have limited data

    return result


# ── run the full pipeline at a point in time ──────────────────────────────────

def classify_at(data: dict) -> str | None:
    """Run the full engine pipeline on reconstructed data and return classification."""
    try:
        sector_result = apply_sector_context(data)

        quality_result = analyze_quality(data)
        financial_result = analyze_financials(data)

        valuation_result = analyze_valuation(
            {**data, "sector_result": sector_result},
            margin_of_safety=MARGIN_OF_SAFETY,
            wacc_adjustment=sector_result["wacc_adjustment"],
            terminal_growth_range=sector_result.get("terminal_growth_range"),
        )

        growth_result = analyze_growth(data)
        risk_result = analyze_risk(data)

        wacc = valuation_result.get("wacc_data", {}).get("wacc")
        red_flag_result = analyze_red_flags(data, wacc=wacc)

        all_critical = (
            (financial_result.get("critical_flags") or []) +
            (risk_result.get("critical_flags") or [])
        )

        record = {
            **data,
            "valuation_metrics": valuation_result["valuation_metrics"],
            "fair_value_weighted": valuation_result["fair_value_weighted"],
            "buy_below_price": valuation_result["buy_below_price"],
            "wacc_data": valuation_result["wacc_data"],
            "red_flags": red_flag_result["red_flags"],
            "critical_flags": all_critical,
            "sector_result": sector_result,
        }

        clf_result = classify_stock(record)
        return clf_result["classification"]

    except Exception as e:
        return None


# ── generate checkpoint dates ─────────────────────────────────────────────────

def _generate_checkpoints(lookback_years: int = 2) -> list[datetime]:
    """Generate quarterly checkpoint dates going back lookback_years from today."""
    today = datetime.today()
    checkpoints = []

    # Go back to the start of lookback period, generate quarter-end dates
    start = today - timedelta(days=lookback_years * 365)

    # Find first quarter-end after start
    year = start.year
    quarter_ends = [
        datetime(year, 3, 31),
        datetime(year, 6, 30),
        datetime(year, 9, 30),
        datetime(year, 12, 31),
    ]

    current = start
    while current < today - timedelta(days=90):  # need at least 3M forward data
        for qe in quarter_ends:
            if qe > start and qe < today - timedelta(days=90):
                checkpoints.append(qe)
        year += 1
        quarter_ends = [
            datetime(year, 3, 31),
            datetime(year, 6, 30),
            datetime(year, 9, 30),
            datetime(year, 12, 31),
        ]
        if quarter_ends[0] > today:
            break

    # Deduplicate and sort
    checkpoints = sorted(set(checkpoints))
    return checkpoints


# ── main backtest function ────────────────────────────────────────────────────

def run_backtest(tickers: list[str], lookback_years: int = DEFAULT_LOOKBACK) -> dict:
    """
    Run the full strategy backtest.

    Returns dict with:
      - signals:    list of {ticker, date, classification, fwd_3m, fwd_6m, fwd_12m}
      - by_tier:    aggregated stats per classification tier
      - summary:    overall summary statistics
    """
    checkpoints = _generate_checkpoints(lookback_years)
    print(f"\n{'═' * 70}")
    print(f"  STRATEGY BACKTEST")
    print(f"{'═' * 70}")
    print(f"  Tickers:      {len(tickers)} stocks")
    print(f"  Lookback:     {lookback_years} years ({len(checkpoints)} quarterly checkpoints)")
    print(f"  Checkpoints:  {checkpoints[0].strftime('%Y-%m-%d')} → {checkpoints[-1].strftime('%Y-%m-%d')}")
    print(f"  Forward:      {FORWARD_PERIODS} months")
    print(f"{'═' * 70}\n")

    signals = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker} ... ", end="", flush=True)

        try:
            raw = collect_raw_yfinance(ticker)
        except Exception as e:
            print(f"FAILED (data error: {e})")
            continue

        if raw["income_statement"] is None:
            print("SKIPPED (no financials)")
            continue

        ticker_signals = 0
        for cp_date in checkpoints:
            data = reconstruct_data_at(raw, cp_date)
            if data is None:
                continue

            classification = classify_at(data)
            if classification is None:
                continue

            # Measure forward returns
            price_at_cp = data["current_price"]
            fwd_returns = {}
            for months in FORWARD_PERIODS:
                fwd_price = _get_forward_price(raw["price_history"], cp_date, months)
                if fwd_price and price_at_cp and price_at_cp > 0:
                    fwd_returns[f"fwd_{months}m"] = (fwd_price - price_at_cp) / price_at_cp
                else:
                    fwd_returns[f"fwd_{months}m"] = None

            signals.append({
                "ticker": ticker,
                "date": cp_date,
                "classification": classification,
                "price": price_at_cp,
                **fwd_returns,
            })
            ticker_signals += 1

        print(f"{ticker_signals} signals")

    # Aggregate results
    by_tier = _aggregate_by_tier(signals)
    summary = _compute_summary(signals, by_tier)

    return {
        "signals": signals,
        "by_tier": by_tier,
        "summary": summary,
        "checkpoints": checkpoints,
        "tickers": tickers,
    }


# ── aggregation ───────────────────────────────────────────────────────────────

def _aggregate_by_tier(signals: list[dict]) -> dict:
    """Compute average forward returns and hit rates by classification tier."""
    tiers = {}

    for tier in TIER_ORDER:
        tier_signals = [s for s in signals if s["classification"] == tier]
        if not tier_signals:
            continue

        tier_data = {"count": len(tier_signals), "signals": tier_signals}

        for period in FORWARD_PERIODS:
            key = f"fwd_{period}m"
            returns = [s[key] for s in tier_signals if s[key] is not None]
            if returns:
                tier_data[f"avg_{period}m"] = sum(returns) / len(returns)
                tier_data[f"median_{period}m"] = sorted(returns)[len(returns) // 2]
                tier_data[f"hit_rate_{period}m"] = sum(1 for r in returns if r > 0) / len(returns)
                tier_data[f"n_{period}m"] = len(returns)
                tier_data[f"best_{period}m"] = max(returns)
                tier_data[f"worst_{period}m"] = min(returns)
            else:
                tier_data[f"avg_{period}m"] = None
                tier_data[f"median_{period}m"] = None
                tier_data[f"hit_rate_{period}m"] = None
                tier_data[f"n_{period}m"] = 0

        tiers[tier] = tier_data

    return tiers


def _compute_summary(signals: list[dict], by_tier: dict) -> dict:
    """Compute overall backtest summary statistics."""
    total_signals = len(signals)
    if total_signals == 0:
        return {"total_signals": 0, "verdict": "No signals generated"}

    # Signal effectiveness: do buy-tier stocks beat avoid-tier stocks?
    buy_tiers = ["STRONG BUY", "BUY"]
    avoid_tiers = ["AVOID", "STRONG AVOID"]

    buy_returns_6m = []
    avoid_returns_6m = []
    for s in signals:
        r = s.get("fwd_6m")
        if r is None:
            continue
        if s["classification"] in buy_tiers:
            buy_returns_6m.append(r)
        elif s["classification"] in avoid_tiers:
            avoid_returns_6m.append(r)

    avg_buy = sum(buy_returns_6m) / len(buy_returns_6m) if buy_returns_6m else None
    avg_avoid = sum(avoid_returns_6m) / len(avoid_returns_6m) if avoid_returns_6m else None

    spread = None
    if avg_buy is not None and avg_avoid is not None:
        spread = avg_buy - avg_avoid

    # Classification distribution
    distribution = {}
    for tier in TIER_ORDER:
        count = sum(1 for s in signals if s["classification"] == tier)
        if count > 0:
            distribution[tier] = count

    return {
        "total_signals": total_signals,
        "unique_tickers": len(set(s["ticker"] for s in signals)),
        "avg_buy_6m": avg_buy,
        "avg_avoid_6m": avg_avoid,
        "buy_vs_avoid_spread_6m": spread,
        "n_buy_signals": len(buy_returns_6m),
        "n_avoid_signals": len(avoid_returns_6m),
        "distribution": distribution,
    }


# ── display ───────────────────────────────────────────────────────────────────

def print_backtest_results(results: dict):
    """Print formatted backtest results to console."""
    by_tier = results["by_tier"]
    summary = results["summary"]

    print(f"\n{'═' * 70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'═' * 70}")
    print(f"  Total signals: {summary['total_signals']}  "
          f"({summary.get('unique_tickers', 0)} tickers × multiple checkpoints)")

    # Distribution
    print(f"\n  Signal Distribution:")
    dist = summary.get("distribution", {})
    for tier in TIER_ORDER:
        if tier in dist:
            pct = dist[tier] / summary["total_signals"]
            bar = "█" * int(pct * 30)
            print(f"    {tier:<14} {dist[tier]:>4} signals ({pct:>5.1%})  {bar}")

    # Returns by tier
    print(f"\n  {'─' * 66}")
    print(f"  {'TIER':<14} {'Count':>6} │ {'Avg 3M':>8} {'Avg 6M':>8} {'Avg 12M':>8} │ {'Hit 6M':>7}")
    print(f"  {'─' * 66}")

    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        t = by_tier[tier]
        avg_3 = f"{t['avg_3m']:+.1%}" if t.get("avg_3m") is not None else "  N/A "
        avg_6 = f"{t['avg_6m']:+.1%}" if t.get("avg_6m") is not None else "  N/A "
        avg_12 = f"{t['avg_12m']:+.1%}" if t.get("avg_12m") is not None else "  N/A "
        hit_6 = f"{t['hit_rate_6m']:.0%}" if t.get("hit_rate_6m") is not None else " N/A "
        print(f"  {tier:<14} {t['count']:>6} │ {avg_3:>8} {avg_6:>8} {avg_12:>8} │ {hit_6:>7}")

    print(f"  {'─' * 66}")

    # Key verdict
    spread = summary.get("buy_vs_avoid_spread_6m")
    print(f"\n  KEY FINDINGS:")
    if summary.get("avg_buy_6m") is not None:
        print(f"    • BUY-tier avg 6M return:   {summary['avg_buy_6m']:+.1%}  "
              f"({summary['n_buy_signals']} signals)")
    if summary.get("avg_avoid_6m") is not None:
        print(f"    • AVOID-tier avg 6M return: {summary['avg_avoid_6m']:+.1%}  "
              f"({summary['n_avoid_signals']} signals)")
    if spread is not None:
        direction = "outperformed" if spread > 0 else "underperformed"
        print(f"    • BUY vs AVOID spread:      {spread:+.1%}  "
              f"(BUY {direction} AVOID by {abs(spread):.1%})")

        if spread > 0.05:
            print(f"\n  VERDICT: Strategy shows positive signal value — "
                  f"BUY classifications outperformed AVOID by {spread:.1%} over 6 months.")
        elif spread > 0:
            print(f"\n  VERDICT: Weak positive signal — BUY marginally beat AVOID. "
                  f"Consider widening margin of safety or refining criteria.")
        else:
            print(f"\n  VERDICT: Signal inversion detected — AVOID stocks outperformed BUY. "
                  f"Review classification thresholds or check for value-trap bias.")
    else:
        print(f"\n  VERDICT: Insufficient BUY/AVOID signals to compute spread.")

    print(f"{'═' * 70}")

    # Per-ticker breakdown
    _print_ticker_summary(results["signals"])


def _print_ticker_summary(signals: list[dict]):
    """Print per-ticker signal summary."""
    tickers = sorted(set(s["ticker"] for s in signals))
    if not tickers:
        return

    print(f"\n  PER-TICKER BREAKDOWN:")
    print(f"  {'Ticker':<10} {'Signals':>8} {'Most Common':>14} {'Avg 6M Fwd':>12} {'Best':>8} {'Worst':>8}")
    print(f"  {'─' * 62}")

    for ticker in tickers:
        t_signals = [s for s in signals if s["ticker"] == ticker]
        classifications = [s["classification"] for s in t_signals]
        most_common = max(set(classifications), key=classifications.count)

        returns_6m = [s["fwd_6m"] for s in t_signals if s["fwd_6m"] is not None]
        avg_6m = sum(returns_6m) / len(returns_6m) if returns_6m else None
        best = max(returns_6m) if returns_6m else None
        worst = min(returns_6m) if returns_6m else None

        avg_str = f"{avg_6m:+.1%}" if avg_6m is not None else "N/A"
        best_str = f"{best:+.1%}" if best is not None else "N/A"
        worst_str = f"{worst:+.1%}" if worst is not None else "N/A"

        print(f"  {ticker:<10} {len(t_signals):>8} {most_common:>14} {avg_str:>12} {best_str:>8} {worst_str:>8}")


# ── Excel export ──────────────────────────────────────────────────────────────

def export_to_excel(results: dict, filepath: str | None = None) -> str:
    """Export backtest results to an Excel workbook."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        print("  openpyxl not installed — skipping Excel export")
        return ""

    if filepath is None:
        filepath = str(_ROOT / f"backtest_results_{datetime.today().strftime('%Y%m%d')}.xlsx")

    wb = Workbook()

    # Sheet 1: Summary by tier
    ws = wb.active
    ws.title = "By Classification"
    headers = ["Tier", "Signals", "Avg 3M", "Avg 6M", "Avg 12M", "Hit Rate 6M", "Best 6M", "Worst 6M"]
    for col, h in enumerate(headers, 1):
        ws.cell(row=1, column=col, value=h).font = Font(bold=True)

    row = 2
    for tier in TIER_ORDER:
        if tier not in results["by_tier"]:
            continue
        t = results["by_tier"][tier]
        ws.cell(row=row, column=1, value=tier)
        ws.cell(row=row, column=2, value=t["count"])
        ws.cell(row=row, column=3, value=t.get("avg_3m"))
        ws.cell(row=row, column=4, value=t.get("avg_6m"))
        ws.cell(row=row, column=5, value=t.get("avg_12m"))
        ws.cell(row=row, column=6, value=t.get("hit_rate_6m"))
        ws.cell(row=row, column=7, value=t.get("best_6m"))
        ws.cell(row=row, column=8, value=t.get("worst_6m"))
        # Format percentages
        for col in range(3, 9):
            cell = ws.cell(row=row, column=col)
            if cell.value is not None:
                cell.number_format = '0.0%'
        row += 1

    # Sheet 2: All signals
    ws2 = wb.create_sheet("All Signals")
    headers2 = ["Ticker", "Date", "Classification", "Price", "Fwd 3M", "Fwd 6M", "Fwd 12M"]
    for col, h in enumerate(headers2, 1):
        ws2.cell(row=1, column=col, value=h).font = Font(bold=True)

    for row_idx, s in enumerate(results["signals"], 2):
        ws2.cell(row=row_idx, column=1, value=s["ticker"])
        ws2.cell(row=row_idx, column=2, value=s["date"].strftime("%Y-%m-%d"))
        ws2.cell(row=row_idx, column=3, value=s["classification"])
        ws2.cell(row=row_idx, column=4, value=s["price"])
        ws2.cell(row=row_idx, column=5, value=s.get("fwd_3m"))
        ws2.cell(row=row_idx, column=6, value=s.get("fwd_6m"))
        ws2.cell(row=row_idx, column=7, value=s.get("fwd_12m"))
        for col in (5, 6, 7):
            cell = ws2.cell(row=row_idx, column=col)
            if cell.value is not None:
                cell.number_format = '0.0%'

    wb.save(filepath)
    return filepath


# ── CLI entry point ───────────────────────────────────────────────────────────

DEFAULT_TICKERS = [
    "MAIN", "BTI", "SHEL", "O", "MNG.L", "AGNC", "BMO", "BNS",
    "BMW.DE", "CNQ", "MBG.DE", "PFE", "RIO", "STAG", "TD", "UPS",
]


def main():
    # Handle CLI arguments
    tickers = DEFAULT_TICKERS
    lookback = DEFAULT_LOOKBACK

    args = sys.argv[1:]
    ticker_args = []
    i = 0
    while i < len(args):
        if args[i] == "--lookback" and i + 1 < len(args):
            lookback = int(args[i + 1])
            i += 2
        elif args[i] == "--help":
            print(__doc__)
            return
        else:
            ticker_args.append(args[i])
            i += 1

    if ticker_args:
        tickers = ticker_args

    print(f"\nStrategy Backtest — {datetime.today().strftime('%Y-%m-%d')}")

    results = run_backtest(tickers, lookback_years=lookback)
    print_backtest_results(results)

    # Excel export
    try:
        path = export_to_excel(results)
        if path:
            print(f"\n  Excel report saved → {path}")
    except Exception as e:
        print(f"\n  Excel export failed: {e}")


if __name__ == "__main__":
    main()
