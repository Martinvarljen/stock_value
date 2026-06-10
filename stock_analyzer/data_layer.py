"""
data_layer.py  —  Phase 1: yfinance data collection & cleaning

collect_data(ticker) → dict with all raw & derived fields needed by scoring engines.
All missing values are stored as None; callers must handle None gracefully.
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)


# ── EUR exchange rate cache ───────────────────────────────────────────────────
# Fetched once per session, keyed by source currency code.
_YF_CACHE_DIR = None
for candidate in filter(None, [
    Path(os.environ["LOCALAPPDATA"]) / "StockAnalyzer" / "yfinance-cache"
    if os.environ.get("LOCALAPPDATA") else None,
    Path(__file__).resolve().parents[1] / ".cache" / "yfinance",
]):
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(candidate))
        _YF_CACHE_DIR = candidate
        break
    except Exception:
        continue

_EUR_RATE_CACHE: dict[str, float | None] = {"EUR": 1.0}

def get_eur_rate(currency: str) -> float | None:
    """
    Return the live rate: 1 unit of *currency* → EUR.
    Uses yfinance forex ticker  e.g. 'USDEUR=X', 'GBPEUR=X'.
    Returns None if the rate cannot be fetched.
    Caches results so each currency is only fetched once per run.
    """
    if currency is None:
        return None
    if currency in _EUR_RATE_CACHE:
        return _EUR_RATE_CACHE[currency]
    try:
        pair = f"{currency}EUR=X"
        tick = yf.Ticker(pair)
        hist = tick.history(period="1d")
        if not hist.empty:
            rate = float(hist["Close"].iloc[-1])
            _EUR_RATE_CACHE[currency] = rate
            return rate
    except (OSError, ValueError, KeyError) as exc:
        _log.debug("EUR rate fetch failed for %s: %s", currency, exc)
    _EUR_RATE_CACHE[currency] = None
    return None


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(value, default=None):
    """Return value unless it is NaN / inf / None, then return default."""
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return value
    except Exception:
        return default


def _series_to_list(series, n=5):
    """
    Convert a pandas Series (index = dates, values = numbers) to a list
    of the last *n* annual values, oldest-first.  Returns None-padded list.
    """
    if series is None or series.empty:
        return [None] * n
    series = series.dropna().sort_index()
    values = [_safe(v) for v in series.values]
    # keep last n
    if len(values) >= n:
        return values[-n:]
    # pad with None on the left if fewer than n years
    return [None] * (n - len(values)) + values


def _cagr(start, end, years):
    """Compound annual growth rate; returns None if inputs invalid."""
    try:
        if start is None or end is None or years <= 0:
            return None
        if start <= 0 or end <= 0:
            return None
        ratio = end / start
        if ratio <= 0:
            return None
        return ratio ** (1 / years) - 1
    except Exception:
        return None


def _pct_change_list(lst):
    """Year-over-year % changes for a list; None entries propagate as None."""
    changes = []
    for i in range(1, len(lst)):
        a, b = lst[i - 1], lst[i]
        if a is None or b is None or a == 0:
            changes.append(None)
        else:
            changes.append((b - a) / abs(a))
    return changes


def _rsi(prices, period=14):
    """Classic Wilder RSI on a list / Series of closing prices."""
    if prices is None or len(prices) < period + 1:
        return None
    s = pd.Series(prices)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return _safe(val)


# ── main collector ────────────────────────────────────────────────────────────

def collect_data(ticker: str) -> dict:
    """
    Pull all required data from yfinance for *ticker*.
    Returns a flat dict.  Never raises — missing fields become None.
    """
    result = {
        "ticker": ticker,
        "error": None,
        "data_quality_score": 0,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # ── identity & price ──────────────────────────────────────────────
        result["company_name"]      = _safe(info.get("longName") or info.get("shortName"))
        result["quote_type"]        = _safe(info.get("quoteType"))
        result["sector"]            = _safe(info.get("sector"))
        result["industry"]          = _safe(info.get("industry"))
        result["exchange"]          = _safe(info.get("exchange"))
        currency = _safe(info.get("currency"))
        result["currency"] = currency

        raw_price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))
        # London exchange (e.g. SHEL.L) quotes in GBp (pence) — convert to GBP
        if currency == "GBp" and raw_price is not None:
            raw_price = raw_price / 100
            result["currency"] = "GBP"
        result["current_price"] = raw_price

        # EUR conversion using live rate
        native_currency = result["currency"]
        eur_rate = get_eur_rate(native_currency)
        result["eur_rate"]      = eur_rate
        result["price_eur"]     = round(raw_price * eur_rate, 4) if (raw_price is not None and eur_rate is not None) else None

        result["market_cap"]        = _safe(info.get("marketCap"))
        result["shares_outstanding"]= _safe(info.get("sharesOutstanding"))
        result["beta"]              = _safe(info.get("beta"))
        result["week52_high"]       = _safe(info.get("fiftyTwoWeekHigh"))
        result["week52_low"]        = _safe(info.get("fiftyTwoWeekLow"))
        result["forward_pe"]        = _safe(info.get("forwardPE"))
        result["trailing_pe"]       = _safe(info.get("trailingPE"))
        result["pb_ratio"]          = _safe(info.get("priceToBook"))
        result["dividend_yield"]    = _safe(info.get("dividendYield"))
        result["enterprise_value"]  = _safe(info.get("enterpriseValue"))
        result["ev_ebitda"]         = _safe(info.get("enterpriseToEbitda"))
        result["ev_revenue"]        = _safe(info.get("enterpriseToRevenue"))
        result["peg_ratio"]         = _safe(info.get("pegRatio"))

        # ── annual financials ─────────────────────────────────────────────
        try:
            inc = tk.financials          # columns = dates (newest left)
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

        def _row(df, *keys):
            """Try multiple key names; return Series or empty Series."""
            if df is None or df.empty:
                return pd.Series(dtype=float)
            for k in keys:
                if k in df.index:
                    return df.loc[k]
            return pd.Series(dtype=float)

        # Income statement rows
        rev_s   = _row(inc_a, "Total Revenue", "Revenue")
        gp_s    = _row(inc_a, "Gross Profit")
        ebit_s  = _row(inc_a, "EBIT", "Operating Income")
        ni_s    = _row(inc_a, "Net Income")
        eps_s   = _row(inc_a, "Basic EPS", "Diluted EPS")
        int_s   = _row(inc_a, "Interest Expense")
        dep_s   = _row(inc_a, "Reconciled Depreciation", "Depreciation And Amortization")
        tax_s   = _row(inc_a, "Tax Provision", "Income Tax Expense")
        pbt_s   = _row(inc_a, "Pretax Income")

        # Balance sheet rows
        assets_s  = _row(bal_a, "Total Assets")
        liab_s    = _row(bal_a, "Total Liabilities Net Minority Interest", "Total Liabilities")
        ltd_s     = _row(bal_a, "Long Term Debt")
        std_s     = _row(bal_a, "Current Debt", "Short Term Debt", "Current Portion Of Long Term Debt And Capital Lease Obligation")
        cash_s    = _row(bal_a, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
        equity_s  = _row(bal_a, "Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")
        shares_s  = _row(bal_a, "Share Issued", "Ordinary Shares Number")
        cur_assets= _row(bal_a, "Current Assets")
        cur_liab  = _row(bal_a, "Current Liabilities")
        bv_s      = _row(bal_a, "Book Value", "Tangible Book Value")

        # Cash flow rows
        ocf_s   = _row(cf_a, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        capex_s = _row(cf_a, "Capital Expenditure")
        div_s   = _row(cf_a, "Common Stock Dividend Paid", "Cash Dividends Paid")
        sbc_s   = _row(cf_a, "Stock Based Compensation")

        # ── store 5-year annual lists (oldest → newest) ───────────────────
        result["revenue_5y"]       = _series_to_list(rev_s)
        result["gross_profit_5y"]  = _series_to_list(gp_s)
        result["ebit_5y"]          = _series_to_list(ebit_s)
        result["net_income_5y"]    = _series_to_list(ni_s)
        result["eps_5y"]           = _series_to_list(eps_s)
        result["interest_expense_5y"] = _series_to_list(int_s)
        result["depreciation_5y"]  = _series_to_list(dep_s)
        result["tax_provision_5y"] = _series_to_list(tax_s)
        result["pretax_income_5y"] = _series_to_list(pbt_s)

        result["total_assets_5y"]  = _series_to_list(assets_s)
        result["total_liabilities_5y"] = _series_to_list(liab_s)
        result["total_debt_5y"]    = _series_to_list(
            (ltd_s.fillna(0) + std_s.fillna(0)) if (not ltd_s.empty or not std_s.empty) else pd.Series(dtype=float)
        )
        result["cash_5y"]          = _series_to_list(cash_s)
        result["equity_5y"]        = _series_to_list(equity_s)
        result["shares_5y"]        = _series_to_list(shares_s)
        result["current_assets_5y"]= _series_to_list(cur_assets)
        result["current_liab_5y"]  = _series_to_list(cur_liab)

        result["ocf_5y"]           = _series_to_list(ocf_s)
        result["capex_5y"]         = _series_to_list(capex_s)
        result["dividends_5y"]     = _series_to_list(div_s)
        result["sbc_5y"]           = _series_to_list(sbc_s)

        # FCF = OCF - |capex|  (capex is usually negative in yfinance)
        fcf_list = []
        for o, c in zip(result["ocf_5y"], result["capex_5y"]):
            if o is None:
                fcf_list.append(None)
            elif c is None:
                fcf_list.append(o)
            else:
                fcf_list.append(o - abs(c))
        result["fcf_5y"] = fcf_list

        # ── latest-year snapshots ─────────────────────────────────────────
        def _last(lst):
            for v in reversed(lst):
                if v is not None:
                    return v
            return None

        revenue      = _last(result["revenue_5y"])
        gross_profit = _last(result["gross_profit_5y"])
        ebit         = _last(result["ebit_5y"])
        net_income   = _last(result["net_income_5y"])
        ocf          = _last(result["ocf_5y"])
        capex        = _last(result["capex_5y"])
        fcf          = _last(result["fcf_5y"])
        total_debt   = _last(result["total_debt_5y"])
        cash         = _last(result["cash_5y"])
        equity       = _last(result["equity_5y"])
        interest_exp = _last(result["interest_expense_5y"])
        dep          = _last(result["depreciation_5y"])
        cur_a        = _last(result["current_assets_5y"])
        cur_l        = _last(result["current_liab_5y"])
        tax_prov     = _last(result["tax_provision_5y"])
        pretax       = _last(result["pretax_income_5y"])

        result.update({
            "revenue": revenue,
            "gross_profit": gross_profit,
            "ebit": ebit,
            "net_income": net_income,
            "ocf": ocf,
            "capex": capex,
            "fcf": fcf,
            "total_debt": total_debt,
            "cash": cash,
            "equity": equity,
            "interest_expense": interest_exp,
            "depreciation": dep,
        })

        # ── derived ratios ────────────────────────────────────────────────
        price = result["current_price"]
        mktcap = result["market_cap"]
        ev = result["enterprise_value"]

        result["gross_margin"]     = _safe(gross_profit / revenue) if (revenue not in (None, 0) and gross_profit is not None) else None
        result["operating_margin"] = _safe(ebit / revenue) if (revenue not in (None, 0) and ebit is not None) else None
        result["net_margin"]       = _safe(net_income / revenue) if (revenue not in (None, 0) and net_income is not None) else None

        # EBITDA
        ebitda = None
        if ebit is not None and dep is not None:
            ebitda = ebit + dep
        elif ebit is not None:
            ebitda = ebit
        result["ebitda"] = ebitda

        # ROE, ROA
        result["roe"]  = _safe(net_income / equity) if (net_income is not None and equity not in (None, 0)) else None
        assets_last = _last(result["total_assets_5y"])
        result["roa"]  = _safe(net_income / assets_last) if (net_income is not None and assets_last not in (None, 0)) else None

        # ROIC  =  EBIT*(1-t) / (equity + net_debt)
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

        # Debt ratios
        result["debt_equity"]    = _safe(total_debt / equity) if (total_debt is not None and equity not in (None, 0)) else None
        result["net_debt_ebitda"]= _safe(net_debt / ebitda) if (net_debt is not None and ebitda not in (None, 0)) else None
        result["interest_coverage"] = _safe(abs(ebit) / abs(interest_exp)) if (ebit is not None and interest_exp not in (None, 0)) else None
        result["current_ratio"]  = _safe(cur_a / cur_l) if (cur_a is not None and cur_l not in (None, 0)) else None

        # FCF yield
        result["fcf_yield"] = _safe(fcf / mktcap) if (fcf is not None and mktcap not in (None, 0)) else None

        # EV multiples
        result["ev_ebit"]  = _safe(ev / ebit) if (ev is not None and ebit not in (None, 0)) else None
        result["ev_fcf"]   = _safe(ev / fcf)  if (ev is not None and fcf not in (None, 0)) else None

        # ── growth CAGRs (single pass per list via indexed valid values) ──
        def _list_cagr(lst, require_positive_start=False):
            indexed = [(i, v) for i, v in enumerate(lst) if v is not None]
            if len(indexed) < 2:
                return None
            _, v_start = indexed[0]
            _, v_end   = indexed[-1]
            years = indexed[-1][0] - indexed[0][0]
            if years <= 0:
                return None
            if require_positive_start and v_start <= 0:
                return None
            if v_end <= 0:
                return None
            return _cagr(v_start, v_end, years)

        result["revenue_cagr_5y"] = _list_cagr(result["revenue_5y"])
        result["eps_cagr_5y"]     = _list_cagr(result["eps_5y"], require_positive_start=True)
        result["fcf_cagr_5y"]     = _list_cagr(result["fcf_5y"], require_positive_start=True)

        result["revenue_yoy_changes"] = _pct_change_list(result["revenue_5y"])

        # Shares dilution (now vs 5y ago)
        sh_now  = result["shares_outstanding"] or _last(result["shares_5y"])
        sh_old  = next((v for v in result["shares_5y"] if v is not None), None)
        if sh_now and sh_old and sh_old != 0:
            result["shares_change_pct"] = (sh_now - sh_old) / sh_old
        else:
            result["shares_change_pct"] = None

        # Gross capex / revenue (trailing average)
        capex_ratios = []
        for r, c in zip(result["revenue_5y"], result["capex_5y"]):
            if r and c and r != 0:
                capex_ratios.append(abs(c) / r)
        result["capex_pct_revenue"] = float(np.mean(capex_ratios)) if capex_ratios else None

        # Net capex / revenue = (capex - D&A) / revenue — the real reinvestment burden.
        # D&A is already absorbed in operating margins, so only the excess over D&A
        # represents true cash outflow beyond what the income statement reflects.
        net_capex_ratios = []
        dep_5y = result.get("depreciation_5y") or []
        for r, c, d in zip(result["revenue_5y"], result["capex_5y"], dep_5y):
            if r and c and r != 0:
                gross_cx = abs(c)
                da       = abs(d) if d else 0.0
                net_cx   = max(gross_cx - da, 0.0)   # floor at 0: D&A can't "fund" capex
                net_capex_ratios.append(net_cx / r)
        result["net_capex_pct_revenue"] = float(np.mean(net_capex_ratios)) if net_capex_ratios else None

        # ── price history (1Y daily, 5Y monthly) ──────────────────────────
        end_dt   = datetime.today()
        start_1y = end_dt - timedelta(days=400)
        start_5y = end_dt - timedelta(days=365 * 5 + 30)

        hist_1y_raw = None
        try:
            hist_1y_raw = tk.history(start=start_1y.strftime("%Y-%m-%d"), interval="1d")
            if hist_1y_raw is not None and not hist_1y_raw.empty:
                hist_1y_raw = hist_1y_raw.sort_index()
                close_1y = hist_1y_raw["Close"].dropna()
            else:
                close_1y = pd.Series(dtype=float)
                hist_1y_raw = None
        except Exception:
            hist_1y_raw = None
            close_1y = pd.Series(dtype=float)

        try:
            hist_5y = tk.history(start=start_5y.strftime("%Y-%m-%d"), interval="1mo")
            close_5y = hist_5y["Close"].dropna() if not hist_5y.empty else pd.Series(dtype=float)
        except Exception:
            close_5y = pd.Series(dtype=float)

        result["close_1y"] = close_1y.tolist() if not close_1y.empty else []
        # OHLCV aligned to close_1y (for ML extended technical features)
        if hist_1y_raw is not None and not close_1y.empty:
            idx = close_1y.index
            h = hist_1y_raw["High"].reindex(idx).astype(float)
            l = hist_1y_raw["Low"].reindex(idx).astype(float)
            v = hist_1y_raw["Volume"].reindex(idx).astype(float)
            result["high_1y"] = h.fillna(close_1y).tolist()
            result["low_1y"] = l.fillna(close_1y).tolist()
            result["volume_1y"] = v.fillna(0.0).tolist()
            o = hist_1y_raw["Open"].reindex(idx).astype(float)
            result["open_1y"] = o.fillna(close_1y).tolist()
        else:
            result["high_1y"] = []
            result["low_1y"] = []
            result["volume_1y"] = []
            result["open_1y"] = []

        result["close_5y_monthly"] = close_5y.tolist() if not close_5y.empty else []

        # Moving averages & RSI
        if len(close_1y) >= 50:
            result["ma50"]  = float(close_1y.iloc[-50:].mean())
        else:
            result["ma50"]  = None

        if len(close_1y) >= 200:
            result["ma200"] = float(close_1y.iloc[-200:].mean())
        else:
            result["ma200"] = None

        result["rsi14"] = _rsi(close_1y.tolist())

        # ~1 trading year return (≈252 sessions): last close vs close ~252 bars earlier
        try:
            p_now = price
            n = len(close_1y)
            if p_now is not None and n >= 2:
                last_i = n - 1
                bars = min(252, last_i)
                ref_i = last_i - bars
                p_1y_ago = float(close_1y.iloc[ref_i])
                result["return_1y"] = (
                    _safe((p_now - p_1y_ago) / p_1y_ago) if p_1y_ago not in (None, 0) else None
                )
            else:
                result["return_1y"] = None
        except Exception:
            result["return_1y"] = None

        # 3Y return — derived from already-fetched 5Y monthly series (no extra API call)
        try:
            # 5Y monthly has ~60 bars; 3 years back ≈ bar index 36 from the end
            if len(close_5y) >= 36 and price is not None:
                p_3y_ago = float(close_5y.iloc[-36])
                result["return_3y"] = _safe((price - p_3y_ago) / p_3y_ago) if p_3y_ago != 0 else None
            else:
                result["return_3y"] = None
        except Exception:
            result["return_3y"] = None

        # ── historical P/E range (5Y) ─────────────────────────────────────
        # approximate: use monthly close / trailing EPS
        trailing_eps = _safe(info.get("trailingEps"))
        result["trailing_eps"] = trailing_eps
        if trailing_eps and trailing_eps > 0 and len(close_5y) > 0:
            pe_hist = [p / trailing_eps for p in close_5y.tolist() if p > 0]
            result["pe_5y_min"]    = _safe(np.percentile(pe_hist, 5))
            result["pe_5y_max"]    = _safe(np.percentile(pe_hist, 95))
            result["pe_5y_median"] = _safe(np.median(pe_hist))
        else:
            result["pe_5y_min"]    = None
            result["pe_5y_max"]    = None
            result["pe_5y_median"] = None

        # ── data quality score ─────────────────────────────────────────────
        critical_fields = [
            "current_price", "revenue", "ebit", "net_income", "fcf",
            "equity", "total_debt", "ocf", "revenue_cagr_5y",
            "operating_margin", "gross_margin", "roic",
        ]
        present = sum(1 for f in critical_fields if result.get(f) is not None)
        result["data_quality_score"] = round(100 * present / len(critical_fields))

    except Exception as exc:
        result["error"] = str(exc)
        result["data_quality_score"] = 0

    return result


# ── quick CLI test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    test_ticker = "ASML.AS"
    print(f"Fetching data for {test_ticker} ...")
    data = collect_data(test_ticker)

    # Print a readable summary
    summary_keys = [
        "ticker", "company_name", "sector", "currency", "current_price",
        "market_cap", "revenue", "operating_margin", "net_margin",
        "roic", "debt_equity", "net_debt_ebitda",
        "revenue_cagr_5y", "fcf", "data_quality_score", "error",
    ]
    print("\n── Data Summary ──")
    for k in summary_keys:
        v = data.get(k)
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"\n  revenue_5y : {data['revenue_5y']}")
    print(f"  fcf_5y     : {data['fcf_5y']}")
