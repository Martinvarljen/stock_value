"""
financial_strength.py  —  Financial Strength Analysis

analyze_financials(data) → dict with every calculated metric, its value,
a plain-English assessment, and the benchmark used.

No made-up point system. Every output is a real number or observation.
"""

import numpy as np
from utils import _pct, _x, _bn, _assess, _count_positive, _cv


# ── main function ─────────────────────────────────────────────────────────────

def analyze_financials(data: dict) -> dict:
    """
    Returns a dict of metric blocks. Each block has:
        value       — the raw number
        formatted   — human-readable string
        assessment  — e.g. "Strong", "Adequate", "Concerning"
        benchmark   — the threshold used to make that call
        detail      — extra context
    """
    metrics = {}
    flags   = []
    critical_flags = []   # hard overrides for the classifier

    # ── Leverage ──────────────────────────────────────────────────────────
    de = data.get("debt_equity")
    if de is None:
        de_assess = "No data"
    elif de < 0:
        de_assess = "Negative equity — liabilities exceed assets"
        critical_flags.append("NEGATIVE_EQUITY")
    elif de <= 0.30:
        de_assess = "Low leverage (<0.3x) — conservative balance sheet"
    elif de <= 0.70:
        de_assess = "Moderate leverage (0.3–0.7x)"
    elif de <= 1.20:
        de_assess = "Elevated leverage (0.7–1.2x)"
    elif de <= 2.00:
        de_assess = "High leverage (1.2–2x)"
    else:
        de_assess = f"Very high leverage ({de:.1f}x)"
        flags.append(f"Debt/Equity of {de:.1f}x — highly leveraged balance sheet")

    metrics["debt_equity"] = {
        "label":      "Debt / Equity",
        "value":      de,
        "formatted":  _x(de),
        "assessment": de_assess,
        "benchmark":  "<0.5x = conservative, 0.5–1.5x = typical, >2x = risky",
        "detail":     "Total debt / shareholders' equity",
    }

    # ── Net Debt / EBITDA ─────────────────────────────────────────────────
    nd_ebitda = data.get("net_debt_ebitda")
    nd        = data.get("net_debt")
    ebitda    = data.get("ebitda")

    if nd is not None and ebitda is not None:
        detail_nd = f"Net debt: {_bn(nd)}, EBITDA: {_bn(ebitda)}"
    else:
        detail_nd = "Insufficient data to compute"

    if nd_ebitda is None:
        nd_assess = "No data"
    elif nd_ebitda <= 0:
        nd_assess = "Net cash (zero net debt) — debt-free or overcapitalized"
    elif nd_ebitda <= 1.0:
        nd_assess = "Very low leverage (<1x EBITDA) — pays off debt in <1 year"
    elif nd_ebitda <= 2.0:
        nd_assess = "Low-moderate (1–2x EBITDA)"
    elif nd_ebitda <= 3.0:
        nd_assess = "Moderate (2–3x EBITDA) — manageable for stable cash flows"
    elif nd_ebitda <= 5.0:
        nd_assess = f"Elevated ({nd_ebitda:.1f}x EBITDA) — watch refinancing risk"
        flags.append(f"Net Debt/EBITDA of {nd_ebitda:.1f}x — elevated leverage")
    else:
        nd_assess = f"Dangerous ({nd_ebitda:.1f}x EBITDA) — stress scenario risk"
        flags.append(f"Net Debt/EBITDA of {nd_ebitda:.1f}x — dangerously high leverage")
        critical_flags.append("NET_DEBT_EBITDA_OVER_5")

    metrics["net_debt_ebitda"] = {
        "label":      "Net Debt / EBITDA",
        "value":      nd_ebitda,
        "formatted":  _x(nd_ebitda) if nd_ebitda is not None else "N/A",
        "assessment": nd_assess,
        "benchmark":  "<1x = fortress, 1–3x = normal, >5x = danger zone",
        "detail":     detail_nd,
    }

    # ── Interest Coverage ─────────────────────────────────────────────────
    ic       = data.get("interest_coverage")
    ebit     = data.get("ebit")
    int_exp  = data.get("interest_expense")

    if int_exp is None or int_exp == 0:
        ic_assess = "No interest expense — effectively debt-free or no financial debt"
        ic_detail = "No interest payments detected"
    elif ic is None:
        ic_assess = "Cannot calculate (missing EBIT or interest data)"
        ic_detail = "N/A"
    elif ic < 0:
        ic_assess = "Negative — operating losses cannot cover any interest"
        flags.append("EBIT negative — cannot service interest from operations")
        critical_flags.append("NEGATIVE_INTEREST_COVERAGE")
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"
    elif ic < 1.5:
        ic_assess = f"Critically thin ({ic:.1f}x) — one bad quarter from distress"
        flags.append(f"Interest coverage only {ic:.1f}x — very thin margin")
        critical_flags.append("INTEREST_COVERAGE_BELOW_1.5")
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"
    elif ic < 3.0:
        ic_assess = f"Weak ({ic:.1f}x) — vulnerable to earnings downturn"
        flags.append(f"Interest coverage of {ic:.1f}x — below comfort zone")
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"
    elif ic < 6.0:
        ic_assess = f"Adequate ({ic:.1f}x)"
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"
    elif ic < 10.0:
        ic_assess = f"Strong ({ic:.1f}x)"
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"
    else:
        ic_assess = f"Very strong ({ic:.1f}x) — interest is not a concern"
        ic_detail = f"EBIT: {_bn(ebit)}, Interest expense: {_bn(int_exp)}"

    metrics["interest_coverage"] = {
        "label":      "Interest Coverage (EBIT / Interest)",
        "value":      ic,
        "formatted":  _x(ic) if ic is not None else "N/A",
        "assessment": ic_assess,
        "benchmark":  ">8x = strong, 3–8x = adequate, <3x = concerning, <1.5x = danger",
        "detail":     ic_detail,
    }

    # ── Liquidity ─────────────────────────────────────────────────────────
    cr = data.get("current_ratio")
    metrics["current_ratio"] = {
        "label":      "Current Ratio",
        "value":      cr,
        "formatted":  _x(cr) if cr is not None else "N/A",
        "assessment": (
            f"Strong ({cr:.2f}x) — ample short-term liquidity"    if cr is not None and cr >= 2.0 else
            f"Healthy ({cr:.2f}x)"                                  if cr is not None and cr >= 1.5 else
            f"Adequate ({cr:.2f}x)"                                 if cr is not None and cr >= 1.2 else
            f"Tight ({cr:.2f}x) — limited short-term cushion"      if cr is not None and cr >= 1.0 else
            f"Below 1.0 ({cr:.2f}x) — current liabilities exceed current assets"
            if cr is not None else "No data"
        ),
        "benchmark":  ">1.5x = healthy, <1.0x = short-term liquidity risk",
        "detail":     "Current assets / current liabilities",
    }
    if cr is not None and cr < 1.0:
        flags.append(f"Current ratio of {cr:.2f}x — current liabilities exceed current assets")
        if cr < 0.75:
            critical_flags.append("CURRENT_RATIO_CRITICALLY_LOW")

    # ── FCF Analysis ──────────────────────────────────────────────────────
    fcf_5y = data.get("fcf_5y") or [None]*5
    fcf_pos, fcf_total = _count_positive(fcf_5y)
    fcf_formatted = [_bn(v) for v in fcf_5y]
    ocf_5y  = data.get("ocf_5y")  or [None]*5
    capex_5y= data.get("capex_5y") or [None]*5

    metrics["fcf_consistency"] = {
        "label":      "FCF Consistency",
        "value":      (fcf_pos / fcf_total) if fcf_total else None,
        "formatted":  f"{fcf_pos}/{fcf_total} years positive FCF",
        "assessment": (
            "All years FCF positive — reliable cash generator"    if fcf_total and fcf_pos == fcf_total else
            f"{fcf_pos}/{fcf_total} years positive — mostly reliable"  if fcf_total and fcf_pos >= fcf_total * 0.8 else
            f"{fcf_pos}/{fcf_total} years positive — inconsistent"     if fcf_total and fcf_pos >= fcf_total * 0.6 else
            f"{fcf_pos}/{fcf_total} years positive — mostly negative FCF"
            if fcf_total else "No data"
        ),
        "benchmark":  "4+ of 5 years positive = reliable; <3 = cash drain",
        "detail":     f"FCF by year: {fcf_formatted}",
    }
    if fcf_total and fcf_pos < fcf_total * 0.4:
        critical_flags.append("FCF_NEGATIVE_3_PLUS_YEARS")
        flags.append(f"FCF negative in {fcf_total - fcf_pos} of {fcf_total} years")

    # FCF volatility
    fcf_cv = _cv(fcf_5y)
    if fcf_cv is not None:
        metrics["fcf_volatility"] = {
            "label":      "FCF Volatility (Coeff. of Variation)",
            "value":      fcf_cv,
            "formatted":  f"{fcf_cv:.2f}",
            "assessment": (
                "Very stable FCF (<30% variation)"  if fcf_cv < 0.30 else
                "Moderate variation (30–60%)"        if fcf_cv < 0.60 else
                "Highly variable FCF (>60%)"
            ),
            "benchmark":  "<30% CV = stable, >60% = unpredictable",
            "detail":     "Standard deviation / mean absolute FCF",
        }

    # ── Debt Structure ─────────────────────────────────────────────────────
    cash      = data.get("cash")
    total_debt= data.get("total_debt")
    net_debt  = data.get("net_debt")

    metrics["cash_position"] = {
        "label":      "Cash & Equivalents",
        "value":      cash,
        "formatted":  _bn(cash),
        "assessment": "No data" if cash is None else (
            f"{_bn(cash)} on hand"
        ),
        "benchmark":  "Context-dependent — compare to debt and annual FCF",
        "detail":     f"Net debt: {_bn(net_debt)}  |  Total debt: {_bn(total_debt)}",
    }

    if cash is not None and total_debt is not None and total_debt > 0:
        cash_ratio = cash / total_debt
        metrics["cash_vs_debt"] = {
            "label":      "Cash as % of Total Debt",
            "value":      cash_ratio,
            "formatted":  _pct(cash_ratio),
            "assessment": (
                "Cash exceeds total debt — net cash balance sheet"  if cash_ratio >= 1.0 else
                "Strong buffer (50–100% of debt covered by cash)"   if cash_ratio >= 0.50 else
                "Moderate buffer (25–50% cash coverage)"             if cash_ratio >= 0.25 else
                "Thin buffer (<25% cash coverage)"                   if cash_ratio >= 0.10 else
                "Very low cash relative to debt — refinancing risk"
            ),
            "benchmark":  ">100% = net cash, 25–50% = adequate buffer",
            "detail":     f"Cash: {_bn(cash)}, Total debt: {_bn(total_debt)}",
        }

    # ── Earnings Volatility ───────────────────────────────────────────────
    ni_5y = data.get("net_income_5y") or [None]*5
    ni_cv = _cv(ni_5y)
    if ni_cv is not None:
        metrics["earnings_volatility"] = {
            "label":      "Earnings Volatility (Coeff. of Variation)",
            "value":      ni_cv,
            "formatted":  f"{ni_cv:.2f}",
            "assessment": (
                "Very stable earnings"   if ni_cv < 0.20 else
                "Moderate variation"      if ni_cv < 0.50 else
                "Highly volatile earnings — cyclical or operationally unstable"
            ),
            "benchmark":  "<20% CV = stable, >50% = high cyclicality / volatility",
            "detail":     f"Net income by year: {[_bn(v) for v in ni_5y]}",
        }

    # ── Revenue volatility (cyclicality proxy) ────────────────────────────
    rev_5y = data.get("revenue_5y") or [None]*5
    rev_cv = _cv(rev_5y)
    if rev_cv is not None:
        metrics["revenue_volatility"] = {
            "label":      "Revenue Volatility (Coeff. of Variation)",
            "value":      rev_cv,
            "formatted":  f"{rev_cv:.2f}",
            "assessment": (
                "Highly stable revenue — defensive business"   if rev_cv < 0.10 else
                "Low volatility (10–20%)"                       if rev_cv < 0.20 else
                "Moderate cyclicality (20–40%)"                 if rev_cv < 0.40 else
                "High cyclicality (>40%) — revenue swings significantly"
            ),
            "benchmark":  "<10% = defensive/staple, >40% = cyclical",
            "detail":     f"Revenue by year: {[_bn(v) for v in rev_5y]}",
        }

    # ── Dividend Track Record ─────────────────────────────────────────────
    div_5y = data.get("dividends_5y") or [None]*5
    div_yield = data.get("dividend_yield")
    div_paid  = [abs(v) for v in div_5y if v is not None and v != 0]
    if div_yield or div_paid:
        metrics["dividend"] = {
            "label":      "Dividend Yield",
            "value":      div_yield,
            "formatted":  _pct(div_yield) if div_yield else "0%",
            "assessment": (
                f"Pays dividend ({_pct(div_yield)} yield); "
                f"paid in {len(div_paid)} of 5 years"
            ) if div_yield else "No dividend or data unavailable",
            "benchmark":  "Consistency matters more than absolute yield",
            "detail":     f"Dividends paid by year: {[_bn(v) for v in div_5y]}",
        }

    # ── Balance sheet trend check ─────────────────────────────────────────
    equity_5y = data.get("equity_5y") or [None]*5
    eq_valid = [(i, v) for i, v in enumerate(equity_5y) if v is not None]
    if len(eq_valid) >= 2:
        eq_trend = eq_valid[-1][1] - eq_valid[0][1]
        metrics["equity_trend"] = {
            "label":      "Equity Trend (5Y change)",
            "value":      eq_trend,
            "formatted":  _bn(eq_trend),
            "assessment": (
                f"Equity growing (+{_bn(eq_trend)}) — retained earnings accumulating"  if eq_trend > 0 else
                f"Equity shrinking ({_bn(eq_trend)}) — check buybacks or losses"
            ),
            "benchmark":  "Growing equity over time = capital accumulation",
            "detail":     f"Equity by year: {[_bn(v) for v in equity_5y]}",
        }

    # ── Effective tax rate ────────────────────────────────────────────────
    eff_tax = data.get("effective_tax_rate")
    metrics["effective_tax_rate"] = {
        "label":      "Effective Tax Rate",
        "value":      eff_tax,
        "formatted":  _pct(eff_tax),
        "assessment": (
            "Low (<15%) — check for tax havens or one-time items"  if eff_tax is not None and eff_tax < 0.15 else
            "Normal range (15–30%)"                                 if eff_tax is not None and eff_tax <= 0.30 else
            "High (>30%) — may drag on net income"                  if eff_tax is not None else
            "Using default 22%"
        ),
        "benchmark":  "15–30% = typical corporate rate",
        "detail":     "Tax provision / pretax income (using 22% default if unavailable)",
    }

    return {
        "financial_metrics": metrics,
        "financial_flags":   flags,
        "critical_flags":    critical_flags,
    }


# ── display helper ────────────────────────────────────────────────────────────

def print_financials(result: dict, ticker: str = ""):
    header = f"FINANCIAL STRENGTH — {ticker}" if ticker else "FINANCIAL STRENGTH"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    for key, m in result["financial_metrics"].items():
        print(f"  {m['label']:<42} {m['formatted']:<12}  {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")
    if result["financial_flags"]:
        print(f"\n  ⚠  Flags:")
        for f in result["financial_flags"]:
            print(f"     • {f}")
    if result["critical_flags"]:
        print(f"\n  🚨  Critical:")
        for f in result["critical_flags"]:
            print(f"     • {f}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    healthy = {
        "debt_equity": 0.25,
        "net_debt_ebitda": 0.8,
        "net_debt": 2e9, "ebitda": 2.5e9,
        "interest_coverage": 15.0,
        "ebit": 2e9, "interest_expense": 133e6,
        "current_ratio": 1.8,
        "fcf_5y": [1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9],
        "ocf_5y": [1.2e9]*5, "capex_5y": [-200e6]*5,
        "net_income_5y": [0.9e9, 1.0e9, 1.1e9, 1.2e9, 1.3e9],
        "revenue_5y": [4e9, 4.4e9, 4.8e9, 5.2e9, 5.6e9],
        "equity_5y": [5e9, 5.5e9, 6e9, 6.5e9, 7e9],
        "cash": 3e9, "total_debt": 5e9,
        "dividends_5y": [-300e6]*5,
        "dividend_yield": 0.018,
        "effective_tax_rate": 0.22,
    }
    r = analyze_financials(healthy)
    print_financials(r, "HEALTHY.CO")
