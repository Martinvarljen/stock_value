"""
red_flags.py  —  Pattern-based red flag detection

analyze_red_flags(data, wacc=None) → dict with:
  - red_flags:    list of {pattern, severity, detail}
  - summary:      plain-English verdict
  - count_high/medium/low

Severity levels: HIGH, MEDIUM, LOW

Patterns checked:
  1.  CAPEX_TRAP              Rising capex/revenue with low/no growth
  2.  MARGIN_COMPRESSION      Operating margin declining over the period
  3.  HEAVY_DILUTION          Share count growing > 3%/yr
  4.  WEAK_FCF_CONVERSION     FCF / Net Income persistently below 60%
  5.  SBC_BURDEN              Stock-based comp > 3% of revenue
  6.  REVENUE_FCF_DIVERGENCE  Revenue growing but FCF flat or declining
  7.  CAPITAL_DESTRUCTION     ROIC below WACC (requires wacc argument)
  8.  COVERAGE_DETERIORATION  Interest coverage < 3x or EBIT trending down
  9.  UNSUSTAINABLE_DIVIDEND  Dividends exceed or nearly exceed free cash flow
  10. LEVERAGE_EXPANSION      Net Debt / EBITDA > 3.5x or rising vs declining EBIT
"""

from utils import _bn


# ── helpers ────────────────────────────────────────────────────────────────────

def _valid(lst):
    return [v for v in (lst or []) if v is not None]


def _simple_cagr(start, end, years):
    if years <= 0 or start is None or end is None or start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _trend_slope(lst):
    """Linear regression slope — positive = improving, negative = deteriorating."""
    vals = _valid(lst)
    if len(vals) < 3:
        return None
    n = len(vals)
    x_mean = (n - 1) / 2
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _flag(pattern, severity, detail):
    return {"pattern": pattern, "severity": severity, "detail": detail}


def _half_averages(series):
    """Split a list into first/second half and return their averages."""
    n = len(series)
    half = max(1, n // 2)
    early = sum(series[:half]) / half
    late  = sum(series[half:]) / max(1, n - half)
    return early, late


# ── individual detectors ───────────────────────────────────────────────────────

def _check_capex_trap(data):
    capex_5y = _valid(data.get("capex_5y") or [])
    rev_5y   = _valid(data.get("revenue_5y") or [])
    rev_cagr = data.get("revenue_cagr_5y")

    if len(capex_5y) < 3 or len(rev_5y) < 3:
        return None

    pairs = [(abs(c), r) for c, r in zip(capex_5y, rev_5y) if r and r > 0]
    if len(pairs) < 3:
        return None

    cpx_pct = [c / r for c, r in pairs]
    early, late = _half_averages(cpx_pct)

    rising     = late > early * 1.15
    low_growth = rev_cagr is not None and rev_cagr < 0.04

    if rising and low_growth:
        return _flag(
            "CAPEX_TRAP", "HIGH",
            f"Capex/revenue rose ~{early:.1%} → ~{late:.1%} while revenue CAGR is only {rev_cagr:.1%} — "
            f"capital intensity increasing with diminishing growth returns"
        )
    if rising and rev_cagr is not None and rev_cagr < 0.08:
        return _flag(
            "CAPEX_TRAP", "MEDIUM",
            f"Capex/revenue rising (~{early:.1%} → ~{late:.1%}) with modest revenue growth ({rev_cagr:.1%}) — "
            f"watch for declining capital efficiency"
        )
    return None


def _check_margin_compression(data):
    ebit_5y = _valid(data.get("ebit_5y") or [])
    rev_5y  = _valid(data.get("revenue_5y") or [])

    if len(ebit_5y) < 3 or len(rev_5y) < 3:
        return None

    margins = [e / r for e, r in zip(ebit_5y, rev_5y) if r and r > 0]
    if len(margins) < 3:
        return None

    early, late = _half_averages(margins)
    delta = late - early

    if delta < -0.05:
        return _flag(
            "MARGIN_COMPRESSION", "HIGH",
            f"Operating margin fell ~{early:.1%} → ~{late:.1%} ({delta:+.1%}) — "
            f"sustained structural pressure on profitability"
        )
    if delta < -0.02:
        return _flag(
            "MARGIN_COMPRESSION", "MEDIUM",
            f"Operating margin drifting lower (~{early:.1%} → ~{late:.1%}, {delta:+.1%}) — "
            f"monitor for continued compression"
        )
    return None


def _check_heavy_dilution(data):
    shares_5y = _valid(data.get("shares_5y") or [])
    if len(shares_5y) < 2:
        return None

    years = len(shares_5y) - 1
    cagr  = _simple_cagr(shares_5y[0], shares_5y[-1], years)
    if cagr is None:
        return None

    if cagr > 0.05:
        return _flag(
            "HEAVY_DILUTION", "HIGH",
            f"Share count growing {cagr:.1%}/yr — heavy dilution steadily eroding per-share value"
        )
    if cagr > 0.03:
        return _flag(
            "HEAVY_DILUTION", "MEDIUM",
            f"Share count growing {cagr:.1%}/yr — meaningful dilution; check SBC and equity issuance"
        )
    if cagr > 0.015:
        return _flag(
            "HEAVY_DILUTION", "LOW",
            f"Mild share count growth ({cagr:.1%}/yr) — minor dilution, worth monitoring"
        )
    return None


def _check_fcf_conversion(data):
    fcf_5y = _valid(data.get("fcf_5y") or [])
    ni_5y  = _valid(data.get("net_income_5y") or [])

    if len(fcf_5y) < 3 or len(ni_5y) < 3:
        return None

    ratios = [f / n for f, n in zip(fcf_5y, ni_5y) if n and n > 0 and f is not None]
    if len(ratios) < 2:
        return None

    avg       = sum(ratios) / len(ratios)
    low_count = sum(1 for r in ratios if r < 0.60)

    if avg < 0.50 or low_count >= len(ratios) - 1:
        return _flag(
            "WEAK_FCF_CONVERSION", "HIGH",
            f"Average FCF/Net Income = {avg:.0%} — earnings are not converting to cash. "
            f"Check for accrual accounting, high capex, or working capital deterioration"
        )
    if avg < 0.70:
        return _flag(
            "WEAK_FCF_CONVERSION", "MEDIUM",
            f"FCF conversion averaging {avg:.0%} — below the 70–90% benchmark for healthy cash-generative businesses"
        )
    return None


def _check_sbc_burden(data):
    sbc_5y = _valid(data.get("sbc_5y") or [])
    rev_5y = _valid(data.get("revenue_5y") or [])

    if len(sbc_5y) < 2 or len(rev_5y) < 2:
        return None

    pcts = [abs(s) / r for s, r in zip(sbc_5y, rev_5y) if r and r > 0 and s is not None]
    if not pcts:
        return None

    latest = pcts[-1]

    if latest > 0.05:
        return _flag(
            "SBC_BURDEN", "HIGH",
            f"SBC = {latest:.1%} of revenue — very high. Real economic earnings are materially "
            f"below GAAP; shareholders are subsidising employee compensation"
        )
    if latest > 0.03:
        return _flag(
            "SBC_BURDEN", "MEDIUM",
            f"SBC = {latest:.1%} of revenue — elevated; reduces economic earnings below reported figures"
        )
    return None


def _check_revenue_fcf_divergence(data):
    rev_cagr = data.get("revenue_cagr_5y")
    fcf_cagr = data.get("fcf_cagr_5y")

    if rev_cagr is None or fcf_cagr is None:
        return None

    gap = rev_cagr - fcf_cagr

    if rev_cagr > 0.05 and fcf_cagr < 0:
        return _flag(
            "REVENUE_FCF_DIVERGENCE", "HIGH",
            f"Revenue growing {rev_cagr:.1%}/yr but FCF declining ({fcf_cagr:.1%}/yr) — "
            f"growth is consuming cash rather than generating it"
        )
    if rev_cagr > 0.04 and gap > 0.06:
        return _flag(
            "REVENUE_FCF_DIVERGENCE", "MEDIUM",
            f"Revenue CAGR ({rev_cagr:.1%}) outpacing FCF CAGR ({fcf_cagr:.1%}) by {gap:.1%} — "
            f"growth efficiency declining"
        )
    return None


def _check_capital_destruction(data, wacc):
    if wacc is None:
        return None
    roic = data.get("roic")
    if roic is None:
        return None

    spread = roic - wacc
    if spread < -0.05:
        return _flag(
            "CAPITAL_DESTRUCTION", "HIGH",
            f"ROIC ({roic:.1%}) is {abs(spread):.1%} below WACC ({wacc:.1%}) — "
            f"every unit of capital deployed is destroying shareholder value"
        )
    if spread < 0:
        return _flag(
            "CAPITAL_DESTRUCTION", "MEDIUM",
            f"ROIC ({roic:.1%}) slightly below WACC ({wacc:.1%}) — marginal value destruction; "
            f"management must improve returns or reduce reinvestment pace"
        )
    return None


def _check_interest_coverage(data):
    coverage = data.get("interest_coverage")
    ebit_5y  = data.get("ebit_5y") or []

    if coverage is None:
        return None

    if coverage < 1.5:
        return _flag(
            "COVERAGE_DETERIORATION", "HIGH",
            f"Interest coverage = {coverage:.1f}x — EBIT barely covers interest expense. "
            f"High default risk in any earnings downturn"
        )
    if coverage < 3.0:
        return _flag(
            "COVERAGE_DETERIORATION", "MEDIUM",
            f"Interest coverage = {coverage:.1f}x — below the 3x safety threshold; "
            f"limited buffer against earnings volatility"
        )

    # Even if currently OK, warn if EBIT trend is down and coverage is modest
    ebit_vals = _valid(ebit_5y)
    if len(ebit_vals) >= 3:
        slope = _trend_slope(ebit_vals)
        if slope is not None and slope < 0 and coverage < 6.0:
            return _flag(
                "COVERAGE_DETERIORATION", "LOW",
                f"Interest coverage = {coverage:.1f}x but EBIT trending downward — "
                f"coverage may erode if trend continues"
            )
    return None


def _check_unsustainable_dividend(data):
    div_yield = data.get("dividend_yield")
    if not div_yield or div_yield <= 0:
        return None   # No dividend

    fcf_5y = _valid(data.get("fcf_5y") or [])
    div_5y = _valid(data.get("dividends_5y") or [])

    if len(fcf_5y) < 1 or len(div_5y) < 1:
        return None

    latest_fcf = fcf_5y[-1]
    latest_div = div_5y[-1]

    if latest_fcf is None or latest_div is None or latest_div <= 0:
        return None

    if latest_fcf <= 0:
        return _flag(
            "UNSUSTAINABLE_DIVIDEND", "HIGH",
            f"Paying dividends ({_bn(latest_div)}) while generating negative FCF ({_bn(latest_fcf)}) — "
            f"dividend is funded by debt or asset sales, not operations"
        )

    payout = latest_div / latest_fcf
    if payout > 1.0:
        return _flag(
            "UNSUSTAINABLE_DIVIDEND", "HIGH",
            f"Dividends ({_bn(latest_div)}) exceed FCF ({_bn(latest_fcf)}) — "
            f"payout = {payout:.0%} of FCF. Dividend at risk without significant FCF recovery"
        )
    if payout > 0.80:
        return _flag(
            "UNSUSTAINABLE_DIVIDEND", "MEDIUM",
            f"Dividend payout = {payout:.0%} of FCF — high but covered. "
            f"Little room for earnings misses or increased investment needs"
        )
    return None


def _check_leverage_expansion(data):
    nd_ebitda = data.get("net_debt_ebitda")
    net_debt  = data.get("net_debt")
    ebit_5y   = data.get("ebit_5y") or []

    if nd_ebitda is not None:
        if nd_ebitda > 5.0:
            return _flag(
                "LEVERAGE_EXPANSION", "HIGH",
                f"Net Debt/EBITDA = {nd_ebitda:.1f}x — very high leverage. "
                f"Vulnerable to rate rises or an earnings shortfall"
            )
        if nd_ebitda > 3.5:
            return _flag(
                "LEVERAGE_EXPANSION", "MEDIUM",
                f"Net Debt/EBITDA = {nd_ebitda:.1f}x — elevated leverage with limited incremental borrowing room"
            )

    # Rising effective leverage: EBIT declining + meaningful debt
    ebit_vals = _valid(ebit_5y)
    if len(ebit_vals) >= 3 and net_debt and net_debt > 0 and nd_ebitda and nd_ebitda > 2.0:
        slope = _trend_slope(ebit_vals)
        if slope is not None and slope < 0:
            return _flag(
                "LEVERAGE_EXPANSION", "LOW",
                f"EBIT declining while Net Debt/EBITDA = {nd_ebitda:.1f}x — "
                f"effective leverage ratio rising as earnings erode"
            )
    return None


# ── main function ──────────────────────────────────────────────────────────────

def analyze_red_flags(data: dict, wacc: float | None = None) -> dict:
    """
    Run all 10 red flag detectors. Pass wacc from valuation engine to enable
    the capital destruction check.
    """
    raw = [
        _check_capex_trap(data),
        _check_margin_compression(data),
        _check_heavy_dilution(data),
        _check_fcf_conversion(data),
        _check_sbc_burden(data),
        _check_revenue_fcf_divergence(data),
        _check_capital_destruction(data, wacc),
        _check_interest_coverage(data),
        _check_unsustainable_dividend(data),
        _check_leverage_expansion(data),
    ]

    flags = [f for f in raw if f is not None]

    # Sort: HIGH → MEDIUM → LOW
    _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags.sort(key=lambda f: _order.get(f["severity"], 3))

    highs   = sum(1 for f in flags if f["severity"] == "HIGH")
    mediums = sum(1 for f in flags if f["severity"] == "MEDIUM")
    lows    = sum(1 for f in flags if f["severity"] == "LOW")

    if highs >= 3:
        summary = f"SERIOUS CONCERNS — {highs} high-severity patterns detected"
    elif highs >= 1:
        summary = f"{highs} high-severity + {mediums} medium pattern(s) — warrants deep due diligence"
    elif mediums >= 2:
        summary = f"No critical issues but {mediums} medium patterns — monitor closely"
    elif flags:
        summary = f"Minor patterns only ({lows} low) — not alarming individually"
    else:
        summary = "No red flag patterns detected across all 10 checks"

    return {
        "red_flags":    flags,
        "summary":      summary,
        "count_high":   highs,
        "count_medium": mediums,
        "count_low":    lows,
    }


# ── display helper ─────────────────────────────────────────────────────────────

_SEVERITY_LABEL = {"HIGH": "[HIGH]  ", "MEDIUM": "[MEDIUM]", "LOW": "[LOW]   "}

def print_red_flags(result: dict, ticker: str = ""):
    header = f"RED FLAGS — {ticker}" if ticker else "RED FLAGS"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")

    flags = result.get("red_flags") or []
    if not flags:
        print(f"  No red flag patterns detected.")
    else:
        for f in flags:
            label = _SEVERITY_LABEL.get(f["severity"], "       ")
            print(f"  {label}  {f['pattern']}")
            print(f"           {f['detail']}")

    print(f"\n  Summary: {result['summary']}")
    if result["count_high"] or result["count_medium"] or result["count_low"]:
        print(
            f"  Counts:  HIGH {result['count_high']}  |  "
            f"MEDIUM {result['count_medium']}  |  LOW {result['count_low']}"
        )
