"""
red_flags.py  —  Pattern-based red flag detection (counter-only)

``analyze_red_flags(data, wacc=None) -> dict`` with:

* ``red_flags``      list of ``{pattern, severity, detail}`` dicts. ``detail``
                     is a compact ``key=value`` string ("capex_pct 8.0%->12.0%
                     rev_cagr 3.0%") suitable for log lines and ML features —
                     **not** rich human prose. Reporting that wants prose can
                     reconstruct it from the structured fields.
* ``count_high/medium/low``  integer severity counts (the only thing the ML
                             feature pipeline and rule composite consume).
* ``summary``        deterministic short string ("3H/2M/0L"). The earlier
                     verbose-prose summary was eye candy with no caller.

This module was 437 lines of verbose detail prose feeding a single output
field used by a dashboard that no longer exists. Detection logic is
preserved; prose was the bloat.
"""

from __future__ import annotations

from typing import Any


# ── helpers ────────────────────────────────────────────────────────────────

def _valid(lst):
    return [v for v in (lst or []) if v is not None]


def _simple_cagr(start, end, years):
    if years <= 0 or start is None or end is None or start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def _trend_slope(lst):
    vals = _valid(lst)
    if len(vals) < 3:
        return None
    n = len(vals)
    x_mean = (n - 1) / 2
    y_mean = sum(vals) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def _half_averages(series):
    n = len(series)
    half = max(1, n // 2)
    early = sum(series[:half]) / half
    late = sum(series[half:]) / max(1, n - half)
    return early, late


def _flag(pattern: str, severity: str, detail: str) -> dict[str, Any]:
    return {"pattern": pattern, "severity": severity, "detail": detail}


# ── detectors ──────────────────────────────────────────────────────────────
# Each detector returns at most ONE flag (HIGH > MEDIUM > LOW priority).
# Detail strings are key=value pairs only; no narrative.

def _check_capex_trap(data):
    capex_5y = _valid(data.get("capex_5y") or [])
    rev_5y = _valid(data.get("revenue_5y") or [])
    rev_cagr = data.get("revenue_cagr_5y")
    if len(capex_5y) < 3 or len(rev_5y) < 3:
        return None
    pairs = [(abs(c), r) for c, r in zip(capex_5y, rev_5y) if r and r > 0]
    if len(pairs) < 3:
        return None
    cpx_pct = [c / r for c, r in pairs]
    early, late = _half_averages(cpx_pct)
    rising = late > early * 1.15
    if rising and rev_cagr is not None and rev_cagr < 0.04:
        return _flag("CAPEX_TRAP", "HIGH",
                     f"capex_pct {early:.1%}->{late:.1%} rev_cagr {rev_cagr:.1%}")
    if rising and rev_cagr is not None and rev_cagr < 0.08:
        return _flag("CAPEX_TRAP", "MEDIUM",
                     f"capex_pct {early:.1%}->{late:.1%} rev_cagr {rev_cagr:.1%}")
    return None


def _check_margin_compression(data):
    ebit_5y = _valid(data.get("ebit_5y") or [])
    rev_5y = _valid(data.get("revenue_5y") or [])
    if len(ebit_5y) < 3 or len(rev_5y) < 3:
        return None
    margins = [e / r for e, r in zip(ebit_5y, rev_5y) if r and r > 0]
    if len(margins) < 3:
        return None
    early, late = _half_averages(margins)
    delta = late - early
    if delta < -0.05:
        return _flag("MARGIN_COMPRESSION", "HIGH",
                     f"op_margin {early:.1%}->{late:.1%} delta {delta:+.1%}")
    if delta < -0.02:
        return _flag("MARGIN_COMPRESSION", "MEDIUM",
                     f"op_margin {early:.1%}->{late:.1%} delta {delta:+.1%}")
    return None


def _check_heavy_dilution(data):
    shares_5y = _valid(data.get("shares_5y") or [])
    if len(shares_5y) < 2:
        return None
    cagr = _simple_cagr(shares_5y[0], shares_5y[-1], len(shares_5y) - 1)
    if cagr is None:
        return None
    if cagr > 0.05:
        return _flag("HEAVY_DILUTION", "HIGH", f"shares_cagr {cagr:.1%}")
    if cagr > 0.03:
        return _flag("HEAVY_DILUTION", "MEDIUM", f"shares_cagr {cagr:.1%}")
    if cagr > 0.015:
        return _flag("HEAVY_DILUTION", "LOW", f"shares_cagr {cagr:.1%}")
    return None


def _check_fcf_conversion(data):
    fcf_5y = _valid(data.get("fcf_5y") or [])
    ni_5y = _valid(data.get("net_income_5y") or [])
    if len(fcf_5y) < 3 or len(ni_5y) < 3:
        return None
    ratios = [f / n for f, n in zip(fcf_5y, ni_5y) if n and n > 0 and f is not None]
    if len(ratios) < 2:
        return None
    avg = sum(ratios) / len(ratios)
    low_count = sum(1 for r in ratios if r < 0.60)
    if avg < 0.50 or low_count >= len(ratios) - 1:
        return _flag("WEAK_FCF_CONVERSION", "HIGH", f"fcf_to_ni_avg {avg:.0%}")
    if avg < 0.70:
        return _flag("WEAK_FCF_CONVERSION", "MEDIUM", f"fcf_to_ni_avg {avg:.0%}")
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
        return _flag("SBC_BURDEN", "HIGH", f"sbc_pct {latest:.1%}")
    if latest > 0.03:
        return _flag("SBC_BURDEN", "MEDIUM", f"sbc_pct {latest:.1%}")
    return None


def _check_revenue_fcf_divergence(data):
    rev_cagr = data.get("revenue_cagr_5y")
    fcf_cagr = data.get("fcf_cagr_5y")
    if rev_cagr is None or fcf_cagr is None:
        return None
    gap = rev_cagr - fcf_cagr
    if rev_cagr > 0.05 and fcf_cagr < 0:
        return _flag("REVENUE_FCF_DIVERGENCE", "HIGH",
                     f"rev_cagr {rev_cagr:.1%} fcf_cagr {fcf_cagr:.1%}")
    if rev_cagr > 0.04 and gap > 0.06:
        return _flag("REVENUE_FCF_DIVERGENCE", "MEDIUM",
                     f"rev_cagr {rev_cagr:.1%} fcf_cagr {fcf_cagr:.1%} gap {gap:.1%}")
    return None


def _check_capital_destruction(data, wacc):
    if wacc is None:
        return None
    roic = data.get("roic")
    if roic is None:
        return None
    spread = roic - wacc
    if spread < -0.05:
        return _flag("CAPITAL_DESTRUCTION", "HIGH",
                     f"roic {roic:.1%} wacc {wacc:.1%} spread {spread:+.1%}")
    if spread < 0:
        return _flag("CAPITAL_DESTRUCTION", "MEDIUM",
                     f"roic {roic:.1%} wacc {wacc:.1%} spread {spread:+.1%}")
    return None


def _check_interest_coverage(data):
    coverage = data.get("interest_coverage")
    if coverage is None:
        return None
    if coverage < 1.5:
        return _flag("COVERAGE_DETERIORATION", "HIGH", f"int_coverage {coverage:.1f}x")
    if coverage < 3.0:
        return _flag("COVERAGE_DETERIORATION", "MEDIUM", f"int_coverage {coverage:.1f}x")
    ebit_vals = _valid(data.get("ebit_5y") or [])
    if len(ebit_vals) >= 3:
        slope = _trend_slope(ebit_vals)
        if slope is not None and slope < 0 and coverage < 6.0:
            return _flag("COVERAGE_DETERIORATION", "LOW",
                         f"int_coverage {coverage:.1f}x ebit_slope {slope:+.0f}")
    return None


def _check_unsustainable_dividend(data):
    div_yield = data.get("dividend_yield")
    if not div_yield or div_yield <= 0:
        return None
    fcf_5y = _valid(data.get("fcf_5y") or [])
    div_5y = _valid(data.get("dividends_5y") or [])
    if len(fcf_5y) < 1 or len(div_5y) < 1:
        return None
    latest_fcf = fcf_5y[-1]
    latest_div = div_5y[-1]
    if latest_fcf is None or latest_div is None or latest_div <= 0:
        return None
    if latest_fcf <= 0:
        return _flag("UNSUSTAINABLE_DIVIDEND", "HIGH", f"fcf<=0 div>0")
    payout = latest_div / latest_fcf
    if payout > 1.0:
        return _flag("UNSUSTAINABLE_DIVIDEND", "HIGH", f"payout_of_fcf {payout:.0%}")
    if payout > 0.80:
        return _flag("UNSUSTAINABLE_DIVIDEND", "MEDIUM", f"payout_of_fcf {payout:.0%}")
    return None


def _check_leverage_expansion(data):
    nd_ebitda = data.get("net_debt_ebitda")
    if nd_ebitda is not None:
        if nd_ebitda > 5.0:
            return _flag("LEVERAGE_EXPANSION", "HIGH", f"nd_ebitda {nd_ebitda:.1f}x")
        if nd_ebitda > 3.5:
            return _flag("LEVERAGE_EXPANSION", "MEDIUM", f"nd_ebitda {nd_ebitda:.1f}x")
    ebit_vals = _valid(data.get("ebit_5y") or [])
    net_debt = data.get("net_debt")
    if (
        len(ebit_vals) >= 3
        and net_debt and net_debt > 0
        and nd_ebitda and nd_ebitda > 2.0
    ):
        slope = _trend_slope(ebit_vals)
        if slope is not None and slope < 0:
            return _flag("LEVERAGE_EXPANSION", "LOW",
                         f"nd_ebitda {nd_ebitda:.1f}x ebit_slope {slope:+.0f}")
    return None


# ── main ───────────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def analyze_red_flags(data: dict, wacc: float | None = None) -> dict:
    """Run all 10 red-flag detectors. ``wacc`` enables the capital
    destruction check; ``None`` skips it."""
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
    flags.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 3))

    highs = sum(1 for f in flags if f["severity"] == "HIGH")
    mediums = sum(1 for f in flags if f["severity"] == "MEDIUM")
    lows = sum(1 for f in flags if f["severity"] == "LOW")
    summary = f"{highs}H/{mediums}M/{lows}L"

    return {
        "red_flags": flags,
        "summary": summary,
        "count_high": highs,
        "count_medium": mediums,
        "count_low": lows,
    }
