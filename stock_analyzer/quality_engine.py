"""
quality_engine.py  —  Business Quality Analysis

analyze_quality(data) → dict with every calculated metric, its value,
a plain-English assessment, and the benchmark used.

No made-up point system. Every output is a real number or observation.
"""

import numpy as np
from utils import _pct, _x, _assess, _count_positive, _cagr_from_list


def _growth_years(yoy_changes):
    valid = [c for c in yoy_changes if c is not None]
    pos   = sum(1 for c in valid if c > 0)
    return pos, len(valid)


# ── main function ─────────────────────────────────────────────────────────────

def analyze_quality(data: dict) -> dict:
    """
    Returns a dict of metric blocks. Each block has:
        value       — the raw number
        formatted   — human-readable string
        assessment  — e.g. "Strong", "Adequate", "Weak"
        benchmark   — the threshold used to make that call
        detail      — extra context (trend, years, etc.)
    """
    metrics = {}
    flags   = []

    # ── Revenue Growth ────────────────────────────────────────────────────
    rev_5y = data.get("revenue_5y") or [None]*5
    cagr   = data.get("revenue_cagr_5y") or _cagr_from_list(rev_5y)[0]
    yoy    = data.get("revenue_yoy_changes") or []

    pos_years, total_years = _growth_years(yoy)

    metrics["revenue_cagr_5y"] = {
        "label":      "Revenue CAGR (5Y)",
        "value":      cagr,
        "formatted":  _pct(cagr),
        "assessment": _assess(cagr, [
            (0.08, "Strong (>8% p.a.)"),
            (0.05, "Moderate (5–8% p.a.)"),
            (0.03, "Slow (3–5% p.a.)"),
            (0.00, "Flat (<3% p.a.)"),
            (-99,  "Declining (negative)"),
        ]),
        "benchmark":  ">8% = strong, 3–8% = moderate, <0% = declining",
        "detail":     f"Revenue grew in {pos_years} of {total_years} measurable years",
    }

    metrics["revenue_growth_consistency"] = {
        "label":      "Revenue Growth Consistency",
        "value":      (pos_years / total_years) if total_years else None,
        "formatted":  f"{pos_years}/{total_years} years positive" if total_years else "N/A",
        "assessment": _assess(
            (pos_years / total_years) if total_years else None,
            [(0.80, "Consistent"), (0.60, "Mostly consistent"),
             (0.40, "Erratic"), (-99,  "Very erratic")]
        ),
        "benchmark":  "≥80% of years growing = consistent",
        "detail":     f"YoY growth rates: {[f'{c:.1%}' if c is not None else 'N/A' for c in yoy]}",
    }

    # Revenue acceleration / deceleration
    if len(yoy) >= 4:
        first_half  = [c for c in yoy[:len(yoy)//2] if c is not None]
        second_half = [c for c in yoy[len(yoy)//2:] if c is not None]
        if first_half and second_half:
            accel = np.mean(second_half) - np.mean(first_half)
            metrics["revenue_acceleration"] = {
                "label":      "Revenue Acceleration",
                "value":      accel,
                "formatted":  f"{accel:+.1%} pp change in avg growth",
                "assessment": "Accelerating" if accel > 0.01 else ("Decelerating" if accel < -0.01 else "Stable"),
                "benchmark":  ">+1pp = accelerating, <-1pp = decelerating",
                "detail":     f"Early avg: {np.mean(first_half):.1%}, Recent avg: {np.mean(second_half):.1%}",
            }

    # ── Earnings Quality ──────────────────────────────────────────────────
    ni_5y = data.get("net_income_5y") or [None]*5
    ni_pos, ni_total = _count_positive(ni_5y)

    ni_formatted = [f"{v/1e9:.2f}B" if v is not None else "N/A" for v in ni_5y]
    metrics["earnings_consistency"] = {
        "label":      "Earnings Consistency",
        "value":      (ni_pos / ni_total) if ni_total else None,
        "formatted":  f"{ni_pos}/{ni_total} profitable years",
        "assessment": _assess(
            (ni_pos / ni_total) if ni_total else None,
            [(1.00, "Always profitable"),
             (0.80, "Usually profitable"),
             (0.60, "Inconsistent"),
             (-99,  "Persistent losses")]
        ),
        "benchmark":  "All years profitable = strongest signal",
        "detail":     f"Net income by year: {ni_formatted}",
    }

    # EPS CAGR
    eps_cagr = data.get("eps_cagr_5y")
    metrics["eps_cagr_5y"] = {
        "label":      "EPS CAGR (5Y)",
        "value":      eps_cagr,
        "formatted":  _pct(eps_cagr),
        "assessment": _assess(eps_cagr, [
            (0.10, "Strong (>10% p.a.)"),
            (0.05, "Moderate (5–10%)"),
            (0.00, "Flat"),
            (-99,  "Declining / negative"),
        ]) if eps_cagr is not None else "No data",
        "benchmark":  ">10% EPS growth = strong compounder",
        "detail":     "Based on basic EPS; N/A if any year had losses",
    }

    # ── FCF Quality ───────────────────────────────────────────────────────
    fcf_5y = data.get("fcf_5y") or [None]*5
    fcf_pos, fcf_total = _count_positive(fcf_5y)
    fcf_formatted = [f"{v/1e9:.2f}B" if v is not None else "N/A" for v in fcf_5y]

    metrics["fcf_consistency"] = {
        "label":      "FCF Consistency",
        "value":      (fcf_pos / fcf_total) if fcf_total else None,
        "formatted":  f"{fcf_pos}/{fcf_total} years positive FCF",
        "assessment": _assess(
            (fcf_pos / fcf_total) if fcf_total else None,
            [(1.00, "Always FCF positive"),
             (0.80, "Usually FCF positive"),
             (0.60, "Mixed"),
             (-99,  "Mostly FCF negative")]
        ),
        "benchmark":  "4+ of 5 years positive = strong",
        "detail":     f"FCF by year: {fcf_formatted}",
    }

    fcf_cagr = data.get("fcf_cagr_5y")
    metrics["fcf_cagr_5y"] = {
        "label":      "FCF CAGR (5Y)",
        "value":      fcf_cagr,
        "formatted":  _pct(fcf_cagr),
        "assessment": _assess(fcf_cagr, [
            (0.10, "Growing strongly"),
            (0.05, "Growing"),
            (0.00, "Flat"),
            (-99,  "Declining"),
        ]) if fcf_cagr is not None else "No data",
        "benchmark":  "FCF growing ≥10% p.a. = strong cash generation",
        "detail":     "N/A if FCF was negative in early years",
    }

    # ── Margin Analysis ───────────────────────────────────────────────────
    gm = data.get("gross_margin")
    metrics["gross_margin"] = {
        "label":      "Gross Margin",
        "value":      gm,
        "formatted":  _pct(gm),
        "assessment": _assess(gm, [
            (0.60, "Exceptional (>60%) — pricing power / asset-light"),
            (0.50, "Strong (50–60%)"),
            (0.35, "Decent (35–50%)"),
            (0.20, "Low (20–35%) — commodity or heavy COGS"),
            (-99,  "Very low (<20%)"),
        ]),
        "benchmark":  ">50% = pricing power, <20% = commodity business",
        "detail":     "Gross profit / revenue (latest year)",
    }

    om = data.get("operating_margin")
    metrics["operating_margin"] = {
        "label":      "Operating Margin (EBIT margin)",
        "value":      om,
        "formatted":  _pct(om),
        "assessment": _assess(om, [
            (0.25, "Excellent (>25%)"),
            (0.15, "Strong (15–25%)"),
            (0.08, "Adequate (8–15%)"),
            (0.03, "Thin (3–8%)"),
            (0.00, "Breakeven or marginally profitable"),
            (-99,  "Operating losses"),
        ]),
        "benchmark":  ">15% = strong, <5% = structurally challenged",
        "detail":     "EBIT / revenue (latest year)",
    }

    nm = data.get("net_margin")
    metrics["net_margin"] = {
        "label":      "Net Margin",
        "value":      nm,
        "formatted":  _pct(nm),
        "assessment": _assess(nm, [
            (0.20, "High (>20%)"),
            (0.10, "Good (10–20%)"),
            (0.05, "Moderate (5–10%)"),
            (0.00, "Low but positive"),
            (-99,  "Net losses"),
        ]),
        "benchmark":  ">10% = good bottom-line efficiency",
        "detail":     "Net income / revenue (latest year)",
    }

    # ── Capital Efficiency ────────────────────────────────────────────────
    roic = data.get("roic")
    metrics["roic"] = {
        "label":      "ROIC (Return on Invested Capital)",
        "value":      roic,
        "formatted":  _pct(roic),
        "assessment": _assess(roic, [
            (0.20, "Exceptional (>20%) — wide economic moat"),
            (0.15, "Strong (15–20%)"),
            (0.10, "Adequate (10–15%)"),
            (0.06, "Below average (6–10%) — marginal returns"),
            (0.00, "Poor (<6%)"),
            (-99,  "Destroying capital (negative)"),
        ]),
        "benchmark":  ">15% = strong moat, <10% = questionable competitive advantage",
        "detail":     "EBIT × (1 − tax rate) / (equity + net debt)",
    }

    roe = data.get("roe")
    metrics["roe"] = {
        "label":      "ROE (Return on Equity)",
        "value":      roe,
        "formatted":  _pct(roe),
        "assessment": _assess(roe, [
            (0.20, "Strong (>20%)"),
            (0.12, "Decent (12–20%)"),
            (0.06, "Weak (6–12%)"),
            (0.00, "Very weak"),
            (-99,  "Negative"),
        ]),
        "benchmark":  ">15% sustained = quality business",
        "detail":     "Net income / shareholders' equity",
    }

    roa = data.get("roa")
    metrics["roa"] = {
        "label":      "ROA (Return on Assets)",
        "value":      roa,
        "formatted":  _pct(roa),
        "assessment": _assess(roa, [
            (0.10, "Strong (>10%)"),
            (0.05, "Decent (5–10%)"),
            (0.02, "Weak (2–5%)"),
            (0.00, "Very weak"),
            (-99,  "Negative"),
        ]),
        "benchmark":  ">5% = asset-efficient",
        "detail":     "Net income / total assets",
    }

    # ── Shareholder Treatment ─────────────────────────────────────────────
    dilution = data.get("shares_change_pct")
    if dilution is not None:
        if dilution <= -0.05:
            dil_assess = f"Active buybacks — share count down {abs(dilution):.1%} over 5Y"
        elif dilution <= 0.0:
            dil_assess = f"Slight buybacks / flat — {dilution:.1%} change over 5Y"
        elif dilution <= 0.05:
            dil_assess = f"Minor dilution — share count up {dilution:.1%} over 5Y"
        elif dilution <= 0.10:
            dil_assess = f"Noticeable dilution — share count up {dilution:.1%} over 5Y"
        else:
            dil_assess = f"Heavy dilution — share count up {dilution:.1%} over 5Y"
            flags.append(f"Share count grew {dilution:.1%} over 5Y — hurts per-share value")
    else:
        dil_assess = "No data"

    metrics["share_dilution"] = {
        "label":      "Share Count Change (5Y)",
        "value":      dilution,
        "formatted":  _pct(dilution) if dilution is not None else "N/A",
        "assessment": dil_assess,
        "benchmark":  "Negative = buybacks (good), >10% = significant dilution",
        "detail":     "Compares current shares outstanding to 5 years ago",
    }

    # SBC check
    sbc_5y   = data.get("sbc_5y") or [None]*5
    rev_last  = data.get("revenue")
    sbc_last  = next((v for v in reversed(sbc_5y) if v is not None), None)
    if sbc_last is not None and rev_last and rev_last > 0:
        sbc_pct = abs(sbc_last) / rev_last
        metrics["stock_based_compensation"] = {
            "label":      "Stock-Based Compensation (% of Revenue)",
            "value":      sbc_pct,
            "formatted":  _pct(sbc_pct),
            "assessment": (
                "High (>10%) — watch for hidden dilution cost"
                if sbc_pct > 0.10 else
                "Elevated (5–10%)" if sbc_pct > 0.05 else
                "Moderate (<5%)"
            ),
            "benchmark":  ">10% of revenue in SBC = dilution concern",
            "detail":     "SBC reduces real FCF available to shareholders",
        }

    # ── Reinvestment ──────────────────────────────────────────────────────
    capex_pct = data.get("capex_pct_revenue")
    metrics["capex_intensity"] = {
        "label":      "Capex as % of Revenue (5Y avg)",
        "value":      capex_pct,
        "formatted":  _pct(capex_pct),
        "assessment": _assess(capex_pct, [
            (0.15, "Capital-heavy business (>15%)"),
            (0.08, "Moderate capex (8–15%)"),
            (0.03, "Light capex (3–8%) — asset-light model"),
            (-99,  "Very low capex (<3%) — minimal reinvestment"),
        ]) if capex_pct is not None else "No data",
        "benchmark":  "<5% capex/revenue = asset-light; >15% = capital-intensive",
        "detail":     "Average of available years; lower = more cash-generative",
    }

    dep_5y   = data.get("depreciation_5y") or [None]*5
    capex_5y = data.get("capex_5y")        or [None]*5
    reinv_ratios = []
    for d, c in zip(dep_5y, capex_5y):
        if d and c and d != 0:
            reinv_ratios.append(abs(c) / d)
    if reinv_ratios:
        avg_reinv = np.mean(reinv_ratios)
        metrics["reinvestment_ratio"] = {
            "label":      "Reinvestment Ratio (Capex / Depreciation)",
            "value":      avg_reinv,
            "formatted":  f"{avg_reinv:.2f}x",
            "assessment": (
                "Investing for growth (>1.2x)"   if avg_reinv > 1.2 else
                "Maintenance only (~1x)"          if avg_reinv >= 0.8 else
                "Underinvesting (<0.8x) — may signal declining asset base"
            ),
            "benchmark":  ">1x = growth capex, <0.8x = possibly milking assets",
            "detail":     f"Avg ratio: {avg_reinv:.2f}x over available years",
        }

    # ── Notable flags ─────────────────────────────────────────────────────
    if cagr is not None and cagr < 0:
        flags.append("Revenue is shrinking — negative 5Y CAGR")
    if (ni_pos / ni_total if ni_total else 1) < 0.4:
        flags.append("Less than 40% of years were profitable — persistent loss-maker")
    if (fcf_pos / fcf_total if fcf_total else 1) < 0.4:
        flags.append("FCF negative in majority of years — cash burn concern")
    if roic is not None and roic < 0:
        flags.append("Negative ROIC — business is destroying capital")
    if om is not None and om < 0:
        flags.append("Negative operating margin — core operations unprofitable")

    return {
        "quality_metrics": metrics,
        "quality_flags":   flags,
    }


# ── display helper ────────────────────────────────────────────────────────────

def print_quality(result: dict, ticker: str = ""):
    header = f"BUSINESS QUALITY — {ticker}" if ticker else "BUSINESS QUALITY"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    for key, m in result["quality_metrics"].items():
        print(f"  {m['label']:<42} {m['formatted']:<12}  {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")
    if result["quality_flags"]:
        print(f"\n  ⚠  Flags:")
        for f in result["quality_flags"]:
            print(f"     • {f}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "revenue_cagr_5y": 0.10,
        "revenue_yoy_changes": [0.08, 0.12, 0.09, 0.11, 0.10],
        "net_income_5y": [1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9],
        "fcf_5y": [0.8e9, 0.9e9, 1.0e9, 1.1e9, 1.2e9],
        "fcf_cagr_5y": 0.107,
        "eps_cagr_5y": 0.095,
        "gross_margin": 0.55,
        "operating_margin": 0.22,
        "net_margin": 0.16,
        "roic": 0.18,
        "roe": 0.24,
        "roa": 0.08,
        "shares_change_pct": -0.05,
        "capex_pct_revenue": 0.04,
        "depreciation_5y": [200e6]*5,
        "capex_5y": [-250e6]*5,
        "sbc_5y": [50e6]*5,
        "revenue": 5e9,
    }
    result = analyze_quality(sample)
    print_quality(result, "SAMPLE.CO")
