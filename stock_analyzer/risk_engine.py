"""
risk_engine.py  —  Phase 4b: Risk Profile Analysis

analyze_risk(data) → dict with:
  - Market risk (beta)
  - Earnings & FCF volatility (coefficient of variation)
  - Revenue cyclicality
  - Leverage-induced risk
  - Dilution risk
  - Distance from 52-week high / low (downside already absorbed?)
  - Combined risk summary

Higher score = LOWER risk (safer stock).
But we show actual numbers, not scores.
"""

import numpy as np
from typing import Optional
from utils import _pct, _x, _bn, _valid, _cv


def _max_drawdown(lst) -> Optional[float]:
    """Largest peak-to-trough drop in a list of values."""
    valid = _valid(lst)
    if len(valid) < 2:
        return None
    peak = valid[0]
    max_dd = 0.0
    for v in valid[1:]:
        if v > peak:
            peak = v
        elif peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)
    return max_dd


# ── main function ─────────────────────────────────────────────────────────────

def analyze_risk(data: dict) -> dict:
    metrics = {}
    flags   = []
    critical_flags = []

    # ── Market risk: Beta ─────────────────────────────────────────────────
    beta = data.get("beta")
    if beta is None:
        beta_assess = "Beta unavailable — cannot measure systematic risk"
        beta_detail = "Using 1.0 as proxy in WACC calculations"
    elif beta < 0:
        beta_assess = f"Negative beta ({beta:.2f}) — moves opposite to market (rare: gold, defensive)"
        beta_detail = "Negative beta stocks often provide portfolio hedging but may be data anomalies"
    elif beta < 0.50:
        beta_assess = f"Very low beta ({beta:.2f}) — much less volatile than market"
        beta_detail = "Examples: utilities, consumer staples, healthcare"
    elif beta < 0.80:
        beta_assess = f"Low beta ({beta:.2f}) — below-market volatility"
        beta_detail = "Defensive characteristics — less upside and downside than market"
    elif beta < 1.20:
        beta_assess = f"Market beta ({beta:.2f}) — moves roughly with the market"
        beta_detail = "Average market sensitivity"
    elif beta < 1.50:
        beta_assess = f"Elevated beta ({beta:.2f}) — more volatile than market"
        beta_detail = "Amplified market movements — higher upside and downside"
    else:
        beta_assess = f"High beta ({beta:.2f}) — significantly more volatile than market"
        beta_detail = "Aggressive risk profile — suitable only for high risk tolerance"
        flags.append(f"High beta of {beta:.2f} — stock amplifies market swings significantly")

    metrics["beta"] = {
        "label":     "Beta (Market Sensitivity)",
        "value":     beta,
        "formatted": f"{beta:.2f}" if beta is not None else "N/A",
        "assessment": beta_assess,
        "benchmark": "<0.8 = defensive, 0.8–1.2 = market, >1.5 = aggressive",
        "detail":    beta_detail,
    }

    # ── Earnings volatility ───────────────────────────────────────────────
    ni_5y = data.get("net_income_5y") or [None]*5
    ni_cv = _cv(ni_5y)
    ni_max_dd = _max_drawdown(ni_5y)

    if ni_cv is not None:
        if ni_cv < 0.15:
            ni_assess = f"Very stable earnings (CV {ni_cv:.2f}) — highly predictable"
        elif ni_cv < 0.30:
            ni_assess = f"Stable earnings (CV {ni_cv:.2f}) — low variability"
        elif ni_cv < 0.60:
            ni_assess = f"Moderate earnings volatility (CV {ni_cv:.2f}) — some cyclicality"
        elif ni_cv < 1.0:
            ni_assess = f"High earnings volatility (CV {ni_cv:.2f}) — significantly cyclical"
            flags.append(f"Earnings are highly volatile (CV {ni_cv:.2f}) — difficult to forecast")
        else:
            ni_assess = f"Extreme earnings volatility (CV {ni_cv:.2f}) — unpredictable"
            flags.append(f"Extreme earnings volatility (CV {ni_cv:.2f}) — very high uncertainty")
    else:
        ni_assess = "Insufficient data for earnings volatility"

    metrics["earnings_volatility"] = {
        "label":     "Earnings Volatility (Coeff. of Variation)",
        "value":     ni_cv,
        "formatted": f"{ni_cv:.2f}" if ni_cv is not None else "N/A",
        "assessment": ni_assess,
        "benchmark": "CV <0.15 = very stable, 0.30–0.60 = moderate, >1.0 = extreme",
        "detail": (
            f"Net income by year: {[_bn(v) for v in ni_5y]}  "
            f"| Worst peak-to-trough drop: {_pct(ni_max_dd)}"
        ),
    }

    # ── FCF volatility ────────────────────────────────────────────────────
    fcf_5y = data.get("fcf_5y") or [None]*5
    fcf_cv = _cv(fcf_5y)
    fcf_max_dd = _max_drawdown(fcf_5y)

    if fcf_cv is not None:
        if fcf_cv < 0.20:
            fcf_assess = f"Very stable FCF (CV {fcf_cv:.2f})"
        elif fcf_cv < 0.40:
            fcf_assess = f"Moderate FCF variation (CV {fcf_cv:.2f})"
        elif fcf_cv < 0.80:
            fcf_assess = f"High FCF volatility (CV {fcf_cv:.2f}) — cash generation is lumpy"
            flags.append(f"FCF volatility is high (CV {fcf_cv:.2f}) — harder to value reliably")
        else:
            fcf_assess = f"Extreme FCF volatility (CV {fcf_cv:.2f})"
    else:
        fcf_assess = "Insufficient FCF data"

    metrics["fcf_volatility"] = {
        "label":     "FCF Volatility (Coeff. of Variation)",
        "value":     fcf_cv,
        "formatted": f"{fcf_cv:.2f}" if fcf_cv is not None else "N/A",
        "assessment": fcf_assess,
        "benchmark": "CV <0.20 = stable, 0.40–0.80 = lumpy, >0.80 = very unpredictable",
        "detail": (
            f"FCF by year: {[_bn(v) for v in fcf_5y]}  "
            f"| Worst peak-to-trough drop: {_pct(fcf_max_dd)}"
        ),
    }

    # ── Revenue cyclicality ───────────────────────────────────────────────
    rev_5y = data.get("revenue_5y") or [None]*5
    rev_cv = _cv(rev_5y)
    rev_max_dd = _max_drawdown(rev_5y)

    if rev_cv is not None:
        if rev_cv < 0.08:
            rev_assess = f"Highly stable revenue (CV {rev_cv:.2f}) — defensive / staples-like"
        elif rev_cv < 0.15:
            rev_assess = f"Low cyclicality (CV {rev_cv:.2f})"
        elif rev_cv < 0.30:
            rev_assess = f"Moderate cyclicality (CV {rev_cv:.2f}) — some economic sensitivity"
        elif rev_cv < 0.50:
            rev_assess = f"High cyclicality (CV {rev_cv:.2f}) — revenue swings meaningfully with economy"
            flags.append(f"Revenue is highly cyclical (CV {rev_cv:.2f}) — vulnerable to economic downturns")
        else:
            rev_assess = f"Very high cyclicality (CV {rev_cv:.2f}) — commodity/project-based or distressed"
            flags.append(f"Extreme revenue cyclicality (CV {rev_cv:.2f})")
    else:
        rev_assess = "Insufficient revenue data"

    metrics["revenue_cyclicality"] = {
        "label":     "Revenue Cyclicality (Coeff. of Variation)",
        "value":     rev_cv,
        "formatted": f"{rev_cv:.2f}" if rev_cv is not None else "N/A",
        "assessment": rev_assess,
        "benchmark": "CV <0.08 = defensive, 0.15–0.30 = moderate, >0.50 = highly cyclical",
        "detail": (
            f"Revenue by year: {[_bn(v) for v in rev_5y]}  "
            f"| Worst revenue decline: {_pct(rev_max_dd)}"
        ),
    }

    # ── Leverage risk ──────────────────────────────────────────────────────
    nd_ebitda = data.get("net_debt_ebitda")
    ic        = data.get("interest_coverage")
    de        = data.get("debt_equity")

    lev_signals = []
    if nd_ebitda is not None:
        if nd_ebitda > 4.0:
            lev_signals.append(f"Net Debt/EBITDA {nd_ebitda:.1f}x — high refinancing risk")
            critical_flags.append("HIGH_LEVERAGE_RISK")
        elif nd_ebitda > 2.5:
            lev_signals.append(f"Net Debt/EBITDA {nd_ebitda:.1f}x — moderate leverage")
        else:
            lev_signals.append(f"Net Debt/EBITDA {nd_ebitda:.1f}x — manageable leverage")

    if ic is not None and ic > 0:
        if ic < 2.0:
            lev_signals.append(f"Interest coverage {ic:.1f}x — earnings barely cover debt service")
            critical_flags.append("THIN_INTEREST_COVERAGE")
        elif ic < 5.0:
            lev_signals.append(f"Interest coverage {ic:.1f}x — adequate but not comfortable")
        else:
            lev_signals.append(f"Interest coverage {ic:.1f}x — comfortable")

    lev_summary = "  |  ".join(lev_signals) if lev_signals else "No leverage data available"

    metrics["leverage_risk"] = {
        "label":     "Leverage Risk Summary",
        "value":     nd_ebitda,
        "formatted": f"{nd_ebitda:.1f}x ND/EBITDA" if nd_ebitda is not None else "N/A",
        "assessment": lev_summary,
        "benchmark": "ND/EBITDA >4x + IC <2x = distress territory",
        "detail":    f"D/E: {de:.2f}x" if de is not None else "D/E: N/A",
    }

    # ── Dilution risk ─────────────────────────────────────────────────────
    dilution = data.get("shares_change_pct")
    sbc_5y   = data.get("sbc_5y") or [None]*5
    rev_last = data.get("revenue")
    sbc_last = next((abs(v) for v in reversed(sbc_5y) if v is not None), None)
    sbc_pct  = (sbc_last / rev_last) if (sbc_last and rev_last) else None

    if dilution is not None:
        if dilution < -0.03:
            dil_assess = f"Buyback program: share count down {abs(dilution):.1%} — per-share value increasing"
        elif dilution <= 0.03:
            dil_assess = f"Negligible dilution ({dilution:.1%}) — share count stable"
        elif dilution <= 0.08:
            dil_assess = f"Moderate dilution ({dilution:.1%} over 5Y) — watch SBC and equity issuance"
            flags.append(f"Moderate dilution ({dilution:.1%} over 5Y)")
        elif dilution <= 0.15:
            dil_assess = f"Significant dilution ({dilution:.1%} over 5Y) — meaningful per-share drag"
            flags.append(f"Significant share dilution ({dilution:.1%} over 5Y) — eroding per-share value")
        else:
            dil_assess = f"Heavy dilution ({dilution:.1%} over 5Y) — major per-share value destruction"
            flags.append(f"Heavy dilution ({dilution:.1%} over 5Y) — serious risk to per-share returns")
            critical_flags.append("HEAVY_DILUTION")
    else:
        dil_assess = "No share count data"

    sbc_detail = f"SBC as % of revenue: {_pct(sbc_pct)}" if sbc_pct else "SBC data unavailable"

    metrics["dilution_risk"] = {
        "label":     "Dilution Risk (Share Count Change 5Y)",
        "value":     dilution,
        "formatted": _pct(dilution) if dilution is not None else "N/A",
        "assessment": dil_assess,
        "benchmark": "Share count decrease = buybacks (good); >10% increase = dilution concern",
        "detail":    sbc_detail,
    }

    # ── 52-week range position ────────────────────────────────────────────
    price     = data.get("current_price")
    high_52w  = data.get("week52_high")
    low_52w   = data.get("week52_low")

    if price and high_52w and low_52w and high_52w != low_52w:
        pct_from_high = (price - high_52w) / high_52w      # negative = below high
        pct_from_low  = (price - low_52w)  / low_52w       # positive = above low
        range_pct     = (price - low_52w) / (high_52w - low_52w)  # 0=at low, 1=at high

        if pct_from_high > -0.05:
            range_assess = f"Near 52W high ({pct_from_high:.1%} below high) — limited near-term margin of safety"
        elif pct_from_high > -0.15:
            range_assess = f"Moderate pullback from 52W high ({abs(pct_from_high):.1%} below)"
        elif pct_from_high > -0.30:
            range_assess = f"Significant pullback ({abs(pct_from_high):.1%} below 52W high) — potential entry zone"
        else:
            range_assess = f"Deep pullback ({abs(pct_from_high):.1%} below 52W high) — either opportunity or fundamental deterioration"
            flags.append(f"Stock is {abs(pct_from_high):.0%} below its 52W high — investigate the cause")

        metrics["price_range_position"] = {
            "label":     "52-Week Range Position",
            "value":     range_pct,
            "formatted": f"{range_pct:.0%} of 52W range",
            "assessment": range_assess,
            "benchmark": "Near low = more margin of safety; near high = less cushion",
            "detail": (
                f"Current: {price:.2f}  "
                f"| 52W high: {high_52w:.2f}  "
                f"| 52W low: {low_52w:.2f}  "
                f"| Distance from high: {pct_from_high:.1%}  "
                f"| Distance from low: {pct_from_low:.1%}"
            ),
        }

    # ── Negative earnings risk ────────────────────────────────────────────
    ni_5y_valid = _valid(ni_5y)
    neg_earnings = sum(1 for v in ni_5y_valid if v < 0)
    if neg_earnings >= 2:
        flags.append(f"Negative earnings in {neg_earnings} of last {len(ni_5y_valid)} years")
        critical_flags.append("PERSISTENT_LOSSES")

    fcf_neg = sum(1 for v in _valid(fcf_5y) if v < 0)
    if fcf_neg >= 3:
        critical_flags.append("FCF_NEGATIVE_MAJORITY")

    # ── Combined risk assessment ──────────────────────────────────────────
    risk_signals = []
    if beta is not None:
        if beta < 0.8:    risk_signals.append("Low market risk")
        elif beta < 1.3:  risk_signals.append("Moderate market risk")
        else:             risk_signals.append("High market risk")

    if ni_cv is not None:
        if ni_cv < 0.20:  risk_signals.append("stable earnings")
        elif ni_cv < 0.50:risk_signals.append("moderate earnings volatility")
        else:             risk_signals.append("volatile earnings")

    if nd_ebitda is not None:
        if nd_ebitda <= 0:    risk_signals.append("net cash (no leverage risk)")
        elif nd_ebitda <= 2:  risk_signals.append("low leverage")
        elif nd_ebitda <= 4:  risk_signals.append("moderate leverage")
        else:                 risk_signals.append("HIGH leverage risk")

    if dilution is not None:
        if dilution < 0:      risk_signals.append("buybacks (no dilution risk)")
        elif dilution > 0.08: risk_signals.append("dilution risk")

    combined = " / ".join(risk_signals) if risk_signals else "Insufficient data for combined assessment"

    metrics["combined_risk"] = {
        "label":     "Combined Risk Assessment",
        "value":     None,
        "formatted": "",
        "assessment": combined,
        "benchmark": "Synthesis of market, earnings, leverage, and dilution risk",
        "detail":    f"Critical flags: {', '.join(critical_flags)}" if critical_flags else "No critical risk flags",
    }

    return {
        "risk_metrics":   metrics,
        "risk_flags":     flags,
        "critical_flags": critical_flags,
    }


# ── display helper ────────────────────────────────────────────────────────────

def print_risk(result: dict, ticker: str = ""):
    header = f"RISK PROFILE — {ticker}" if ticker else "RISK PROFILE"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    for key, m in result["risk_metrics"].items():
        if m["formatted"]:
            print(f"  {m['label']:<42} {m['formatted']:<12}  {m['assessment']}")
        else:
            print(f"  {m['label']:<42} {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")
    if result["risk_flags"]:
        print(f"\n  ⚠  Risk flags:")
        for f in result["risk_flags"]:
            print(f"     • {f}")
    if result["critical_flags"]:
        print(f"\n  🚨  Critical risk flags:")
        for f in result["critical_flags"]:
            print(f"     • {f}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "beta": 1.1,
        "net_income_5y": [3e9, 3.3e9, 3.7e9, 4.1e9, 4.8e9],
        "fcf_5y": [2.8e9, 3.1e9, 3.5e9, 3.8e9, 4.5e9],
        "revenue_5y": [20e9, 22e9, 24e9, 26e9, 29e9],
        "net_debt_ebitda": 0.8,
        "interest_coverage": 15.0,
        "debt_equity": 0.3,
        "shares_change_pct": -0.04,
        "sbc_5y": [300e6]*5,
        "revenue": 29e9,
        "current_price": 210.0,
        "week52_high": 240.0,
        "week52_low": 160.0,
    }
    result = analyze_risk(sample)
    print_risk(result, "SAMPLE.CO")
