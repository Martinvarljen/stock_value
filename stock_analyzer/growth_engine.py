"""
growth_engine.py  —  Phase 4a: Growth Durability Analysis

analyze_growth(data) → dict with:
  - Margin trend (expanding or contracting over time)
  - Earnings quality (FCF conversion ratio)
  - Revenue per share growth (dilution-adjusted)
  - EPS vs revenue growth gap (implies margin direction)
  - FCF growth trend
  - Reinvestment effectiveness

No made-up point scores. All outputs are real calculations.
"""

import numpy as np
from typing import Optional
from utils import _pct, _bn, _valid, _cagr


def _slope_trend(values: list) -> Optional[float]:
    """
    Fit a linear trend to a list of values (oldest → newest).
    Returns the slope normalised by the mean — positive = improving, negative = declining.
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 3:
        return None
    xs = np.array([i for i, _ in valid], dtype=float)
    ys = np.array([v for _, v in valid], dtype=float)
    mean_y = np.mean(ys)
    if mean_y == 0:
        return None
    slope = np.polyfit(xs, ys, 1)[0]
    return slope / abs(mean_y)   # normalised slope


# ── main function ─────────────────────────────────────────────────────────────

def analyze_growth(data: dict) -> dict:
    metrics = {}
    flags   = []

    # ── Margin trends ─────────────────────────────────────────────────────
    # Build margin series from raw 5Y data
    rev_5y  = data.get("revenue_5y")    or [None]*5
    gp_5y   = data.get("gross_profit_5y") or [None]*5
    ebit_5y = data.get("ebit_5y")        or [None]*5
    ni_5y   = data.get("net_income_5y")  or [None]*5
    fcf_5y  = data.get("fcf_5y")         or [None]*5

    gm_series  = []
    om_series  = []
    nm_series  = []
    fcf_m_series = []

    for r, gp, eb, ni, fc in zip(rev_5y, gp_5y, ebit_5y, ni_5y, fcf_5y):
        gm_series.append(gp / r  if r and gp  else None)
        om_series.append(eb / r  if r and eb  else None)
        nm_series.append(ni / r  if r and ni  else None)
        fcf_m_series.append(fc / r if r and fc else None)

    def _margin_trend_block(label, series, benchmark):
        valid = _valid(series)
        if len(valid) < 2:
            return {"label": label, "value": None, "formatted": "N/A",
                    "assessment": "Insufficient data", "benchmark": benchmark,
                    "detail": "Need at least 2 years of data"}

        trend = _slope_trend(series)
        first_half_avg = np.mean(_valid(series[:len(series)//2])) if _valid(series[:len(series)//2]) else None
        second_half_avg = np.mean(_valid(series[len(series)//2:])) if _valid(series[len(series)//2:]) else None

        if first_half_avg and second_half_avg:
            change = second_half_avg - first_half_avg
            if change > 0.01:
                direction = f"Expanding (+{change:.1%} pp, recent avg {second_half_avg:.1%} vs early avg {first_half_avg:.1%})"
            elif change < -0.01:
                direction = f"Contracting ({change:.1%} pp, recent avg {second_half_avg:.1%} vs early avg {first_half_avg:.1%})"
            else:
                direction = f"Stable (recent avg {second_half_avg:.1%})"
        else:
            direction = "Insufficient data for trend"

        formatted_series = [f"{v:.1%}" if v is not None else "N/A" for v in series]

        return {
            "label":     label,
            "value":     valid[-1],
            "formatted": _pct(valid[-1]),
            "assessment": direction,
            "benchmark": benchmark,
            "detail":    f"By year: {formatted_series}",
        }

    metrics["gross_margin_trend"] = _margin_trend_block(
        "Gross Margin Trend (5Y)",
        gm_series,
        "Expanding = pricing power improving; contracting = cost pressure or mix shift"
    )
    metrics["operating_margin_trend"] = _margin_trend_block(
        "Operating Margin Trend (5Y)",
        om_series,
        "Expanding = operating leverage; contracting = rising costs or competition"
    )
    metrics["net_margin_trend"] = _margin_trend_block(
        "Net Margin Trend (5Y)",
        nm_series,
        "Net margin expansion is the strongest signal of compounding profitability"
    )

    # ── Earnings quality: FCF conversion ──────────────────────────────────
    # FCF conversion = FCF / Net Income — measures how much reported profit is real cash
    conversion_series = []
    for fc, ni in zip(fcf_5y, ni_5y):
        if ni and ni > 0 and fc is not None:
            conversion_series.append(fc / ni)
        else:
            conversion_series.append(None)

    valid_conv = _valid(conversion_series)
    avg_conv   = np.mean(valid_conv) if valid_conv else None

    if avg_conv is not None:
        if avg_conv >= 1.10:
            conv_assess = f"Excellent ({avg_conv:.2f}x avg) — FCF consistently exceeds reported earnings"
        elif avg_conv >= 0.85:
            conv_assess = f"Good ({avg_conv:.2f}x avg) — earnings largely backed by cash"
        elif avg_conv >= 0.60:
            conv_assess = f"Moderate ({avg_conv:.2f}x avg) — some gap between earnings and cash"
        elif avg_conv >= 0.30:
            conv_assess = f"Weak ({avg_conv:.2f}x avg) — significant portion of earnings not converting to cash"
            flags.append(f"Low FCF conversion ({avg_conv:.2f}x) — earnings quality concern")
        else:
            conv_assess = f"Very weak ({avg_conv:.2f}x avg) — earnings are largely non-cash"
            flags.append(f"Very low FCF conversion ({avg_conv:.2f}x) — high accruals or accounting concern")
    else:
        conv_assess = "Cannot calculate (negative earnings years excluded)"

    metrics["fcf_conversion"] = {
        "label":     "Earnings Quality (FCF / Net Income)",
        "value":     avg_conv,
        "formatted": f"{avg_conv:.2f}x" if avg_conv is not None else "N/A",
        "assessment": conv_assess,
        "benchmark": ">1x = cash earnings exceed reported earnings (high quality); <0.5x = red flag",
        "detail":    f"By year: {[f'{v:.2f}x' if v is not None else 'N/A' for v in conversion_series]}",
    }

    # ── Revenue per share growth (dilution-adjusted) ───────────────────────
    shares_5y  = data.get("shares_5y") or [None]*5
    rev_ps_series = []
    for r, s in zip(rev_5y, shares_5y):
        rev_ps_series.append(r / s if r and s and s > 0 else None)

    valid_rps = [(i, v) for i, v in enumerate(rev_ps_series) if v is not None]
    if len(valid_rps) >= 2:
        rps_start = valid_rps[0][1]
        rps_end   = valid_rps[-1][1]
        rps_years = valid_rps[-1][0] - valid_rps[0][0]
        rps_cagr  = _cagr(rps_start, rps_end, rps_years)
        rev_cagr  = data.get("revenue_cagr_5y")
        gap = (rps_cagr - rev_cagr) if (rps_cagr is not None and rev_cagr is not None) else None

        if gap is not None:
            if gap >= 0.01:
                gap_assess = f"Revenue/share growing faster than total revenue (+{gap:.1%} gap) — buybacks adding per-share value"
            elif gap > -0.01:
                gap_assess = "Revenue/share roughly in line with total revenue — minimal dilution effect"
            else:
                gap_assess = f"Revenue/share lagging total revenue ({gap:.1%} gap) — dilution is eroding per-share growth"
                flags.append(f"Share dilution is reducing per-share revenue growth by {abs(gap):.1%} p.a.")
        else:
            gap_assess = "Cannot compare — missing data"

        metrics["revenue_per_share_growth"] = {
            "label":     "Revenue per Share CAGR (5Y)",
            "value":     rps_cagr,
            "formatted": _pct(rps_cagr),
            "assessment": gap_assess,
            "benchmark": "Should be close to or above total revenue CAGR — gap = dilution drag",
            "detail":    f"Total revenue CAGR: {_pct(rev_cagr)}  | Revenue/share CAGR: {_pct(rps_cagr)}",
        }
    else:
        metrics["revenue_per_share_growth"] = {
            "label": "Revenue per Share CAGR (5Y)", "value": None,
            "formatted": "N/A", "assessment": "Insufficient shares data",
            "benchmark": "", "detail": "",
        }

    # ── EPS vs Revenue growth gap (margin direction signal) ──────────────
    eps_cagr = data.get("eps_cagr_5y")
    rev_cagr = data.get("revenue_cagr_5y")
    if eps_cagr is not None and rev_cagr is not None:
        eps_rev_gap = eps_cagr - rev_cagr
        if eps_rev_gap > 0.02:
            eps_assess = f"EPS growing {eps_rev_gap:.1%} faster than revenue — margins expanding, operating leverage at work"
        elif eps_rev_gap > -0.02:
            eps_assess = f"EPS growing in line with revenue — margins stable"
        else:
            eps_assess = f"EPS growing {abs(eps_rev_gap):.1%} slower than revenue — margin compression or rising share count"
            flags.append(f"EPS growth ({_pct(eps_cagr)}) lagging revenue growth ({_pct(rev_cagr)}) — margins shrinking or dilution")

        metrics["eps_vs_revenue_growth"] = {
            "label":     "EPS CAGR vs Revenue CAGR",
            "value":     eps_rev_gap,
            "formatted": f"{eps_rev_gap:+.1%} gap",
            "assessment": eps_assess,
            "benchmark": "Positive gap = margin expansion or buybacks; negative = margin compression",
            "detail":    f"EPS CAGR: {_pct(eps_cagr)}  | Revenue CAGR: {_pct(rev_cagr)}",
        }

    # ── FCF growth trend ───────────────────────────────────────────────────
    fcf_cagr = data.get("fcf_cagr_5y")
    rev_cagr = data.get("revenue_cagr_5y")
    if fcf_cagr is not None:
        if rev_cagr and fcf_cagr > rev_cagr + 0.02:
            fcf_vs_rev = f"FCF growing faster than revenue ({_pct(fcf_cagr)} vs {_pct(rev_cagr)}) — FCF margins expanding"
        elif rev_cagr and fcf_cagr < rev_cagr - 0.02:
            fcf_vs_rev = f"FCF growing slower than revenue ({_pct(fcf_cagr)} vs {_pct(rev_cagr)}) — FCF margin compression"
        else:
            fcf_vs_rev = f"FCF growth roughly in line with revenue"

        metrics["fcf_growth_trend"] = {
            "label":     "FCF CAGR (5Y)",
            "value":     fcf_cagr,
            "formatted": _pct(fcf_cagr),
            "assessment": fcf_vs_rev,
            "benchmark": "FCF CAGR > revenue CAGR = improving cash conversion",
            "detail":    f"FCF by year: {[_bn(v) for v in fcf_5y]}",
        }

    # ── FCF margin trend ───────────────────────────────────────────────────
    metrics["fcf_margin_trend"] = _margin_trend_block(
        "FCF Margin Trend (5Y)",
        fcf_m_series,
        "FCF margin = FCF / revenue — rising is the gold standard of compounding quality"
    )

    # ── Reinvestment effectiveness ─────────────────────────────────────────
    capex_5y = data.get("capex_5y") or [None]*5
    avg_capex_pct = data.get("capex_pct_revenue")  # already computed in data_layer
    dep_5y = data.get("depreciation_5y") or [None]*5
    reinv_ratios = []
    for d, c in zip(dep_5y, capex_5y):
        if d and c and d > 0:
            reinv_ratios.append(abs(c) / d)

    avg_reinv = np.mean(reinv_ratios) if reinv_ratios else None

    if avg_capex_pct is not None and rev_cagr is not None:
        # High capex + high growth = efficient reinvestment
        # High capex + low growth = capital trap
        if avg_capex_pct > 0.10 and (rev_cagr or 0) > 0.08:
            reinv_assess = f"Capital-heavy ({_pct(avg_capex_pct)} capex/rev) but generating strong growth — efficient reinvestment"
        elif avg_capex_pct > 0.10 and (rev_cagr or 0) <= 0.03:
            reinv_assess = f"Capital-heavy ({_pct(avg_capex_pct)} capex/rev) with weak growth — potential capital trap"
            flags.append("High capex intensity with low revenue growth — question the reinvestment return")
        elif avg_capex_pct < 0.05:
            reinv_assess = f"Asset-light ({_pct(avg_capex_pct)} capex/rev) — minimal reinvestment needed to sustain growth"
        else:
            reinv_assess = f"Moderate capex intensity ({_pct(avg_capex_pct)} capex/rev)"

        if avg_reinv:
            reinv_assess += f"  |  Capex/Depreciation: {avg_reinv:.2f}x ({'growth capex' if avg_reinv > 1.1 else 'maintenance capex'})"

        metrics["reinvestment_effectiveness"] = {
            "label":     "Reinvestment Effectiveness",
            "value":     avg_capex_pct,
            "formatted": _pct(avg_capex_pct),
            "assessment": reinv_assess,
            "benchmark": "Asset-light + high growth = ideal; heavy capex + low growth = trap",
            "detail":    f"Avg capex/revenue: {_pct(avg_capex_pct)}  | Avg capex/depreciation: {f'{avg_reinv:.2f}x' if avg_reinv else 'N/A'}",
        }

    # ── Revenue acceleration ──────────────────────────────────────────────
    yoy = data.get("revenue_yoy_changes") or []
    if len(yoy) >= 4:
        first_half  = _valid(yoy[:len(yoy)//2])
        second_half = _valid(yoy[len(yoy)//2:])
        if first_half and second_half:
            accel = np.mean(second_half) - np.mean(first_half)
            if accel > 0.02:
                accel_assess = f"Accelerating — recent growth ({np.mean(second_half):.1%} avg) above earlier period ({np.mean(first_half):.1%} avg)"
            elif accel < -0.02:
                accel_assess = f"Decelerating — recent growth ({np.mean(second_half):.1%} avg) below earlier period ({np.mean(first_half):.1%} avg)"
                if np.mean(second_half) < 0:
                    flags.append("Revenue is shrinking in the most recent period")
            else:
                accel_assess = f"Stable growth rate (~{np.mean(second_half):.1%} avg recent)"

            metrics["revenue_acceleration"] = {
                "label":     "Revenue Growth Acceleration",
                "value":     accel,
                "formatted": f"{accel:+.1%} pp",
                "assessment": accel_assess,
                "benchmark": "Positive = momentum building; negative = slowing or maturing",
                "detail":    f"YoY changes by year: {[f'{c:.1%}' if c is not None else 'N/A' for c in yoy]}",
            }

    return {
        "growth_metrics": metrics,
        "growth_flags":   flags,
    }


# ── display helper ────────────────────────────────────────────────────────────

def print_growth(result: dict, ticker: str = ""):
    header = f"GROWTH ANALYSIS — {ticker}" if ticker else "GROWTH ANALYSIS"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    for key, m in result["growth_metrics"].items():
        print(f"  {m['label']:<42} {m['formatted']:<12}  {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")
    if result["growth_flags"]:
        print(f"\n  ⚠  Growth flags:")
        for f in result["growth_flags"]:
            print(f"     • {f}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "revenue_5y":     [20e9, 22e9, 24e9, 26e9, 29e9],
        "gross_profit_5y":[10e9, 11e9, 12.5e9, 14e9, 16e9],
        "ebit_5y":        [4e9,  4.5e9, 5e9, 5.5e9, 6.5e9],
        "net_income_5y":  [3e9,  3.3e9, 3.7e9, 4.1e9, 4.8e9],
        "fcf_5y":         [3.2e9,3.5e9, 4e9, 4.3e9, 5e9],
        "shares_5y":      [500e6,490e6, 480e6, 470e6, 460e6],
        "capex_5y":       [-1e9, -1.1e9,-1.2e9,-1.3e9,-1.4e9],
        "depreciation_5y":[1.2e9,1.3e9, 1.4e9, 1.5e9, 1.6e9],
        "revenue_cagr_5y": 0.095,
        "eps_cagr_5y":     0.125,
        "fcf_cagr_5y":     0.118,
        "capex_pct_revenue": 0.048,
        "revenue_yoy_changes": [0.08, 0.09, 0.09, 0.10, 0.115],
    }
    result = analyze_growth(sample)
    print_growth(result, "SAMPLE.CO")
