"""
scenario_engine.py  —  Bear / Base / Bull DCF projections

Key features:
  • Growth DECAY CURVES — revenue growth linearly decays from the starting rate
    toward the terminal growth rate over the 10-year projection window, rather
    than holding a single static multiplier throughout.
  • MARGIN MEAN-REVERSION — operating margins gradually revert toward a
    long-run competitive steady-state (85% of base margin) instead of staying
    flat, which is more realistic for most businesses.
  • DYNAMIC SCENARIO WEIGHTS — instead of hard-coded 25/50/25 probabilities,
    weights are adjusted based on the company's actual leverage and earnings
    volatility. A highly leveraged, volatile company shifts weight toward bear.
"""

from utils import _pct, _bn, _num, _cv

# ── Scenario parameter table ──────────────────────────────────────────────────
SCENARIO_PARAMS = {
    "bear": {
        "label":               "Bear Case",
        "revenue_growth_mult": 0.50,   # applied to 5Y CAGR for starting growth
        "margin_mult":         0.85,   # applied to trailing margin for starting margin
        "terminal_growth":     0.015,
        "probability":         0.25,   # default; overridden by dynamic weights
    },
    "base": {
        "label":               "Base Case",
        "revenue_growth_mult": 0.85,
        "margin_mult":         1.00,
        "terminal_growth":     0.025,
        "probability":         0.50,
    },
    "bull": {
        "label":               "Bull Case",
        "revenue_growth_mult": 1.20,
        "margin_mult":         1.10,
        "terminal_growth":     0.030,
        "probability":         0.25,
    },
}


# ── Decay helpers ─────────────────────────────────────────────────────────────

def _decay_value(start: float, end: float, year: int, total_years: int) -> float:
    """
    Linear interpolation from `start` (year 1) toward `end` (year total_years).
    Models gradual mean-reversion rather than a step-change.
    """
    if total_years <= 1:
        return end
    alpha = (year - 1) / (total_years - 1)
    return start * (1 - alpha) + end * alpha


# ── Dynamic probability weights ───────────────────────────────────────────────

def calculate_dynamic_weights(data: dict) -> dict:
    """
    Adjust default 25/50/25 weights based on the company's risk profile.

    Rules:
      High leverage  → shift toward bear (more downside risk)
      High earnings volatility → spread weights wider (more uncertainty)
      All FCF years positive  → shift slightly toward bull
      Net cash position        → slight bull tilt
    """
    bear_w, base_w, bull_w = 0.25, 0.50, 0.25
    reasons = []

    nd_ebitda = data.get("net_debt_ebitda")
    if nd_ebitda is not None:
        if nd_ebitda > 5.0:
            bear_w += 0.15; base_w -= 0.05; bull_w -= 0.10
            reasons.append(f"Dangerous leverage ND/EBITDA {nd_ebitda:.1f}x → bear +15%")
        elif nd_ebitda > 3.0:
            bear_w += 0.08; bull_w -= 0.08
            reasons.append(f"High leverage ND/EBITDA {nd_ebitda:.1f}x → bear +8%")
        elif nd_ebitda <= 0:
            bull_w += 0.05; bear_w -= 0.05
            reasons.append("Net cash position → bull +5%")

    ni_cv = _cv(data.get("net_income_5y") or [])
    if ni_cv is not None:
        if ni_cv > 0.60:
            bear_w += 0.05; bull_w += 0.05; base_w -= 0.10
            reasons.append(f"High earnings volatility CV {ni_cv:.2f} → spread wider")
        elif ni_cv < 0.15:
            base_w += 0.05; bear_w -= 0.025; bull_w -= 0.025
            reasons.append(f"Very stable earnings CV {ni_cv:.2f} → base +5%")

    fcf_5y = data.get("fcf_5y") or []
    fcf_pos   = sum(1 for v in fcf_5y if v is not None and v > 0)
    fcf_total = sum(1 for v in fcf_5y if v is not None)
    if fcf_total >= 4 and fcf_pos == fcf_total:
        bull_w += 0.05; bear_w -= 0.05
        reasons.append("All FCF years positive → bull +5%")
    elif fcf_total >= 3 and fcf_pos < fcf_total * 0.5:
        bear_w += 0.08; bull_w -= 0.08
        reasons.append("FCF mostly negative → bear +8%")

    # Clamp minimums, then normalise
    bear_w = max(0.05, bear_w)
    base_w = max(0.20, base_w)
    bull_w = max(0.05, bull_w)
    total  = bear_w + base_w + bull_w

    return {
        "bear":    round(bear_w / total, 3),
        "base":    round(base_w / total, 3),
        "bull":    round(bull_w / total, 3),
        "reasons": reasons,
    }


# ── Core DCF projection ───────────────────────────────────────────────────────

def run_scenario(
    revenue:     float,
    base_growth: float,
    base_margin: float,
    tax_rate:    float,
    capex_pct:   float,
    wacc:        float,
    net_debt:    float,
    shares:      float,
    params:      dict,
    years:       int = 10,
) -> dict | None:
    """
    Project FCF for `years` years under a single scenario with decay curves.

    Growth path: starts at base_growth × growth_mult, linearly decays to
    terminal_growth by year 10 (not a flat constant).

    Margin path: starts at base_margin × margin_mult, gradually reverts toward
    a steady-state of 85% of base_margin (competitive equilibrium).
    """
    if not (revenue and revenue > 0 and shares and shares > 0 and wacc and wacc > 0):
        return None

    growth_start  = (base_growth or 0.03) * params["revenue_growth_mult"]
    margin_start  = (base_margin or 0.10) * params["margin_mult"]
    tg            = params["terminal_growth"]

    # Safety clamps
    growth_start = max(min(growth_start, 0.40), -0.10)
    margin_start = max(min(margin_start, 0.60), -0.05)
    if tg >= wacc:
        tg = wacc - 0.005

    tax = max(0.0, min(tax_rate or 0.22, 0.50))
    cx  = max(0.0, min(capex_pct or 0.03, 0.30))

    # Long-run steady-state margin: slight competitive erosion toward 85% of base
    # Bear already starts below this, so for bear we revert upward toward steady state
    steady_state = max((base_margin or 0.10) * 0.85, 0.02)

    yearly   = []
    total_pv = 0.0
    rev      = revenue

    for y in range(1, years + 1):
        yr_growth = _decay_value(growth_start, tg,           y, years)
        yr_margin = _decay_value(margin_start, steady_state, y, years)

        rev    = rev * (1 + yr_growth)
        nopat  = rev * yr_margin * (1 - tax)
        fcf    = nopat - rev * cx
        pv     = fcf / (1 + wacc) ** y
        total_pv += pv
        yearly.append({
            "year":   y,
            "growth": yr_growth,
            "margin": yr_margin,
            "revenue": rev,
            "fcf":    fcf,
            "pv_fcf": pv,
        })

    fcf_y10 = yearly[-1]["fcf"]
    tv      = fcf_y10 * (1 + tg) / (wacc - tg)
    pv_tv   = tv / (1 + wacc) ** years

    ev       = total_pv + pv_tv
    tv_pct   = pv_tv / ev if ev > 0 else None
    net_d    = net_debt or 0.0
    eq_val   = ev - net_d
    per_share = eq_val / shares if shares else None

    return {
        "label":              params["label"],
        "growth_rate":        growth_start,
        "operating_margin":   margin_start,
        "steady_state_margin":steady_state,
        "terminal_growth":    tg,
        "wacc":               wacc,
        "yearly":             yearly,
        "pv_fcfs":            total_pv,
        "terminal_value":     tv,
        "pv_terminal_value":  pv_tv,
        "tv_pct_of_ev":       tv_pct,
        "enterprise_value":   ev,
        "net_debt_used":      net_d,
        "equity_value":       eq_val,
        "per_share_value":    per_share,
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all_scenarios(data: dict, wacc: float) -> dict:
    """
    Run all 3 scenarios with dynamic weights and return results + weights used.
    """
    revenue   = data.get("revenue")
    growth    = data.get("revenue_cagr_5y") or 0.03
    margin    = data.get("operating_margin") or 0.10
    tax       = data.get("effective_tax_rate") or 0.22
    capex_pct = data.get("capex_pct_revenue") or 0.03
    net_debt  = data.get("net_debt") or 0.0
    shares    = data.get("shares_outstanding")

    dyn_weights = calculate_dynamic_weights(data)
    tg_range = data.get("terminal_growth_range")
    if isinstance(tg_range, (tuple, list)) and len(tg_range) == 2:
        tg_floor, tg_ceiling = tg_range
    else:
        tg_floor, tg_ceiling = None, None

    results = {}
    for name, params in SCENARIO_PARAMS.items():
        adjusted = {**params, "probability": dyn_weights[name]}
        # Sector-specific realism: clamp terminal growth into sector range if provided.
        if tg_floor is not None and tg_ceiling is not None:
            adjusted_tg = adjusted.get("terminal_growth")
            if adjusted_tg is not None:
                adjusted["terminal_growth"] = max(tg_floor, min(adjusted_tg, tg_ceiling))
        results[name] = run_scenario(
            revenue=revenue, base_growth=growth, base_margin=margin,
            tax_rate=tax, capex_pct=capex_pct, wacc=wacc,
            net_debt=net_debt, shares=shares, params=adjusted,
        )

    results["_weights"] = dyn_weights
    return results


def _weighted_per_share(scenarios: dict) -> float | None:
    weights = scenarios.get("_weights") or {
        "bear": SCENARIO_PARAMS["bear"]["probability"],
        "base": SCENARIO_PARAMS["base"]["probability"],
        "bull": SCENARIO_PARAMS["bull"]["probability"],
    }
    total = 0.0
    for name in ["bear", "base", "bull"]:
        s = scenarios.get(name)
        if s is None or s.get("per_share_value") is None:
            return None
        total += s["per_share_value"] * weights[name]
    return total


# ── Display ───────────────────────────────────────────────────────────────────

def print_scenarios(scenarios: dict, ticker: str = ""):
    header = f"DCF SCENARIOS — {ticker}" if ticker else "DCF SCENARIOS"
    print(f"\n{'─' * 70}")
    print(f"  {header}  (growth decay + margin mean-reversion model)")
    print(f"{'─' * 70}")

    weights = scenarios.get("_weights") or {}
    w_reasons = weights.get("reasons") or []

    rows = [
        ("Starting revenue growth",   "growth_rate",        lambda v: _pct(v)),
        ("Starting operating margin",  "operating_margin",   lambda v: _pct(v)),
        ("Steady-state margin (yr 10)","steady_state_margin",lambda v: _pct(v)),
        ("Terminal growth rate",       "terminal_growth",    lambda v: _pct(v)),
        ("WACC",                       "wacc",               lambda v: _pct(v)),
        ("PV of 10Y FCFs",             "pv_fcfs",            lambda v: _bn(v)),
        ("PV of terminal value",       "pv_terminal_value",  lambda v: _bn(v)),
        ("Terminal value % of EV",     "tv_pct_of_ev",       lambda v: _pct(v)),
        ("Enterprise value",           "enterprise_value",   lambda v: _bn(v)),
        ("Equity value (EV − net debt)","equity_value",      lambda v: _bn(v)),
        ("Intrinsic value per share",  "per_share_value",    lambda v: _num(v)),
    ]

    bw = weights.get("bear", 0.25)
    baw = weights.get("base", 0.50)
    blw = weights.get("bull", 0.25)

    print(f"  {'':38}  {'Bear':>12}  {'Base':>12}  {'Bull':>12}")
    print(f"  {'Probability weight':<38}  {bw:>11.0%}  {baw:>11.0%}  {blw:>11.0%}")
    print(f"  {'─'*38}  {'─'*12}  {'─'*12}  {'─'*12}")

    for label, key, fmt in rows:
        vals = []
        for name in ["bear", "base", "bull"]:
            s = scenarios.get(name)
            v = s.get(key) if s else None
            vals.append(fmt(v) if v is not None else "N/A")
        print(f"  {label:<38}  {vals[0]:>12}  {vals[1]:>12}  {vals[2]:>12}")

    weighted = _weighted_per_share(scenarios)
    if weighted is not None:
        print(f"\n  {'Probability-weighted fair value':<38}  {'':>12}  {weighted:>11.2f}")

    if w_reasons:
        print(f"\n  Weight adjustments from defaults (25/50/25):")
        for r in w_reasons:
            print(f"    • {r}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "revenue": 30e9, "revenue_cagr_5y": 0.08,
        "operating_margin": 0.25, "effective_tax_rate": 0.22,
        "capex_pct_revenue": 0.04, "net_debt": -5e9,
        "shares_outstanding": 400e6,
        "net_debt_ebitda": -0.5,
        "net_income_5y": [2e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9],
        "fcf_5y": [1.8e9, 2.0e9, 2.2e9, 2.4e9, 2.6e9],
    }
    scenarios = run_all_scenarios(sample, wacc=0.085)
    print_scenarios(scenarios, "SAMPLE.CO")
