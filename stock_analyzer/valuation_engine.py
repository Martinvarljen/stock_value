"""
valuation_engine.py  —  Phase 3: Valuation Analysis

analyze_valuation(data, margin_of_safety=0.25) → dict with:
  - WACC breakdown (every component shown)
  - 3-scenario DCF results (via scenario_engine)
  - Probability-weighted fair value
  - Buy-below price
  - Current price vs fair value (upside / downside %)
  - Historical P/E range positioning
  - Relative multiple snapshot
  - Flags for high terminal value dependency, overvaluation, etc.

No made-up point scores. Every output is a real number or observation.
"""

from scenario_engine import run_all_scenarios, run_scenario, _weighted_per_share, SCENARIO_PARAMS, print_scenarios
from utils import _pct, _x, _bn, _num

# ── constants ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.035   # EU 10Y Bund approx — update as needed
EQUITY_RISK_PREM = 0.055   # Damodaran long-run ERP
DEFAULT_COST_OF_DEBT = 0.045
DEFAULT_TAX_RATE     = 0.22

# Damodaran-style sector median multiples (updated ~2024)
SECTOR_BENCHMARKS = {
    "Technology":             {"pe": 28, "ev_ebitda": 18, "ev_ebit": 22, "pb": 6.0, "fcf_yield": 0.030},
    "Healthcare":             {"pe": 22, "ev_ebitda": 14, "ev_ebit": 17, "pb": 4.0, "fcf_yield": 0.035},
    "Financial Services":     {"pe": 14, "ev_ebitda": None,"ev_ebit": None,"pb": 1.5,"fcf_yield": None},
    "Consumer Cyclical":      {"pe": 18, "ev_ebitda": 12, "ev_ebit": 15, "pb": 3.5, "fcf_yield": 0.040},
    "Consumer Defensive":     {"pe": 20, "ev_ebitda": 13, "ev_ebit": 16, "pb": 4.5, "fcf_yield": 0.038},
    "Industrials":            {"pe": 20, "ev_ebitda": 13, "ev_ebit": 16, "pb": 3.5, "fcf_yield": 0.040},
    "Basic Materials":        {"pe": 14, "ev_ebitda": 8,  "ev_ebit": 11, "pb": 2.0, "fcf_yield": 0.055},
    "Energy":                 {"pe": 11, "ev_ebitda": 5,  "ev_ebit": 8,  "pb": 1.5, "fcf_yield": 0.070},
    "Utilities":              {"pe": 17, "ev_ebitda": 10, "ev_ebit": 14, "pb": 1.8, "fcf_yield": 0.045},
    "Real Estate":            {"pe": 40, "ev_ebitda": 18, "ev_ebit": 25, "pb": 2.0, "fcf_yield": 0.040},
    "Communication Services": {"pe": 18, "ev_ebitda": 10, "ev_ebit": 14, "pb": 2.5, "fcf_yield": 0.045},
}


# ── WACC ──────────────────────────────────────────────────────────────────────

def calculate_wacc(data: dict) -> dict:
    """
    Returns every component of the WACC calculation with full transparency.

    Cost of equity  = risk_free + beta × ERP          (CAPM)
    Cost of debt    = interest_expense / total_debt    (or default 4.5%)
    After-tax CoD   = cost_of_debt × (1 − tax_rate)
    WACC            = (E/V) × ke  +  (D/V) × kd_after_tax
    """
    beta     = data.get("beta")
    mktcap   = data.get("market_cap")
    debt     = data.get("total_debt") or 0.0
    int_exp  = data.get("interest_expense")
    tax      = data.get("effective_tax_rate") or DEFAULT_TAX_RATE

    # Cost of equity
    if beta is None or beta <= 0:
        beta_used   = 1.0
        beta_note   = "Beta unavailable — using 1.0 (market average)"
    else:
        beta_used   = beta
        beta_note   = f"Beta from yfinance: {beta:.2f}"

    cost_of_equity = RISK_FREE_RATE + beta_used * EQUITY_RISK_PREM

    # Cost of debt
    if int_exp and int_exp != 0 and debt and debt > 0:
        cost_of_debt = abs(int_exp) / debt
        debt_note    = f"Interest expense {_bn(int_exp)} / total debt {_bn(debt)}"
    else:
        cost_of_debt = DEFAULT_COST_OF_DEBT
        debt_note    = f"No reliable interest/debt data — using default {DEFAULT_COST_OF_DEBT:.1%}"

    cost_of_debt = max(0.01, min(cost_of_debt, 0.15))   # sanity clamp
    kd_after_tax = cost_of_debt * (1 - tax)

    # Weights (use market cap for equity, book debt for debt)
    if mktcap and mktcap > 0:
        total_capital = mktcap + debt
        w_equity = mktcap  / total_capital
        w_debt   = debt    / total_capital
        weight_note = f"Market cap {_bn(mktcap)} / total capital {_bn(total_capital)}"
    else:
        w_equity = 0.80
        w_debt   = 0.20
        weight_note = "Market cap unavailable — assuming 80/20 equity/debt split"

    wacc = w_equity * cost_of_equity + w_debt * kd_after_tax
    wacc = max(0.04, min(wacc, 0.20))   # floor 4%, cap 20%

    return {
        "risk_free_rate":   RISK_FREE_RATE,
        "equity_risk_prem": EQUITY_RISK_PREM,
        "beta":             beta_used,
        "beta_note":        beta_note,
        "cost_of_equity":   cost_of_equity,
        "cost_of_debt":     cost_of_debt,
        "kd_after_tax":     kd_after_tax,
        "debt_note":        debt_note,
        "w_equity":         w_equity,
        "w_debt":           w_debt,
        "weight_note":      weight_note,
        "wacc":             wacc,
        "tax_rate_used":    tax,
    }


# ── historical P/E positioning ─────────────────────────────────────────────────

def _pe_history_position(data: dict) -> dict | None:
    current_pe = data.get("trailing_pe")
    pe_min     = data.get("pe_5y_min")
    pe_max     = data.get("pe_5y_max")
    pe_median  = data.get("pe_5y_median")

    if any(v is None for v in [current_pe, pe_min, pe_max, pe_median]):
        return None
    if pe_max == pe_min:
        return None

    percentile = (current_pe - pe_min) / (pe_max - pe_min)
    percentile = max(0.0, min(percentile, 1.0))

    if percentile <= 0.25:
        pos = f"Bottom quartile of 5Y range — historically cheap"
    elif percentile <= 0.50:
        pos = f"Below median of 5Y range — below-average valuation"
    elif percentile <= 0.75:
        pos = f"Above median of 5Y range — above-average valuation"
    else:
        pos = f"Top quartile of 5Y range — historically expensive"

    return {
        "current_pe":  current_pe,
        "pe_5y_min":   pe_min,
        "pe_5y_max":   pe_max,
        "pe_5y_median":pe_median,
        "percentile":  percentile,
        "assessment":  pos,
    }


# ── ROIC vs WACC spread ───────────────────────────────────────────────────────

def _roic_wacc_spread(data: dict, wacc: float) -> dict:
    """
    ROIC − WACC spread: positive = value creation, negative = capital destruction.
    Uses trailing ROIC from data_layer (EBIT*(1-t) / invested_capital).
    """
    roic = data.get("roic")
    if roic is None:
        return {"available": False}

    spread = roic - wacc

    if spread > 0.10:
        verdict = f"Strong economic moat — earning {spread:.1%} above cost of capital"
    elif spread > 0.03:
        verdict = f"Value-creating — earning {spread:.1%} above cost of capital"
    elif spread > 0.0:
        verdict = f"Marginally above cost of capital ({spread:.1%} spread)"
    elif spread > -0.03:
        verdict = f"Roughly at cost of capital — minimal value creation ({spread:.1%})"
    elif spread > -0.08:
        verdict = f"Destroying capital — ROIC trails WACC by {abs(spread):.1%}"
    else:
        verdict = f"Severe capital destruction — ROIC trails WACC by {abs(spread):.1%}"

    return {
        "available":  True,
        "roic":       roic,
        "wacc":       wacc,
        "spread":     spread,
        "verdict":    verdict,
        "flag":       spread < 0,
    }


# ── Terminal value sensitivity grid ───────────────────────────────────────────

def _tv_sensitivity(data: dict, base_wacc: float) -> dict | None:
    """
    Grid of per-share intrinsic values across WACC ± 2% and terminal growth ± 1%.
    Uses base-case scenario parameters (margin_mult=1.0, growth_mult=0.85).
    Returns a dict with `wacc_steps`, `tg_steps`, and `grid` (list of rows).
    """
    revenue   = data.get("revenue")
    shares    = data.get("shares_outstanding")
    if not (revenue and revenue > 0 and shares and shares > 0):
        return None

    base_growth   = data.get("revenue_cagr_5y") or 0.03
    base_margin   = data.get("operating_margin") or 0.10
    tax           = data.get("effective_tax_rate") or DEFAULT_TAX_RATE
    capex_pct     = data.get("capex_pct_revenue") or 0.03
    net_debt      = data.get("net_debt") or 0.0

    wacc_steps = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]
    tg_steps   = [0.010, 0.015, 0.020, 0.025, 0.030]

    base_params = {**SCENARIO_PARAMS["base"], "terminal_growth": 0.025}  # overridden per cell

    grid = []
    for wc in wacc_steps:
        row = []
        for tg in tg_steps:
            cell_params = {**base_params, "terminal_growth": tg}
            res = run_scenario(
                revenue=revenue, base_growth=base_growth, base_margin=base_margin,
                tax_rate=tax, capex_pct=capex_pct, wacc=wc,
                net_debt=net_debt, shares=shares, params=cell_params,
            )
            row.append(res["per_share_value"] if res else None)
        grid.append(row)

    return {
        "wacc_steps": wacc_steps,
        "tg_steps":   tg_steps,
        "grid":       grid,
        "base_wacc":  base_wacc,
    }


# ── Reverse DCF ───────────────────────────────────────────────────────────────

def _reverse_dcf(data: dict, wacc: float) -> dict | None:
    """
    Binary-search for the implied revenue CAGR that makes the DCF base-case
    per-share value equal to the current market price.
    Reports whether the market is pricing in aggressive or modest growth.
    """
    price  = data.get("current_price")
    shares = data.get("shares_outstanding")
    revenue = data.get("revenue")
    if not (price and price > 0 and shares and shares > 0 and revenue and revenue > 0):
        return None

    base_margin = data.get("operating_margin") or 0.10
    tax         = data.get("effective_tax_rate") or DEFAULT_TAX_RATE
    capex_pct   = data.get("capex_pct_revenue") or 0.03
    net_debt    = data.get("net_debt") or 0.0

    # Use base-case margin mult and terminal growth; only vary growth
    base_params = {**SCENARIO_PARAMS["base"]}

    def _val_at_growth(g: float) -> float | None:
        res = run_scenario(
            revenue=revenue, base_growth=g, base_margin=base_margin,
            tax_rate=tax, capex_pct=capex_pct, wacc=wacc,
            net_debt=net_debt, shares=shares,
            params={**base_params, "revenue_growth_mult": 1.0},
        )
        return res["per_share_value"] if res else None

    # Binary search between -5% and 50%
    lo, hi = -0.05, 0.50
    implied = None
    for _ in range(40):
        mid = (lo + hi) / 2
        val = _val_at_growth(mid)
        if val is None:
            return None
        if abs(val - price) < 0.01:
            implied = mid
            break
        if val < price:
            lo = mid
        else:
            hi = mid
    if implied is None:
        implied = (lo + hi) / 2

    hist_growth = data.get("revenue_cagr_5y") or 0.03

    if implied > hist_growth * 1.5 and implied > 0.10:
        assessment = (
            f"Market implies {implied:.1%} revenue CAGR — significantly above trailing "
            f"{hist_growth:.1%}. Pricing in strong acceleration."
        )
    elif implied > hist_growth * 1.1:
        assessment = (
            f"Market implies {implied:.1%} CAGR — modestly above trailing {hist_growth:.1%}. "
            f"Reasonable if recent momentum holds."
        )
    elif implied >= 0:
        assessment = (
            f"Market implies {implied:.1%} CAGR — at or below trailing {hist_growth:.1%}. "
            f"Conservative pricing vs. recent growth."
        )
    else:
        assessment = (
            f"Market implies {implied:.1%} CAGR (revenue contraction). "
            f"Pricing in declining business."
        )

    return {
        "implied_growth":  implied,
        "trailing_growth": hist_growth,
        "current_price":   price,
        "assessment":      assessment,
    }


# ── Sector benchmark comparison ───────────────────────────────────────────────

def _sector_comparison(data: dict) -> dict | None:
    sector = data.get("sector")
    if not sector:
        return None
    bench = SECTOR_BENCHMARKS.get(sector)
    if not bench:
        return None

    key_map = {
        "pe":        ("trailing_pe",  "P/E",       False),   # (data_key, label, higher_is_cheaper)
        "ev_ebitda": ("ev_ebitda",    "EV/EBITDA", False),
        "ev_ebit":   ("ev_ebit",      "EV/EBIT",   False),
        "pb":        ("pb_ratio",     "P/B",       False),
        "fcf_yield": ("fcf_yield",    "FCF Yield", True),
    }

    comparisons = []
    for mult_key, (data_key, label, higher_cheaper) in key_map.items():
        bench_val = bench.get(mult_key)
        cur_val   = data.get(data_key)
        if bench_val is None or cur_val is None or cur_val <= 0:
            continue

        if higher_cheaper:
            # FCF yield: higher = cheaper; negative premium = expensive vs sector
            premium = (cur_val - bench_val) / bench_val
            direction = "above" if premium > 0 else "below"
            cheap_flag = premium > 0
        else:
            premium = (cur_val - bench_val) / bench_val
            direction = "premium" if premium > 0 else "discount"
            cheap_flag = premium < 0

        comparisons.append({
            "label":     label,
            "current":   cur_val,
            "benchmark": bench_val,
            "premium":   premium,
            "direction": direction,
            "cheap":     cheap_flag,
        })

    return {"sector": sector, "comparisons": comparisons}


# ── main function ──────────────────────────────────────────────────────────────

def analyze_valuation(data: dict, margin_of_safety: float = 0.25, wacc_adjustment: float = 0.0) -> dict:
    """
    Full valuation analysis. Returns a flat dict of metric blocks.
    wacc_adjustment: sector-specific float to add to the CAPM WACC (from sector_engine).
    """
    metrics = {}
    flags   = []

    # ── WACC ─────────────────────────────────────────────────────────────
    wacc_data = calculate_wacc(data)

    # Apply sector WACC adjustment (e.g. +0.5% for Tech, -0.8% for Utilities)
    if wacc_adjustment:
        raw_wacc = wacc_data["wacc"]
        adjusted = max(0.04, min(raw_wacc + wacc_adjustment, 0.20))
        wacc_data["wacc"]             = adjusted
        wacc_data["sector_adjustment"] = wacc_adjustment
        wacc_data["wacc_before_adj"]  = raw_wacc

    wacc = wacc_data["wacc"]

    adj_note = ""
    if wacc_adjustment:
        adj_note = (
            f"  |  Sector adj: {wacc_adjustment:+.1%} "
            f"({_pct(wacc_data['wacc_before_adj'])} → {_pct(wacc)})"
        )

    metrics["wacc_components"] = {
        "label":     "WACC Calculation",
        "value":     wacc,
        "formatted": _pct(wacc),
        "assessment": (
            f"Risk-free {_pct(wacc_data['risk_free_rate'])}  +  "
            f"Beta {wacc_data['beta']:.2f} × ERP {_pct(wacc_data['equity_risk_prem'])}  "
            f"→  Cost of equity {_pct(wacc_data['cost_of_equity'])}"
        ),
        "benchmark": "Typical range 6–12%; lower WACC → higher fair value",
        "detail": (
            f"Cost of debt (pre-tax): {_pct(wacc_data['cost_of_debt'])}  "
            f"| After-tax: {_pct(wacc_data['kd_after_tax'])}  "
            f"| Weights: {wacc_data['w_equity']:.0%} equity / {wacc_data['w_debt']:.0%} debt  "
            f"| {wacc_data['beta_note']}"
            + adj_note
        ),
    }

    # ── Run 3-scenario DCF ────────────────────────────────────────────────
    scenarios = run_all_scenarios(data, wacc)

    revenue   = data.get("revenue")
    if revenue is None:
        flags.append("No revenue data — DCF cannot be calculated")
        return {"valuation_metrics": metrics, "valuation_flags": flags,
                "scenarios": scenarios, "wacc_data": wacc_data,
                "fair_value_weighted": None, "buy_below_price": None}

    # Inputs summary
    metrics["dcf_inputs"] = {
        "label":     "DCF Inputs",
        "value":     None,
        "formatted": "",
        "assessment": (
            f"Base revenue: {_bn(revenue)}  "
            f"| Base growth (5Y CAGR): {_pct(data.get('revenue_cagr_5y'))}  "
            f"| Base op. margin: {_pct(data.get('operating_margin'))}  "
            f"| Tax rate: {_pct(data.get('effective_tax_rate') or DEFAULT_TAX_RATE)}  "
            f"| Capex/revenue: {_pct(data.get('capex_pct_revenue'))}"
        ),
        "benchmark": "All inputs derived from trailing 5Y financials — no analyst estimates",
        "detail":    f"Net debt: {_bn(data.get('net_debt'))}  | Shares: {_bn(data.get('shares_outstanding'))}",
    }

    # Per-scenario results
    for name in ["bear", "base", "bull"]:
        s = scenarios.get(name)
        if s is None:
            continue
        tv_flag = ""
        if s.get("tv_pct_of_ev") is not None and s["tv_pct_of_ev"] > 0.75:
            tv_flag = f"   Terminal value is {s['tv_pct_of_ev']:.0%} of total EV — heavily assumption-dependent"
            if s["tv_pct_of_ev"] > 0.80:
                flags.append(f"{name.title()} case: terminal value = {s['tv_pct_of_ev']:.0%} of EV — treat with caution")

        metrics[f"dcf_{name}"] = {
            "label":     f"DCF {name.title()} Case",
            "value":     s.get("per_share_value"),
            "formatted": _num(s.get("per_share_value")),
            "assessment": (
                f"Growth {_pct(s['growth_rate'])} / Margin {_pct(s['operating_margin'])} / "
                f"Terminal {_pct(s['terminal_growth'])} → "
                f"EV {_bn(s['enterprise_value'])}, equity {_bn(s['equity_value'])}"
                + tv_flag
            ),
            "benchmark": f"Probability weight: {SCENARIO_PARAMS[name]['probability']:.0%}",
            "detail": (
                f"PV of FCFs: {_bn(s['pv_fcfs'])}  "
                f"| PV of terminal value: {_bn(s['pv_terminal_value'])}  "
                f"| Terminal % of EV: {_pct(s.get('tv_pct_of_ev'))}"
            ),
        }

    # Probability-weighted fair value
    fair_value = _weighted_per_share(scenarios)
    metrics["fair_value_weighted"] = {
        "label":     "Probability-Weighted Fair Value",
        "value":     fair_value,
        "formatted": _num(fair_value),
        "assessment": (
            f"25% Bear ({_num(scenarios['bear'].get('per_share_value') if scenarios.get('bear') else None)})  +  "
            f"50% Base ({_num(scenarios['base'].get('per_share_value') if scenarios.get('base') else None)})  +  "
            f"25% Bull ({_num(scenarios['bull'].get('per_share_value') if scenarios.get('bull') else None)})"
            if fair_value else "Cannot calculate — missing scenario data"
        ),
        "benchmark": "This is a DCF estimate, not a guarantee. Treat as a range, not a precise target.",
        "detail":    "Weighted: 25% bear + 50% base + 25% bull",
    }

    # Buy-below price
    buy_below = fair_value * (1 - margin_of_safety) if fair_value else None
    metrics["buy_below_price"] = {
        "label":     f"Buy-Below Price ({margin_of_safety:.0%} margin of safety)",
        "value":     buy_below,
        "formatted": _num(buy_below),
        "assessment": (
            f"Fair value {_num(fair_value)} × (1 − {margin_of_safety:.0%}) = {_num(buy_below)}"
            if buy_below else "Cannot calculate"
        ),
        "benchmark": f"{margin_of_safety:.0%} margin of safety means paying at most {1-margin_of_safety:.0%} of estimated fair value",
        "detail":    "Provides a buffer for DCF estimation errors and unexpected downturns",
    }

    # Current price vs fair value
    price = data.get("current_price")
    if price and fair_value:
        upside = (fair_value - price) / price
        if buy_below:
            if price <= buy_below:
                zone = f"IN BUY ZONE (price {_num(price)} ≤ buy-below {_num(buy_below)})"
            elif price <= fair_value:
                zone = f"WATCHLIST (above buy-below {_num(buy_below)}, below fair value {_num(fair_value)})"
            else:
                zone = f"ABOVE FAIR VALUE (overvalued by {abs(upside):.1%})"
        else:
            zone = "N/A"

        metrics["price_vs_fair_value"] = {
            "label":     "Current Price vs Fair Value",
            "value":     upside,
            "formatted": f"{upside:+.1%}",
            "assessment": zone,
            "benchmark": "Negative = overvalued vs DCF, positive = undervalued",
            "detail": (
                f"Current price: {_num(price)}  "
                f"| Fair value: {_num(fair_value)}  "
                f"| Buy-below: {_num(buy_below)}  "
                f"| Upside to fair value: {upside:+.1%}"
            ),
        }

        if upside < -0.30:
            flags.append(f"Current price is {abs(upside):.0%} above estimated fair value")
    else:
        metrics["price_vs_fair_value"] = {
            "label": "Current Price vs Fair Value", "value": None,
            "formatted": "N/A", "assessment": "Price or fair value unavailable",
            "benchmark": "", "detail": "",
        }

    # ── Historical P/E range ──────────────────────────────────────────────
    pe_hist = _pe_history_position(data)
    if pe_hist:
        metrics["pe_historical_range"] = {
            "label":     "P/E vs Own 5Y History",
            "value":     pe_hist["percentile"],
            "formatted": f"{pe_hist['percentile']:.0%} percentile",
            "assessment": pe_hist["assessment"],
            "benchmark": "Bottom quartile = cheap vs own history, top = expensive",
            "detail": (
                f"Current P/E: {pe_hist['current_pe']:.1f}x  "
                f"| 5Y range: {pe_hist['pe_5y_min']:.1f}x – {pe_hist['pe_5y_max']:.1f}x  "
                f"| 5Y median: {pe_hist['pe_5y_median']:.1f}x"
            ),
        }
    else:
        metrics["pe_historical_range"] = {
            "label": "P/E vs Own 5Y History", "value": None,
            "formatted": "N/A",
            "assessment": "Insufficient P/E history data",
            "benchmark": "", "detail": "",
        }

    # ── Current multiples snapshot ────────────────────────────────────────
    pe      = data.get("trailing_pe")
    fwd_pe  = data.get("forward_pe")
    ev_ebit = data.get("ev_ebit")
    ev_fcf  = data.get("ev_fcf")
    ev_ebitda = data.get("ev_ebitda")
    pb      = data.get("pb_ratio")
    peg     = data.get("peg_ratio")
    fcf_yield = data.get("fcf_yield")

    metrics["trailing_pe"] = {
        "label":     "Trailing P/E",
        "value":     pe,
        "formatted": _x(pe),
        "assessment": _pe_assessment(pe),
        "benchmark": "Market avg ~20x; <15x cheap, >30x expensive (context-dependent)",
        "detail":    f"Forward P/E: {_x(fwd_pe)}",
    }

    metrics["ev_ebit"] = {
        "label":     "EV / EBIT",
        "value":     ev_ebit,
        "formatted": _x(ev_ebit),
        "assessment": _ev_ebit_assessment(ev_ebit),
        "benchmark": "<10x = cheap, 10–20x = fair, >25x = expensive",
        "detail":    "Enterprise value / EBIT — better than P/E for leverage comparison",
    }

    metrics["ev_ebitda"] = {
        "label":     "EV / EBITDA",
        "value":     ev_ebitda,
        "formatted": _x(ev_ebitda),
        "assessment": _ev_ebitda_assessment(ev_ebitda),
        "benchmark": "<8x = cheap, 8–15x = fair, >20x = expensive",
        "detail":    "Widely used cross-sector comparison multiple",
    }

    metrics["ev_fcf"] = {
        "label":     "EV / FCF",
        "value":     ev_fcf,
        "formatted": _x(ev_fcf),
        "assessment": _ev_fcf_assessment(ev_fcf),
        "benchmark": "<15x = cheap, 15–25x = fair, >30x = expensive",
        "detail":    "Most cash-flow-faithful multiple — immune to D&A manipulation",
    }

    metrics["fcf_yield"] = {
        "label":     "FCF Yield",
        "value":     fcf_yield,
        "formatted": _pct(fcf_yield),
        "assessment": (
            f"High yield ({_pct(fcf_yield)}) — significant free cash return vs market cap"  if fcf_yield and fcf_yield > 0.06 else
            f"Moderate yield ({_pct(fcf_yield)})"   if fcf_yield and fcf_yield > 0.03 else
            f"Low yield ({_pct(fcf_yield)})"         if fcf_yield and fcf_yield > 0 else
            "Negative FCF yield — burning cash"      if fcf_yield and fcf_yield <= 0 else
            "No data"
        ),
        "benchmark": ">5% FCF yield = attractive; <2% = priced for high growth",
        "detail":    "FCF / market cap — think of it like an earnings yield but cash-based",
    }

    metrics["pb_ratio"] = {
        "label":     "Price / Book",
        "value":     pb,
        "formatted": _x(pb),
        "assessment": (
            f"Below book value ({_x(pb)}) — potential deep value or distress"   if pb is not None and pb < 1.0 else
            f"Near book ({_x(pb)}) — modest premium"                             if pb is not None and pb < 2.0 else
            f"Moderate premium ({_x(pb)})"                                        if pb is not None and pb < 4.0 else
            f"High premium ({_x(pb)}) — market pricing in significant intangibles/growth"
            if pb is not None else "No data"
        ),
        "benchmark": "<1x = asset play or distress; most relevant for banks/industrials",
        "detail":    "PEG ratio: " + _x(peg),
    }

    # ── ROIC vs WACC spread ───────────────────────────────────────────────
    roic_spread = _roic_wacc_spread(data, wacc)
    if roic_spread.get("available"):
        spread = roic_spread["spread"]
        metrics["roic_wacc_spread"] = {
            "label":     "ROIC vs WACC Spread",
            "value":     spread,
            "formatted": f"{spread:+.1%}",
            "assessment": roic_spread["verdict"],
            "benchmark": ">3% spread = durable moat; <0% = capital destruction",
            "detail": (
                f"ROIC: {_pct(roic_spread['roic'])}  "
                f"| WACC: {_pct(roic_spread['wacc'])}  "
                f"| Spread: {spread:+.1%}"
            ),
        }
        if roic_spread["flag"]:
            flags.append(f"ROIC ({_pct(roic_spread['roic'])}) below WACC ({_pct(roic_spread['wacc'])}) — destroying shareholder value")
    else:
        metrics["roic_wacc_spread"] = {
            "label": "ROIC vs WACC Spread", "value": None,
            "formatted": "N/A", "assessment": "ROIC data unavailable",
            "benchmark": "", "detail": "",
        }

    # ── Sector multiples comparison ───────────────────────────────────────
    sector_comp = _sector_comparison(data)

    # ── Terminal value sensitivity grid ───────────────────────────────────
    tv_sens = _tv_sensitivity(data, wacc)

    # ── Reverse DCF ───────────────────────────────────────────────────────
    rdcf = _reverse_dcf(data, wacc)
    if rdcf:
        metrics["reverse_dcf"] = {
            "label":     "Reverse DCF — Implied Growth",
            "value":     rdcf["implied_growth"],
            "formatted": _pct(rdcf["implied_growth"]),
            "assessment": rdcf["assessment"],
            "benchmark": "Compare implied CAGR to trailing growth — gap shows market expectations",
            "detail": (
                f"Trailing 5Y revenue CAGR: {_pct(rdcf['trailing_growth'])}  "
                f"| Price used: {_num(rdcf['current_price'])}"
            ),
        }
    else:
        metrics["reverse_dcf"] = {
            "label": "Reverse DCF — Implied Growth", "value": None,
            "formatted": "N/A", "assessment": "Insufficient data",
            "benchmark": "", "detail": "",
        }

    return {
        "valuation_metrics": metrics,
        "valuation_flags":   flags,
        "scenarios":         scenarios,
        "wacc_data":         wacc_data,
        "fair_value_weighted": fair_value,
        "buy_below_price":   buy_below,
        "sector_comparison": sector_comp,
        "tv_sensitivity":    tv_sens,
    }


# ── assessment helpers ─────────────────────────────────────────────────────────

def _pe_assessment(pe):
    if pe is None:        return "No data"
    if pe < 0:            return "Negative earnings — P/E not meaningful"
    if pe < 10:           return f"Very low ({pe:.1f}x) — cheap or value trap"
    if pe < 15:           return f"Low ({pe:.1f}x) — below market average"
    if pe < 22:           return f"Moderate ({pe:.1f}x) — around market average"
    if pe < 30:           return f"Elevated ({pe:.1f}x) — premium to market"
    if pe < 45:           return f"High ({pe:.1f}x) — priced for strong growth"
    return                       f"Very high ({pe:.1f}x) — requires exceptional growth to justify"

def _ev_ebit_assessment(v):
    if v is None: return "No data"
    if v < 0:     return "Negative EBIT — multiple not meaningful"
    if v < 8:     return f"Very cheap ({v:.1f}x)"
    if v < 14:    return f"Cheap-to-fair ({v:.1f}x)"
    if v < 20:    return f"Fair ({v:.1f}x)"
    if v < 28:    return f"Elevated ({v:.1f}x)"
    return               f"Expensive ({v:.1f}x)"

def _ev_ebitda_assessment(v):
    if v is None: return "No data"
    if v < 0:     return "Negative EBITDA"
    if v < 6:     return f"Very cheap ({v:.1f}x)"
    if v < 10:    return f"Below average ({v:.1f}x)"
    if v < 15:    return f"Average ({v:.1f}x)"
    if v < 20:    return f"Elevated ({v:.1f}x)"
    return               f"High ({v:.1f}x)"

def _ev_fcf_assessment(v):
    if v is None: return "No data"
    if v < 0:     return "Negative FCF"
    if v < 12:    return f"Very cheap ({v:.1f}x)"
    if v < 18:    return f"Attractive ({v:.1f}x)"
    if v < 25:    return f"Fair ({v:.1f}x)"
    if v < 35:    return f"Elevated ({v:.1f}x)"
    return               f"Expensive ({v:.1f}x)"


# ── display helper ────────────────────────────────────────────────────────────

def _print_tv_sensitivity(tv_sens: dict):
    """Print terminal value sensitivity grid."""
    if not tv_sens:
        return
    print(f"\n  Terminal Value Sensitivity  (Base-case per-share value by WACC × Terminal Growth)")
    print(f"  {'WACC \\ TG':<12}", end="")
    for tg in tv_sens["tg_steps"]:
        print(f"  {tg:.1%}   ", end="")
    print()
    print(f"  {'─' * 68}")
    for i, wc in enumerate(tv_sens["wacc_steps"]):
        marker = " ◄" if abs(wc - tv_sens["base_wacc"]) < 0.001 else ""
        print(f"  {wc:.1%}{marker:<7}", end="")
        for val in tv_sens["grid"][i]:
            cell = _num(val) if val else " N/A "
            print(f"  {cell:<8}", end="")
        print()


def _print_sector_comparison(sector_comp: dict):
    """Print sector benchmark multiples comparison."""
    if not sector_comp or not sector_comp.get("comparisons"):
        return
    print(f"\n  Sector Multiples Comparison  (vs {sector_comp['sector']} median)")
    print(f"  {'Multiple':<14} {'Current':>9}  {'Sector':>9}  {'vs Sector':>10}  {'Signal'}")
    print(f"  {'─' * 60}")
    for c in sector_comp["comparisons"]:
        prem = c["premium"]
        if c["label"] == "FCF Yield":
            signal = "CHEAPER"  if prem > 0.10 else ("cheaper" if prem > 0 else ("pricier" if prem > -0.10 else "PRICIER"))
            cur_fmt  = _pct(c["current"])
            bench_fmt = _pct(c["benchmark"])
        else:
            signal = "CHEAPER"  if prem < -0.15 else ("cheaper" if prem < 0 else ("pricier" if prem < 0.15 else "PRICIER"))
            cur_fmt  = f"{c['current']:.1f}x"
            bench_fmt = f"{c['benchmark']:.1f}x"
        print(f"  {c['label']:<14} {cur_fmt:>9}  {bench_fmt:>9}  {prem:>+9.0%}  {signal}")


def print_valuation(result: dict, ticker: str = ""):
    header = f"VALUATION — {ticker}" if ticker else "VALUATION"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    for key, m in result["valuation_metrics"].items():
        if m["formatted"]:
            print(f"  {m['label']:<42} {m['formatted']:<12}  {m['assessment']}")
        else:
            print(f"  {m['label']:<42} {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")

    print_scenarios(result["scenarios"], ticker)
    _print_tv_sensitivity(result.get("tv_sensitivity"))
    _print_sector_comparison(result.get("sector_comparison"))

    if result["valuation_flags"]:
        print(f"\n  ⚠  Valuation flags:")
        for f in result["valuation_flags"]:
            print(f"     • {f}")


# ── quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "ticker": "SAMPLE.CO",
        "revenue": 30e9, "revenue_cagr_5y": 0.08,
        "operating_margin": 0.25, "effective_tax_rate": 0.22,
        "capex_pct_revenue": 0.04,
        "net_debt": -5e9, "shares_outstanding": 400e6,
        "market_cap": 120e9, "total_debt": 3e9,
        "interest_expense": 120e6, "beta": 1.1,
        "current_price": 300.0, "trailing_pe": 28.0, "forward_pe": 24.0,
        "pe_5y_min": 18.0, "pe_5y_max": 40.0, "pe_5y_median": 26.0,
        "ev_ebit": 20.0, "ev_ebitda": 16.0, "ev_fcf": 22.0,
        "pb_ratio": 8.5, "peg_ratio": 2.8,
        "fcf_yield": 0.035, "enterprise_value": 118e9, "ebitda": 7.5e9,
    }
    result = analyze_valuation(sample, margin_of_safety=0.25)
    print_valuation(result, "SAMPLE.CO")
