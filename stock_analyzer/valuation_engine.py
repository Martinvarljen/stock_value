"""
valuation_engine.py — Compact base-case DCF + WACC for the trading stack

This module replaces a 974-line single-name research artefact (3-scenario
probability-weighted DCF + sensitivity grid + reverse DCF + sector
multiple comparison + P/E history percentile + verbose prose verdicts).
For a *trading* strategy whose decisions come from a calibrated ML model,
the only valuation outputs that materially affect P&L are:

* ``wacc_data["wacc"]`` — used by ``red_flags`` (capital destruction
  check) and ``projection_engine`` quality scoring.
* ``fair_value_weighted`` — used by ``projection_engine._score_valuation``.
* ``buy_below_price`` — used by ``trade_setup_engine`` for human-readable
  watch levels in the memory log.
* ``scenarios["base"]`` — kept as a structured object so reporting code
  that walks the 3-scenario shape doesn't crash. Bear/Bull are no longer
  computed; if you want richer reporting, build it from the base case
  PV path.

Everything else (`_tv_sensitivity`, `_reverse_dcf`, `_sector_comparison`,
`_pe_history_position`, the metrics-prose dictionary, ``print_scenarios``)
was research / dashboard residue and is gone. The previous
``scenario_engine.py`` is folded inline because it had no other caller.

Constants are still hard-coded — that's a known limitation flagged in the
audit (see ``backtesting/yearly_top100_universe.py`` for the related PIT
warning). For trading-mode runs the only consumer of ``calculate_wacc``
is feature engineering, so the impact is bounded.
"""

from __future__ import annotations


# ── constants ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.035
EQUITY_RISK_PREM = 0.055
DEFAULT_COST_OF_DEBT = 0.045
DEFAULT_TAX_RATE = 0.22

CYCLICAL_SECTORS = {"Energy", "Basic Materials"}
CYCLICAL_GROWTH_FLOOR = 0.00


# ── WACC ─────────────────────────────────────────────────────────────────────

def calculate_wacc(data: dict) -> dict:
    """Returns the WACC components transparently. CAPM-based; debt cost from
    interest expense / total debt when both are known. Output is clamped
    to a 4–20% sanity band."""
    beta = data.get("beta")
    mktcap = data.get("market_cap")
    debt = data.get("total_debt") or 0.0
    int_exp = data.get("interest_expense")
    tax = data.get("effective_tax_rate") or DEFAULT_TAX_RATE

    if beta is None or beta <= 0:
        beta_used, beta_note = 1.0, "Beta unavailable - using 1.0"
    else:
        beta_used, beta_note = beta, f"Beta={beta:.2f}"

    cost_of_equity = RISK_FREE_RATE + beta_used * EQUITY_RISK_PREM

    if int_exp and int_exp != 0 and debt and debt > 0:
        cost_of_debt = abs(int_exp) / debt
        debt_note = "interest_expense/total_debt"
    else:
        cost_of_debt = DEFAULT_COST_OF_DEBT 
        debt_note = f"default {DEFAULT_COST_OF_DEBT:.1%}"
    cost_of_debt = max(0.01, min(cost_of_debt, 0.15))
    kd_after_tax = cost_of_debt * (1 - tax)

    if mktcap and mktcap > 0:
        total_capital = mktcap + debt
        w_equity = mktcap / total_capital
        w_debt = debt / total_capital
        weight_note = "market_cap/(mktcap+debt)"
    else:
        w_equity, w_debt = 0.80, 0.20
        weight_note = "80/20 fallback"

    wacc = w_equity * cost_of_equity + w_debt * kd_after_tax
    wacc = max(0.04, min(wacc, 0.20))

    return {
        "risk_free_rate": RISK_FREE_RATE,
        "equity_risk_prem": EQUITY_RISK_PREM,
        "beta": beta_used,
        "beta_note": beta_note,
        "cost_of_equity": cost_of_equity,
        "cost_of_debt": cost_of_debt,
        "kd_after_tax": kd_after_tax,
        "debt_note": debt_note,
        "w_equity": w_equity,
        "w_debt": w_debt,
        "weight_note": weight_note,
        "wacc": wacc,
        "tax_rate_used": tax,
    }


# ── ROIC vs WACC spread (single ratio; consumed by red_flags too) ────────────

def _roic_wacc_spread(data: dict, wacc: float) -> dict:
    roic = data.get("roic")
    if roic is None:
        return {"available": False}
    spread = roic - wacc
    return {
        "available": True,
        "roic": roic,
        "wacc": wacc,
        "spread": spread,
        "flag": spread < 0,
    }


# ── Base-case DCF (folds the prior scenario_engine in-line) ──────────────────

_BASE_PARAMS = {
    "label": "Base Case",
    "revenue_growth_mult": 0.85,
    "margin_mult": 1.00,
    "terminal_growth": 0.025,
}


def _decay(start: float, end: float, year: int, total_years: int) -> float:
    if total_years <= 1:
        return end
    alpha = (year - 1) / (total_years - 1)
    return start * (1 - alpha) + end * alpha


def _run_base_case_dcf(data: dict, wacc: float, years: int = 10) -> dict | None:
    """Single-scenario DCF with revenue-growth decay and margin
    mean-reversion to 85% of starting margin. The 3-scenario probability-
    weighting that the audit flagged as engineering theatre is gone — base
    case IS the fair value estimate, full stop."""
    revenue = data.get("revenue")
    shares = data.get("shares_outstanding")
    if not (revenue and revenue > 0 and shares and shares > 0 and wacc and wacc > 0):
        return None

    base_growth = data.get("revenue_cagr_5y") or 0.03
    base_margin = data.get("operating_margin") or 0.10
    tax = max(0.0, min(data.get("effective_tax_rate") or DEFAULT_TAX_RATE, 0.50))
    capex_pct = max(0.0, min(data.get("capex_pct_revenue") or 0.03, 0.30))
    net_debt = data.get("net_debt") or 0.0

    growth_start = max(min(base_growth * _BASE_PARAMS["revenue_growth_mult"], 0.40), -0.10)
    margin_start = max(min(base_margin * _BASE_PARAMS["margin_mult"], 0.60), -0.05)
    tg = _BASE_PARAMS["terminal_growth"]

    # Sector terminal-growth realism, if surfaced from sector_engine.
    tg_range = data.get("terminal_growth_range")
    if isinstance(tg_range, (tuple, list)) and len(tg_range) == 2:
        tg_floor, tg_ceiling = tg_range
        tg = max(tg_floor, min(tg, tg_ceiling))
    if tg >= wacc:
        tg = wacc - 0.005

    steady_state = max(base_margin * 0.85, 0.02)
    rev = revenue
    total_pv = 0.0
    yearly: list[dict] = []

    for y in range(1, years + 1):
        yr_growth = _decay(growth_start, tg, y, years)
        yr_margin = _decay(margin_start, steady_state, y, years)
        rev = rev * (1 + yr_growth)
        nopat = rev * yr_margin * (1 - tax)
        fcf = nopat - rev * capex_pct
        pv = fcf / (1 + wacc) ** y
        total_pv += pv
        yearly.append({"year": y, "growth": yr_growth, "margin": yr_margin,
                        "revenue": rev, "fcf": fcf, "pv_fcf": pv})

    fcf_y10 = yearly[-1]["fcf"]
    tv = fcf_y10 * (1 + tg) / (wacc - tg)
    pv_tv = tv / (1 + wacc) ** years
    ev = total_pv + pv_tv
    eq = ev - net_debt
    per_share = eq / shares if shares else None
    tv_pct = pv_tv / ev if ev > 0 else None

    return {
        "label": _BASE_PARAMS["label"],
        "growth_rate": growth_start,
        "operating_margin": margin_start,
        "steady_state_margin": steady_state,
        "terminal_growth": tg,
        "wacc": wacc,
        "yearly": yearly,
        "pv_fcfs": total_pv,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "tv_pct_of_ev": tv_pct,
        "enterprise_value": ev,
        "net_debt_used": net_debt,
        "equity_value": eq,
        "per_share_value": per_share,
    }


# ── main entry ───────────────────────────────────────────────────────────────

def analyze_valuation(
    data: dict,
    margin_of_safety: float = 0.25,
    wacc_adjustment: float = 0.0,
    terminal_growth_range: tuple[float, float] | None = None,
) -> dict:
    """Returns a flat dict matching the legacy keys consumed by
    ``stock_analyzer.pipeline``:

    * ``wacc_data``        WACC breakdown.
    * ``fair_value_weighted`` per-share base-case DCF value (the legacy
                              key name is kept for back-compat; it's no
                              longer probability-weighted).
    * ``buy_below_price``  fair_value × (1 - margin_of_safety).
    * ``scenarios``        ``{"base": <scenario dict>}`` only.
    * ``tv_sensitivity``   Always ``None`` (the grid was removed).
    * ``valuation_metrics``Empty dict — reporting prose lives in reporting/.
    * ``valuation_flags``  List of short structured flag strings.
    """
    flags: list[str] = []
    wacc_data = calculate_wacc(data)
    if wacc_adjustment:
        raw_wacc = wacc_data["wacc"]
        adjusted = max(0.04, min(raw_wacc + wacc_adjustment, 0.20))
        wacc_data["wacc"] = adjusted
        wacc_data["sector_adjustment"] = wacc_adjustment
        wacc_data["wacc_before_adj"] = raw_wacc
    wacc = wacc_data["wacc"]

    sector = data.get("sector") or ""
    dcf_data = dict(data)
    if terminal_growth_range and isinstance(terminal_growth_range, (tuple, list)):
        dcf_data["terminal_growth_range"] = terminal_growth_range

    skip_dcf = False
    # DCF is structurally unreliable for banks/insurers — same skip rule
    # as the legacy module.
    if sector == "Financial Services":
        skip_dcf = True
        flags.append("dcf_skipped:financial_services")

    # Cyclical-sector mid-cycle normalisation.
    if sector in CYCLICAL_SECTORS:
        rev_5y = [r for r in (data.get("revenue_5y") or []) if r is not None and r > 0]
        trailing_growth = data.get("revenue_cagr_5y") or 0.0
        if len(rev_5y) >= 3:
            dcf_data["revenue"] = sum(rev_5y) / len(rev_5y)
            flags.append(f"cyclical_norm:revenue=mid_cycle_5y_avg")
        dcf_data["revenue_cagr_5y"] = max(trailing_growth, CYCLICAL_GROWTH_FLOOR)
        net_cx = data.get("net_capex_pct_revenue")
        if net_cx is not None:
            dcf_data["capex_pct_revenue"] = net_cx

    base = None if skip_dcf else _run_base_case_dcf(dcf_data, wacc)
    fair_value = (base or {}).get("per_share_value") if base else None
    buy_below = fair_value * (1 - margin_of_safety) if fair_value else None

    if base is not None:
        tv_pct = base.get("tv_pct_of_ev")
        if tv_pct is not None and tv_pct > 0.80:
            flags.append(f"dcf_tv_dependency_high:{tv_pct:.0%}")

    spread = _roic_wacc_spread(data, wacc)
    # ``valuation_metrics`` is the legacy block consumed by
    # ``classification_engine._metric_val`` (it expects a nested
    # ``{"key": {"value": x}}`` shape). We populate just enough to keep the
    # ROIC/WACC bullet alive in classification; reverse-DCF and sector-PE
    # narrative paths gracefully degrade to "no bullet" when absent.
    val_metrics: dict = {}
    if spread.get("available"):
        val_metrics["roic_wacc_spread"] = {"value": spread.get("spread")}

    return {
        "wacc_data": wacc_data,
        "fair_value_weighted": fair_value,
        "buy_below_price": buy_below,
        "scenarios": {"base": base} if base else {},
        "tv_sensitivity": None,
        "valuation_metrics": val_metrics,
        "valuation_flags": flags,
        "roic_wacc_spread": spread,
    }
