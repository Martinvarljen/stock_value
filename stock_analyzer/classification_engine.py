"""
classification_engine.py  —  Final stock classification

classify_stock(record) → dict with:
  - classification:   STRONG BUY / BUY / WATCHLIST / HOLD / AVOID / STRONG AVOID
  - reasons_for:      list of specific, number-backed reasons supporting a buy
  - reasons_against:  list of specific risks / negatives
  - key_opportunity:  single most important positive
  - key_risk:         single most important risk

No arbitrary point system. Every reason cites a real number.
Hard overrides (critical flags, extreme overvaluation) take precedence over
soft factors.  Valuation (DCF upside) is the primary tiebreaker.
"""

from utils import _pct, _num, _x


# ── helpers ────────────────────────────────────────────────────────────────────

def _metric_val(record: dict, engine_key: str, metric_key: str):
    """Safely pull .value from a metric block inside an engine result dict."""
    return (record.get(engine_key) or {}).get(metric_key, {}).get("value")


# ── main function ──────────────────────────────────────────────────────────────

def classify_stock(record: dict) -> dict:
    """
    Synthesises all engine results into a single classification.
    Returns a dict with classification label, supporting/opposing reasons,
    and the single most important opportunity and risk.
    """
    reasons_for     = []
    reasons_against = []

    price          = record.get("current_price")
    fair_value     = record.get("fair_value_weighted")
    buy_below      = record.get("buy_below_price")
    critical_flags = record.get("critical_flags") or []
    red_flags      = record.get("red_flags") or []
    high_rf        = [f for f in red_flags if (f.get("severity") == "HIGH")]
    medium_rf      = [f for f in red_flags if (f.get("severity") == "MEDIUM")]

    upside = None
    if price and fair_value and price > 0:
        upside = (fair_value - price) / price

    # ── 1. DCF valuation vs price ─────────────────────────────────────────
    if upside is not None:
        if upside > 0.35:
            reasons_for.append(
                f"Large DCF margin of safety: price {_num(price)} vs fair value {_num(fair_value)} "
                f"({upside:.0%} upside)"
            )
        elif upside > 0.15:
            reasons_for.append(
                f"Price below DCF fair value by {upside:.0%} "
                f"({_num(price)} vs {_num(fair_value)})"
            )
        elif upside > 0:
            pass  # trivial upside — neutral
        elif upside > -0.15:
            reasons_against.append(
                f"Price slightly above DCF fair value "
                f"({_num(price)} vs est. {_num(fair_value)}, {upside:.0%})"
            )
        elif upside > -0.30:
            reasons_against.append(
                f"Price {abs(upside):.0%} above DCF fair value "
                f"({_num(price)} vs est. {_num(fair_value)})"
            )
        else:
            reasons_against.append(
                f"Significantly overvalued: price {abs(upside):.0%} above DCF fair value "
                f"({_num(price)} vs est. {_num(fair_value)})"
            )

    if buy_below and price and price <= buy_below:
        reasons_for.append(
            f"Trading below buy-below price ({_num(price)} ≤ {_num(buy_below)}) — "
            f"within 25% margin-of-safety zone"
        )

    # ── 2. ROIC vs WACC spread ────────────────────────────────────────────
    spread = _metric_val(record, "valuation_metrics", "roic_wacc_spread")
    roic   = record.get("roic")
    wacc   = (record.get("wacc_data") or {}).get("wacc")
    if spread is not None:
        if spread > 0.08:
            reasons_for.append(
                f"Strong economic moat: ROIC/WACC spread = {spread:+.1%} "
                f"(ROIC {_pct(roic)} vs WACC {_pct(wacc)})"
            )
        elif spread > 0.03:
            reasons_for.append(
                f"Value-creating business: ROIC/WACC spread = {spread:+.1%}"
            )
        elif spread < -0.05:
            reasons_against.append(
                f"Capital destruction: ROIC ({_pct(roic)}) trails WACC ({_pct(wacc)}) "
                f"by {abs(spread):.1%}"
            )
        elif spread < 0:
            reasons_against.append(
                f"Marginal capital destruction: ROIC/WACC spread = {spread:+.1%}"
            )

    # ── 3. Reverse DCF — implied vs trailing growth ───────────────────────
    implied_growth  = _metric_val(record, "valuation_metrics", "reverse_dcf")
    trailing_growth = record.get("revenue_cagr_5y")
    if implied_growth is not None and trailing_growth is not None:
        if implied_growth < trailing_growth * 0.75 and implied_growth >= 0:
            reasons_for.append(
                f"Market only pricing in {_pct(implied_growth)} revenue CAGR vs "
                f"trailing {_pct(trailing_growth)} — conservative expectations"
            )
        elif implied_growth > trailing_growth * 1.5 and implied_growth > 0.12:
            reasons_against.append(
                f"Market pricing in {_pct(implied_growth)} revenue CAGR vs trailing "
                f"{_pct(trailing_growth)} — aggressive growth expectations baked in"
            )
        elif implied_growth < 0:
            reasons_against.append(
                f"Market implies revenue contraction ({_pct(implied_growth)}) at current price"
            )

    # ── 3b. Sector growth sanity — “train already left” ───────────────────
    sector_ctx = record.get("sector_result") or {}
    cagr_range = sector_ctx.get("growth_cagr_range")
    if isinstance(cagr_range, (tuple, list)) and len(cagr_range) == 2:
        _, cagr_ceiling = cagr_range
        if implied_growth is not None and implied_growth > cagr_ceiling * 1.25 and implied_growth > 0.10:
            reasons_against.append(
                f"Growth expectations look stretched for the sector: reverse DCF implies {_pct(implied_growth)} "
                f"vs typical {sector_ctx.get('sector') or 'sector'} ceiling ~{_pct(cagr_ceiling)}"
            )

    # ── 4. Revenue growth ─────────────────────────────────────────────────
    rev_cagr = record.get("revenue_cagr_5y")
    if rev_cagr is not None:
        if rev_cagr > 0.15:
            reasons_for.append(f"High revenue growth: {_pct(rev_cagr)} 5Y CAGR")
        elif rev_cagr > 0.07:
            reasons_for.append(f"Solid revenue growth: {_pct(rev_cagr)} 5Y CAGR")
        elif rev_cagr < -0.02:
            reasons_against.append(f"Revenue declining: {_pct(rev_cagr)} 5Y CAGR")

    # ── 5. Financial strength ─────────────────────────────────────────────
    nd_ebitda = record.get("net_debt_ebitda")
    if nd_ebitda is not None:
        if nd_ebitda < 0:
            reasons_for.append(f"Net cash position (Net Debt/EBITDA = {nd_ebitda:.1f}x)")
        elif nd_ebitda < 1.5:
            reasons_for.append(f"Conservative leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x")
        elif nd_ebitda > 4.5:
            reasons_against.append(f"High leverage: Net Debt/EBITDA = {nd_ebitda:.1f}x")

    coverage = record.get("interest_coverage")
    if coverage is not None:
        if coverage > 15:
            reasons_for.append(f"Very strong interest coverage: {coverage:.1f}x")
        elif coverage < 2.0:
            reasons_against.append(f"Thin interest coverage: {coverage:.1f}x")

    # ── 6. Critical flags (hard negatives) ───────────────────────────────
    for cf in critical_flags:
        reasons_against.append(f"Critical flag: {cf}")

    # ── 7. High red flags ─────────────────────────────────────────────────
    for f in high_rf:
        reasons_against.append(f"High-severity pattern: {f['pattern']}")

    # ── 8. Quality — operating margin ─────────────────────────────────────
    op_margin = record.get("operating_margin")
    if op_margin is not None:
        if op_margin > 0.25:
            reasons_for.append(f"High operating margin: {_pct(op_margin)}")
        elif op_margin < 0:
            reasons_against.append(f"Negative operating margin: {_pct(op_margin)}")

    # ── 9. FCF yield ──────────────────────────────────────────────────────
    fcf_yield = record.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield > 0.06:
            reasons_for.append(f"Attractive FCF yield: {_pct(fcf_yield)}")
        elif fcf_yield < 0:
            reasons_against.append(f"Negative FCF yield — burning cash")

    # ── Derive classification ─────────────────────────────────────────────
    n_for     = len(reasons_for)
    n_against = len(reasons_against)

    # Hard overrides — these override everything else
    if len(critical_flags) >= 2 or (critical_flags and len(high_rf) >= 2):
        classification = "STRONG AVOID"

    elif len(critical_flags) >= 1 or len(high_rf) >= 3:
        classification = "AVOID"

    elif upside is not None and upside < -0.35:
        classification = "STRONG AVOID"

    elif upside is not None and upside < -0.20:
        classification = "AVOID"

    # Valuation-driven soft classification
    elif upside is not None and upside > 0.30 and not critical_flags and not high_rf:
        classification = "STRONG BUY"

    elif upside is not None and upside > 0.15 and not critical_flags and len(high_rf) <= 1:
        classification = "BUY"

    elif upside is not None and upside > 0 and not critical_flags:
        # Small upside — tip by fundamental quality
        if n_for > n_against + 1:
            classification = "BUY"
        else:
            classification = "WATCHLIST"

    elif upside is not None and upside > -0.20:
        classification = "HOLD"

    elif upside is not None and upside > -0.25:
        classification = "AVOID"

    else:
        # No fair value or edge cases
        if critical_flags or len(high_rf) >= 2:
            classification = "AVOID"
        elif len(high_rf) == 1 or len(medium_rf) >= 3:
            classification = "HOLD"
        elif n_for > n_against:
            classification = "WATCHLIST"
        else:
            classification = "HOLD"

    # ── Key opportunity and risk ──────────────────────────────────────────
    key_opportunity = reasons_for[0] if reasons_for else "No clear positive catalyst identified"
    key_risk        = reasons_against[0] if reasons_against else "No major risk factors detected"

    return {
        "classification":  classification,
        "reasons_for":     reasons_for,
        "reasons_against": reasons_against,
        "key_opportunity": key_opportunity,
        "key_risk":        key_risk,
    }


# ── display helper ─────────────────────────────────────────────────────────────

_LABELS = {
    "STRONG BUY":   "STRONG BUY  ▲▲",
    "BUY":          "BUY         ▲",
    "WATCHLIST":    "WATCHLIST   →",
    "HOLD":         "HOLD        —",
    "AVOID":        "AVOID       ▼",
    "STRONG AVOID": "STRONG AVOID▼▼",
}

def print_classification(result: dict, ticker: str = ""):
    header = f"CLASSIFICATION — {ticker}" if ticker else "CLASSIFICATION"
    label  = _LABELS.get(result["classification"], result["classification"])
    print(f"\n{'═' * 70}")
    print(f"  {header}")
    print(f"{'═' * 70}")
    print(f"  Verdict:  {label}")
    print()

    if result["reasons_for"]:
        print(f"  Positives:")
        for r in result["reasons_for"]:
            print(f"    +  {r}")

    if result["reasons_against"]:
        print()
        print(f"  Negatives / Risks:")
        for r in result["reasons_against"]:
            print(f"    -  {r}")

    print()
    print(f"  Key opportunity:  {result['key_opportunity']}")
    print(f"  Key risk:         {result['key_risk']}")
    print(f"{'═' * 70}")
