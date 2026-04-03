"""
explanation_engine.py  —  Plain-language narrative generation

generate_explanation(record) → dict with:
  - paragraphs:  list of (heading, text) tuples in reading order
  - one_liner:   single sentence summary for quick scanning

Template-based — no LLM needed. Every sentence references a computed number.
"""

from utils import _pct, _num, _x, _bn


def _or(value, fallback: str = "not available") -> str:
    return str(value) if value is not None else fallback


def _growth_word(cagr) -> str:
    if cagr is None: return "an unknown rate"
    if cagr > 0.20:  return f"exceptional growth ({_pct(cagr)}/yr)"
    if cagr > 0.12:  return f"strong growth ({_pct(cagr)}/yr)"
    if cagr > 0.06:  return f"solid growth ({_pct(cagr)}/yr)"
    if cagr > 0.01:  return f"modest growth ({_pct(cagr)}/yr)"
    if cagr >= 0:    return f"near-flat revenue ({_pct(cagr)}/yr)"
    return                  f"revenue contraction ({_pct(cagr)}/yr)"


def _margin_word(margin) -> str:
    if margin is None: return "unknown"
    if margin > 0.30:  return f"high ({_pct(margin)})"
    if margin > 0.18:  return f"solid ({_pct(margin)})"
    if margin > 0.08:  return f"moderate ({_pct(margin)})"
    if margin > 0.0:   return f"thin ({_pct(margin)})"
    return                    f"negative ({_pct(margin)})"


def _leverage_word(nd_ebitda) -> str:
    if nd_ebitda is None:  return "leverage is unknown"
    if nd_ebitda < 0:      return f"a net cash position (Net Debt/EBITDA {nd_ebitda:.1f}x)"
    if nd_ebitda < 1.0:    return f"minimal debt (Net Debt/EBITDA {nd_ebitda:.1f}x)"
    if nd_ebitda < 2.5:    return f"manageable leverage (Net Debt/EBITDA {nd_ebitda:.1f}x)"
    if nd_ebitda < 4.0:    return f"elevated leverage (Net Debt/EBITDA {nd_ebitda:.1f}x)"
    return                        f"high leverage (Net Debt/EBITDA {nd_ebitda:.1f}x)"


def generate_explanation(record: dict) -> dict:
    ticker   = record.get("ticker", "")
    company  = record.get("company_name") or ticker
    sector   = record.get("sector") or "an unclassified sector"
    currency = record.get("currency", "")
    price    = record.get("current_price")
    fv       = record.get("fair_value_weighted")
    bb       = record.get("buy_below_price")
    clf      = (record.get("classification_result") or {}).get("classification", "N/A")

    rev_cagr  = record.get("revenue_cagr_5y")
    op_margin = record.get("operating_margin")
    gm        = record.get("gross_margin")
    net_margin = record.get("net_margin")
    roic      = record.get("roic")
    wacc      = (record.get("wacc_data") or {}).get("wacc")
    spread    = (roic - wacc) if (roic is not None and wacc is not None) else None
    nd_ebitda = record.get("net_debt_ebitda")
    coverage  = record.get("interest_coverage")
    fcf_yield = record.get("fcf_yield")
    rev       = record.get("revenue")
    dq        = record.get("data_quality_score", 0)

    red_flags  = record.get("red_flags") or []
    high_flags = [f for f in red_flags if f["severity"] == "HIGH"]
    crit_flags = record.get("critical_flags") or []

    clf_result    = record.get("classification_result") or {}
    key_opp       = clf_result.get("key_opportunity", "")
    key_risk      = clf_result.get("key_risk", "")
    reasons_for   = clf_result.get("reasons_for") or []
    reasons_against = clf_result.get("reasons_against") or []

    paragraphs = []

    # ── 1. Business overview ──────────────────────────────────────────────
    rev_str = f"with revenue of {_bn(rev)}" if rev else ""
    para1 = (
        f"{company} ({ticker}) is a {sector} company {rev_str}. "
        f"Over the past five years, it has grown revenue at {_growth_word(rev_cagr)}, "
        f"with {_margin_word(op_margin)} operating margins"
    )
    if gm is not None:
        para1 += f" on top of a gross margin of {_pct(gm)}"
    para1 += "."
    paragraphs.append(("Business Overview", para1))

    # ── 2. Financial health ───────────────────────────────────────────────
    coverage_str = (
        f"Interest is covered {coverage:.1f}x by EBIT. " if coverage is not None else ""
    )
    roic_str = ""
    if roic is not None and wacc is not None:
        if spread > 0:
            roic_str = (
                f"The business earns a return on invested capital of {_pct(roic)}, "
                f"which is {abs(spread):.1%} above its cost of capital ({_pct(wacc)}) — "
                f"a sign of genuine value creation. "
            )
        else:
            roic_str = (
                f"ROIC of {_pct(roic)} falls short of the {_pct(wacc)} cost of capital "
                f"by {abs(spread):.1%} — the business is currently destroying economic value. "
            )

    para2 = (
        f"{company} carries {_leverage_word(nd_ebitda)}. "
        f"{coverage_str}"
        f"{roic_str}"
    )
    if fcf_yield is not None:
        if fcf_yield > 0.04:
            para2 += f"Free cash flow yield of {_pct(fcf_yield)} suggests strong cash generation relative to market cap."
        elif fcf_yield > 0:
            para2 += f"FCF yield of {_pct(fcf_yield)} is positive but modest."
        else:
            para2 += f"FCF yield is negative ({_pct(fcf_yield)}), indicating the business is consuming more cash than it generates."
    paragraphs.append(("Financial Health", para2.strip()))

    # ── 3. Valuation ──────────────────────────────────────────────────────
    if fv and price:
        upside = (fv - price) / price
        if upside > 0.15:
            val_verdict = (
                f"At {currency} {_num(price)}, the stock trades {upside:.0%} below our DCF "
                f"fair value estimate of {_num(fv)}. "
            )
        elif upside > -0.10:
            val_verdict = (
                f"At {currency} {_num(price)}, the stock is broadly in line with our DCF "
                f"fair value estimate of {_num(fv)} ({upside:+.0%}). "
            )
        else:
            val_verdict = (
                f"At {currency} {_num(price)}, the stock appears {abs(upside):.0%} above our DCF "
                f"fair value estimate of {_num(fv)} — suggesting limited margin of safety. "
            )
        val_verdict += (
            f"A 25% margin of safety implies a buy-below price of {_num(bb)}. "
            if bb else ""
        )
    else:
        val_verdict = f"Insufficient data to generate a DCF fair value. Use multiples-based valuation as a cross-check. "

    # Add reverse DCF note
    rdcf_metric = (record.get("valuation_metrics") or {}).get("reverse_dcf", {})
    implied_g   = rdcf_metric.get("value")
    if implied_g is not None and rev_cagr is not None:
        if implied_g > rev_cagr * 1.3:
            val_verdict += (
                f"The reverse DCF suggests the market is pricing in {_pct(implied_g)} revenue CAGR — "
                f"significantly above the trailing {_pct(rev_cagr)}, embedding optimistic assumptions."
            )
        elif implied_g < rev_cagr * 0.8:
            val_verdict += (
                f"The reverse DCF implies only {_pct(implied_g)} CAGR is needed to justify the current price — "
                f"below the trailing {_pct(rev_cagr)}, suggesting conservative market pricing."
            )

    paragraphs.append(("Valuation", val_verdict.strip()))

    # ── 4. Key risks ──────────────────────────────────────────────────────
    risk_parts = []
    if crit_flags:
        risk_parts.append(
            f"Critical concerns include: {'; '.join(crit_flags[:2])}."
        )
    if high_flags:
        names = ", ".join(f["pattern"] for f in high_flags[:3])
        risk_parts.append(
            f"The pattern-detection system flagged {len(high_flags)} high-severity issue(s): {names}."
        )
    if key_risk and key_risk != "No major risk factors detected":
        risk_parts.append(key_risk + ".")

    if not risk_parts:
        risk_parts.append("No material risk flags were detected across the 10-pattern screening.")

    paragraphs.append(("Key Risks", " ".join(risk_parts)))

    # ── 5. Timing / Momentum signal ───────────────────────────────────────
    trend         = record.get("momentum_trend", "UNKNOWN")
    mom_metrics   = record.get("momentum_metrics") or {}
    rsi_val       = (mom_metrics.get("rsi") or {}).get("value")
    ret_3m_val    = (mom_metrics.get("return_3m") or {}).get("value")
    range_pct     = (mom_metrics.get("range_52w") or {}).get("value")

    timing_parts = []
    if trend == "UPTREND":
        timing_parts.append("Price trend is bullish — MA50 is above MA200.")
    elif trend == "DOWNTREND":
        timing_parts.append("Price trend is bearish (Death Cross) — consider waiting for trend stabilisation before entry.")
    elif trend == "SIDEWAYS":
        timing_parts.append("Price is moving sideways — no clear directional trend.")
    else:
        timing_parts.append("Insufficient price history for moving average trend assessment.")

    if rsi_val is not None:
        if rsi_val > 70:
            timing_parts.append(f"RSI is overbought at {rsi_val:.0f} — near-term pullback risk is elevated.")
        elif rsi_val < 30:
            timing_parts.append(f"RSI is oversold at {rsi_val:.0f} — potential tactical entry opportunity.")
        else:
            timing_parts.append(f"RSI is neutral at {rsi_val:.0f}.")

    if ret_3m_val is not None:
        if ret_3m_val > 0.10:
            timing_parts.append(f"3-month momentum is strong ({ret_3m_val:+.1%}), supporting near-term continuation.")
        elif ret_3m_val < -0.10:
            timing_parts.append(f"3-month momentum is weak ({ret_3m_val:+.1%}) — caution on near-term entry.")

    if range_pct is not None:
        if range_pct > 0.85:
            timing_parts.append(f"Stock is near its 52-week high ({range_pct:.0%} of range) — limited near-term technical upside.")
        elif range_pct < 0.20:
            timing_parts.append(f"Stock is near its 52-week low ({range_pct:.0%} of range) — either value opportunity or ongoing weakness.")

    paragraphs.append(("Timing Signal", " ".join(timing_parts)))

    # ── 6. Verdict ────────────────────────────────────────────────────────
    verdict_map = {
        "STRONG BUY":   "a compelling investment opportunity",
        "BUY":          "an attractive investment at current levels",
        "WATCHLIST":    "a stock to monitor but not yet at an attractive entry",
        "HOLD":         "a hold — fairly valued with no strong directional signal",
        "AVOID":        "a stock to avoid at current levels",
        "STRONG AVOID": "a stock to avoid — significant fundamental concerns",
    }
    verdict_desc = verdict_map.get(clf, "an uncertain investment")

    para5 = (
        f"Overall, {company} is classified as {verdict_desc} ({clf}). "
    )
    if reasons_for:
        para5 += f"The primary positive is: {reasons_for[0].rstrip('.')}. "
    if reasons_against:
        para5 += f"The main concern is: {reasons_against[0].rstrip('.')}. "

    if dq < 60:
        para5 += (
            f"Note: data quality score is low ({dq}/100) — some inputs may be incomplete "
            f"and conclusions should be treated with additional caution."
        )
    paragraphs.append(("Verdict", para5.strip()))

    # ── One-liner ─────────────────────────────────────────────────────────
    one_liner = (
        f"{company} ({ticker}): {clf} — "
        f"{_growth_word(rev_cagr)}, {_margin_word(op_margin)} margins, "
        f"{_leverage_word(nd_ebitda)}"
        + (f", {upside:+.0%} DCF upside" if (fv and price) else "")
        + "."
    )

    return {
        "paragraphs": paragraphs,
        "one_liner":  one_liner,
    }


# ── display helper ─────────────────────────────────────────────────────────────

def print_explanation(result: dict, ticker: str = ""):
    header = f"NARRATIVE SUMMARY — {ticker}" if ticker else "NARRATIVE SUMMARY"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    print(f"  {result['one_liner']}")

    for heading, text in result["paragraphs"]:
        print(f"\n  [{heading}]")
        # Wrap text at ~68 chars
        words  = text.split()
        line   = "  "
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)
