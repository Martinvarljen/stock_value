"""
backtest_engine.py  —  Historical price performance analysis

analyze_price_history(data) → dict with:
  - period_returns:    1Y, 3Y, 5Y total price returns
  - annualised:        annualised returns for each period
  - max_drawdown:      maximum peak-to-trough decline over 5Y monthly series
  - volatility_annual: annualised monthly return volatility (5Y)
  - sharpe_approx:     (annualised return - 3.5% risk-free) / volatility
  - up_months / down_months: count over 5Y
  - price_vs_fundamentals:  price CAGR vs revenue CAGR vs EPS CAGR
  - consistency:       how often the stock beats a simple 5%/yr hurdle

No extra API calls — uses close_1y (daily) and close_5y_monthly already fetched.
"""

import math
from utils import _pct

RISK_FREE = 0.035   # same as valuation engine


# ── helpers ────────────────────────────────────────────────────────────────────

def _valid(lst):
    return [v for v in (lst or []) if v is not None and v > 0]


def _annualise(total_return: float, years: float) -> float | None:
    if years <= 0 or total_return <= -1:
        return None
    return (1 + total_return) ** (1 / years) - 1


def _max_drawdown(prices: list) -> float | None:
    """Maximum peak-to-trough percentage decline."""
    if len(prices) < 2:
        return None
    peak = prices[0]
    max_dd = 0.0
    for p in prices[1:]:
        if p > peak:
            peak = p
        dd = (p - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _monthly_returns(prices: list) -> list:
    rets = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            rets.append((prices[i] - prices[i - 1]) / prices[i - 1])
    return rets


def _stdev(values: list) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


# ── main function ──────────────────────────────────────────────────────────────

def analyze_price_history(data: dict) -> dict:
    metrics = {}
    flags   = []

    price        = data.get("current_price")
    close_5y_mo  = _valid(data.get("close_5y_monthly") or [])
    return_1y    = data.get("return_1y")
    return_3y    = data.get("return_3y")
    rev_cagr_5y  = data.get("revenue_cagr_5y")

    # ── Period returns ────────────────────────────────────────────────────
    ret_5y = None
    if price and len(close_5y_mo) >= 12:
        p_5y_ago = close_5y_mo[0]
        if p_5y_ago and p_5y_ago > 0:
            ret_5y = (price - p_5y_ago) / p_5y_ago

    ann_1y = return_1y   # already 1-year, no annualisation needed
    ann_3y = _annualise(return_3y, 3) if return_3y is not None else None
    ann_5y = _annualise(ret_5y,    5) if ret_5y    is not None else None

    def _ret_note(total, ann):
        if total is None:
            return "No data"
        gain = "gain" if total >= 0 else "loss"
        ann_str = f" ({ann:+.1%}/yr annualised)" if ann is not None else ""
        return f"{total:+.1%} total {gain}{ann_str}"

    metrics["return_1y"] = {
        "label":     "1-Year Price Return",
        "value":     return_1y,
        "formatted": f"{return_1y:+.1%}" if return_1y is not None else "N/A",
        "assessment": _ret_note(return_1y, ann_1y),
        "benchmark": "S&P 500 long-run average ~10%/yr; MSCI World ~8%/yr",
        "detail":    "",
    }
    metrics["return_3y"] = {
        "label":     "3-Year Price Return",
        "value":     return_3y,
        "formatted": f"{return_3y:+.1%}" if return_3y is not None else "N/A",
        "assessment": _ret_note(return_3y, ann_3y),
        "benchmark": "3Y annualised >10% = strong long-term performer",
        "detail":    f"Annualised: {_pct(ann_3y)}" if ann_3y is not None else "",
    }
    metrics["return_5y"] = {
        "label":     "5-Year Price Return",
        "value":     ret_5y,
        "formatted": f"{ret_5y:+.1%}" if ret_5y is not None else "N/A",
        "assessment": _ret_note(ret_5y, ann_5y),
        "benchmark": "5Y annualised >8% = solid long-term compounder",
        "detail":    f"Annualised: {_pct(ann_5y)}" if ann_5y is not None else "",
    }

    # ── Volatility and drawdown from 5Y monthly ───────────────────────────
    mo_rets = _monthly_returns(close_5y_mo)

    if len(mo_rets) >= 12:
        vol_monthly = _stdev(mo_rets)
        vol_annual  = vol_monthly * math.sqrt(12) if vol_monthly is not None else None

        if vol_annual is not None:
            if vol_annual > 0.45:
                vol_note = f"Very high volatility ({_pct(vol_annual)}/yr) — significant price swings"
                flags.append(f"Annualised volatility {_pct(vol_annual)} — high risk profile")
            elif vol_annual > 0.30:
                vol_note = f"High volatility ({_pct(vol_annual)}/yr) — above-average price swings"
            elif vol_annual > 0.18:
                vol_note = f"Moderate volatility ({_pct(vol_annual)}/yr)"
            else:
                vol_note = f"Low volatility ({_pct(vol_annual)}/yr) — stable price behaviour"

            metrics["volatility"] = {
                "label":     "Price Volatility (Annualised)",
                "value":     vol_annual,
                "formatted": _pct(vol_annual),
                "assessment": vol_note,
                "benchmark": "<20% = low, 20–35% = moderate, >35% = high",
                "detail":    f"Based on {len(mo_rets)} monthly return observations",
            }

            # Approx Sharpe
            if ann_5y is not None and vol_annual > 0:
                sharpe = (ann_5y - RISK_FREE) / vol_annual
                sharpe_note = (
                    f"Excellent risk-adjusted return (Sharpe {sharpe:.2f})"  if sharpe > 1.0 else
                    f"Good risk-adjusted return (Sharpe {sharpe:.2f})"        if sharpe > 0.5 else
                    f"Acceptable risk-adjusted return (Sharpe {sharpe:.2f})"  if sharpe > 0.0 else
                    f"Poor risk-adjusted return (Sharpe {sharpe:.2f}) — below risk-free rate"
                )
                metrics["sharpe"] = {
                    "label":     "Approx. Sharpe Ratio (5Y)",
                    "value":     sharpe,
                    "formatted": f"{sharpe:.2f}",
                    "assessment": sharpe_note,
                    "benchmark": ">1.0 = excellent, 0.5–1.0 = good, <0 = worse than cash",
                    "detail":    f"({_pct(ann_5y)} return − {_pct(RISK_FREE)} risk-free) / {_pct(vol_annual)} volatility",
                }

    # ── Max drawdown ──────────────────────────────────────────────────────
    mdd = _max_drawdown(close_5y_mo)
    if mdd is not None:
        if mdd < -0.50:
            mdd_note = f"Severe drawdown ({mdd:.0%}) — significant capital impairment risk"
            flags.append(f"Max drawdown over 5Y: {mdd:.0%}")
        elif mdd < -0.30:
            mdd_note = f"Large drawdown ({mdd:.0%}) — meaningful peak-to-trough decline"
        elif mdd < -0.15:
            mdd_note = f"Moderate drawdown ({mdd:.0%})"
        else:
            mdd_note = f"Limited drawdown ({mdd:.0%}) — relatively stable price history"

        metrics["max_drawdown"] = {
            "label":     "Max Drawdown (5Y monthly)",
            "value":     mdd,
            "formatted": f"{mdd:.1%}",
            "assessment": mdd_note,
            "benchmark": ">-20% = resilient; -30 to -50% = typical cyclical; <-50% = high distress risk",
            "detail":    "Maximum peak-to-trough price decline over the 5-year period",
        }

    # ── Up/down month ratio ───────────────────────────────────────────────
    if mo_rets:
        up   = sum(1 for r in mo_rets if r > 0)
        down = len(mo_rets) - up
        ratio = up / len(mo_rets)
        metrics["up_down_ratio"] = {
            "label":     "Up / Down Month Ratio (5Y)",
            "value":     ratio,
            "formatted": f"{up} up / {down} down",
            "assessment": (
                f"{ratio:.0%} of months positive — "
                + ("consistent upward trend" if ratio > 0.65 else
                   "slight upward bias" if ratio > 0.55 else
                   "roughly balanced" if ratio > 0.45 else
                   "more down months — weak price trend")
            ),
            "benchmark": ">60% up months = trending stock; <40% = persistent weakness",
            "detail":    f"Over {len(mo_rets)} monthly observations",
        }

    # ── Price vs fundamental growth ───────────────────────────────────────
    if ann_5y is not None and rev_cagr_5y is not None:
        gap = ann_5y - rev_cagr_5y
        if gap > 0.08:
            pf_note = (
                f"Price CAGR ({_pct(ann_5y)}) significantly outpacing revenue CAGR ({_pct(rev_cagr_5y)}) — "
                f"multiple expansion ({gap:+.1%} gap). Valuation has stretched vs fundamentals."
            )
        elif gap > 0.02:
            pf_note = (
                f"Price CAGR ({_pct(ann_5y)}) modestly ahead of revenue growth ({_pct(rev_cagr_5y)}) — "
                f"slight multiple expansion."
            )
        elif gap > -0.02:
            pf_note = (
                f"Price CAGR ({_pct(ann_5y)}) broadly in line with revenue growth ({_pct(rev_cagr_5y)}) — "
                f"fundamentals and price have moved together."
            )
        else:
            pf_note = (
                f"Price CAGR ({_pct(ann_5y)}) lagging revenue growth ({_pct(rev_cagr_5y)}) by {abs(gap):.1%} — "
                f"multiple compression. Business growing faster than market recognises."
            )

        metrics["price_vs_fundamentals"] = {
            "label":     "Price CAGR vs Revenue CAGR (5Y)",
            "value":     gap,
            "formatted": f"{gap:+.1%}",
            "assessment": pf_note,
            "benchmark": "Gap > +5% = multiple expansion (re-rating); gap < -5% = de-rating",
            "detail":    f"Price 5Y CAGR: {_pct(ann_5y)}  |  Revenue 5Y CAGR: {_pct(rev_cagr_5y)}",
        }

    return {
        "backtest_metrics": metrics,
        "backtest_flags":   flags,
    }


# ── display helper ─────────────────────────────────────────────────────────────

def print_price_history(result: dict, ticker: str = ""):
    header = f"PRICE HISTORY & PERFORMANCE - {ticker}" if ticker else "PRICE HISTORY"
    print(f"\n{'-' * 70}")
    print(f"  {header}")
    print(f"{'-' * 70}")

    for _, m in result["backtest_metrics"].items():
        if m["formatted"] and m["formatted"] != "N/A":
            print(f"  {m['label']:<44} {m['formatted']:<12}  {m['assessment']}")
        else:
            print(f"  {m['label']:<44} {m['assessment']}")
        if m.get("detail"):
            print(f"    - {m['detail']}")

    if result["backtest_flags"]:
        print(f"\n  Performance flags:")
        for f in result["backtest_flags"]:
            print(f"     - {f}")
