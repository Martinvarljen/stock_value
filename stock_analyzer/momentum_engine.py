"""
momentum_engine.py  —  Price and technical momentum

analyze_momentum(data) → dict with:
  - price_returns:  1M, 3M, 6M, 12M, 3Y price returns
  - trend:          UPTREND / DOWNTREND / SIDEWAYS (MA50 vs MA200)
  - ma_signal:      Golden Cross / Death Cross / Neutral
  - rsi:            14-day RSI with overbought/oversold assessment
  - vs_52w:         position within 52-week range (percentile)
  - momentum_flags: notable signals

All inputs come from data_layer fields — no extra API calls.
"""

from utils import _pct, _num


# ── helpers ────────────────────────────────────────────────────────────────────

def _return(series: list, lookback: int) -> float | None:
    """Return from `lookback` bars ago to the last bar."""
    if not series or len(series) < lookback + 1:
        return None
    base = series[-lookback - 1]
    if not base or base <= 0:
        return None
    return (series[-1] - base) / base


def _avg(series: list) -> float | None:
    vals = [v for v in series if v is not None]
    return sum(vals) / len(vals) if vals else None


# ── main function ──────────────────────────────────────────────────────────────

def analyze_momentum(data: dict) -> dict:
    metrics = {}
    flags   = []

    price       = data.get("current_price")
    close_1y    = data.get("close_1y") or []          # daily, ~252 bars
    close_5y_mo = data.get("close_5y_monthly") or []  # monthly, ~60 bars
    ma50        = data.get("ma50")
    ma200       = data.get("ma200")
    rsi         = data.get("rsi14")
    w52_high    = data.get("week52_high")
    w52_low     = data.get("week52_low")
    ret_1y      = data.get("return_1y")
    ret_3y      = data.get("return_3y")

    # ── Short-term returns from monthly series ────────────────────────────
    ret_1m  = _return(close_5y_mo, 1)
    ret_3m  = _return(close_5y_mo, 3)
    ret_6m  = _return(close_5y_mo, 6)
    ret_12m = _return(close_5y_mo, 12)

    def _ret_assessment(r, period):
        if r is None:
            return "No data"
        if r > 0.30:  return f"Very strong {period} gain ({r:+.1%})"
        if r > 0.10:  return f"Strong {period} gain ({r:+.1%})"
        if r > 0.02:  return f"Positive {period} ({r:+.1%})"
        if r > -0.02: return f"Flat {period} ({r:+.1%})"
        if r > -0.10: return f"Mild {period} decline ({r:+.1%})"
        if r > -0.25: return f"Significant {period} decline ({r:+.1%})"
        return              f"Sharp {period} sell-off ({r:+.1%})"

    metrics["return_1m"] = {
        "label":     "1-Month Return",
        "value":     ret_1m,
        "formatted": f"{ret_1m:+.1%}" if ret_1m is not None else "N/A",
        "assessment": _ret_assessment(ret_1m, "1M"),
        "benchmark": "Relative to recent trend — single months are noisy",
        "detail":    "",
    }
    metrics["return_3m"] = {
        "label":     "3-Month Return",
        "value":     ret_3m,
        "formatted": f"{ret_3m:+.1%}" if ret_3m is not None else "N/A",
        "assessment": _ret_assessment(ret_3m, "3M"),
        "benchmark": "3M momentum is the most studied short-term signal",
        "detail":    "",
    }
    metrics["return_6m"] = {
        "label":     "6-Month Return",
        "value":     ret_6m,
        "formatted": f"{ret_6m:+.1%}" if ret_6m is not None else "N/A",
        "assessment": _ret_assessment(ret_6m, "6M"),
        "benchmark": "6–12M momentum tends to persist; <-20% = potential distress or value opportunity",
        "detail":    "",
    }
    metrics["return_12m"] = {
        "label":     "12-Month Return",
        "value":     ret_12m,
        "formatted": f"{ret_12m:+.1%}" if ret_12m is not None else "N/A",
        "assessment": _ret_assessment(ret_12m, "1Y"),
        "benchmark": "1Y return captures a full earnings cycle",
        "detail":    f"3Y cumulative: {ret_3y:+.1%}" if ret_3y is not None else "",
    }

    # ── Moving average trend ──────────────────────────────────────────────
    if ma50 and ma200 and price:
        if ma50 > ma200 * 1.02:
            trend     = "UPTREND"
            ma_signal = "Golden Cross — MA50 above MA200 (bullish)"
        elif ma50 < ma200 * 0.98:
            trend     = "DOWNTREND"
            ma_signal = "Death Cross — MA50 below MA200 (bearish)"
            flags.append("MA50 below MA200 — price in a downtrend")
        else:
            trend     = "SIDEWAYS"
            ma_signal = "MA50 ≈ MA200 — no clear directional trend"

        price_vs_ma50  = (price - ma50)  / ma50
        price_vs_ma200 = (price - ma200) / ma200

        metrics["moving_averages"] = {
            "label":     "Moving Average Trend",
            "value":     None,
            "formatted": trend,
            "assessment": ma_signal,
            "benchmark": "Price above both MAs = strength; below both = weakness",
            "detail": (
                f"Price {_num(price)}  |  MA50: {_num(ma50)} ({price_vs_ma50:+.1%})  "
                f"|  MA200: {_num(ma200)} ({price_vs_ma200:+.1%})"
            ),
        }
        if price < ma200:
            flags.append(f"Price ({_num(price)}) below 200-day MA ({_num(ma200)}) — technically weak")
    else:
        trend = "UNKNOWN"
        metrics["moving_averages"] = {
            "label": "Moving Average Trend", "value": None,
            "formatted": "N/A", "assessment": "Insufficient price history for MA calculation",
            "benchmark": "", "detail": "",
        }

    # ── RSI ───────────────────────────────────────────────────────────────
    if rsi is not None:
        if rsi > 75:
            rsi_note = f"Overbought ({rsi:.0f}) — short-term pullback risk elevated"
            flags.append(f"RSI {rsi:.0f} — overbought territory")
        elif rsi > 60:
            rsi_note = f"Mildly overbought ({rsi:.0f}) — momentum positive but extended"
        elif rsi > 40:
            rsi_note = f"Neutral ({rsi:.0f}) — no extreme reading"
        elif rsi > 25:
            rsi_note = f"Mildly oversold ({rsi:.0f}) — potential mean-reversion opportunity"
        else:
            rsi_note = f"Oversold ({rsi:.0f}) — depressed price; possible value entry or distress signal"
            flags.append(f"RSI {rsi:.0f} — deeply oversold territory")

        metrics["rsi"] = {
            "label":     "RSI (14-day)",
            "value":     rsi,
            "formatted": f"{rsi:.0f}",
            "assessment": rsi_note,
            "benchmark": ">70 = overbought, <30 = oversold, 40–60 = neutral",
            "detail":    "Relative Strength Index — measures speed and magnitude of recent price moves",
        }
    else:
        metrics["rsi"] = {
            "label": "RSI (14-day)", "value": None, "formatted": "N/A",
            "assessment": "No data", "benchmark": "", "detail": "",
        }

    # ── 52-week range position ────────────────────────────────────────────
    if price and w52_high and w52_low and w52_high > w52_low:
        pct_of_range = (price - w52_low) / (w52_high - w52_low)
        pct_of_range = max(0.0, min(pct_of_range, 1.0))

        if pct_of_range > 0.90:
            range_note = f"Near 52-week high ({pct_of_range:.0%} of range) — price at recent peak"
        elif pct_of_range > 0.70:
            range_note = f"Upper part of 52-week range ({pct_of_range:.0%})"
        elif pct_of_range > 0.40:
            range_note = f"Mid 52-week range ({pct_of_range:.0%})"
        elif pct_of_range > 0.20:
            range_note = f"Lower part of 52-week range ({pct_of_range:.0%}) — near recent lows"
        else:
            range_note = f"Near 52-week low ({pct_of_range:.0%} of range) — significant recent weakness"
            flags.append(f"Near 52-week low ({_num(price)} vs low {_num(w52_low)})")

        metrics["range_52w"] = {
            "label":     "52-Week Range Position",
            "value":     pct_of_range,
            "formatted": f"{pct_of_range:.0%} of range",
            "assessment": range_note,
            "benchmark": "Top 10% = price momentum positive; bottom 10% = weakness or opportunity",
            "detail": (
                f"Current: {_num(price)}  |  52W Low: {_num(w52_low)}  |  52W High: {_num(w52_high)}  "
                f"|  From high: {(price/w52_high - 1):+.1%}"
            ),
        }
    else:
        metrics["range_52w"] = {
            "label": "52-Week Range Position", "value": None,
            "formatted": "N/A", "assessment": "No data",
            "benchmark": "", "detail": "",
        }

    # ── Momentum consistency (consecutive positive months) ────────────────
    if len(close_5y_mo) >= 6:
        recent_6mo = close_5y_mo[-7:]   # 7 bars → 6 returns
        mo_rets    = []
        for j in range(1, len(recent_6mo)):
            prev = recent_6mo[j - 1]
            curr = recent_6mo[j]
            if prev and prev > 0 and curr is not None:
                mo_rets.append((curr - prev) / prev)

        pos_months = sum(1 for r in mo_rets if r > 0)
        neg_months = len(mo_rets) - pos_months
        consistency = (
            f"{pos_months}/{len(mo_rets)} months positive in the past 6 months — "
            + ("consistent upward momentum" if pos_months >= 5 else
               "mostly positive" if pos_months >= 4 else
               "mixed price action" if pos_months >= 3 else
               "mostly negative recent trend" if pos_months <= 2 else "")
        )
        metrics["momentum_consistency"] = {
            "label":     "6-Month Momentum Consistency",
            "value":     pos_months,
            "formatted": f"{pos_months}/{len(mo_rets)} months positive",
            "assessment": consistency,
            "benchmark": "5-6/6 positive = strong momentum; 0-2/6 = weak/distribution phase",
            "detail":    "",
        }

    return {
        "momentum_metrics": metrics,
        "momentum_flags":   flags,
        "trend":            trend,
    }


# ── display helper ─────────────────────────────────────────────────────────────

def print_momentum(result: dict, ticker: str = ""):
    header = f"MOMENTUM — {ticker}" if ticker else "MOMENTUM"
    print(f"\n{'─' * 70}")
    print(f"  {header}  [Trend: {result['trend']}]")
    print(f"{'─' * 70}")

    for key, m in result["momentum_metrics"].items():
        if m["formatted"] and m["formatted"] != "N/A":
            print(f"  {m['label']:<42} {m['formatted']:<14}  {m['assessment']}")
        else:
            print(f"  {m['label']:<42} {m['assessment']}")
        if m.get("detail"):
            print(f"    └─ {m['detail']}")

    if result["momentum_flags"]:
        print(f"\n  ⚠  Momentum flags:")
        for f in result["momentum_flags"]:
            print(f"     • {f}")
