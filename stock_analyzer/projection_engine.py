"""
projection_engine.py  —  Forward projection & probability estimation

generate_projections(record) → dict with:
  - p_up_20d / p_up_60d / p_up_120d:    probability price will be higher
  - expected_return_20d / 60d / 120d:    expected return over horizon
  - signal:       BULLISH / LEAN_BULLISH / NEUTRAL / LEAN_BEARISH / BEARISH
  - confidence:   HIGH / MEDIUM_HIGH / MEDIUM / LOW
  - paths:        bull/base/bear projection paths (list of price points)

Phase 1: Weighted scoring heuristic using existing engine outputs.
Phase 2: Swap in ML model (XGBoost/LightGBM) trained on historical features.
"""

import sys
import math
from utils import _pct

# Windows console fix
for _s in ("stdout", "stderr"):
    _stream = getattr(sys, _s, None)
    _reconf = getattr(_stream, "reconfigure", None)
    if callable(_reconf):
        try:
            _reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── signal weights ─────────────────────────────────────────────────────────────

WEIGHTS = {
    "valuation_upside":  0.30,   # DCF fair value vs price
    "momentum_trend":    0.20,   # MA50/200 trend + recent returns
    "rsi_signal":        0.10,   # RSI mean-reversion
    "quality_score":     0.15,   # margins, ROIC, consistency
    "risk_penalty":      0.15,   # red flags, leverage, volatility
    "growth_signal":     0.10,   # revenue/FCF growth trajectory
}


# ── scoring functions ──────────────────────────────────────────────────────────

def _score_valuation(record: dict) -> float:
    """Score -1 to +1 based on DCF upside/downside."""
    fair_value = record.get("fair_value_weighted")
    price = record.get("current_price")
    if not fair_value or not price or price <= 0:
        return 0.0

    upside = (fair_value - price) / price
    # Sigmoid-like mapping: ±50% upside → ±0.9
    return max(-1.0, min(1.0, upside / 0.5 * 0.9))


def _score_momentum(record: dict) -> float:
    """Score -1 to +1 based on trend and recent returns."""
    score = 0.0

    trend = record.get("momentum_trend", "UNKNOWN")
    if trend == "UPTREND":
        score += 0.4
    elif trend == "DOWNTREND":
        score -= 0.4

    # Price vs MA200
    price = record.get("current_price")
    ma200 = record.get("ma200")
    if price and ma200 and ma200 > 0:
        pct_above = (price - ma200) / ma200
        score += max(-0.3, min(0.3, pct_above))

    # 3-month momentum
    mom_metrics = record.get("momentum_metrics") or {}
    ret_3m = (mom_metrics.get("return_3m") or {}).get("value")
    if ret_3m is not None:
        score += max(-0.3, min(0.3, ret_3m))

    return max(-1.0, min(1.0, score))


def _score_rsi(record: dict) -> float:
    """Score -1 to +1. Oversold = bullish (mean reversion), overbought = bearish."""
    rsi = record.get("rsi14")
    if rsi is None:
        return 0.0

    # RSI 30 → +0.8 (oversold, expect bounce)
    # RSI 50 → 0.0 (neutral)
    # RSI 70 → -0.8 (overbought, expect pullback)
    return max(-1.0, min(1.0, (50 - rsi) / 25))


def _score_quality(record: dict) -> float:
    """Score 0 to +1 based on business quality metrics."""
    score = 0.0
    count = 0

    op_margin = record.get("operating_margin")
    if op_margin is not None:
        if op_margin > 0.25:
            score += 1.0
        elif op_margin > 0.15:
            score += 0.6
        elif op_margin > 0.05:
            score += 0.3
        elif op_margin < 0:
            score -= 0.5
        count += 1

    roic = record.get("roic")
    wacc = (record.get("wacc_data") or {}).get("wacc")
    if roic is not None and wacc is not None:
        spread = roic - wacc
        if spread > 0.08:
            score += 1.0
        elif spread > 0.03:
            score += 0.6
        elif spread > 0:
            score += 0.2
        else:
            score -= 0.3
        count += 1

    fcf_yield = record.get("fcf_yield")
    if fcf_yield is not None:
        if fcf_yield > 0.06:
            score += 0.8
        elif fcf_yield > 0.03:
            score += 0.4
        elif fcf_yield < 0:
            score -= 0.5
        count += 1

    if count == 0:
        return 0.0
    return max(-1.0, min(1.0, score / count))


def _score_risk(record: dict) -> float:
    """Score 0 to -1 based on risk factors (always a penalty)."""
    penalty = 0.0

    critical_flags = record.get("critical_flags") or []
    red_flags = record.get("red_flags") or []
    high_rf = [f for f in red_flags if f.get("severity") == "HIGH"]

    penalty += len(critical_flags) * 0.4
    penalty += len(high_rf) * 0.2
    penalty += len([f for f in red_flags if f.get("severity") == "MEDIUM"]) * 0.08

    nd_ebitda = record.get("net_debt_ebitda")
    if nd_ebitda is not None and nd_ebitda > 4.0:
        penalty += 0.3

    return -min(1.0, penalty)


def _score_growth(record: dict) -> float:
    """Score -1 to +1 based on growth trajectory."""
    score = 0.0

    rev_cagr = record.get("revenue_cagr_5y")
    if rev_cagr is not None:
        if rev_cagr > 0.15:
            score += 0.8
        elif rev_cagr > 0.07:
            score += 0.4
        elif rev_cagr > 0:
            score += 0.1
        elif rev_cagr > -0.05:
            score -= 0.2
        else:
            score -= 0.6

    fcf_cagr = record.get("fcf_cagr_5y")
    if fcf_cagr is not None:
        if fcf_cagr > 0.15:
            score += 0.4
        elif fcf_cagr > 0.05:
            score += 0.2
        elif fcf_cagr is not None and fcf_cagr < -0.05:
            score -= 0.3

    return max(-1.0, min(1.0, score))


# ── composite score → probability ─────────────────────────────────────────────

def _composite_score(record: dict) -> float:
    """Weighted composite score from -1 (very bearish) to +1 (very bullish)."""
    scores = {
        "valuation_upside": _score_valuation(record),
        "momentum_trend":   _score_momentum(record),
        "rsi_signal":       _score_rsi(record),
        "quality_score":    _score_quality(record),
        "risk_penalty":     _score_risk(record),
        "growth_signal":    _score_growth(record),
    }

    composite = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    return max(-1.0, min(1.0, composite)), scores


def _score_to_probability(score: float, horizon_days: int) -> float:
    """
    Convert composite score to probability of positive return.
    Longer horizons have higher base probability (stocks tend to go up over time).
    """
    # Base probability (market drift): ~53% for 20d, ~57% for 60d, ~62% for 120d
    base_prob = 0.50 + 0.003 * horizon_days

    # Score adjustment: ±0.25 probability swing at max signal
    adjustment = score * 0.25

    return max(0.05, min(0.95, base_prob + adjustment))


def _score_to_expected_return(score: float, horizon_days: int, volatility: float = 0.25) -> float:
    """
    Convert composite score to expected annualized-equivalent return.
    Uses score as directional bias and volatility for magnitude.
    """
    # Annualised drift assumption: market ~8%, adjusted by score
    annual_drift = 0.08 + score * 0.20  # range: -12% to +28% annualised

    # Scale to horizon
    horizon_years = horizon_days / 252
    expected = annual_drift * horizon_years

    # Volatility widens the range but doesn't change expectation
    return expected


# ── projection paths ───────────────────────────────────────────────────────────

def _generate_paths(
    current_price: float,
    fair_value: float | None,
    score: float,
    volatility: float,
    horizon_days: int = 120,
    n_points: int = 25,
) -> dict:
    """
    Generate bull/base/bear smooth projection paths.
    Uses exponential convergence toward target price, with volatility bands.
    """
    if fair_value is None or fair_value <= 0:
        fair_value = current_price * (1 + score * 0.3)

    # Target prices for each scenario
    upside = (fair_value - current_price) / current_price
    bull_target = current_price * (1 + max(upside * 1.3, 0.05))
    base_target = current_price * (1 + upside * 0.7)
    bear_target = current_price * (1 + min(upside * 0.3, -0.05))

    # Speed of convergence (momentum-adjusted)
    # Higher score = faster convergence to bullish target
    base_speed = 2.5 / horizon_days  # reaches ~92% of target by end
    bull_speed = base_speed * (1.0 + max(score, 0) * 0.5)
    bear_speed = base_speed * (1.0 + max(-score, 0) * 0.5)

    paths = {"bull": [], "base": [], "bear": [], "days": []}

    for i in range(n_points + 1):
        t = i / n_points  # 0 to 1
        day = int(t * horizon_days)
        paths["days"].append(day)

        # Exponential convergence: price + (target - price) * (1 - e^(-speed*t*horizon))
        bull_p = current_price + (bull_target - current_price) * (1 - math.exp(-bull_speed * day))
        base_p = current_price + (base_target - current_price) * (1 - math.exp(-base_speed * day))
        bear_p = current_price + (bear_target - current_price) * (1 - math.exp(-bear_speed * day))

        paths["bull"].append(round(bull_p, 2))
        paths["base"].append(round(base_p, 2))
        paths["bear"].append(round(bear_p, 2))

    return paths


# ── signal classification ──────────────────────────────────────────────────────

def _classify_signal(score: float) -> str:
    if score > 0.35:
        return "BULLISH"
    elif score > 0.15:
        return "LEAN_BULLISH"
    elif score > -0.15:
        return "NEUTRAL"
    elif score > -0.35:
        return "LEAN_BEARISH"
    else:
        return "BEARISH"


def _classify_confidence(scores: dict, record: dict) -> str:
    """Confidence based on signal agreement across factors."""
    positive = sum(1 for v in scores.values() if v > 0.1)
    negative = sum(1 for v in scores.values() if v < -0.1)
    total = len(scores)

    agreement = max(positive, negative) / total

    # Reduce confidence if data quality is low
    dq = record.get("data_quality_score", 50)
    if dq < 50:
        agreement *= 0.7

    if agreement > 0.8:
        return "HIGH"
    elif agreement > 0.6:
        return "MEDIUM_HIGH"
    elif agreement > 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# ── main function ──────────────────────────────────────────────────────────────

def generate_projections(record: dict) -> dict:
    """
    Generate forward projections from a fully-analyzed stock record.
    Expects record to contain outputs from all engines (valuation, momentum, etc.)
    """
    score, sub_scores = _composite_score(record)
    price = record.get("current_price")
    fair_value = record.get("fair_value_weighted")

    if price is None or price <= 0:
        return {"error": "No price data available"}

    # Estimate volatility from beta or default
    beta = record.get("beta") or 1.0
    volatility = 0.20 * beta  # approximate annualised vol

    # Probabilities
    p_up_20d = _score_to_probability(score, 20)
    p_up_60d = _score_to_probability(score, 60)
    p_up_120d = _score_to_probability(score, 120)

    # Expected returns
    er_20d = _score_to_expected_return(score, 20, volatility)
    er_60d = _score_to_expected_return(score, 60, volatility)
    er_120d = _score_to_expected_return(score, 120, volatility)

    # Signal & confidence
    signal = _classify_signal(score)
    confidence = _classify_confidence(sub_scores, record)

    # Projection paths
    paths = _generate_paths(
        current_price=price,
        fair_value=fair_value,
        score=score,
        volatility=volatility,
        horizon_days=120,
        n_points=25,
    )

    # Volatility band (1-sigma channel around base path)
    vol_daily = volatility / math.sqrt(252)
    upper_band = []
    lower_band = []
    for i, base_p in enumerate(paths["base"]):
        day = paths["days"][i]
        band_width = base_p * vol_daily * math.sqrt(max(day, 1))
        upper_band.append(round(base_p + band_width, 2))
        lower_band.append(round(base_p - band_width, 2))

    return {
        "composite_score": round(score, 3),
        "sub_scores": {k: round(v, 3) for k, v in sub_scores.items()},

        "p_up_20d": round(p_up_20d, 3),
        "p_up_60d": round(p_up_60d, 3),
        "p_up_120d": round(p_up_120d, 3),

        "expected_return_20d": round(er_20d, 4),
        "expected_return_60d": round(er_60d, 4),
        "expected_return_120d": round(er_120d, 4),

        "signal": signal,
        "confidence": confidence,

        "paths": paths,
        "upper_band": upper_band,
        "lower_band": lower_band,

        "targets": {
            "bull": paths["bull"][-1],
            "base": paths["base"][-1],
            "bear": paths["bear"][-1],
            "fair_value": fair_value,
        },

        "current_price": price,
        "horizon_days": 120,
    }


# ── display helper ─────────────────────────────────────────────────────────────

def print_projections(result: dict, ticker: str = ""):
    if result.get("error"):
        print(f"  Projection error: {result['error']}")
        return

    header = f"PROJECTIONS — {ticker}" if ticker else "PROJECTIONS"
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")

    print(f"  Signal:      {result['signal']}")
    print(f"  Confidence:  {result['confidence']}")
    print(f"  Score:       {result['composite_score']:+.3f}")
    print()

    print(f"  Probability of being UP:")
    print(f"    20 trading days:  {result['p_up_20d']:.0%}")
    print(f"    60 trading days:  {result['p_up_60d']:.0%}")
    print(f"   120 trading days:  {result['p_up_120d']:.0%}")
    print()

    print(f"  Expected Return:")
    print(f"    20d:  {result['expected_return_20d']:+.1%}")
    print(f"    60d:  {result['expected_return_60d']:+.1%}")
    print(f"   120d:  {result['expected_return_120d']:+.1%}")
    print()

    print(f"  Price Targets (120d):")
    t = result["targets"]
    print(f"    Bull:  {t['bull']:.2f}")
    print(f"    Base:  {t['base']:.2f}")
    print(f"    Bear:  {t['bear']:.2f}")
    if t.get("fair_value"):
        print(f"    Fair Value (DCF): {t['fair_value']:.2f}")
    print()

    print(f"  Sub-scores:")
    for k, v in result["sub_scores"].items():
        bar = "+" * int(max(v, 0) * 10) + "-" * int(max(-v, 0) * 10)
        print(f"    {k:<22} {v:+.2f}  {bar}")
    print(f"{'─' * 70}")


# ── quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample = {
        "current_price": 45.0,
        "fair_value_weighted": 62.0,
        "buy_below_price": 43.4,
        "momentum_trend": "UPTREND",
        "ma200": 42.0,
        "rsi14": 55.0,
        "operating_margin": 0.22,
        "roic": 0.14,
        "wacc_data": {"wacc": 0.09},
        "fcf_yield": 0.055,
        "revenue_cagr_5y": 0.08,
        "fcf_cagr_5y": 0.12,
        "critical_flags": [],
        "red_flags": [],
        "net_debt_ebitda": 1.5,
        "beta": 0.9,
        "data_quality_score": 85,
        "momentum_metrics": {
            "return_3m": {"value": 0.08},
        },
    }
    result = generate_projections(sample)
    print_projections(result, "SAMPLE")
