"""
projection_engine.py — Forward projection & probability estimation

generate_projections(record) → dict with:
  - p_up_5d / p_up_20d / p_up_60d / p_up_120d   probability price will be higher
  - expected_return_5d / 20d / 60d / 120d       expected return over horizon
  - signal        BULLISH / LEAN_BULLISH / NEUTRAL / LEAN_BEARISH / BEARISH
  - confidence    HIGH / MEDIUM_HIGH / MEDIUM / LOW
  - paths         bull/base/bear projection paths (list of price points)
  - news          news sentiment signal (if available)
  - ml_used       whether ML model was used for probability estimation

Hybrid model:
  Phase 1 (active): weighted scoring heuristic from engine outputs
  Phase 1.5 (active): news sentiment from FinBERT + Claude
  Phase 2 (active if trained): LightGBM classifiers on technical features
"""

import sys
import math
from pathlib import Path

# Ensure stock_analyzer is importable
_root = Path(__file__).resolve().parents[1]
_sa = str(_root / "stock_analyzer")
if _sa not in sys.path:
    sys.path.insert(0, _sa)

from utils import _pct
from projection_settings import load_projection_settings

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
    "valuation_upside": 0.28,
    "momentum_trend":   0.18,
    "rsi_signal":       0.08,
    "quality_score":    0.14,
    "risk_penalty":     0.14,
    "growth_signal":    0.08,
    "news_sentiment":   0.10,   # news signal weight (0 if unavailable → redistributed)
}


# ── scoring functions ──────────────────────────────────────────────────────────

def _score_valuation(record: dict) -> float:
    fair_value = record.get("fair_value_weighted")
    price = record.get("current_price")
    if not fair_value or not price or price <= 0:
        return 0.0
    upside = (fair_value - price) / price
    return max(-1.0, min(1.0, upside / 0.5 * 0.9))


def _score_momentum(record: dict) -> float:
    score = 0.0
    trend = record.get("momentum_trend", "UNKNOWN")
    if trend == "UPTREND":
        score += 0.4
    elif trend == "DOWNTREND":
        score -= 0.4

    price = record.get("current_price")
    ma200 = record.get("ma200")
    if price and ma200 and ma200 > 0:
        score += max(-0.3, min(0.3, (price - ma200) / ma200))

    mom_metrics = record.get("momentum_metrics") or {}
    ret_3m = (mom_metrics.get("return_3m") or {}).get("value")
    if ret_3m is not None:
        score += max(-0.3, min(0.3, ret_3m))

    return max(-1.0, min(1.0, score))


def _score_rsi(record: dict) -> float:
    rsi = record.get("rsi14")
    if rsi is None:
        return 0.0
    return max(-1.0, min(1.0, (50 - rsi) / 25))


def _score_quality(record: dict) -> float:
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
        elif fcf_cagr < -0.05:
            score -= 0.3

    return max(-1.0, min(1.0, score))


def _score_news(news_result: dict | None) -> float:
    """Convert news sentiment result to -1..+1 score."""
    if not news_result or not news_result.get("available"):
        return None  # signals "no news data" — weight is redistributed
    return max(-1.0, min(1.0, news_result.get("sentiment_score", 0.0)))


# ── composite score ────────────────────────────────────────────────────────────

def _composite_score(
    record: dict,
    news_result: dict | None = None,
    *,
    exclude_valuation: bool = False,
) -> tuple[float, dict]:
    """Weighted composite score -1 (very bearish) to +1 (very bullish)."""
    news_score = _score_news(news_result)

    raw_scores = {
        "valuation_upside": 0.0 if exclude_valuation else _score_valuation(record),
        "momentum_trend":   _score_momentum(record),
        "rsi_signal":       _score_rsi(record),
        "quality_score":    _score_quality(record),
        "risk_penalty":     _score_risk(record),
        "growth_signal":    _score_growth(record),
    }

    weights = dict(WEIGHTS)
    if exclude_valuation:
        vw = weights.pop("valuation_upside", 0.0)
        tot = sum(weights.values())
        if tot > 0:
            weights = {k: v + (v / tot) * vw for k, v in weights.items()}

    if news_score is None:
        # Redistribute news weight proportionally to other factors
        news_w = weights.pop("news_sentiment", 0.0)
        total_other = sum(weights.values())
        if total_other > 0:
            weights = {k: v + v / total_other * news_w for k, v in weights.items()}
        scores = raw_scores
    else:
        raw_scores["news_sentiment"] = news_score
        scores = raw_scores

    composite = sum(scores[k] * weights[k] for k in weights if k in scores)
    return max(-1.0, min(1.0, composite)), scores


# ── probability estimation ─────────────────────────────────────────────────────

def _rule_based_probability(score: float, horizon_days: int) -> float:
    """
    Rule-based probability of positive return.
    Base drift: ~52% for 20d, ~56% for 60d, ~62% for 120d.
    """
    base_prob = 0.50 + 0.001 * horizon_days   # 20d→0.52, 60d→0.56, 120d→0.62
    adjustment = score * 0.25
    return max(0.05, min(0.95, base_prob + adjustment))


def _ml_probability(record: dict, horizon_days: int) -> float | None:
    """Try ML model inference; return None if model not available."""
    try:
        _proj_dir = str(Path(__file__).parent)
        if _proj_dir not in sys.path:
            sys.path.insert(0, _proj_dir)
        from ml_model.predictor import ml_predict
        from ml_model.features import extract_features

        features = extract_features(record)
        preds = ml_predict(features, horizons=[horizon_days])
        if preds and horizon_days in preds:
            return preds[horizon_days]
    except Exception:
        pass
    return None


def _score_to_expected_return(score: float, horizon_days: int) -> float:
    annual_drift = 0.08 + score * 0.20
    return annual_drift * (horizon_days / 252)


# ── projection paths ───────────────────────────────────────────────────────────

def _generate_paths(
    current_price: float,
    fair_value: float | None,
    score: float,
    horizon_days: int = 120,
    n_points: int = 25,
) -> dict:
    if fair_value is None or fair_value <= 0:
        fair_value = current_price * (1 + score * 0.3)

    upside = (fair_value - current_price) / current_price
    bull_target = current_price * (1 + max(upside * 1.3, 0.05))
    base_target = current_price * (1 + upside * 0.7)
    bear_target = current_price * (1 + min(upside * 0.3, -0.05))

    base_speed = 2.5 / horizon_days
    bull_speed = base_speed * (1.0 + max(score, 0) * 0.5)
    bear_speed = base_speed * (1.0 + max(-score, 0) * 0.5)

    paths: dict = {"bull": [], "base": [], "bear": [], "days": []}
    for i in range(n_points + 1):
        t = i / n_points
        day = int(t * horizon_days)
        paths["days"].append(day)
        paths["bull"].append(round(current_price + (bull_target - current_price) * (1 - math.exp(-bull_speed * day)), 2))
        paths["base"].append(round(current_price + (base_target - current_price) * (1 - math.exp(-base_speed * day)), 2))
        paths["bear"].append(round(current_price + (bear_target - current_price) * (1 - math.exp(-bear_speed * day)), 2))

    return paths


# ── signal & confidence ────────────────────────────────────────────────────────

def _classify_signal(score: float) -> str:
    if score > 0.35:
        return "BULLISH"
    elif score > 0.15:
        return "LEAN_BULLISH"
    elif score > -0.15:
        return "NEUTRAL"
    elif score > -0.35:
        return "LEAN_BEARISH"
    return "BEARISH"


def _classify_confidence(
    scores: dict,
    record: dict,
    ml_used: bool,
    ml_rule_disagreement: bool = False,
) -> str:
    positive = sum(1 for v in scores.values() if v > 0.1)
    negative = sum(1 for v in scores.values() if v < -0.1)
    agreement = max(positive, negative) / len(scores)

    dq = record.get("data_quality_score", 50)
    if dq < 50:
        agreement *= 0.7

    if ml_used:
        agreement = min(1.0, agreement * 1.1)  # slight boost when ML confirms

    if agreement > 0.8:
        label = "HIGH"
    elif agreement > 0.6:
        label = "MEDIUM_HIGH"
    elif agreement > 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    if ml_rule_disagreement and label != "LOW":
        order = ["HIGH", "MEDIUM_HIGH", "MEDIUM", "LOW"]
        label = order[min(order.index(label) + 1, len(order) - 1)]
    return label


# ── main function ──────────────────────────────────────────────────────────────

def generate_projections(
    record: dict,
    horizon_days: int = 120,
    news_result: dict | None = None,
    *,
    exclude_valuation: bool = False,
) -> dict:
    """
    Generate forward projections from a fully-analyzed stock record.

    Args:
        record:       Output from the stock_analyzer pipeline.
        horizon_days: Projection horizon in trading days.
        news_result:  Output from news_engine.analyze_news() — optional.
        exclude_valuation: If True, drop DCF upside from the composite (weight redistributed
            to momentum, RSI, quality, risk, growth, news).
    """
    score, sub_scores = _composite_score(record, news_result, exclude_valuation=exclude_valuation)
    price_raw = record.get("current_price")
    fair_value = record.get("fair_value_weighted")

    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        return {"error": "No price data available"}
    if math.isnan(price) or math.isinf(price) or price <= 0:
        return {"error": "No price data available"}

    beta = record.get("beta") or 1.0
    volatility = 0.20 * beta

    # Try ML model first; fall back to rule-based
    ml_p5 = _ml_probability(record, 5)
    ml_p20 = _ml_probability(record, 20)
    ml_p60 = _ml_probability(record, 60)
    ml_p120 = _ml_probability(record, horizon_days)
    ml_used = any(x is not None for x in (ml_p5, ml_p20, ml_p60, ml_p120))

    settings = load_projection_settings()
    ml_w = settings.ml_blend_weight
    dthr = settings.ml_rule_disagreement_threshold
    unc = settings.probability_uncertainty_half_width

    def _blend(ml_p: float | None, rule_p: float) -> tuple[float, float | None, float]:
        if ml_p is None:
            return rule_p, None, rule_p
        return ml_w * ml_p + (1.0 - ml_w) * rule_p, ml_p, rule_p

    r5 = _rule_based_probability(score, 5)
    r20 = _rule_based_probability(score, 20)
    r60 = _rule_based_probability(score, 60)
    r120 = _rule_based_probability(score, horizon_days)

    p_up_5d, ml5, rule5 = _blend(ml_p5, r5)
    p_up_20d, ml20, rule20 = _blend(ml_p20, r20)
    p_up_60d, ml60, rule60 = _blend(ml_p60, r60)
    p_up_120d, ml120, rule120 = _blend(ml_p120, r120)

    ml_vs_rule: dict[str, dict] = {}
    for key, ml_p, rule_p in (
        ("5d", ml5, r5),
        ("20d", ml20, r20),
        ("60d", ml60, r60),
        ("horizon", ml120, r120),
    ):
        if ml_p is not None:
            sp = abs(float(ml_p) - float(rule_p))
            ml_vs_rule[key] = {
                "ml": round(float(ml_p), 4),
                "rule": round(float(rule_p), 4),
                "spread": round(sp, 4),
                "disagree": sp > dthr,
            }
    ml_rule_disagreement = any(d["disagree"] for d in ml_vs_rule.values()) if ml_vs_rule else False

    def _band(p: float) -> tuple[float, float]:
        lo = max(0.05, float(p) - unc)
        hi = min(0.95, float(p) + unc)
        return round(lo, 3), round(hi, 3)

    probability_bands = {
        "5d": _band(p_up_5d),
        "20d": _band(p_up_20d),
        "60d": _band(p_up_60d),
        "horizon": _band(p_up_120d),
    }

    er_5d = _score_to_expected_return(score, 5)
    er_20d = _score_to_expected_return(score, 20)
    er_60d = _score_to_expected_return(score, 60)
    er_120d = _score_to_expected_return(score, horizon_days)

    signal = _classify_signal(score)
    confidence = _classify_confidence(
        sub_scores, record, ml_used, ml_rule_disagreement=ml_rule_disagreement
    )

    paths = _generate_paths(
        current_price=price,
        fair_value=fair_value,
        score=score,
        horizon_days=horizon_days,
        n_points=25,
    )

    vol_daily = volatility / math.sqrt(252)
    upper_band, lower_band = [], []
    for i, base_p in enumerate(paths["base"]):
        day = paths["days"][i]
        bw = base_p * vol_daily * math.sqrt(max(day, 1))
        upper_band.append(round(base_p + bw, 2))
        lower_band.append(round(base_p - bw, 2))

    return {
        "composite_score":       round(score, 3),
        "sub_scores":            {k: round(v, 3) for k, v in sub_scores.items()},
        "ml_used":               ml_used,

        "p_up_5d":               round(p_up_5d, 3),
        "p_up_20d":              round(p_up_20d, 3),
        "p_up_60d":              round(p_up_60d, 3),
        "p_up_120d":             round(p_up_120d, 3),
        # Aliases for the active horizon_days (p_up_120d / ER_120d are horizon-parameterized)
        "p_up_horizon":          round(p_up_120d, 3),
        "expected_return_horizon": round(er_120d, 4),

        "expected_return_5d":    round(er_5d, 4),
        "expected_return_20d":   round(er_20d, 4),
        "expected_return_60d":   round(er_60d, 4),
        "expected_return_120d":  round(er_120d, 4),

        "signal":                signal,
        "confidence":            confidence,

        "paths":                 paths,
        "upper_band":            upper_band,
        "lower_band":            lower_band,

        "targets": {
            "bull":       paths["bull"][-1],
            "base":       paths["base"][-1],
            "bear":       paths["bear"][-1],
            "fair_value": fair_value,
        },

        "news":          news_result,
        "current_price": price,
        "horizon_days":  horizon_days,

        "ml_blend_weight_used": round(ml_w, 3),
        "ml_rule_disagreement": ml_rule_disagreement,
        "ml_vs_rule":     ml_vs_rule,
        "probability_bands": probability_bands,
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
    print(f"  ML model:    {'YES' if result['ml_used'] else 'no (rule-based fallback)'}")
    print()
    print(
        f"  P(up)   5d: {result['p_up_5d']:.0%}   20d: {result['p_up_20d']:.0%}   "
        f"60d: {result['p_up_60d']:.0%}   {result['horizon_days']}d: {result.get('p_up_horizon', result['p_up_120d']):.0%}"
    )
    print(
        f"  ExpRet  5d: {result['expected_return_5d']:+.1%}   20d: {result['expected_return_20d']:+.1%}   "
        f"60d: {result['expected_return_60d']:+.1%}   {result['horizon_days']}d: "
        f"{result.get('expected_return_horizon', result['expected_return_120d']):+.1%}"
    )

    t = result["targets"]
    print(f"\n  Targets ({result['horizon_days']}d):  Bull {t['bull']:.2f}  Base {t['base']:.2f}  Bear {t['bear']:.2f}")

    if result.get("news") and result["news"].get("available"):
        n = result["news"]
        print(f"\n  News ({n['n_articles']} articles): {n['signal']}  score={n['sentiment_score']:+.2f}")

    print(f"\n  Sub-scores:")
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
        "momentum_metrics": {"return_3m": {"value": 0.08}},
    }
    result = generate_projections(sample, horizon_days=120)
    print_projections(result, "SAMPLE")
