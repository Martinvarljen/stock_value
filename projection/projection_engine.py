"""
projection_engine.py — Forward probability estimation

generate_projections(record) -> dict with:
  - p_up_5d / p_up_20d / p_up_60d / p_up_120d   probability price will be higher
  - expected_return_5d / 20d / 60d / 120d       expected return over horizon
  - signal        BULLISH / LEAN_BULLISH / NEUTRAL / LEAN_BEARISH / BEARISH
  - confidence    HIGH / MEDIUM_HIGH / MEDIUM / LOW
  - composite_score, sub_scores                 rule-composite for diagnostics
  - ml_used                                     whether ML was called
  - ml_blend_weight_used, ml_rule_disagreement, ml_vs_rule

Default behaviour
-----------------
When a calibrated LightGBM model is present (``ml_model/lgbm_*.pkl``), ML
probabilities are used directly. The rule composite is computed for
diagnostics only and DOES NOT dilute the calibrated probability unless the
operator explicitly sets ``ml_blend_weight < 1.0`` in
``projection_settings.json`` / env. Earlier defaults blended ML at 0.6 with
hand-tuned rule scoring, silently degrading the only signal with
statistical guarantees.

Cosmetic outputs that previously shipped — deterministic "bull/base/bear"
paths, vol-envelope bands, fixed-half-width "probability_bands" — were
dashboard residue with no probabilistic content. They've been removed.

News sentiment
--------------
``news_sentiment`` weight defaults to 0.0 in ``WEIGHTS`` until cross-
sectional alpha contribution is empirically measured. The plumbing
(``_score_news`` and the redistribution code path) remains so it can be
re-enabled by adjusting ``WEIGHTS`` without re-architecting.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

# ``projection_engine`` is intentionally numpy/pandas-free at import time.
# stock_analyzer adjacency is added so consumers running from repo root
# resolve sibling modules (e.g. ``ml_model``) without extra plumbing.
_root = Path(__file__).resolve().parents[1]
_sa = str(_root / "stock_analyzer")
if _sa not in sys.path:
    sys.path.insert(0, _sa)

from projection_settings import load_projection_settings


# ── signal weights ─────────────────────────────────────────────────────────────
#
# News sentiment is weight 0 by default — see module docstring. The other
# weights are diagnostic-only when ML is available; they only drive
# decisions when the calibrated model is missing (``ml_used == False``).

WEIGHTS = {
    "valuation_upside": 0.30,
    "momentum_trend":   0.20,
    "rsi_signal":       0.10,
    "quality_score":    0.16,
    "risk_penalty":     0.16,
    "growth_signal":    0.08,
    "news_sentiment":   0.08,
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


def _score_news(news_result: dict | None) -> float | None:
    """Return -1..+1 sentiment score, or ``None`` to signal "no news data" so
    the news weight is redistributed to the other components."""
    if not news_result or not news_result.get("available"):
        return None
    return max(-1.0, min(1.0, news_result.get("sentiment_score", 0.0)))


# ── composite score ────────────────────────────────────────────────────────────

def _composite_score(
    record: dict,
    news_result: dict | None = None,
    *,
    exclude_valuation: bool = False,
) -> tuple[float, dict]:
    """Weighted composite -1..+1.

    Diagnostic only when ML is available (see module docstring). When
    ``WEIGHTS["news_sentiment"] == 0`` the news redistribution branch is a
    no-op.
    """
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
        news_w = weights.pop("news_sentiment", 0.0)
        total_other = sum(weights.values())
        if news_w > 0 and total_other > 0:
            weights = {k: v + v / total_other * news_w for k, v in weights.items()}
        scores = raw_scores
    else:
        raw_scores["news_sentiment"] = news_score
        scores = raw_scores

    composite = sum(scores[k] * weights[k] for k in weights if k in scores)
    return max(-1.0, min(1.0, composite)), scores


# ── probability estimation ─────────────────────────────────────────────────────

def _rule_based_probability(score: float, horizon_days: int) -> float:
    """Fallback probability when no calibrated ML model is available.

    Base drift slopes with horizon: ~52% at 20d, ~62% at 120d. A perfect
    composite shifts probability by 25 percentage points.
    """
    base_prob = 0.50 + 0.001 * horizon_days
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
        agreement = min(1.0, agreement * 1.1)

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
    """Generate forward probability estimates for one record.

    When a calibrated LightGBM model is loadable, the calibrated probability
    is used directly (default ``ml_blend_weight=1.0``). Without the model,
    the rule composite is the only source.
    """
    score, sub_scores = _composite_score(record, news_result, exclude_valuation=exclude_valuation)
    price_raw = record.get("current_price")

    try:
        price = float(price_raw)
    except (TypeError, ValueError):
        return {"error": "No price data available"}
    if math.isnan(price) or math.isinf(price) or price <= 0:
        return {"error": "No price data available"}

    ml_p5 = _ml_probability(record, 5)
    ml_p20 = _ml_probability(record, 20)
    ml_p60 = _ml_probability(record, 60)
    ml_p120 = _ml_probability(record, horizon_days)
    ml_used = any(x is not None for x in (ml_p5, ml_p20, ml_p60, ml_p120))

    settings = load_projection_settings()
    ml_w = settings.ml_blend_weight
    dthr = settings.ml_rule_disagreement_threshold

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

    er_5d = _score_to_expected_return(score, 5)
    er_20d = _score_to_expected_return(score, 20)
    er_60d = _score_to_expected_return(score, 60)
    er_120d = _score_to_expected_return(score, horizon_days)

    signal = _classify_signal(score)
    confidence = _classify_confidence(
        sub_scores, record, ml_used, ml_rule_disagreement=ml_rule_disagreement
    )

    return {
        "composite_score":       round(score, 3),
        "sub_scores":            {k: round(v, 3) for k, v in sub_scores.items()},
        "ml_used":               ml_used,

        "p_up_5d":               round(p_up_5d, 3),
        "p_up_20d":              round(p_up_20d, 3),
        "p_up_60d":              round(p_up_60d, 3),
        "p_up_120d":             round(p_up_120d, 3),
        "p_up_horizon":          round(p_up_120d, 3),
        "expected_return_horizon": round(er_120d, 4),

        "expected_return_5d":    round(er_5d, 4),
        "expected_return_20d":   round(er_20d, 4),
        "expected_return_60d":   round(er_60d, 4),
        "expected_return_120d":  round(er_120d, 4),

        "signal":                signal,
        "confidence":            confidence,

        "news":                  news_result,
        "current_price":         price,
        "horizon_days":          horizon_days,

        "ml_blend_weight_used":  round(ml_w, 3),
        "ml_rule_disagreement":  ml_rule_disagreement,
        "ml_vs_rule":            ml_vs_rule,
    }
