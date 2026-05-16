"""Deterministic outcome reflection.

Same role as primerjava's ``Reflector`` (``tradingagents/graph/reflection.py``):
turn the realised outcome of a past decision into a short prose lesson that
can be re-read later. Two key differences:

  - **No LLM**: same inputs always produce identical text, so the memory loop
    is reproducible. (The primerjava version is non-deterministic and was
    one of the main institutional-scrutiny concerns in our comparison.)
  - **Quant-grounded**: the prose references the actual numeric drivers
    (alpha sign, magnitude, exit reason, ML calibration delta, regime).

The output is the string written to ``REFLECTION:`` blocks in the
:mod:`portfolio.memory_log`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OutcomeContext:
    """Numeric + categorical context needed to write a reflection.

    Fields are intentionally primitive so the reflection function stays
    pure: same struct -> same string, every time.
    """

    ticker: str
    trade_date: str
    rating: str  # 5-tier (Buy / Overweight / Hold / Underweight / Sell)
    action: str  # Action.value (ENTER_LONG / EXIT / NO_TRADE / ...)
    raw_return: float  # decimal, e.g. 0.0345 = +3.45%
    alpha_return: float  # raw - benchmark over same window
    holding_days: int
    benchmark: str = "SPY"
    p_up_20d: float | None = None  # model probability at decision time
    ml_score: float | None = None
    regime_scale: float | None = None
    spy_bull: bool | None = None
    exit_reason: str | None = None  # only meaningful for EXIT actions


_LONG_ACTIONS = {"ENTER_LONG", "HOLD", "Buy", "Overweight"}
_SHORT_ACTIONS = {"ENTER_SHORT", "Sell", "Underweight"}


def _direction_of(ctx: OutcomeContext) -> int:
    """+1 if the decision was directionally long, -1 if short, 0 if flat."""
    if ctx.action in _LONG_ACTIONS or ctx.rating in {"Buy", "Overweight"}:
        return 1
    if ctx.action in _SHORT_ACTIONS or ctx.rating in {"Sell", "Underweight"}:
        return -1
    return 0


def _ml_calibration_note(ctx: OutcomeContext) -> str:
    """Compare the realised direction to the model's probability at entry."""
    if ctx.p_up_20d is None:
        return ""
    realised_up = ctx.raw_return > 0
    p = float(ctx.p_up_20d)
    confidence = abs(p - 0.5) * 2.0
    if confidence < 0.10:
        return f"Model was near-flat (P(up)={p:.0%}); outcome carries little signal about calibration."
    if (realised_up and p >= 0.5) or ((not realised_up) and p <= 0.5):
        return f"Model called direction correctly (P(up)={p:.0%}; realised {'+' if realised_up else '−'})."
    return f"Model mis-called direction (P(up)={p:.0%}; realised {'+' if realised_up else '−'}) — calibration debt."


def _exit_reason_note(ctx: OutcomeContext) -> str:
    if not ctx.exit_reason:
        return ""
    r = ctx.exit_reason.lower()
    if "stop" in r:
        return "Stop fired — risk limit did its job, but check for over-tight stop on noise."
    if "take-profit" in r or "take profit" in r:
        return "Take-profit captured — confirm the target was anchored on regime-realistic ATR / vol."
    if "max hold" in r:
        return "Time exit — signal failed to convert into PnL inside the horizon; may need shorter or sharper rule."
    if "p(up)" in r or "p_up" in r:
        return "ML-driven exit — verify model retraining cadence to avoid stale calibration."
    if "critical" in r or "flag" in r:
        return "Risk-flag exit — fundamental override saved exposure; keep flag taxonomy current."
    return f"Exit reason: {ctx.exit_reason}."


def _alpha_magnitude_note(alpha: float, bench: str) -> str:
    a = abs(alpha)
    if a < 0.005:
        return f"Effectively flat vs {bench} (~{alpha:+.1%})."
    if a < 0.02:
        return f"Marginal alpha vs {bench} ({alpha:+.1%}) — inside noise band."
    if a < 0.05:
        return f"Meaningful alpha vs {bench} ({alpha:+.1%})."
    return f"Large alpha vs {bench} ({alpha:+.1%}) — confirm it's not a single-name outlier."


def _regime_note(ctx: OutcomeContext) -> str:
    if ctx.regime_scale is None and ctx.spy_bull is None:
        return ""
    parts: list[str] = []
    if ctx.spy_bull is not None:
        parts.append("risk-on" if ctx.spy_bull else "risk-off")
    if ctx.regime_scale is not None:
        parts.append(f"gross scale {ctx.regime_scale:.0%}")
    return f"Regime at entry: {', '.join(parts)}." if parts else ""


def reflect_on_outcome(ctx: OutcomeContext) -> str:
    """Return a 2–4 sentence deterministic reflection.

    Identical inputs always produce identical output — required for an
    auditable memory log and for snapshot-based tests.
    """
    direction = _direction_of(ctx)
    raw = ctx.raw_return
    alpha = ctx.alpha_return
    correct_call = (direction == 1 and raw > 0) or (direction == -1 and raw < 0) or (
        direction == 0 and abs(raw) < 0.01
    )

    if direction == 0:
        verdict = (
            f"Flat call held: {ctx.ticker} returned {raw:+.1%} over {ctx.holding_days}d, "
            f"alpha {alpha:+.1%} vs {ctx.benchmark}."
        )
    elif correct_call:
        verdict = (
            f"Directional call was right: {ctx.action} on {ctx.ticker} returned "
            f"{raw:+.1%} over {ctx.holding_days}d (alpha {alpha:+.1%} vs {ctx.benchmark})."
        )
    else:
        verdict = (
            f"Directional call was wrong: {ctx.action} on {ctx.ticker} returned "
            f"{raw:+.1%} over {ctx.holding_days}d (alpha {alpha:+.1%} vs {ctx.benchmark})."
        )

    sentences = [verdict]

    mag = _alpha_magnitude_note(alpha, ctx.benchmark)
    if mag:
        sentences.append(mag)

    ml = _ml_calibration_note(ctx)
    if ml:
        sentences.append(ml)

    exit_note = _exit_reason_note(ctx)
    if exit_note:
        sentences.append(exit_note)

    regime = _regime_note(ctx)
    if regime:
        sentences.append(regime)

    # Cap to 4 sentences so the log stays compact when re-injected as context.
    sentences = sentences[:4]
    return " ".join(sentences)
