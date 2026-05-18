"""Structured decision report for the daily / backtest pipelines.

Ports the *structured-output* idea from primerjava (TradingAgents schemas) but
keeps everything deterministic and stdlib-only:

  - ``DecisionReport`` is a plain dataclass — no LLM, no pydantic dependency.
  - ``Action`` (from :mod:`portfolio.decisions`) is mapped onto the same
    5-tier rating scale primerjava uses (Buy / Overweight / Hold /
    Underweight / Sell), so the memory log and any downstream reporting
    share one canonical vocabulary.
  - ``render_decision()`` produces a stable markdown shape suitable for
    storage in :mod:`portfolio.memory_log` and for human review in daily
    snapshots — the rating line is parseable with :func:`parse_rating`.

The mapping is intentionally conservative: an ML signal that fires an
``ENTER_LONG`` becomes ``Buy``, a ``HOLD`` for an existing long becomes
``Overweight`` (still positive), a passive ``NO_TRADE`` becomes ``Hold``,
and any ``EXIT`` becomes ``Underweight`` (reducing risk) or ``Sell`` when
the analysis itself is bearish.

This module is deliberately read-only on the rest of the system — it does
not modify decisions, it only renders + classifies them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Tuple

from portfolio.decisions import Action, TickerDecision


# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy",
    "Overweight",
    "Hold",
    "Underweight",
    "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from rendered decision text.

    Two-pass strategy (matches primerjava semantics):

    1. Look for an explicit ``Rating: X`` label (tolerant of markdown bold).
    2. Fall back to the first 5-tier word found anywhere in the text.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default


def action_to_rating(action: Action, *, has_position: bool, p_up: float | None = None) -> str:
    """Deterministic mapping from a daily-agent ``Action`` to the 5-tier rating.

    - ``ENTER_LONG`` -> Buy
    - ``ENTER_SHORT`` -> Sell
    - ``HOLD``        -> Overweight (long held) / Underweight (short held)
    - ``EXIT``        -> Underweight (long) / Overweight (short cover)
    - ``NO_TRADE``    -> Hold (default) or biased by p_up if available
    """
    if action == Action.ENTER_LONG:
        return "Buy"
    if action == Action.ENTER_SHORT:
        return "Sell"
    if action == Action.HOLD:
        return "Overweight" if not has_position else "Overweight"
    if action == Action.EXIT:
        return "Underweight"
    if p_up is not None:
        if p_up >= 0.58:
            return "Overweight"
        if p_up <= 0.42:
            return "Underweight"
    return "Hold"


@dataclass
class DecisionReport:
    """Structured wrapper around a ``TickerDecision`` for storage / display.

    Every field that ends up in a memory-log tag is a primitive (str / float
    / int / None) so the tag line is grep-friendly and JSON-serialisable.
    """

    ticker: str
    trade_date: str  # ISO date
    action: str
    rating: str
    reason: str
    price: float | None = None
    ml_score: float | None = None
    p_up_20d: float | None = None
    quintile: int | None = None
    regime_scale: float | None = None
    spy_bull: bool | None = None
    had_position: bool = False
    past_context: str = ""
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_decision(
        cls,
        d: TickerDecision,
        *,
        trade_date: date,
        regime: dict | None = None,
        had_position: bool = False,
        past_context: str = "",
        extras: dict | None = None,
    ) -> "DecisionReport":
        regime = regime or {}
        rating = action_to_rating(d.action, has_position=had_position, p_up=d.p_up_20d)
        return cls(
            ticker=d.ticker.upper(),
            trade_date=trade_date.isoformat(),
            action=d.action.value,
            rating=rating,
            reason=d.reason,
            price=d.price,
            ml_score=d.ml_score,
            p_up_20d=d.p_up_20d,
            quintile=d.quintile,
            regime_scale=regime.get("gross_exposure_scale"),
            spy_bull=regime.get("spy_bull"),
            had_position=had_position,
            past_context=past_context or "",
            extras=dict(extras or {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def render_decision(report: DecisionReport) -> str:
    """Render a ``DecisionReport`` to deterministic markdown.

    Shape mirrors primerjava's ``render_pm_decision`` so the memory log can
    grep a ``**Rating**`` line back out, but the contents are quant-grade
    (action, score, p_up, regime) rather than prose.
    """
    lines: list[str] = [
        f"**Rating**: {report.rating}",
        "",
        f"**Ticker**: {report.ticker}  |  **Date**: {report.trade_date}  |  **Action**: {report.action}",
    ]

    facts: list[str] = []
    if report.price is not None:
        facts.append(f"price={report.price:.4f}")
    if report.ml_score is not None:
        facts.append(f"ml_score={report.ml_score:.3f}")
    if report.p_up_20d is not None:
        facts.append(f"p_up_20d={report.p_up_20d:.2%}")
    if report.quintile is not None:
        facts.append(f"quintile=Q{report.quintile}")
    if report.regime_scale is not None:
        regime_word = "risk-on" if report.spy_bull else "risk-off"
        facts.append(f"regime={regime_word}@{report.regime_scale:.0%}")
    if report.had_position:
        facts.append("had_position=yes")
    if facts:
        lines.extend(["", "**Signal**: " + " | ".join(facts)])

    lines.extend(["", f"**Reason**: {report.reason}"])

    for key, value in report.extras.items():
        lines.append(f"**{key}**: {value}")

    if report.past_context:
        lines.extend(["", "**Past context**:", report.past_context.strip()])

    return "\n".join(lines)
