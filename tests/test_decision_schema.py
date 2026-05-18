"""Tests for portfolio.decision_schema."""

from __future__ import annotations

import unittest
from datetime import date

from portfolio.decision_schema import (
    RATINGS_5_TIER,
    DecisionReport,
    action_to_rating,
    parse_rating,
    render_decision,
)
from portfolio.decisions import Action, TickerDecision


class TestParseRating(unittest.TestCase):
    def test_explicit_rating_label(self) -> None:
        self.assertEqual(parse_rating("**Rating**: Buy\nMore text"), "Buy")
        self.assertEqual(parse_rating("Rating - Sell\n..."), "Sell")

    def test_falls_back_to_first_rating_word(self) -> None:
        text = "Some prose. We recommend a moderate Overweight position."
        self.assertEqual(parse_rating(text), "Overweight")

    def test_default_when_missing(self) -> None:
        self.assertEqual(parse_rating("no rating word here"), "Hold")
        self.assertEqual(parse_rating("none", default="Underweight"), "Underweight")

    def test_all_ratings_are_recognised(self) -> None:
        for r in RATINGS_5_TIER:
            self.assertEqual(parse_rating(f"Rating: {r}"), r)


class TestActionToRating(unittest.TestCase):
    def test_long_short_flat(self) -> None:
        self.assertEqual(action_to_rating(Action.ENTER_LONG, has_position=False), "Buy")
        self.assertEqual(action_to_rating(Action.ENTER_SHORT, has_position=False), "Sell")
        self.assertEqual(action_to_rating(Action.EXIT, has_position=True), "Underweight")
        self.assertEqual(action_to_rating(Action.HOLD, has_position=True), "Overweight")

    def test_no_trade_uses_p_up(self) -> None:
        self.assertEqual(action_to_rating(Action.NO_TRADE, has_position=False, p_up=0.7), "Overweight")
        self.assertEqual(action_to_rating(Action.NO_TRADE, has_position=False, p_up=0.3), "Underweight")
        self.assertEqual(action_to_rating(Action.NO_TRADE, has_position=False, p_up=0.5), "Hold")


class TestDecisionReport(unittest.TestCase):
    def _make(self) -> DecisionReport:
        td = TickerDecision(
            ticker="aapl",
            action=Action.ENTER_LONG,
            reason="Q5 P(up)20d=64%",
            ml_score=0.71,
            quintile=5,
            p_up_20d=0.64,
            price=189.34,
        )
        return DecisionReport.from_decision(
            td,
            trade_date=date(2026, 5, 16),
            regime={"spy_bull": True, "gross_exposure_scale": 1.0},
            had_position=False,
            past_context="[2026-03-01 | AAPL | Buy | raw +5.0% | alpha +1.2% | 20d]\nDirectional call was right.",
        )

    def test_round_trip(self) -> None:
        rep = self._make()
        self.assertEqual(rep.ticker, "AAPL")
        self.assertEqual(rep.rating, "Buy")
        self.assertEqual(rep.action, "ENTER_LONG")
        d = rep.to_dict()
        self.assertEqual(d["ticker"], "AAPL")
        self.assertEqual(d["rating"], "Buy")

    def test_render_is_deterministic_and_parseable(self) -> None:
        rep = self._make()
        md1 = render_decision(rep)
        md2 = render_decision(rep)
        self.assertEqual(md1, md2)
        self.assertEqual(parse_rating(md1), "Buy")
        self.assertIn("ml_score=0.710", md1)
        self.assertIn("p_up_20d=64.00%", md1)
        self.assertIn("Past context", md1)

    def test_extras_render_as_named_sections(self) -> None:
        td = TickerDecision(
            ticker="AAPL",
            action=Action.ENTER_LONG,
            reason="signal",
            ml_score=0.5,
            quintile=4,
            p_up_20d=0.6,
            price=180.0,
        )
        rep = DecisionReport.from_decision(
            td,
            trade_date=date(2026, 5, 16),
            regime={"spy_bull": True, "gross_exposure_scale": 1.0},
            had_position=False,
            extras={
                "explanation": "Trades 25% below DCF fair value with high ROIC.",
                "classification": "BUY",
            },
        )
        md = render_decision(rep)
        self.assertIn("**explanation**: Trades 25% below DCF fair value", md)
        self.assertIn("**classification**: BUY", md)

    def test_trade_setup_extras_are_human_readable(self) -> None:
        """Lock in the shape ``daily_run`` flattens ``trade_setup`` into."""
        td = TickerDecision(
            ticker="SHEL",
            action=Action.ENTER_LONG,
            reason="ml signal",
            ml_score=0.71,
            quintile=5,
            p_up_20d=0.64,
            price=72.5,
        )
        rep = DecisionReport.from_decision(
            td,
            trade_date=date(2026, 5, 16),
            regime={"spy_bull": True, "gross_exposure_scale": 1.0},
            had_position=False,
            extras={
                "setup_bias": "trend up; MACD bullish; structure hint: bull",
                "watch_levels": "Buy-below (valuation)=68.40, Fair value=85.10, Fib 0.382=70.20",
            },
        )
        md = render_decision(rep)
        self.assertIn("**setup_bias**: trend up; MACD bullish", md)
        self.assertIn("**watch_levels**: Buy-below (valuation)=68.40", md)
        self.assertIn("Fair value=85.10", md)
        self.assertEqual(parse_rating(md), "Buy")


if __name__ == "__main__":
    unittest.main()
