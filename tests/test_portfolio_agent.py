"""Tests for daily agent decision rules."""

from __future__ import annotations

import unittest
from datetime import date

from portfolio.decisions import Action, decide_ticker, prioritize_entries
from portfolio.store import Position, PortfolioState


class TestDecisions(unittest.TestCase):
    def test_enter_long_top_quintile(self) -> None:
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 100.0,
            "p_up_20d": 0.65,
            "ml_score": 0.65,
            "critical_flags": [],
        }
        regime = {"spy_bull": True, "gross_exposure_scale": 1.0}
        cfg = {"min_p_up_long": 0.60, "long_quintile_min": 4, "long_p_up_quintile_5_floor": 0.55, "regime_filter": True}
        d = decide_ticker(analysis, None, quintile=5, regime=regime, cfg=cfg, as_of=date(2026, 5, 15))
        self.assertEqual(d.action, Action.ENTER_LONG)

    def test_exit_low_p_up(self) -> None:
        pos = Position(
            ticker="AAA",
            side="long",
            entry_date="2026-05-01",
            entry_price=100.0,
            notional=0.1,
            stop_price=90.0,
            take_profit_price=125.0,
            max_hold_days=25,
        )
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 102.0,
            "p_up_20d": 0.40,
            "ml_score": 0.40,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis,
            pos,
            quintile=2,
            regime={"spy_bull": False, "gross_exposure_scale": 0.5},
            cfg={
                "exit_p_up_long": 0.45,
                "score_exit_long_only_bear_regime": False,
                "min_hold_days_before_score_exit_long": 0,
                "max_hold_days": 25,
                "estimated_hold_days": 20,
            },
            as_of=date(2026, 5, 15),
        )
        self.assertEqual(d.action, Action.EXIT)

    def test_entry_cap(self) -> None:
        from portfolio.decisions import TickerDecision

        decs = [
            TickerDecision("A", Action.ENTER_LONG, "a", ml_score=0.9),
            TickerDecision("B", Action.ENTER_LONG, "b", ml_score=0.8),
            TickerDecision("C", Action.ENTER_LONG, "c", ml_score=0.7),
        ]
        out = prioritize_entries(decs, {"max_new_entries_per_day": 2})
        allowed = sum(1 for d in out if d.action == Action.ENTER_LONG)
        self.assertEqual(allowed, 2)


if __name__ == "__main__":
    unittest.main()
