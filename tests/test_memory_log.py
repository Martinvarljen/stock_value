"""Tests for portfolio.memory_log (no network, no LLM)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from portfolio.decision_schema import DecisionReport
from portfolio.decisions import Action, TickerDecision
from portfolio.memory_log import DecisionMemoryLog


def _make_report(ticker: str, trade_date: date, action: Action = Action.ENTER_LONG) -> DecisionReport:
    td = TickerDecision(
        ticker=ticker,
        action=action,
        reason="unit test",
        ml_score=0.62,
        quintile=5,
        p_up_20d=0.61,
        price=100.0,
    )
    return DecisionReport.from_decision(
        td,
        trade_date=trade_date,
        regime={"spy_bull": True, "gross_exposure_scale": 1.0},
        had_position=False,
    )


class TestDecisionMemoryLog(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log = DecisionMemoryLog(Path(self._tmp.name) / "memory.md")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_store_then_load_pending(self) -> None:
        rep = _make_report("AAPL", date(2026, 5, 16))
        self.assertTrue(self.log.store_decision(rep))
        entries = self.log.load_entries()
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0].pending)
        self.assertEqual(entries[0].ticker, "AAPL")
        self.assertEqual(entries[0].rating, "Buy")

    def test_store_decision_is_idempotent(self) -> None:
        rep = _make_report("AAPL", date(2026, 5, 16))
        self.assertTrue(self.log.store_decision(rep))
        self.assertFalse(self.log.store_decision(rep))
        self.assertEqual(len(self.log.load_entries()), 1)

    def test_update_with_outcome_resolves_tag(self) -> None:
        rep = _make_report("AAPL", date(2026, 5, 16))
        self.log.store_decision(rep)
        ok = self.log.update_with_outcome(
            ticker="AAPL",
            trade_date="2026-05-16",
            raw_return=0.0345,
            alpha_return=0.012,
            holding_days=20,
            reflection="Directional call was right.",
        )
        self.assertTrue(ok)
        e = self.log.load_entries()[0]
        self.assertFalse(e.pending)
        self.assertEqual(e.raw_return, "+3.5%")
        self.assertEqual(e.alpha_return, "+1.2%")
        self.assertEqual(e.holding_days, "20d")
        self.assertIn("Directional call was right.", e.reflection_md)

    def test_batch_update_only_resolves_matching(self) -> None:
        self.log.store_decision(_make_report("AAPL", date(2026, 5, 16)))
        self.log.store_decision(_make_report("MSFT", date(2026, 5, 16)))
        n = self.log.batch_update_with_outcomes([
            {
                "ticker": "AAPL",
                "trade_date": "2026-05-16",
                "raw_return": 0.05,
                "alpha_return": 0.02,
                "holding_days": 15,
                "reflection": "OK",
            },
            {
                "ticker": "GOOGL",  # not in log
                "trade_date": "2026-05-16",
                "raw_return": -0.01,
                "alpha_return": -0.005,
                "holding_days": 10,
                "reflection": "Missing entry; should not match.",
            },
        ])
        self.assertEqual(n, 1)
        statuses = {e.ticker: e.pending for e in self.log.load_entries()}
        self.assertFalse(statuses["AAPL"])
        self.assertTrue(statuses["MSFT"])

    def test_past_context_returns_resolved_only(self) -> None:
        self.log.store_decision(_make_report("AAPL", date(2026, 5, 1)))
        self.log.update_with_outcome(
            ticker="AAPL",
            trade_date="2026-05-01",
            raw_return=0.03,
            alpha_return=0.01,
            holding_days=14,
            reflection="Right call.",
        )
        # Still pending entry should be ignored by past_context
        self.log.store_decision(_make_report("AAPL", date(2026, 5, 16)))
        ctx = self.log.get_past_context("AAPL")
        self.assertIn("Past resolved decisions for AAPL", ctx)
        self.assertIn("Right call.", ctx)
        # Pending entries by definition do not appear
        self.assertNotIn("2026-05-16", ctx)

    def test_rotation_drops_oldest_resolved(self) -> None:
        log = DecisionMemoryLog(Path(self._tmp.name) / "memory_rot.md", max_entries=2)
        for i, tk in enumerate(["AAA", "BBB", "CCC", "DDD"]):
            log.store_decision(_make_report(tk, date(2026, 5, 10 + i)))
            log.update_with_outcome(
                ticker=tk,
                trade_date=f"2026-05-{10 + i:02d}",
                raw_return=0.01 * i,
                alpha_return=0.005 * i,
                holding_days=10,
                reflection="x",
            )
        tickers = [e.ticker for e in log.load_entries()]
        # Oldest two rotated out; newest two kept.
        self.assertEqual(tickers, ["CCC", "DDD"])


if __name__ == "__main__":
    unittest.main()
