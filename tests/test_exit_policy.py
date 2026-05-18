"""Exit policy — regime score exit, min hold, TP vs trail."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decisions import Action, decide_ticker
from portfolio.exit_policy import (
    long_entry_allowed,
    long_score_exit_threshold,
    short_entry_allowed,
    short_score_exit_threshold,
    take_profit_enabled,
)
from portfolio.store import Position
from types import SimpleNamespace


CFG = {
    "exit_p_up_long": 0.30,
    "score_exit_long_only_bear_regime": True,
    "min_hold_days_before_score_exit_long": 25,
    "max_hold_days": 28,
    "estimated_hold_days": 20,
    "use_trailing_stop": True,
    "trailing_stop_pct": 0.12,
    "use_take_profit": False,
    "take_profit_pct": 0.35,
}


class TestExitPolicy(unittest.TestCase):
    def test_no_score_exit_in_bull_regime(self) -> None:
        self.assertIsNone(
            long_score_exit_threshold({"spy_bull": True}, CFG),
        )

    def test_score_exit_in_bear_regime(self) -> None:
        self.assertAlmostEqual(
            long_score_exit_threshold({"spy_bull": False}, CFG),
            0.30,
        )

    def test_tp_off_when_trailing_without_override(self) -> None:
        self.assertFalse(take_profit_enabled(CFG))

    def test_bull_holds_despite_low_p_up(self) -> None:
        pos = Position(
            ticker="AAA",
            side="long",
            entry_date="2026-01-01",
            entry_price=100.0,
            notional=1000.0,
            stop_price=80.0,
            take_profit_price=1e12,
            max_hold_days=28,
        )
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 105.0,
            "p_up_20d": 0.32,
            "ml_score": 0.32,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis,
            pos,
            quintile=2,
            regime={"spy_bull": True, "gross_exposure_scale": 1.0},
            cfg=CFG,
            as_of=date(2026, 1, 20),
        )
        self.assertEqual(d.action, Action.HOLD)

    def test_bear_exits_after_min_hold(self) -> None:
        pos = Position(
            ticker="AAA",
            side="long",
            entry_date="2025-12-01",
            entry_price=100.0,
            notional=1000.0,
            stop_price=80.0,
            take_profit_price=1e12,
            max_hold_days=35,
        )
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 95.0,
            "p_up_20d": 0.30,
            "ml_score": 0.30,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis,
            pos,
            quintile=2,
            regime={"spy_bull": False, "gross_exposure_scale": 0.5},
            cfg=CFG,
            as_of=date(2026, 1, 15),
        )
        self.assertEqual(d.action, Action.EXIT)

    def test_bear_no_score_exit_before_min_hold(self) -> None:
        pos = Position(
            ticker="AAA",
            side="long",
            entry_date="2026-01-10",
            entry_price=100.0,
            notional=1000.0,
            stop_price=80.0,
            take_profit_price=1e12,
            max_hold_days=35,
        )
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 95.0,
            "p_up_20d": 0.30,
            "ml_score": 0.30,
            "critical_flags": [],
        }
        cfg = {**CFG, "exit_long_when_regime_not_bull": False}
        d = decide_ticker(
            analysis,
            pos,
            quintile=2,
            regime={"spy_bull": False, "regime_signal": "bear", "gross_exposure_scale": 0.5},
            cfg=cfg,
            as_of=date(2026, 1, 20),
        )
        self.assertEqual(d.action, Action.HOLD)


    def test_long_only_when_regime_bull(self) -> None:
        cfg = {"long_entry_requires_bull_regime": True}
        self.assertFalse(long_entry_allowed({"regime_signal": "bear"}, cfg, 0.35))
        self.assertFalse(long_entry_allowed({"regime_signal": "unknown"}, cfg, 0.35))
        self.assertTrue(long_entry_allowed({"regime_signal": "bull"}, cfg, 1.0))

    def test_exit_long_when_not_bull(self) -> None:
        from portfolio.exit_policy import should_exit_long_on_regime

        cfg = {"exit_long_when_regime_not_bull": True, "long_entry_requires_bull_regime": True}
        self.assertTrue(should_exit_long_on_regime({"regime_signal": "bear"}, cfg))
        self.assertFalse(should_exit_long_on_regime({"regime_signal": "bull"}, cfg))

    def test_short_only_when_regime_bear(self) -> None:
        cfg = {"enable_short": True, "short_entry_requires_bear_regime": True}
        self.assertFalse(short_entry_allowed({"regime_signal": "bull"}, cfg))
        self.assertFalse(short_entry_allowed({"regime_signal": "unknown"}, cfg))
        self.assertTrue(short_entry_allowed({"regime_signal": "bear"}, cfg))

    def test_strict_short_requires_stress(self) -> None:
        cfg = {
            "enable_short": True,
            "short_entry_requires_bear_regime": True,
            "short_requires_full_risk_off": True,
            "bear_scale": 0.35,
            "short_min_spy_below_ma_pct": 0.02,
            "short_max_spy_return_20d": -0.04,
        }
        weak = {
            "regime_signal": "bear",
            "gross_exposure_scale": 0.35,
            "spy_pct_below_ma200": 0.01,
            "spy_return_20d": -0.02,
        }
        self.assertFalse(short_entry_allowed(weak, cfg))
        strong = {
            "regime_signal": "bear",
            "gross_exposure_scale": 0.35,
            "spy_pct_below_ma200": 0.03,
            "spy_return_20d": -0.06,
        }
        self.assertTrue(short_entry_allowed(strong, cfg))

    def test_short_cover_threshold_tied_to_entry(self) -> None:
        pos = SimpleNamespace(p_up_20d_at_entry=0.35)
        cfg = {
            "exit_p_up_short": 0.48,
            "short_exit_p_up_relative_to_entry": True,
            "short_exit_p_up_delta": 0.10,
        }
        self.assertAlmostEqual(short_score_exit_threshold(pos, cfg), 0.45)


if __name__ == "__main__":
    unittest.main()
