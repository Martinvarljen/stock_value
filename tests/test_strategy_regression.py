"""Frozen decision scenarios — strategy logic must not drift unintentionally."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decisions import Action, decide_ticker, decide_universe, prioritize_entries  # noqa: E402
from portfolio.store import PortfolioState, Position  # noqa: E402


def _bull_regime() -> dict:
    return {"spy_bull": True, "regime_signal": "bull", "gross_exposure_scale": 1.0}


def _bear_regime() -> dict:
    return {"spy_bull": False, "regime_signal": "bear", "gross_exposure_scale": 0.35}


def _base_cfg() -> dict:
    return {
        "min_p_up_long": 0.58,
        "long_quintile_min": 4,
        "long_p_up_quintile_5_floor": 0.54,
        "max_p_up_short": 0.32,
        "short_quintile_max": 1,
        "regime_filter": True,
        "long_entry_requires_bull_regime": True,
        "short_entry_requires_bear_regime": True,
        "enable_short": True,
        "max_new_entries_per_day": 3,
        "estimated_hold_days": 20,
        "max_hold_days": 25,
        "min_p_up_long_abs_buffer": 0.04,
        "max_p_up_short_abs_buffer": 0.04,
        "long_entry_min_regime_scale": 0.35,
        "score_exit_long_only_bear_regime": True,
        "min_hold_days_before_score_exit_long": 20,
    }


class TestStrategyRegression(unittest.TestCase):
    def test_q5_long_entry_in_bull(self) -> None:
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 100.0,
            "p_up_20d": 0.62,
            "ml_score": 0.62,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis, None, quintile=5, regime=_bull_regime(),
            cfg=_base_cfg(), as_of=date(2024, 6, 15),
        )
        self.assertEqual(d.action, Action.ENTER_LONG)

    def test_short_blocked_in_bull(self) -> None:
        analysis = {
            "ticker": "ZZZ",
            "ok": True,
            "price": 50.0,
            "p_up_20d": 0.20,
            "ml_score": 0.20,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis, None, quintile=1, regime=_bull_regime(),
            cfg=_base_cfg(), as_of=date(2024, 6, 15),
        )
        self.assertNotEqual(d.action, Action.ENTER_SHORT)

    def test_hold_long_with_stop_not_hit(self) -> None:
        pos = Position(
            ticker="AAA",
            side="long",
            entry_date="2024-06-01",
            entry_price=100.0,
            notional=5000.0,
            stop_price=84.0,
            take_profit_price=122.0,
            max_hold_days=25,
            margin=1000.0,
        )
        analysis = {
            "ticker": "AAA",
            "ok": True,
            "price": 105.0,
            "p_up_20d": 0.55,
            "ml_score": 0.55,
            "critical_flags": [],
        }
        d = decide_ticker(
            analysis, pos, quintile=4, regime=_bull_regime(),
            cfg=_base_cfg(), as_of=date(2024, 6, 10),
        )
        self.assertEqual(d.action, Action.HOLD)

    def test_prioritize_entries_respects_cap(self) -> None:
        cfg = {**_base_cfg(), "max_new_entries_per_day": 1}
        analyses = [
            {
                "ticker": f"T{i}",
                "ok": True,
                "price": 10.0,
                "p_up_20d": 0.70 - i * 0.01,
                "ml_score": 0.70 - i * 0.01,
                "critical_flags": [],
            }
            for i in range(3)
        ]
        state = PortfolioState()
        as_of = date(2024, 6, 15)
        decisions = decide_universe(analyses, state, _bull_regime(), cfg, as_of)
        decisions = prioritize_entries(decisions, cfg)
        entries = [d for d in decisions if d.action == Action.ENTER_LONG]
        self.assertLessEqual(len(entries), 1)


if __name__ == "__main__":
    unittest.main()
