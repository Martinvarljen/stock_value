"""Daily agent and backtest share the same decision core."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decisions import (  # noqa: E402
    Action,
    decide_ticker,
    decide_universe,
    prioritize_entries,
)
from portfolio.store import PortfolioState  # noqa: E402


class TestDailyBacktestParity(unittest.TestCase):
    def _sample_analysis(self, ticker: str = "AAA") -> dict:
        return {
            "ticker": ticker,
            "ok": True,
            "price": 100.0,
            "p_up_20d": 0.72,
            "ml_score": 0.72,
            "critical_flags": [],
            "sector": "Technology",
            "beta": 1.1,
            "vol_60d_annual": 0.25,
        }

    def test_decide_universe_matches_decide_ticker(self) -> None:
        regime = {"spy_bull": True, "gross_exposure_scale": 1.0}
        cfg = {
            "min_p_up_long": 0.58,
            "long_quintile_min": 4,
            "long_p_up_quintile_5_floor": 0.54,
            "regime_filter": True,
            "max_new_entries_per_day": 3,
        }
        analyses = [self._sample_analysis("AAA"), self._sample_analysis("BBB")]
        state = PortfolioState()
        as_of = date(2024, 6, 15)
        batch = decide_universe(analyses, state, regime, cfg, as_of)
        for a in analyses:
            single = decide_ticker(
                a, None, quintile=5, regime=regime, cfg=cfg, as_of=as_of,
            )
            batch_one = next(d for d in batch if d.ticker == a["ticker"])
            self.assertEqual(single.action, batch_one.action, msg=a["ticker"])

    def test_prioritize_entries_cap(self) -> None:
        cfg = {"max_new_entries_per_day": 1}
        decisions = [
            decide_ticker(self._sample_analysis("A"), None, quintile=5,
                          regime={"spy_bull": True, "gross_exposure_scale": 1.0},
                          cfg={"min_p_up_long": 0.58, "long_quintile_min": 4,
                               "long_p_up_quintile_5_floor": 0.54, "regime_filter": True},
                          as_of=date(2024, 1, 10)),
            decide_ticker(self._sample_analysis("B"), None, quintile=5,
                          regime={"spy_bull": True, "gross_exposure_scale": 1.0},
                          cfg={"min_p_up_long": 0.58, "long_quintile_min": 4,
                               "long_p_up_quintile_5_floor": 0.54, "regime_filter": True},
                          as_of=date(2024, 1, 10)),
        ]
        out = prioritize_entries(decisions, cfg)
        entries = [d for d in out if d.action == Action.ENTER_LONG]
        self.assertEqual(len(entries), 1)


if __name__ == "__main__":
    unittest.main()
