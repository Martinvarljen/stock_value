"""Tests for backtest post-run invariant checks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.backtest_invariants import validate_backtest_run  # noqa: E402
class TestBacktestInvariants(unittest.TestCase):
    def test_passes_clean_run(self) -> None:
        curve = pd.DataFrame(
            {"strategy": [1.0, 1.05, 1.02], "spy_bh": [1.0, 1.03, 1.04]},
            index=pd.date_range("2024-01-02", periods=3, freq="B"),
        )
        ledger = [
            {"date": "2024-01-02", "ticker": "AAA", "action": "ENTER_LONG", "side": "long", "price": 100.0},
            {"date": "2024-01-04", "ticker": "AAA", "action": "EXIT", "side": "long", "price": 102.0},
        ]
        snapshots = [{"date": "2024-01-04", "nav": 1.02, "positions": []}]
        cfg = {"risk_limits": {"enabled": True, "max_gross_exposure_pct": 1.5}}
        errs = validate_backtest_run(
            curve=curve,
            ledger=ledger,
            snapshots=snapshots,
            cfg=cfg,
            stats={"strategy_max_dd": -0.03, "max_gross_exposure": 0.5},
        )
        self.assertEqual(errs, [])

    def test_catches_orphan_exit(self) -> None:
        curve = pd.DataFrame({"strategy": [1.0, 0.99]}, index=pd.date_range("2024-01-02", periods=2, freq="B"))
        ledger = [{"date": "2024-01-03", "ticker": "ZZZ", "action": "EXIT", "side": "long", "price": 1.0}]
        errs = validate_backtest_run(
            curve=curve, ledger=ledger, snapshots=[], cfg={}, stats={"strategy_max_dd": -0.01},
        )
        self.assertTrue(any("EXIT without" in e for e in errs))


if __name__ == "__main__":
    unittest.main()
