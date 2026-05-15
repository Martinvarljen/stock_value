"""Checkpoint schedule: quarter-end vs month-end."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stock_analyzer"))

import backtesting.strategy_backtest as st  # noqa: E402


class TestCheckpointFreq(unittest.TestCase):
    def test_monthly_has_more_dates_than_quarterly(self) -> None:
        q = st._generate_checkpoints(4, "Q")
        m = st._generate_checkpoints(4, "M")
        self.assertGreater(len(m), len(q))
        self.assertGreaterEqual(len(m), len(q) * 2)

    def test_normalize_freq(self) -> None:
        self.assertEqual(st._normalize_checkpoint_freq(None), "Q")
        self.assertEqual(st._normalize_checkpoint_freq("M"), "M")
        self.assertEqual(st._normalize_checkpoint_freq("monthly"), "M")
        self.assertEqual(st._normalize_checkpoint_freq("Q"), "Q")


if __name__ == "__main__":
    unittest.main()
