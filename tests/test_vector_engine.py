"""Tests for vector backtest engine (next-bar open, costs)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.vector_engine import run_vector_backtest  # noqa: E402


class TestVectorEngine(unittest.TestCase):
    def test_flat_price_zero_signal(self) -> None:
        n = 30
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        ohlc = pd.DataFrame({"open": np.ones(n) * 100.0}, index=idx)
        sig = np.zeros(n)
        r = run_vector_backtest(ohlc, sig, commission_bps=0, slippage_bps=0)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(float(r["equity"][-1]), 1.0, places=6)

    def test_long_rails_match_buy_hold_open(self) -> None:
        n = 20
        idx = pd.date_range("2020-01-01", periods=n, freq="B")
        opens = 100.0 * np.cumprod(np.concatenate([[1.0], np.repeat(1.001, n - 1)]))
        ohlc = pd.DataFrame({"open": opens}, index=idx)
        sig = np.ones(n)
        r = run_vector_backtest(ohlc, sig, commission_bps=0, slippage_bps=0)
        self.assertTrue(r["ok"])
        o = np.asarray(opens, dtype=float)
        expected = float(o[-1] / o[1])
        self.assertAlmostEqual(float(r["equity"][-1]), expected, places=5)


if __name__ == "__main__":
    unittest.main()
