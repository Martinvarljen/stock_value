"""Tests for regime snapshot helpers."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.regime import build_regime_snapshot, spy_pct_below_ma, spy_trailing_return  # noqa: E402


class TestRegimeSnapshot(unittest.TestCase):
    def test_spy_features_on_synthetic_series(self) -> None:
        idx = pd.date_range("2020-01-01", periods=260, freq="B")
        close = pd.Series([200.0 + i * 0.5 for i in range(len(idx))], index=idx, dtype=float)
        close.iloc[-25:] = [float(close.iloc[-26]) * (1.0 - 0.01 * i) for i in range(1, 26)]
        as_of = idx[-1].to_pydatetime()
        below = spy_pct_below_ma(close, as_of, ma_days=200)
        ret = spy_trailing_return(close, as_of, days=20)
        self.assertIsNotNone(below)
        self.assertIsNotNone(ret)
        self.assertGreater(float(below), 0.0)
        self.assertLess(float(ret), 0.0)
        snap = build_regime_snapshot(close, as_of, bear_scale=0.35)
        self.assertIn("spy_return_20d", snap)
        self.assertEqual(snap["regime_signal"], "bear")


if __name__ == "__main__":
    unittest.main()
