"""Smoke tests for confirmed swing market structure."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SA = _ROOT / "stock_analyzer"
if str(_SA) not in sys.path:
    sys.path.insert(0, str(_SA))

from market_structure import analyze_market_structure  # noqa: E402


class TestMarketStructure(unittest.TestCase):
    def test_uptrending_synthetic(self) -> None:
        n = 120
        close = (100 + np.linspace(0, 15, n) + np.random.default_rng(42).normal(0, 0.2, n)).tolist()
        high = [c + 0.3 for c in close]
        low = [c - 0.3 for c in close]
        data = {"close_1y": close, "high_1y": high, "low_1y": low}
        r = analyze_market_structure(data, order=3)
        self.assertTrue(r.get("available"))
        self.assertIn("regime_hint", r)
        self.assertGreater(r.get("n_confirmed_pivots", 0), 0)


if __name__ == "__main__":
    unittest.main()
