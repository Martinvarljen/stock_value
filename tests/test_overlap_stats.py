"""Tests for overlap-adjusted effective sample sizes."""

from __future__ import annotations

import unittest
from datetime import datetime

from backtesting.overlap_stats import (
    effective_n_for_signals,
    effective_n_greedy,
    overlap_inflation_factor,
)


class TestOverlapStats(unittest.TestCase):
    def test_greedy_non_overlapping_6m(self) -> None:
        # Four quarterly checkpoints within one year → only ~2 non-overlapping 6M windows
        dates = [
            datetime(2023, 1, 1),
            datetime(2023, 4, 1),
            datetime(2023, 7, 1),
            datetime(2023, 10, 1),
        ]
        self.assertEqual(effective_n_greedy(dates, 6), 2)

    def test_effective_n_sums_tickers(self) -> None:
        signals = [
            {"ticker": "A", "date": datetime(2023, 1, 1), "fwd_6m": 0.1},
            {"ticker": "A", "date": datetime(2023, 4, 1), "fwd_6m": 0.2},
            {"ticker": "B", "date": datetime(2023, 1, 1), "fwd_6m": 0.05},
            {"ticker": "B", "date": datetime(2023, 4, 1), "fwd_6m": -0.01},
        ]
        # Jan + Apr are <6M apart → one non-overlapping window per ticker → 2 total
        self.assertEqual(effective_n_for_signals(signals, 6), 2.0)

    def test_overlap_factor(self) -> None:
        self.assertAlmostEqual(overlap_inflation_factor(8, 4.0), 2.0**0.5)
        self.assertIsNone(overlap_inflation_factor(0, 0))


if __name__ == "__main__":
    unittest.main()
