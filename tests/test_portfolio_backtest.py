"""Smoke tests for agent backtest helpers."""

from __future__ import annotations

import unittest
from datetime import date, datetime

from backtesting.strategy_backtest import reconstruct_data_at, reconstruct_price_only_at
from portfolio.backtest import _cagr, _max_drawdown
import pandas as pd


class TestPointInTimeMl(unittest.TestCase):
    def test_price_only_when_fundamentals_sparse(self) -> None:
        import backtesting.strategy_backtest as st

        raw = st.collect_raw_yfinance("AAPL")
        as_of = datetime(2019, 6, 15)
        self.assertIsNone(reconstruct_data_at(raw, as_of))
        data = reconstruct_price_only_at(raw, as_of)
        self.assertIsNotNone(data)
        self.assertGreaterEqual(len(data.get("close_1y") or []), 220)


class TestBacktestMetrics(unittest.TestCase):
    def test_cagr(self) -> None:
        self.assertAlmostEqual(_cagr(1.0, 2.0, 1.0), 1.0)

    def test_mdd(self) -> None:
        s = pd.Series([1.0, 1.2, 0.9, 1.1])
        self.assertLess(_max_drawdown(s), 0)


if __name__ == "__main__":
    unittest.main()
