"""Regression tests for strategy_backtest / dynamic backtest methodology fixes."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import pandas as pd

from backtesting import strategy_backtest as st
from backtesting.dynamic_portfolio_backtest import _next_trading_row, _norm_day


class TestForwardPrice(unittest.TestCase):
    def _hist(self, start: str, n: int) -> pd.DataFrame:
        idx = pd.bdate_range(start, periods=n)
        return pd.DataFrame({"Close": range(100, 100 + n)}, index=idx)

    def test_returns_none_when_history_too_short(self) -> None:
        hist = self._hist("2024-01-02", 30)
        cp = datetime(2024, 1, 2)
        self.assertIsNone(st._get_forward_price(hist, cp, months=6))

    def test_no_fallback_to_latest_bar(self) -> None:
        hist = self._hist("2024-01-02", 120)
        cp = datetime(2024, 1, 2)
        # Only ~4 months of history after checkpoint — 6M forward should fail.
        self.assertIsNone(st._get_forward_price(hist, cp, months=6))

    def test_returns_price_near_target(self) -> None:
        hist = self._hist("2023-01-03", 400)
        cp = datetime(2023, 1, 3)
        px = st._get_forward_price(hist, cp, months=6)
        self.assertIsNotNone(px)
        target = cp + timedelta(days=6 * 30.44)
        bar = hist[hist.index >= pd.Timestamp(target)].iloc[0]
        self.assertEqual(px, float(bar["Close"]))


class TestSharesAtCheckpoint(unittest.TestCase):
    def test_prefers_balance_sheet_shares_over_info(self) -> None:
        raw = {
            "ticker": "TST",
            "info": {"sharesOutstanding": 1_000_000, "currency": "USD"},
            "income_statement": pd.DataFrame(
                {"2023-12-31": [1e9], "2022-12-31": [9e8]},
                index=["Total Revenue"],
            ),
            "balance_sheet": pd.DataFrame(
                {
                    "2023-12-31": [500_000],
                    "2022-12-31": [480_000],
                },
                index=["Share Issued"],
            ),
            "cash_flow": pd.DataFrame({"2023-12-31": [0]}, index=["Operating Cash Flow"]),
            "price_history": pd.DataFrame(
                {"Close": [100.0]},
                index=pd.DatetimeIndex([pd.Timestamp("2024-06-01")]),
            ),
        }
        as_of = datetime(2024, 6, 1)
        data = st.reconstruct_data_at(raw, as_of)
        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["shares_outstanding"], 500_000.0)
        self.assertEqual(data["market_cap"], 50_000_000.0)


class TestNextTradingRow(unittest.TestCase):
    def test_next_session_after_signal_day(self) -> None:
        idx = pd.bdate_range("2024-01-02", periods=5)
        df = pd.DataFrame({"Open": [10, 11, 12, 13, 14]}, index=idx)
        nxt = _next_trading_row(df, _norm_day(idx[1]))
        self.assertIsNotNone(nxt)
        assert nxt is not None
        self.assertEqual(float(nxt["Open"]), 12.0)


if __name__ == "__main__":
    unittest.main()
