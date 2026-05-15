"""Tests for Dolt feather helpers (no live Dolt required)."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from projection.data.dolt_source import (
    dolt_to_yahoo_symbol,
    top_liquidity_tickers,
    ticker_histories_from_feather,
)


class TestDoltSource(unittest.TestCase):
    def test_symbol_maps(self):
        self.assertEqual(dolt_to_yahoo_symbol("BRK.B"), "BRK-B")

    def test_top_liquidity(self):
        rows = []
        for sym, vol in [("AAA", 1e6), ("BBB", 5e6), ("CCC", 2e6)]:
            rows.append({"act_symbol": sym, "date": "2023-06-01", "close": 10.0, "volume": vol})
        df = pd.DataFrame(rows)
        top = top_liquidity_tickers(df, 2023, top_n=2)
        self.assertEqual(top[0], "BBB")
        self.assertEqual(len(top), 2)

    def test_feather_histories(self):
        dates = pd.date_range("2022-01-01", periods=300, freq="B")
        rows = []
        for d in dates:
            rows.append({
                "act_symbol": "AAPL",
                "date": d,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1e6,
            })
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.feather"
            df.to_feather(path)
            hists = ticker_histories_from_feather(
                ["AAPL"],
                dates[0].to_pydatetime(),
                dates[-1].to_pydatetime(),
                path,
            )
        self.assertIn("AAPL", hists)
        self.assertIn("Close", hists["AAPL"].columns)
        self.assertGreaterEqual(len(hists["AAPL"]), 50)


if __name__ == "__main__":
    unittest.main()
