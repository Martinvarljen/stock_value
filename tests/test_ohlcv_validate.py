"""Unit tests for OHLCV validation (implementation brief §3, §12)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SA = _ROOT / "stock_analyzer"
if str(_SA) not in sys.path:
    sys.path.insert(0, str(_SA))

from ohlcv_validate import validate_ohlcv_dataframe, validate_ohlcv_from_data_dict  # noqa: E402


class TestOhlcvValidate(unittest.TestCase):
    def test_valid_frame_ok(self) -> None:
        ix = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [10, 10.1, 10.2, 10.3, 10.4],
                "high": [10.5, 10.6, 10.7, 10.8, 10.9],
                "low": [9.9, 10.0, 10.1, 10.2, 10.3],
                "close": [10.2, 10.3, 10.4, 10.5, 10.6],
                "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
            },
            index=ix,
        )
        r = validate_ohlcv_dataframe(df)
        self.assertTrue(r["ok"])
        self.assertEqual(r["n_bars"], 5)
        self.assertEqual(len(r["errors"]), 0)

    def test_high_below_body_fails(self) -> None:
        df = pd.DataFrame(
            {
                "open": [10.0],
                "high": [9.5],
                "low": [9.0],
                "close": [9.8],
            }
        )
        r = validate_ohlcv_dataframe(df)
        self.assertFalse(r["ok"])
        self.assertIn("high_below_body", r["errors"])

    def test_unsorted_timestamp_fails(self) -> None:
        ix = pd.to_datetime(["2024-01-03", "2024-01-01"])
        df = pd.DataFrame(
            {
                "open": [10.0, 10.1],
                "high": [10.5, 10.6],
                "low": [9.9, 10.0],
                "close": [10.2, 10.3],
            },
            index=ix,
        )
        r = validate_ohlcv_dataframe(df)
        self.assertFalse(r["ok"])
        self.assertIn("timestamp_not_sorted", r["errors"])

    def test_from_data_dict_mismatch(self) -> None:
        d = {"close_1y": [1, 2, 3], "high_1y": [1.1, 2.1], "low_1y": [0.9, 1.9, 2.9], "open_1y": [1, 2, 3]}
        r = validate_ohlcv_from_data_dict(d)
        self.assertFalse(r["ok"])
        self.assertIn("ohlc_length_mismatch", r["errors"])


if __name__ == "__main__":
    unittest.main()
