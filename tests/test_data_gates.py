"""Unit tests for the OHLCV integrity gate.

Pure stdlib; no numpy/pandas imports so the suite runs in any sandbox.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.data_gates import filter_for_bad_ohlcv, is_ohlcv_ok


def _good(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "ok": True,
        "ohlcv_quality": {"ok": True, "errors": [], "warnings": [], "n_bars": 252},
    }


def _bad(ticker: str, errors: list[str]) -> dict:
    return {
        "ticker": ticker,
        "ok": True,
        "ohlcv_quality": {"ok": False, "errors": errors, "warnings": [], "n_bars": 5},
    }


class TestIsOhlcvOk(unittest.TestCase):
    def test_missing_quality_is_ok(self) -> None:
        ok, reasons = is_ohlcv_ok({"ticker": "AAA", "ok": True})
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_clean_quality_is_ok(self) -> None:
        ok, reasons = is_ohlcv_ok(_good("AAA"))
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_fatal_high_below_body_is_rejected(self) -> None:
        ok, reasons = is_ohlcv_ok(_bad("AAA", ["high_below_body"]))
        self.assertFalse(ok)
        self.assertEqual(reasons, ["high_below_body"])

    def test_fatal_timestamp_not_sorted_is_rejected(self) -> None:
        ok, reasons = is_ohlcv_ok(_bad("AAA", ["timestamp_not_sorted"]))
        self.assertFalse(ok)
        self.assertIn("timestamp_not_sorted", reasons)

    def test_warnings_only_are_kept(self) -> None:
        a = {
            "ticker": "AAA",
            "ok": True,
            "ohlcv_quality": {
                "ok": True,
                "errors": [],
                "warnings": ["zero_range_bars:3", "volume_missing_or_misaligned_skipped"],
                "n_bars": 252,
            },
        }
        ok, reasons = is_ohlcv_ok(a)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])

    def test_no_close_1y_is_fatal(self) -> None:
        ok, reasons = is_ohlcv_ok(_bad("AAA", ["no_close_1y"]))
        self.assertFalse(ok)
        self.assertEqual(reasons, ["no_close_1y"])

    def test_all_nan_with_column_suffix_is_fatal(self) -> None:
        ok, reasons = is_ohlcv_ok(_bad("AAA", ["all_nan:close"]))
        self.assertFalse(ok)
        self.assertEqual(reasons, ["all_nan:close"])

    def test_unknown_error_with_ok_false_still_rejects(self) -> None:
        ok, reasons = is_ohlcv_ok(_bad("AAA", ["weird_new_error_type"]))
        self.assertFalse(ok)
        self.assertEqual(reasons, ["weird_new_error_type"])


class TestFilterForBadOhlcv(unittest.TestCase):
    def test_drops_only_unhealthy(self) -> None:
        analyses = [
            _good("AAA"),
            _bad("BBB", ["high_below_body"]),
            _good("CCC"),
            _bad("DDD", ["ohlc_length_mismatch", "all_nan:open"]),
        ]
        kept, dropped = filter_for_bad_ohlcv(analyses)
        self.assertEqual([a["ticker"] for a in kept], ["AAA", "CCC"])
        self.assertEqual([t for t, _ in dropped], ["BBB", "DDD"])
        self.assertEqual(dropped[1][1], ["ohlc_length_mismatch", "all_nan:open"])

    def test_failed_analyses_pass_through(self) -> None:
        analyses = [
            {"ticker": "ZZZ", "ok": False, "error": "ETFs not supported"},
            _good("AAA"),
            _bad("BBB", ["timestamp_not_sorted"]),
        ]
        kept, dropped = filter_for_bad_ohlcv(analyses)
        kept_tickers = [a["ticker"] for a in kept]
        self.assertIn("ZZZ", kept_tickers)
        self.assertIn("AAA", kept_tickers)
        self.assertNotIn("BBB", kept_tickers)
        self.assertEqual(len(dropped), 1)


if __name__ == "__main__":
    unittest.main()
