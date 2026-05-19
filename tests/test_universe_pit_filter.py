"""Tests for PIT universe filtering on daily resolve_tickers."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.universe import resolve_tickers  # noqa: E402


class TestUniversePitFilter(unittest.TestCase):
    def test_explicit_tickers_filtered(self) -> None:
        pit = {"AAPL", "MSFT"}
        with mock.patch("backtesting.sp500_pit_universe.members_as_of", return_value=pit):
            out = resolve_tickers(
                explicit=["AAPL", "GOOG"],
                universe="top100",
                max_tickers=None,
                universe_source="pit_filter",
                as_of=date(2020, 6, 1),
            )
        self.assertEqual(out, ["AAPL"])

    def test_legacy_keeps_all_explicit(self) -> None:
        out = resolve_tickers(
            explicit=["AAPL", "GOOG"],
            universe="top100",
            max_tickers=None,
            universe_source="legacy",
            as_of=date(2020, 6, 1),
        )
        self.assertEqual(out, ["AAPL", "GOOG"])


if __name__ == "__main__":
    unittest.main()
