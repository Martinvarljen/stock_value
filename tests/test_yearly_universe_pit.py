"""Tests for PIT integration in yearly top-100 universe builder."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.yearly_top100_universe import (  # noqa: E402
    build_top_n_for_year,
    default_universe_cache_dir,
    normalize_universe_source,
    write_ticker_lines,
)


class TestYearlyUniversePit(unittest.TestCase):
    def test_normalize_universe_source(self) -> None:
        self.assertEqual(normalize_universe_source("pit"), "pit")
        self.assertEqual(normalize_universe_source("legacy"), "legacy")
        self.assertEqual(normalize_universe_source("sp500_pit"), "pit")

    def test_cache_dirs_differ(self) -> None:
        leg = default_universe_cache_dir(_ROOT, "legacy")
        pit = default_universe_cache_dir(_ROOT, "pit")
        self.assertNotEqual(leg, pit)
        self.assertIn("pit", str(pit))

    def test_build_top_n_uses_pit_top_n(self) -> None:
        with mock.patch("backtesting.sp500_pit_universe.pit_top_n", return_value=["AAPL", "MSFT"]) as m:
            out = build_top_n_for_year(2022, universe_source="pit", top_n=2, verbose=False)
        m.assert_called_once()
        self.assertEqual(out, ["AAPL", "MSFT"])

    def test_year_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_ticker_lines(d / "2023.txt", ["AAPL", "MSFT"])
            from backtesting.yearly_top100_universe import read_ticker_lines

            self.assertEqual(read_ticker_lines(d / "2023.txt"), ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
