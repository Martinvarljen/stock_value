import tempfile
import unittest
from pathlib import Path

from backtesting.yearly_top100_universe import (
    load_universe_map_for_lag_years,
    normalize_yahoo_symbol,
    read_ticker_lines,
    write_ticker_lines,
)


class TestYearlyUniverse(unittest.TestCase):
    def test_normalize_yahoo_symbol(self) -> None:
        self.assertEqual(normalize_yahoo_symbol("brk.b"), "BRK-B")

    def test_read_write_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2024.txt"
            write_ticker_lines(p, ["AAPL", "MSFT", "# ignore", "", "GOOG"])
            got = read_ticker_lines(p)
            self.assertEqual(got, ["AAPL", "MSFT", "GOOG"])

    def test_load_universe_map(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            write_ticker_lines(base / "2023.txt", ["XOM", "CVX"])
            write_ticker_lines(base / "2024.txt", ["AAPL"])
            m = load_universe_map_for_lag_years([2024, 2023], base, auto_build_missing=False, verbose=False)
            self.assertEqual(m[2023], ["XOM", "CVX"])
            self.assertEqual(m[2024], ["AAPL"])


if __name__ == "__main__":
    unittest.main()
