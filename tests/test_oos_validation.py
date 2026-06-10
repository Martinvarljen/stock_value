"""Tests for OOS validation helpers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from portfolio.oos_validation import format_yearly_table, write_frozen_config, yearly_performance


class TestYearlyPerformance(unittest.TestCase):
    def test_yearly_beat_spy_flags(self) -> None:
        idx = pd.date_range("2023-01-03", periods=504, freq="B")
        strat = [1.0 + 0.001 * i for i in range(len(idx))]
        spy = [1.0 + 0.0005 * i for i in range(len(idx))]
        df = pd.DataFrame({"strategy": strat, "spy_bh": spy}, index=idx)
        rows = yearly_performance(df)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(r.get("beat_spy") for r in rows))

    def test_yearly_string_index(self) -> None:
        """Backtest index from date column is often object dtype, not DatetimeIndex."""
        dates = [f"2023-{m:02d}-15" for m in range(1, 13)]
        df = pd.DataFrame(
            {"strategy": [1.0 + 0.01 * i for i in range(12)], "spy_bh": [1.0 + 0.005 * i for i in range(12)]},
            index=dates,
        )
        rows = yearly_performance(df)
        self.assertEqual(rows[0]["year"], 2023)


class TestFrozenConfig(unittest.TestCase):
    def test_write_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "frozen.json"
            write_frozen_config(
                p,
                train_through_year=2022,
                oos_from_year=2023,
                cfg={"profile": "research_ls", "min_p_up_long": 0.58},
            )
            self.assertTrue(p.is_file())
            self.assertIn("oos_test_from_year", p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
