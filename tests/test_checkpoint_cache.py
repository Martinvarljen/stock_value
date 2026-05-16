"""Tests for backtesting.checkpoint_cache."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from backtesting.checkpoint_cache import BacktestCheckpointCache


class TestCheckpointCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "cp.db"
        self.cache = BacktestCheckpointCache(self.path, strategy_mode="ml", schema_version="v1")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_miss_then_hit(self) -> None:
        cp = datetime(2024, 6, 28)
        self.assertIsNone(self.cache.get("AAPL", cp))
        self.cache.put("AAPL", cp, {"classification": "BUY", "fwd_6m": 0.05})
        row = self.cache.get("AAPL", cp)
        self.assertIsNotNone(row)
        self.assertEqual(row["classification"], "BUY")
        self.assertAlmostEqual(row["fwd_6m"], 0.05)

    def test_normalises_ticker_case_and_date_tz(self) -> None:
        cp_naive = datetime(2024, 6, 28)
        self.cache.put("aapl", cp_naive, {"x": 1})
        row = self.cache.get("AAPL", cp_naive)
        self.assertIsNotNone(row)
        self.assertEqual(row["x"], 1)

    def test_isolation_by_strategy_mode(self) -> None:
        cp = datetime(2024, 6, 28)
        dcf = BacktestCheckpointCache(self.path, strategy_mode="dcf", schema_version="v1")
        self.cache.put("AAPL", cp, {"mode": "ml"})
        self.assertIsNone(dcf.get("AAPL", cp))
        self.assertEqual(self.cache.count(), 1)
        self.assertEqual(dcf.count(), 0)

    def test_isolation_by_schema_version(self) -> None:
        cp = datetime(2024, 6, 28)
        v2 = BacktestCheckpointCache(self.path, strategy_mode="ml", schema_version="v2")
        self.cache.put("AAPL", cp, {"v": 1})
        self.assertIsNone(v2.get("AAPL", cp))

    def test_overwrite_on_put(self) -> None:
        cp = datetime(2024, 6, 28)
        self.cache.put("AAPL", cp, {"v": 1})
        self.cache.put("AAPL", cp, {"v": 2})
        self.assertEqual(self.cache.get("AAPL", cp)["v"], 2)
        self.assertEqual(self.cache.count(), 1)

    def test_clear_scoped(self) -> None:
        cp = datetime(2024, 6, 28)
        dcf = BacktestCheckpointCache(self.path, strategy_mode="dcf", schema_version="v1")
        self.cache.put("AAPL", cp, {"x": 1})
        dcf.put("AAPL", cp, {"x": 2})
        dropped = self.cache.clear()
        self.assertEqual(dropped, 1)
        self.assertEqual(self.cache.count(), 0)
        self.assertEqual(dcf.count(), 1)


if __name__ == "__main__":
    unittest.main()
