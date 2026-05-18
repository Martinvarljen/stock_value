"""Tests for the pluggable PIT fundamentals layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "stock_analyzer"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fundamentals_source import (  # noqa: E402
    CSVPointInTimeSource,
    FundamentalsSourceNotConfigured,
    SimFinSource,
    YfinanceRestatedSource,
    get_fundamentals_source,
)


class TestCSVPointInTimeSource(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        ticker_dir = self.root / "AAPL"
        ticker_dir.mkdir()
        (ticker_dir / "2018-09-29.json").write_text(json.dumps({
            "revenue": 265.6e9, "operating_margin": 0.27,
        }))
        (ticker_dir / "2019-09-28.json").write_text(json.dumps({
            "revenue": 260.2e9, "operating_margin": 0.245,
        }))
        (ticker_dir / "2020-09-26.json").write_text(json.dumps({
            "revenue": 274.5e9, "operating_margin": 0.241,
        }))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_returns_most_recent_filing_before_cutoff(self) -> None:
        src = CSVPointInTimeSource(root=self.root, reporting_lag_days=90)
        # On 2020-04-01 we should see the FY2019 filing
        # (lag-cutoff = 2020-04-01 - 90d = 2020-01-02; the 2019-09-28
        # filing is the most recent <= cutoff).
        out = src.get_as_of("AAPL", date(2020, 4, 1))
        self.assertEqual(out["fiscal_period_end"], "2019-09-28")
        self.assertAlmostEqual(out["operating_margin"], 0.245)

    def test_filters_lookahead(self) -> None:
        # On 2018-09-30 the FY2018 filing has just ended its quarter
        # (2018-09-29). With 90-day lag the cutoff is 2018-07-02 — the
        # FY2018 row at 2018-09-29 must NOT be returned.
        src = CSVPointInTimeSource(root=self.root, reporting_lag_days=90)
        out = src.get_as_of("AAPL", date(2018, 9, 30))
        # No filing predates the cutoff in our fixture, so we expect
        # an "no_pit_filing_before" sentinel rather than the 2018 row.
        self.assertIn("error", out)
        self.assertIn("no_pit_filing_before", out["error"])

    def test_missing_ticker_raises_typed_error(self) -> None:
        src = CSVPointInTimeSource(root=self.root)
        with self.assertRaises(FundamentalsSourceNotConfigured):
            src.get_as_of("ZZZZ", date(2020, 1, 1))

    def test_is_pit_true(self) -> None:
        src = CSVPointInTimeSource(root=self.root)
        self.assertTrue(src.is_pit())


class TestYfinanceRestatedSource(unittest.TestCase):
    def test_is_pit_false(self) -> None:
        # The whole point of this adapter is to flag itself as NOT PIT.
        self.assertFalse(YfinanceRestatedSource().is_pit())


class TestSimFinSourceStub(unittest.TestCase):
    def test_raises_without_api_key(self) -> None:
        # No SIMFIN_API_KEY in env (test sandbox) -> not configured.
        import os
        prior = os.environ.pop("SIMFIN_API_KEY", None)
        try:
            src = SimFinSource()
            with self.assertRaises(FundamentalsSourceNotConfigured):
                src.get_as_of("AAPL", date(2020, 1, 1))
        finally:
            if prior is not None:
                os.environ["SIMFIN_API_KEY"] = prior

    def test_raises_when_stub_called_with_key(self) -> None:
        # With a key but unimplemented body, we still raise so callers
        # can't accidentally rely on the stub returning empty data.
        src = SimFinSource(api_key="dummy")
        with self.assertRaises(FundamentalsSourceNotConfigured):
            src.get_as_of("AAPL", date(2020, 1, 1))


class TestFactory(unittest.TestCase):
    def test_csv_pit_requires_root(self) -> None:
        with self.assertRaises(FundamentalsSourceNotConfigured):
            get_fundamentals_source("csv_pit")

    def test_csv_pit_with_root(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = get_fundamentals_source("csv_pit", root=d)
            self.assertEqual(src.name(), f"csv_pit:{Path(d)}")

    def test_yfinance_restated(self) -> None:
        src = get_fundamentals_source("yfinance_restated")
        self.assertEqual(src.name(), "yfinance_restated")
        self.assertFalse(src.is_pit())

    def test_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            get_fundamentals_source("does_not_exist")


if __name__ == "__main__":
    unittest.main()
