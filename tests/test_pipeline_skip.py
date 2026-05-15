"""Tests for pipeline skip rules (mocked data layer)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
_SA = _ROOT / "stock_analyzer"
if str(_SA) not in sys.path:
    sys.path.insert(0, str(_SA))

from pipeline import build_analysis_bundle  # noqa: E402


class TestPipelineSkip(unittest.TestCase):
    @patch("pipeline.collect_data")
    def test_skips_etf(self, mock_collect: MagicMock) -> None:
        mock_collect.return_value = {
            "ticker": "SPY",
            "quote_type": "ETF",
            "error": None,
        }
        bundle, err = build_analysis_bundle("SPY", 0.25, include_explanation=False)
        self.assertIsNone(bundle)
        self.assertIsNotNone(err)
        self.assertIn("ETF", err)

    @patch("pipeline.collect_data")
    def test_skips_on_data_error(self, mock_collect: MagicMock) -> None:
        mock_collect.return_value = {
            "ticker": "INVALID",
            "quote_type": "EQUITY",
            "error": "HTTP 404",
            "current_price": None,
        }
        bundle, err = build_analysis_bundle("INVALID", 0.25, include_explanation=False)
        self.assertIsNone(bundle)
        self.assertIn("Data error", err or "")

    @patch("pipeline.collect_data")
    def test_skips_without_price(self, mock_collect: MagicMock) -> None:
        mock_collect.return_value = {
            "ticker": "NOPRICE",
            "quote_type": "EQUITY",
            "error": None,
            "current_price": None,
        }
        bundle, err = build_analysis_bundle("NOPRICE", 0.25, include_explanation=False)
        self.assertIsNone(bundle)
        self.assertIn("price", (err or "").lower())


if __name__ == "__main__":
    unittest.main()
