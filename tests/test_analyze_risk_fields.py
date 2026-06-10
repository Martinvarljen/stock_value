"""Ensure analyze_ticker exposes sector/beta and extended technical risk fields."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio import analyze as pa  # noqa: E402


class TestAnalyzeRiskFields(unittest.TestCase):
    @patch("portfolio.analyze.generate_projections")
    @patch("portfolio.analyze.build_analysis_bundle")
    def test_sector_beta_and_bars_from_record(self, mock_bundle, mock_proj) -> None:
        record = {
            "company_name": "Test Co",
            "current_price": 100.0,
            "classification": "Medium",
            "momentum_trend": "UPTREND",
            "sector": "Technology",
            "beta": 1.25,
            "extended_technicals": {
                "atr_14": {"pct_of_price": 0.025},
                "realised_vol_60d_annual": 0.30,
                "last_bar": {"low": 98.0, "high": 102.0, "open": 99.5},
            },
            "critical_flags": [],
        }
        bundle = MagicMock()
        bundle.record = record
        mock_bundle.return_value = (bundle, None)
        mock_proj.return_value = {
            "p_up_5d": 0.55,
            "p_up_20d": 0.62,
            "p_up_60d": 0.58,
            "composite_score": 0.6,
            "signal": "LEAN_BULLISH",
            "confidence": "MEDIUM",
            "ml_used": True,
            "expected_return_20d": 0.03,
        }

        out = pa.analyze_ticker("AAA", include_explanation=False)
        assert out is not None
        self.assertTrue(out["ok"])
        self.assertEqual(out["sector"], "Technology")
        self.assertAlmostEqual(out["beta"], 1.25)
        self.assertAlmostEqual(out["atr_pct"], 0.025)
        self.assertAlmostEqual(out["bar_low"], 98.0)
        self.assertAlmostEqual(out["bar_high"], 102.0)


if __name__ == "__main__":
    unittest.main()
