"""Unit tests for projection_engine (no network)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "stock_analyzer", _ROOT / "projection"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from projection_engine import (  # noqa: E402
    _composite_score,
    generate_projections,
)


class TestProjectionEngine(unittest.TestCase):
    def test_generate_projections_rejects_bad_price(self) -> None:
        out = generate_projections(
            {"current_price": None, "fair_value_weighted": 100.0},
            horizon_days=120,
        )
        self.assertIn("error", out)

    def test_generate_projections_rejects_nan_price(self) -> None:
        out = generate_projections(
            {"current_price": float("nan"), "fair_value_weighted": 100.0},
            horizon_days=60,
        )
        self.assertIn("error", out)

    def test_horizon_aliases_match_parameterized_fields(self) -> None:
        record = {
            "current_price": 50.0,
            "fair_value_weighted": 55.0,
            "buy_below_price": 40.0,
            "momentum_trend": "UPTREND",
            "ma200": 45.0,
            "rsi14": 50.0,
            "operating_margin": 0.18,
            "roic": 0.12,
            "wacc_data": {"wacc": 0.09},
            "fcf_yield": 0.04,
            "revenue_cagr_5y": 0.06,
            "fcf_cagr_5y": 0.05,
            "critical_flags": [],
            "red_flags": [],
            "net_debt_ebitda": 2.0,
            "beta": 1.0,
            "data_quality_score": 80,
            "momentum_metrics": {"return_3m": {"value": 0.02}},
        }
        out = generate_projections(record, horizon_days=90, news_result=None)
        self.assertNotIn("error", out)
        self.assertEqual(out["horizon_days"], 90)
        self.assertEqual(out["p_up_horizon"], out["p_up_120d"])
        self.assertEqual(out["expected_return_horizon"], out["expected_return_120d"])
        self.assertIn("ml_vs_rule", out)
        self.assertIn("probability_bands", out)
        self.assertIn("ml_blend_weight_used", out)

    def test_composite_score_without_news_redistributes_weights(self) -> None:
        record = {
            "current_price": 40.0,
            "fair_value_weighted": 44.0,
            "momentum_trend": "NEUTRAL",
            "ma200": 40.0,
            "rsi14": 50.0,
            "operating_margin": 0.10,
            "roic": 0.08,
            "wacc_data": {"wacc": 0.09},
            "fcf_yield": 0.03,
            "revenue_cagr_5y": 0.04,
            "fcf_cagr_5y": 0.02,
            "critical_flags": [],
            "red_flags": [],
            "net_debt_ebitda": 2.0,
            "beta": 1.0,
            "data_quality_score": 70,
            "momentum_metrics": {},
        }
        score, sub = _composite_score(record, news_result=None)
        self.assertIsInstance(score, float)
        self.assertTrue(-1.0 <= score <= 1.0)
        self.assertNotIn("news_sentiment", sub)
        self.assertGreater(len(sub), 0)

    def test_exclude_valuation_drops_dcf_from_composite(self) -> None:
        record = {
            "current_price": 50.0,
            "fair_value_weighted": 150.0,
            "momentum_trend": "NEUTRAL",
            "ma200": 50.0,
            "rsi14": 50.0,
            "operating_margin": 0.10,
            "roic": 0.08,
            "wacc_data": {"wacc": 0.09},
            "fcf_yield": 0.03,
            "revenue_cagr_5y": 0.04,
            "fcf_cagr_5y": 0.02,
            "critical_flags": [],
            "red_flags": [],
            "net_debt_ebitda": 2.0,
            "beta": 1.0,
            "data_quality_score": 70,
            "momentum_metrics": {},
        }
        s_with, sub_with = _composite_score(record, None, exclude_valuation=False)
        s_ex, sub_ex = _composite_score(record, None, exclude_valuation=True)
        self.assertEqual(sub_ex.get("valuation_upside"), 0.0)
        self.assertGreater(abs(s_with - s_ex), 1e-6)


if __name__ == "__main__":
    unittest.main()
