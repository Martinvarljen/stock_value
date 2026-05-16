"""Tests for the multi-factor regime classifier — pure-Python so they
run without numpy/pandas. Each component is exercised in isolation
plus a few composition smoke tests."""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.regime_multifactor import (  # noqa: E402
    RegimeView,
    _score_breadth_component,
    _score_term_component,
    classify_regime,
)


class TestBreadthScore(unittest.TestCase):
    def test_high_breadth_scores_full_risk_on(self) -> None:
        s, det = _score_breadth_component(0.75)
        self.assertEqual(s, 1.0)
        self.assertTrue(det["available"])

    def test_low_breadth_scores_full_risk_off(self) -> None:
        s, _ = _score_breadth_component(0.30)
        self.assertEqual(s, 0.0)

    def test_middle_interpolates(self) -> None:
        s, _ = _score_breadth_component(0.50)
        self.assertAlmostEqual(s, 0.5, places=4)

    def test_none_returns_none(self) -> None:
        s, det = _score_breadth_component(None)
        self.assertIsNone(s)
        self.assertFalse(det["available"])

    def test_out_of_range_returns_none(self) -> None:
        s, _ = _score_breadth_component(1.7)
        self.assertIsNone(s)


class TestTermScore(unittest.TestCase):
    def test_normal_curve_is_risk_on(self) -> None:
        s, _ = _score_term_component(100.0)  # max curve slope
        self.assertEqual(s, 1.0)

    def test_inversion_is_risk_off(self) -> None:
        s, _ = _score_term_component(-50.0)
        self.assertEqual(s, 0.0)

    def test_flat_curve_middle(self) -> None:
        s, _ = _score_term_component(25.0)
        self.assertAlmostEqual(s, 0.50, places=4)


class TestComposition(unittest.TestCase):
    def test_no_components_returns_unknown(self) -> None:
        view = classify_regime(as_of=datetime(2024, 1, 1))
        self.assertEqual(view.label, "unknown")
        self.assertAlmostEqual(view.score, 0.5)
        # gross exposure scale falls back to unknown_scale (=bear_scale).
        self.assertAlmostEqual(view.gross_exposure_scale, view.unknown_scale)

    def test_breadth_plus_term_only_renormalises_weights(self) -> None:
        # No SPY series, no trend/vol -> only breadth+term contribute.
        # Both at extremes -> score should saturate cleanly.
        view = classify_regime(
            as_of=datetime(2024, 1, 1),
            breadth_pct_above_200ma=0.75,        # 1.0 score
            ten_minus_two_slope_bps=120,         # capped at 1.0
        )
        # Both available, both score 1.0 -> total = 1.0.
        self.assertAlmostEqual(view.score, 1.0, places=4)
        self.assertEqual(view.label, "bull")
        # gross exposure = bear_scale + (1 - bear_scale) * 1 = 1.0
        self.assertAlmostEqual(view.gross_exposure_scale, 1.0, places=4)

    def test_breadth_low_term_inverted_drives_bear(self) -> None:
        view = classify_regime(
            as_of=datetime(2024, 1, 1),
            breadth_pct_above_200ma=0.30,
            ten_minus_two_slope_bps=-50.0,
        )
        self.assertAlmostEqual(view.score, 0.0, places=4)
        self.assertEqual(view.label, "bear")
        # gross exposure floor = bear_scale.
        self.assertAlmostEqual(view.gross_exposure_scale, view.bear_scale)

    def test_mixed_components_can_yield_unknown_band(self) -> None:
        # breadth at 0.5 (score=0.5), term at 0bps (score=0.333). Mean
        # of two equal weights = 0.4167 — sits in the unknown band
        # (0.40, 0.55).
        view = classify_regime(
            as_of=datetime(2024, 1, 1),
            breadth_pct_above_200ma=0.50,
            ten_minus_two_slope_bps=0.0,
        )
        self.assertEqual(view.label, "unknown")


class TestRegimeView(unittest.TestCase):
    def test_to_json_roundtrips_components(self) -> None:
        view = classify_regime(
            as_of=datetime(2024, 1, 1),
            breadth_pct_above_200ma=0.55,
        )
        out = view.to_json()
        self.assertIn("score", out)
        self.assertIn("label", out)
        self.assertIn("breadth", out["components"])
        self.assertEqual(out["components"]["breadth"]["pct_above_200ma"], 0.55)


if __name__ == "__main__":
    unittest.main()
