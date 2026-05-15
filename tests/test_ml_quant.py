"""Tests for ML quintile and regime helpers."""

import unittest
from datetime import datetime

import pandas as pd

from backtesting.ml_quant import aggregate_quintile_forward_returns, assign_quintile, ml_score_from_signal
from backtesting.regime import gross_exposure_scale, spy_bull_regime


class TestMlQuant(unittest.TestCase):
    def test_quintile_monotonic_synthetic(self) -> None:
        signals = []
        for i in range(50):
            sc = i / 50.0
            signals.append({
                "ml_score": sc,
                "p_up_20d": sc,
                "fwd_6m": sc * 0.2 - 0.05,
            })
        q = aggregate_quintile_forward_returns(signals, horizon_months=6)
        self.assertGreater(q[5]["avg_fwd"], q[1]["avg_fwd"])

    def test_ml_score_prefers_p_up(self) -> None:
        self.assertEqual(ml_score_from_signal({"p_up_20d": 0.7, "composite_score": 0.1}), 0.7)


class TestRegime(unittest.TestCase):
    def test_spy_bull_and_bear(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=250, freq="B")
        bull = pd.Series(range(100, 350), index=idx, dtype=float)
        self.assertTrue(spy_bull_regime(bull, idx[-1], ma_days=200))
        self.assertEqual(gross_exposure_scale(bull, idx[-1], bear_scale=0.3), 1.0)
        bear_vals = [300.0] * 200 + [float(80 - i * 0.5) for i in range(50)]
        bear = pd.Series(bear_vals, index=idx, dtype=float)
        self.assertFalse(spy_bull_regime(bear, idx[-1], ma_days=200))
        self.assertAlmostEqual(gross_exposure_scale(bear, idx[-1], bear_scale=0.3), 0.3)


if __name__ == "__main__":
    unittest.main()
