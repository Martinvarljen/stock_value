"""Smoke tests for institutional Sharpe machinery in performance_metrics.

The DSR / PSR implementations follow Bailey & Lopez de Prado (2014). We
don't try to match scipy down to the last decimal — we sanity-check
monotonicities, edge cases, and that risk-free subtraction lowers Sharpe.
"""

from __future__ import annotations

import math
import unittest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

if _HAS_NUMPY:
    from backtesting.performance_metrics import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
        summarize_backtest,
        t_stat_of_sharpe,
    )


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed in this environment")
class TestPerformanceMetrics(unittest.TestCase):
    def _build(self, mean_per_day=0.0008, vol_per_day=0.01, n=2520, seed=42):
        rng = np.random.default_rng(seed)
        r = rng.normal(mean_per_day, vol_per_day, size=n)
        eq = np.empty(n + 1)
        eq[0] = 1.0
        for i in range(n):
            eq[i + 1] = eq[i] * (1.0 + r[i])
        return r, eq

    def test_rf_subtraction_lowers_sharpe(self) -> None:
        r, eq = self._build()
        s0 = summarize_backtest(r, eq, periods_per_year=252.0, risk_free_rate_annual=0.0)
        s4 = summarize_backtest(r, eq, periods_per_year=252.0, risk_free_rate_annual=0.04)
        self.assertGreater(s0["sharpe"], s4["sharpe"])

    def test_t_stat_scales_with_years(self) -> None:
        # Same Sharpe, twice as many periods -> larger t-stat by sqrt(2).
        t1 = t_stat_of_sharpe(1.0, n_periods=252, periods_per_year=252.0)
        t2 = t_stat_of_sharpe(1.0, n_periods=504, periods_per_year=252.0)
        self.assertAlmostEqual(t2 / t1, math.sqrt(2.0), places=6)

    def test_psr_in_unit_interval(self) -> None:
        r, _ = self._build()
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
        self.assertTrue(0.0 <= psr <= 1.0)

    def test_psr_high_for_strong_signal(self) -> None:
        r, _ = self._build(mean_per_day=0.0015, vol_per_day=0.005, n=2520)
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
        self.assertGreater(psr, 0.95)

    def test_psr_low_for_zero_signal(self) -> None:
        r, _ = self._build(mean_per_day=0.0, vol_per_day=0.01, n=2520, seed=7)
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=1.0)
        self.assertLess(psr, 0.5)

    def test_dsr_lower_than_psr_with_many_trials(self) -> None:
        r, _ = self._build()
        psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
        dsr_few = deflated_sharpe_ratio(r, n_trials=2)
        dsr_many = deflated_sharpe_ratio(r, n_trials=200)
        self.assertGreaterEqual(psr, dsr_few - 1e-6)
        self.assertGreaterEqual(dsr_few, dsr_many - 1e-6)

    def test_summary_includes_new_fields(self) -> None:
        r, eq = self._build()
        out = summarize_backtest(r, eq, n_trials_for_dsr=50)
        self.assertIn("sharpe_t_stat", out)
        self.assertIn("psr_vs_zero", out)
        self.assertIn("dsr", out)
        self.assertIn("risk_free_rate_annual", out)

    def test_summary_no_dsr_when_no_trials(self) -> None:
        r, eq = self._build()
        out = summarize_backtest(r, eq)
        self.assertIsNone(out["dsr"])


if __name__ == "__main__":
    unittest.main()
