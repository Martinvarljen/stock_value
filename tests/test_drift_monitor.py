"""Tests for the feature-drift monitor — pure-Python, no numpy.

Stable distributions should report PSI < 0.10 / KS < 0.10. Shifted
distributions should breach the high threshold (PSI >= 0.25)."""

from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "projection"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from projection.ml_model.drift_monitor import (  # noqa: E402
    DriftMonitor,
    ks_statistic,
    population_stability_index,
)


class TestPSI(unittest.TestCase):
    def test_identical_samples_psi_close_to_zero(self) -> None:
        random.seed(0)
        a = [random.gauss(0, 1) for _ in range(2000)]
        b = list(a)
        psi = population_stability_index(a, b)
        self.assertLess(psi, 0.01)

    def test_shifted_samples_psi_high(self) -> None:
        random.seed(1)
        a = [random.gauss(0, 1) for _ in range(2000)]
        b = [random.gauss(2.5, 1) for _ in range(2000)]
        psi = population_stability_index(a, b)
        self.assertGreater(psi, 0.25)


class TestKS(unittest.TestCase):
    def test_identical_distributions_ks_small(self) -> None:
        random.seed(2)
        a = [random.gauss(0, 1) for _ in range(2000)]
        b = [random.gauss(0, 1) for _ in range(2000)]
        ks = ks_statistic(a, b)
        self.assertLess(ks, 0.1)

    def test_shifted_distribution_ks_large(self) -> None:
        random.seed(3)
        a = [random.gauss(0, 1) for _ in range(2000)]
        b = [random.gauss(0, 1) + 1.5 for _ in range(2000)]
        ks = ks_statistic(a, b)
        self.assertGreater(ks, 0.3)


class TestDriftMonitorRoundtrip(unittest.TestCase):
    def test_fit_save_load_compute_stable(self) -> None:
        random.seed(4)
        train = {
            "rsi_14": [random.gauss(50, 15) for _ in range(2000)],
            "atr_pct": [abs(random.gauss(0.02, 0.005)) for _ in range(2000)],
        }
        monitor = DriftMonitor.fit(train)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            monitor.save(path)
            loaded = DriftMonitor.load(path)
            actual = {
                "rsi_14": [random.gauss(50, 15) for _ in range(500)],
                "atr_pct": [abs(random.gauss(0.02, 0.005)) for _ in range(500)],
            }
            report = loaded.compute(actual)
            self.assertEqual(report.severity, "ok")
            for r in report.rows:
                self.assertLess(r.psi, 0.25)
        finally:
            path.unlink()

    def test_compute_detects_high_severity_on_shift(self) -> None:
        random.seed(5)
        train = {"rsi_14": [random.gauss(50, 15) for _ in range(2000)]}
        monitor = DriftMonitor.fit(train)
        actual = {"rsi_14": [random.gauss(80, 5) for _ in range(500)]}  # right tail only
        report = monitor.compute(actual)
        self.assertEqual(report.severity, "high")
        self.assertEqual(report.rows[0].severity, "high")
        # Out-of-range should also flag.
        self.assertGreater(report.rows[0].out_of_range_pct, 0.30)


if __name__ == "__main__":
    unittest.main()
