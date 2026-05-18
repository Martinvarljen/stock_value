"""Tests for the walk-forward retraining harness — pure-Python parts.

Window enumeration, calibration metrics, and drift detection don't
need numpy/pandas. The actual training orchestration is exercised via
mocks elsewhere; here we cover correctness of the helpers since they
gate every walk-forward run.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "projection"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from projection.ml_model.walk_forward import (  # noqa: E402
    brier_score,
    detect_calibration_drift,
    log_loss,
    reliability_curve,
    reliability_linear_fit,
    walk_forward_windows,
)


class TestWindowEnumeration(unittest.TestCase):
    def test_basic_walk_forward_step(self) -> None:
        wins = list(walk_forward_windows(
            date(2018, 1, 1), date(2024, 12, 31),
            train_years=5, val_months=6, step_months=6, embargo_days=5,
        ))
        # 5y train + 6m val each; window 1: train 2018-01-01..2022-12-31,
        # val 2023-01-06..2023-07-05.
        self.assertGreater(len(wins), 0)
        w0 = wins[0]
        self.assertEqual(w0.train_start, date(2018, 1, 1))
        self.assertEqual(w0.train_end, date(2022, 12, 31))
        self.assertEqual(w0.val_start, date(2023, 1, 6))
        # Each window slides by 6 months.
        if len(wins) >= 2:
            self.assertEqual(wins[1].train_start, date(2018, 7, 1))

    def test_stops_when_val_end_exceeds_global_end(self) -> None:
        # 5y train (2020-01..2024-12) + 5d embargo + 6m val pushes
        # val_end past end=2025-06-30 (lands at 2025-07-04). Use a
        # later end to fit one window.
        wins = list(walk_forward_windows(
            date(2020, 1, 1), date(2025, 7, 31),
            train_years=5, val_months=6, step_months=6,
        ))
        self.assertEqual(len(wins), 1)


class TestCalibrationMetrics(unittest.TestCase):
    def test_brier_perfect_is_zero(self) -> None:
        self.assertAlmostEqual(brier_score([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]), 0.0)

    def test_brier_random_is_quarter(self) -> None:
        # Predicting 0.5 against balanced labels -> Brier = 0.25.
        self.assertAlmostEqual(
            brier_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]), 0.25,
        )

    def test_log_loss_perfect_is_near_zero(self) -> None:
        self.assertLess(log_loss([1, 0, 1, 0], [0.99, 0.01, 0.99, 0.01]), 0.05)

    def test_reliability_curve_shape(self) -> None:
        # int(0.1 * 10) = 1, int(0.15*10) = 1 -> both in bin "0.1-0.2".
        y = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        p = [0.1, 0.4, 0.2, 0.6, 0.55, 0.05, 0.7, 0.15, 0.85, 0.95]
        bins = reliability_curve(y, p, bins=10)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins[1]["n"], 2)   # 0.1, 0.15
        self.assertEqual(bins[4]["n"], 1)   # 0.4
        self.assertEqual(bins[9]["n"], 1)   # 0.95
        self.assertEqual(sum(b["n"] for b in bins), 10)

    def test_reliability_linear_fit_returns_slope_close_to_1_when_calibrated(self) -> None:
        # 10000 samples drawn so y = bernoulli(p) -> slope should be ~1.
        # Use deterministic stand-in data.
        n_per_bucket = 200
        y, p = [], []
        for bucket_p in (0.1, 0.3, 0.5, 0.7, 0.9):
            n_pos = int(round(bucket_p * n_per_bucket))
            y.extend([1] * n_pos + [0] * (n_per_bucket - n_pos))
            p.extend([bucket_p] * n_per_bucket)
        slope_intercept = reliability_linear_fit(y, p)
        self.assertIsNotNone(slope_intercept)
        slope, intercept = slope_intercept
        self.assertAlmostEqual(slope, 1.0, places=2)
        self.assertAlmostEqual(intercept, 0.0, places=2)


class TestDriftDetection(unittest.TestCase):
    def _write_log(self, rows: list[dict]) -> Path:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.close()
        return Path(f.name)

    def test_no_alerts_on_first_n_windows(self) -> None:
        rows = [
            {"window_id": str(i), "val_start": f"2024-0{i}-01", "brier": 0.24}
            for i in range(1, 4)
        ]
        path = self._write_log(rows)
        alerts = detect_calibration_drift(path, rolling_n=5)
        self.assertEqual(alerts, [])
        path.unlink()

    def test_drift_breach_detected(self) -> None:
        # 5 windows of stable Brier 0.24, then a sudden 0.40 -> z-score
        # large -> breach.
        baseline = [
            {"window_id": f"w{i}", "val_start": f"2023-0{i}-01", "brier": 0.24}
            for i in range(1, 6)
        ]
        spike = {"window_id": "w6", "val_start": "2023-06-01", "brier": 0.40}
        path = self._write_log(baseline + [spike])
        alerts = detect_calibration_drift(path, rolling_n=5, z_threshold=2.0)
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0].breach)
        self.assertEqual(alerts[0].window_id, "w6")
        path.unlink()

    def test_no_breach_when_within_threshold(self) -> None:
        rows = [
            {"window_id": f"w{i}", "val_start": f"2023-0{i}-01", "brier": 0.24 + 0.01 * (i % 2)}
            for i in range(1, 8)
        ]
        path = self._write_log(rows)
        alerts = detect_calibration_drift(path, rolling_n=5, z_threshold=2.5)
        # Some windows may trigger alerts but none should breach.
        self.assertTrue(all(not a.breach for a in alerts))
        path.unlink()


if __name__ == "__main__":
    unittest.main()
