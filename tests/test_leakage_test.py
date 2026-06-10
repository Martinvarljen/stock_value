"""Tests for permutation leakage detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projection.ml_model.leakage_test import run_permutation_leakage_test  # noqa: E402


class TestLeakageTest(unittest.TestCase):
    def test_passes_on_noise_features(self) -> None:
        rng = np.random.default_rng(1)
        n = 300
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.integers(0, 2, size=n))
        split = int(n * 0.8)
        result = run_permutation_leakage_test(
            X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
            n_repeats=2, ic_threshold=0.25, num_boost_round=20,
        )
        self.assertIsInstance(result.passed, bool)
        self.assertEqual(len(result.iterations), 2)


if __name__ == "__main__":
    unittest.main()
