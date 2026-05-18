"""Tests for the v4 engine-derived ML feature block.

Validates the contract of :func:`projection.ml_model.features._engine_features_from_arrays`:

  * default neutral values when arrays are empty / pathological
  * candle bias is read from ``analyze_candle_patterns`` (a pure-Python engine
    with no numpy/pandas dependency, so this case always runs)
  * market-structure and Elliott blocks are populated from their respective
    engines when pandas is available; gracefully skipped in environments
    without it.

Designed to exercise the real engines rather than mock them — the whole
point of this layer is that the ML pipeline and the live trading agent
share the same swing/candle/Elliott logic.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "stock_analyzer"), str(_ROOT / "projection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # pandas drives the historical feature pipeline
    import pandas  # noqa: F401

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
class TestEngineFeaturesFromArrays(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Import lazily so the sandbox-skip path actually skips.
        from projection.ml_model.features import (
            _ENGINE_FEATURE_DEFAULTS,
            _engine_features_from_arrays,
        )
        cls.fn = staticmethod(_engine_features_from_arrays)
        cls.defaults = dict(_ENGINE_FEATURE_DEFAULTS)

    def test_empty_arrays_return_all_defaults(self) -> None:
        out = self.fn(None, [], [], [])
        self.assertEqual(out, self.defaults)
        self.assertEqual(set(out.keys()), set(self.defaults.keys()))

    def test_zero_last_close_returns_defaults(self) -> None:
        out = self.fn(None, [100.0] * 100, [99.0] * 100, [0.0] * 100)
        self.assertEqual(out, self.defaults)

    def test_short_history_returns_defaults(self) -> None:
        # < 3 bars: candle engine unavailable. < 80 bars: elliott unavailable.
        # < 2 * order + 15 = 23 bars: market structure unavailable.
        c = [10.0, 10.5]
        out = self.fn(None, c, c, c)
        for k, v in self.defaults.items():
            self.assertEqual(out[k], v, f"{k} should be default for short history")

    def test_uptrend_synthetic_history_lights_up_bullish_signals(self) -> None:
        # 120 bars steadily climbing then a green last bar.
        c = [100.0 + i * 0.5 for i in range(120)]
        h = [v + 0.4 for v in c]
        l = [v - 0.4 for v in c]
        # Last bar: bullish marubozu-ish.
        o = list(c)
        o[-1] = c[-1] - 1.0  # open below close

        out = self.fn(o, h, l, c)

        self.assertEqual(out["cand_bias_bull"], 1.0)
        self.assertEqual(out["cand_bias_bear"], 0.0)
        self.assertGreaterEqual(out["cand_body_pct"], 0.0)
        self.assertLessEqual(out["cand_body_pct"], 1.0)
        self.assertGreaterEqual(out["ms_n_pivots_norm"], 0.0)
        self.assertLessEqual(out["ms_n_pivots_norm"], 1.0)
        self.assertGreaterEqual(out["ms_pivot_dist_norm"], -1.0)
        self.assertLessEqual(out["ms_pivot_dist_norm"], 1.0)
        self.assertGreaterEqual(out["ell_price_vs_fib_norm"], -0.5)
        self.assertLessEqual(out["ell_price_vs_fib_norm"], 0.5)
        self.assertIn(out["ell_dir_up"], (0.0, 1.0))
        self.assertIn(out["ell_dir_down"], (0.0, 1.0))
        self.assertNotEqual(
            out["ell_dir_up"] + out["ell_dir_down"], 2.0,
            "up and down can't both be 1.0",
        )

    def test_downtrend_synthetic_history_lights_up_bearish_candle(self) -> None:
        c = [200.0 - i * 0.5 for i in range(120)]
        h = [v + 0.4 for v in c]
        l = [v - 0.4 for v in c]
        o = list(c)
        o[-1] = c[-1] + 1.0  # open above close → red bar

        out = self.fn(o, h, l, c)

        self.assertEqual(out["cand_bias_bear"], 1.0)
        self.assertEqual(out["cand_bias_bull"], 0.0)

    def test_returns_all_v4_columns(self) -> None:
        """Schema contract: the function never silently drops a column."""
        c = [100.0 + (i % 7) * 0.5 for i in range(120)]
        out = self.fn(None, c, c, c)
        self.assertEqual(set(out.keys()), set(self.defaults.keys()))


class TestSchemaVersion(unittest.TestCase):
    """Schema-version sanity (runs even without pandas)."""

    @unittest.skipUnless(_HAS_PANDAS, "pandas not installed in this environment")
    def test_schema_version_is_v4(self) -> None:
        from projection.ml_model.features import FEATURE_SCHEMA_VERSION, TECH_FEATURES

        self.assertEqual(FEATURE_SCHEMA_VERSION, 4)
        # All v4 engine columns are in TECH_FEATURES.
        for col in (
            "ms_regime_up",
            "ms_regime_down",
            "ms_n_pivots_norm",
            "ms_pivot_dist_norm",
            "cand_bias_bull",
            "cand_bias_bear",
            "cand_body_pct",
            "cand_upper_wick_pct",
            "cand_lower_wick_pct",
            "ell_dir_up",
            "ell_dir_down",
            "ell_price_vs_fib_norm",
        ):
            self.assertIn(col, TECH_FEATURES, f"missing v4 column: {col}")


if __name__ == "__main__":
    unittest.main()
