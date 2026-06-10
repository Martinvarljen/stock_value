"""Tests for portfolio risk scaling scalars."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.risk_scaling import (  # noqa: E402
    apply_risk_scalar_to_regime,
    drawdown_scalar,
    spread_scalar_from_scores,
    yang_zhang_median_vol,
)


class TestRiskScaling(unittest.TestCase):
    def test_drawdown_kill(self) -> None:
        cfg = {"risk_scaling": {"enabled": True, "dd_kill_threshold": -0.20}}
        self.assertEqual(drawdown_scalar(0.75, 1.0, cfg), 0.0)

    def test_spread_scalar_low_dispersion(self) -> None:
        cfg = {"risk_scaling": {"enabled": True, "min_score_std": 0.05, "spread_floor": 0.5}}
        s = spread_scalar_from_scores([0.6] * 20, cfg)
        self.assertLess(s, 1.0)

    def test_yang_zhang_positive(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        rows = []
        for sym in ("A", "B", "C"):
            px = 100 + rng.normal(0, 1, len(dates)).cumsum()
            for i, d in enumerate(dates):
                rows.append({
                    "date": d, "ticker": sym,
                    "open": px[i] * 0.99, "high": px[i] * 1.01,
                    "low": px[i] * 0.98, "close": px[i],
                })
        med = yang_zhang_median_vol(pd.DataFrame(rows), window=10)
        self.assertIsNotNone(med)
        assert med is not None
        self.assertGreater(med, 0)

    def test_apply_to_regime(self) -> None:
        regime = {"gross_exposure_scale": 1.0, "spy_bull": True}
        out = apply_risk_scalar_to_regime(regime, {"scalar": 0.8, "components": {}})
        self.assertAlmostEqual(out["gross_exposure_scale"], 0.8)


if __name__ == "__main__":
    unittest.main()
