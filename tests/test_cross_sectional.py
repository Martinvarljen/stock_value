"""Tests for cross-sectional training features."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projection.ml_model.cross_sectional import (  # noqa: E402
    attach_cross_sectional_features,
    diffusion_index,
)


class TestCrossSectional(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-02", periods=40, freq="B")
        rows = []
        for sym, drift in (("AAA", 0.002), ("BBB", -0.001), ("CCC", 0.001)):
            px = 100.0
            for d in dates:
                rows.append({"date": d, "act_symbol": sym, "close": px})
                px *= 1 + drift + np.random.default_rng(hash(sym) % 2**32).normal(0, 0.005)
        return pd.DataFrame(rows)

    def test_diffusion_index_range(self) -> None:
        di = diffusion_index(self._panel(), timeperiod=5)
        self.assertFalse(di.dropna().empty)
        self.assertTrue((di.dropna() >= 0).all() and (di.dropna() <= 1).all())

    def test_attach_merges_on_date(self) -> None:
        panel = self._panel()
        rows = panel[["date", "act_symbol"]].rename(columns={"act_symbol": "ticker"}).head(10).copy()
        out = attach_cross_sectional_features(rows, panel)
        self.assertIn("cs_diffusion_21d", out.columns)


if __name__ == "__main__":
    unittest.main()
