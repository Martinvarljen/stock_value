"""Tests for backtest strategy mode normalization."""

import unittest

from backtesting.strategy_modes import (
    is_ml_strategy,
    normalize_signal_mode,
    strategy_display_name,
)


class TestStrategyModes(unittest.TestCase):
    def test_ml_aliases(self) -> None:
        for name in ("ml", "tech_ai", "tech-ai", "dolt_ml", "ML"):
            self.assertEqual(normalize_signal_mode(name), "tech_ai")
            self.assertTrue(is_ml_strategy(name))

    def test_dcf_default(self) -> None:
        self.assertEqual(normalize_signal_mode(None), "dcf")
        self.assertEqual(normalize_signal_mode("dcf"), "dcf")
        self.assertFalse(is_ml_strategy("dcf"))

    def test_display_name(self) -> None:
        self.assertIn("LightGBM", strategy_display_name("ml"))


if __name__ == "__main__":
    unittest.main()
