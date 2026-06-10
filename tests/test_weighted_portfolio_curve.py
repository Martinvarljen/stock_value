import unittest
from datetime import datetime

from backtesting.strategy_backtest import (
    sequential_weighted_equity_curve,
    weighted_basket_returns_at_date,
)


class TestWeightedPortfolioCurve(unittest.TestCase):
    def test_weighted_basket_tier(self) -> None:
        d = datetime(2024, 3, 31)
        same = [
            {"classification": "STRONG BUY", "fwd_6m": 0.10, "spy_fwd_6m": 0.04},
            {"classification": "BUY", "fwd_6m": 0.0, "spy_fwd_6m": 0.04},
        ]
        r_s, r_b, n = weighted_basket_returns_at_date(same, weight_mode="tier", hold_months=6)
        self.assertEqual(n, 2)
        # weights 2/3 and 1/3 → stock (2/3)*0.1 + 0 = 0.0666...
        self.assertAlmostEqual(r_s, (2 / 3) * 0.10 + (1 / 3) * 0.0, places=6)
        self.assertAlmostEqual(r_b, 0.04, places=6)

    def test_weighted_basket_equal(self) -> None:
        same = [
            {"classification": "BUY", "fwd_6m": 0.08, "spy_fwd_6m": 0.02},
            {"classification": "BUY", "fwd_6m": 0.02, "spy_fwd_6m": 0.02},
        ]
        r_s, r_b, n = weighted_basket_returns_at_date(same, weight_mode="equal", hold_months=6)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(r_s, 0.05, places=6)
        self.assertAlmostEqual(r_b, 0.02, places=6)

    def test_sequential_skips_inside_hold(self) -> None:
        """Second quarter inside 6M window must not start a new basket."""
        s = [
            {
                "date": datetime(2023, 3, 31),
                "ticker": "AAA",
                "classification": "BUY",
                "fwd_6m": 0.10,
                "spy_fwd_6m": 0.05,
            },
            {
                "date": datetime(2023, 6, 30),
                "ticker": "BBB",
                "classification": "BUY",
                "fwd_6m": 0.20,
                "spy_fwd_6m": 0.06,
            },
            {
                "date": datetime(2023, 12, 31),
                "ticker": "CCC",
                "classification": "BUY",
                "fwd_6m": 0.05,
                "spy_fwd_6m": 0.01,
            },
        ]
        df = sequential_weighted_equity_curve(s, weight_mode="equal", hold_months=6)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df.iloc[0]["ret_stock"], 0.10, places=6)
        self.assertAlmostEqual(df.iloc[1]["ret_stock"], 0.05, places=6)
        self.assertAlmostEqual(df.iloc[1]["equity_stock"], 1.1 * 1.05, places=6)


if __name__ == "__main__":
    unittest.main()
