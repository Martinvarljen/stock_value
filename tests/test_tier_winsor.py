import unittest

from backtesting.strategy_backtest import TIER_STATS_WINSOR, _aggregate_by_tier, _tier_stat_clip


class TestTierWinsor(unittest.TestCase):
    def test_clip(self) -> None:
        self.assertEqual(_tier_stat_clip(10.0), TIER_STATS_WINSOR)
        self.assertEqual(_tier_stat_clip(-10.0), -TIER_STATS_WINSOR)
        self.assertAlmostEqual(_tier_stat_clip(0.03), 0.03)

    def test_aggregate_avg_uses_winsor(self) -> None:
        signals = [
            {
                "classification": "AVOID",
                "fwd_3m": 0.0,
                "fwd_6m": 100.0,
                "fwd_12m": 0.0,
                "excess_fwd_3m": None,
                "excess_fwd_6m": 100.0,
                "excess_fwd_12m": None,
            },
            {
                "classification": "AVOID",
                "fwd_3m": 0.0,
                "fwd_6m": 0.04,
                "fwd_12m": 0.0,
                "excess_fwd_3m": None,
                "excess_fwd_6m": 0.02,
                "excess_fwd_12m": None,
            },
        ]
        tiers = _aggregate_by_tier(signals)
        self.assertAlmostEqual(tiers["AVOID"]["avg_6m"], (TIER_STATS_WINSOR + 0.04) / 2, places=5)
        self.assertAlmostEqual(tiers["AVOID"]["avg_excess_6m"], (TIER_STATS_WINSOR + 0.02) / 2, places=5)
        self.assertEqual(tiers["AVOID"]["best_6m"], 100.0)


if __name__ == "__main__":
    unittest.main()
