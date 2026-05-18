"""Trailing stop ratchet and exit labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.store import Position
from portfolio.trailing_stop import (
    seed_trail_fields,
    stop_exit_label,
    update_position_trail,
)


CFG = {
    "stop_loss_pct": 0.20,
    "use_trailing_stop": True,
    "trailing_stop_pct": 0.10,
    "trail_activate_profit_pct": 0.0,
}


class TestTrailingStop(unittest.TestCase):
    def test_long_trail_ratchets_up_not_down(self) -> None:
        pos = Position(
            ticker="AAPL",
            side="long",
            entry_date="2024-01-01",
            entry_price=100.0,
            notional=1000.0,
            stop_price=80.0,
            take_profit_price=135.0,
            max_hold_days=50,
        )
        seed_trail_fields(pos, CFG)
        self.assertAlmostEqual(pos.initial_stop_price, 80.0, places=4)

        update_position_trail(pos, 110.0, 105.0, CFG)
        self.assertAlmostEqual(pos.peak_price, 110.0, places=4)
        self.assertAlmostEqual(pos.stop_price, 99.0, places=4)  # 110 * 0.9

        update_position_trail(pos, 105.0, 100.0, CFG)
        self.assertAlmostEqual(pos.stop_price, 99.0, places=4)

        update_position_trail(pos, 130.0, 120.0, CFG)
        self.assertAlmostEqual(pos.stop_price, 117.0, places=4)  # 130 * 0.9

    def test_trail_activates_after_min_profit(self) -> None:
        cfg = {**CFG, "trail_activate_profit_pct": 0.05}
        pos = Position(
            ticker="X",
            side="long",
            entry_date="2024-01-01",
            entry_price=100.0,
            notional=1000.0,
            stop_price=80.0,
            take_profit_price=135.0,
            max_hold_days=50,
        )
        seed_trail_fields(pos, cfg)
        stop_before = pos.stop_price
        update_position_trail(pos, 103.0, 101.0, cfg)
        self.assertEqual(pos.stop_price, stop_before)
        update_position_trail(pos, 106.0, 104.0, cfg)
        self.assertGreater(pos.stop_price, stop_before)

    def test_exit_label_distinguishes_trail(self) -> None:
        pos = Position(
            ticker="X",
            side="long",
            entry_date="2024-01-01",
            entry_price=100.0,
            notional=1000.0,
            stop_price=95.0,
            take_profit_price=135.0,
            max_hold_days=50,
            initial_stop_price=80.0,
        )
        self.assertIn("Trailing", stop_exit_label(pos))


if __name__ == "__main__":
    unittest.main()
