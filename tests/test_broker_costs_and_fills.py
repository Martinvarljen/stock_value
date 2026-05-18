"""Deterministic broker invariants — cost arithmetic and intraday-stop fills.

These tests pin specific numeric outcomes for the cost model + intraday
stop logic. They run in pure Python (no numpy / pandas / yfinance) so
they're sandbox-friendly and act as a regression net for accidental
changes to entry/exit math.
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.broker import accrue_short_overnight_interest, apply_decisions  # noqa: E402
from portfolio.decisions import Action, TickerDecision  # noqa: E402
from portfolio.store import PortfolioState, Position  # noqa: E402


def _state(cash: float = 100_000.0) -> PortfolioState:
    return PortfolioState(positions=[], cash=cash, nav=cash)


def _entry_decision(ticker: str, price: float, side: Action, **kw) -> TickerDecision:
    return TickerDecision(
        ticker=ticker, action=side, reason="test", ml_score=0.7,
        quintile=5, p_up_20d=0.65, price=price, **kw,
    )


CFG_FLAT_NO_COST = {
    "position_frac": 0.10,
    "max_positions": 5,
    "stop_loss_pct": 0.10,
    "take_profit_pct": 0.20,
    "max_hold_days": 25,
    "commission_bps": 0,
    "slippage_bps": 0,
    "borrow_bps_annual": 0,
    "cfd_leverage": 1,
}

CFG_CFD_5X = {**CFG_FLAT_NO_COST, "cfd_leverage": 5}

CFG_WITH_COSTS = {
    **CFG_FLAT_NO_COST,
    "commission_bps": 1.0,
    "slippage_bps": 2.0,
    "borrow_bps_annual": 100.0,
    "cfd_leverage": 1,
}


class TestEntryArithmetic(unittest.TestCase):
    def test_zero_cost_entry_invests_full_budget(self) -> None:
        st = _state(100_000.0)
        d = _entry_decision("AAPL", 100.0, Action.ENTER_LONG)
        rows = apply_decisions(st, [d], run_date=date(2024, 1, 5),
                               cfg=CFG_FLAT_NO_COST)
        # Budget = 10% of NAV = 10_000 with zero costs -> invested 10_000.
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["notional"], 10_000.0, places=4)
        self.assertAlmostEqual(rows[0]["entry_cost"], 0.0, places=4)
        # Cash drops by full budget.
        self.assertAlmostEqual(st.cash, 90_000.0, places=4)

    def test_costs_reduce_invested_principal(self) -> None:
        st = _state(100_000.0)
        d = _entry_decision("AAPL", 100.0, Action.ENTER_LONG)
        rows = apply_decisions(st, [d], run_date=date(2024, 1, 5),
                               cfg=CFG_WITH_COSTS)
        # Budget 10_000; one-way cost = (1+2)/10000 = 0.0003 -> 3.0
        # Invested = 10_000 - 3.0 = 9_997.0
        self.assertAlmostEqual(rows[0]["notional"], 9_997.0, places=4)
        self.assertAlmostEqual(rows[0]["entry_cost"], 3.0, places=4)
        # Cash drops by margin (budget minus entry cost).
        self.assertAlmostEqual(st.cash, 90_003.0, places=0)


class TestIntradayStopFill(unittest.TestCase):
    """The day's range touched the stop — broker must fill at the stop
    level (with slippage), not at the (much lower) closing print."""

    def _open_long(self, st: PortfolioState, entry: float = 100.0) -> Position:
        d = _entry_decision("AAPL", entry, Action.ENTER_LONG)
        apply_decisions(st, [d], run_date=date(2024, 1, 5), cfg=CFG_FLAT_NO_COST)
        pos = st.position_for("AAPL")
        assert pos is not None
        return pos

    def test_long_stop_touched_intraday_fills_at_stop_not_close(self) -> None:
        st = _state(100_000.0)
        pos = self._open_long(st, entry=100.0)
        stop = pos.stop_price  # 90 with 10% stop
        # Bar: low pierces stop, close is far below -> fill at STOP.
        exit_d = TickerDecision(
            ticker="AAPL", action=Action.EXIT, reason="bar low <= stop",
            price=70.0,                  # close - terrible
            intraday_low=89.0,           # touched stop @ 90
            intraday_high=92.0,
            open_price=91.0,             # opened above stop
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10),
                               cfg=CFG_FLAT_NO_COST)
        self.assertEqual(rows[0]["fill_kind"], "stop_touched")
        # Realized pnl_pct must reflect fill at stop (90), not close (70).
        self.assertAlmostEqual(rows[0]["pnl_pct"], (stop - 100.0) / 100.0, places=4)
        self.assertAlmostEqual(rows[0]["price"], stop, places=4)

    def test_long_gap_through_stop_fills_at_open_not_stop(self) -> None:
        st = _state(100_000.0)
        pos = self._open_long(st, entry=100.0)
        # Open=85 (below stop 90), low=80, close=82. Worst real fill is
        # the open — broker must use open, not the (no-longer-reachable)
        # stop level.
        exit_d = TickerDecision(
            ticker="AAPL", action=Action.EXIT, reason="gap-down through stop",
            price=82.0, intraday_low=80.0, intraday_high=86.0, open_price=85.0,
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10),
                               cfg=CFG_FLAT_NO_COST)
        self.assertEqual(rows[0]["fill_kind"], "stop_touched")
        self.assertAlmostEqual(rows[0]["price"], 85.0, places=4)
        self.assertAlmostEqual(rows[0]["pnl_pct"], -0.15, places=4)

    def test_long_no_intraday_data_falls_back_to_close(self) -> None:
        st = _state(100_000.0)
        self._open_long(st, entry=100.0)
        exit_d = TickerDecision(
            ticker="AAPL", action=Action.EXIT, reason="no intraday data",
            price=95.0,  # close only
            intraday_low=None, intraday_high=None, open_price=None,
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10),
                               cfg=CFG_FLAT_NO_COST)
        self.assertEqual(rows[0]["fill_kind"], "close")
        self.assertAlmostEqual(rows[0]["price"], 95.0, places=4)

    def test_long_take_profit_touched_fills_at_tp(self) -> None:
        st = _state(100_000.0)
        pos = self._open_long(st, entry=100.0)
        tp = pos.take_profit_price  # 120 with 20% take-profit
        exit_d = TickerDecision(
            ticker="AAPL", action=Action.EXIT, reason="hit tp",
            price=125.0,                # close above tp
            intraday_low=110.0,
            intraday_high=121.0,        # tp touched
            open_price=115.0,
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10),
                               cfg=CFG_FLAT_NO_COST)
        self.assertEqual(rows[0]["fill_kind"], "tp_touched")
        self.assertAlmostEqual(rows[0]["price"], tp, places=4)


class TestCfdLeverageAndOvernight(unittest.TestCase):
    def test_long_5x_reserves_margin_not_full_exposure(self) -> None:
        st = _state(100_000.0)
        d = _entry_decision("AAPL", 100.0, Action.ENTER_LONG)
        rows = apply_decisions(st, [d], run_date=date(2024, 1, 5), cfg=CFG_CFD_5X)
        self.assertAlmostEqual(rows[0]["margin"], 10_000.0, places=0)
        self.assertAlmostEqual(rows[0]["exposure"], 50_000.0, places=0)
        self.assertAlmostEqual(st.cash, 90_000.0, places=0)

    def test_long_5x_pnl_scales_on_exposure(self) -> None:
        st = _state(100_000.0)
        apply_decisions(
            st,
            [_entry_decision("AAPL", 100.0, Action.ENTER_LONG)],
            run_date=date(2024, 1, 5),
            cfg=CFG_CFD_5X,
        )
        exit_d = TickerDecision(
            ticker="AAPL",
            action=Action.EXIT,
            reason="sell",
            price=110.0,
            intraday_low=109.0,
            intraday_high=111.0,
            open_price=110.0,
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10), cfg=CFG_CFD_5X)
        exit_row = [r for r in rows if r.get("action") == Action.EXIT.value][0]
        self.assertAlmostEqual(exit_row["pnl_pct"], 0.10, places=4)
        self.assertAlmostEqual(exit_row["net_proceeds"], 15_000.0, places=0)

    def test_short_5x_reserves_margin_not_full_exposure(self) -> None:
        st = _state(100_000.0)
        cfg = CFG_CFD_5X
        d = _entry_decision("AAPL", 100.0, Action.ENTER_SHORT)
        rows = apply_decisions(st, [d], run_date=date(2024, 1, 5), cfg=cfg)
        self.assertAlmostEqual(rows[0]["margin"], 10_000.0, places=0)
        self.assertAlmostEqual(rows[0]["exposure"], 50_000.0, places=0)
        self.assertAlmostEqual(st.cash, 90_000.0, places=0)
        pos = st.position_for("AAPL")
        assert pos is not None
        self.assertAlmostEqual(pos.short_margin(), 10_000.0, places=0)
        self.assertAlmostEqual(pos.notional, 50_000.0, places=0)

    def test_short_5x_pnl_scales_on_exposure(self) -> None:
        st = _state(100_000.0)
        cfg = CFG_CFD_5X
        apply_decisions(
            st,
            [_entry_decision("AAPL", 100.0, Action.ENTER_SHORT)],
            run_date=date(2024, 1, 5),
            cfg=cfg,
        )
        # Price drops 10%: PnL = 50_000 * 0.10 = 5_000 on top of 10_000 margin.
        exit_d = TickerDecision(
            ticker="AAPL",
            action=Action.EXIT,
            reason="cover",
            price=90.0,
            intraday_low=89.0,
            intraday_high=91.0,
            open_price=90.0,
        )
        rows = apply_decisions(st, [exit_d], run_date=date(2024, 1, 10), cfg=cfg)
        exit_row = [r for r in rows if r.get("action") == Action.EXIT.value][0]
        self.assertAlmostEqual(exit_row["pnl_pct"], 0.10, places=4)
        self.assertAlmostEqual(exit_row["net_proceeds"], 15_000.0, places=0)

    def test_overnight_interest_on_exposure_one_day(self) -> None:
        st = _state(100_000.0)
        pos = Position(
            ticker="AAPL",
            side="short",
            entry_date="2024-01-01",
            entry_price=100.0,
            notional=50_000.0,
            margin=10_000.0,
            stop_price=110.0,
            take_profit_price=80.0,
            max_hold_days=25,
        )
        st.positions.append(pos)
        st.cash -= 10_000.0
        cfg = {**CFG_WITH_COSTS, "cfd_leverage": 5, "overnight_interest_bps_annual": 100.0}
        rows = accrue_short_overnight_interest(st, run_date=date(2024, 1, 2), cfg=cfg)
        expected = 50_000.0 * (100.0 / 10_000.0 / 252.0)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["overnight_charge"], expected, places=2)
        self.assertAlmostEqual(st.cash, 100_000.0 - 10_000.0 - expected, places=0)

    def test_overnight_on_long_exposure(self) -> None:
        st = _state(100_000.0)
        pos = Position(
            ticker="AAPL",
            side="long",
            entry_date="2024-01-01",
            entry_price=100.0,
            notional=50_000.0,
            margin=10_000.0,
            stop_price=90.0,
            take_profit_price=120.0,
            max_hold_days=25,
        )
        st.positions.append(pos)
        st.cash -= 10_000.0
        cfg = {**CFG_WITH_COSTS, "cfd_leverage": 5, "overnight_interest_bps_annual": 100.0}
        rows = accrue_short_overnight_interest(st, run_date=date(2024, 1, 2), cfg=cfg)
        expected = 50_000.0 * (100.0 / 10_000.0 / 252.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["side"], "long")
        self.assertAlmostEqual(rows[0]["overnight_charge"], expected, places=2)


if __name__ == "__main__":
    unittest.main()
