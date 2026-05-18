"""Tests for portfolio-level pre-trade risk limits."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decisions import Action, TickerDecision  # noqa: E402
from portfolio.risk_limits import RiskLimits, apply_pre_trade_limits  # noqa: E402
from portfolio.store import PortfolioState, Position  # noqa: E402


def _state(nav: float = 100_000.0) -> PortfolioState:
    return PortfolioState(positions=[], cash=nav, nav=nav)


def _entry(ticker: str, side: Action) -> TickerDecision:
    return TickerDecision(
        ticker=ticker, action=side, reason="entry",
        ml_score=0.7, p_up_20d=0.65, price=100.0,
    )


CFG = {"position_frac": 0.10, "_regime_scale": 1.0}


class TestSectorLimit(unittest.TestCase):
    def test_blocks_third_entry_when_sector_cap_30_pct(self) -> None:
        # 3 entries x 10% = 30% in semis -> the 3rd is the boundary;
        # cap is 30% -> third allowed. 4th -> blocked.
        st = _state()
        decisions = [_entry(t, Action.ENTER_LONG) for t in ("NVDA", "AMD", "INTC", "AVGO")]
        sector_lookup = lambda tk: "Technology"
        limits = RiskLimits(max_sector_pct=0.30, max_gross_exposure_pct=2.0,
                            max_beta_to_spy=5.0, enforce_var=False)
        out, dropped = apply_pre_trade_limits(
            decisions, st, limits=limits, cfg=CFG,
            sector_lookup=sector_lookup,
        )
        actions = [d.action for d in out]
        # First three pass, fourth blocked.
        self.assertEqual(actions[:3], [Action.ENTER_LONG] * 3)
        self.assertEqual(actions[3], Action.NO_TRADE)
        self.assertEqual(len(dropped), 1)
        self.assertIn("sector", dropped[0]["reason"])


class TestGrossExposureLimit(unittest.TestCase):
    def test_blocks_when_gross_cap_breached(self) -> None:
        st = _state()
        # 11 entries x 10% = 110% gross -> cap 100% means the 11th is
        # blocked (10 fit at exactly 100%; 11th would push to 110%).
        decisions = [_entry(f"T{i}", Action.ENTER_LONG) for i in range(11)]
        limits = RiskLimits(max_gross_exposure_pct=1.00, max_sector_pct=2.0,
                            max_net_exposure_pct=2.0, max_beta_to_spy=5.0,
                            enforce_var=False)
        out, dropped = apply_pre_trade_limits(
            decisions, st, limits=limits, cfg=CFG,
        )
        passed = sum(1 for d in out if d.action == Action.ENTER_LONG)
        self.assertEqual(passed, 10)
        self.assertEqual(len(dropped), 1)
        self.assertIn("gross", dropped[0]["reason"])


class TestBetaLimit(unittest.TestCase):
    def test_blocks_high_beta_pile_on(self) -> None:
        st = _state()
        decisions = [_entry(t, Action.ENTER_LONG) for t in ("ARKK", "TSLA", "ROKU")]
        # All beta 2.5 -> portfolio beta = 2.5 * gross. With 30% gross
        # contribution -> portfolio beta ~0.75 each fold so by the 3rd
        # we're above 1.30.
        beta_lookup = lambda tk: 2.5
        limits = RiskLimits(max_beta_to_spy=1.30, max_gross_exposure_pct=2.0,
                            max_sector_pct=2.0, enforce_var=False)
        out, dropped = apply_pre_trade_limits(
            decisions, st, limits=limits, cfg=CFG,
            beta_lookup=beta_lookup,
        )
        passed = sum(1 for d in out if d.action == Action.ENTER_LONG)
        # 1 * 0.10 NAV * beta=2.5 / 100k = 0.25 beta -> ok
        # 2 * 0.10 NAV * beta=2.5 / 100k = 0.50 beta -> ok
        # ... clearly all 3 fit if beta cap is 1.3 (only 0.75 max).
        # So we need a tighter beta cap to test.
        self.assertEqual(passed, 3)
        # Now run with beta cap of 0.5
        limits2 = RiskLimits(max_beta_to_spy=0.50, max_gross_exposure_pct=2.0,
                             max_sector_pct=2.0, enforce_var=False)
        out2, dropped2 = apply_pre_trade_limits(
            decisions, st, limits=limits2, cfg=CFG, beta_lookup=beta_lookup,
        )
        passed2 = sum(1 for d in out2 if d.action == Action.ENTER_LONG)
        # 0.10 * 2.5 = 0.25 (ok), next +0.25 = 0.50 (boundary, ok),
        # next +0.25 = 0.75 (>0.5, blocked).
        self.assertEqual(passed2, 2)
        self.assertEqual(len(dropped2), 1)
        self.assertIn("beta", dropped2[0]["reason"])


class TestExitsPassThrough(unittest.TestCase):
    def test_exits_and_holds_unchanged(self) -> None:
        st = _state()
        st.positions.append(Position(
            ticker="AAPL", side="long", entry_date="2024-01-01",
            entry_price=100, notional=10_000, stop_price=90, take_profit_price=120,
            max_hold_days=25,
        ))
        exit_d = TickerDecision(
            ticker="AAPL", action=Action.EXIT, reason="stop", price=95,
        )
        hold_d = TickerDecision(
            ticker="MSFT", action=Action.HOLD, reason="hold", price=400,
        )
        out, _ = apply_pre_trade_limits(
            [exit_d, hold_d], st,
            limits=RiskLimits(), cfg=CFG,
        )
        self.assertEqual([d.action for d in out], [Action.EXIT, Action.HOLD])


class TestVarLimit(unittest.TestCase):
    def test_high_vol_basket_breaches_var_cap(self) -> None:
        st = _state()
        # 5 entries x 10% NAV @ 80% annualised vol. zero correlation
        # variance = 5 * (0.10 * 0.80)^2 = 0.032
        # daily sigma = sqrt(0.032) / sqrt(252) ≈ 0.0113
        # 95% VaR ≈ 1.645 * 0.0113 ≈ 0.0186 (<5% so ok)
        # Tighten cap to force breach.
        decisions = [_entry(f"VOL{i}", Action.ENTER_LONG) for i in range(5)]
        vol_lookup = lambda tk: 0.80
        limits = RiskLimits(max_single_day_var_pct=0.005,
                            max_gross_exposure_pct=2.0,
                            max_sector_pct=2.0, max_beta_to_spy=5.0)
        out, dropped = apply_pre_trade_limits(
            decisions, st, limits=limits, cfg=CFG, vol_lookup=vol_lookup,
        )
        # At least one should be dropped due to VaR.
        var_drops = [d for d in dropped if "var" in d["reason"].lower()]
        self.assertGreaterEqual(len(var_drops), 1)


if __name__ == "__main__":
    unittest.main()
