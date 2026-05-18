"""Tests for the decision-threshold scaffolding."""

from __future__ import annotations

import unittest

from portfolio.decision_thresholds import (
    DEFAULT_DECISION_CFG,
    JOINT_SWEEP_RANGES,
    THRESHOLD_RANGES,
    n_trials_for,
    n_univariate_trials_for,
    sweep_decision_cfgs,
    univariate_sensitivity_sweep,
)


class TestDecisionThresholds(unittest.TestCase):
    def test_defaults_have_required_keys(self) -> None:
        # Sanity: every key the broker / decisions module reads should
        # have a documented default here so callers don't have to guess.
        for required in (
            "min_p_up_long", "max_p_up_short",
            "long_quintile_min", "short_quintile_max",
            "min_p_up_long_abs_buffer", "max_p_up_short_abs_buffer",
            "max_positions", "position_frac",
            "stop_loss_pct", "take_profit_pct", "max_hold_days",
            "commission_bps", "slippage_bps", "borrow_bps_annual",
            "atr_stop_mult", "atr_tp_mult",
            "vol_target_annual_pct",
        ):
            self.assertIn(required, DEFAULT_DECISION_CFG, f"missing {required}")

    def test_sweep_yields_cartesian_product(self) -> None:
        axes = {
            "min_p_up_long": [0.55, 0.60],
            "stop_loss_pct": [0.15, 0.20, 0.25],
        }
        out = list(sweep_decision_cfgs({}, axes))
        self.assertEqual(len(out), 6)
        labels = {label for _, label in out}
        self.assertEqual(len(labels), 6)
        for cfg, label in out:
            self.assertIn("min_p_up_long=", label)
            self.assertIn("stop_loss_pct=", label)
            self.assertIn(cfg["min_p_up_long"], axes["min_p_up_long"])

    def test_sweep_does_not_mutate_base(self) -> None:
        base = {"foo": 1, "min_p_up_long": 0.50}
        for cfg, _ in sweep_decision_cfgs(base, {"min_p_up_long": [0.55, 0.60]}):
            cfg["foo"] = 999
        self.assertEqual(base["foo"], 1, "base cfg leaked across iterations")
        self.assertEqual(base["min_p_up_long"], 0.50)

    def test_n_trials_for_matches_sweep_size(self) -> None:
        axes = {
            "min_p_up_long": [0.55, 0.60],
            "stop_loss_pct": [0.15, 0.20, 0.25],
            "position_frac": [0.10, 0.12],
        }
        self.assertEqual(n_trials_for(axes), 12)
        self.assertEqual(len(list(sweep_decision_cfgs({}, axes))), 12)

    def test_univariate_sweep_size_is_modest(self) -> None:
        # Univariate is meant to be a quick diagnostic — the full
        # ``THRESHOLD_RANGES`` recipe should add up to ~25-35 trials.
        n = n_univariate_trials_for(THRESHOLD_RANGES)
        self.assertGreater(n, 10)
        self.assertLessEqual(n, 50)
        self.assertEqual(n, len(list(univariate_sensitivity_sweep({}, THRESHOLD_RANGES))))

    def test_joint_sweep_is_deliberately_tiny(self) -> None:
        # 27 is the soft cap. Anything larger is overfit territory and
        # the DSR with N>50 already kills most "good" Sharpes.
        self.assertLessEqual(n_trials_for(JOINT_SWEEP_RANGES), 27)

    def test_univariate_keeps_other_axes_at_base(self) -> None:
        base = {"min_p_up_long": 0.58, "stop_loss_pct": 0.20}
        axes = {"min_p_up_long": [0.55, 0.60]}
        out = list(univariate_sensitivity_sweep(base, axes))
        self.assertEqual(len(out), 2)
        for cfg, _ in out:
            self.assertEqual(cfg["stop_loss_pct"], 0.20)


if __name__ == "__main__":
    unittest.main()
