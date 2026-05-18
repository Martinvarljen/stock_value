"""Tests for portfolio.reflection (deterministic, pure-function)."""

from __future__ import annotations

import unittest

from portfolio.reflection import OutcomeContext, reflect_on_outcome


def _ctx(**overrides) -> OutcomeContext:
    base = dict(
        ticker="AAPL",
        trade_date="2026-05-01",
        rating="Buy",
        action="ENTER_LONG",
        raw_return=0.04,
        alpha_return=0.015,
        holding_days=20,
        benchmark="SPY",
        p_up_20d=0.62,
        ml_score=0.62,
        regime_scale=1.0,
        spy_bull=True,
        exit_reason=None,
    )
    base.update(overrides)
    return OutcomeContext(**base)


class TestReflectionDeterminism(unittest.TestCase):
    def test_pure_function_same_input_same_output(self) -> None:
        c = _ctx()
        self.assertEqual(reflect_on_outcome(c), reflect_on_outcome(c))

    def test_correct_directional_call(self) -> None:
        text = reflect_on_outcome(_ctx())
        self.assertIn("Directional call was right", text)
        self.assertIn("alpha", text.lower())

    def test_wrong_directional_call(self) -> None:
        text = reflect_on_outcome(_ctx(raw_return=-0.03, alpha_return=-0.02))
        self.assertIn("Directional call was wrong", text)

    def test_model_calibration_correct(self) -> None:
        text = reflect_on_outcome(_ctx(p_up_20d=0.70, raw_return=0.05))
        self.assertIn("Model called direction correctly", text)

    def test_model_calibration_miss(self) -> None:
        text = reflect_on_outcome(_ctx(p_up_20d=0.70, raw_return=-0.02, alpha_return=-0.01))
        self.assertIn("Model mis-called direction", text)

    def test_near_flat_model_gets_low_signal_note(self) -> None:
        text = reflect_on_outcome(_ctx(p_up_20d=0.52, raw_return=0.001, alpha_return=0.0))
        self.assertIn("near-flat", text)

    def test_exit_reason_stop_loss(self) -> None:
        text = reflect_on_outcome(_ctx(action="EXIT", exit_reason="Stop hit (long)"))
        self.assertIn("Stop fired", text)

    def test_exit_reason_take_profit(self) -> None:
        text = reflect_on_outcome(_ctx(action="EXIT", exit_reason="Take-profit (long)"))
        self.assertIn("Take-profit captured", text)

    def test_alpha_magnitude_buckets(self) -> None:
        flat = reflect_on_outcome(_ctx(alpha_return=0.001))
        marginal = reflect_on_outcome(_ctx(alpha_return=0.012))
        meaningful = reflect_on_outcome(_ctx(alpha_return=0.03))
        large = reflect_on_outcome(_ctx(alpha_return=0.08))
        self.assertIn("Effectively flat", flat)
        self.assertIn("Marginal alpha", marginal)
        self.assertIn("Meaningful alpha", meaningful)
        self.assertIn("Large alpha", large)

    def test_capped_at_four_sentences(self) -> None:
        text = reflect_on_outcome(_ctx(action="EXIT", exit_reason="Stop hit (long)"))
        # 4-sentence cap; we look at period-separated count (rough but adequate).
        self.assertLessEqual(text.count(". "), 4)


if __name__ == "__main__":
    unittest.main()
