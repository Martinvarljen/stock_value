"""Tests for ML drift gates on daily decisions."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decisions import Action, TickerDecision  # noqa: E402
from portfolio.ml_gates import MlGateResult, apply_ml_entry_gates, evaluate_ml_gates  # noqa: E402


class TestMlGates(unittest.TestCase):
    def test_blocks_entries_when_gate_active(self) -> None:
        gate = evaluate_ml_gates({"ml_gates": {"enabled": False}}, [])
        self.assertFalse(gate.block_new_entries)
        decisions = [
            TickerDecision("A", Action.ENTER_LONG, "x", ml_score=0.7),
            TickerDecision("B", Action.EXIT, "y"),
        ]
        out, blocked = apply_ml_entry_gates(
            decisions,
            MlGateResult(block_new_entries=True, reasons=["test drift"]),
        )
        self.assertEqual(len(blocked), 1)
        self.assertEqual(out[0].action, Action.NO_TRADE)
        self.assertEqual(out[1].action, Action.EXIT)

    def test_feature_drift_high_blocks(self) -> None:
        from projection.ml_model.drift_monitor import DriftMonitor

        training = {"ml_score": [0.5 + i * 0.001 for i in range(200)]}
        monitor = DriftMonitor.fit(training)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "baseline.json"
            monitor.save(path)
            analyses = [{"ok": True, "ml_score": 0.95 + i * 0.0001} for i in range(60)]
            cfg = {
                "ml_gates": {
                    "enabled": True,
                    "block_entries_on_high_feature_drift": True,
                    "block_entries_on_calibration_drift": False,
                    "feature_baseline_path": str(path),
                    "warn_if_missing_baseline": False,
                }
            }
            gate = evaluate_ml_gates(cfg, analyses)
            self.assertTrue(gate.block_new_entries)


if __name__ == "__main__":
    unittest.main()
