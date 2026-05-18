"""Tests for the threshold sweep runner — robust-cfg picking and report
output. The actual backtest call is monkeypatched so tests are pure
Python and run in any sandbox."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.run_threshold_sweep import TrialResult, pick_robust_cfg, write_reports  # noqa: E402


class TestPickRobust(unittest.TestCase):
    def test_picks_top_quartile_by_sharpe_then_min_drawdown(self) -> None:
        results = [
            TrialResult("a", {}, 0.10, 0.5, 0.6, -0.30, 50),
            TrialResult("b", {}, 0.12, 1.2, 0.7, -0.20, 50),
            TrialResult("c", {}, 0.15, 1.4, 0.8, -0.18, 50),
            TrialResult("d", {}, 0.18, 1.6, 0.9, -0.25, 50),  # higher Sharpe, deeper DD
        ]
        chosen = pick_robust_cfg(results)
        self.assertIsNotNone(chosen)
        # b/c are bottom-half by Sharpe; the top quartile threshold @
        # 75th percentile sits at index 2 (sorted ascending -> 0.5,1.2,1.4,1.6),
        # giving threshold=1.4. Members: {c, d}. Smallest DD = c (-0.18).
        assert chosen is not None
        self.assertEqual(chosen.label, "c")

    def test_returns_none_on_empty_or_all_errors(self) -> None:
        self.assertIsNone(pick_robust_cfg([]))
        self.assertIsNone(pick_robust_cfg([
            TrialResult("a", {}, None, None, None, None, None, error="oops"),
        ]))


class TestWriteReports(unittest.TestCase):
    def test_writes_json_and_md(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            results = [TrialResult("a", {"min_p_up_long": 0.58},
                                    0.10, 0.9, 0.5, -0.15, 30)]
            write_reports(results, out, n_trials=10, oos_block=None)
            self.assertTrue((out / "threshold_sweep.json").is_file())
            self.assertTrue((out / "threshold_sweep.md").is_file())
            payload = json.loads((out / "threshold_sweep.json").read_text())
            self.assertEqual(payload["n_trials"], 10)
            self.assertEqual(len(payload["results"]), 1)


if __name__ == "__main__":
    unittest.main()
