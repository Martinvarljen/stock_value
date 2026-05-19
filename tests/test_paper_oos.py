"""Tests for forward paper OOS tracking."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import portfolio.paper_oos as po  # noqa: E402


class TestPaperOos(unittest.TestCase):
    def test_record_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            po.PAPER_OOS_DIR = tdir
            po.CURVE_PATH = tdir / "curve.jsonl"
            po.META_PATH = tdir / "meta.json"
            po.REPORT_MD_PATH = tdir / "report.md"
            po.REPORT_JSON_PATH = tdir / "report.json"

            cfg = {
                "profile": "research_ls",
                "paper_oos": {"enabled": True, "oos_start_date": "2026-05-01", "nav_anchor": 1.0},
                "universe_source": "pit_filter",
            }
            regime = {"spy_bull": True, "regime_signal": "bull", "gross_exposure_scale": 1.0}

            pd = __import__("pandas")
            fake_spy = pd.Series(
                [400.0, 401.0, 402.0, 403.0],
                index=pd.date_range("2026-05-01", periods=4, freq="D"),
            )

            with mock.patch.object(po, "_spy_close_history", return_value=fake_spy):
                po.record_paper_day(date(2026, 5, 1), nav=1.0, cash=0.9, n_positions=1, regime=regime, cfg=cfg)
                po.record_paper_day(date(2026, 5, 2), nav=1.01, cash=0.89, n_positions=1, regime=regime, cfg=cfg)
                po.record_paper_day(date(2026, 5, 3), nav=1.02, cash=0.88, n_positions=2, regime=regime, cfg=cfg)

            rows = po.load_curve_rows()
            self.assertEqual(len(rows), 3)
            m = po.compute_oos_metrics(cfg)
            self.assertGreaterEqual(m.get("n_days", 0), 2)
            self.assertIn("strategy", m)

            with mock.patch.object(po, "_spy_close_history", return_value=fake_spy):
                path = po.write_oos_report(cfg)
            self.assertTrue(path.is_file())
            data = json.loads(po.REPORT_JSON_PATH.read_text(encoding="utf-8"))
            self.assertIn("regime_attribution", data)


if __name__ == "__main__":
    unittest.main()
