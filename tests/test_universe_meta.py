"""Tests for universe / PIT metadata."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.universe_meta import pit_coverage, universe_summary  # noqa: E402


class TestUniverseMeta(unittest.TestCase):
    def test_pit_coverage_partial_for_old_window(self) -> None:
        meta = pit_coverage(date(2000, 1, 1), date(2010, 12, 31))
        self.assertTrue(meta.get("pit_available"))
        self.assertIn(meta.get("pit_coverage"), ("partial", "full"))

    def test_universe_summary_legacy(self) -> None:
        s = universe_summary(universe_source="legacy", start=date(2019, 1, 1), end=date(2024, 1, 1))
        self.assertEqual(s["universe_source"], "legacy")
        note = s.get("survivorship_bias_note", "").lower()
        self.assertTrue("delisted" in note or "survivorship" in note)


if __name__ == "__main__":
    unittest.main()
