"""Tests for point-in-time S&P 500 universe reconstruction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.sp500_pit_universe import (  # noqa: E402
    Change,
    ever_members_in_window,
    load_changes,
    members_as_of,
)


class TestPITUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8",
        )
        self._tmp.write(
            "# fixture\n"
            "2018-06-26,remove,GE,GE removed\n"
            "2018-06-26,add,WBA,Walgreens added\n"
            "2020-12-21,add,TSLA,Tesla added\n"
            "2023-03-15,remove,SIVB,SVB failure\n"
        )
        self._tmp.close()
        self.path = Path(self._tmp.name)

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_load_changes_parses_and_normalises(self) -> None:
        changes = load_changes(self.path)
        self.assertEqual(len(changes), 4)
        self.assertEqual(changes[0].ticker, "GE")
        self.assertEqual(changes[0].action, "remove")
        self.assertEqual(changes[0].when, date(2018, 6, 26))

    def test_change_rejects_bad_action(self) -> None:
        with self.assertRaises(ValueError):
            Change(when=date(2020, 1, 1), action="reorg", ticker="X")

    def test_members_as_of_pre_2018_includes_GE_excludes_WBA(self) -> None:
        # Today's pool includes WBA, doesn't include SIVB. Walking back
        # past 2018-06-26 must re-add GE and remove WBA. Walking back
        # past 2023-03-15 must re-add SIVB. We also pre-add SIVB to the
        # current_pool to confirm the "remove not yet happened" branch.
        today_pool = {"AAPL", "MSFT", "WBA", "TSLA"}
        m = members_as_of(date(2017, 1, 1), current_pool=today_pool, changes_path=self.path)
        self.assertIn("GE", m)
        self.assertNotIn("WBA", m)
        self.assertNotIn("TSLA", m)

    def test_members_as_of_after_tesla_add(self) -> None:
        today_pool = {"AAPL", "MSFT", "WBA", "TSLA"}
        m = members_as_of(date(2021, 1, 1), current_pool=today_pool, changes_path=self.path)
        self.assertIn("TSLA", m)
        self.assertIn("WBA", m)
        self.assertNotIn("GE", m)

    def test_ever_members_in_window_picks_up_failed_banks(self) -> None:
        today_pool = {"AAPL", "MSFT"}
        # In a window that spans 2023-03, SIVB was a member at the start
        # then removed mid-window. ever_members must include it.
        ever = ever_members_in_window(
            date(2023, 1, 1), date(2023, 12, 31),
            current_pool=today_pool, changes_path=self.path,
        )
        self.assertIn("SIVB", ever)

    def test_ever_members_window_outside_log_warns(self) -> None:
        today_pool = {"AAPL"}
        with self.assertWarns(UserWarning):
            ever_members_in_window(
                date(1990, 1, 1), date(1990, 12, 31),
                current_pool=today_pool, changes_path=self.path,
            )


if __name__ == "__main__":
    unittest.main()
