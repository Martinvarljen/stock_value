"""Tests for crash-safe atomic file writes."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.atomic_io import atomic_write_json, atomic_write_text  # noqa: E402


class TestAtomicIo(unittest.TestCase):
    def test_atomic_write_json_roundtrip(self) -> None:
        path = _ROOT / "tests" / "_tmp_atomic_state.json"
        try:
            atomic_write_json(path, {"nav": 1.05, "positions": []})
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["nav"], 1.05)
        finally:
            path.unlink(missing_ok=True)

    def test_atomic_write_text_overwrites(self) -> None:
        path = _ROOT / "tests" / "_tmp_atomic.txt"
        try:
            atomic_write_text(path, "first")
            atomic_write_text(path, "second")
            self.assertEqual(path.read_text(encoding="utf-8"), "second")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
