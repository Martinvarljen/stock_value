"""Tests for config profile loading."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.config_loader import config_fingerprint, load_config, profile_path  # noqa: E402


class TestConfigLoader(unittest.TestCase):
    def test_conservative_profile_lowers_leverage(self) -> None:
        base = load_config()
        cons = load_config(profile="conservative")
        self.assertEqual(cons.get("cfd_leverage"), 1)
        self.assertGreaterEqual(base.get("cfd_leverage", 1), cons.get("cfd_leverage", 1))

    def test_fingerprint_stable(self) -> None:
        cfg = {"min_p_up_long": 0.58, "cfd_leverage": 5}
        a = config_fingerprint(cfg)
        b = config_fingerprint(dict(cfg))
        self.assertEqual(a, b)

    def test_profile_files_exist(self) -> None:
        self.assertTrue(profile_path("research").is_file())
        self.assertTrue(profile_path("conservative").is_file())
        self.assertTrue(profile_path("research_ls").is_file())

    def test_research_ls_enables_shorts(self) -> None:
        cfg = load_config(profile="research_ls")
        self.assertTrue(cfg.get("enable_short"))
        self.assertEqual(cfg.get("cfd_leverage"), 5)


if __name__ == "__main__":
    unittest.main()
