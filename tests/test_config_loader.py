"""Tests for config profile loading."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.config_loader import config_fingerprint, load_config, profile_path  # noqa: E402


class TestConfigLoader(unittest.TestCase):
    def test_default_is_winning_research_ls(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.get("profile"), "research_ls")
        self.assertTrue(cfg.get("enable_short"))
        self.assertEqual(cfg.get("long_leverage"), 5)
        self.assertEqual(cfg.get("short_leverage"), 5)
        self.assertEqual(cfg.get("short_quintile_max"), 1)
        self.assertTrue(cfg.get("short_requires_full_risk_off"))

    def test_fingerprint_stable(self) -> None:
        cfg = {"min_p_up_long": 0.58, "cfd_leverage": 5}
        a = config_fingerprint(cfg)
        b = config_fingerprint(dict(cfg))
        self.assertEqual(a, b)

    def test_research_ls_profile_file_exists(self) -> None:
        self.assertTrue(profile_path("research_ls").is_file())


if __name__ == "__main__":
    unittest.main()
