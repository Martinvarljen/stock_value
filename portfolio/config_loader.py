"""Load portfolio config (production strategy: research_ls long+short 5x)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from portfolio.store import CONFIG_PATH, PORTFOLIO_DIR

PROFILES_DIR = PORTFOLIO_DIR / "profiles"
VALID_PROFILES = frozenset({"research_ls"})


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def profile_path(name: str) -> Path:
    if name not in VALID_PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(VALID_PROFILES)}")
    return PROFILES_DIR / f"{name}.json"


def load_config(*, profile: str | None = None) -> dict[str, Any]:
    """Load ``config.json`` and optionally merge a named profile overlay.

    When ``profile`` is omitted, ``default_profile`` from ``config.json`` is
    applied (paper + backtest share one strategy). Pass ``profile=""`` to load
    base config only with no overlay.
    """
    if CONFIG_PATH.is_file():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    else:
        cfg = {}
    effective = profile
    if effective is None:
        effective = cfg.get("default_profile")
    if effective:
        path = profile_path(str(effective))
        if not path.is_file():
            raise FileNotFoundError(f"Profile file missing: {path}")
        overlay = json.loads(path.read_text(encoding="utf-8"))
        cfg = _deep_merge(cfg, overlay)
        cfg["profile"] = str(effective)
    return cfg


def config_fingerprint(cfg: dict[str, Any], *, keys: tuple[str, ...] | None = None) -> str:
    """Stable short hash of decision-relevant config (for OOS audit trails)."""
    if keys is None:
        keys = (
            "profile",
            "cfd_leverage",
            "position_frac",
            "min_p_up_long",
            "long_quintile_min",
            "max_positions",
            "stop_loss_pct",
            "use_trailing_stop",
            "trailing_stop_pct",
            "max_hold_days",
            "regime_filter",
            "bear_scale",
            "risk_limits",
        )
    subset = {k: cfg.get(k) for k in keys if k in cfg}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
