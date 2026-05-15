"""
Runtime settings for projection + news (JSON + environment overrides).

Environment variables (optional):
  FINANCE_ML_BLEND              — float 0..1, weight on ML vs rule-based P(up)
  FINANCE_DISAGREEMENT_THRESH   — float, |ml - rule| above this → disagreement flag
  NEWS_DECAY_HALFLIFE_HOURS     — exponential decay half-life for news aggregation
  ANTHROPIC_MODEL               — Claude model id for high-impact headlines
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_SETTINGS_PATH = Path(__file__).resolve().parent / "projection_settings.json"


@dataclass(frozen=True)
class ProjectionSettings:
    ml_blend_weight: float
    ml_rule_disagreement_threshold: float
    probability_uncertainty_half_width: float
    news_decay_half_life_hours: float
    anthropic_model: str


def load_projection_settings() -> ProjectionSettings:
    data: dict = {}
    if _SETTINGS_PATH.exists():
        try:
            data = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}

    def _f(key: str, default: float) -> float:
        v = data.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    blend = _f("ml_blend_weight", 0.6)
    blend = max(0.0, min(1.0, blend))
    thresh = max(0.01, min(0.5, _f("ml_rule_disagreement_threshold", 0.12)))
    half_w = max(0.0, min(0.25, _f("probability_uncertainty_half_width", 0.08)))
    news_h = max(1.0, min(336.0, _f("news_decay_half_life_hours", 72.0)))

    if os.environ.get("FINANCE_ML_BLEND"):
        try:
            blend = max(0.0, min(1.0, float(os.environ["FINANCE_ML_BLEND"])))
        except ValueError:
            pass
    if os.environ.get("FINANCE_DISAGREEMENT_THRESH"):
        try:
            thresh = max(0.01, min(0.5, float(os.environ["FINANCE_DISAGREEMENT_THRESH"])))
        except ValueError:
            pass
    if os.environ.get("NEWS_DECAY_HALFLIFE_HOURS"):
        try:
            news_h = max(1.0, min(336.0, float(os.environ["NEWS_DECAY_HALFLIFE_HOURS"])))
        except ValueError:
            pass

    model = os.environ.get(
        "ANTHROPIC_MODEL",
        data.get("anthropic_model", "claude-haiku-4-5-20251001"),
    )

    return ProjectionSettings(
        ml_blend_weight=blend,
        ml_rule_disagreement_threshold=thresh,
        probability_uncertainty_half_width=half_w,
        news_decay_half_life_hours=news_h,
        anthropic_model=str(model),
    )
