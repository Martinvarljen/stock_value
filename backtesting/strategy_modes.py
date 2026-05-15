"""Backtest signal mode helpers (DCF vs ML projection stack)."""

from __future__ import annotations

# Canonical modes passed through run_backtest / run_dynamic
MODE_DCF = "dcf"
MODE_ML = "ml"  # user-facing name
MODE_ML_INTERNAL = "tech_ai"  # implementation id in classify_at


def normalize_signal_mode(mode: str | None) -> str:
    """
    Return internal mode for classify_at: ``dcf`` or ``tech_ai``.

    Accepts: ml, tech_ai, tech-ai, projection, dolt_ml → tech_ai.
    """
    if not mode:
        return MODE_DCF
    m = str(mode).strip().lower().replace("-", "_")
    if m in (MODE_ML, "tech_ai", "techai", "projection", "dolt_ml", "ml_projection"):
        return MODE_ML_INTERNAL
    return MODE_DCF


def is_ml_strategy(mode: str | None) -> bool:
    return normalize_signal_mode(mode) == MODE_ML_INTERNAL


def strategy_display_name(mode: str | None) -> str:
    if is_ml_strategy(mode):
        return "ml (technicals + Dolt-trained LightGBM via projection_engine)"
    return "dcf (classification_engine + optional DCF)"
