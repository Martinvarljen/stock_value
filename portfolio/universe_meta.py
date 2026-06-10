"""Universe / survivorship metadata for backtest reports."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from backtesting.sp500_pit_universe import _DEFAULT_CHANGES_CSV, load_changes


def pit_coverage(start: date, end: date, *, changes_path: Path = _DEFAULT_CHANGES_CSV) -> dict[str, Any]:
    """Describe S&P 500 point-in-time change-log coverage for a backtest window."""
    if not changes_path.is_file():
        return {
            "pit_available": False,
            "pit_coverage": "none",
            "pit_warning": "S&P 500 change log missing; survivorship bias likely.",
            "changes_path": str(changes_path),
        }
    changes = load_changes(changes_path)
    if not changes:
        return {
            "pit_available": True,
            "pit_coverage": "empty",
            "pit_warning": "Change log empty; using current index membership only.",
            "changes_path": str(changes_path),
            "change_events": 0,
        }
    log_min, log_max = changes[0].when, changes[-1].when
    partial = start < log_min or end > log_max
    return {
        "pit_available": True,
        "pit_coverage": "partial" if partial else "full",
        "pit_warning": (
            f"Change log covers {log_min}..{log_max}; window {start}..{end} extends "
            f"outside — membership outside log dates is approximate."
            if partial
            else None
        ),
        "changes_path": str(changes_path),
        "change_log_start": log_min.isoformat(),
        "change_log_end": log_max.isoformat(),
        "change_events": len(changes),
    }


def universe_summary(
    *,
    universe_source: str,
    start: date,
    end: date,
) -> dict[str, Any]:
    """Build summary block embedded in backtest JSON/HTML."""
    src = universe_source.lower()
    out: dict[str, Any] = {
        "universe_source": src,
        "universe_description": (
            "Yearly dollar-volume top-100 (lag year); ranks today's S&P 500 by prior-year volume"
            if src == "legacy"
            else (
                "Yearly top-100 from PIT S&P pool (sp500_changes.csv) ranked by prior-year volume"
                if src == "pit"
                else "Top-100 lists filtered to S&P members as-of each trade day"
            )
        ),
    }
    if src in ("pit", "pit_filter"):
        out.update(pit_coverage(start, end))
        out["survivorship_bias_note"] = (
            "PIT filter removes names not in the index on each day; delisted names "
            "only appear if listed in sp500_changes.csv."
        )
    else:
        out["survivorship_bias_note"] = (
            "Legacy mode ranks today's S&P 500 against prior-year volume; "
            "delisted names are absent unless you rebuild with PIT pool."
        )
        out.update(pit_coverage(start, end))
    return out
