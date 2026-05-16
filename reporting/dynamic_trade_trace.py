"""Pipeline traces for dynamic_portfolio_backtest (ML rank / tier entries)."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _quintile_from_p_up(p_up: float | None) -> int:
    if p_up is None:
        return 3
    if p_up >= 0.58:
        return 5
    if p_up >= 0.52:
        return 4
    if p_up >= 0.45:
        return 3
    if p_up >= 0.38:
        return 2
    return 1


def build_entry_reason(
    *,
    p_up: float | None,
    cls: str | None,
    min_p_up: float | None,
    require_breakout: bool,
    regime_scale: float,
    entry_mode: str,
    ml_mode: bool,
) -> str:
    parts: list[str] = []
    if ml_mode and entry_mode == "rank" and p_up is not None:
        parts.append(f"P(up)20d={p_up:.1%}")
        if min_p_up is not None:
            parts.append(f">= {min_p_up:.0%} gate")
    elif cls:
        parts.append(f"tier {cls}")
    if require_breakout:
        parts.append("20d high breakout")
    elif ml_mode:
        parts.append("no breakout required")
    if regime_scale < 0.99:
        parts.append(f"regime size {regime_scale:.0%}")
    return "ENTER_LONG: " + ", ".join(parts)


def build_short_entry_reason(
    *,
    p_up: float | None,
    max_p_up: float,
    regime_scale: float,
) -> str:
    parts: list[str] = []
    if p_up is not None:
        parts.append(f"P(up)20d={p_up:.1%}")
        parts.append(f"<= {max_p_up:.0%} short gate")
    if regime_scale < 0.99:
        parts.append(f"regime size {regime_scale:.0%}")
    return "ENTER_SHORT: " + ", ".join(parts)


def build_entry_trace(
    *,
    ticker: str,
    day: datetime,
    p_up: float | None,
    regime_scale: float,
    reason: str,
    min_p_up: float | None,
) -> dict[str, Any]:
    q = _quintile_from_p_up(p_up)
    stages: list[dict[str, str]] = [
        {"id": "universe", "label": "Universe", "status": "pass"},
        {"id": "data", "label": "Data / ML", "status": "pass"},
    ]
    if p_up is None:
        stages.append({"id": "ml", "label": "ML engine", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        path = ["universe", "data", "ml", "skip"]
    else:
        ml_ok = min_p_up is None or p_up >= min_p_up
        stages.append(
            {
                "id": "ml",
                "label": f"ML P(up)20d={p_up:.1%}",
                "status": "pass" if ml_ok else "fail",
            }
        )
        if not ml_ok:
            stages.append({"id": "skip", "label": "Below gate", "status": "terminal"})
            path = ["universe", "data", "ml", "skip"]
        else:
            q_long = 4
            stages.append(
                {
                    "id": "quintile",
                    "label": f"Rank Q{q}",
                    "status": "bull" if q >= q_long else ("bear" if q <= 2 else "neutral"),
                }
            )
            if regime_scale < 0.2:
                stages.append({"id": "regime", "label": "Regime (risk-off)", "status": "fail"})
            elif regime_scale < 1.0:
                stages.append({"id": "regime", "label": f"Regime ({regime_scale:.0%} size)", "status": "warn"})
            else:
                stages.append({"id": "regime", "label": "Regime (risk-on)", "status": "pass"})
            stages.append({"id": "committee", "label": "Entry committee", "status": "pass"})
            stages.append({"id": "action", "label": "ENTER LONG", "status": "long"})
            path = ["universe", "data", "ml", "quintile", "regime", "committee", "action"]
    return {
        "date": day.date().isoformat() if hasattr(day, "date") else str(day)[:10],
        "ticker": ticker.upper(),
        "action": "ENTER_LONG",
        "reason": reason,
        "p_up_20d": p_up,
        "quintile": q,
        "regime_scale": regime_scale,
        "critical_flags": False,
        "path": path,
        "stages": stages,
    }


def build_short_entry_trace(
    *,
    ticker: str,
    day: datetime,
    p_up: float | None,
    regime_scale: float,
    reason: str,
    max_p_up: float,
) -> dict[str, Any]:
    q = _quintile_from_p_up(p_up)
    stages: list[dict[str, str]] = [
        {"id": "universe", "label": "Universe", "status": "pass"},
        {"id": "data", "label": "Data / ML", "status": "pass"},
    ]
    if p_up is None:
        stages.append({"id": "ml", "label": "ML engine", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        path = ["universe", "data", "ml", "skip"]
    elif p_up > max_p_up:
        stages.append({"id": "ml", "label": f"ML P(up)20d={p_up:.1%}", "status": "fail"})
        stages.append({"id": "skip", "label": "Above short gate", "status": "terminal"})
        path = ["universe", "data", "ml", "skip"]
    else:
        stages.append({"id": "ml", "label": f"ML P(up)20d={p_up:.1%}", "status": "bear"})
        stages.append(
            {
                "id": "quintile",
                "label": f"Rank Q{q}",
                "status": "bear" if q <= 2 else "neutral",
            }
        )
        if regime_scale < 0.2:
            stages.append({"id": "regime", "label": "Regime (risk-off)", "status": "warn"})
        elif regime_scale < 1.0:
            stages.append({"id": "regime", "label": f"Regime ({regime_scale:.0%} size)", "status": "warn"})
        else:
            stages.append({"id": "regime", "label": "Regime (risk-on)", "status": "pass"})
        stages.append({"id": "committee", "label": "Entry committee", "status": "pass"})
        stages.append({"id": "action", "label": "ENTER SHORT", "status": "short"})
        path = ["universe", "data", "ml", "quintile", "regime", "committee", "action"]
    return {
        "date": day.date().isoformat() if hasattr(day, "date") else str(day)[:10],
        "ticker": ticker.upper(),
        "action": "ENTER_SHORT",
        "reason": reason,
        "p_up_20d": p_up,
        "quintile": q,
        "regime_scale": regime_scale,
        "critical_flags": False,
        "path": path,
        "stages": stages,
    }


def build_exit_trace(
    *,
    ticker: str,
    day: datetime,
    reason: str,
    p_up_at_entry: float | None,
) -> dict[str, Any]:
    r = reason.lower()
    if "stop" in r:
        exit_id = "stop_loss"
    elif "take profit" in r or "take-profit" in r:
        exit_id = "take_profit"
    elif "max hold" in r:
        exit_id = "time_exit"
    else:
        exit_id = "time_exit"
    path = ["universe", "data", "ml", "quintile", "regime", "committee", exit_id, "action"]
    stages = [
        {"id": "committee", "label": "Open position", "status": "pass"},
        {"id": exit_id, "label": reason, "status": "exit"},
        {"id": "action", "label": "EXIT", "status": "exit"},
    ]
    return {
        "date": day.date().isoformat() if hasattr(day, "date") else str(day)[:10],
        "ticker": ticker.upper(),
        "action": "EXIT",
        "reason": reason,
        "p_up_20d": p_up_at_entry,
        "quintile": _quintile_from_p_up(p_up_at_entry),
        "regime_scale": None,
        "critical_flags": False,
        "path": path,
        "stages": stages,
    }
