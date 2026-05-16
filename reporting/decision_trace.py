"""Decision pipeline traces for branching-map visualizations."""

from __future__ import annotations

from typing import Any

from portfolio.decisions import Action, TickerDecision


def exit_stage_from_reason(reason: str) -> dict[str, str]:
    """Map exit reason to a dedicated pipeline node (stop / TP / time / signal)."""
    r = (reason or "").lower()
    if "stop" in r:
        return {"id": "stop_loss", "label": "Stop loss", "status": "exit"}
    if "take-profit" in r or "take profit" in r:
        return {"id": "take_profit", "label": "Take profit", "status": "exit"}
    if "max hold" in r:
        return {"id": "time_exit", "label": "Max hold (time)", "status": "exit"}
    if "p(up)" in r or "p_up" in r:
        return {"id": "signal_exit", "label": "ML signal exit", "status": "exit"}
    if "critical" in r or "flag" in r:
        return {"id": "signal_exit", "label": "Risk exit", "status": "exit"}
    return {"id": "time_exit", "label": "Exit rule", "status": "exit"}


def trace_pipeline_stages(
    analysis: dict[str, Any],
    decision: TickerDecision,
    *,
    regime: dict[str, Any],
    cfg: dict[str, Any],
    had_position: bool,
    forced_exit: bool = False,
) -> list[dict[str, str]]:
    """
    Ordered stages a ticker passed through (for graph highlighting).

    Stages: universe → data → ml → quintile → regime → committee → [exit type] → action
    """
    stages: list[dict[str, str]] = [
        {"id": "universe", "label": "Universe", "status": "pass"},
    ]

    if not analysis.get("ok"):
        stages.append({"id": "data", "label": "Data / ML", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        return stages

    stages.append({"id": "data", "label": "Data / ML", "status": "pass"})

    if analysis.get("critical_flags") and not had_position:
        stages.append({"id": "risk", "label": "Risk gate", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        return stages

    if analysis.get("critical_flags") and had_position:
        stages.append({"id": "risk", "label": "Risk gate", "status": "warn"})

    p_up = analysis.get("p_up_20d")
    if p_up is None and decision.ml_score is None and decision.action != Action.EXIT:
        stages.append({"id": "ml", "label": "ML engine", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        return stages

    if p_up is not None or decision.ml_score is not None:
        stages.append({"id": "ml", "label": "ML engine", "status": "pass"})

    q = decision.quintile
    if q is None and decision.action not in (Action.EXIT,):
        stages.append({"id": "quintile", "label": "Quintile rank", "status": "fail"})
        stages.append({"id": "skip", "label": "Skip", "status": "terminal"})
        return stages

    if q is not None:
        q_long = int(cfg.get("long_quintile_min", 4))
        q_short = int(cfg.get("short_quintile_max", 2))
        if q >= q_long:
            stages.append({"id": "quintile", "label": f"Quintile Q{q}", "status": "bull"})
        elif q <= q_short:
            stages.append({"id": "quintile", "label": f"Quintile Q{q}", "status": "bear"})
        else:
            stages.append({"id": "quintile", "label": f"Quintile Q{q}", "status": "neutral"})

    scale = float(regime.get("gross_exposure_scale", 1.0))
    if scale < 0.2:
        stages.append({"id": "regime", "label": "Regime (risk-off)", "status": "fail"})
    elif scale < 1.0:
        stages.append({"id": "regime", "label": "Regime (caution)", "status": "warn"})
    else:
        stages.append({"id": "regime", "label": "Regime (risk-on)", "status": "pass"})

    if "Deferred" in (decision.reason or ""):
        stages.append({"id": "committee", "label": "Entry committee", "status": "cap"})
    elif decision.action in (Action.ENTER_LONG, Action.ENTER_SHORT):
        stages.append({"id": "committee", "label": "Entry committee", "status": "pass"})
    else:
        stages.append({"id": "committee", "label": "Entry committee", "status": "neutral"})

    if decision.action == Action.EXIT:
        stages.append(exit_stage_from_reason(decision.reason or ""))

    act = decision.action.value
    status = {
        "ENTER_LONG": "long",
        "ENTER_SHORT": "short",
        "EXIT": "exit",
        "HOLD": "hold",
        "NO_TRADE": "flat",
    }.get(act, "flat")
    stages.append({"id": "action", "label": act.replace("_", " "), "status": status})
    return stages


def stage_path_ids(stages: list[dict[str, str]]) -> list[str]:
    return [s["id"] for s in stages]
