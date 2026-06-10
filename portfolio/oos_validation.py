"""OOS validation helpers: frozen config export and yearly CAGR breakdown."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio.config_loader import load_config

# Keys not copied into frozen OOS config (runtime / reporting only).
_FROZEN_SKIP = frozenset({
    "paper_oos",
    "daily_run",
    "backtest_defaults",
    "default_profile",
})


def yearly_performance(curve: pd.DataFrame) -> list[dict[str, Any]]:
    """Calendar-year strategy vs SPY from equity curve (columns strategy, spy_bh)."""
    if curve is None or curve.empty:
        return []
    df = curve.sort_index().copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[df.index.notna()]
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for year, grp in df.groupby(df.index.year):
        if len(grp) < 2:
            continue
        s0, s1 = float(grp["strategy"].iloc[0]), float(grp["strategy"].iloc[-1])
        b0, b1 = float(grp["spy_bh"].iloc[0]), float(grp["spy_bh"].iloc[-1])
        days = max((grp.index[-1] - grp.index[0]).days, 1)
        y_frac = days / 365.25
        strat_ret = s1 / s0 - 1.0 if s0 > 0 else None
        spy_ret = b1 / b0 - 1.0 if b0 > 0 else None
        strat_cagr = (s1 / s0) ** (1.0 / y_frac) - 1.0 if s0 > 0 and s1 > 0 else None
        spy_cagr = (b1 / b0) ** (1.0 / y_frac) - 1.0 if b0 > 0 and b1 > 0 else None
        beat = (
            strat_ret is not None
            and spy_ret is not None
            and strat_ret > spy_ret
        )
        rows.append({
            "year": int(year),
            "strategy_return": strat_ret,
            "spy_return": spy_ret,
            "strategy_cagr": strat_cagr,
            "spy_cagr": spy_cagr,
            "beat_spy": beat,
            "n_days": int(len(grp)),
        })
    return rows


def format_yearly_table(yearly: list[dict[str, Any]]) -> str:
    if not yearly:
        return "(no yearly data)"
    lines = [
        "| Year | Strat return | SPY return | Beat SPY | Days |",
        "|------|-------------|------------|----------|------|",
    ]
    for r in yearly:
        sr = r.get("strategy_return")
        br = r.get("spy_return")
        sr_s = f"{sr:+.1%}" if sr is not None else "n/a"
        br_s = f"{br:+.1%}" if br is not None else "n/a"
        beat = "yes" if r.get("beat_spy") else "no"
        lines.append(
            f"| {r['year']} | {sr_s} | {br_s} | {beat} | {r.get('n_days', 0)} |"
        )
    wins = sum(1 for r in yearly if r.get("beat_spy"))
    lines.append("")
    lines.append(f"Beat SPY in **{wins}/{len(yearly)}** calendar years.")
    if wins <= max(1, len(yearly) // 3):
        lines.append(
            "_Flag: edge concentrated in few years — check robustness before scaling capital._"
        )
    return "\n".join(lines)


def write_frozen_config(
    path: Path,
    *,
    train_through_year: int,
    oos_from_year: int,
    source: str = "portfolio/config.json",
    extra: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> Path:
    """
    Snapshot current research_ls config for OOS runs.

    Does not run a sweep — locks whatever is in ``config.json`` today.
    After a threshold sweep, pass ``extra=chosen.cfg`` from sweep output.
    """
    base = dict(cfg or load_config())
    out: dict[str, Any] = {
        "_comment": (
            "Locked thresholds for OOS. Train window should end at "
            "oos_train_through_year; test from oos_test_from_year onward only."
        ),
        "oos_train_through_year": train_through_year,
        "oos_test_from_year": oos_from_year,
        "frozen_at": date.today().isoformat(),
        "frozen_from": source,
    }
    if extra:
        out["sweep_label"] = extra.pop("sweep_label", None)
        out.update({k: v for k, v in extra.items() if not str(k).startswith("_")})
    for key, val in base.items():
        if key.startswith("_") or key in _FROZEN_SKIP:
            continue
        if key not in out:
            out[key] = val
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path
