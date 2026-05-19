"""Attribute strategy returns by SPY bull/bear/unknown regime."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backtesting.performance_metrics import max_drawdown, summarize_backtest
from backtesting.regime import regime_signal, spy_close_series


def _daily_returns(nav: pd.Series) -> pd.Series:
    s = nav.astype(float)
    return s.pct_change().dropna()


def regime_labels_for_index(
    index: pd.DatetimeIndex,
    spy_close: pd.Series,
    *,
    ma_days: int = 200,
) -> pd.Series:
    labels = []
    for ts in index:
        as_of = ts.to_pydatetime().replace(tzinfo=None)
        labels.append(regime_signal(spy_close, as_of, ma_days=ma_days))
    return pd.Series(labels, index=index, name="regime")


def attribute_by_regime(
    curve: pd.DataFrame,
    *,
    nav_col: str = "strategy",
    spy_close: pd.Series | None = None,
    ma_days: int = 200,
) -> dict[str, Any]:
    """Split period returns by regime; return per-regime risk stats."""
    if curve.empty or nav_col not in curve.columns:
        return {"error": "empty_curve"}

    df = curve.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if spy_close is None:
        return {"error": "spy_close_required"}

    spy = spy_close_series(spy_close) if hasattr(spy_close, "columns") else spy_close
    df["regime"] = regime_labels_for_index(df.index, spy, ma_days=ma_days)
    rets = _daily_returns(df[nav_col])

    out: dict[str, Any] = {"regimes": {}, "combined": {}}
    aligned = pd.DataFrame({"return": rets}).join(df[["regime"]], how="inner")

    for label in ("bull", "bear", "unknown"):
        sub = aligned[aligned["regime"] == label]["return"].values
        if len(sub) < 2:
            out["regimes"][label] = {"n_days": int(len(sub)), "skipped": "insufficient_days"}
            continue
        eq = np.cumprod(np.concatenate([[1.0], 1.0 + sub]))
        stats = summarize_backtest(sub, eq, periods_per_year=252.0)
        stats["n_days"] = int(len(sub))
        stats["pct_of_days"] = round(len(sub) / len(aligned), 4) if len(aligned) else 0.0
        out["regimes"][label] = stats

    full_rets = aligned["return"].values
    full_eq = df[nav_col].astype(float).values
    out["combined"] = summarize_backtest(full_rets, full_eq, periods_per_year=252.0)
    out["regime_day_counts"] = aligned["regime"].value_counts().to_dict()
    return out


def attribute_costs_from_ledger(ledger: list[dict[str, Any]]) -> dict[str, float]:
    """Sum modelled costs from trade ledger rows."""
    overnight = 0.0
    exit_cost = 0.0
    enter_rows = 0
    exit_rows = 0
    for row in ledger:
        act = str(row.get("action", ""))
        if act == "OVERNIGHT_INTEREST":
            overnight += float(row.get("overnight_charge", 0) or 0)
        elif act == "EXIT":
            exit_rows += 1
            exit_cost += float(row.get("exit_cost", 0) or 0)
        elif act.startswith("ENTER"):
            enter_rows += 1
    return {
        "overnight_total": round(overnight, 6),
        "exit_cost_total": round(exit_cost, 6),
        "n_entries": enter_rows,
        "n_exits": exit_rows,
    }
