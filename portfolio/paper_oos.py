"""Forward (out-of-sample) paper track vs benchmark and regime breakdown."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from backtesting.performance_metrics import max_drawdown, summarize_backtest
from backtesting.regime import spy_close_series
from portfolio.regime_attribution import attribute_by_regime, attribute_costs_from_ledger
from portfolio.store import DATA_DIR, read_ledger

PAPER_OOS_DIR = DATA_DIR / "paper_oos"
CURVE_PATH = PAPER_OOS_DIR / "curve.jsonl"
META_PATH = PAPER_OOS_DIR / "meta.json"
REPORT_MD_PATH = PAPER_OOS_DIR / "report.md"
REPORT_JSON_PATH = PAPER_OOS_DIR / "report.json"


def _oos_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("paper_oos") or {})


def ensure_paper_oos_dirs() -> None:
    PAPER_OOS_DIR.mkdir(parents=True, exist_ok=True)


def _load_meta() -> dict[str, Any]:
    if META_PATH.is_file():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def _save_meta(meta: dict[str, Any]) -> None:
    ensure_paper_oos_dirs()
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def _spy_close_history(start: date, end: date) -> pd.Series:
    end_ex = (end + timedelta(days=2)).isoformat()
    hist = yf.Ticker("SPY").history(start=start.isoformat(), end=end_ex, interval="1d")
    return spy_close_series(hist)


def _spy_level_on(spy: pd.Series, d: date) -> float | None:
    if spy.empty:
        return None
    ts = pd.Timestamp(d)
    sub = spy[spy.index <= ts]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def record_paper_day(
    run_date: date,
    *,
    nav: float,
    cash: float,
    n_positions: int,
    regime: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Append one OOS observation (idempotent per calendar day)."""
    oos = _oos_cfg(cfg)
    if not oos.get("enabled", True):
        return {}

    ensure_paper_oos_dirs()
    bench = str(oos.get("benchmark", "SPY")).upper()
    start_s = oos.get("oos_start_date")
    if not start_s:
        start_s = run_date.isoformat()
        oos["oos_start_date"] = start_s
        meta_patch = _load_meta()
        meta_patch["oos_start_date"] = start_s
        meta_patch["benchmark"] = bench
        _save_meta(meta_patch)

    oos_start = date.fromisoformat(str(start_s))
    meta = _load_meta()
    if "spy_anchor" not in meta:
        spy_hist = _spy_close_history(oos_start, run_date)
        anchor = _spy_level_on(spy_hist, oos_start)
        if anchor is None or anchor <= 0:
            anchor = 1.0
        meta = {
            "oos_start_date": start_s,
            "benchmark": bench,
            "spy_anchor": anchor,
            "nav_anchor": float(oos.get("nav_anchor", nav)),
        }
        _save_meta(meta)

    spy_hist = _spy_close_history(oos_start, run_date)
    spy_px = _spy_level_on(spy_hist, run_date)
    spy_anchor = float(meta.get("spy_anchor", 1.0))
    spy_bh = (spy_px / spy_anchor) if spy_px and spy_anchor > 0 else float("nan")
    nav_anchor = float(meta.get("nav_anchor", nav))
    nav_norm = nav / nav_anchor if nav_anchor > 0 else nav

    row = {
        "date": run_date.isoformat(),
        "nav": round(nav, 6),
        "nav_norm": round(nav_norm, 6),
        "cash": round(cash, 6),
        "positions": n_positions,
        "spy_bh": round(spy_bh, 6) if spy_bh == spy_bh else None,
        "spy_bull": bool(regime.get("spy_bull")),
        "regime_signal": regime.get("regime_signal"),
        "gross_exposure_scale": regime.get("gross_exposure_scale"),
        "profile": cfg.get("profile"),
        "universe_source": cfg.get("universe_source"),
    }

    existing = {r["date"] for r in load_curve_rows()}
    with CURVE_PATH.open("a", encoding="utf-8") as f:
        if run_date.isoformat() not in existing:
            f.write(json.dumps(row, default=str) + "\n")
        else:
            # rewrite file without this date then append
            rows = [r for r in load_curve_rows() if r["date"] != run_date.isoformat()]
            rows.append(row)
            CURVE_PATH.write_text(
                "\n".join(json.dumps(r, default=str) for r in rows) + "\n",
                encoding="utf-8",
            )

    return row


def load_curve_rows() -> list[dict[str, Any]]:
    if not CURVE_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in CURVE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_curve_df() -> pd.DataFrame:
    rows = load_curve_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def compute_oos_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Metrics since ``paper_oos.oos_start_date``."""
    df = load_curve_df()
    if df.empty or "nav_norm" not in df.columns:
        return {"error": "no_curve", "n_days": 0}

    nav = df["nav_norm"].astype(float)
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return {"error": "insufficient_days", "n_days": len(df)}

    eq = nav.values
    strat = summarize_backtest(rets.values, eq, periods_per_year=252.0)

    spy_line = None
    if "spy_bh" in df.columns and df["spy_bh"].notna().sum() >= 2:
        spy = df["spy_bh"].astype(float)
        spy_rets = spy.pct_change().dropna()
        spy_eq = spy.values
        spy_line = summarize_backtest(spy_rets.values, spy_eq, periods_per_year=252.0)

    meta = _load_meta()
    oos = _oos_cfg(cfg)
    start = oos.get("oos_start_date") or meta.get("oos_start_date")
    end = df.index[-1].date().isoformat()

    spy_hist = _spy_close_history(date.fromisoformat(str(start)), date.fromisoformat(end))
    regime_attr = attribute_by_regime(
        pd.DataFrame({"strategy": nav}),
        nav_col="strategy",
        spy_close=spy_hist,
    )

    ledger = read_ledger()
    costs = attribute_costs_from_ledger(ledger)
    ledger_oos = [
        r for r in ledger
        if r.get("date") and str(r["date"]) >= str(start)
    ]
    costs_oos = attribute_costs_from_ledger(ledger_oos)

    beat = None
    if spy_line and strat.get("cagr") is not None and spy_line.get("cagr") is not None:
        beat = float(strat["cagr"]) > float(spy_line["cagr"])

    return {
        "oos_start": start,
        "oos_end": end,
        "n_days": int(len(df)),
        "strategy": strat,
        "benchmark": spy_line,
        "beat_benchmark_cagr": beat,
        "regime_attribution": regime_attr,
        "costs_all_time": costs,
        "costs_since_oos": costs_oos,
        "final_nav_norm": round(float(nav.iloc[-1]), 6),
        "final_spy_bh": round(float(df["spy_bh"].iloc[-1]), 6) if "spy_bh" in df.columns else None,
        "max_drawdown": round(max_drawdown(nav.values), 6),
    }


def write_oos_report(cfg: dict[str, Any]) -> Path:
    """Write markdown + JSON OOS report under ``portfolio/data/paper_oos/``."""
    ensure_paper_oos_dirs()
    m = compute_oos_metrics(cfg)
    REPORT_JSON_PATH.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Paper OOS track (forward test)",
        "",
        f"Period: **{m.get('oos_start', '?')}** → **{m.get('oos_end', '?')}** ({m.get('n_days', 0)} run-days)",
        "",
        "## Strategy vs benchmark",
        "",
    ]
    s = m.get("strategy") or {}
    b = m.get("benchmark") or {}
    if s.get("cagr") is not None:
        lines.append(f"- Strategy CAGR (OOS): **{s['cagr']:.2%}**")
    if b.get("cagr") is not None:
        lines.append(f"- {cfg.get('paper_oos', {}).get('benchmark', 'SPY')} CAGR: **{b['cagr']:.2%}**")
    lines.append(f"- Beat benchmark on CAGR: **{m.get('beat_benchmark_cagr')}**")
    if m.get("max_drawdown") is not None:
        lines.append(f"- Strategy max DD (OOS): **{m['max_drawdown']:.2%}**")
    if s.get("sharpe") is not None:
        lines.append(f"- Sharpe (OOS): **{s['sharpe']:.2f}**")
    if s.get("psr_vs_zero") is not None:
        lines.append(f"- PSR vs 0: **{s['psr_vs_zero']:.2%}**")

    lines.extend(["", "## Costs (modelled)", ""])
    co = m.get("costs_since_oos") or {}
    lines.append(f"- Overnight (since OOS): **{co.get('overnight_total', 0):.4f}**")
    lines.append(f"- Exit costs (since OOS): **{co.get('exit_cost_total', 0):.4f}**")
    lines.append(f"- Trades: **{co.get('n_entries', 0)}** entries, **{co.get('n_exits', 0)}** exits")

    lines.extend(["", "## Returns by regime (SPY 200d MA)", ""])
    regimes = (m.get("regime_attribution") or {}).get("regimes") or {}
    for label in ("bull", "bear", "unknown"):
        r = regimes.get(label) or {}
        if r.get("skipped"):
            lines.append(f"- **{label}**: n={r.get('n_days', 0)} (too few days)")
            continue
        cagr = r.get("cagr")
        cagr_s = f"{cagr:.2%}" if cagr is not None else "n/a"
        lines.append(
            f"- **{label}**: {r.get('n_days', 0)} days ({100 * r.get('pct_of_days', 0):.0f}%) "
            f"| CAGR {cagr_s} | Sharpe {r.get('sharpe', 'n/a')}"
        )

    lines.extend([
        "",
        "_Research only — not investment advice. OOS paper validates forward behaviour; "
        "long OOS history required before high confidence._",
        "",
    ])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    return REPORT_MD_PATH


def update_after_daily_run(
    run_date: date,
    *,
    state,
    regime: dict[str, Any],
    cfg: dict[str, Any],
) -> Path | None:
    """Record curve point and refresh OOS report."""
    if not _oos_cfg(cfg).get("enabled", True):
        return None
    record_paper_day(
        run_date,
        nav=float(state.nav),
        cash=float(state.cash),
        n_positions=len(state.positions),
        regime=regime,
        cfg=cfg,
    )
    return write_oos_report(cfg)
