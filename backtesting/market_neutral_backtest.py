#!/usr/bin/env python3
"""
Cross-sectional long/short and SPY-hedged backtest on ML scores at each checkpoint.

Uses signals from strategy_backtest (--strategy ml). At each checkpoint:
  - Long top K names by ml_score (P(up) 20d)
  - Short bottom K names (equal weight dollar-neutral book)
  - Optional: subtract SPY 6M return (beta-hedge proxy)

Also prints ML quintile monotonicity and saves a cumulative return HTML chart.

Usage:
    python backtesting/market_neutral_backtest.py
    python backtesting/market_neutral_backtest.py --leg-size 15 --horizon 6
    python backtesting/market_neutral_backtest.py --universe-dir backtesting/universes/dollar_volume_top100
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stock_analyzer"))

import backtesting.strategy_backtest as st
from backtesting.ml_quant import (
    aggregate_quintile_forward_returns,
    ml_score_from_signal,
    print_quintile_table,
)
from backtesting.regime import gross_exposure_scale, spy_close_series
from backtesting.yearly_top100_universe import default_universe_cache_dir


def _simulate_legs(
    signals: list[dict],
    checkpoints: list[datetime],
    *,
    leg_size: int,
    horizon_months: int,
    cost_bps_per_leg: float,
    regime_spy: pd.Series | None,
    bear_scale: float,
) -> pd.DataFrame:
    fwd_key = f"fwd_{horizon_months}m"
    spy_key = f"spy_fwd_{horizon_months}m"
    by_cp: dict[datetime, list[dict]] = defaultdict(list)
    for s in signals:
        cp = s["date"]
        if isinstance(cp, pd.Timestamp):
            cp = cp.to_pydatetime()
        sc = ml_score_from_signal(s)
        if sc is None:
            continue
        r = s.get(fwd_key)
        if r is None:
            continue
        by_cp[cp].append({**s, "ml_score": sc, "_fwd": float(r)})

    rows: list[dict] = []
    cps = sorted(by_cp.keys())
    cost = cost_bps_per_leg / 10_000.0

    for cp in cps:
        pool = by_cp[cp]
        if len(pool) < leg_size * 2:
            continue
        pool.sort(key=lambda x: x["ml_score"], reverse=True)
        longs = pool[:leg_size]
        shorts = pool[-leg_size:]
        long_ret = float(np.mean([x["_fwd"] for x in longs]))
        short_ret = float(np.mean([x["_fwd"] for x in shorts]))
        ls_ret = long_ret - short_ret - 2 * cost

        spy_fwd = None
        if longs and longs[0].get(spy_key) is not None:
            spy_fwd = float(longs[0][spy_key])
        hedged = ls_ret - spy_fwd if spy_fwd is not None else None

        scale = 1.0
        if regime_spy is not None and len(regime_spy):
            scale = gross_exposure_scale(regime_spy, cp, bear_scale=bear_scale)

        rows.append({
            "date": cp,
            "n": len(pool),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_ret": ls_ret * scale,
            "hedged_ret": (hedged * scale) if hedged is not None else None,
            "regime_scale": scale,
            "avg_long_score": float(np.mean([x["ml_score"] for x in longs])),
            "avg_short_score": float(np.mean([x["ml_score"] for x in shorts])),
        })

    return pd.DataFrame(rows)


def _compound_returns(rets: pd.Series) -> pd.Series:
    return (1.0 + rets.fillna(0.0)).cumprod()


def _annualized(rets: pd.Series, periods_per_year: float = 2.0) -> float | None:
    if rets.empty:
        return None
    total = float((1.0 + rets).prod() - 1.0)
    n = len(rets)
    if n <= 0:
        return None
    return (1.0 + total) ** (periods_per_year / n) - 1.0


def _write_html(curve: pd.DataFrame, path: Path, title: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
    fig.add_trace(
        go.Scatter(x=curve["date"], y=curve["ls_cum"], name="L/S (top−bottom)", line=dict(color="#2563eb")),
        row=1,
        col=1,
    )
    if "hedged_cum" in curve.columns:
        fig.add_trace(
            go.Scatter(x=curve["date"], y=curve["hedged_cum"], name="L/S − SPY", line=dict(color="#16a34a")),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(x=curve["date"], y=curve["ls_ret"], name="Period L/S", marker_color="#94a3b8"),
        row=2,
        col=1,
    )
    fig.update_layout(title=title, height=520, template="plotly_white", legend=dict(orientation="h"))
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".1%", row=2, col=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def run_market_neutral_from_results(
    results: dict,
    *,
    leg_size: int = 12,
    horizon_months: int = 6,
    cost_bps: float = 15.0,
    bear_scale: float = 0.35,
) -> dict:
    """Run L/S simulation on an existing ``run_backtest`` result dict."""
    signals = results["signals"]
    checkpoints = results["checkpoints"]

    quintiles = results.get("ml_quintiles") or aggregate_quintile_forward_returns(
        signals, horizon_months=horizon_months
    )
    print_quintile_table(quintiles, horizon_months=horizon_months)

    spy_hist = yf.Ticker("SPY").history(period="max", interval="1d")
    spy_ser = spy_close_series(spy_hist)

    legs = _simulate_legs(
        signals,
        checkpoints,
        leg_size=leg_size,
        horizon_months=horizon_months,
        cost_bps_per_leg=cost_bps,
        regime_spy=spy_ser,
        bear_scale=bear_scale,
    )
    if legs.empty:
        raise RuntimeError("No checkpoint periods with enough scored names for L/S legs.")

    legs["ls_cum"] = _compound_returns(legs["ls_ret"])
    if legs["hedged_ret"].notna().any():
        legs["hedged_cum"] = _compound_returns(legs["hedged_ret"].fillna(0.0))

    ann_ls = _annualized(legs["ls_ret"])
    ann_h = _annualized(legs["hedged_ret"].dropna()) if legs["hedged_ret"].notna().any() else None

    print(f"\n{'═' * 60}")
    print("  MARKET-NEUTRAL ML BACKTEST (checkpoint legs)")
    print(f"{'═' * 60}")
    print(f"  Periods:        {len(legs)}")
    print(f"  Leg size:       {leg_size} long + {leg_size} short")
    print(f"  Horizon:        {horizon_months}M forward per leg")
    print(f"  Costs:          {cost_bps:.0f} bps per leg side")
    print(f"  Bear scaling:   {bear_scale:.0%} gross when SPY < 200d MA")
    if ann_ls is not None:
        print(f"  Ann. L/S return: {ann_ls:+.1%}  (compound of {len(legs)} non-overlapping legs*)")
    if ann_h is not None:
        print(f"  Ann. hedged:     {ann_h:+.1%}  (L/S − SPY fwd)")
    print(f"  Total L/S compound: {float(legs['ls_cum'].iloc[-1]) - 1.0:+.1%}")
    print("  *Legs overlap in calendar time; treat as diagnostic, not live P&L.")

    out_html = _ROOT / f"market_neutral_ml_{datetime.today().strftime('%Y%m%d')}.html"
    _write_html(legs, out_html, "ML long/short by score quintile (checkpoint legs)")

    return {
        "legs": legs,
        "quintiles": quintiles,
        "ann_ls": ann_ls,
        "ann_hedged": ann_h,
        "html": out_html,
        "n_signals": len(signals),
    }


def run_market_neutral(
    *,
    lookback_years: int = 5,
    checkpoint_min_year: int | None = 2023,
    universe_dir: Path | None = None,
    leg_size: int = 12,
    horizon_months: int = 6,
    cost_bps: float = 15.0,
    bear_scale: float = 0.35,
    auto_build_universe: bool = False,
) -> dict:
    print("\nCollecting ML signals for market-neutral simulation…")
    results = st.run_backtest(
        None,
        lookback_years=lookback_years,
        benchmark="SPY",
        yearly_top100=True,
        universe_cache_dir=universe_dir,
        checkpoint_min_year=checkpoint_min_year,
        auto_build_missing_universe=auto_build_universe,
        use_valuation=False,
        signal_mode="ml",
    )
    return run_market_neutral_from_results(
        results,
        leg_size=leg_size,
        horizon_months=horizon_months,
        cost_bps=cost_bps,
        bear_scale=bear_scale,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--checkpoint-min-year", type=int, default=2023)
    ap.add_argument("--universe-dir", type=Path, default=None)
    ap.add_argument("--leg-size", type=int, default=12, help="Names per long and short leg")
    ap.add_argument("--horizon", type=int, default=6, help="Forward return months per leg")
    ap.add_argument("--cost-bps", type=float, default=15.0)
    ap.add_argument("--bear-scale", type=float, default=0.35)
    ap.add_argument("--auto-build-universe", action="store_true")
    args = ap.parse_args()

    udir = args.universe_dir or default_universe_cache_dir(_ROOT)
    out = run_market_neutral(
        lookback_years=args.lookback,
        checkpoint_min_year=args.checkpoint_min_year if args.checkpoint_min_year else None,
        universe_dir=udir,
        leg_size=args.leg_size,
        horizon_months=args.horizon,
        cost_bps=args.cost_bps,
        bear_scale=args.bear_scale,
        auto_build_universe=args.auto_build_universe,
    )
    print(f"\n  Chart -> {out['html']}")


if __name__ == "__main__":
    main()
