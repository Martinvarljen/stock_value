#!/usr/bin/env python3
"""
Backtest the daily agent rules vs SPY buy-and-hold.

Uses the same decision logic as ``portfolio/daily_run.py`` on historical data:
  - Point-in-time ML scores (refreshed every ``--signal-step`` trading days)
  - Daily stop / take-profit / max-hold checks
  - Regime filter on SPY 200d MA

Examples::

    python portfolio/backtest.py --from-year 2015 --max-tickers 100
    python portfolio/backtest.py --from-year 2020 --signal-step 1 --max-tickers 20

Requires yearly universe files (build with build_yearly_top100_universe.py).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import backtesting.strategy_backtest as st
from backtesting.regime import gross_exposure_scale, spy_bull_regime, spy_close_series
from backtesting.yearly_top100_universe import default_universe_cache_dir, load_universe_map_for_lag_years
from portfolio.broker import apply_decisions, mark_nav_prices
from portfolio.decisions import Action, TickerDecision, decide_ticker, prioritize_entries, _quintile_map
from portfolio.score_pt import score_ticker_at
from portfolio.backtest_flow_html import write_flow_map_html
from portfolio.backtest_viz import write_backtest_report
from portfolio.decision_trace import stage_path_ids, trace_pipeline_stages
from portfolio.store import default_state, load_config


def _norm_day(ts) -> datetime:
    d = pd.Timestamp(ts).to_pydatetime()
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _row_asof(df: pd.DataFrame, d: datetime) -> pd.Series | None:
    ts = pd.Timestamp(_norm_day(d))
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
        idx = df.index
    pos = int(idx.searchsorted(ts, side="right") - 1)
    if pos < 0:
        return None
    return df.iloc[pos]


def _intraday_exit_decisions(
    state,
    d: date,
    raw_by: dict,
) -> list[TickerDecision]:
    out: list[TickerDecision] = []
    dt = datetime.combine(d, datetime.min.time())
    for pos in state.positions:
        raw = raw_by.get(pos.ticker.upper())
        if not raw:
            continue
        ph = raw.get("price_history")
        if ph is None or ph.empty:
            continue
        row = _row_asof(ph, dt)
        if row is None:
            continue
        lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
        exit_px = None
        reason = ""
        held = (d - date.fromisoformat(pos.entry_date)).days
        if held >= pos.max_hold_days:
            exit_px, reason = cl, f"Max hold {pos.max_hold_days}d"
        elif pos.side == "long":
            if lo <= pos.stop_price:
                exit_px, reason = pos.stop_price, "Stop hit (long)"
            elif hi >= pos.take_profit_price:
                exit_px, reason = pos.take_profit_price, "Take-profit (long)"
        else:
            if hi >= pos.stop_price:
                exit_px, reason = pos.stop_price, "Stop hit (short)"
            elif lo <= pos.take_profit_price:
                exit_px, reason = pos.take_profit_price, "Take-profit (short)"
        if exit_px is not None:
            out.append(
                TickerDecision(
                    ticker=pos.ticker,
                    action=Action.EXIT,
                    reason=reason,
                    price=exit_px,
                )
            )
    return out


def _universe_for_day(uni: dict[int, list[str]], d: date) -> set[str]:
    ly = d.year - 1
    return {t.upper() for t in uni.get(ly, [])}


def _cagr(start: float, end: float, years: float) -> float | None:
    if years <= 0 or start <= 0 or end <= 0:
        return None
    try:
        r = (end / start) ** (1 / years) - 1
        return r if math.isfinite(r) else None
    except Exception:
        return None


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    peak = series.cummax()
    dd = (series - peak) / peak.replace(0, float("nan"))
    return float(dd.min()) if len(dd) else 0.0


def run_backtest(
    *,
    from_year: int,
    to_year: int | None,
    max_tickers: int | None,
    signal_step: int,
    out_html: Path | None,
    out_json: Path | None = None,
    out_flow_html: Path | None = None,
    initial_capital: float | None = None,
) -> dict:
    cfg = load_config()
    to_year = to_year or datetime.today().year
    start = date(from_year, 1, 15)
    end = date(to_year, 12, 15)
    if end > date.today():
        end = date.today()

    lag_years = list(range(from_year - 1, to_year))
    udir = default_universe_cache_dir(_ROOT)
    try:
        uni = load_universe_map_for_lag_years(lag_years, udir, auto_build_missing=False, verbose=True)
    except FileNotFoundError as e:
        raise SystemExit(
            f"{e}\nBuild: python backtesting/build_yearly_top100_universe.py --from {from_year - 1} --to {to_year - 1}"
        ) from e

    freq: Counter[str] = Counter()
    for y in lag_years:
        for t in uni.get(y, []):
            freq[str(t).upper()] += 1
    all_tickers = sorted(freq.keys(), key=lambda t: (-freq[t], t))
    if max_tickers and max_tickers > 0:
        all_tickers = all_tickers[:max_tickers]

    print(f"\nAgent backtest {start} → {end} | {len(all_tickers)} tickers | signal every {signal_step}d\n")

    spy_hist = yf.Ticker("SPY").history(period="max", interval="1d")
    spy_close = spy_close_series(spy_hist)
    spy_df = spy_hist.copy()
    if spy_df.index.tz is not None:
        spy_df.index = spy_df.index.tz_localize(None)

    raw_by: dict[str, dict] = {}
    for i, tk in enumerate(all_tickers, 1):
        print(f"  Load {tk} ({i}/{len(all_tickers)}) …", flush=True)
        try:
            raw_by[tk] = st.collect_raw_yfinance(tk)
        except Exception as e:
            print(f"    skip: {e}")

    trading_days = [
        _norm_day(x)
        for x in spy_df.index
        if start <= _norm_day(x).date() <= end
    ]
    if not trading_days:
        raise SystemExit("No trading days in range.")

    spy0 = float(_row_asof(spy_df, trading_days[0])["Close"])
    state = default_state(cfg)
    scores: dict[str, dict] = {}
    curve: list[dict] = []
    ledger: list[dict] = []
    snapshots: list[dict] = []
    traces: list[dict] = []

    for di, dt in enumerate(trading_days):
        d = dt.date()
        refresh = di % max(1, signal_step) == 0
        allowed = _universe_for_day(uni, d) | state.open_tickers()

        if refresh:
            as_of = dt
            batch: list[dict] = []
            for tk in sorted(allowed):
                raw = raw_by.get(tk)
                if not raw:
                    continue
                try:
                    sc = score_ticker_at(raw, as_of, spy_close=spy_close)
                except Exception:
                    sc = None
                if sc:
                    scores[tk] = sc
                    if sc.get("ok"):
                        batch.append(sc)
            if len(batch) >= 5:
                qmap = _quintile_map(batch)
            else:
                qmap = {}

        regime = {
            "spy_bull": spy_bull_regime(spy_close, dt),
            "gross_exposure_scale": gross_exposure_scale(spy_close, dt, bear_scale=float(cfg.get("bear_scale", 0.35))),
        }
        cfg_run = {**cfg, "_regime_scale": regime["gross_exposure_scale"]}

        scan = allowed | state.open_tickers()
        closes: dict[str, float] = {}
        analyses: list[dict] = []
        for tk in sorted(scan):
            sc = scores.get(tk)
            if sc and sc.get("price"):
                closes[tk] = float(sc["price"])
            else:
                raw = raw_by.get(tk)
                if raw:
                    row = _row_asof(raw["price_history"], dt)
                    if row is not None:
                        closes[tk] = float(row["Close"])
                        if sc:
                            sc = {**sc, "price": closes[tk]}
            if sc:
                analyses.append(sc)

        qmap_full = _quintile_map([a for a in analyses if a.get("ok") and a.get("ml_score") is not None])

        decisions: list[TickerDecision] = []
        forced = {x.ticker: x for x in _intraday_exit_decisions(state, d, raw_by)}
        for a in analyses:
            tk = a["ticker"]
            pos = state.position_for(tk)
            if tk in forced:
                decisions.append(forced[tk])
                continue
            dec = decide_ticker(
                a,
                pos,
                quintile=qmap_full.get(tk),
                regime=regime,
                cfg=cfg_run,
                as_of=d,
            )
            if dec.price is None and tk in closes:
                dec.price = closes[tk]
            decisions.append(dec)

        decisions = prioritize_entries(decisions, cfg_run)
        day_rows = apply_decisions(state, decisions, run_date=d, cfg=cfg_run, write_ledger=False)
        ledger.extend(day_rows)

        for dec in decisions:
            if dec.action.value == "NO_TRADE" and not refresh:
                continue
            a = next((x for x in analyses if x.get("ticker") == dec.ticker), None)
            if not a:
                if dec.action.value != "EXIT":
                    continue
                a = {
                    "ticker": dec.ticker,
                    "ok": True,
                    "price": dec.price,
                    "ml_score": dec.ml_score,
                    "p_up_20d": dec.p_up_20d,
                    "critical_flags": [],
                }
            had_pos = state.position_for(dec.ticker) is not None or any(
                r.get("ticker") == dec.ticker and r.get("action", "").startswith("ENTER") for r in day_rows
            )
            stages = trace_pipeline_stages(
                a,
                dec,
                regime=regime,
                cfg=cfg,
                had_position=had_pos,
            )
            traces.append(
                {
                    "date": d.isoformat(),
                    "ticker": dec.ticker,
                    "action": dec.action.value,
                    "reason": dec.reason,
                    "ml_score": dec.ml_score,
                    "p_up_20d": dec.p_up_20d,
                    "quintile": dec.quintile,
                    "regime_scale": regime["gross_exposure_scale"],
                    "critical_flags": bool(a.get("critical_flags")),
                    "path": stage_path_ids(stages),
                    "stages": stages,
                }
            )

        mark_nav_prices(state, closes)
        pos_snap = []
        for p in state.positions:
            px = closes.get(p.ticker.upper(), p.entry_price)
            if p.side == "long":
                upnl = (px - p.entry_price) / p.entry_price
            else:
                upnl = (p.entry_price - px) / p.entry_price
            pos_snap.append(
                {
                    "ticker": p.ticker,
                    "side": p.side,
                    "entry_date": p.entry_date,
                    "entry_price": p.entry_price,
                    "notional": p.notional,
                    "unrealized_pnl_pct": round(upnl, 4),
                    "p_up_20d_at_entry": p.p_up_20d_at_entry,
                }
            )
        snapshots.append({"date": d.isoformat(), "nav": state.nav, "positions": pos_snap})

        spy_row = _row_asof(spy_df, dt)
        spy_bh = float(spy_row["Close"]) / spy0 if spy_row is not None else float("nan")
        curve.append({"date": d, "strategy": state.nav, "spy_bh": spy_bh})

        if di % 126 == 0:
            print(f"  … {d} NAV {state.nav:.4f} vs SPY {spy_bh:.4f}", flush=True)

    df = pd.DataFrame(curve).set_index("date")
    years = (df.index[-1] - df.index[0]).days / 365.25
    s0, s1 = float(df["strategy"].iloc[0]), float(df["strategy"].iloc[-1])
    b0, b1 = float(df["spy_bh"].iloc[0]), float(df["spy_bh"].iloc[-1])
    cagr_s = _cagr(s0, s1, years)
    cagr_b = _cagr(b0, b1, years)
    mdd_s = _max_drawdown(df["strategy"])
    mdd_b = _max_drawdown(df["spy_bh"])

    cap = float(initial_capital) if initial_capital and initial_capital > 0 else None
    summary = {
        "from": str(start),
        "to": str(end),
        "years": round(years, 2),
        "tickers": len(raw_by),
        "signal_step_days": signal_step,
        "strategy_cagr": cagr_s,
        "spy_cagr": cagr_b,
        "strategy_max_dd": mdd_s,
        "spy_max_dd": mdd_b,
        "final_nav": s1,
        "final_spy_bh": b1,
        "beat_spy": (cagr_s is not None and cagr_b is not None and cagr_s > cagr_b),
        "initial_capital": cap,
        "final_strategy_usd": (s1 * cap) if cap else None,
        "final_spy_usd": (b1 * cap) if cap else None,
        "strategy_total_return": (s1 / s0 - 1.0) if s0 > 0 else None,
        "spy_total_return": (b1 / b0 - 1.0) if b0 > 0 else None,
    }

    print("\n=== Agent backtest summary ===")
    print(f"  Period:        {start} → {end} ({years:.1f} y)")
    print(f"  Strategy CAGR: {cagr_s:.1%}" if cagr_s is not None else "  Strategy CAGR: n/a")
    print(f"  SPY CAGR:      {cagr_b:.1%}" if cagr_b is not None else "  SPY CAGR: n/a")
    print(f"  Strategy max DD: {mdd_s:.1%}  |  SPY max DD: {mdd_b:.1%}")
    print(f"  Final NAV:     {s1:.4f}  |  SPY B&H: {b1:.4f}")
    if cap:
        print(f"  Start ${cap:,.0f} → Strategy ${s1 * cap:,.2f}  |  SPY B&H ${b1 * cap:,.2f}")
        print(f"  Total return:  {(s1 / s0 - 1):.1%} strategy  |  {(b1 / b0 - 1):.1%} SPY")
    if summary["beat_spy"]:
        print("  → Strategy beat SPY on CAGR (research only — not investment advice).")
    else:
        print("  → Strategy did not beat SPY on CAGR in this run.")

    if out_html:
        write_backtest_report(
            curve=df,
            ledger=ledger,
            snapshots=snapshots,
            summary=summary,
            out_html=out_html,
            out_json=out_json,
        )
        print(f"\nReport → {out_html}")
        if out_json:
            print(f"Trade log JSON → {out_json}")

    if out_flow_html:
        write_flow_map_html(
            traces=traces,
            ledger=ledger,
            summary=summary,
            out_html=out_flow_html,
        )
        print(f"Pipeline map → {out_flow_html}")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest portfolio daily agent vs SPY.")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=None)
    ap.add_argument("--max-tickers", type=int, default=100, help="Universe size cap (top-100 dollar volume list has 100 names)")
    ap.add_argument("--signal-step", type=int, default=5, help="Refresh ML scores every N trading days (5≈weekly)")
    ap.add_argument("--out-html", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None, help="Trade ledger + closed trades JSON")
    ap.add_argument(
        "--initial-capital",
        type=float,
        default=None,
        help="Scale report to dollar portfolio (e.g. 10000 for $10k start)",
    )
    args = ap.parse_args()

    stamp = datetime.today().strftime("%Y%m%d")
    out = args.out_html or (_ROOT / f"portfolio_agent_report_{stamp}.html")
    out_json = args.out_json or (_ROOT / f"portfolio_agent_trades_{stamp}.json")
    out_flow = _ROOT / f"portfolio_agent_flow_{stamp}.html"
    run_backtest(
        from_year=args.from_year,
        to_year=args.to_year,
        max_tickers=args.max_tickers if args.max_tickers > 0 else None,
        signal_step=max(1, args.signal_step),
        out_html=out,
        out_json=out_json,
        out_flow_html=out_flow,
        initial_capital=args.initial_capital,
    )


if __name__ == "__main__":
    main()
