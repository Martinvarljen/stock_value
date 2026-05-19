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
from typing import Any

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import backtesting.strategy_backtest as st
from backtesting.regime import build_regime_snapshot, spy_close_series
from backtesting.yearly_top100_universe import default_universe_cache_dir, load_universe_map_for_lag_years
from portfolio.broker import apply_decisions, mark_nav_prices
from portfolio.decisions import Action, TickerDecision, decide_ticker, prioritize_entries, _quintile_map
from portfolio.score_pt import score_ticker_at
from reporting.backtest_flow_html import write_flow_map_html
from reporting.backtest_viz import write_backtest_report
from reporting.decision_trace import stage_path_ids, trace_pipeline_stages
from portfolio.backtest_invariants import _gross_exposure_pct, validate_backtest_run
from portfolio.config_loader import config_fingerprint
from portfolio.store import default_state, load_config
from portfolio.exit_policy import take_profit_enabled
from portfolio.trailing_stop import stop_exit_label, update_open_trailing_stops
from portfolio.regime_attribution import attribute_by_regime, attribute_costs_from_ledger
from portfolio.universe_meta import universe_summary
from backtesting.performance_metrics import summarize_backtest


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


def _risk_fields_from_bar(row: pd.Series, ph: pd.DataFrame) -> dict[str, float | None]:
    """ATR% and 60d vol for broker sizing (backtest path; mirrors analyze_ticker)."""
    out: dict[str, float | None] = {
        "bar_low": float(row["Low"]),
        "bar_high": float(row["High"]),
        "bar_open": float(row["Open"]) if "Open" in row.index and pd.notna(row["Open"]) else None,
    }
    cl = float(row["Close"])
    if cl <= 0:
        return out
    tail = ph.tail(60)
    if len(tail) >= 15:
        hl = (tail["High"] - tail["Low"]).tail(14).mean()
        out["atr_pct"] = float(hl / cl) if hl and math.isfinite(hl) else None
    if len(tail) >= 21:
        rets = tail["Close"].pct_change().dropna()
        if len(rets) >= 20:
            out["vol_60d_annual"] = float(rets.std(ddof=1) * math.sqrt(252))
    return out


def _intraday_exit_decisions(
    state,
    d: date,
    raw_by: dict,
    cfg: dict | None = None,
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
                exit_px, reason = pos.stop_price, stop_exit_label(pos)
            elif cfg and take_profit_enabled(cfg) and hi >= pos.take_profit_price:
                exit_px, reason = pos.take_profit_price, "Take-profit (long)"
        else:
            if hi >= pos.stop_price:
                exit_px, reason = pos.stop_price, stop_exit_label(pos)
            elif cfg and take_profit_enabled(cfg) and lo <= pos.take_profit_price:
                exit_px, reason = pos.take_profit_price, "Take-profit (short)"
        if exit_px is not None:
            risk = _risk_fields_from_bar(row, ph)
            out.append(
                TickerDecision(
                    ticker=pos.ticker,
                    action=Action.EXIT,
                    reason=reason,
                    price=float(cl),
                    intraday_low=risk.get("bar_low"),
                    intraday_high=risk.get("bar_high"),
                    open_price=risk.get("bar_open"),
                )
            )
    return out


def _universe_for_day(
    uni: dict[int, list[str]],
    d: date,
    *,
    universe_source: str = "legacy",
) -> set[str]:
    ly = d.year - 1
    tickers = {t.upper() for t in uni.get(ly, [])}
    if universe_source in ("pit", "pit_filter"):
        from backtesting.sp500_pit_universe import members_as_of

        pit = members_as_of(d)
        tickers = {t for t in tickers if t in pit}
    return tickers


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
    universe_source: str | None = None,
    frozen_config_path: Path | None = None,
    skip_invariants: bool = False,
) -> dict:
    if frozen_config_path and frozen_config_path.is_file():
        import json as _json

        cfg = _json.loads(frozen_config_path.read_text(encoding="utf-8"))
        cfg["_frozen_config"] = str(frozen_config_path)
    else:
        cfg = load_config()
    uni_src = (
        universe_source
        or cfg.get("backtest_defaults", {}).get("universe_source")
        or "legacy"
    )
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

    prof_label = cfg.get("profile", "default")
    lev_long = cfg.get("long_leverage", cfg.get("cfd_leverage", 1))
    lev_short = cfg.get("short_leverage", cfg.get("cfd_leverage", lev_long))
    print(
        f"\nAgent backtest {start} → {end} | profile={prof_label} | "
        f"leverage long={lev_long}x short={lev_short}x | universe={uni_src} | "
        f"{len(all_tickers)} tickers | signal every {signal_step}d\n"
    )

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
    run_stats: dict[str, Any] = {
        "risk_limit_drops": 0,
        "max_gross_exposure": 0.0,
        "max_positions": 0,
    }

    for di, dt in enumerate(trading_days):
        d = dt.date()
        refresh = di % max(1, signal_step) == 0
        allowed = _universe_for_day(uni, d, universe_source=uni_src) | state.open_tickers()

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
                    ph = raw.get("price_history")
                    if ph is not None and not ph.empty:
                        brow = _row_asof(ph, dt)
                        if brow is not None:
                            sc = {**sc, **_risk_fields_from_bar(brow, ph)}
                    scores[tk] = sc
                    if sc.get("ok"):
                        batch.append(sc)
            if len(batch) >= 5:
                qmap = _quintile_map(batch)
            else:
                qmap = {}

        regime = build_regime_snapshot(
            spy_close, dt, bear_scale=float(cfg.get("bear_scale", 0.35)),
        )
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
                            sc = {
                                **sc,
                                "price": closes[tk],
                                **_risk_fields_from_bar(row, ph),
                            }
            if sc:
                analyses.append(sc)

        qmap_full = _quintile_map([a for a in analyses if a.get("ok") and a.get("ml_score") is not None])

        def _bar_hl(ticker: str):
            raw = raw_by.get(ticker)
            if not raw:
                return None
            ph = raw.get("price_history")
            if ph is None or ph.empty:
                return None
            row = _row_asof(ph, dt)
            if row is None:
                return None
            return float(row["High"]), float(row["Low"])

        update_open_trailing_stops(state, bar_fetcher=_bar_hl, cfg=cfg_run)

        decisions: list[TickerDecision] = []
        forced = {x.ticker: x for x in _intraday_exit_decisions(state, d, raw_by, cfg_run)}
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

        from portfolio.risk_limits import RiskLimits, apply_pre_trade_limits

        limits_cfg = cfg_run.get("risk_limits") or {}
        if limits_cfg.get("enabled", True):
            limits = RiskLimits.from_cfg(limits_cfg)
            by_tk = {a["ticker"].upper(): a for a in analyses if a.get("ok")}
            decisions, dropped_limits = apply_pre_trade_limits(
                decisions,
                state,
                limits=limits,
                cfg=cfg_run,
                sector_lookup=lambda tk: (by_tk.get(tk.upper(), {}) or {}).get("sector"),
                beta_lookup=lambda tk: (by_tk.get(tk.upper(), {}) or {}).get("beta"),
                vol_lookup=lambda tk: (by_tk.get(tk.upper(), {}) or {}).get("vol_60d_annual"),
            )
            run_stats["risk_limit_drops"] += len(dropped_limits)

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
        gross = _gross_exposure_pct(state, cfg_run)
        run_stats["max_gross_exposure"] = max(run_stats["max_gross_exposure"], gross)
        run_stats["max_positions"] = max(run_stats["max_positions"], len(state.positions))
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
        "profile": cfg.get("profile", "default"),
        "cfd_leverage": cfg.get("cfd_leverage"),
        "long_leverage": cfg.get("long_leverage", cfg.get("cfd_leverage")),
        "short_leverage": cfg.get("short_leverage", cfg.get("cfd_leverage")),
        "config_fingerprint": config_fingerprint(cfg),
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
        "run_stats": run_stats,
    }
    summary.update(universe_summary(universe_source=uni_src, start=start, end=end))

    strat_rets = df["strategy"].astype(float).pct_change().dropna().values
    rf = float(cfg.get("risk_free_rate_annual", 0.04))
    n_trials = cfg.get("threshold_search_trials")
    summary["risk_metrics"] = summarize_backtest(
        strat_rets,
        df["strategy"].astype(float).values,
        periods_per_year=252.0,
        risk_free_rate_annual=rf,
        n_trials_for_dsr=int(n_trials) if n_trials else None,
    )
    summary["regime_attribution"] = attribute_by_regime(df, nav_col="strategy", spy_close=spy_close)
    summary["costs"] = attribute_costs_from_ledger(ledger)

    invariant_errors = validate_backtest_run(
        curve=df,
        ledger=ledger,
        snapshots=snapshots,
        cfg=cfg,
        stats={**run_stats, "strategy_max_dd": mdd_s},
    )
    summary["invariants_ok"] = len(invariant_errors) == 0
    summary["invariant_errors"] = invariant_errors
    if invariant_errors and not skip_invariants:
        print("\n⚠ Invariant violations:")
        for err in invariant_errors:
            print(f"  - {err}")

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

    costs = summary.get("costs") or {}
    print(f"  Modelled costs: overnight {costs.get('overnight_total', 0):.4f} | exit fees {costs.get('exit_cost_total', 0):.4f}")
    ra = summary.get("regime_attribution", {}).get("regimes") or {}
    for label in ("bull", "bear", "unknown"):
        r = ra.get(label) or {}
        if r.get("skipped"):
            continue
        cagr = r.get("cagr")
        if cagr is not None:
            print(f"  Regime {label}: CAGR {cagr:.1%} ({r.get('n_days', 0)} days)")

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
    ap.add_argument(
        "--universe-source",
        choices=("legacy", "pit", "pit_filter"),
        default=None,
        help="legacy=yearly top-100; pit*=filter to S&P members as-of each day",
    )
    ap.add_argument(
        "--frozen-config",
        type=Path,
        default=None,
        help="JSON file with locked thresholds for OOS runs (see config.frozen.example.json)",
    )
    ap.add_argument("--skip-invariants", action="store_true", help="Do not fail loudly on invariant violations")
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
        universe_source=args.universe_source,
        frozen_config_path=args.frozen_config,
        skip_invariants=args.skip_invariants,
    )


if __name__ == "__main__":
    main()
