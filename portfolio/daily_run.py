#!/usr/bin/env python3
"""
Daily trading agent — stateless run, persistent files only.

Each invocation is independent: reads portfolio/data/state.json and config,
scans the market, writes decisions, applies paper trades, logs ledger entries,
and leaves notes for the next run.

  python portfolio/daily_run.py
  python portfolio/daily_run.py AAPL MSFT NVDA
  python portfolio/daily_run.py --max-tickers 30 --news
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.analyze import analyze_ticker, market_regime
from portfolio.broker import apply_decisions
from portfolio.data_gates import filter_for_bad_ohlcv
from portfolio.decision_schema import DecisionReport
from portfolio.decisions import Action, decide_universe, prioritize_entries
from portfolio.memory_log import DecisionMemoryLog
from portfolio.reflection import OutcomeContext, reflect_on_outcome
from portfolio.store import (
    DATA_DIR,
    ensure_data_dirs,
    load_config,
    load_state,
    read_daily_notes,
    save_state,
    write_daily_notes,
    write_snapshot,
)
from portfolio.universe import resolve_tickers

import pandas as pd
import yfinance as yf


_DEFAULT_MEMORY_LOG = DATA_DIR / "decision_memory.md"


def _memory_log_for(cfg: dict) -> DecisionMemoryLog:
    path = cfg.get("memory_log_path") or _DEFAULT_MEMORY_LOG
    cap = cfg.get("memory_log_max_entries")
    return DecisionMemoryLog(path, max_entries=cap)


def _fetch_window(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Daily closes between ``start`` and ``end`` (inclusive). None on failure."""
    try:
        df = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception:
        return None


def _resolve_pending_entries(
    memory: DecisionMemoryLog,
    *,
    run_date: date,
    benchmark: str = "SPY",
    min_holding_days: int = 5,
) -> int:
    """Resolve any pending memory-log entries whose horizon has elapsed.

    For each pending entry, fetch ``ticker`` and ``benchmark`` closes from
    the trade date through today, compute raw and alpha returns, build a
    deterministic reflection, then atomically rewrite the log. Entries
    that don't yet have enough trading days simply stay pending.

    Returns the number of entries resolved.
    """
    pending = memory.get_pending_entries()
    if not pending:
        return 0

    # Group by trade_date so we fetch SPY only once per date.
    end = (run_date + timedelta(days=1)).isoformat()
    spy_cache: dict[str, pd.DataFrame] = {}

    updates: list[dict] = []
    for e in pending:
        try:
            entry_dt = date.fromisoformat(e.date)
        except ValueError:
            continue
        if (run_date - entry_dt).days < min_holding_days:
            continue

        start = entry_dt.isoformat()
        stock = _fetch_window(e.ticker, start, end)
        if stock is None or len(stock) < 2:
            continue

        bench = spy_cache.get(start)
        if bench is None:
            bench = _fetch_window(benchmark, start, end)
            if bench is not None:
                spy_cache[start] = bench
        if bench is None or len(bench) < 2:
            continue

        try:
            p0 = float(stock["Close"].iloc[0])
            p1 = float(stock["Close"].iloc[-1])
            b0 = float(bench["Close"].iloc[0])
            b1 = float(bench["Close"].iloc[-1])
        except (KeyError, IndexError, TypeError):
            continue
        if p0 <= 0 or b0 <= 0:
            continue

        raw = (p1 - p0) / p0
        bench_ret = (b1 - b0) / b0
        alpha = raw - bench_ret
        holding_days = (run_date - entry_dt).days

        ctx = OutcomeContext(
            ticker=e.ticker,
            trade_date=e.date,
            rating=e.rating,
            action=e.rating,  # action no longer parseable from tag, fall back to rating
            raw_return=raw,
            alpha_return=alpha,
            holding_days=holding_days,
            benchmark=benchmark,
        )
        updates.append({
            "ticker": e.ticker,
            "trade_date": e.date,
            "raw_return": raw,
            "alpha_return": alpha,
            "holding_days": holding_days,
            "reflection": reflect_on_outcome(ctx),
        })

    if not updates:
        return 0
    return memory.batch_update_with_outcomes(updates)


def _build_notes_for_tomorrow(
    run_date: date,
    regime: dict,
    decisions: list,
    ledger_rows: list,
    state,
    cfg: dict,
) -> dict:
    tomorrow = run_date + timedelta(days=1)
    actions = [
        {"ticker": d.ticker, "action": d.action.value, "reason": d.reason}
        for d in decisions
        if d.action != Action.NO_TRADE
    ]
    entries = [d for d in decisions if d.action in (Action.ENTER_LONG, Action.ENTER_SHORT)]
    exits = [d for d in decisions if d.action == Action.EXIT]
    holds = [d for d in decisions if d.action == Action.HOLD]

    watch = sorted(
        [d for d in decisions if d.quintile and d.quintile >= 4 and d.action == Action.NO_TRADE],
        key=lambda x: x.ml_score or 0,
        reverse=True,
    )[:10]

    bullets = [
        f"Regime: {'risk-on' if regime.get('spy_bull') else 'risk-off'} (gross scale {regime.get('gross_exposure_scale', 1):.0%}).",
        f"NAV {state.nav:.4f} | cash {state.cash:.4f} | {len(state.positions)} open positions.",
        f"Today: {len(entries)} entries, {len(exits)} exits, {len(holds)} holds.",
    ]
    if entries:
        bullets.append("Entered: " + ", ".join(f"{d.ticker} ({d.action.value})" for d in entries))
    if exits:
        bullets.append("Exited: " + ", ".join(d.ticker for d in exits))

    open_detail = []
    est = int(cfg.get("estimated_hold_days", 20))
    for p in state.positions:
        rem = p.estimated_days_remaining(est, run_date)
        open_detail.append(
            {
                "ticker": p.ticker,
                "side": p.side,
                "entry_date": p.entry_date,
                "entry_price": p.entry_price,
                "est_days_remaining": rem,
            }
        )

    return {
        "for_date": tomorrow.isoformat(),
        "written_on": run_date.isoformat(),
        "regime": regime,
        "summary_bullets": bullets,
        "watchlist": [{"ticker": d.ticker, "ml_score": d.ml_score, "reason": d.reason} for d in watch],
        "actions_today": actions,
        "open_positions": open_detail,
        "ledger_count": len(ledger_rows),
    }


def run_daily(
    tickers: list[str],
    *,
    max_tickers: int | None,
    universe: str,
    include_news: bool | None,
) -> None:
    ensure_data_dirs()
    cfg = load_config()
    state = load_state(cfg)
    run_date = date.today()

    yesterday_notes = read_daily_notes(run_date)
    if yesterday_notes:
        print(f"Notes for today (from prior run):\n")
        for b in yesterday_notes.get("summary_bullets", [])[:8]:
            print(f"  • {b}")
        print()

    memory = _memory_log_for(cfg)
    min_hold = int(cfg.get("memory_min_holding_days", cfg.get("estimated_hold_days", 5)))
    n_resolved = _resolve_pending_entries(memory, run_date=run_date, min_holding_days=min_hold)
    if n_resolved:
        print(f"Memory: resolved {n_resolved} pending decision(s) with realised outcomes.\n")

    regime = market_regime(datetime.combine(run_date, datetime.min.time()))
    print(f"=== Daily agent {run_date.isoformat()} ===")
    print(f"Regime: SPY {'above' if regime['spy_bull'] else 'below'} 200d MA → scale {regime['gross_exposure_scale']:.0%}\n")

    symbols = resolve_tickers(explicit=tickers or None, universe=universe, max_tickers=max_tickers)
    if not symbols:
        print("No tickers to scan.")
        return

    news = include_news if include_news is not None else bool(cfg.get("include_news", False))
    mos = float(cfg.get("margin_of_safety", 0.3))
    news_days = int(cfg.get("news_days", 3))

    analyses = []
    for i, tk in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {tk} …", flush=True)
        a = analyze_ticker(tk, margin_of_safety=mos, include_news=news, news_days=news_days)
        if a:
            analyses.append(a)

    analyses, dropped_ohlcv = filter_for_bad_ohlcv(analyses)
    if dropped_ohlcv:
        print(
            f"OHLCV gate: dropped {len(dropped_ohlcv)} ticker(s) with broken bars "
            f"({', '.join(tk for tk, _ in dropped_ohlcv[:8])}"
            f"{'…' if len(dropped_ohlcv) > 8 else ''})."
        )

    cfg_run = {**cfg, "_regime_scale": regime["gross_exposure_scale"]}
    decisions = decide_universe(analyses, state, regime, cfg_run, run_date)
    decisions = prioritize_entries(decisions, cfg_run)

    # Pre-trade portfolio-level risk limits (gross / sector / beta /
    # VaR). Decisions that would breach a cap get downgraded to
    # NO_TRADE here with a structured reason so the memory log can
    # surface "we wanted to but the limits said no".
    from portfolio.risk_limits import RiskLimits, apply_pre_trade_limits
    _limits_cfg = cfg_run.get("risk_limits") or {}
    if _limits_cfg.get("enabled", True):
        limits = RiskLimits.from_cfg(_limits_cfg)
        analyses_by_tk = {a["ticker"].upper(): a for a in analyses if a.get("ok")}
        decisions, dropped_by_limits = apply_pre_trade_limits(
            decisions, state, limits=limits, cfg=cfg_run,
            sector_lookup=lambda tk: (analyses_by_tk.get(tk.upper(), {}) or {}).get("sector"),
            beta_lookup=lambda tk: (analyses_by_tk.get(tk.upper(), {}) or {}).get("beta"),
            vol_lookup=lambda tk: (analyses_by_tk.get(tk.upper(), {}) or {}).get("vol_60d_annual"),
        )
        if dropped_by_limits:
            print(f"Pre-trade risk limits dropped {len(dropped_by_limits)} entrie(s):")
            for d in dropped_by_limits[:5]:
                print(f"  - {d['ticker']}: {d['reason']}")

    ledger_rows = apply_decisions(state, decisions, run_date=run_date, cfg=cfg_run)
    save_state(state)

    analyses_by_ticker = {a["ticker"].upper(): a for a in analyses if a.get("ok")}

    n_stored = 0
    for d in decisions:
        if d.action == Action.NO_TRADE:
            continue
        had_pos = bool(state.position_for(d.ticker))
        past_ctx = memory.get_past_context(d.ticker)
        analysis = analyses_by_ticker.get(d.ticker.upper(), {})
        extras: dict = {}
        explanation = analysis.get("explanation_one_liner")
        if explanation:
            extras["explanation"] = explanation
        if analysis.get("classification"):
            extras["classification"] = analysis["classification"]
        if analysis.get("projection_signal"):
            extras["projection_signal"] = analysis["projection_signal"]

        setup = analysis.get("trade_setup") or {}
        if setup.get("available"):
            bias = setup.get("bias_summary")
            if bias:
                extras["setup_bias"] = bias
            levels = [
                lv for lv in (setup.get("watch_levels") or [])
                if isinstance(lv.get("price"), (int, float))
            ][:4]
            if levels:
                extras["watch_levels"] = ", ".join(
                    f"{lv['name']}={lv['price']:.2f}" for lv in levels
                )

        report = DecisionReport.from_decision(
            d,
            trade_date=run_date,
            regime=regime,
            had_position=had_pos,
            past_context=past_ctx,
            extras=extras,
        )
        if memory.store_decision(report):
            n_stored += 1
    if n_stored:
        print(f"Memory: stored {n_stored} new decision(s) as pending → {memory.path}")

    notes = _build_notes_for_tomorrow(run_date, regime, decisions, ledger_rows, state, cfg)
    notes_path = write_daily_notes(run_date + timedelta(days=1), notes)

    snap = {
        "run_date": run_date.isoformat(),
        "regime": regime,
        "analyses": analyses,
        "decisions": [
            {
                "ticker": d.ticker,
                "action": d.action.value,
                "reason": d.reason,
                "ml_score": d.ml_score,
                "quintile": d.quintile,
                "p_up_20d": d.p_up_20d,
            }
            for d in decisions
        ],
        "ledger": ledger_rows,
        "state_after": {"nav": state.nav, "cash": state.cash, "positions": len(state.positions)},
    }
    snap_path = write_snapshot(run_date, snap)

    print("\n--- Decisions ---")
    for d in sorted(decisions, key=lambda x: (x.action.value, x.ticker)):
        if d.action == Action.NO_TRADE:
            continue
        sc = f"{d.ml_score:.2f}" if d.ml_score is not None else "—"
        print(f"  {d.ticker:6} {d.action.value:12} Q{d.quintile or '-'} score={sc}  {d.reason}")

    print(f"\nNAV {state.nav:.4f} | cash {state.cash:.4f} | positions {len(state.positions)}")
    print(f"Notes for tomorrow → {notes_path}")
    print(f"Snapshot → {snap_path}")
    print("\nWeekly report: python reporting/weekly_report.py")


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily paper-trading agent (stateless run, file memory).")
    ap.add_argument("tickers", nargs="*", help="Explicit symbols (overrides universe)")
    ap.add_argument("--max-tickers", type=int, default=50, help="Cap universe scan (default 50)")
    ap.add_argument("--universe", default=None, help="top100 (default from config)")
    ap.add_argument("--news", action="store_true", help="Include FinBERT/Claude news pass")
    ap.add_argument("--no-news", action="store_true", help="Skip news even if config enables it")
    args = ap.parse_args()

    cfg = load_config()
    universe = args.universe or cfg.get("universe", "top100")
    include_news = None
    if args.news:
        include_news = True
    elif args.no_news:
        include_news = False

    max_t = args.max_tickers if args.max_tickers > 0 else None
    run_daily(
        list(args.tickers),
        max_tickers=max_t,
        universe=universe,
        include_news=include_news,
    )


if __name__ == "__main__":
    main()
