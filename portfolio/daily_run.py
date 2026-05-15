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
from portfolio.decisions import Action, decide_universe, prioritize_entries
from portfolio.store import (
    ensure_data_dirs,
    load_config,
    load_state,
    read_daily_notes,
    save_state,
    write_daily_notes,
    write_snapshot,
)
from portfolio.universe import resolve_tickers


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

    cfg_run = {**cfg, "_regime_scale": regime["gross_exposure_scale"]}
    decisions = decide_universe(analyses, state, regime, cfg_run, run_date)
    decisions = prioritize_entries(decisions, cfg_run)

    ledger_rows = apply_decisions(state, decisions, run_date=run_date, cfg=cfg_run)
    save_state(state)

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
    print("\nWeekly report: python portfolio/weekly_report.py")


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
