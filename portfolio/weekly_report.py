#!/usr/bin/env python3
"""
Weekly paper-trading report from trade ledger + open positions.

  python portfolio/weekly_report.py
  python portfolio/weekly_report.py --days 7
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yfinance as yf

from portfolio.store import (
    ensure_data_dirs,
    load_config,
    load_state,
    read_ledger,
    week_id,
    weekly_report_path,
)


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    out: dict[str, float] = {}
    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
        )
        if data is None or data.empty:
            return out
        if len(tickers) == 1:
            tk = tickers[0]
            if "Close" in data.columns:
                out[tk] = float(data["Close"].dropna().iloc[-1])
            return out
        for tk in tickers:
            try:
                sub = data[tk]
                out[tk] = float(sub["Close"].dropna().iloc[-1])
            except (KeyError, TypeError, IndexError):
                continue
    except Exception:
        pass
    return out


def build_weekly_markdown(days: int = 7) -> str:
    ensure_data_dirs()
    cfg = load_config()
    state = load_state(cfg)
    today = date.today()
    since = today - timedelta(days=days)
    ledger = read_ledger(since=since)
    est_hold = int(cfg.get("estimated_hold_days", 20))

    week_trades = [r for r in ledger if r.get("action") in ("ENTER_LONG", "ENTER_SHORT", "EXIT")]
    realized = [r for r in ledger if r.get("action") == "EXIT" and r.get("pnl_pct") is not None]
    realized_pnl = sum(float(r["notional"]) * float(r["pnl_pct"]) for r in realized)

    open_tickers = [p.ticker for p in state.positions]
    prices = _fetch_prices(open_tickers)

    open_rows: list[str] = []
    unrealized = 0.0
    for p in state.positions:
        px = prices.get(p.ticker, p.entry_price)
        if p.side == "long":
            pnl_pct = (px - p.entry_price) / p.entry_price
        else:
            pnl_pct = (p.entry_price - px) / p.entry_price
        pnl_dollar = p.notional * pnl_pct
        unrealized += pnl_dollar
        rem = p.estimated_days_remaining(est_hold, today)
        open_rows.append(
            f"| {p.ticker} | {p.side} | {p.entry_date} | {p.entry_price:.2f} | {px:.2f} | "
            f"{pnl_pct:+.1%} | ~{rem}d | {p.entry_reason[:40]} |"
        )

    trade_lines = []
    for r in week_trades[-50:]:
        trade_lines.append(
            f"| {r.get('date', '')} | {r.get('ticker', '')} | {r.get('action', '')} | "
            f"{r.get('side', '')} | {r.get('price', '')} | {r.get('notional', '')} | "
            f"{r.get('pnl_pct', '')} | {r.get('reason', '')[:50]} |"
        )

    nav = state.nav
    net_week = realized_pnl + unrealized

    md = f"""# Weekly paper report — {week_id(today)}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary

| Metric | Value |
|--------|-------|
| Period | last {days} days ({since} → {today}) |
| NAV (mark-to-market) | **{nav:.4f}** |
| Cash | {state.cash:.4f} |
| Open positions | {len(state.positions)} |
| Realized P&L (closed this week) | {realized_pnl:+.4f} |
| Unrealized P&L (open) | {unrealized:+.4f} |
| Net week (realized + unrealized MTM) | **{net_week:+.4f}** |

## Trades this week

| Date | Ticker | Action | Side | Price | Notional | PnL % | Reason |
|------|--------|--------|------|-------|----------|-------|--------|
"""
    md += "\n".join(trade_lines) if trade_lines else "| — | — | — | — | — | — | — | — |"

    md += """

## Open positions

| Ticker | Side | Entry | Entry $ | Last $ | P&L % | Est. hold left | Entry reason |
|--------|------|-------|---------|--------|-------|----------------|--------------|
"""
    md += "\n".join(open_rows) if open_rows else "| — | — | — | — | — | — | — | — |"

    md += """

---
*Each daily run is independent; this report reads `portfolio/data/trade_ledger.jsonl` and `state.json` only.*
"""
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--print-only", action="store_true", help="Do not write file")
    args = ap.parse_args()

    md = build_weekly_markdown(args.days)
    print(md)
    if not args.print_only:
        ensure_data_dirs()
        path = weekly_report_path()
        path.write_text(md, encoding="utf-8")
        print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
