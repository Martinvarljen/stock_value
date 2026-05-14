#!/usr/bin/env python3
"""
run_vector_backtest.py — CLI for vector_engine (next-open execution, bps costs).

Examples (from Finance repo root):
  python backtesting/run_vector_backtest.py AAPL
  python backtesting/run_vector_backtest.py MSFT --strategy donchian --window 20
  python backtesting/run_vector_backtest.py AAPL --strategy bollinger --years 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.vector_engine import (  # noqa: E402
    run_vector_backtest,
    signal_bollinger_mean_reversion,
    signal_donchian_high_break,
    signal_sma_cross,
)


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0.0)
    l = (-d).clip(lower=0.0)
    ag = g.ewm(alpha=1.0 / period, adjust=False).mean()
    al = l.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ag / (al.replace(0.0, float("nan")) + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Vector backtest (signal @ close, fill next open).")
    ap.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    ap.add_argument("--years", type=float, default=10.0, help="History years (default 10)")
    ap.add_argument(
        "--strategy",
        choices=("sma", "donchian", "bollinger"),
        default="sma",
        help="Built-in signal generator",
    )
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--window", type=int, default=20, help="Donchian / Bollinger window")
    ap.add_argument("--commission-bps", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--json", action="store_true", help="Print metrics as JSON only")
    args = ap.parse_args()

    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=int(365 * args.years + 10))
    hist = yf.Ticker(args.ticker.upper()).history(start=start, end=end, interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < max(args.slow, args.window) + 5:
        print("Insufficient price history.", file=sys.stderr)
        sys.exit(1)
    hist.index = hist.index.tz_localize(None)
    hist = hist.rename(columns=str.lower)
    if "open" not in hist.columns:
        print("No OHLC data.", file=sys.stderr)
        sys.exit(1)

    close = hist["close"]
    high = hist["high"] if "high" in hist.columns else close
    if args.strategy == "sma":
        sig = signal_sma_cross(close, fast=args.fast, slow=args.slow)
    elif args.strategy == "donchian":
        sig = signal_donchian_high_break(high, close, window=args.window)
    else:
        rsi = _rsi_series(close, 14)
        sig = signal_bollinger_mean_reversion(close, window=args.window, rsi=rsi)

    sig = sig.reindex(hist.index).fillna(0.0)
    out = run_vector_backtest(
        hist,
        sig,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )
    if not out.get("ok"):
        print(out.get("error", "failed"), file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(out["metrics"], indent=2))
        return

    m = out["metrics"]
    print(f"\n{args.ticker.upper()}  strategy={args.strategy}  bars={m.get('n_periods', 0)}")
    print(f"  commission_bps={args.commission_bps}  slippage_bps={args.slippage_bps}")
    print(f"  position_changes={out['n_position_changes']}")
    print(f"  total_return={m.get('total_return')}  cagr={m.get('cagr')}  max_dd={m.get('max_drawdown')}")
    print(f"  sharpe={m.get('sharpe')}  sortino={m.get('sortino')}  win_rate={m.get('win_rate')}")
    print(f"  profit_factor={m.get('profit_factor')}  vol_ann={m.get('vol_annual')}\n")


if __name__ == "__main__":
    main()
