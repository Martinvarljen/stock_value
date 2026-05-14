"""
strategy_stat_tests.py — Walk-forward splits + permutation-style null for a simple rule strategy.

Inspired by common research practice (e.g. permutation / walk-forward robustness checks).
This is a lightweight educational layer: not a full replacement for dedicated research stacks.

Example (Donchian-style breakout on daily closes):
  python backtesting/strategy_stat_tests.py AAPL
  python backtesting/strategy_stat_tests.py MSFT --window 20 --perm 400
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "stock_analyzer"))


def _annualized_sharpe(daily_rets: np.ndarray, trading_days: int = 252) -> float:
    x = daily_rets[np.isfinite(daily_rets)]
    if len(x) < 10:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd < 1e-12:
        return 0.0
    return (mu / sd) * math.sqrt(trading_days)


def donchian_position(close: pd.Series, window: int) -> pd.Series:
    """1 when long (close breaks prior N-day high), 0 flat — simplified."""
    upper = close.rolling(window).max().shift(1)
    pos = (close > upper).astype(int)
    return pos


def strategy_daily_returns(close: pd.Series, window: int) -> pd.Series:
    pos = donchian_position(close, window)
    ret = close.pct_change()
    return pos.shift(1).fillna(0) * ret  # enter next bar


def walk_forward_segments(n: int, n_folds: int = 4, min_train: int = 120) -> list[tuple[slice, slice]]:
    """Return list of (train_slice, test_slice) index slices on integer positions."""
    if n < min_train + 40:
        return []
    usable = n
    fold_len = max(20, (usable - min_train) // n_folds)
    out: list[tuple[slice, slice]] = []
    start_test = min_train
    while start_test + fold_len <= usable:
        out.append((slice(0, start_test), slice(start_test, start_test + fold_len)))
        start_test += fold_len
    return out


def permutation_sharpe_null(
    close: pd.Series,
    window: int,
    n_perm: int = 500,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """
    Observed Sharpe on real strategy vs Sharpe on bar-shuffled returns (destroys serial correlation).
    Returns (observed_sharpe, p_value_approx, null_samples).
    """
    rng = np.random.default_rng(seed)
    r = strategy_daily_returns(close, window).dropna().values
    obs = _annualized_sharpe(r)

    nulls = []
    base_ret = close.pct_change().dropna().values
    pos_template = donchian_position(close, window).shift(1).fillna(0)
    pos_template = pos_template.reindex(close.index).fillna(0).astype(int)

    for _ in range(n_perm):
        shuffled = rng.permutation(base_ret)
        # align length: use same position mask length as shuffled
        m = min(len(shuffled), len(pos_template) - 1)
        if m < 20:
            continue
        strat = pos_template.iloc[-m:].values * shuffled[-m:]
        nulls.append(_annualized_sharpe(strat))

    nulls_arr = np.array(nulls, dtype=float)
    p = float(np.mean(nulls_arr >= obs)) if len(nulls_arr) else 1.0
    return obs, p, nulls_arr


def run_report(ticker: str, window: int = 20, n_perm: int = 400, years: int = 5) -> None:
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * years + 40)
    hist = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"), interval="1d")
    if hist.empty or len(hist) < 200:
        print(f"No enough data for {ticker}")
        return
    hist.index = hist.index.tz_localize(None)
    close = hist["Close"].astype(float)

    obs, pval, nulls = permutation_sharpe_null(close, window, n_perm=n_perm)
    print(f"\n=== {ticker}  Donchian({window}) daily breakout (simplified) ===")
    print(f"In-sample annualized Sharpe (approx): {obs:.3f}")
    print(f"Permutation p-value (>= observed under shuffled returns): {pval:.3f}")

    segs = walk_forward_segments(len(close), n_folds=4, min_train=180)
    if segs:
        print("\nWalk-forward OOS Sharpe (by segment):")
        for i, (tr, te) in enumerate(segs, 1):
            r_te = strategy_daily_returns(close, window).iloc[te].dropna().values
            print(f"  Fold {i}: OOS Sharpe ~ {_annualized_sharpe(r_te):.3f} (n={len(r_te)})")
    else:
        print("\nWalk-forward: not enough length for folds at this min_train.")

    print("\nNote: Use this as a sanity check layer; combine with your classification pipeline separately.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=["AAPL"], help="Symbols to test")
    ap.add_argument("--window", type=int, default=20, help="Donchian lookback")
    ap.add_argument("--perm", type=int, default=400, help="Permutation draws")
    args = ap.parse_args()
    for t in args.tickers:
        run_report(t.upper(), window=args.window, n_perm=args.perm)


if __name__ == "__main__":
    main()
