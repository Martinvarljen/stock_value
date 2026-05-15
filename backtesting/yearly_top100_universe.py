"""
Yearly 'top 100' universe for historical backtests.

We cannot recover true point-in-time index membership or market cap from Yahoo
alone. This module uses a transparent proxy:

  * Candidate pool: current S&P 500 constituents (Wikipedia table).
  * For calendar year Y: rank candidates by total dollar volume
    sum(Close * Volume) over Y on daily bars from yfinance.
  * Take the top N symbols (default 100).

A checkpoint dated in calendar year C uses the universe built from year C - 1
(no look-ahead into the checkpoint year).

Symbols are normalized for Yahoo (e.g. BRK.B -> BRK-B).
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import pandas as pd
import yfinance as yf


def default_universe_cache_dir(root: Path) -> Path:
    return root / "backtesting" / "universes" / "dollar_volume_top100"


def normalize_yahoo_symbol(sym: str) -> str:
    s = sym.strip().upper()
    return s.replace(".", "-")


def read_ticker_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(normalize_yahoo_symbol(line.split()[0]))
    return out


def write_ticker_lines(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tickers) + "\n", encoding="utf-8")


def fetch_sp500_symbols() -> list[str]:
    """Current S&P 500 tickers from Wikipedia (Symbol column)."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        html = resp.read()
    tables = pd.read_html(html)
    if not tables:
        raise RuntimeError("No tables found on Wikipedia S&P 500 page")
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("Wikipedia S&P 500 table missing Symbol column")
    raw = df["Symbol"].astype(str).tolist()
    return [normalize_yahoo_symbol(s) for s in raw if s and s != "nan"]


def dollar_volume_sum(ticker: str, year: int) -> float | None:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-15"
    try:
        h = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
    except Exception:
        return None
    if h is None or h.empty:
        return None
    if "Close" not in h.columns or "Volume" not in h.columns:
        return None
    h = h[h.index.year == year]
    if h.empty:
        return None
    vol = h["Volume"].fillna(0).astype(float)
    close = h["Close"].astype(float)
    return float((close * vol).sum())


def build_dollar_volume_top_n(
    year: int,
    *,
    top_n: int = 100,
    pool: list[str] | None = None,
    sleep_s: float = 0.05,
    verbose: bool = True,
) -> list[str]:
    """
    Rank ``pool`` (default: current S&P 500) by dollar volume in ``year``;
    return top ``top_n`` tickers descending.
    """
    symbols = list(pool) if pool is not None else fetch_sp500_symbols()
    scores: list[tuple[str, float]] = []
    for i, sym in enumerate(symbols, 1):
        if verbose and (i == 1 or i % 50 == 0 or i == len(symbols)):
            print(f"    [{i}/{len(symbols)}] scoring {sym} …", flush=True)
        dv = dollar_volume_sum(sym, year)
        if dv is not None and dv > 0:
            scores.append((sym, dv))
        if sleep_s:
            time.sleep(sleep_s)
    scores.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scores[:top_n]]


def load_or_build_year_file(
    year: int,
    cache_dir: Path,
    *,
    auto_build: bool,
    top_n: int = 100,
    verbose: bool = True,
) -> list[str]:
    path = cache_dir / f"{year}.txt"
    if path.is_file():
        return read_ticker_lines(path)
    if not auto_build:
        raise FileNotFoundError(
            f"Missing universe file {path}. "
            f"Run: python backtesting/build_yearly_top100_universe.py --from {year} --to {year}"
        )
    if verbose:
        print(f"  Building universe for {year} (this may take a while)…")
    tickers = build_dollar_volume_top_n(year, top_n=top_n, verbose=verbose)
    write_ticker_lines(path, tickers)
    if verbose:
        print(f"  Wrote {len(tickers)} tickers -> {path}")
    return tickers


def load_universe_map_for_lag_years(
    lag_years: list[int],
    cache_dir: Path,
    *,
    auto_build_missing: bool,
    verbose: bool = True,
) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for y in sorted(set(lag_years)):
        out[y] = load_or_build_year_file(y, cache_dir, auto_build=auto_build_missing, verbose=verbose)
    return out
