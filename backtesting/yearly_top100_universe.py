"""
Yearly 'top 100' universe for historical backtests.

⚠️ SURVIVORSHIP-BIAS WARNING — read before trusting any backtest run on
top of this universe.

We cannot recover true point-in-time index membership or market cap from
Yahoo alone. This module uses a transparent proxy:

  * Candidate pool: **current** S&P 500 constituents (Wikipedia table).
  * For calendar year Y: rank candidates by total dollar volume
    sum(Close * Volume) over Y on daily bars from yfinance.
  * Take the top N symbols (default 100).

What this is NOT
----------------
This is **not** a survivorship-bias-free universe. The candidate pool is
the S&P 500 *as it stands today*. Names that were in the index during the
backtest window but were later delisted, acquired, or downgraded out of
the index (Lehman, Bear Stearns, Sears, Wachovia, GE pre-2018, …) are
silently absent from the candidate pool. The reverse problem also exists:
recent additions (Tesla, Snowflake) appear in the pool earlier than they
should.

Empirically, surviving-only US large-cap universes inflate Sharpe by
roughly **0.2-0.4** vs PIT-correct universes (Brown, Goetzmann, Ross
1992; Elton, Gruber, Blake 1996). Treat any Sharpe produced from this
module as **upper-bound, not realised**.

Tier-1 fix (now shipped — use it)
---------------------------------
Use ``backtesting.sp500_pit_universe`` instead of this module. It builds
a per-year membership set by walking today's pool *backwards* through a
curated S&P 500 change-log (``backtesting/sp500_changes.csv``), so
delisted names (Lehman, Bear, GE, SVB, FRC, …) appear in the candidate
pool for the years they were actually in the index, and post-window
additions (TSLA before Dec 2020) don't.

The seed change-log covers the largest 30+ changes 2008-2025; extend it
from Wikipedia's full revision history or by purchasing CRSP / Norgate
for fully-authoritative coverage.

Drop-in usage::

    from backtesting.yearly_top100_universe import build_pit_universe_map
    from backtesting.dynamic_portfolio_backtest import run_dynamic

    universe_map = build_pit_universe_map(
        years=range(2018, 2025),   # one entry per backtest year
        top_n=100,
    )
    run_dynamic(..., universe_map=universe_map)

The ``--pit-universe`` flag on the dynamic backtest CLI does the same
thing without code (see ``run_dynamic`` ``__main__`` block).

Symbols are normalized for Yahoo (e.g. BRK.B -> BRK-B).
"""

from __future__ import annotations

import time
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf


# Emitted once per process so users see it during a backtest run rather
# than only when they read the docstring.
_SURVIVORSHIP_WARNING = (
    "Universe built from current S&P 500 only — survivorship bias inflates "
    "backtest Sharpe by an estimated 0.2-0.4. Use the PIT path instead: "
    "pass `pit_universe=True` to run_dynamic, or call build_pit_universe_map("
    "years=...) directly. See module docstring for details."
)
_SURVIVORSHIP_WARNING_EMITTED = False


def _warn_survivorship_once() -> None:
    global _SURVIVORSHIP_WARNING_EMITTED
    if _SURVIVORSHIP_WARNING_EMITTED:
        return
    warnings.warn(_SURVIVORSHIP_WARNING, stacklevel=2)
    _SURVIVORSHIP_WARNING_EMITTED = True


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

    Emits a one-time survivorship-bias warning the first time the default
    Wikipedia pool is used. Pass an explicit ``pool`` to suppress (e.g. a
    point-in-time CRSP snapshot).
    """
    if pool is None:
        _warn_survivorship_once()
        symbols = fetch_sp500_symbols()
    else:
        symbols = list(pool)
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


def build_pit_universe_map(
    years,
    *,
    top_n: int = 100,
    cache_dir: Path | None = None,
    sleep_s: float = 0.05,
    verbose: bool = True,
) -> dict[int, list[str]]:
    """Drop-in PIT replacement for ``load_universe_map_for_lag_years``.

    Builds ``{year: top_n_tickers}`` using the survivorship-free
    membership set from ``backtesting.sp500_pit_universe``. Files are
    cached on disk (one per year) so subsequent runs are instant.

    Pair with ``run_dynamic(..., universe_map=...)``::

        universe_map = build_pit_universe_map(years=range(2018, 2025))
        run_dynamic(..., universe_map=universe_map)

    The PIT module emits a one-time coverage warning when the seed
    change-log doesn't cover the requested year range; expand
    ``backtesting/sp500_changes.csv`` to silence it.
    """
    from backtesting.sp500_pit_universe import pit_top_n

    if cache_dir is None:
        cache_dir = default_universe_cache_dir(
            Path(__file__).resolve().parents[1]
        ).parent / "dollar_volume_top100_pit"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[int, list[str]] = {}
    for y in sorted(set(int(yr) for yr in years)):
        path = cache_dir / f"{y}.txt"
        if path.is_file():
            out[y] = read_ticker_lines(path)
            if verbose:
                print(f"  PIT universe {y}: cached ({len(out[y])} tickers)")
            continue
        if verbose:
            print(f"  PIT universe {y}: building (one-time, may take a while)")
        tickers = pit_top_n(y, top_n=top_n, sleep_s=sleep_s, verbose=verbose)
        write_ticker_lines(path, tickers)
        out[y] = tickers
        if verbose:
            print(f"  Wrote {len(tickers)} tickers -> {path}")
    return out
