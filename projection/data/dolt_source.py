"""
dolt_source.py — Local Dolt SQL or feather cache for ML training.

Requires post-no-preference Dolt databases and `dolt sql-server` on 127.0.0.1:3306,
or a pre-built feather file from setup_dolt_cache.py.

Docs: https://www.dolthub.com/users/post-no-preference
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATHER = Path(__file__).resolve().parent / "cache" / "all_ohlcv_no_etfs.feather"

_DOLT_HOST = os.environ.get("DOLT_HOST", "127.0.0.1")
_DOLT_PORT = int(os.environ.get("DOLT_PORT", "3306"))
_DOLT_USER = os.environ.get("DOLT_USER", "root")
_DOLT_PASSWORD = os.environ.get("DOLT_PASSWORD", "")


def default_feather_path() -> Path:
    env = os.environ.get("FINANCE_DOLT_FEATHER")
    if env:
        return Path(env)
    return DEFAULT_FEATHER


def _connect():
    from mysql import connector as cnc

    return cnc.connect(
        host=_DOLT_HOST,
        port=_DOLT_PORT,
        user=_DOLT_USER,
        password=_DOLT_PASSWORD,
        database=None,
    )


def dolt_available() -> bool:
    """True if MySQL endpoint answers (Dolt sql-server running)."""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchone()
        conn.close()
        return True
    except Exception:
        return False


def normalize_yahoo_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-")


def dolt_to_yahoo_symbol(act_symbol: str) -> str:
    """Map Dolt act_symbol to Yahoo-style ticker when needed."""
    s = str(act_symbol).strip().upper()
    if len(s) >= 4 and s[-3] == "-" and s[-2:].isalpha():
        return s
    return s.replace(".", "-")


def yahoo_to_dolt_symbol(ticker: str) -> str:
    """Best-effort reverse map for feather queries."""
    s = ticker.strip().upper().replace("-", ".")
    return s


def load_ohlcv_feather(path: Path | None = None) -> pd.DataFrame:
    """Load full OHLCV panel; columns: act_symbol, date, open, high, low, close, volume."""
    p = path or default_feather_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"Dolt feather cache not found: {p}\n"
            "Run: python projection/data/setup_dolt_cache.py"
        )
    df = pd.read_feather(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def export_ohlcv_from_dolt(
    dest: Path,
    *,
    exclude_etfs: bool = True,
    verbose: bool = True,
) -> Path:
    """Pull stocks.ohlcv from Dolt and write feather (same layout as StockMarketTool)."""
    conn = _connect()
    if verbose:
        print("Reading stocks.ohlcv from Dolt...", flush=True)
    df = pd.read_sql("SELECT * FROM stocks.ohlcv", conn)
    if exclude_etfs:
        try:
            etf = pd.read_sql(
                "SELECT act_symbol FROM stocks.symbol WHERE is_etf = '1'", conn
            )
            etfs = set(etf["act_symbol"].astype(str))
            before = len(df)
            df = df[~df["act_symbol"].isin(etfs)]
            if verbose:
                print(f"  Excluded {before - len(df):,} ETF rows ({len(etfs)} symbols)", flush=True)
        except Exception as e:
            if verbose:
                print(f"  ETF filter skipped: {e}", flush=True)
    conn.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(dest)
    if verbose:
        print(f"Saved {len(df):,} rows -> {dest}", flush=True)
    return dest


def top_liquidity_tickers(
    df: pd.DataFrame,
    year: int,
    top_n: int = 300,
) -> list[str]:
    """Prior-calendar-year average dollar volume (StockMarketTool-style universe seed)."""
    dcol = pd.to_datetime(df["date"])
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")
    mask = (dcol >= start) & (dcol <= end)
    sub = df.loc[mask, ["act_symbol", "close", "volume"]].copy()
    if sub.empty:
        return []
    sub["dv"] = sub["close"].astype(float) * sub["volume"].astype(float)
    avg = sub.groupby("act_symbol")["dv"].mean().sort_values(ascending=False)
    return [dolt_to_yahoo_symbol(s) for s in avg.head(top_n).index.tolist()]


def _ohlcv_to_yfinance_hist(g: pd.DataFrame) -> pd.DataFrame:
    """Single-ticker daily bars indexed like yfinance history."""
    g = g.sort_values("date").copy()
    g = g.set_index("date")
    out = pd.DataFrame(
        {
            "Open": g["open"].astype(float),
            "High": g["high"].astype(float),
            "Low": g["low"].astype(float),
            "Close": g["close"].astype(float),
            "Volume": g["volume"].astype(float),
        },
        index=g.index,
    )
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out


def ticker_histories_from_feather(
    tickers: list[str],
    start_dt: datetime,
    end_dt: datetime,
    feather_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Return {yahoo_ticker: OHLCV DataFrame} for training.
    Uses one feather read; filters in memory.
    """
    df = load_ohlcv_feather(feather_path)
    t0 = pd.Timestamp(start_dt)
    t1 = pd.Timestamp(end_dt)
    df = df[(df["date"] >= t0) & (df["date"] <= t1)]

    dolt_syms = {yahoo_to_dolt_symbol(t): t for t in tickers}
    df = df[df["act_symbol"].isin(dolt_syms.keys())]

    out: dict[str, pd.DataFrame] = {}
    for act, grp in df.groupby("act_symbol"):
        ysym = dolt_syms.get(act, dolt_to_yahoo_symbol(act))
        hist = _ohlcv_to_yfinance_hist(grp)
        if len(hist) >= 50:
            out[ysym] = hist
    return out
