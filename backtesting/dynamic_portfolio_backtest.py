#!/usr/bin/env python3
"""
Dynamic paper portfolio vs SPY buy-and-hold.

Rules (MVP, transparent defaults):
  1) Fundamentals: quarterly checkpoints reuse the same pipeline as strategy_backtest
     (reconstruct_data_at + classify_at). Only STRONG BUY / BUY are eligible for entries.
     Use ``--signal-tech-ai`` for BUY tiers from momentum/technicals + projection_engine
     (no DCF in composite; ML when a saved model exists). Use ``--no-valuation`` for DCF-off
     ``classification_engine`` only (same CLI as strategy_backtest.py).
  2) Technical entry: first day on or after the checkpoint where Close breaks above the
     prior N-day high (Donchian-style breakout; default N=20).
  3) Exit: stop-loss 5% below entry close; take-profit 25% above entry ("price right" proxy);
     or max hold 130 calendar days. Stops take precedence using that day's Low/High.
  4) Sizing: each new position uses ``position_frac`` of current NAV (default 10%), capped by
     cash and ``max_positions`` concurrent names.
  5) Benchmark: buy-and-hold SPY normalized to $1 on the first simulation day (never sold).

Output: standalone Plotly HTML with a slider/Play animation of monthly strategy vs SPY.

Requires yearly top-100 cache files (same as strategy_backtest). Example:

    python backtesting/build_yearly_top100_universe.py --for-checkpoints-from-year 2023
    python backtesting/dynamic_portfolio_backtest.py
    python backtesting/dynamic_portfolio_backtest.py --signal-tech-ai
    # (defaults: checkpoints from 2023 onward vs SPY buy-and-hold; use --checkpoint-min-year 0 for no year filter)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "stock_analyzer"))

import backtesting.strategy_backtest as st
from backtesting.yearly_top100_universe import default_universe_cache_dir, load_universe_map_for_lag_years


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    stop: float
    take_profit: float
    entry_day: datetime
    peak: float


def _norm_day(ts) -> datetime:
    d = pd.Timestamp(ts).to_pydatetime()
    if d.tzinfo is not None:
        d = d.replace(tzinfo=None)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _prep_price_df(hist: pd.DataFrame) -> pd.DataFrame | None:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    df = hist[["Open", "High", "Low", "Close"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()
    return df


def _last_class_on_or_before(rows: list[tuple[datetime, str | None]], d: datetime) -> str | None:
    best: str | None = None
    for cp, cl in rows:
        if cp <= d:
            best = cl
        else:
            break
    return best


def _row_asof(df: pd.DataFrame, d: datetime) -> pd.Series | None:
    ts = pd.Timestamp(_norm_day(d))
    idx = df.index
    if len(idx) == 0:
        return None
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]


def _breakout_today(df: pd.DataFrame, d: datetime, n: int) -> bool:
    row_today = _row_asof(df, d)
    if row_today is None:
        return False
    ts = pd.Timestamp(_norm_day(d))
    pos = int(df.index.searchsorted(ts, side="right") - 1)
    if pos < n + 1:
        return False
    window = df.iloc[pos - n : pos]
    if window.empty:
        return False
    prior_high = float(window["High"].max())
    c = float(row_today["Close"])
    return c > prior_high and prior_high > 0


def _write_monthly_animation_html(df: pd.DataFrame, out: Path, title: str) -> None:
    import plotly.express as px

    if df.empty:
        return
    rows: list[dict] = []
    for i in range(len(df)):
        sub = df.iloc[: i + 1]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "month": r["month"],
                    "nav": r["strategy"],
                    "series": "Strategy",
                    "frame": i,
                }
            )
            rows.append(
                {
                    "month": r["month"],
                    "nav": r["spy_bh"],
                    "series": "SPY buy & hold",
                    "frame": i,
                }
            )
    pdf = pd.DataFrame(rows)
    y_max = float(max(pdf["nav"].max(), 1.05))
    fig = px.line(
        pdf,
        x="month",
        y="nav",
        color="series",
        animation_frame="frame",
        range_y=[0, y_max],
        title=title,
        template="plotly_dark",
        markers=True,
    )
    fig.update_layout(height=700, legend_title_text="")
    fig.update_yaxes(title="Growth of $1")
    fig.update_xaxes(title="Month")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", config={"displayModeBar": True})


def run_dynamic(
    *,
    lookback_years: int,
    checkpoint_min_year: int | None,
    universe_dir: Path | None,
    auto_build_universe: bool,
    breakout_days: int,
    stop_loss: float,
    take_profit: float,
    max_hold_days: int,
    position_frac: float,
    max_positions: int,
    max_tickers: int | None,
    out_html: Path | None,
    tickers: list[str] | None = None,
    universe_map: dict[int, list[str]] | None = None,
    use_valuation: bool = True,
    signal_mode: str = "dcf",
    checkpoint_freq: str | None = None,
) -> Path:
    look = lookback_years
    if checkpoint_min_year is not None:
        y_now = datetime.today().year
        look = max(look, y_now - checkpoint_min_year + 2)

    cf = st._normalize_checkpoint_freq(checkpoint_freq)
    checkpoints = st._generate_checkpoints(look, cf)
    if checkpoint_min_year is not None:
        checkpoints = [c for c in checkpoints if c.year >= checkpoint_min_year]
    if not checkpoints:
        raise RuntimeError("No checkpoints after filters.")

    lag_years = sorted({c.year - 1 for c in checkpoints})
    if tickers is None:
        udir = universe_dir or default_universe_cache_dir(_ROOT)
        try:
            uni = load_universe_map_for_lag_years(
                lag_years, udir, auto_build_missing=auto_build_universe, verbose=True
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{e}\nBuild cache: python backtesting/build_yearly_top100_universe.py --from ... --to ..."
            ) from e
        tickers_list = sorted({t for y in lag_years for t in uni.get(y, [])})
        uni_for_cp = uni
    else:
        tickers_list = sorted(set(tickers))
        uni_for_cp = universe_map if universe_map is not None else {}

    if max_tickers is not None and max_tickers > 0:
        tickers_list = tickers_list[:max_tickers]
    if not tickers_list:
        raise RuntimeError("No tickers to simulate.")

    spy = yf.Ticker("SPY").history(period="max", interval="1d")
    spy_df = _prep_price_df(spy)
    if spy_df is None:
        raise RuntimeError("Could not load SPY history.")

    first_d = max(_norm_day(checkpoints[0]), _norm_day(spy_df.index[0]))
    last_d = min(_norm_day(checkpoints[-1]) + timedelta(days=400), _norm_day(spy_df.index[-1]))
    spy_df = spy_df.loc[(spy_df.index >= pd.Timestamp(first_d)) & (spy_df.index <= pd.Timestamp(last_d))]

    print(
        f"\nDynamic portfolio - {len(tickers_list)} tickers, {len(checkpoints)} checkpoints "
        f"({'month-end' if cf == 'M' else 'quarter-end'}) {checkpoints[0].date()} -> {checkpoints[-1].date()}"
    )
    print(
        f"Rules: BUY tiers only, {breakout_days}d high breakout, "
        f"stop {stop_loss:.0%}, TP {take_profit:.0%}, max hold {max_hold_days}d, "
        f"{position_frac:.0%} NAV/trade, max {max_positions} positions."
        + ("" if use_valuation else " Valuation engine OFF.")
        + "\n"
    )

    raw_by: dict[str, dict] = {}
    classes: dict[str, list[tuple[datetime, str | None]]] = {}

    for i, tk in enumerate(tickers_list, 1):
        print(f"  [{i}/{len(tickers_list)}] fundamentals {tk} …", flush=True)
        try:
            raw = st.collect_raw_yfinance(tk)
        except Exception as e:
            print(f"    skip ({e})")
            continue
        if raw.get("income_statement") is None:
            print("    skip (no financials)")
            continue
        px = _prep_price_df(raw.get("price_history"))
        if px is None:
            print("    skip (no prices)")
            continue
        qrows: list[tuple[datetime, str | None]] = []
        for cp in checkpoints:
            ly = cp.year - 1
            if uni_for_cp:
                allowed = set(uni_for_cp.get(ly, []))
                if allowed and tk not in allowed:
                    continue
            data = st.reconstruct_data_at(raw, cp)
            if data is None:
                continue
            cl = st.classify_at(
                data,
                raw,
                cp,
                use_valuation=use_valuation,
                signal_mode=signal_mode,
            )
            qrows.append((_norm_day(cp), cl))
        if not qrows:
            continue
        qrows.sort(key=lambda x: x[0])
        raw_by[tk] = raw
        classes[tk] = qrows

    trading_days = sorted({_norm_day(x) for x in spy_df.index})
    trading_days = [d for d in trading_days if d >= first_d and d <= last_d]
    spy0_row = _row_asof(spy_df, trading_days[0])
    if spy0_row is None:
        raise RuntimeError("SPY has no price on first trading day.")
    spy0 = float(spy0_row["Close"])

    cash = 1.0
    positions: list[Position] = []
    daily_rows: list[tuple[datetime, float, float]] = []

    for d in trading_days:
        spy_row = _row_asof(spy_df, d)
        if spy_row is None:
            continue
        spy_close = float(spy_row["Close"])

        # --- exits first ---
        still: list[Position] = []
        for p in positions:
            df = _prep_price_df(raw_by[p.ticker]["price_history"])
            if df is None:
                still.append(p)
                continue
            row = _row_asof(df, d)
            if row is None:
                still.append(p)
                continue
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            p.peak = max(p.peak, cl)
            exit_px = None
            if lo <= p.stop:
                exit_px = p.stop
            elif hi >= p.take_profit:
                exit_px = p.take_profit
            elif (d - p.entry_day).days >= max_hold_days:
                exit_px = cl
            if exit_px is not None:
                cash += p.shares * exit_px
            else:
                still.append(p)
        positions = still

        # --- NAV for sizing (cash + open positions MTM) ---
        nav_pre = cash
        for p in positions:
            df = _prep_price_df(raw_by[p.ticker]["price_history"])
            if df is None:
                continue
            row = _row_asof(df, d)
            if row is None:
                continue
            nav_pre += p.shares * float(row["Close"])

        # --- new entries ---
        if len(positions) < max_positions and cash > 1e-6:
            for tk in sorted(raw_by.keys()):
                if len(positions) >= max_positions:
                    break
                if any(x.ticker == tk for x in positions):
                    continue
                cls = _last_class_on_or_before(classes[tk], d)
                if cls not in st.BUY_TIERS:
                    continue
                df = _prep_price_df(raw_by[tk]["price_history"])
                if df is None:
                    continue
                if not _breakout_today(df, d, breakout_days):
                    continue
                row = _row_asof(df, d)
                if row is None:
                    continue
                entry = float(row["Close"])
                if entry <= 0:
                    continue
                notional = min(position_frac * nav_pre, cash * 0.999)
                if notional < 1e-4:
                    continue
                shares = notional / entry
                cash -= notional
                positions.append(
                    Position(
                        ticker=tk,
                        shares=shares,
                        entry_price=entry,
                        stop=entry * (1.0 - stop_loss),
                        take_profit=entry * (1.0 + take_profit),
                        entry_day=d,
                        peak=entry,
                    )
                )

        nav_end = cash
        for p in positions:
            df = _prep_price_df(raw_by[p.ticker]["price_history"])
            if df is None:
                continue
            row = _row_asof(df, d)
            if row is None:
                continue
            nav_end += p.shares * float(row["Close"])

        spy_bh = spy_close / spy0 if spy0 > 0 else float("nan")
        daily_rows.append((d, nav_end, spy_bh))

    comb = (
        pd.DataFrame(daily_rows, columns=["date", "strategy", "spy_bh"])
        .drop_duplicates(subset=["date"])
        .set_index("date")
        .sort_index()
    )
    n0 = float(comb["strategy"].iloc[0]) if len(comb) else 1.0
    if n0 > 0:
        comb["strategy"] = comb["strategy"] / n0

    monthly = comb.resample("ME").last().dropna(how="all")
    monthly = monthly.reset_index().rename(columns={"date": "month"})
    monthly["month"] = pd.to_datetime(monthly["month"])

    out = out_html or (_ROOT / f"dynamic_vs_spy_{datetime.today().strftime('%Y%m%d')}.html")
    _write_monthly_animation_html(
        monthly,
        Path(out),
        "Dynamic strategy (fundamental + breakout) vs SPY buy-and-hold — monthly (animated)",
    )
    return Path(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback", type=int, default=st.DEFAULT_LOOKBACK)
    ap.add_argument(
        "--checkpoint-min-year",
        type=int,
        default=2023,
        help="Only simulate checkpoints in this calendar year or later (default: 2023). Use 0 for no filter.",
    )
    ap.add_argument("--universe-dir", type=Path, default=None)
    ap.add_argument("--auto-build-universe", action="store_true")
    ap.add_argument("--breakout-days", type=int, default=20)
    ap.add_argument("--stop-loss", type=float, default=0.05)
    ap.add_argument("--take-profit", type=float, default=0.25)
    ap.add_argument("--max-hold-days", type=int, default=130)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-positions", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=None, help="Limit universe size for a quick run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--no-valuation",
        action="store_true",
        help="Skip DCF / valuation_engine in dcf mode (classification_engine only).",
    )
    ap.add_argument(
        "--signal-tech-ai",
        action="store_true",
        help="Use technicals + projection_engine (no DCF in composite) mapped to BUY tiers.",
    )
    ap.add_argument(
        "--checkpoint-freq",
        type=str,
        default="Q",
        metavar="Q|M",
        help="Q=quarter-end (default), M=month-end — more frequent fundamental/signal refresh.",
    )
    args = ap.parse_args()

    cp_min = args.checkpoint_min_year if args.checkpoint_min_year else None
    cp_cf = st._normalize_checkpoint_freq(args.checkpoint_freq)

    try:
        p = run_dynamic(
            lookback_years=args.lookback,
            checkpoint_min_year=cp_min,
            universe_dir=args.universe_dir,
            auto_build_universe=args.auto_build_universe,
            breakout_days=args.breakout_days,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_hold_days=args.max_hold_days,
            position_frac=args.position_frac,
            max_positions=args.max_positions,
            max_tickers=args.max_tickers,
            out_html=args.out,
            tickers=None,
            universe_map=None,
            use_valuation=not args.no_valuation,
            signal_mode="tech_ai" if args.signal_tech_ai else "dcf",
            checkpoint_freq=cp_cf,
        )
        print(f"\nSaved animated HTML -> {p}")
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
