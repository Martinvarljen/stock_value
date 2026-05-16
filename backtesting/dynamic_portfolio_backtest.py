#!/usr/bin/env python3
"""
Dynamic paper portfolio vs SPY buy-and-hold.

Rules (MVP, transparent defaults):
  1) Fundamentals: quarterly checkpoints reuse the same pipeline as strategy_backtest
     (reconstruct_data_at + classify_at). Only STRONG BUY / BUY are eligible for entries.
     Use ``--signal-tech-ai`` for BUY tiers from momentum/technicals + projection_engine
     (no DCF in composite; ML when a saved model exists). Use ``--no-valuation`` for DCF-off
     ``classification_engine`` only (same CLI as strategy_backtest.py).
  2) Technical entry (DCF / tier mode): 20-day high breakout on or after a BUY-tier
     checkpoint. ML rank mode (default): enter on any session while P(up)20d passes the
     gate — no breakout required — using the latest month-end (or quarter-end) score.
  3) Exit: stop-loss 20% below entry; take-profit 40% above entry (default); or max hold 130
     calendar days. Stops take precedence using that day's Low/High.
  4) Sizing: each new position uses ``position_frac`` of current NAV (default 10%), capped by
     cash and ``max_positions`` concurrent names.
  5) Benchmark: buy-and-hold SPY normalized to $1 on the first simulation day (never sold).

Output: standalone Plotly HTML with a slider/Play animation of monthly strategy vs SPY.

Requires yearly top-100 cache files (same as strategy_backtest). Example:

    python backtesting/build_yearly_top100_universe.py --for-checkpoints-from-year 2023
    python backtesting/dynamic_portfolio_backtest.py
    python backtesting/dynamic_portfolio_backtest.py --strategy ml
    python backtesting/run_ml_backtest.ps1
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
from backtesting.regime import gross_exposure_scale, spy_close_series
from reporting.dynamic_trade_trace import (
    build_entry_reason,
    build_entry_trace,
    build_exit_trace,
    build_short_entry_reason,
    build_short_entry_trace,
)
from backtesting.strategy_modes import is_ml_strategy, normalize_signal_mode, strategy_display_name
from backtesting.yearly_top100_universe import default_universe_cache_dir, load_universe_map_for_lag_years

from typing import Optional, Tuple

# (checkpoint_day, classification, optional p_up_20d for ML gate)
CheckpointSignal = Tuple[datetime, Optional[str], Optional[float]]


@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    stop: float
    take_profit: float
    entry_day: datetime
    peak: float
    entry_p_up: float | None = None
    entry_cls: str | None = None
    entry_reason: str = ""
    regime_scale_at_entry: float = 1.0
    side: str = "long"  # "long" | "short"


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


def _last_signal_on_or_before(rows: list[CheckpointSignal], d: datetime) -> tuple[str | None, float | None]:
    best_cl: str | None = None
    best_p: float | None = None
    for cp, cl, p_up in rows:
        if cp <= d:
            best_cl = cl
            best_p = p_up
        else:
            break
    return best_cl, best_p


def _row_asof(df: pd.DataFrame, d: datetime) -> pd.Series | None:
    ts = pd.Timestamp(_norm_day(d))
    idx = df.index
    if len(idx) == 0:
        return None
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return df.iloc[int(pos)]


def _next_bar_after(df: pd.DataFrame, d: datetime) -> pd.Series | None:
    """First bar strictly after ``d``. Used for t+1 open fills so a signal
    knowable only at today's close is executed on the next session's open.
    Returns ``None`` if no later bar exists in the frame."""
    ts = pd.Timestamp(_norm_day(d))
    idx = df.index
    if len(idx) == 0:
        return None
    pos = int(idx.searchsorted(ts, side="right"))
    if pos >= len(idx):
        return None
    return df.iloc[pos]


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
    pit_universe: bool = False,
    use_valuation: bool = True,
    signal_mode: str = "dcf",
    checkpoint_freq: str | None = None,
    min_p_up_20d: float | None = None,
    regime_filter: bool = False,
    bear_scale: float = 0.35,
    entry_mode: str = "tier",
    require_breakout: bool | None = None,
    enable_short: bool = False,
    max_p_up_short: float = 0.48,
    # --- execution knobs ---
    # Defaults match the pre-audit backtest (same-day close fills, zero
    # explicit costs). Pass commission/slippage or fill_at="next_open" for
    # the institutional-realism path.
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_bps_annual: float = 0.0,
    fill_at: str = "close",
) -> Path:
    signal_mode = normalize_signal_mode(signal_mode)
    entry_mode = entry_mode.strip().lower()
    if require_breakout is None:
        # ML rank: enter when the model says go, not only on a Donchian breakout.
        require_breakout = not (entry_mode == "rank" and is_ml_strategy(signal_mode))
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
            if pit_universe:
                # Survivorship-free path: per-year membership comes from
                # the curated S&P 500 change-log + delisted overlay
                # (see ``backtesting.sp500_pit_universe``). Cached on
                # disk under ``dollar_volume_top100_pit/`` so subsequent
                # runs are instant.
                from backtesting.yearly_top100_universe import build_pit_universe_map
                uni = build_pit_universe_map(lag_years, verbose=True)
            else:
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
    ml_note = ""
    if is_ml_strategy(signal_mode):
        ml_note = f" Signals: {strategy_display_name(signal_mode)}."
        if min_p_up_20d is not None:
            ml_note += f" Long when P(up) 20d >= {min_p_up_20d:.0%}."
        if enable_short:
            ml_note += f" Short when P(up) 20d <= {max_p_up_short:.0%}."
    if require_breakout:
        entry_desc = f"BUY tiers + {breakout_days}d breakout"
    elif entry_mode == "rank" and is_ml_strategy(signal_mode):
        entry_desc = f"ML rank (P(up) gate, no breakout; scores refresh each {'month' if cf == 'M' else 'quarter'})"
    else:
        entry_desc = "BUY tiers"
    print(
        f"Rules: {entry_desc}, "
        f"stop {stop_loss:.0%}, TP {take_profit:.0%}, max hold {max_hold_days}d, "
        f"{position_frac:.0%} NAV/trade, max {max_positions} positions."
        + ("" if use_valuation or is_ml_strategy(signal_mode) else " Valuation engine OFF.")
        + ml_note
        + "\n"
    )

    spy_feat = spy_df["Close"].astype(float)

    def _eligible_long(cls: str | None, p_up: float | None) -> bool:
        if entry_mode == "rank" and is_ml_strategy(signal_mode):
            if p_up is None:
                return False
            if min_p_up_20d is not None and p_up < min_p_up_20d:
                return False
            return True
        return cls in st.BUY_TIERS

    def _eligible_short(cls: str | None, p_up: float | None) -> bool:
        if not enable_short:
            return False
        if entry_mode == "rank" and is_ml_strategy(signal_mode):
            return p_up is not None and p_up <= max_p_up_short
        return cls in getattr(st, "SELL_TIERS", ("STRONG AVOID", "AVOID"))

    raw_by: dict[str, dict] = {}
    classes: dict[str, list[CheckpointSignal]] = {}
    # Cache the cleaned price frame per ticker once. Earlier versions called
    # ``_prep_price_df(raw["price_history"])`` inside the per-day inner loop,
    # producing O(positions × days × universe) full-frame copies — the single
    # biggest CPU cost in the backtest.
    px_by: dict[str, pd.DataFrame] = {}

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
        qrows: list[CheckpointSignal] = []
        for cp in checkpoints:
            ly = cp.year - 1
            if uni_for_cp:
                allowed = set(uni_for_cp.get(ly, []))
                if allowed and tk not in allowed:
                    continue
            data = st.reconstruct_data_at(raw, cp)
            if data is None:
                continue
            sig_meta: dict = {}
            cl = st.classify_at(
                data,
                raw,
                cp,
                use_valuation=use_valuation,
                signal_mode=signal_mode,
                signal_meta=sig_meta if is_ml_strategy(signal_mode) else None,
                spy_close_series=spy_feat,
            )
            p_up = sig_meta.get("p_up_20d") if is_ml_strategy(signal_mode) else None
            if p_up is not None:
                try:
                    p_up = float(p_up)
                except (TypeError, ValueError):
                    p_up = None
            qrows.append((_norm_day(cp), cl, p_up))
        if not qrows:
            continue
        qrows.sort(key=lambda x: x[0])
        raw_by[tk] = raw
        px_by[tk] = px
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
    ledger: list[dict] = []
    traces: list[dict] = []

    # Per-trade frictions. Hard-coded conservative defaults — easy to plumb
    # to argparse later. Kept here rather than in the broker because this
    # simulator is the production-path event-driven backtest today.
    cost_one_way = float(commission_bps + slippage_bps) / 10_000.0  # round-trip = 2 × this
    borrow_daily = float(borrow_bps_annual) / 10_000.0 / 252.0  # only meaningful for shorts

    for d in trading_days:
        spy_row = _row_asof(spy_df, d)
        if spy_row is None:
            continue
        spy_close = float(spy_row["Close"])

        # --- exits first ---
        still: list[Position] = []
        for p in positions:
            df = px_by.get(p.ticker)
            if df is None:
                still.append(p)
                continue
            row = _row_asof(df, d)
            if row is None:
                still.append(p)
                continue
            lo, hi, cl = float(row["Low"]), float(row["High"]), float(row["Close"])
            side = getattr(p, "side", "long")
            if side == "long":
                p.peak = max(p.peak, cl)
            exit_px = None
            exit_reason = ""
            if side == "long":
                if lo <= p.stop:
                    exit_px = p.stop
                    exit_reason = f"Stop loss ({stop_loss:.0%} below entry)"
                elif hi >= p.take_profit:
                    exit_px = p.take_profit
                    exit_reason = f"Take profit ({take_profit:.0%} above entry)"
                elif (d - p.entry_day).days >= max_hold_days:
                    exit_px = cl
                    exit_reason = f"Max hold {max_hold_days}d"
            else:
                if hi >= p.stop:
                    exit_px = p.stop
                    exit_reason = f"Stop loss ({stop_loss:.0%} above entry)"
                elif lo <= p.take_profit:
                    exit_px = p.take_profit
                    exit_reason = f"Take profit ({take_profit:.0%} below entry)"
                elif (d - p.entry_day).days >= max_hold_days:
                    exit_px = cl
                    exit_reason = f"Max hold {max_hold_days}d"
            if exit_px is not None:
                if side == "long":
                    exit_px_net = exit_px * (1.0 - cost_one_way)
                    cash += p.shares * exit_px_net
                    pnl_pct = (exit_px_net / p.entry_price - 1.0) if p.entry_price > 0 else None
                else:
                    exit_px_net = exit_px * (1.0 + cost_one_way)
                    cash -= p.shares * exit_px_net
                    pnl_pct = (
                        (p.entry_price - exit_px_net) / p.entry_price if p.entry_price > 0 else None
                    )
                ledger.append(
                    {
                        "date": _norm_day(d).date().isoformat(),
                        "ticker": p.ticker,
                        "action": "EXIT",
                        "side": side,
                        "price": exit_px_net,
                        "reason": exit_reason,
                        "pnl_pct": pnl_pct,
                        "p_up_20d": p.entry_p_up,
                    }
                )
                traces.append(
                    build_exit_trace(
                        ticker=p.ticker,
                        day=d,
                        reason=exit_reason,
                        p_up_at_entry=p.entry_p_up,
                    )
                )
            else:
                still.append(p)
        positions = still

        # --- NAV for sizing (cash + open positions MTM) ---
        nav_pre = cash
        for p in positions:
            df = px_by.get(p.ticker)
            if df is None:
                continue
            row = _row_asof(df, d)
            if row is None:
                continue
            cl = float(row["Close"])
            if getattr(p, "side", "long") == "long":
                nav_pre += p.shares * cl
            else:
                nav_pre -= p.shares * cl

        # --- new entries ---
        # ``unknown_scale=bear_scale`` makes the regime gate *abstain* when
        # SPY history is too short to classify (legacy behaviour was to
        # default risk-on, i.e. trade into uncertainty).
        scale = (
            gross_exposure_scale(spy_feat, d, bear_scale=bear_scale)
            if regime_filter
            else 1.0
        )
        eff_frac = position_frac * scale
        if scale < 0.05:
            eff_frac = 0.0

        if len(positions) < max_positions and eff_frac > 0:

            def _fill_entry(df_tk: pd.DataFrame) -> tuple[float, datetime] | None:
                if fill_at == "close":
                    row = _row_asof(df_tk, d)
                    if row is None:
                        return None
                    return float(row["Close"]), d
                nxt = _next_bar_after(df_tk, d)
                if nxt is None:
                    return None
                entry_day = nxt.name.to_pydatetime() if hasattr(nxt.name, "to_pydatetime") else d
                return float(nxt["Open"]), entry_day

            long_cand: list[tuple[float, str]] = []
            short_cand: list[tuple[float, str]] = []
            for tk in raw_by.keys():
                if any(x.ticker == tk for x in positions):
                    continue
                cls, p_up = _last_signal_on_or_before(classes[tk], d)
                df_tk = px_by.get(tk)
                if df_tk is None:
                    continue
                if _eligible_long(cls, p_up):
                    if require_breakout and not _breakout_today(df_tk, d, breakout_days):
                        pass
                    else:
                        long_cand.append((float(p_up) if p_up is not None else 0.0, tk))
                if _eligible_short(cls, p_up):
                    short_cand.append((float(p_up) if p_up is not None else 1.0, tk))
            long_cand.sort(key=lambda r: (-r[0], r[1]))
            short_cand.sort(key=lambda r: (r[0], r[1]))

            entry_queue: list[tuple[bool, str]] = [(False, tk) for _, tk in long_cand] + [
                (True, tk) for _, tk in short_cand
            ]
            for is_short, tk in entry_queue:
                if len(positions) >= max_positions:
                    break
                if any(x.ticker == tk for x in positions):
                    continue
                cls_ent, p_up_ent = _last_signal_on_or_before(classes[tk], d)
                df_tk = px_by[tk]
                filled = _fill_entry(df_tk)
                if filled is None:
                    continue
                entry, entry_day = filled
                if entry <= 0:
                    continue
                notional = min(eff_frac * nav_pre, max(cash, nav_pre) * 0.999)
                if notional < 1e-4:
                    continue
                cost = notional * cost_one_way
                shares = (notional - cost) / entry
                if is_short:
                    cash += notional - cost
                    entry_reason = build_short_entry_reason(
                        p_up=p_up_ent,
                        max_p_up=max_p_up_short,
                        regime_scale=scale,
                    )
                    positions.append(
                        Position(
                            ticker=tk,
                            shares=shares,
                            entry_price=entry,
                            stop=entry * (1.0 + stop_loss),
                            take_profit=entry * (1.0 - take_profit),
                            entry_day=entry_day,
                            peak=entry,
                            entry_p_up=p_up_ent,
                            entry_cls=cls_ent,
                            entry_reason=entry_reason,
                            regime_scale_at_entry=scale,
                            side="short",
                        )
                    )
                    ledger.append(
                        {
                            "date": _norm_day(entry_day).date().isoformat(),
                            "ticker": tk,
                            "action": "ENTER_SHORT",
                            "side": "short",
                            "price": entry,
                            "notional": notional,
                            "reason": entry_reason,
                            "p_up_20d": p_up_ent,
                        }
                    )
                    traces.append(
                        build_short_entry_trace(
                            ticker=tk,
                            day=entry_day,
                            p_up=p_up_ent,
                            regime_scale=scale,
                            reason=entry_reason,
                            max_p_up=max_p_up_short,
                        )
                    )
                else:
                    if cash < notional * 0.999:
                        continue
                    cash -= notional
                    entry_reason = build_entry_reason(
                        p_up=p_up_ent,
                        cls=cls_ent,
                        min_p_up=min_p_up_20d,
                        require_breakout=require_breakout,
                        regime_scale=scale,
                        entry_mode=entry_mode,
                        ml_mode=is_ml_strategy(signal_mode),
                    )
                    positions.append(
                        Position(
                            ticker=tk,
                            shares=shares,
                            entry_price=entry,
                            stop=entry * (1.0 - stop_loss),
                            take_profit=entry * (1.0 + take_profit),
                            entry_day=entry_day,
                            peak=entry,
                            entry_p_up=p_up_ent,
                            entry_cls=cls_ent,
                            entry_reason=entry_reason,
                            regime_scale_at_entry=scale,
                            side="long",
                        )
                    )
                    ledger.append(
                        {
                            "date": _norm_day(entry_day).date().isoformat(),
                            "ticker": tk,
                            "action": "ENTER_LONG",
                            "side": "long",
                            "price": entry,
                            "notional": notional,
                            "reason": entry_reason,
                            "p_up_20d": p_up_ent,
                        }
                    )
                    traces.append(
                        build_entry_trace(
                            ticker=tk,
                            day=entry_day,
                            p_up=p_up_ent,
                            regime_scale=scale,
                            reason=entry_reason,
                            min_p_up=min_p_up_20d,
                        )
                    )
                nav_pre = cash
                for p in positions:
                    df = px_by.get(p.ticker)
                    if df is None:
                        continue
                    row = _row_asof(df, d)
                    if row is None:
                        continue
                    cl = float(row["Close"])
                    if getattr(p, "side", "long") == "long":
                        nav_pre += p.shares * cl
                    else:
                        nav_pre -= p.shares * cl

        # --- daily borrow accrual on shorts ---
        if borrow_daily > 0 and positions:
            for p in positions:
                if p.side == "short":
                    df_tk = px_by.get(p.ticker)
                    if df_tk is None:
                        continue
                    row = _row_asof(df_tk, d)
                    if row is None:
                        continue
                    cash -= borrow_daily * p.shares * float(row["Close"])

        nav_end = cash
        for p in positions:
            df = px_by.get(p.ticker)
            if df is None:
                continue
            row = _row_asof(df, d)
            if row is None:
                continue
            cl = float(row["Close"])
            if getattr(p, "side", "long") == "long":
                nav_end += p.shares * cl
            else:
                nav_end -= p.shares * cl

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
    out = Path(out)
    _write_monthly_animation_html(
        monthly,
        out,
        "Dynamic strategy (fundamental + breakout) vs SPY buy-and-hold — monthly (animated)",
    )

    if ledger:
        stem = out.with_suffix("")
        trades_html = Path(f"{stem}_trades.html")
        flow_html = Path(f"{stem}_pipeline.html")
        trades_json = Path(f"{stem}_trades.json")
        s0 = float(comb["strategy"].iloc[0]) if len(comb) else 1.0
        s1 = float(comb["strategy"].iloc[-1]) if len(comb) else 1.0
        b0 = float(comb["spy_bh"].iloc[0]) if len(comb) else 1.0
        b1 = float(comb["spy_bh"].iloc[-1]) if len(comb) else 1.0
        years = max((comb.index[-1] - comb.index[0]).days / 365.25, 1e-6) if len(comb) > 1 else 1.0
        cagr_s = (s1 / s0) ** (1.0 / years) - 1.0 if s0 > 0 else None
        cagr_b = (b1 / b0) ** (1.0 / years) - 1.0 if b0 > 0 else None
        summary = {
            "from": str(comb.index[0].date()) if len(comb) else "",
            "to": str(comb.index[-1].date()) if len(comb) else "",
            "years": round(years, 2),
            "strategy_cagr": cagr_s,
            "spy_cagr": cagr_b,
            "final_nav": s1,
            "final_spy_bh": b1,
            "entry_mode": entry_mode,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "max_hold_days": max_hold_days,
            "require_breakout": require_breakout,
            "checkpoint_freq": cf,
            "enable_short": enable_short,
            "max_p_up_short": max_p_up_short,
        }
        from reporting.backtest_flow_html import write_flow_map_html
        from reporting.backtest_viz import write_backtest_report

        write_backtest_report(
            curve=comb,
            ledger=ledger,
            snapshots=[],
            summary=summary,
            out_html=trades_html,
            out_json=trades_json,
        )
        write_flow_map_html(
            traces=traces,
            ledger=ledger,
            summary=summary,
            out_html=flow_html,
        )
        n_round = len([r for r in ledger if r.get("action") == "EXIT"])
        n_long = len([r for r in ledger if r.get("action") == "ENTER_LONG"])
        n_short = len([r for r in ledger if r.get("action") == "ENTER_SHORT"])
        print(
            f"\n  Trade journal ({len(ledger)} events, {n_round} round-trips, "
            f"{n_long} long entries, {n_short} short entries):\n"
            f"    NAV + markers  -> {trades_html}\n"
            f"    Pipeline map   -> {flow_html}\n"
            f"    JSON log       -> {trades_json}"
        )

    return out


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
    ap.add_argument("--stop-loss", type=float, default=0.20)
    ap.add_argument(
        "--take-profit",
        type=float,
        default=0.40,
        help="Take-profit as fraction above entry (default 0.40 = 40%%).",
    )
    ap.add_argument("--max-hold-days", type=int, default=130)
    ap.add_argument("--position-frac", type=float, default=0.10)
    ap.add_argument("--max-positions", type=int, default=10)
    ap.add_argument("--max-tickers", type=int, default=None, help="Limit universe size for a quick run")
    ap.add_argument(
        "--ticker",
        nargs="+",
        metavar="SYM",
        help="Explicit ticker list (skips universe file; e.g. --ticker TSLA).",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--enable-short",
        action="store_true",
        help="ML: also enter shorts when P(up)20d <= --max-p-up-short (default 0.42).",
    )
    ap.add_argument(
        "--max-p-up-short",
        type=float,
        default=0.48,
        help="Short entry when P(up)20d is at or below this (default 0.48; long gate is 0.52).",
    )
    ap.add_argument(
        "--no-valuation",
        action="store_true",
        help="Skip DCF / valuation_engine in dcf mode (classification_engine only).",
    )
    ap.add_argument(
        "--signal-tech-ai",
        action="store_true",
        help="Same as --strategy ml (legacy flag).",
    )
    ap.add_argument(
        "--strategy",
        choices=("dcf", "ml"),
        default="dcf",
        help="dcf=valuation classifier (default); ml=Dolt-trained LightGBM via projection_engine.",
    )
    ap.add_argument(
        "--min-p-up",
        type=float,
        default=None,
        metavar="P",
        help="ML mode only: require P(up) 20d >= P for breakout entry (default 0.52 when --strategy ml).",
    )
    ap.add_argument(
        "--regime-filter",
        action="store_true",
        help="Scale down / skip new entries when SPY is below 200d MA.",
    )
    ap.add_argument(
        "--entry-mode",
        choices=("tier", "rank"),
        default="tier",
        help="tier=BUY/STRONG BUY only; rank=ML score gate (use with --strategy ml).",
    )
    ap.add_argument(
        "--checkpoint-freq",
        type=str,
        default="Q",
        metavar="Q|M",
        help="Q=quarter-end (default), M=month-end — more frequent fundamental/signal refresh.",
    )
    ap.add_argument(
        "--realistic-costs",
        action="store_true",
        help="Institutional fills: 1+2 bps per leg, 50 bps borrow, entries at next open.",
    )
    ap.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="Per-leg commission in bps (default 0; use --realistic-costs for 1.0).",
    )
    ap.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="Per-leg slippage in bps (default 0; use --realistic-costs for 2.0).",
    )
    ap.add_argument(
        "--borrow-bps-annual",
        type=float,
        default=None,
        help="Annualised short borrow bps (default 0; use --realistic-costs for 50).",
    )
    ap.add_argument(
        "--fill-at",
        choices=("next_open", "close"),
        default=None,
        help="Entry fill: close (default, legacy) or next_open (no same-day lookahead).",
    )
    ap.add_argument(
        "--pit-universe",
        action="store_true",
        help=(
            "Use the survivorship-free PIT S&P 500 universe (delisted "
            "overlay from backtesting/sp500_changes.csv). Default is the "
            "legacy current-list-only universe — see "
            "yearly_top100_universe.py docstring for the bias warning."
        ),
    )
    args = ap.parse_args()

    cp_min = args.checkpoint_min_year if args.checkpoint_min_year else None
    cp_cf = st._normalize_checkpoint_freq(args.checkpoint_freq)

    strat = "ml" if args.signal_tech_ai else args.strategy
    min_p = args.min_p_up
    if min_p is None and strat == "ml":
        min_p = 0.52
    # ML backtests use rank gate (P(up) threshold), not BUY-tier labels.
    entry_m = "rank" if is_ml_strategy(strat) else args.entry_mode
    enable_short = args.enable_short or is_ml_strategy(strat)

    if args.realistic_costs:
        commission_bps = 1.0
        slippage_bps = 2.0
        borrow_bps = 50.0
        fill_at = "next_open"
    else:
        commission_bps = 0.0 if args.commission_bps is None else args.commission_bps
        slippage_bps = 0.0 if args.slippage_bps is None else args.slippage_bps
        borrow_bps = 0.0 if args.borrow_bps_annual is None else args.borrow_bps_annual
        fill_at = "close" if args.fill_at is None else args.fill_at

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
            tickers=[t.upper() for t in args.ticker] if args.ticker else None,
            universe_map=None,
            pit_universe=args.pit_universe,
            use_valuation=not args.no_valuation,
            signal_mode=strat,
            checkpoint_freq=cp_cf,
            min_p_up_20d=min_p,
            regime_filter=args.regime_filter or strat == "ml",
            entry_mode=entry_m,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            borrow_bps_annual=borrow_bps,
            fill_at=fill_at,
            enable_short=enable_short,
            max_p_up_short=args.max_p_up_short,
        )
        print(f"\nSaved animated HTML -> {p}")
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
