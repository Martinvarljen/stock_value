"""
Stock Projection Dashboard — Streamlit + Plotly

Architecture
────────────
  Heavy  : run_fundamentals()  valuation, quality, risk, classification
           → @st.cache_data(ttl=3600)  run once, cached 1 hour
  Live   : get_live_data()     current price + 1Y OHLCV chart
           → @st.cache_data(ttl=30)   refreshed every 30 s automatically
  Candle : optional LSTM (train_candle_sequence.py) predicts next 12 daily OHLC
           bars; inputs follow Kaufman-style change/range channels when IN_DIM=6.
  TSM    : extended technicals include Kaufman *Trading Systems & Methods* metrics
           (momentum, efficiency ratio, linear regression trend) — see kaufman_tsm.py.
  News   : run_news()          FinBERT + Claude headline sentiment
           → @st.cache_data(ttl=300)  refreshed every 5 min

The live section uses @st.fragment(run_every=30) so the price, chart, and
projections update automatically without touching the heavy pipeline.

Run with (from the Finance folder — important for imports):
    powershell -ExecutionPolicy Bypass -File .\\run_dashboard.ps1
    python -m streamlit run dashboard/app.py
"""

import hashlib
import math
import sys
import time
from pathlib import Path
from random import Random
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(_ROOT / "stock_analyzer"), str(_ROOT / "projection")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from projection_engine import generate_projections
from data_layer import collect_data
from quality_engine import analyze_quality
from financial_strength import analyze_financials
from valuation_engine import analyze_valuation
from growth_engine import analyze_growth
from risk_engine import analyze_risk
from red_flags import analyze_red_flags
from classification_engine import classify_stock
from sector_engine import apply_sector_context
from momentum_engine import analyze_momentum
from technical_extended import analyze_extended_technicals
from elliott_engine import analyze_elliott_context
from trade_setup_engine import build_trade_setup
from candle_patterns import analyze_candle_patterns
from ohlcv_validate import validate_ohlcv_from_data_dict
from market_structure import analyze_market_structure
from news_engine import analyze_news
try:
    from ml_model.predictor import models_available, model_summary, models_load_hint
except ImportError:
    from ml_model.predictor import models_available, model_summary

    def models_load_hint() -> str:
        return ""

try:
    from ml_model.candle_seq_infer import (
        candle_sequence_available,
        candle_sequence_summary,
        predict_future_ohlc,
    )
except ImportError:

    def candle_sequence_available() -> bool:
        return False

    def candle_sequence_summary() -> str:
        return ""

    def predict_future_ohlc(hist, anchor_close=None):
        return None

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock Projection Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: #0e1117; border-radius: 8px; padding: 10px;
        border: 1px solid #262730;
    }
    .signal-bullish  { color: #00d4aa; font-size: 1.5rem; font-weight: bold; }
    .signal-bearish  { color: #ff4757; font-size: 1.5rem; font-weight: bold; }
    .signal-neutral  { color: #ffa502; font-size: 1.5rem; font-weight: bold; }
    .live-badge      { color: #00d4aa; font-size: 0.75rem; }
    .stale-badge     { color: #888;    font-size: 0.75rem; }
    .news-card       { background: #1a1a2e; border-radius: 8px; padding: 10px;
                       margin-bottom: 6px; border-left: 3px solid #333; }
    .news-pos        { border-left-color: #00d4aa; }
    .news-neg        { border-left-color: #ff4757; }
</style>
""", unsafe_allow_html=True)


# ── sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Projection Dashboard")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL", max_chars=10).upper().strip()

    st.markdown("---")
    st.subheader("Settings")
    margin_of_safety = st.slider("Margin of Safety", 0.10, 0.50, 0.30, 0.05, key="mos")
    horizon = st.selectbox("Projection Horizon (days)", [60, 90, 120, 180, 252], index=2, key="horizon")
    refresh_sec = st.selectbox("Live refresh interval", [15, 30, 60, 120], index=1,
                               format_func=lambda s: f"{s}s", key="refresh_sec")

    st.markdown("---")
    st.subheader("News Analysis")
    enable_news = st.toggle("Enable news sentiment", value=True, key="enable_news")
    news_days = st.slider("News lookback (days)", 3, 30, 7, key="news_days") if enable_news else 7

    st.markdown("---")
    if models_available():
        st.success("🤖 ML model active")
        with st.expander("Model info"):
            st.caption(model_summary())
    else:
        st.warning("No ML model — rule-based")
        _hint = models_load_hint()
        if _hint:
            st.caption(_hint)
        with st.expander("Train ML model"):
            st.code(
                "cd <Finance folder>\npython projection/ml_model/trainer.py",
                language="bash",
            )

    st.markdown("---")
    st.subheader("Candle-sequence LSTM")
    if candle_sequence_available():
        st.success("OHLC sequence model loaded")
        st.caption(candle_sequence_summary())
    else:
        st.info("Not trained — chart uses blended fundamentals path.")
        with st.expander("Train candle LSTM"):
            st.caption(
                "Learns next daily OHLC bars from past windows (PyTorch). "
                "Inputs include daily return and (H−L)/C per bar (Kaufman-style path vs. noise). "
                "Separate from the P(up) LightGBM models."
            )
            st.code(
                "python projection/ml_model/train_candle_sequence.py\n"
                "python projection/ml_model/train_candle_sequence.py --quick",
                language="bash",
            )

    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    st.caption("Data: yfinance · 15min delay")


# ── cached data functions ──────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def run_fundamentals(ticker: str, margin_of_safety: float) -> tuple:
    """Heavy pipeline — run once, cache 1 hour."""
    data = collect_data(ticker)
    if data.get("error") and data.get("data_quality_score", 0) < 20:
        return None, f"Data error: {data.get('error')}"
    if data.get("quote_type") == "ETF":
        return None, "ETFs are not supported (no fundamental data)"

    sector_result    = apply_sector_context(data)
    quality_result   = analyze_quality(data)
    financial_result = analyze_financials(data)

    valuation_result = analyze_valuation(
        {**data, "sector_result": sector_result},
        margin_of_safety=margin_of_safety,
        wacc_adjustment=sector_result["wacc_adjustment"],
        terminal_growth_range=sector_result.get("terminal_growth_range"),
    )

    growth_result    = analyze_growth(data)
    risk_result      = analyze_risk(data)
    wacc             = valuation_result.get("wacc_data", {}).get("wacc")
    red_flag_result  = analyze_red_flags(data, wacc=wacc)
    momentum_result  = analyze_momentum(data)
    extended_tech    = analyze_extended_technicals(data)
    elliott_ctx      = analyze_elliott_context(data)
    candle_ctx       = analyze_candle_patterns(data)
    ohlcv_qc         = validate_ohlcv_from_data_dict(data)
    mkt_struct       = analyze_market_structure(data)

    all_critical = (
        (financial_result.get("critical_flags") or []) +
        (risk_result.get("critical_flags") or [])
    )

    record = {
        **data,
        "valuation_metrics":   valuation_result["valuation_metrics"],
        "fair_value_weighted": valuation_result["fair_value_weighted"],
        "buy_below_price":     valuation_result["buy_below_price"],
        "wacc_data":           valuation_result["wacc_data"],
        "scenarios":           valuation_result["scenarios"],
        "growth_metrics":      growth_result["growth_metrics"],
        "risk_metrics":        risk_result["risk_metrics"],
        "red_flags":           red_flag_result["red_flags"],
        "critical_flags":      all_critical,
        "momentum_metrics":    momentum_result["momentum_metrics"],
        "momentum_trend":      momentum_result["trend"],
        "sector_result":       sector_result,
        "extended_technicals": extended_tech,
        "elliott_context":     elliott_ctx,
        "candle_patterns":     candle_ctx,
        "ohlcv_quality":       ohlcv_qc,
        "market_structure":    mkt_struct,
    }

    clf = classify_stock(record)
    record["classification"]        = clf["classification"]
    record["classification_result"] = clf
    record["trade_setup"]           = build_trade_setup(record)

    return record, None


@st.cache_data(ttl=30, show_spinner=False)
def get_live_data(ticker: str) -> dict:
    """
    Fast refresh — current price + 1Y daily chart.
    Called every 30 s (or whatever refresh_sec is set to).
    """
    tk = yf.Ticker(ticker)

    price = None
    prev_close = None
    try:
        fi = tk.fast_info
        price      = float(fi.last_price)
        prev_close = float(fi.previous_close)
    except Exception:
        pass

    hist = tk.history(period="1y", interval="1d")
    if not hist.empty:
        hist.index = hist.index.tz_localize(None)
        if price is None:
            price = float(hist["Close"].iloc[-1])
        if prev_close is None and len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])

    return {
        "price":      price,
        "prev_close": prev_close,
        "hist":       hist,
        "fetched_at": datetime.now(),
    }


@st.cache_data(ttl=300, show_spinner=False)
def run_news(ticker: str, days_back: int) -> dict:
    return analyze_news(ticker, days_back=days_back)


# ── on "Run Analysis" ──────────────────────────────────────────────────────────

if run_btn:
    with st.spinner(f"Running fundamental analysis for {ticker}…"):
        base_record, error = run_fundamentals(ticker, margin_of_safety)

    if error:
        st.error(error)
        st.stop()

    st.session_state.base_record    = base_record
    st.session_state.active_ticker  = ticker

    if enable_news:
        with st.spinner("Fetching news…"):
            st.session_state.news_result = run_news(ticker, news_days)
    else:
        st.session_state.news_result = None


def _hist_atr_dollars(hist, lookback: int = 24) -> float | None:
    """Typical daily range ($) from recent history for ghost-candle wicks."""
    if hist is None or hist.empty or len(hist) < 2:
        return None
    tail = hist.tail(min(lookback, len(hist)))
    try:
        r = (tail["High"].astype(float) - tail["Low"].astype(float)).mean()
        v = float(r)
        if math.isnan(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


def _news_fingerprint(news_result: dict | None) -> str:
    """Stable string from current headlines/scores so candle microstructure tracks news."""
    if not news_result or not news_result.get("available"):
        return "no_news"
    parts = [
        f"{float(news_result.get('sentiment_score') or 0):.5f}",
        str(int(news_result.get("n_articles") or 0)),
        str(news_result.get("signal") or ""),
    ]
    for a in (news_result.get("articles") or [])[:6]:
        parts.append((a.get("title") or "")[:120])
    return "|".join(parts)


def _interp_from_pivots(day: int, pivot_days: list, values: list) -> float:
    """Piecewise-linear value at trading-day `day` using engine pivot nodes."""
    if not pivot_days or not values:
        return 0.0
    if day <= int(pivot_days[0]):
        return float(values[0])
    if day >= int(pivot_days[-1]):
        return float(values[-1])
    for i in range(len(pivot_days) - 1):
        d0, d1 = int(pivot_days[i]), int(pivot_days[i + 1])
        if d0 <= day <= d1:
            span = max(d1 - d0, 1)
            t = (day - d0) / span
            return float(values[i] + t * (values[i + 1] - values[i]))
    return float(values[-1])


def _scenario_blend_weights(composite_score: float) -> tuple[float, float, float]:
    """
    Turn the three engine scenarios into one weight triple (bull, base, bear).
    Neutral score → mostly base; bullish → bull + base; bearish → bear + base.
    """
    s = max(-1.0, min(1.0, float(composite_score)))
    w_base = 1.0 - abs(s)
    w_bull = abs(s) if s >= 0 else 0.0
    w_bear = abs(s) if s < 0 else 0.0
    t = w_base + w_bull + w_bear
    if t <= 0:
        return 0.0, 1.0, 0.0
    return w_bull / t, w_base / t, w_bear / t


def _consensus_close_path(
    composite_score: float,
    path_bull: list,
    path_base: list,
    path_bear: list,
) -> list[float]:
    """Single blended close path (same length as resampled scenario arrays)."""
    wb, w0, wr = _scenario_blend_weights(composite_score)
    n = len(path_base)
    return [
        wb * float(path_bull[i]) + w0 * float(path_base[i]) + wr * float(path_bear[i])
        for i in range(n)
    ]


def _band_around_consensus(
    path_consensus: list[float],
    path_base: list[float],
    upper: list[float] | None,
    lower: list[float] | None,
) -> tuple[list[float] | None, list[float] | None]:
    """Keep base-path σ width but re-anchor the channel on the blended forecast."""
    if upper is None or lower is None:
        return None, None
    if len(path_consensus) != len(path_base) or len(upper) != len(path_base):
        return None, None
    u_out = [path_consensus[i] + (upper[i] - path_base[i]) for i in range(len(path_base))]
    l_out = [path_consensus[i] + (lower[i] - path_base[i]) for i in range(len(path_base))]
    return u_out, l_out


def _ghost_resampled_series(
    paths: dict,
    upper_band: list | None,
    lower_band: list | None,
    last_date,
    *,
    early_trading_days: int = 8,
) -> tuple[list, list, list, list, list, list | None, list | None]:
    """
    Resample scenario paths for charting: **one candle per trading day** for the
    first `early_trading_days`, then only original pivot days — dense near-term
    tape without hundreds of bars for the full horizon.
    """
    pivot_days = paths["days"]
    if not pivot_days:
        return [], [], [], [], [], None, None
    H = int(pivot_days[-1])
    early = max(1, min(int(early_trading_days), H))
    day_grid = sorted(set(range(0, early + 1)) | {int(d) for d in pivot_days if int(d) > early} | {H})

    future_dates = [last_date + timedelta(days=int(d * 365 / 252)) for d in day_grid]
    base = [_interp_from_pivots(d, pivot_days, paths["base"]) for d in day_grid]
    bear = [_interp_from_pivots(d, pivot_days, paths["bear"]) for d in day_grid]
    bull = [_interp_from_pivots(d, pivot_days, paths["bull"]) for d in day_grid]

    ub = lb = None
    if upper_band is not None and len(upper_band) == len(pivot_days):
        ub = [_interp_from_pivots(d, pivot_days, upper_band) for d in day_grid]
    if lower_band is not None and len(lower_band) == len(pivot_days):
        lb = [_interp_from_pivots(d, pivot_days, lower_band) for d in day_grid]

    return day_grid, future_dates, base, bear, bull, ub, lb


def _forecast_ghost_candles(
    future_dates,
    path_close: list,
    upper_band: list | None,
    lower_band: list | None,
    anchor_open: float,
    hist,
    news_result: dict | None,
    ticker: str,
    scenario: str = "base",
):
    """
    Build OHLC rows that follow the scenario path but look like real candles:
    ATR-sized bodies/wicks, alternating stress, and a sentiment tilt so when
    news changes the whole ghost pattern shifts and reshapes (deterministic RNG).
    """
    n = len(path_close)
    if n < 2 or len(future_dates) != n:
        return None, None, None, None, None

    atr = _hist_atr_dollars(hist)
    mid = float(path_close[max(1, n // 2)])
    if atr is None or atr <= 0:
        atr = max(mid * 0.01, 0.02)

    seed_src = f"{ticker}|{scenario}|{_news_fingerprint(news_result)}"
    seed = int(hashlib.sha256(seed_src.encode("utf-8", errors="ignore")).hexdigest()[:12], 16)
    rng = Random(seed)

    ns = 0.0
    if news_result and news_result.get("available"):
        try:
            ns = float(news_result.get("sentiment_score") or 0.0)
        except (TypeError, ValueError):
            ns = 0.0
    ns = max(-1.0, min(1.0, ns))

    anchor = float(anchor_open)
    # End-of-horizon price lift from news (bullish headlines nudge the cloud upward)
    news_lift = anchor * ns * 0.005

    xs, O, H, L, C = [], [], [], [], []
    for i in range(1, n):
        t = i / max(n - 1, 1)
        shift = news_lift * t

        raw_s = float(path_close[i - 1])
        raw_e = float(path_close[i])
        tgt_e = raw_e + shift

        if i == 1:
            o = anchor
        else:
            gap = rng.gauss(0, 0.28 * atr)
            o = C[-1] + gap

        # Path-faithful close with room for red/green bodies
        c = tgt_e + rng.gauss(0, 0.22 * atr)
        c = 0.58 * c + 0.42 * tgt_e

        # Occasional counter-trend bar (long wick, small body) like real tape
        if rng.random() < 0.22:
            mid_body = 0.5 * (o + c)
            span = abs(c - o)
            flip = 1 if rng.random() < 0.5 else -1
            c = mid_body + flip * span * rng.uniform(0.15, 0.55)
            c = 0.55 * c + 0.45 * tgt_e

        body_hi = max(o, c)
        body_lo = min(o, c)
        wick_up = rng.uniform(0.4, 1.15) * atr
        wick_dn = rng.uniform(0.4, 1.15) * atr
        hi = body_hi + wick_up
        lo = body_lo - wick_dn

        if upper_band is not None and i < len(upper_band):
            hi = max(hi, float(upper_band[i]) + shift)
        if lower_band is not None and i < len(lower_band):
            lo = min(lo, float(lower_band[i]) + shift)

        hi = max(hi, o, c)
        lo = min(lo, o, c)

        xs.append(future_dates[i])
        O.append(o)
        H.append(hi)
        L.append(lo)
        C.append(c)

    return xs, O, H, L, C


def _prepare_live_projections(projections: dict) -> dict | None:
    """
    Return a dict safe for the live UI, or None when projections failed
    (e.g. no price). Older projection_engine builds may omit 5d fields — fill
    from 20d so metrics do not KeyError.
    """
    if projections.get("error"):
        return None
    projections.setdefault("ml_used", False)
    if "p_up_5d" not in projections:
        projections["p_up_5d"] = projections.get("p_up_20d", 0.5)
    if "expected_return_5d" not in projections:
        projections["expected_return_5d"] = projections.get("expected_return_20d", 0.0)
    return projections


# ── live dashboard (auto-refreshes) ───────────────────────────────────────────

if (
    st.session_state.get("base_record") is not None
    and st.session_state.get("active_ticker") == ticker
):

    @st.fragment(run_every=st.session_state.get("refresh_sec", 30))
    def live_dashboard():
        base_record = st.session_state.base_record
        _horizon    = st.session_state.get("horizon", 120)

        # ── get live price ─────────────────────────────────────────────────
        live = get_live_data(st.session_state.active_ticker)

        def _finite_price(v, fallback):
            try:
                x = float(v)
                if math.isnan(x) or math.isinf(x) or x <= 0:
                    raise ValueError
                return x
            except (TypeError, ValueError):
                try:
                    fb = float(fallback)
                    if math.isnan(fb) or math.isinf(fb) or fb <= 0:
                        return 0.0
                    return fb
                except (TypeError, ValueError):
                    return 0.0

        price = _finite_price(live.get("price"), base_record.get("current_price", 0))
        _pc_raw = live.get("prev_close")
        try:
            prev_close = float(_pc_raw)
            if math.isnan(prev_close) or math.isinf(prev_close):
                prev_close = None
        except (TypeError, ValueError):
            prev_close = None
        fetched_at  = live["fetched_at"]
        hist        = live["hist"]

        # Refresh headlines on the live cadence (cached ~5m) so projections + ghosts react
        if st.session_state.get("enable_news", True):
            _nd = int(st.session_state.get("news_days", 7))
            news_result = run_news(st.session_state.active_ticker, _nd)
            st.session_state.news_result = news_result
        else:
            news_result = st.session_state.get("news_result")

        # price change
        price_delta     = price - prev_close if prev_close else None
        price_delta_pct = price_delta / prev_close if prev_close else None

        # merge live price into record → projections always reflect current price
        live_record = {**base_record, "current_price": price}

        raw_projections = generate_projections(
            live_record, horizon_days=_horizon, news_result=news_result
        )
        projections = _prepare_live_projections(raw_projections)
        if projections is None:
            st.error(
                raw_projections.get("error")
                or "Projections unavailable (no valid price)."
            )
            return

        fv       = base_record.get("fair_value_weighted")
        bb       = base_record.get("buy_below_price")
        currency = base_record.get("currency", "USD")
        name     = base_record.get("company_name") or st.session_state.active_ticker
        age_s    = int((datetime.now() - fetched_at).total_seconds())

        # ── header ────────────────────────────────────────────────────────
        col_title, col_live, col_signal = st.columns([3, 1, 1])

        with col_title:
            st.markdown(f"## {name} ({st.session_state.active_ticker})")
            sect = base_record.get("sector", "Unknown")
            ind  = base_record.get("industry", "")
            st.markdown(f"*{sect} · {ind}*")
            if projections.get("ml_used"):
                st.caption(f"🤖 ML model active")

        with col_live:
            delta_str = ""
            if price_delta is not None:
                sign = "▲" if price_delta >= 0 else "▼"
                clr  = "#00d4aa" if price_delta >= 0 else "#ff4757"
                delta_str = (
                    f'<span style="color:{clr}">'
                    f'{sign} {abs(price_delta):.2f} ({price_delta_pct:+.2%})'
                    f"</span>"
                )
            st.markdown(
                f"<div style='font-size:2rem; font-weight:bold;'>"
                f"{currency} {price:.2f}</div>"
                f"{delta_str}",
                unsafe_allow_html=True,
            )
            badge_cls = "live-badge" if age_s < 60 else "stale-badge"
            st.markdown(
                f'<div class="{badge_cls}">⟳ updated {age_s}s ago</div>',
                unsafe_allow_html=True,
            )

        with col_signal:
            signal    = projections["signal"]
            sig_class = "bullish" if "BULL" in signal else ("bearish" if "BEAR" in signal else "neutral")
            st.markdown(f'<div class="signal-{sig_class}">{signal}</div>', unsafe_allow_html=True)
            st.markdown(f"Confidence: **{projections['confidence']}**")
            st.markdown(f"Class: **{base_record.get('classification', 'N/A')}**")

        # ── news bar ──────────────────────────────────────────────────────
        if news_result and news_result.get("available"):
            ns   = news_result["sentiment_score"]
            nsig = news_result["signal"]
            clr  = "#00d4aa" if "BULL" in nsig else ("#ff4757" if "BEAR" in nsig else "#ffa502")
            st.markdown(
                f'📰 **News:** <span style="color:{clr}; font-weight:bold;">{nsig}</span> '
                f'(score {ns:+.2f}, {news_result["n_articles"]} articles)',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── probability metrics ────────────────────────────────────────────
        st.subheader("Probability & Expected Returns")
        r1 = st.columns(4)
        with r1[0]:
            st.metric("P(up 5d)", f"{projections['p_up_5d']:.0%}",
                      delta=f"{projections['expected_return_5d']:+.1%} exp.")
        with r1[1]:
            st.metric("P(up 20d)", f"{projections['p_up_20d']:.0%}",
                      delta=f"{projections['expected_return_20d']:+.1%} exp.")
        with r1[2]:
            st.metric("P(up 60d)", f"{projections['p_up_60d']:.0%}",
                      delta=f"{projections['expected_return_60d']:+.1%} exp.")
        with r1[3]:
            st.metric(f"P(up {_horizon}d)", f"{projections['p_up_120d']:.0%}",
                      delta=f"{projections['expected_return_120d']:+.1%} exp.")
        r2 = st.columns(4)
        with r2[0]:
            if fv:
                upside = (fv - price) / price
                st.metric("Fair Value", f"{fv:.2f}", delta=f"{upside:+.0%}")
            else:
                st.metric("Fair Value", "N/A")
        with r2[1]:
            st.metric("Composite Score", f"{projections['composite_score']:+.2f}")
        with r2[2]:
            if bb:
                label = "in buy zone" if price <= bb else "above buy-below"
                st.metric("Buy Below", f"{bb:.2f}", delta=label)
            else:
                st.metric("Buy Below", "N/A")
        with r2[3]:
            st.caption("P(up): LightGBM + rules. Run trainer.py after upgrades — feature schema v3 (long history + SPY regime).")

        st.markdown("---")

        # ── price chart ────────────────────────────────────────────────────
        st.subheader(f"Live Chart & {_horizon}-Day Projections")

        if not hist.empty:
            last_date = hist.index[-1]
            anchor = float(price)

            lstm_out = None
            if candle_sequence_available():
                try:
                    lstm_out = predict_future_ohlc(hist, anchor_close=anchor)
                except Exception:
                    lstm_out = None

            paths = projections["paths"]
            ub_full = projections.get("upper_band")
            lb_full = projections.get("lower_band")
            score = float(projections.get("composite_score") or 0.0)

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
            )

            # Historical — opaque, full chroma (past)
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],   close=hist["Close"],
                name="Price (past)",
                increasing_line_color="#00d4aa",
                decreasing_line_color="#ff4757",
                increasing_fillcolor="#00d4aa",
                decreasing_fillcolor="#ff4757",
            ), row=1, col=1)

            if lstm_out is not None:
                seq_ix, seq_oh = lstm_out
                fig.add_trace(
                    go.Candlestick(
                        x=seq_ix,
                        open=seq_oh[:, 0],
                        high=seq_oh[:, 1],
                        low=seq_oh[:, 2],
                        close=seq_oh[:, 3],
                        name="Forecast (LSTM)",
                        increasing_fillcolor="rgba(155, 200, 235, 0.52)",
                        decreasing_fillcolor="rgba(120, 155, 195, 0.52)",
                        increasing_line_color="rgba(185, 225, 255, 0.88)",
                        decreasing_line_color="rgba(130, 165, 205, 0.88)",
                        whiskerwidth=0.88,
                    ),
                    row=1,
                    col=1,
                )
                targets = projections["targets"]
                wb, w0, wr = _scenario_blend_weights(score)
                blend_long = wb * targets["bull"] + w0 * targets["base"] + wr * targets["bear"]
                st.caption(
                    f"LSTM OHLC sequence: next {len(seq_ix)} trading sessions (trained model). "
                    f"Last predicted close {float(seq_oh[-1, 3]):.2f} {currency}. "
                    f"Blended fundamental {_horizon}d target ≈ {blend_long:.2f} (table below)."
                )
            else:
                (
                    _,
                    future_dates,
                    path_base,
                    path_bear,
                    path_bull,
                    ub_rs,
                    lb_rs,
                ) = _ghost_resampled_series(
                    paths,
                    ub_full,
                    lb_full,
                    last_date,
                    early_trading_days=8,
                )

                path_cons = _consensus_close_path(score, path_bull, path_base, path_bear)
                ub_c, lb_c = _band_around_consensus(path_cons, path_base, ub_rs, lb_rs)

                if ub_c is not None and lb_c is not None and len(future_dates) == len(ub_c):
                    fig.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=ub_c + lb_c[::-1],
                        fill="toself",
                        fillcolor="rgba(190, 155, 75, 0.10)",
                        line=dict(color="rgba(210, 175, 95, 0.22)", width=0),
                        name="1σ Band",
                    ), row=1, col=1)

                gx, gO, gH, gL, gC = _forecast_ghost_candles(
                    future_dates,
                    path_cons,
                    ub_c,
                    lb_c,
                    anchor_open=anchor,
                    hist=hist,
                    news_result=news_result,
                    ticker=st.session_state.active_ticker,
                    scenario="consensus",
                )
                if gx:
                    fig.add_trace(
                        go.Candlestick(
                            x=gx,
                            open=gO,
                            high=gH,
                            low=gL,
                            close=gC,
                            name="Forecast",
                            increasing_fillcolor="rgba(175, 178, 190, 0.48)",
                            decreasing_fillcolor="rgba(145, 148, 162, 0.48)",
                            increasing_line_color="rgba(205, 208, 220, 0.82)",
                            decreasing_line_color="rgba(165, 168, 182, 0.82)",
                            whiskerwidth=0.84,
                        ),
                        row=1,
                        col=1,
                    )

                wb, w0, wr = _scenario_blend_weights(score)
                st.caption(
                    f"Single forecast path — blend of bull / base / bear "
                    f"(weights {wb:.0%} / {w0:.0%} / {wr:.0%} from composite {score:+.2f}). "
                    f"Horizon blended close ≈ {path_cons[-1]:.2f} {currency}."
                )

            # Soft vertical split: last historical session vs forecast zone
            fig.add_vline(
                x=last_date,
                line_width=1,
                line_color="rgba(255, 255, 255, 0.12)",
                row=1,
                col=1,
            )

            if fv:
                fig.add_hline(y=fv, line_dash="dash", line_color="#7c4dff",
                              annotation_text=f"Fair Value {fv:.2f}",
                              annotation_position="top left", row=1, col=1)
            if bb:
                fig.add_hline(y=bb, line_dash="dot", line_color="#00bcd4",
                              annotation_text=f"Buy Below {bb:.2f}",
                              annotation_position="bottom left", row=1, col=1)

            ec = base_record.get("elliott_context") or {}
            if ec.get("available") and isinstance(ec.get("fib_retracement"), dict):
                for fk, fib_px in ec["fib_retracement"].items():
                    fig.add_hline(
                        y=float(fib_px),
                        line_dash="dot",
                        line_color="rgba(160,160,255,0.45)",
                        annotation_text=fk.replace("fib_", ""),
                        annotation_position="left",
                        row=1, col=1,
                    )

            # Live price marker on top of forecast layer
            fig.add_trace(go.Scatter(
                x=[hist.index[-1]], y=[price],
                mode="markers",
                marker=dict(color="#ffffff", size=8, symbol="circle",
                            line=dict(color="#00d4aa", width=2)),
                name="Live Price",
                hovertemplate=f"Live: {currency} {price:.2f}<extra></extra>",
            ), row=1, col=1)

            # Volume
            vol_colors = [
                "#00d4aa" if c >= o else "#ff4757"
                for o, c in zip(hist["Open"], hist["Close"])
            ]
            fig.add_trace(go.Bar(
                x=hist.index, y=hist["Volume"],
                marker_color=vol_colors, name="Volume", showlegend=False,
            ), row=2, col=1)

            fig.update_layout(
                template="plotly_dark", height=650,
                margin=dict(l=50, r=50, t=30, b=30),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1),
                paper_bgcolor="#000000",
                plot_bgcolor="#000000",
            )
            fig.update_xaxes(gridcolor="#141414", row=1, col=1)
            fig.update_xaxes(gridcolor="#141414", row=2, col=1)
            fig.update_yaxes(gridcolor="#141414", title_text="Price", row=1, col=1)
            fig.update_yaxes(gridcolor="#141414", title_text="Vol",   row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No price history available for charting")

        st.markdown("---")

        # ── signal breakdown + targets ─────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Signal Breakdown")
            for factor, score in projections["sub_scores"].items():
                label   = factor.replace("_", " ").title()
                color   = "#00d4aa" if score > 0.1 else ("#ff4757" if score < -0.1 else "#ffa502")
                bar_pct = int((score + 1) / 2 * 100)
                st.markdown(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="color:#ccc;">{label}</span>'
                    f'<span style="color:{color}; float:right;">{score:+.2f}</span>'
                    f'<div style="background:#1a1a2e; border-radius:4px; height:8px; margin-top:4px;">'
                    f'<div style="background:{color}; width:{bar_pct}%; height:100%; border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

        with col_right:
            st.subheader(f"Price Target ({_horizon} days)")
            targets = projections["targets"]
            sc = float(projections.get("composite_score") or 0.0)
            wb, w0, wr = _scenario_blend_weights(sc)
            blend_px = wb * targets["bull"] + w0 * targets["base"] + wr * targets["bear"]
            st.table(pd.DataFrame({
                "":             ["Blended forecast"],
                "Target price": [f"{blend_px:.2f}"],
                "vs live":      [f"{(blend_px - price) / price:+.1%}"],
            }))
            with st.expander("How the blend is built (bull / base / bear envelope)", expanded=False):
                st.markdown(
                    f"Composite **{sc:+.2f}** → weights **bull {wb:.0%} · base {w0:.0%} · bear {wr:.0%}**. "
                    "The chart shows one path; the table lists raw scenario anchors."
                )
                st.table(pd.DataFrame({
                    "Anchor":       ["Bull", "Base", "Bear"],
                    "Target price": [
                        f"{targets['bull']:.2f}",
                        f"{targets['base']:.2f}",
                        f"{targets['bear']:.2f}",
                    ],
                    "vs live": [
                        f"{(targets['bull'] - price) / price:+.1%}",
                        f"{(targets['base'] - price) / price:+.1%}",
                        f"{(targets['bear'] - price) / price:+.1%}",
                    ],
                }))

            st.markdown("**Key Fundamentals:**")
            for label, val in {
                "ROIC":            base_record.get("roic"),
                "Op. Margin":      base_record.get("operating_margin"),
                "Rev CAGR (5Y)":   base_record.get("revenue_cagr_5y"),
                "FCF Yield":       base_record.get("fcf_yield"),
                "Net Debt/EBITDA": base_record.get("net_debt_ebitda"),
                "Beta":            base_record.get("beta"),
            }.items():
                if val is not None:
                    is_pct = any(k in label.lower() for k in ("margin","cagr","yield","roic"))
                    st.text(f"  {label}: {val:.1%}" if is_pct else f"  {label}: {val:.2f}")

        st.markdown("---")
        with st.expander("Technical analysis, Elliott-style structure & trade setup", expanded=False):
            ext = base_record.get("extended_technicals") or {}
            ell = base_record.get("elliott_context") or {}
            ts = base_record.get("trade_setup") or {}
            if not ext.get("available"):
                st.caption(ext.get("reason", "Extended technicals unavailable."))
            else:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("MACD hist", f"{ext['macd']['histogram']:.4f}", delta=ext["macd"]["bias"])
                with m2:
                    st.metric("ADX", f"{ext['adx_14']['adx']:.1f}", delta=ext["adx_14"]["trend_strength"])
                with m3:
                    st.metric("Stoch %K", f"{ext['stochastic_14_3']['k']:.0f}", delta=ext["stochastic_14_3"]["zone"])
                with m4:
                    st.metric(
                        "BB %B",
                        f"{ext['bollinger_20']['percent_b']:.2f}",
                        delta=f"w {ext['bollinger_20']['band_width_pct']:.3f}",
                    )
                r14 = ext.get("rsi_14") or {}
                dc = ext.get("donchian_20") or {}
                volx = ext.get("volume_context") or {}
                m5, m6, m7 = st.columns(3)
                with m5:
                    st.metric("RSI 14", f"{r14.get('value', 0):.1f}", delta=r14.get("zone", ""))
                with m6:
                    st.metric(
                        "Donchian 20",
                        f"{dc.get('close_vs_channel', '')}",
                        help=f"Hi {dc.get('upper')} / Lo {dc.get('lower')}",
                    )
                with m7:
                    rv = volx.get("relative_vs_ma20")
                    st.metric(
                        "Rel vol vs MA20",
                        f"{rv:.2f}" if isinstance(rv, (int, float)) else "—",
                        delta=volx.get("obv_slope_hint") or "",
                    )
                tsm = ext.get("kaufman_tsm") or {}
                if tsm.get("available"):
                    st.markdown("**Kaufman / TSM-style (P. Kaufman, *Trading Systems & Methods*)**")
                    st.caption(tsm.get("disclaimer", ""))
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.metric("Eff. ratio 10d", f"{tsm['efficiency_ratio_10']:.3f}")
                        st.metric("Eff. ratio 20d", f"{tsm['efficiency_ratio_20']:.3f}")
                    with k2:
                        st.metric("Mom 10d (rel)", f"{tsm['momentum_10d_rel']:.4f}")
                        st.metric("LinReg 20d R²", f"{tsm['linreg_20d']['r2']:.3f}")
                    with k3:
                        st.metric("Reg 20d 1d fcst", f"{tsm['linreg_20d']['forecast_1d_return']:+.2%}")
                        st.metric("Combined bias", f"{tsm['combined_bias']:+.2f}")
                    st.caption(f"Direction hint: **{tsm.get('direction_hint', '?')}**")

            qc = base_record.get("ohlcv_quality") or {}
            st.markdown("**OHLCV data QC**")
            if qc.get("ok"):
                st.caption(f"OK — {qc.get('n_bars', 0)} bars checked.")
            else:
                st.warning(" · ".join(qc.get("errors") or ["QC failed"]))
            for w in qc.get("warnings") or []:
                st.caption(f"⚠ {w}")

            ms = base_record.get("market_structure") or {}
            st.markdown("**Market structure (confirmed swings)**")
            if not ms.get("available"):
                st.caption(ms.get("reason", "Unavailable."))
            else:
                st.caption(ms.get("disclaimer", ""))
                st.write(
                    f"Regime hint: **{ms.get('regime_hint', '?')}** · "
                    f"Confirmed pivots: **{ms.get('n_confirmed_pivots', 0)}**"
                )
                if ms.get("recent_swings"):
                    st.dataframe(pd.DataFrame(ms["recent_swings"]), hide_index=True, use_container_width=True)

            st.markdown("**Recent candlestick patterns**")
            cp = base_record.get("candle_patterns") or {}
            if not cp.get("available"):
                st.caption(cp.get("reason", ""))
            else:
                st.write(f"Bias: **{cp.get('candle_bias', '?')}** — {cp.get('summary', '')}")
                an = cp.get("last_bar_anatomy")
                if isinstance(an, dict):
                    st.caption(
                        f"Last bar anatomy: body {an.get('body_pct', 0):.0%} of range · "
                        f"↑wick {an.get('upper_wick_pct', 0):.0%} · ↓wick {an.get('lower_wick_pct', 0):.0%}"
                    )
                for p in (cp.get("patterns") or [])[:5]:
                    st.caption(f"- {p}")

            st.markdown("**Elliott-style context (heuristic)**")
            if not ell.get("available"):
                st.caption(ell.get("reason", "Elliott context unavailable."))
            else:
                st.caption(ell.get("disclaimer", ""))
                st.write(
                    f"Direction: **{ell.get('dominant_direction', '?')}** · "
                    f"Leg high/low: **{ell.get('last_leg_high')}** / **{ell.get('last_leg_low')}**"
                )
                st.write(ell.get("structure_hint", ""))
                fib = ell.get("fib_retracement") or {}
                if fib:
                    st.dataframe(
                        pd.DataFrame([{"level": k, "price": v} for k, v in fib.items()]),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.markdown("**Trade setup (journal-style)**")
            st.caption(
                "Strategy robustness: run `python backtesting/strategy_stat_tests.py TICKER` "
                "for Donchian-style permutation / walk-forward diagnostics; "
                "`python backtesting/run_vector_backtest.py TICKER --strategy sma|donchian|bollinger` "
                "for next-open vector P/L with bps costs."
            )
            if not ts.get("available"):
                st.caption(ts.get("reason", ""))
            else:
                st.write(ts.get("bias_summary", ""))
                if ts.get("candle_note"):
                    st.write(f"Candles: {ts['candle_note']}")
                if ts.get("elliott_note"):
                    st.write(ts["elliott_note"])
                st.caption(ts.get("risk_notes", ""))
                wl = ts.get("watch_levels") or []
                if wl:
                    st.dataframe(pd.DataFrame(wl), hide_index=True, use_container_width=True)

        st.markdown("---")

        # ── news detail ────────────────────────────────────────────────────
        if news_result and news_result.get("available") and news_result.get("articles"):
            with st.expander(f"📰 News Detail ({news_result['n_articles']} articles)", expanded=False):
                HIGH = {"earnings", "guidance", "merger", "fda", "sec"}
                for a in news_result["articles"]:
                    score    = a.get("final_score", 0.0)
                    card_cls = "news-pos" if score > 0.1 else ("news-neg" if score < -0.1 else "news-card")
                    badge    = " 🔴 HIGH IMPACT" if a["category"] in HIGH else ""
                    reasoning = a.get("claude_reasoning", "")
                    st.markdown(
                        f'<div class="news-card {card_cls}">'
                        f"<b>{a['title']}</b>{badge}<br>"
                        f'<small style="color:#888;">'
                        f"{a['publisher']} · {a['published_at'][:10]} · "
                        f"category: {a['category']} · score: {score:+.2f}"
                        f"</small>"
                        + (f'<br><small style="color:#aaa;">{reasoning}</small>' if reasoning else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )

        # ── classification & red flags ─────────────────────────────────────
        col_clf, col_flags = st.columns(2)

        with col_clf:
            st.subheader("Classification")
            clf = base_record.get("classification_result", {})
            st.markdown(f"**Verdict: {base_record.get('classification', 'N/A')}**")
            if clf.get("reasons_for"):
                st.markdown("**Positives:**")
                for r in clf["reasons_for"][:5]:
                    st.markdown(f"- ✅ {r}")
            if clf.get("reasons_against"):
                st.markdown("**Risks:**")
                for r in clf["reasons_against"][:5]:
                    st.markdown(f"- ⚠️ {r}")

        with col_flags:
            st.subheader("Red Flags")
            red_flags = base_record.get("red_flags") or []
            if red_flags:
                for f in red_flags:
                    sev  = f.get("severity", "MEDIUM")
                    icon = "🔴" if sev == "HIGH" else "🟡"
                    st.markdown(f"{icon} **{f['pattern']}** ({sev})")
                    if f.get("detail"):
                        st.caption(f"  {f['detail']}")
            else:
                st.success("No red flags detected")

            critical = base_record.get("critical_flags") or []
            if critical:
                st.error("Critical Flags:")
                for c in critical:
                    st.markdown(f"🚨 {c}")

    live_dashboard()


# ── landing page ───────────────────────────────────────────────────────────────

else:
    st.markdown("# 📈 Stock Projection Dashboard")
    st.markdown("""
    Enter a ticker and click **🚀 Run Analysis**.

    The dashboard runs heavy fundamental analysis **once** (cached 1 hour),
    then keeps the **live price, chart and projections updating automatically**
    every 30 seconds without re-running the full pipeline.

    ---

    | Layer | Refresh | What it does |
    |---|---|---|
    | Fundamentals | On demand (1h cache) | DCF valuation, quality, risk, classification |
    | Live price & chart | Every 30 s | Current price from yfinance fast_info |
    | Projections | Every 30 s | Recomputed with live price each time |
    | News sentiment | Every 5 min | FinBERT on all articles, Claude on key events |

    ### Train the ML model
    ```bash
    python projection/ml_model/trainer.py
    ```

    *yfinance data has ~15 min delay. Not financial advice.*
    """)
