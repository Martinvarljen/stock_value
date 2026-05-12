"""
Stock Projection Dashboard — Streamlit + Plotly

Architecture
────────────
  Heavy  : run_fundamentals()  valuation, quality, risk, classification
           → @st.cache_data(ttl=3600)  run once, cached 1 hour
  Live   : get_live_data()     current price + 1Y OHLCV chart
           → @st.cache_data(ttl=30)   refreshed every 30 s automatically
  News   : run_news()          FinBERT + Claude headline sentiment
           → @st.cache_data(ttl=300)  refreshed every 5 min

The live section uses @st.fragment(run_every=30) so the price, chart, and
projections update automatically without touching the heavy pipeline.

Run with:
    streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path
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

from projection_engine import generate_projections
from news_engine import analyze_news
from ml_model.predictor import models_available, model_summary

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
        st.warning("⚙️ No ML model — rule-based")
        with st.expander("Train ML model"):
            st.code("python projection/ml_model/trainer.py", language="bash")

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
    }

    clf = classify_stock(record)
    record["classification"]        = clf["classification"]
    record["classification_result"] = clf

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


# ── live dashboard (auto-refreshes) ───────────────────────────────────────────

if (
    st.session_state.get("base_record") is not None
    and st.session_state.get("active_ticker") == ticker
):

    @st.fragment(run_every=st.session_state.get("refresh_sec", 30))
    def live_dashboard():
        base_record = st.session_state.base_record
        news_result = st.session_state.get("news_result")
        _horizon    = st.session_state.get("horizon", 120)

        # ── get live price ─────────────────────────────────────────────────
        live        = get_live_data(st.session_state.active_ticker)
        price       = live["price"] or base_record.get("current_price", 0)
        prev_close  = live["prev_close"]
        fetched_at  = live["fetched_at"]
        hist        = live["hist"]

        # price change
        price_delta     = price - prev_close if prev_close else None
        price_delta_pct = price_delta / prev_close if prev_close else None

        # merge live price into record → projections always reflect current price
        live_record = {**base_record, "current_price": price}

        projections = generate_projections(
            live_record, horizon_days=_horizon, news_result=news_result
        )

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
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            st.metric("P(up 20d)",  f"{projections['p_up_20d']:.0%}",
                      delta=f"{projections['expected_return_20d']:+.1%} exp.")
        with c2:
            st.metric("P(up 60d)",  f"{projections['p_up_60d']:.0%}",
                      delta=f"{projections['expected_return_60d']:+.1%} exp.")
        with c3:
            st.metric(f"P(up {_horizon}d)", f"{projections['p_up_120d']:.0%}",
                      delta=f"{projections['expected_return_120d']:+.1%} exp.")
        with c4:
            if fv:
                upside = (fv - price) / price
                st.metric("Fair Value", f"{fv:.2f}", delta=f"{upside:+.0%}")
            else:
                st.metric("Fair Value", "N/A")
        with c5:
            st.metric("Composite Score", f"{projections['composite_score']:+.2f}")
        with c6:
            if bb:
                label = "IN ZONE ✓" if price <= bb else "above"
                st.metric("Buy Below", f"{bb:.2f}", delta=label)
            else:
                st.metric("Buy Below", "N/A")

        st.markdown("---")

        # ── price chart ────────────────────────────────────────────────────
        st.subheader(f"Live Chart & {_horizon}-Day Projections")

        if not hist.empty:
            last_date    = hist.index[-1]
            paths        = projections["paths"]
            future_dates = [
                last_date + timedelta(days=int(d * 365 / 252))
                for d in paths["days"]
            ]

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.8, 0.2],
            )

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"],   close=hist["Close"],
                name="Price",
                increasing_line_color="#00d4aa",
                decreasing_line_color="#ff4757",
            ), row=1, col=1)

            # Live price dot on last candle
            fig.add_trace(go.Scatter(
                x=[hist.index[-1]], y=[price],
                mode="markers",
                marker=dict(color="#ffffff", size=8, symbol="circle",
                            line=dict(color="#00d4aa", width=2)),
                name="Live Price",
                hovertemplate=f"Live: {currency} {price:.2f}<extra></extra>",
            ), row=1, col=1)

            # Projection paths
            for path_key, color, dash, width in [
                ("bull", "#00d4aa", "dot",   2.0),
                ("base", "#ffa502", "solid", 2.5),
                ("bear", "#ff4757", "dot",   2.0),
            ]:
                fig.add_trace(go.Scatter(
                    x=future_dates, y=paths[path_key],
                    mode="lines",
                    name=f"{path_key.title()} Path",
                    line=dict(color=color, width=width, dash=dash),
                ), row=1, col=1)

            # Volatility band
            fig.add_trace(go.Scatter(
                x=future_dates + future_dates[::-1],
                y=projections["upper_band"] + projections["lower_band"][::-1],
                fill="toself",
                fillcolor="rgba(255,165,2,0.08)",
                line=dict(color="rgba(255,165,2,0.2)", width=0),
                name="1σ Band",
            ), row=1, col=1)

            if fv:
                fig.add_hline(y=fv, line_dash="dash", line_color="#7c4dff",
                              annotation_text=f"Fair Value {fv:.2f}",
                              annotation_position="top left", row=1, col=1)
            if bb:
                fig.add_hline(y=bb, line_dash="dot", line_color="#00bcd4",
                              annotation_text=f"Buy Below {bb:.2f}",
                              annotation_position="bottom left", row=1, col=1)

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
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
            )
            fig.update_xaxes(gridcolor="#1a1a2e", row=1, col=1)
            fig.update_yaxes(gridcolor="#1a1a2e", title_text="Price", row=1, col=1)
            fig.update_yaxes(gridcolor="#1a1a2e", title_text="Vol",   row=2, col=1)

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
            st.subheader(f"Price Targets ({_horizon} days)")
            targets = projections["targets"]
            st.table(pd.DataFrame({
                "Scenario":     ["🟢 Bull", "🟡 Base", "🔴 Bear"],
                "Target Price": [f"{targets['bull']:.2f}",
                                 f"{targets['base']:.2f}",
                                 f"{targets['bear']:.2f}"],
                "Return": [
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
