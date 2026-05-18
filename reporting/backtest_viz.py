"""Interactive HTML report: NAV vs SPY, position map, trade log."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def closed_trades_from_ledger(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair entries with exits into round-trip rows."""
    open_lots: dict[tuple[str, str], dict] = {}
    closed: list[dict[str, Any]] = []
    for row in sorted(ledger, key=lambda r: (r.get("date", ""), r.get("ticker", ""))):
        tk = str(row.get("ticker", "")).upper()
        side = row.get("side") or ("long" if "LONG" in str(row.get("action", "")) else "short")
        key = (tk, side)
        act = row.get("action", "")
        if act in ("ENTER_LONG", "ENTER_SHORT"):
            open_lots[key] = row
        elif act == "EXIT" and key in open_lots:
            ent = open_lots.pop(key)
            d0 = date.fromisoformat(str(ent["date"])[:10])
            d1 = date.fromisoformat(str(row["date"])[:10])
            closed.append(
                {
                    "ticker": tk,
                    "side": side,
                    "entry_date": str(ent["date"])[:10],
                    "exit_date": str(row["date"])[:10],
                    "hold_days": (d1 - d0).days,
                    "entry_price": ent.get("price"),
                    "exit_price": row.get("price"),
                    "notional": ent.get("notional"),
                    "pnl_pct": row.get("pnl_pct"),
                    "entry_reason": ent.get("reason", ""),
                    "exit_reason": row.get("reason", ""),
                    "p_up_20d_at_entry": ent.get("p_up_20d"),
                }
            )
    return closed


def _weekly_position_matrix(
    snapshots: list[dict[str, Any]],
    tickers: list[str],
) -> tuple[list, list, np.ndarray]:
    """Build heatmap matrix (tickers × week dates): 1 long, -1 short, 0 flat."""
    if not snapshots:
        return [], [], np.array([])
    df = pd.DataFrame(snapshots)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    weeks = pd.date_range(df.index.min(), df.index.max(), freq="W-FRI")
    tk_list = sorted(tickers)
    mat = np.zeros((len(tk_list), len(weeks)), dtype=float)
    tk_ix = {t: i for i, t in enumerate(tk_list)}
    for wi, w in enumerate(weeks):
        sub = df[df.index <= w]
        if sub.empty:
            continue
        row = sub.iloc[-1]
        for p in row.get("positions") or []:
            t = str(p.get("ticker", "")).upper()
            if t not in tk_ix:
                continue
            side = p.get("side", "long")
            mat[tk_ix[t], wi] = 1.0 if side == "long" else -1.0
    return tk_list, [w.strftime("%Y-%m-%d") for w in weeks], mat


def write_backtest_report(
    *,
    curve: pd.DataFrame,
    ledger: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    summary: dict[str, Any],
    out_html: Path,
    out_json: Path | None = None,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_html.parent.mkdir(parents=True, exist_ok=True)
    closed = closed_trades_from_ledger(ledger)
    traded_tickers = sorted({str(t["ticker"]).upper() for t in closed} | {str(t["ticker"]).upper() for t in ledger})

    curve = curve.copy()
    if not isinstance(curve.index, pd.DatetimeIndex):
        curve.index = pd.to_datetime(curve.index)

    cap = summary.get("initial_capital")
    dollar_mode = cap is not None and float(cap) > 0
    if dollar_mode:
        cap = float(cap)
        curve = curve.assign(
            strategy=curve["strategy"] * cap,
            spy_bh=curve["spy_bh"] * cap,
        )
    growth_label = f"growth of ${cap:,.0f}" if dollar_mode else "growth of $1"

    # ── Figure 1: NAV + trade markers ─────────────────────────────────────
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.38, 0.37, 0.25],
        subplot_titles=(
            f"Strategy vs SPY ({growth_label})",
            "Position map (weekly) — green long, red short",
            "Open position count",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve["strategy"],
            name="Agent",
            line=dict(color="#2ecc71", width=2),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve["spy_bh"],
            name="SPY B&H",
            line=dict(color="#3498db", width=2),
        ),
        row=1,
        col=1,
    )

    def _nav_at(dstr: str) -> float | None:
        try:
            ts = pd.Timestamp(dstr)
            if ts in curve.index:
                return float(curve.loc[ts, "strategy"])
            pos = curve.index.searchsorted(ts, side="right") - 1
            if pos >= 0:
                return float(curve["strategy"].iloc[pos])
        except Exception:
            pass
        return None

    long_ent = [t for t in ledger if t.get("action") == "ENTER_LONG"]
    short_ent = [t for t in ledger if t.get("action") == "ENTER_SHORT"]
    exits = [t for t in ledger if t.get("action") == "EXIT"]

    if long_ent:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in long_ent],
                y=[_nav_at(str(t["date"])[:10]) for t in long_ent],
                mode="markers",
                name="Long entry",
                marker=dict(symbol="triangle-up", size=10, color="#2ecc71"),
                text=[f"{t['ticker']}<br>{t.get('reason','')}" for t in long_ent],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if short_ent:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in short_ent],
                y=[_nav_at(str(t["date"])[:10]) for t in short_ent],
                mode="markers",
                name="Short entry",
                marker=dict(symbol="triangle-down", size=10, color="#e74c3c"),
                text=[f"{t['ticker']}<br>{t.get('reason','')}" for t in short_ent],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if exits:
        fig.add_trace(
            go.Scatter(
                x=[t["date"] for t in exits],
                y=[_nav_at(str(t["date"])[:10]) for t in exits],
                mode="markers",
                name="Exit",
                marker=dict(symbol="x", size=9, color="#f1c40f"),
                text=[
                    f"{t['ticker']} PnL {float(t.get('pnl_pct', 0) or 0):+.1%}<br>{t.get('reason','')}"
                    for t in exits
                ],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # ── Heatmap ───────────────────────────────────────────────────────────
    y_tk, x_wk, mat = _weekly_position_matrix(snapshots, traded_tickers or [])
    if mat.size:
        fig.add_trace(
            go.Heatmap(
                z=mat,
                x=x_wk,
                y=y_tk,
                colorscale=[
                    [0.0, "#1a1a2e"],
                    [0.45, "#1a1a2e"],
                    [0.5, "#2ecc71"],
                    [0.55, "#e74c3c"],
                    [1.0, "#e74c3c"],
                ],
                zmid=0,
                showscale=True,
                colorbar=dict(title="Position", len=0.35, y=0.35),
                hovertemplate="%{y} @ %{x}<br>side: %{z}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # ── Position count ────────────────────────────────────────────────────
    n_open = [len(s.get("positions") or []) for s in snapshots]
    snap_dates = [s["date"] for s in snapshots]
    fig.add_trace(
        go.Scatter(
            x=snap_dates,
            y=n_open,
            name="# open",
            fill="tozeroy",
            line=dict(color="#9b59b6"),
        ),
        row=3,
        col=1,
    )

    title = (
        f"Agent backtest {summary.get('from')} → {summary.get('to')} | "
        f"Strat {summary.get('strategy_cagr', 0):.1%} vs SPY {summary.get('spy_cagr', 0):.1%} | "
        f"{len(closed)} closed trades"
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=1100,
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
    )
    fig.update_yaxes(title_text="Portfolio ($)" if dollar_mode else "NAV", row=1, col=1)
    fig.update_yaxes(title_text="Ticker", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    # ── Animated NAV (monthly frames) ───────────────────────────────────
    monthly = curve.resample("ME").last().dropna()
    if len(monthly) > 1:
        anim_rows: list[dict] = []
        for i in range(len(monthly)):
            sub = monthly.iloc[: i + 1]
            for idx, r in sub.iterrows():
                anim_rows.append({"month": idx, "nav": r["strategy"], "series": "Agent", "frame": i})
                anim_rows.append({"month": idx, "nav": r["spy_bh"], "series": "SPY B&H", "frame": i})
        pdf = pd.DataFrame(anim_rows)
        import plotly.express as px

        fig_anim = px.line(
            pdf,
            x="month",
            y="nav",
            color="series",
            animation_frame="frame",
            range_y=[0, float(max(pdf["nav"].max(), 1.05))],
            title="Animated: strategy vs SPY (monthly)",
            template="plotly_dark",
        )
        fig_anim.update_layout(height=500)
        anim_html = fig_anim.to_html(full_html=False, include_plotlyjs=False)
    else:
        anim_html = ""

    # ── Trade table HTML ──────────────────────────────────────────────────
    def _row_html(t: dict) -> str:
        pnl = t.get("pnl_pct")
        pnl_s = f"{float(pnl):+.1%}" if pnl is not None else "—"
        pu = t.get("p_up_20d_at_entry")
        pu_s = f"{float(pu):.0%}" if pu is not None else "—"
        return (
            f"<tr><td>{t.get('ticker')}</td><td>{t.get('side')}</td>"
            f"<td>{t.get('entry_date')}</td><td>{t.get('exit_date')}</td>"
            f"<td>{t.get('hold_days')}</td><td>{pnl_s}</td><td>{pu_s}</td>"
            f"<td>{t.get('entry_reason', '')[:60]}</td>"
            f"<td>{t.get('exit_reason', '')[:60]}</td></tr>"
        )

    table_rows = "".join(_row_html(t) for t in closed[:500])
    if not table_rows:
        table_rows = "<tr><td colspan='9'>No closed trades in period.</td></tr>"

    cagr_s = summary.get("strategy_cagr")
    cagr_b = summary.get("spy_cagr")
    cagr_s_s = f"{cagr_s:.1%}" if cagr_s is not None else "n/a"
    cagr_b_s = f"{cagr_b:.1%}" if cagr_b is not None else "n/a"
    uni_block = ""
    pit_warn = summary.get("pit_warning")
    if summary.get("universe_source") or pit_warn or summary.get("survivorship_bias_note"):
        uni_block = (
            f"<br><b>Universe</b>: {summary.get('universe_source', 'legacy')} — "
            f"{summary.get('universe_description', '')}"
        )
        if pit_warn:
            uni_block += f"<br><span style='color:#f39c12'>⚠ {pit_warn}</span>"
        note = summary.get("survivorship_bias_note")
        if note:
            uni_block += f"<br><span style='color:#aaa;font-size:12px'>{note}</span>"
    prof_block = ""
    if summary.get("profile"):
        lev_l = summary.get("long_leverage", summary.get("cfd_leverage"))
        lev_s = summary.get("short_leverage", summary.get("cfd_leverage"))
        prof_block = (
            f"<br>Profile: <b>{summary.get('profile')}</b> &nbsp;|&nbsp; "
            f"Leverage: long <b>{lev_l}x</b> / short <b>{lev_s}x</b> &nbsp;|&nbsp; "
            f"Config hash: <code>{summary.get('config_fingerprint', '—')}</code>"
        )
    inv_block = ""
    if summary.get("invariant_errors"):
        inv_block = (
            "<br><span style='color:#e74c3c'>Invariant issues: "
            + "; ".join(summary["invariant_errors"][:3])
            + "</span>"
        )
    elif summary.get("invariants_ok"):
        inv_block = "<br><span style='color:#2ecc71'>Invariants: OK</span>"

    cap_block = ""
    if dollar_mode:
        fs = summary.get("final_strategy_usd")
        fb = summary.get("final_spy_usd")
        tr_s = summary.get("strategy_total_return")
        tr_b = summary.get("spy_total_return")
        tr_s_s = f"{tr_s:.1%}" if tr_s is not None else "n/a"
        tr_b_s = f"{tr_b:.1%}" if tr_b is not None else "n/a"
        cap_block = (
            f"<br>Starting capital: <b>${cap:,.0f}</b><br>"
            f"Ending value: strategy <b>${fs:,.2f}</b> &nbsp;|&nbsp; SPY <b>${fb:,.2f}</b><br>"
            f"Total return: strategy <b>{tr_s_s}</b> &nbsp;|&nbsp; SPY <b>{tr_b_s}</b>"
        )
    main_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="main-charts")
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Agent backtest report</title>
<style>
  body {{ background:#0f0f1a; color:#ddd; margin:0; padding:12px; font-family:system-ui,sans-serif; }}
  h2 {{ margin:24px 0 8px; color:#2ecc71; }}
  table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  th, td {{ border:1px solid #333; padding:6px 8px; text-align:left; }}
  th {{ background:#1a1a2e; }}
  tr:nth-child(even) {{ background:#1a1a28; }}
</style></head><body>
<h1>Portfolio agent backtest</h1>
<div style="font-family:system-ui;color:#eee;background:#16213e;padding:16px;margin:12px 0;border-radius:8px;">
  <b>Summary</b><br>
  Period: {summary.get('from')} → {summary.get('to')} ({summary.get('years')} y)<br>
  Strategy CAGR: {cagr_s_s} &nbsp;|&nbsp; SPY: {cagr_b_s}<br>
  Max DD: strategy {summary.get('strategy_max_dd', 0):.1%} &nbsp;|&nbsp; SPY {summary.get('spy_max_dd', 0):.1%}<br>
  Trades: {len(ledger)} events, {len(closed)} round-trips &nbsp;|&nbsp;
  Beat SPY: {'yes' if summary.get('beat_spy') else 'no'}{prof_block}{uni_block}{inv_block}{cap_block}
</div>
{main_html}
<h2>Animated equity curve</h2>
{anim_html}
<h2>Closed trades (entry → exit)</h2>
<table>
<thead><tr>
<th>Ticker</th><th>Side</th><th>Entry</th><th>Exit</th><th>Days</th><th>PnL</th><th>P(up)@entry</th>
<th>Entry reason</th><th>Exit reason</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
<p style="color:#888;font-size:12px;">Heatmap: green = long, red = short. NAV markers: ▲ long, ▼ short, ✕ exit.</p>
</body></html>"""

    out_html.write_text(page, encoding="utf-8")

    if out_json:
        payload = {
            "summary": summary,
            "ledger": ledger,
            "closed_trades": closed,
            "snapshots_sample": snapshots[:: max(1, len(snapshots) // 200)],
        }
        out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
