"""Branching pipeline + position swimlane HTML (alpha-discovery style)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from portfolio.backtest_viz import closed_trades_from_ledger


# Fixed pipeline topology (edges used for layout)
PIPELINE_EDGES = [
    ("universe", "data"),
    ("data", "ml"),
    ("data", "skip"),
    ("ml", "quintile"),
    ("ml", "skip"),
    ("quintile", "regime"),
    ("quintile", "skip"),
    ("regime", "committee"),
    ("regime", "skip"),
    ("committee", "stop_loss"),
    ("committee", "take_profit"),
    ("committee", "time_exit"),
    ("committee", "signal_exit"),
    ("stop_loss", "action"),
    ("take_profit", "action"),
    ("time_exit", "action"),
    ("signal_exit", "action"),
    ("committee", "action"),
    ("risk", "skip"),
]

PIPELINE_LABELS = {
    "universe": "Universe scan",
    "data": "Data / ML",
    "ml": "ML engine",
    "quintile": "Quintile rank",
    "regime": "Regime filter",
    "committee": "Entry committee",
    "action": "Trade action",
    "risk": "Risk gate",
    "stop_loss": "Stop loss",
    "take_profit": "Take profit",
    "time_exit": "Max hold (time)",
    "signal_exit": "ML / risk exit",
    "skip": "No trade",
}


def _position_bars(closed: list[dict], ledger: list[dict]) -> list[dict]:
    """Gantt-style segments for swimlane chart."""
    bars: list[dict] = []
    for t in closed:
        bars.append(
            {
                "ticker": t["ticker"],
                "side": t["side"],
                "start": t["entry_date"],
                "end": t["exit_date"],
                "pnl_pct": t.get("pnl_pct"),
                "entry_reason": (t.get("entry_reason") or "")[:80],
                "exit_reason": (t.get("exit_reason") or "")[:80],
            }
        )
    open_entries = [r for r in ledger if r.get("action", "").startswith("ENTER")]
    exit_keys = {(str(t["ticker"]).upper(), t.get("side")) for t in closed}
    for e in open_entries:
        key = (str(e["ticker"]).upper(), e.get("side"))
        if key not in exit_keys:
            bars.append(
                {
                    "ticker": e["ticker"],
                    "side": e.get("side", "long"),
                    "start": str(e["date"])[:10],
                    "end": None,
                    "pnl_pct": None,
                    "entry_reason": (e.get("reason") or "")[:80],
                    "exit_reason": "still open",
                }
            )
    return bars


def _agent_scores(trace: dict | None) -> list[dict]:
    """Pseudo-agent panel scores for detail view (mirrors hedgefund UI)."""
    if not trace:
        return []
    p_up = trace.get("p_up_20d")
    q = trace.get("quintile")
    scale = trace.get("regime_scale", 1.0)
    agents = [
        {
            "name": "ML model",
            "score": int((p_up or 0) * 100),
            "verdict": "PASS" if (p_up or 0) >= 0.58 else ("HOLD" if (p_up or 0) >= 0.45 else "FAIL"),
        },
        {
            "name": "Quintile rank",
            "score": int((q or 3) * 20),
            "verdict": "PASS" if (q or 0) >= 4 else ("FAIL" if (q or 3) <= 2 else "HOLD"),
        },
        {
            "name": "Regime (SPY)",
            "score": int(scale * 100),
            "verdict": "PASS" if scale >= 0.99 else ("HOLD" if scale >= 0.35 else "FAIL"),
        },
        {
            "name": "Risk gate",
            "score": 100 if not trace.get("critical_flags") else 0,
            "verdict": "PASS" if not trace.get("critical_flags") else "FAIL",
        },
    ]
    return agents


def write_flow_map_html(
    *,
    traces: list[dict[str, Any]],
    ledger: list[dict[str, Any]],
    summary: dict[str, Any],
    out_html: Path,
) -> None:
    closed = closed_trades_from_ledger(ledger)
    bars = _position_bars(closed, ledger)

    # Aggregate edge counts from traces
    edge_counts: dict[str, int] = {}
    for tr in traces:
        path = tr.get("path") or []
        for i in range(len(path) - 1):
            key = f"{path[i]}->{path[i+1]}"
            edge_counts[key] = edge_counts.get(key, 0) + 1

    feed = [t for t in traces if t.get("action") not in (None, "NO_TRADE", "HOLD")][-80:]
    feed.reverse()

    payload = {
        "summary": summary,
        "traces": traces[-1500:],
        "feed": feed,
        "closed": closed[:300],
        "bars": bars,
        "edge_counts": edge_counts,
        "pipeline_labels": PIPELINE_LABELS,
        "pipeline_edges": PIPELINE_EDGES,
    }

    data_json = json.dumps(payload, default=str)
    html = _sanitize_html(_HTML_TEMPLATE.replace("__DATA_JSON__", data_json))
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html, encoding="utf-8")


def _sanitize_html(html: str) -> str:
    """Fix legacy typo tags and broken feed markup."""
    html = html.replace("<motion", "<div").replace("</motion>", "</div>")
    html = html.replace("</span></div>", "</span>")
    html = html.replace('<motion class="tk-big">', '<div class="tk-big">')
    html = html.replace(
        'class="seg ${b.side}" style="left:',
        'class="seg ${b.side}" data-ticker="${b.ticker}" data-start="${b.start}" style="left:',
    )
    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Agent pipeline map</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
<style>
  :root {
    --bg: #0a0e17; --panel: #12182a; --border: #1e2a45;
    --green: #00e676; --red: #ff5252; --blue: #448aff; --muted: #8892a8;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: #e0e6f0; font-family: 'Segoe UI', system-ui, sans-serif; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  header h1 { margin: 0; font-size: 1.1rem; letter-spacing: 0.06em; color: var(--green); }
  header .sub { color: var(--muted); font-size: 0.85rem; }
  .grid { display: grid; grid-template-columns: 1.1fr 0.75fr 0.85fr; gap: 10px; padding: 10px; height: calc(100vh - 56px); }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; display: flex; flex-direction: column; overflow: hidden; }
  .panel h2 { margin: 0; padding: 10px 14px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); border-bottom: 1px solid var(--border); }
  #cy { flex: 1; min-height: 320px; }
  .feed { flex: 1; overflow-y: auto; padding: 8px; }
  .feed-item { padding: 10px 12px; margin-bottom: 6px; border-radius: 8px; background: #0d1220; border-left: 3px solid var(--blue); cursor: pointer; font-size: 0.85rem; }
  .feed-item.long { border-left-color: var(--green); }
  .feed-item.short { border-left-color: var(--red); }
  .feed-item.exit { border-left-color: #ffc107; }
  .feed-item:hover, .feed-item.active { background: #1a2540; }
  .feed-item .tk { font-weight: 700; color: #fff; }
  .feed-item .meta { color: var(--muted); font-size: 0.75rem; margin-top: 4px; }
  .detail { flex: 1; overflow-y: auto; padding: 12px; }
  .detail .tk-big { font-size: 1.4rem; font-weight: 700; color: var(--green); }
  .agent { margin: 12px 0; }
  .agent .row { display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; }
  .bar-bg { height: 6px; background: #1e2a45; border-radius: 3px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 3px; }
  .verdict { font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; }
  .verdict.PASS { background: #1b4332; color: var(--green); }
  .verdict.HOLD { background: #3d3200; color: #ffc107; }
  .verdict.FAIL { background: #3d1515; color: var(--red); }
  #swimlane { flex: 1; min-height: 200px; overflow: auto; padding: 8px; }
  .lane { display: flex; align-items: center; margin-bottom: 6px; font-size: 0.75rem; }
  .lane .name { width: 52px; color: var(--muted); flex-shrink: 0; }
  .lane .track { flex: 1; height: 18px; background: #0d1220; border-radius: 4px; position: relative; }
  .seg { position: absolute; height: 100%; border-radius: 4px; opacity: 0.9; cursor: pointer; }
  .seg.long { background: linear-gradient(90deg, #00c853, #69f0ae); }
  .seg.short { background: linear-gradient(90deg, #d50000, #ff5252); }
  .hint { padding: 8px 14px; font-size: 0.72rem; color: var(--muted); border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <div>
    <h1>PORTFOLIO AGENT — PIPELINE MAP</h1>
    <span class="sub">Branching decision path · click a trade to highlight its route</span>
  </div>
  <div class="sub" id="hdr-stats"></div>
</header>
<div class="grid">
  <div class="panel">
    <h2>Decision pipeline</h2>
    <div id="cy"></div>
    <div class="hint">Width = how many tickers flowed through that branch</div>
  </div>
  <div class="panel">
    <h2>Trade feed</h2>
    <div class="feed" id="feed"></div>
    <h2 style="border-top: 1px solid var(--border)">Position timeline</h2>
    <div id="swimlane"></div>
  </div>
  <div class="panel">
    <h2>Selected trade</h2>
    <div class="detail" id="detail"><p style="color:var(--muted)">Click a trade in the feed or a bar in the timeline.</p></div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;

function initHeader() {
  const s = DATA.summary || {};
  document.getElementById('hdr-stats').textContent =
    `${s.from || ''} → ${s.to || ''} | Strat ${((s.strategy_cagr||0)*100).toFixed(1)}% vs SPY ${((s.spy_cagr||0)*100).toFixed(1)}% | ${(DATA.closed||[]).length} round-trips`;
}

function buildCy() {
  const labels = DATA.pipeline_labels || {};
  const nodeIds = new Set();
  (DATA.pipeline_edges || []).forEach(([a,b]) => { nodeIds.add(a); nodeIds.add(b); });
  const elements = [];
  nodeIds.forEach(id => {
    elements.push({ data: { id, label: labels[id] || id } });
  });
  const ec = DATA.edge_counts || {};
  (DATA.pipeline_edges || []).forEach(([src, tgt]) => {
    const key = src + '->' + tgt;
    const w = ec[key] || 0;
    elements.push({ data: { id: key, source: src, target: tgt, weight: Math.max(1, w), label: w ? String(w) : '' } });
  });
  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    style: [
      { selector: 'node', style: {
        'label': 'data(label)', 'text-valign': 'center', 'text-halign': 'center',
        'font-size': '10px', 'color': '#e0e6f0', 'text-wrap': 'wrap', 'text-max-width': '90px',
        'background-color': '#1a2540', 'border-width': 2, 'border-color': '#448aff',
        'width': 90, 'height': 50, 'shape': 'roundrectangle'
      }},
      { selector: 'node.highlight', style: { 'border-color': '#00e676', 'background-color': '#0d3320' } },
      { selector: 'edge', style: {
        'width': 'mapData(weight, 1, 200, 1, 8)', 'line-color': '#2a3a5c',
        'target-arrow-color': '#2a3a5c', 'target-arrow-shape': 'triangle',
        'curve-style': 'bezier', 'label': 'data(label)', 'font-size': '9px', 'color': '#8892a8'
      }},
      { selector: 'edge.highlight', style: { 'line-color': '#00e676', 'target-arrow-color': '#00e676', 'width': 4 } }
    ],
    layout: { name: 'dagre', rankDir: 'LR', nodeSep: 40, edgeSep: 20, rankSep: 70 }
  });
  return cy;
}

function highlightPath(cy, path) {
  cy.elements().removeClass('highlight');
  if (!path || !path.length) return;
  for (let i = 0; i < path.length - 1; i++) {
    const eid = path[i] + '->' + path[i+1];
    cy.getElementById(eid).addClass('highlight');
    cy.getElementById(path[i]).addClass('highlight');
    cy.getElementById(path[i+1]).addClass('highlight');
  }
  if (path.length) cy.getElementById(path[path.length-1]).addClass('highlight');
}

function renderFeed(cy) {
  const el = document.getElementById('feed');
  el.innerHTML = '';
  const items = DATA.feed && DATA.feed.length ? DATA.feed : (DATA.traces || []).filter(t => ['ENTER_LONG','ENTER_SHORT','EXIT'].includes(t.action)).slice(-60).reverse();
  items.forEach((t, idx) => {
    const div = document.createElement('div');
    const cls = t.action === 'EXIT' ? 'exit' : (t.action === 'ENTER_SHORT' ? 'short' : 'long');
    const exitTag = (t.reason || '').toLowerCase().includes('stop') ? ' [SL]' :
      (t.reason || '').toLowerCase().includes('take-profit') ? ' [TP]' :
      (t.reason || '').toLowerCase().includes('max hold') ? ' [TIME]' : '';
    div.className = 'feed-item ' + cls;
    div.innerHTML = `<motion class="tk">${t.ticker}${exitTag}</motion> <span style="color:var(--muted)">${t.action}</span>`;
    div.onclick = () => {
      document.querySelectorAll('.feed-item').forEach(x => x.classList.remove('active'));
      div.classList.add('active');
      highlightPath(cy, t.path);
      renderDetail(t);
    };
    el.appendChild(div);
  });
}

function renderDetail(t) {
  const el = document.getElementById('detail');
  if (!t) { el.innerHTML = '<p style="color:var(--muted)">Select a trade.</p>'; return; }
  const agents = [
    { name: 'ML model', score: Math.round((t.p_up_20d||0)*100), verdict: (t.p_up_20d||0) >= 0.58 ? 'PASS' : ((t.p_up_20d||0) >= 0.45 ? 'HOLD' : 'FAIL') },
    { name: 'Quintile rank', score: (t.quintile||3)*20, verdict: (t.quintile||0) >= 4 ? 'PASS' : ((t.quintile||3) <= 2 ? 'FAIL' : 'HOLD') },
    { name: 'Regime (SPY)', score: Math.round((t.regime_scale||1)*100), verdict: (t.regime_scale||1) >= 0.99 ? 'PASS' : ((t.regime_scale||0) >= 0.35 ? 'HOLD' : 'FAIL') },
    { name: 'Risk gate', score: t.critical_flags ? 0 : 100, verdict: t.critical_flags ? 'FAIL' : 'PASS' },
  ];
  let html = `<motion class="tk-big">${t.ticker}</motion> <span class="verdict ${t.action==='EXIT'?'HOLD':(t.action.includes('LONG')?'PASS':'FAIL')}">${t.action}</span>
    <p style="color:var(--muted);font-size:0.85rem">${t.date} — ${t.reason || ''}</p>
    <p style="font-size:0.8rem">Path: ${(t.path||[]).join(' → ')}</p>`;
  agents.forEach(a => {
    const col = a.verdict === 'PASS' ? '#00e676' : (a.verdict === 'HOLD' ? '#ffc107' : '#ff5252');
    html += `<div class="agent"><div class="row"><span>${a.name}</span><span class="verdict ${a.verdict}">${a.verdict}</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${a.score}%;background:${col}"></div></div></div>`;
  });
  el.innerHTML = html;
}

function renderSwimlane(cy) {
  const el = document.getElementById('swimlane');
  const bars = DATA.bars || [];
  if (!bars.length) { el.innerHTML = '<p style="color:var(--muted);padding:8px">No positions in period.</p>'; return; }
  const dates = [];
  bars.forEach(b => { dates.push(b.start); if (b.end) dates.push(b.end); });
  const t0 = new Date(Math.min(...dates.map(d => new Date(d))));
  const t1 = new Date(Math.max(...dates.map(d => new Date(d))));
  const span = Math.max(t1 - t0, 86400000);
  const tickers = [...new Set(bars.map(b => b.ticker))].sort();
  el.innerHTML = tickers.map(tk => {
    const segs = bars.filter(b => b.ticker === tk);
    const inner = segs.map(b => {
      const left = ((new Date(b.start) - t0) / span) * 100;
      const end = b.end ? new Date(b.end) : t1;
      const w = Math.max(2, ((end - new Date(b.start)) / span) * 100);
      const pnl = b.pnl_pct != null ? ` PnL ${(b.pnl_pct*100).toFixed(1)}%` : '';
      return `<div class="seg ${b.side}" style="left:${left}%;width:${w}%" title="${b.start}→${b.end||'open'}${pnl}"></div>`;
    }).join('');
    return `<div class="lane"><div class="name">${tk}</div><div class="track">${inner}</div></div>`;
  }).join('');
  el.querySelectorAll('.seg').forEach(seg => {
    seg.onclick = () => {
      const tr = (DATA.traces||[]).find(t => t.ticker === seg.dataset.ticker && t.action && t.action.startsWith('ENTER') && t.date === seg.dataset.start);
      if (tr) { highlightPath(cy, tr.path); renderDetail(tr); }
    };
  });
}

const cy = buildCy();
initHeader();
renderFeed(cy);
renderSwimlane(cy);
if ((DATA.closed||[]).length) {
  const last = (DATA.traces||[]).filter(t => t.action === 'ENTER_LONG' || t.action === 'ENTER_SHORT').pop();
  if (last) { highlightPath(cy, last.path); renderDetail(last); }
}
</script>
</body>
</html>"""
