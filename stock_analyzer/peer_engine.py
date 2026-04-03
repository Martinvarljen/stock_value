"""
peer_engine.py  —  Cross-ticker peer comparison

analyze_peers(records) → dict with:
  - groups:   list of {sector, peers: [{ticker, metrics...}]} ranked tables
  - rankings: flat dict of ticker → {rank, out_of, percentile} per metric

Works on the already-analyzed records list — no new data fetches.
Tickers in the same sector are compared automatically.
If fewer than 2 tickers share a sector, all tickers are compared together.
"""

from utils import _pct, _num, _x

# Metrics included in peer comparison — (record_key, label, higher_is_better, format_fn)
_PEER_METRICS = [
    ("revenue_cagr_5y",   "Rev CAGR 5Y",       True,  _pct),
    ("operating_margin",  "Op Margin",          True,  _pct),
    ("roic",              "ROIC",               True,  _pct),
    ("net_debt_ebitda",   "ND/EBITDA",          False, lambda v: f"{v:.1f}x"),
    ("interest_coverage", "Int Coverage",       True,  lambda v: f"{v:.1f}x"),
    ("fcf_yield",         "FCF Yield",          True,  _pct),
    ("fair_value_weighted","Fair Value (DCF)",  None,  _num),   # directional N/A — show only
]

# Metrics ranked (higher_is_better explicitly set)
_RANK_METRICS = [
    ("revenue_cagr_5y",  True),
    ("operating_margin", True),
    ("roic",             True),
    ("net_debt_ebitda",  False),
    ("interest_coverage",True),
    ("fcf_yield",        True),
]


def _rank_group(records: list[dict], metric_key: str, higher_is_better: bool) -> dict:
    """
    Returns {ticker: rank} for a group of records on a single metric.
    Ties share the same rank. Records missing the metric are ranked last.
    """
    pairs = []
    for r in records:
        v = r.get(metric_key)
        if v is not None:
            pairs.append((r["ticker"], v))

    pairs.sort(key=lambda x: x[1], reverse=higher_is_better)

    ranks = {}
    for i, (ticker, _) in enumerate(pairs, 1):
        ranks[ticker] = i

    # Missing data → rank last
    n = len(records)
    for r in records:
        if r["ticker"] not in ranks:
            ranks[r["ticker"]] = n

    return ranks


def analyze_peers(records: list[dict]) -> dict:
    """
    Groups records by sector and produces ranked comparison tables.
    Falls back to a single all-tickers group if no sector has ≥ 2 members.
    """
    if not records:
        return {"groups": [], "rankings": {}}

    # Group by sector
    sector_groups: dict[str, list] = {}
    for r in records:
        sec = r.get("sector") or "Unknown"
        sector_groups.setdefault(sec, []).append(r)

    # Keep groups with ≥ 2 members; put loners in a catch-all
    multi  = {s: g for s, g in sector_groups.items() if len(g) >= 2}
    loners = [r for g in sector_groups.values() if len(g) == 1 for r in g]

    if not multi:
        # No sector has 2+ tickers — compare everything together
        multi = {"All Tickers": records}
        loners = []
    elif loners:
        multi["Other / Mixed"] = loners

    groups    = []
    rankings  = {r["ticker"]: {} for r in records}

    for sector, group in multi.items():
        # Build metric rankings for this group
        metric_ranks = {}
        for mk, hib in _RANK_METRICS:
            metric_ranks[mk] = _rank_group(group, mk, hib)

        # Build peer rows
        peer_rows = []
        for r in group:
            ticker = r["ticker"]
            row = {
                "ticker":   ticker,
                "company":  (r.get("company_name") or ticker)[:30],
                "currency": r.get("currency", ""),
                "price":    r.get("current_price"),
                "classification": (r.get("classification_result") or {}).get("classification", "N/A"),
                "metrics":  {},
            }
            for mk, label, hib, fmt in _PEER_METRICS:
                val = r.get(mk)
                row["metrics"][mk] = {
                    "value":     val,
                    "formatted": fmt(val) if val is not None else "N/A",
                    "rank":      metric_ranks.get(mk, {}).get(ticker),
                    "out_of":    len([x for x in group if x.get(mk) is not None]),
                    "label":     label,
                }

            # Overall score: average rank percentile across ranked metrics
            rank_pcts = []
            for mk, hib in _RANK_METRICS:
                r_rank = metric_ranks.get(mk, {}).get(ticker)
                n_valid = len([x for x in group if x.get(mk) is not None])
                if r_rank is not None and n_valid > 1:
                    # Convert rank to percentile (1st = 100%, last = 0%)
                    rank_pcts.append(1 - (r_rank - 1) / (n_valid - 1))

            row["overall_percentile"] = sum(rank_pcts) / len(rank_pcts) if rank_pcts else None

            # Store in global rankings dict
            for mk in _RANK_METRICS:
                mk_key = mk[0]
                rankings[ticker][mk_key] = {
                    "rank":   metric_ranks.get(mk_key, {}).get(ticker),
                    "out_of": len([x for x in group if x.get(mk_key) is not None]),
                }
            rankings[ticker]["overall_percentile"] = row["overall_percentile"]
            rankings[ticker]["sector_group"]        = sector

            peer_rows.append(row)

        # Sort group by overall percentile descending
        peer_rows.sort(key=lambda x: x["overall_percentile"] or 0, reverse=True)

        groups.append({
            "sector":    sector,
            "peers":     peer_rows,
            "n_tickers": len(group),
        })

    return {"groups": groups, "rankings": rankings}


# ── display helper ─────────────────────────────────────────────────────────────

def print_peers(result: dict):
    if not result["groups"]:
        return

    print(f"\n{'─' * 70}")
    print(f"  PEER COMPARISON")
    print(f"{'─' * 70}")

    for group in result["groups"]:
        sector = group["sector"]
        peers  = group["peers"]
        n      = group["n_tickers"]

        print(f"\n  [{sector}]  —  {n} ticker(s)")

        # Header row
        metric_labels = [m[1] for m in _PEER_METRICS]
        header = f"  {'Ticker':<10}  {'Company':<28}  {'Clf':<14}"
        for lbl in metric_labels:
            header += f"  {lbl:>11}"
        header += f"  {'Score':>7}"
        print(f"  {'─' * (len(header) - 2)}")
        print(header)
        print(f"  {'─' * (len(header) - 2)}")

        for row in peers:
            line = f"  {row['ticker']:<10}  {row['company']:<28}  {row['classification']:<14}"
            for mk, label, hib, fmt in _PEER_METRICS:
                m = row["metrics"][mk]
                cell = m["formatted"]
                # Add rank badge if metric is ranked
                if m["rank"] is not None and m["out_of"] > 1:
                    cell = f"{cell} #{m['rank']}"
                line += f"  {cell:>11}"
            score = row["overall_percentile"]
            line += f"  {score:.0%}" if score is not None else "    N/A"
            print(line)

        print()
        print(f"  Score = average rank percentile across {len(_RANK_METRICS)} metrics (100% = best in group)")
