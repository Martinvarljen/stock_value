"""ML score ranking and quintile diagnostics for backtests."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np


def ml_score_from_signal(sig: dict) -> float | None:
    """Primary cross-sectional score: blended P(up) 20d, else composite."""
    for key in ("ml_score", "p_up_20d", "p_up_60d", "composite_score"):
        v = sig.get(key)
        if v is not None:
            try:
                f = float(v)
                if np.isfinite(f):
                    return f
            except (TypeError, ValueError):
                continue
    return None


def assign_quintile(scores: list[float]) -> list[int]:
    """Map scores to quintiles 1 (low) .. 5 (high)."""
    if not scores:
        return []
    import pandas as pd

    s = pd.Series(scores, dtype=float)
    try:
        labels = pd.qcut(s.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        return [int(x) for x in labels]
    except Exception:
        arr = np.array(scores, dtype=float)
        ranks = arr.argsort().argsort()
        n = len(arr)
        return [min(5, max(1, int(1 + 5 * ranks[i] / max(n - 1, 1)))) for i in range(n)]


def aggregate_quintile_forward_returns(
    signals: list[dict],
    *,
    horizon_months: int = 6,
) -> dict[int, dict]:
    """Average forward return by ML score quintile (pooled across checkpoints)."""
    fwd_key = f"fwd_{horizon_months}m"
    rows = []
    for s in signals:
        sc = ml_score_from_signal(s)
        r = s.get(fwd_key)
        if sc is None or r is None:
            continue
        rows.append((sc, float(r)))
    if len(rows) < 10:
        return {}

    scores = [x[0] for x in rows]
    quintiles = assign_quintile(scores)
    buckets: dict[int, list[float]] = defaultdict(list)
    for (_, ret), q in zip(rows, quintiles):
        buckets[int(q)].append(ret)

    out: dict[int, dict] = {}
    for q in sorted(buckets):
        rets = buckets[q]
        out[q] = {
            "n": len(rets),
            "avg_fwd": float(np.mean(rets)),
            "hit_rate": float(np.mean([1.0 if r > 0 else 0.0 for r in rets])),
        }
    return out


def print_quintile_table(quintiles: dict[int, dict], *, horizon_months: int = 6) -> None:
    if not quintiles:
        print("  ML quintiles: insufficient scored signals.")
        return
    print(f"\n  ML score quintiles (pooled, forward {horizon_months}M) — monotonicity check:")
    print(f"  {'Q':>3} {'N':>6} {'Avg fwd':>10} {'Hit rate':>10}")
    for q in sorted(quintiles):
        t = quintiles[q]
        print(f"  {q:>3} {t['n']:>6} {t['avg_fwd']:>+10.1%} {t['hit_rate']:>10.0%}")
    q5 = quintiles.get(5, {}).get("avg_fwd")
    q1 = quintiles.get(1, {}).get("avg_fwd")
    if q5 is not None and q1 is not None:
        spread = q5 - q1
        print(f"  Q5−Q1 spread: {spread:+.1%}  ({'OK' if spread > 0 else 'inverted — prefer rank-based L/S'})")
