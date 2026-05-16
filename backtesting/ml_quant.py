"""ML score ranking and quintile diagnostics for backtests.

Scale-discipline note
=====================
Earlier versions returned the first non-null among
``("ml_score", "p_up_20d", "p_up_60d", "composite_score")``. Those four
quantities live on **different scales** — the calibrated probabilities are
in ``[0, 1]`` while ``composite_score`` is in ``[-1, +1]``. Mixing them in a
single ``assign_quintile`` call meant cross-sectional ranks were sometimes
comparing scales rather than relative quality.

Quintiles are still rank-based and therefore invariant to monotone
transforms within a fixed sample, but mixing sources across rows in the
same sample is *not* invariant: a row with calibrated ``ml_score=0.55``
and a row with raw ``composite_score=0.55`` should not be ranked identically.

We now expose three explicit selectors:

* ``calibrated_score(sig)`` — calibrated probability only (``ml_score`` then
  ``p_up_20d`` / ``p_up_60d``). The expected primary input.
* ``composite_score(sig)`` — fallback for the legacy rule-composite path.
* ``ml_score_from_signal(sig, prefer="calibrated")`` — convenience wrapper
  that callers should pass an explicit ``prefer`` to. Default keeps the
  legacy "first non-null" behaviour for back-compat but emits no implicit
  scale mixing because the canonical pipeline always sets ``ml_score`` when
  a calibrated model is present.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np


_CALIBRATED_KEYS = ("ml_score", "p_up_20d", "p_up_60d")
_COMPOSITE_KEYS = ("composite_score",)


def _first_finite(sig: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = sig.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            return f
    return None


def calibrated_score(sig: dict) -> float | None:
    """Calibrated probability in [0,1]: ml_score, then p_up_20d / p_up_60d."""
    return _first_finite(sig, _CALIBRATED_KEYS)


def composite_score(sig: dict) -> float | None:
    """Rule-composite score in [-1, +1]."""
    return _first_finite(sig, _COMPOSITE_KEYS)


def ml_score_from_signal(sig: dict, *, prefer: str = "calibrated") -> float | None:
    """Pick a single scalar score for cross-sectional ranking.

    ``prefer`` selects which scale the caller intends:
      * ``"calibrated"`` (default) — use ml_score / p_up_*. Falls back to
        composite only if no calibrated value is present.
      * ``"composite"`` — use composite_score directly.
      * ``"any"`` — legacy first-non-null among (ml, p_up, composite). Use
        only when you've verified all input rows are on the same scale.

    The recommended pattern is to construct sample groups so every row in a
    given quintile call shares one scale, then pass ``prefer`` accordingly.
    """
    p = prefer.lower()
    if p == "calibrated":
        v = calibrated_score(sig)
        if v is not None:
            return v
        return composite_score(sig)
    if p == "composite":
        v = composite_score(sig)
        if v is not None:
            return v
        return calibrated_score(sig)
    if p == "any":
        return _first_finite(sig, _CALIBRATED_KEYS + _COMPOSITE_KEYS)
    raise ValueError(f"Unknown prefer={prefer!r}")


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
