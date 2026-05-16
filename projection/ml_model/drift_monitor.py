"""
drift_monitor.py — Feature-drift monitoring (PSI + KS).

Why
===
A model is only valid for inputs that look like its training data. As
the world shifts (rate-cut cycles, vol regime breaks, new sector
leadership) the *distribution* of features at prediction time slides
away from the training distribution. PSI/KS scores quantify that
slide and let us alert before the model's calibration falls apart.

The drift monitor is decoupled from training so the prediction path
can compute drift cheaply per-day:

    monitor = DriftMonitor.load(<saved_models_dir>/feature_baseline.json)
    drift = monitor.compute(current_features)
    if drift.severity == "high":
        log.alert(...)

PSI (Population Stability Index)
--------------------------------
For a numeric feature binned into K bins:

    PSI = sum_i ( (a_i - e_i) * ln(a_i / e_i) )

where e_i = expected (training) frequency in bin i, a_i = actual
(prediction-time) frequency. Conventional thresholds:

    PSI < 0.10            no shift
    0.10 <= PSI < 0.25    moderate
    PSI >= 0.25           high — investigate / retrain

KS (Kolmogorov-Smirnov)
-----------------------
Maximum CDF distance between two empirical distributions; captures
shape changes PSI may miss when the bin boundaries are coarse.
Threshold ~0.10 for moderate, ~0.20 for high.

The monitor stores a baseline of {feature → (bin_edges, frequencies,
sample_quantiles)} so it doesn't need access to the original training
data at predict time.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


# ── pure-Python distribution helpers ─────────────────────────────────────────

def _quantiles(xs: list[float], qs: list[float]) -> list[float]:
    if not xs:
        return [float("nan")] * len(qs)
    s = sorted(xs)
    n = len(s)
    out = []
    for q in qs:
        if q <= 0:
            out.append(s[0])
            continue
        if q >= 1:
            out.append(s[-1])
            continue
        idx = q * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        out.append(s[lo] * (1 - frac) + s[hi] * frac)
    return out


def _bin_index(x: float, edges: list[float]) -> int:
    """Return bin index for ``x`` given monotonic edges. Edges are
    treated as right-inclusive interior boundaries; values below
    ``edges[0]`` go to bin 0, above ``edges[-1]`` go to last bin."""
    for i, e in enumerate(edges):
        if x <= e:
            return i
    return len(edges)


def _binned_freq(xs: list[float], edges: list[float]) -> list[float]:
    counts = [0] * (len(edges) + 1)
    for x in xs:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        counts[_bin_index(float(x), edges)] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def population_stability_index(
    expected: list[float], actual: list[float], *, n_bins: int = 10,
    smooth: float = 1e-4,
) -> float:
    """PSI between expected (baseline) and actual samples."""
    expected = [float(x) for x in expected if x is not None and not (isinstance(x, float) and math.isnan(x))]
    actual = [float(x) for x in actual if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not expected or not actual:
        return float("nan")
    qs = [(i + 1) / n_bins for i in range(n_bins - 1)]
    edges = _quantiles(expected, qs)
    if len(set(edges)) < len(edges):
        # collapsed bins -> distribution is degenerate; return 0 but flag.
        warnings.warn("PSI: degenerate baseline bins (constant feature?)", stacklevel=2)
    e_freq = _binned_freq(expected, edges)
    a_freq = _binned_freq(actual, edges)
    psi = 0.0
    for e, a in zip(e_freq, a_freq):
        e = max(e, smooth)
        a = max(a, smooth)
        psi += (a - e) * math.log(a / e)
    return psi


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max |CDF_a - CDF_b|)."""
    a = sorted(float(x) for x in a if x is not None and not (isinstance(x, float) and math.isnan(x)))
    b = sorted(float(x) for x in b if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not a or not b:
        return float("nan")
    i = j = 0
    cdf_a = cdf_b = 0.0
    n_a = len(a)
    n_b = len(b)
    d_max = 0.0
    while i < n_a and j < n_b:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / n_a
        else:
            j += 1
            cdf_b = j / n_b
        d_max = max(d_max, abs(cdf_a - cdf_b))
    while i < n_a:
        i += 1
        cdf_a = i / n_a
        d_max = max(d_max, abs(cdf_a - cdf_b))
    while j < n_b:
        j += 1
        cdf_b = j / n_b
        d_max = max(d_max, abs(cdf_a - cdf_b))
    return d_max


# ── monitor ──────────────────────────────────────────────────────────────────

@dataclass
class FeatureBaseline:
    feature: str
    edges: list[float]
    expected_freq: list[float]
    quantiles_5_95: tuple[float, float]
    n_samples: int


@dataclass
class DriftRow:
    feature: str
    psi: float
    ks: float
    severity: str            # "ok" | "moderate" | "high"
    n_actual: int
    out_of_range_pct: float

    def to_json(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks": self.ks,
            "severity": self.severity,
            "n_actual": self.n_actual,
            "out_of_range_pct": self.out_of_range_pct,
        }


@dataclass
class DriftReport:
    rows: list[DriftRow] = field(default_factory=list)

    @property
    def severity(self) -> str:
        if any(r.severity == "high" for r in self.rows):
            return "high"
        if any(r.severity == "moderate" for r in self.rows):
            return "moderate"
        return "ok"

    def to_json(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "rows": [r.to_json() for r in self.rows],
        }


def _classify(psi: float, ks: float) -> str:
    if psi != psi or ks != ks:  # NaN
        return "ok"
    if psi >= 0.25 or ks >= 0.20:
        return "high"
    if psi >= 0.10 or ks >= 0.10:
        return "moderate"
    return "ok"


@dataclass
class DriftMonitor:
    """Holds the training-time baseline for a list of features."""

    baselines: dict[str, FeatureBaseline] = field(default_factory=dict)
    psi_bins: int = 10

    @classmethod
    def fit(
        cls,
        training_features: dict[str, Iterable[float]],
        *,
        psi_bins: int = 10,
    ) -> "DriftMonitor":
        baselines: dict[str, FeatureBaseline] = {}
        for name, raw in training_features.items():
            xs = [float(x) for x in raw if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if len(xs) < 50:
                continue
            qs = [(i + 1) / psi_bins for i in range(psi_bins - 1)]
            edges = _quantiles(xs, qs)
            expected_freq = _binned_freq(xs, edges)
            q05, q95 = _quantiles(xs, [0.05, 0.95])
            baselines[name] = FeatureBaseline(
                feature=name, edges=edges, expected_freq=expected_freq,
                quantiles_5_95=(q05, q95), n_samples=len(xs),
            )
        return cls(baselines=baselines, psi_bins=psi_bins)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "psi_bins": self.psi_bins,
            "baselines": {
                name: {
                    "feature": b.feature,
                    "edges": b.edges,
                    "expected_freq": b.expected_freq,
                    "quantiles_5_95": list(b.quantiles_5_95),
                    "n_samples": b.n_samples,
                }
                for name, b in self.baselines.items()
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DriftMonitor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        baselines = {
            name: FeatureBaseline(
                feature=v["feature"],
                edges=list(v["edges"]),
                expected_freq=list(v["expected_freq"]),
                quantiles_5_95=tuple(v["quantiles_5_95"]),  # type: ignore
                n_samples=int(v["n_samples"]),
            )
            for name, v in payload.get("baselines", {}).items()
        }
        return cls(baselines=baselines, psi_bins=int(payload.get("psi_bins", 10)))

    def compute(self, actual_features: dict[str, Iterable[float]]) -> DriftReport:
        rows: list[DriftRow] = []
        for name, base in self.baselines.items():
            xs = [float(x) for x in actual_features.get(name, [])
                  if x is not None and not (isinstance(x, float) and math.isnan(x))]
            if not xs:
                continue
            # Use PSI on baseline frequencies via shared edges; KS via
            # the (regenerated) quantiles is too expensive without the
            # raw baseline samples, so we approximate KS with a
            # per-bin frequency comparison which is a strict upper
            # bound on KS for the same binning.
            psi = 0.0
            actual_freq = _binned_freq(xs, base.edges)
            for e, a in zip(base.expected_freq, actual_freq):
                e_smooth = max(e, 1e-4)
                a_smooth = max(a, 1e-4)
                psi += (a_smooth - e_smooth) * math.log(a_smooth / e_smooth)
            # Approx KS = max cumulative |a - e| across the bins.
            cum_e = 0.0
            cum_a = 0.0
            ks = 0.0
            for e, a in zip(base.expected_freq, actual_freq):
                cum_e += e
                cum_a += a
                ks = max(ks, abs(cum_a - cum_e))
            # Out-of-range rate: how many samples sit outside the 5-95 band.
            q05, q95 = base.quantiles_5_95
            out = sum(1 for x in xs if x < q05 or x > q95) / len(xs)
            rows.append(DriftRow(
                feature=name, psi=psi, ks=ks,
                severity=_classify(psi, ks),
                n_actual=len(xs), out_of_range_pct=out,
            ))
        return DriftReport(rows=rows)
