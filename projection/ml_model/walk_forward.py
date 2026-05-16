"""
walk_forward.py — Walk-forward ML retraining harness.

Why
===
The current ``trainer.train()`` fits one model on a fixed window and
ships it. That ignores the elephant: financial relationships are
non-stationary, so a model trained 2018-2022 starts decaying around
late 2023. Walk-forward retraining is the institutional standard for
catching that — retrain on a rolling window, evaluate on the next
period before promoting, and alert when calibration drifts.

What this module provides
-------------------------
* ``WalkForwardWindows`` — pure-Python date-arithmetic struct that
  enumerates ``(train_start, train_end, val_start, val_end)`` windows
  given a global start, global end, train length, val length, and
  step. Tested separately from anything network/pandas.

* ``run_walk_forward`` — orchestrates one full pass: for each window,
  train via ``projection.ml_model.trainer.train``, predict on the val
  window, compute Brier / log-loss / AUC / calibration histogram, and
  append a row to ``calibration_log.jsonl``.

* ``detect_calibration_drift`` — given the JSONL log, compare the
  latest window's Brier and reliability slope against a rolling
  baseline of the prior N windows; raise an alert when delta exceeds
  the configured threshold.

Output schema (``calibration_log.jsonl`` per row)::

    {
      "window_id": "2018-01-01__2022-12-31__2023-01-01__2023-06-30",
      "train_start": "2018-01-01", "train_end": "2022-12-31",
      "val_start":   "2023-01-01", "val_end":   "2023-06-30",
      "horizon": 20,
      "n_train": 12345, "n_val": 3211,
      "brier": 0.241, "log_loss": 0.681, "auc": 0.564,
      "reliability_slope": 0.93, "reliability_intercept": 0.04,
      "histogram": [{"bin": "0.0-0.1", "n": 230, "freq_actual": 0.12}, ...]
    }

CLI
---
::

    python -m projection.ml_model.walk_forward run \\
        --train-years 5 --val-months 6 --step-months 6 \\
        --start 2014-01-01 --end 2025-01-01 \\
        --out-dir reports/walkforward/2025_q1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator


# ── window enumeration ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Window:
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    def label(self) -> str:
        return (f"{self.train_start.isoformat()}__{self.train_end.isoformat()}"
                f"__{self.val_start.isoformat()}__{self.val_end.isoformat()}")


def _add_months(d: date, months: int) -> date:
    """Calendar-month math without dateutil."""
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    last_day = (date(y + (1 if m == 12 else 0),
                     1 if m == 12 else m + 1, 1) - timedelta(days=1)).day
    return date(y, m, min(d.day, last_day))


def walk_forward_windows(
    start: date,
    end: date,
    *,
    train_years: int = 5,
    val_months: int = 6,
    step_months: int = 6,
    embargo_days: int = 5,
) -> Iterator[Window]:
    """Enumerate sliding train/val windows from ``start`` to ``end``.

    Each train window has length ``train_years``; the val window starts
    ``embargo_days`` after train_end and runs for ``val_months``; we
    slide by ``step_months``. Stops when ``val_end > end``.

    Embargo is the gap between train_end and val_start that prevents
    label-leakage at the boundary (a 20-day forward label trained at
    train_end is "known" through train_end + 20d; by waiting 5d before
    validating we're not overlapping training labels with eval inputs,
    though the safer threshold is ``embargo_days >= max_horizon``).
    """
    current_train_start = start
    while True:
        train_end = _add_months(current_train_start, train_years * 12) - timedelta(days=1)
        val_start = train_end + timedelta(days=embargo_days + 1)
        val_end = _add_months(val_start, val_months) - timedelta(days=1)
        if val_end > end:
            return
        yield Window(
            train_start=current_train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        )
        current_train_start = _add_months(current_train_start, step_months)


# ── calibration metrics (pure Python) ────────────────────────────────────────

def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true or len(y_true) != len(y_prob):
        return float("nan")
    n = len(y_true)
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / n


def log_loss(y_true: list[int], y_prob: list[float], *, eps: float = 1e-12) -> float:
    if not y_true or len(y_true) != len(y_prob):
        return float("nan")
    n = len(y_true)
    s = 0.0
    for y, p in zip(y_true, y_prob):
        p = min(max(p, eps), 1 - eps)
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / n


def reliability_curve(
    y_true: list[int], y_prob: list[float], *, bins: int = 10,
) -> list[dict[str, Any]]:
    """Calibration histogram. Each bin records (count, mean_predicted,
    mean_actual). Used both for plotting and as a regression baseline."""
    if not y_true or len(y_true) != len(y_prob):
        return []
    bin_y: list[list[int]] = [[] for _ in range(bins)]
    bin_p: list[list[float]] = [[] for _ in range(bins)]
    for y, p in zip(y_true, y_prob):
        idx = min(bins - 1, max(0, int(p * bins)))
        bin_y[idx].append(int(y))
        bin_p[idx].append(float(p))
    out = []
    for i in range(bins):
        if not bin_y[i]:
            out.append({"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
                        "n": 0, "mean_predicted": None, "freq_actual": None})
            continue
        out.append({
            "bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}",
            "n": len(bin_y[i]),
            "mean_predicted": sum(bin_p[i]) / len(bin_p[i]),
            "freq_actual": sum(bin_y[i]) / len(bin_y[i]),
        })
    return out


def reliability_linear_fit(
    y_true: list[int], y_prob: list[float],
) -> tuple[float, float] | None:
    """OLS slope/intercept of actual freq vs predicted prob (per-name).

    A perfectly-calibrated model has slope=1, intercept=0. Slope < 0.7
    or > 1.3 is the classic drift threshold.
    """
    n = len(y_true)
    if n < 30 or n != len(y_prob):
        return None
    mean_p = sum(y_prob) / n
    mean_y = sum(y_true) / n
    cov = sum((p - mean_p) * (y - mean_y) for p, y in zip(y_prob, y_true)) / n
    var_p = sum((p - mean_p) ** 2 for p in y_prob) / n
    if var_p < 1e-12:
        return None
    slope = cov / var_p
    intercept = mean_y - slope * mean_p
    return slope, intercept


# ── drift detection ──────────────────────────────────────────────────────────

@dataclass
class DriftAlert:
    window_id: str
    metric: str
    current: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    breach: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "metric": self.metric,
            "current": self.current,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "z_score": self.z_score,
            "breach": self.breach,
        }


def detect_calibration_drift(
    log_path: Path,
    *,
    rolling_n: int = 5,
    z_threshold: float = 2.0,
    metric: str = "brier",
) -> list[DriftAlert]:
    """For each window in ``calibration_log.jsonl``, compare ``metric``
    against the rolling-mean of the prior ``rolling_n`` windows and
    raise an alert when the absolute z-score exceeds ``z_threshold``.

    Returns the list of alerts (one per window once we have enough
    history; empty for the first ``rolling_n`` windows).
    """
    if not log_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("val_start", ""))
    alerts: list[DriftAlert] = []
    for i, r in enumerate(rows):
        if i < rolling_n:
            continue
        prior = rows[i - rolling_n: i]
        vals = [p.get(metric) for p in prior if isinstance(p.get(metric), (int, float))]
        if len(vals) < 2:
            continue
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var) if var > 1e-12 else 1e-6
        cur = r.get(metric)
        if not isinstance(cur, (int, float)):
            continue
        z = (cur - mean) / std
        breach = abs(z) >= z_threshold
        alerts.append(DriftAlert(
            window_id=r.get("window_id", ""),
            metric=metric,
            current=float(cur),
            baseline_mean=mean,
            baseline_std=std,
            z_score=z,
            breach=breach,
        ))
    return alerts


# ── orchestration (lazy / optional pandas) ────────────────────────────────────

def append_log_row(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_walk_forward(
    *,
    start: date,
    end: date,
    train_years: int = 5,
    val_months: int = 6,
    step_months: int = 6,
    embargo_days: int = 25,
    horizon: int = 20,
    out_dir: Path,
    tickers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the full walk-forward loop. Heavy imports happen here only
    so the module remains importable in numpy-less environments.
    """
    from projection.ml_model.trainer import collect_training_data, train  # type: ignore
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "calibration_log.jsonl"

    rows: list[dict[str, Any]] = []
    for window in walk_forward_windows(
        start, end,
        train_years=train_years, val_months=val_months,
        step_months=step_months, embargo_days=embargo_days,
    ):
        print(f"\n=== Window {window.label()} ===", flush=True)
        # Collect once over the union, then split — same feature cache.
        full_df = collect_training_data(
            tickers, lookback_years=train_years + (val_months // 12) + 1,
            sample_step=5,
        )
        if full_df is None or full_df.empty:
            print("  no data, skipping window")
            continue
        # Filter columns by date range. Trainer-side rows have a 'date'
        # column added by ``_build_rows_from_hist``.
        if "date" not in full_df.columns:
            print("  trainer output has no 'date' column; aborting")
            return rows
        train_df = full_df[(full_df["date"] >= window.train_start.isoformat())
                           & (full_df["date"] <= window.train_end.isoformat())]
        val_df = full_df[(full_df["date"] >= window.val_start.isoformat())
                         & (full_df["date"] <= window.val_end.isoformat())]
        if len(train_df) < 500 or len(val_df) < 100:
            print(f"  skipping (n_train={len(train_df)}, n_val={len(val_df)})")
            continue
        models = train(df=train_df, save=False, calibrate=True, purged_cv=False)
        if not models:
            continue
        model = models.get(horizon)
        if model is None:
            continue
        feat_cols = [c for c in train_df.columns if c not in
                     {"date", "ticker", f"target_{horizon}", "open", "high", "low", "close", "volume"}]
        X_val = val_df[feat_cols].values
        y_val = val_df[f"target_{horizon}"].astype(int).tolist()
        try:
            y_prob = list(model.predict_proba(X_val)[:, 1])
        except Exception as e:
            print(f"  predict_proba failed: {e}")
            continue
        b = brier_score(y_val, y_prob)
        ll = log_loss(y_val, y_prob)
        rel = reliability_linear_fit(y_val, y_prob)
        hist = reliability_curve(y_val, y_prob)
        row = {
            "window_id": window.label(),
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "val_start": window.val_start.isoformat(),
            "val_end": window.val_end.isoformat(),
            "horizon": horizon,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "brier": b,
            "log_loss": ll,
            "reliability_slope": rel[0] if rel else None,
            "reliability_intercept": rel[1] if rel else None,
            "histogram": hist,
        }
        append_log_row(log_path, row)
        rows.append(row)
        print(f"  brier={b:.4f}  log_loss={ll:.4f}  "
              f"slope={(rel[0] if rel else float('nan')):.3f}", flush=True)
    return rows


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run a full walk-forward pass")
    run.add_argument("--start", required=True, type=date.fromisoformat)
    run.add_argument("--end", required=True, type=date.fromisoformat)
    run.add_argument("--train-years", type=int, default=5)
    run.add_argument("--val-months", type=int, default=6)
    run.add_argument("--step-months", type=int, default=6)
    run.add_argument("--embargo-days", type=int, default=25)
    run.add_argument("--horizon", type=int, default=20)
    run.add_argument("--out-dir", type=Path, default=Path("reports/walkforward"))
    drift = sub.add_parser("drift", help="scan calibration log for drift")
    drift.add_argument("--log", type=Path, required=True)
    drift.add_argument("--rolling-n", type=int, default=5)
    drift.add_argument("--z-threshold", type=float, default=2.0)
    drift.add_argument("--metric", default="brier")
    args = p.parse_args(argv)
    if args.cmd == "run":
        rows = run_walk_forward(
            start=args.start, end=args.end,
            train_years=args.train_years, val_months=args.val_months,
            step_months=args.step_months, embargo_days=args.embargo_days,
            horizon=args.horizon, out_dir=args.out_dir,
        )
        print(f"\nFinished {len(rows)} windows.")
        return 0
    if args.cmd == "drift":
        alerts = detect_calibration_drift(
            args.log, rolling_n=args.rolling_n,
            z_threshold=args.z_threshold, metric=args.metric,
        )
        breaches = [a for a in alerts if a.breach]
        print(json.dumps({
            "n_windows_scanned": len(alerts),
            "n_breaches": len(breaches),
            "breaches": [a.to_json() for a in breaches],
        }, indent=2))
        return 1 if breaches else 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
