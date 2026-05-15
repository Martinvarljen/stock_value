"""
Walk-forward evaluation by calendar year (research / diagnostics).

Trains a fresh LightGBM per train-year block and scores the following calendar
year — avoids evaluating the shipped production model on data it may have
seen during fitting.

  cd Finance
  python projection/ml_model/evaluate.py --quick
  python projection/ml_model/evaluate.py --horizon 20 --lookback 6 --sample-step 3

Writes: projection/ml_model/saved_models/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
for _p in [str(_root / "stock_analyzer"), str(_root / "projection")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_model.trainer import (  # noqa: E402
    DEFAULT_TICKERS,
    collect_training_data,
)
from ml_model.features import TECH_FEATURES  # noqa: E402


def walk_forward_by_year(
    df: pd.DataFrame,
    horizon: int,
    *,
    deep: bool = False,
) -> dict:
    """Return report dict with per-year test metrics."""
    target = f"target_up_{horizon}d"
    need = TECH_FEATURES + [target, "date"]
    for c in need:
        if c not in df.columns:
            return {"error": f"missing column {c}"}

    sub = df[need].dropna().copy()
    sub["date"] = pd.to_datetime(sub["date"], utc=False)
    sub = sub.sort_values(["date", "ticker"] if "ticker" in sub.columns else ["date"])
    sub["year"] = sub["date"].dt.year
    sub = sub.reset_index(drop=True)

    years = sorted(sub["year"].unique())
    if len(years) < 3:
        return {"error": "need_at_least_3_distinct_years", "years": years}

    try:
        import lightgbm as lgb
    except ImportError:
        return {"error": "lightgbm_not_installed"}

    from sklearn.metrics import brier_score_loss, roc_auc_score

    lgbm_params = dict(
        n_estimators=600 if not deep else 1200,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    by_year: dict[str, dict] = {}
    Xall = sub[TECH_FEATURES]
    yall = sub[target].astype(int).to_numpy()

    for y_test in years[2:]:  # earliest two years only for warm-up train
        train_mask = (sub["year"] < y_test).to_numpy()
        test_mask = (sub["year"] == y_test).to_numpy()
        if int(train_mask.sum()) < 500 or int(test_mask.sum()) < 50:
            continue
        X_tr, y_tr = Xall.iloc[train_mask], yall[train_mask]
        X_te, y_te = Xall.iloc[test_mask], yall[test_mask]
        m = lgb.LGBMClassifier(**lgbm_params)
        m.fit(X_tr, y_tr)
        prob = m.predict_proba(X_te)[:, 1]
        try:
            auc = float(roc_auc_score(y_te, prob))
        except ValueError:
            auc = float("nan")
        brier = float(brier_score_loss(y_te, prob))
        by_year[str(y_test)] = {
            "auc": round(auc, 4) if np.isfinite(auc) else None,
            "brier": round(brier, 5),
            "n_test": int(test_mask.sum()),
            "n_train": int(train_mask.sum()),
        }

    return {
        "generated_at": datetime.now().isoformat(),
        "horizon_days": horizon,
        "label_note": "Uses same labels as collect_training_data (see trainer --label-mode).",
        "by_year": by_year,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward ML evaluation by year")
    ap.add_argument("--quick", action="store_true", help="Small ticker set + shorter lookback")
    ap.add_argument("--horizon", type=int, default=20, choices=[5, 20, 60, 120])
    ap.add_argument("--lookback", type=int, default=6)
    ap.add_argument("--sample-step", type=int, default=4)
    ap.add_argument("--label-mode", choices=("raw", "alpha_vs_spy"), default="raw")
    ap.add_argument("--deep", action="store_true")
    args = ap.parse_args()

    tickers = ["AAPL", "MSFT", "JPM", "XOM", "PG"] if args.quick else DEFAULT_TICKERS[:40]
    lookback = 3 if args.quick else args.lookback
    step = 5 if args.quick else args.sample_step

    print(f"Collecting data: {len(tickers)} tickers, {lookback}Y, step={step} ...", flush=True)
    df = collect_training_data(tickers, lookback, step, label_mode=args.label_mode)
    if df.empty:
        print("No data — abort.")
        sys.exit(1)

    print(f"Walk-forward evaluation horizon={args.horizon}d ...", flush=True)
    report = walk_forward_by_year(df, args.horizon, deep=args.deep)
    if report.get("error"):
        print(json.dumps(report, indent=2))
        sys.exit(1)

    out_dir = Path(__file__).parent / "saved_models"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "evaluation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(report["by_year"], indent=2))


if __name__ == "__main__":
    main()
