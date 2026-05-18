#!/usr/bin/env python3
"""Build feature_baseline.json for daily ML drift gates (from Dolt feather cache or CSV)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projection.ml_model.drift_monitor import DriftMonitor  # noqa: E402

_DEFAULT_OUT = _ROOT / "projection" / "ml_model" / "saved_models" / "feature_baseline.json"
_GATE_FEATURES = ("ml_score", "p_up_20d", "atr_pct", "vol_60d_annual")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ML feature drift baseline JSON.")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument(
        "--from-training-csv",
        type=Path,
        default=None,
        help="CSV with columns matching gate features (one row per observation)",
    )
    args = ap.parse_args()

    if args.from_training_csv is None:
        print(
            "Provide --from-training-csv with columns: "
            + ", ".join(_GATE_FEATURES)
            + "\nOr run after exporting features from ml_model/trainer.py."
        )
        raise SystemExit(1)

    import pandas as pd

    df = pd.read_csv(args.from_training_csv)
    training = {col: df[col].dropna().tolist() for col in _GATE_FEATURES if col in df.columns}
    if not training:
        raise SystemExit(f"No columns found in {_GATE_FEATURES}")

    monitor = DriftMonitor.fit(training)
    monitor.save(args.out)
    print(f"Wrote {args.out} ({len(monitor.baselines)} features)")


if __name__ == "__main__":
    main()
