#!/usr/bin/env python3
"""Build feature_baseline.json from portfolio daily_snapshots (ML gate features)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from projection.ml_model.drift_monitor import DriftMonitor  # noqa: E402

from portfolio.store import SNAPSHOTS_DIR  # noqa: E402

_GATE_FEATURES = ("ml_score", "p_up_20d", "atr_pct", "vol_60d_annual")
_DEFAULT_OUT = _ROOT / "projection" / "ml_model" / "saved_models" / "feature_baseline.json"


def _collect_from_snapshots(snap_dir: Path, max_files: int | None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in _GATE_FEATURES}
    paths = sorted(snap_dir.glob("*.json"), reverse=True)
    if max_files:
        paths = paths[:max_files]
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for a in data.get("analyses") or []:
            if not a.get("ok"):
                continue
            for key in _GATE_FEATURES:
                val = a.get(key)
                if isinstance(val, (int, float)) and val == val:
                    out[key].append(float(val))
    return {k: v for k, v in out.items() if len(v) >= 10}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ML drift baseline from daily snapshots.")
    ap.add_argument("--snap-dir", type=Path, default=SNAPSHOTS_DIR)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--max-files", type=int, default=60, help="Use last N snapshot files")
    args = ap.parse_args()

    training = _collect_from_snapshots(args.snap_dir, args.max_files)
    if not training:
        raise SystemExit(
            f"No gate features found under {args.snap_dir}. "
            "Run daily_run a few times first or use tools/build_feature_baseline.py with CSV."
        )

    monitor = DriftMonitor.fit(training)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    monitor.save(args.out)
    print(f"Wrote {args.out} from {args.snap_dir} ({', '.join(f'{k}={len(v)}' for k, v in training.items())})")


if __name__ == "__main__":
    main()
