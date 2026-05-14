"""
ohlcv_validate.py — OHLCV schema checks (implementation brief §3).

Validates sorted timestamps, numeric OHLC, bar sanity (high/low vs body),
optional volume. Used for research / QC — not a trading guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def validate_ohlcv_dataframe(
    df: pd.DataFrame,
    *,
    require_volume: bool = False,
    timestamp_col: str | None = None,
) -> dict[str, Any]:
    """
    Return {"ok": bool, "errors": [...], "warnings": [...], "n_bars": int}.

    If timestamp_col is None, uses the DataFrame index when it is DatetimeIndex;
    otherwise skips monotonic timestamp checks.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or len(df) == 0:
        return {"ok": False, "errors": ["empty_frame"], "warnings": [], "n_bars": 0}

    need = ("open", "high", "low", "close")
    miss = [c for c in need if c not in df.columns]
    if miss:
        return {"ok": False, "errors": [f"missing_columns:{miss}"], "warnings": [], "n_bars": 0}

    sub = df[list(need) + (["volume"] if "volume" in df.columns else [])].copy()
    for c in need:
        if sub[c].isna().all():
            errors.append(f"all_nan:{c}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings, "n_bars": int(len(sub))}

    if require_volume:
        if "volume" not in sub.columns:
            errors.append("volume_required_but_missing")
        elif sub["volume"].isna().all():
            warnings.append("volume_all_nan")

    ts_series = None
    if timestamp_col and timestamp_col in df.columns:
        ts_series = df[timestamp_col]
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_series = pd.Series(df.index)

    if ts_series is not None:
        if not ts_series.is_monotonic_increasing:
            errors.append("timestamp_not_sorted")
        if ts_series.duplicated().any():
            errors.append("duplicate_timestamp")

    o = sub["open"].astype(float)
    h = sub["high"].astype(float)
    l = sub["low"].astype(float)
    c = sub["close"].astype(float)

    body_hi = np.maximum(o.values, c.values)
    body_lo = np.minimum(o.values, c.values)
    if ((h.values + 1e-9) < body_hi).any():
        errors.append("high_below_body")
    if ((l.values - 1e-9) > body_lo).any():
        errors.append("low_above_body")
    if (h.values < l.values).any():
        errors.append("high_below_low")

    zr = (h.values - l.values) <= 1e-12
    if zr.any():
        warnings.append(f"zero_range_bars:{int(zr.sum())}")

    if "volume" in sub.columns:
        vol = sub["volume"].astype(float)
        if (vol < 0).any():
            errors.append("negative_volume")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings, "n_bars": int(len(sub))}


def validate_ohlcv_from_data_dict(data: dict) -> dict[str, Any]:
    """
    Build a frame from collect_data() 1Y lists (open/high/low/close/volume_1y).
    Falls back to close-only rows if OHLC lengths diverge (not ok for full rules).
    """
    c = data.get("close_1y") or []
    if not c:
        return {"ok": False, "errors": ["no_close_1y"], "warnings": [], "n_bars": 0}

    n = len(c)
    o = data.get("open_1y") or []
    h = data.get("high_1y") or []
    l = data.get("low_1y") or []
    v = data.get("volume_1y") or []

    if len(o) != n or len(h) != n or len(l) != n:
        return {
            "ok": False,
            "errors": ["ohlc_length_mismatch"],
            "warnings": [],
            "n_bars": n,
        }

    vol_col: list[float] | None
    if len(v) == n:
        vol_col = [float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan for x in v]
    else:
        vol_col = None

    df = pd.DataFrame(
        {
            "open": [float(x) for x in o],
            "high": [float(x) for x in h],
            "low": [float(x) for x in l],
            "close": [float(x) for x in c],
        }
    )
    if vol_col is not None:
        df["volume"] = vol_col

    out = validate_ohlcv_dataframe(df, require_volume=False)
    if vol_col is None:
        out.setdefault("warnings", []).append("volume_missing_or_misaligned_skipped")
    return out
