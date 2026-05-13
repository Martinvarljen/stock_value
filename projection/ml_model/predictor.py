"""
predictor.py — Load trained LightGBM models and run inference.

Falls back gracefully to None when no trained models are present
so projection_engine can use rule-based scoring instead.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "saved_models"

# Module-level cache — reload when metadata on disk changes (e.g. after training)
_cache: dict[int, object] = {}
_meta: dict | None = None
_disk_mtime_loaded: float | None = None
_last_load_error: str | None = None


def _ensure_loaded() -> None:
    """
    Load or refresh models when metadata.json is new or cache is empty.

    Streamlit keeps the process alive across reruns; a failed first load must
    not permanently block later loads after `trainer.py` writes new files.
    """
    global _cache, _meta, _disk_mtime_loaded, _last_load_error

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        _cache.clear()
        _meta = None
        _disk_mtime_loaded = None
        _last_load_error = None
        return

    try:
        mt = meta_path.stat().st_mtime
    except OSError:
        return

    if _cache and _meta is not None and _disk_mtime_loaded == mt:
        return

    try:
        import joblib

        _meta = json.loads(meta_path.read_text())
        new_cache: dict[int, object] = {}
        for h in _meta.get("horizons", []):
            hi = int(h)
            model_path = MODELS_DIR / f"lgbm_{hi}d.pkl"
            if model_path.exists():
                new_cache[hi] = joblib.load(model_path)
        _cache = new_cache
        _disk_mtime_loaded = mt
        _last_load_error = None if _cache else "metadata.json found but no matching .pkl files"
    except Exception as e:
        _last_load_error = f"{type(e).__name__}: {e}"
        print(f"  [predictor] Model load error: {_last_load_error}")
        _cache.clear()
        _meta = None
        _disk_mtime_loaded = None


def ml_predict(features: dict, horizons: list[int] = None) -> dict[int, float] | None:
    """
    Run ML inference for given horizons.

    Args:
        features:  dict from ml_model.features.extract_features()
        horizons:  list of horizon days to predict (default: all trained)

    Returns:
        dict {horizon_days: probability} or None if no models available.
    """
    _ensure_loaded()
    if not _cache:
        return None

    targets = horizons or list(_cache.keys())
    feat_cols = (_meta or {}).get("feature_cols", [])
    if not feat_cols:
        return None

    row = _build_vector(features, feat_cols)
    X = pd.DataFrame(row, columns=feat_cols)

    results: dict[int, float] = {}
    for h in targets:
        if h not in _cache:
            continue
        try:
            prob = float(_cache[h].predict_proba(X)[0][1])
            results[h] = round(prob, 4)
        except Exception as e:
            print(f"  [predictor] Inference error ({h}d): {e}")

    return results if results else None


def _build_vector(features: dict, feat_cols: list[str]) -> np.ndarray:
    vals = []
    for col in feat_cols:
        v = features.get(col, 0.0)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 0.0
        vals.append(float(v))
    return np.array(vals, dtype=np.float32).reshape(1, -1)


def models_available() -> bool:
    _ensure_loaded()
    return bool(_cache)


def models_load_hint() -> str:
    """
    Short message for UI when models are missing or failed to load.
    Empty string if models are OK.
    """
    _ensure_loaded()
    if _cache:
        return ""
    if not (MODELS_DIR / "metadata.json").exists():
        return f"No trained models (expected under {MODELS_DIR})."
    if _last_load_error:
        return f"Load error: {_last_load_error}"
    return "Models not loaded."


def get_metadata() -> dict | None:
    _ensure_loaded()
    return _meta


def model_summary() -> str:
    """Human-readable model status for the dashboard."""
    if not models_available():
        return "No trained models found — using rule-based projections. Run ml_model/trainer.py to train."
    meta = get_metadata() or {}
    trained_at = meta.get("trained_at", "unknown")[:10]
    aucs = {
        f"{h}d": f"{v['auc_mean']:.3f}"
        for h, v in {int(k): v for k, v in meta.get("metrics", {}).items()}.items()
    }
    auc_str = "  ".join(f"{k}: AUC {v}" for k, v in aucs.items())
    n = meta.get("metrics", {})
    n_samples = list(n.values())[0].get("n_samples", "?") if n else "?"
    return f"LightGBM models trained {trained_at} | {n_samples:,} samples | {auc_str}"
