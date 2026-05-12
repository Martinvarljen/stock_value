"""
predictor.py — Load trained LightGBM models and run inference.

Falls back gracefully to None when no trained models are present
so projection_engine can use rule-based scoring instead.
"""

import json
import math
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "saved_models"

# Module-level cache — models loaded once per process
_cache: dict[int, object] = {}
_meta: dict | None = None
_loaded: bool = False


def _ensure_loaded() -> None:
    global _cache, _meta, _loaded
    if _loaded:
        return
    _loaded = True

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return

    try:
        import joblib
        _meta = json.loads(meta_path.read_text())
        for h in _meta.get("horizons", []):
            model_path = MODELS_DIR / f"lgbm_{h}d.pkl"
            if model_path.exists():
                _cache[h] = joblib.load(model_path)
    except Exception as e:
        print(f"  [predictor] Model load error: {e}")
        _cache.clear()


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

    X = _build_vector(features, feat_cols)

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
