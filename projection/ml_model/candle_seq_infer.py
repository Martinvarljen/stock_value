"""
candle_seq_infer.py — Load LSTM candle-sequence model and predict next H OHLC bars.

Trained by train_candle_sequence.py. Saved under saved_models/candle_seq/lstm.pt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEQ_DIR = Path(__file__).parent / "saved_models" / "candle_seq"

_model = None
_blob: dict | None = None
_disk_mtime: float | None = None


def _sanitize_bar(o: float, h: float, l: float, c: float) -> tuple[float, float, float, float]:
    o, h, l, c = float(o), float(h), float(l), float(c)
    body_lo, body_hi = min(o, c), max(o, c)
    l = min(l, body_lo)
    h = max(h, body_hi)
    if h <= l:
        h = l + 1e-4
    return o, h, l, c


def candle_sequence_available() -> bool:
    p = SEQ_DIR / "lstm.pt"
    return p.exists() and p.stat().st_size > 0


def candle_sequence_summary() -> str:
    if not candle_sequence_available():
        return "No candle-sequence model (run train_candle_sequence.py)."
    _ensure_loaded()
    if _blob is None:
        return "Candle-sequence checkpoint unreadable."
    h = int(_blob.get("horizon", 0))
    lb = int(_blob.get("lookback", 0))
    ind = int(_blob.get("in_dim", 4))
    suf = " (OHLC + return + range)" if ind == 6 else ""
    return f"Candle LSTM: lookback={lb}d, {ind} features/bar{suf} → next {h} OHLC bars."


def _ensure_loaded() -> None:
    global _model, _blob, _disk_mtime
    ckpt = SEQ_DIR / "lstm.pt"
    if not ckpt.exists():
        _model = None
        _blob = None
        _disk_mtime = None
        return
    try:
        mt = ckpt.stat().st_mtime
    except OSError:
        return
    if _model is not None and _disk_mtime == mt:
        return

    import torch
    import torch.nn as nn

    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(ckpt, map_location="cpu")
    except Exception:
        _model = None
        _blob = None
        _disk_mtime = None
        return

    if not isinstance(blob, dict) or "state_dict" not in blob:
        _model = None
        _blob = None
        _disk_mtime = None
        return

    lookback = int(blob["lookback"])
    horizon = int(blob["horizon"])
    hidden = int(blob.get("hidden", 96))
    num_layers = int(blob.get("num_layers", 2))
    in_dim = int(blob.get("in_dim", 4))

    class CandleSeqLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.horizon = horizon
            self.lstm = nn.LSTM(
                in_dim, hidden, num_layers, batch_first=True, dropout=0.15 if num_layers > 1 else 0.0
            )
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, horizon * 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            o, _ = self.lstm(x)
            y = self.head(o[:, -1, :])
            return y.view(-1, self.horizon, 4)

    m = CandleSeqLSTM()
    try:
        m.load_state_dict(blob["state_dict"], strict=True)
    except Exception:
        _model = None
        _blob = None
        _disk_mtime = None
        return

    m.eval()
    _model = m
    _blob = blob
    _disk_mtime = mt


def predict_future_ohlc(
    hist: pd.DataFrame,
    anchor_close: float | None = None,
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """
    Predict the next H daily OHLC bars from the last LOOKBACK rows of `hist`.

    Args:
        hist:  Must have columns Open, High, Low, Close (yfinance-style).
        anchor_close: Scale outputs to this last price (default: last Close in hist).

    Returns:
        (business_day_index length H, float array shape (H,4) columns O,H,L,C)
        or None if model missing / insufficient history.
    """
    import torch

    if hist is None or hist.empty:
        return None
    _ensure_loaded()
    if _model is None or _blob is None:
        return None

    lookback = int(_blob["lookback"])
    horizon = int(_blob["horizon"])
    need = lookback + 5
    if len(hist) < need:
        return None

    h = hist.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col not in h.columns:
            return None

    tail = h.iloc[-lookback:]
    anchor = float(anchor_close) if anchor_close is not None else float(tail["Close"].iloc[-1])
    if not np.isfinite(anchor) or anchor <= 0:
        return None

    ohl = tail[["Open", "High", "Low", "Close"]].astype(np.float64).values
    in_dim = int(_blob.get("in_dim", 4))
    if in_dim == 6:
        n = ohl.shape[0]
        ret = np.zeros(n, dtype=np.float64)
        ret[1:] = np.clip(ohl[1:, 3] / np.maximum(ohl[:-1, 3], 1e-9) - 1.0, -0.35, 0.35)
        hl_pct = np.clip((ohl[:, 1] - ohl[:, 2]) / np.maximum(ohl[:, 3], 1e-9), 0.0, 0.5)
        x = np.column_stack([ohl / anchor, ret, hl_pct])
    else:
        x = ohl / anchor
    if not np.all(np.isfinite(x)):
        return None

    xt = torch.from_numpy(x.reshape(1, lookback, in_dim).astype(np.float32))
    with torch.no_grad():
        y = _model(xt).numpy()[0]  # (H,4) normalized

    out = np.zeros((horizon, 4), dtype=np.float64)
    for i in range(horizon):
        o, hi, lo, c = y[i]
        o, hi, lo, c = o * anchor, hi * anchor, lo * anchor, c * anchor
        out[i] = _sanitize_bar(o, hi, lo, c)

    last_dt = pd.Timestamp(hist.index[-1])
    idx = pd.bdate_range(last_dt + pd.offsets.BDay(1), periods=horizon)
    return idx, out
