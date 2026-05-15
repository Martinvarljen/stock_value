"""
train_candle_sequence.py — Train LSTM to predict next H daily OHLC bars from lookback L.

Uses normalized OHLC windows (scaled by last close) plus per-bar return and
(H−L)/C (Kaufman-style emphasis on change vs. noise). Cross-ticker training.

**Research hygiene:** sequence models overfit easily. Prefer many tickers, time-based
train/val splits, early stopping on validation loss, and treat outputs as illustrative
unless you publish clear out-of-sample metrics (see projection/ml_model/evaluate.py
for the LightGBM side).

Run (from Finance folder):
    python projection/ml_model/train_candle_sequence.py
    python projection/ml_model/train_candle_sequence.py --quick
    python projection/ml_model/train_candle_sequence.py --all-tickers --lookback-years 8 --epochs 22 --sample-step 1
    powershell -File run_ml_train_best.ps1   # LSTM + LightGBM overnight preset

Output:
    projection/ml_model/saved_models/candle_seq/lstm.pt
    projection/ml_model/saved_models/candle_seq/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
for _p in (str(_root / "stock_analyzer"), str(_root / "projection")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ml_model.trainer import DEFAULT_TICKERS
except Exception:
    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

OUT_DIR = Path(__file__).parent / "saved_models" / "candle_seq"

LOOKBACK = 50
HORIZON = 12
HIDDEN = 112  # wider LSTM = more capacity (checkpoint stores hidden for inference)
NUM_LAYERS = 2
IN_DIM = 6  # OHLC (norm) + clipped daily return + (H-L)/C
DEFAULT_SAMPLE_STEP = 2


def _windows_for_hist(hist: pd.DataFrame, sample_step: int):
    """Return list of X (L, IN_DIM), Y (H,4), end_date for each training window."""
    if hist is None or len(hist) < LOOKBACK + HORIZON + 2:
        return [], [], []
    h = hist.copy()
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)

    for col in ("Open", "High", "Low", "Close"):
        if col not in h.columns:
            return [], [], []

    n = len(h)
    xs, ys, dates = [], [], []
    c = h["Close"].astype(float).values

    for end in range(LOOKBACK - 1, n - HORIZON, sample_step):
        anchor = float(c[end])
        if not np.isfinite(anchor) or anchor <= 0:
            continue
        block = h.iloc[end - LOOKBACK + 1 : end + 1]
        fut = h.iloc[end + 1 : end + 1 + HORIZON]
        ohl = block[["Open", "High", "Low", "Close"]].astype(np.float64).values
        Lb = ohl.shape[0]
        ret = np.zeros(Lb, dtype=np.float64)
        ret[1:] = np.clip(ohl[1:, 3] / np.maximum(ohl[:-1, 3], 1e-9) - 1.0, -0.35, 0.35)
        hl_pct = np.clip((ohl[:, 1] - ohl[:, 2]) / np.maximum(ohl[:, 3], 1e-9), 0.0, 0.5)
        x = np.column_stack([ohl / anchor, ret, hl_pct])
        y = fut[["Open", "High", "Low", "Close"]].astype(np.float64).values / anchor
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            continue
        xs.append(x)
        ys.append(y)
        dates.append(h.index[end])
    return xs, ys, dates


def collect_xy(
    tickers: list[str],
    lookback_years: int,
    sample_step: int,
) -> tuple[np.ndarray, np.ndarray, list]:
    import yfinance as yf

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_years * 365 + 60)
    all_x, all_y, all_d = [], [], []

    for i, tk in enumerate(tickers, 1):
        print(f"[{i:>3}/{len(tickers)}] {tk:<6}", end=" ", flush=True)
        try:
            hist = yf.Ticker(tk).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
            )
            if hist.empty:
                print("- empty")
                continue
            xs, ys, ds = _windows_for_hist(hist, sample_step)
            all_x.extend(xs)
            all_y.extend(ys)
            all_d.extend(ds)
            print(f"-> {len(xs)} windows")
        except Exception as e:
            print(f"- err {e}")
        time.sleep(0.12)

    if not all_x:
        return np.array([]), np.array([]), []
    X = np.stack(all_x, axis=0).astype(np.float32)
    Y = np.stack(all_y, axis=0).astype(np.float32)
    print(f"\nTotal windows: {len(X):,}")
    return X, Y, all_d


def train_main(
    tickers: list[str],
    lookback_years: int,
    epochs: int,
    batch_size: int,
    lr: float,
    sample_step: int,
    hidden: int | None = None,
) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, Y, dates = collect_xy(tickers, lookback_years, sample_step)
    if len(X) < 300:
        print(f"Need at least ~300 windows; got {len(X)}. Add tickers, years, or lower --sample-step.")
        return

    order = np.argsort(pd.DatetimeIndex(dates).astype(np.int64))
    cut = int(len(order) * 0.88)
    tr, va = order[:cut], order[cut:]
    X_tr, Y_tr = X[tr], Y[tr]
    X_va, Y_va = X[va], Y[va]

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(Y_va))
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    hid = int(hidden) if hidden is not None else int(HIDDEN)

    class CandleSeqLSTM(nn.Module):
        def __init__(self, h: int) -> None:
            super().__init__()
            self.horizon = HORIZON
            self.lstm = nn.LSTM(
                IN_DIM, h, NUM_LAYERS, batch_first=True, dropout=0.15 if NUM_LAYERS > 1 else 0.0
            )
            self.head = nn.Sequential(
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Linear(h, HORIZON * 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            o, _ = self.lstm(x)
            y = self.head(o[:, -1, :])
            return y.view(-1, self.horizon, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CandleSeqLSTM(hid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_epoch(train: bool) -> float:
        model.train(train)
        total, n = 0.0, 0
        loader = dl_tr if train else dl_va
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += float(loss.item()) * len(xb)
            n += len(xb)
        return total / max(n, 1)

    best = 1e9
    best_state = None
    for ep in range(1, epochs + 1):
        tr_loss = run_epoch(True)
        va_loss = run_epoch(False)
        print(f"Epoch {ep:>2}/{epochs}  train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")
        if va_loss < best:
            best = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    ckpt = OUT_DIR / "lstm.pt"
    payload = {
        "state_dict": best_state,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "hidden": hid,
        "num_layers": NUM_LAYERS,
        "in_dim": IN_DIM,
        "trained_at": datetime.now().isoformat(),
        "tickers_used": len(tickers),
        "n_windows": len(X),
        "best_val_mse": float(best),
    }
    torch.save(payload, ckpt)
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "best_val_mse": float(best),
                "n_windows": len(X),
                "lookback": LOOKBACK,
                "horizon": HORIZON,
                "in_dim": IN_DIM,
                "hidden": hid,
            },
            indent=2,
        )
    )
    print(f"\nSaved -> {ckpt}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LSTM candle-sequence model")
    ap.add_argument("--tickers", nargs="+", default=None, help="Ticker symbols (default: trainer universe)")
    ap.add_argument("--lookback-years", type=int, default=6, help="Years of yfinance history")
    ap.add_argument("--epochs", type=int, default=12, help="More epochs usually help val MSE (diminishing returns)")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--hidden",
        type=int,
        default=None,
        help=f"LSTM hidden size (default {HIDDEN})",
    )
    ap.add_argument(
        "--sample-step",
        type=int,
        default=DEFAULT_SAMPLE_STEP,
        help=f"Stride between windows per ticker (default {DEFAULT_SAMPLE_STEP})",
    )
    ap.add_argument("--quick", action="store_true", help="Fewer tickers & epochs for smoke test")
    ap.add_argument(
        "--all-tickers",
        action="store_true",
        help="Use full DEFAULT_TICKERS list (~180) instead of first 120",
    )
    args = ap.parse_args()

    if args.quick:
        tks = (args.tickers or DEFAULT_TICKERS)[:18]
        epochs = min(args.epochs, 4)
        ly = min(args.lookback_years, 4)
        print("Quick mode: small ticker set, 4 epochs max, 4Y history")
    else:
        base = args.tickers or DEFAULT_TICKERS
        tks = base if (args.all_tickers or args.tickers) else base[:120]
        epochs = args.epochs
        ly = args.lookback_years

    print(f"LOOKBACK={LOOKBACK}  HORIZON={HORIZON}  IN_DIM={IN_DIM}  tickers={len(tks)}")
    train_main(tks, ly, epochs, args.batch_size, args.lr, args.sample_step, hidden=args.hidden)


if __name__ == "__main__":
    main()
