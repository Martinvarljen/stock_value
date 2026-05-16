"""
trainer.py — Train LightGBM projection models on historical price data.

Trains binary classifiers for P(up 5d), P(up 20d), etc.
Uses extended OHLCV technical features plus long-horizon / crisis-style proxies
(drawdown, vol stress, panic days) and SPY-relative regime. Feature schema v3 — retrain after upgrading code.

Run:
    cd Finance
    python projection/ml_model/trainer.py
    python projection/ml_model/trainer.py --tickers AAPL MSFT --lookback 8 --sample-step 1
    python projection/ml_model/trainer.py --label-mode alpha_vs_spy --deep
    python projection/ml_model/trainer.py --purged-cv --optuna --optuna-trials 40
    python projection/data/setup_dolt_cache.py
    python projection/ml_model/trainer.py --data-source dolt-feather --dolt-top 500 --purged-cv --optuna
    python projection/ml_model/trainer.py --save-training-csv train_rows.csv
    python projection/ml_model/evaluate.py --quick   # walk-forward report

Output:
    projection/ml_model/saved_models/lgbm_5d.pkl
    projection/ml_model/saved_models/lgbm_20d.pkl
    projection/ml_model/saved_models/lgbm_60d.pkl
    projection/ml_model/saved_models/lgbm_120d.pkl
    projection/ml_model/saved_models/metadata.json
"""

import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup
_root = Path(__file__).resolve().parents[2]  # Finance/ (contains stock_analyzer, projection)
for _p in [str(_root / "stock_analyzer"), str(_root / "projection")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ml_model.features import (
    TECH_FEATURES,
    FEATURE_SCHEMA_VERSION,
    extract_historical_features,
    MIN_OHLCV_BARS,
)
from ml_model.splits import (
    chronological_holdout_split,
    grouped_purged_time_series_splits,
    purged_time_series_splits,
)
from ml_model.optuna_tune import tune_lgbm_classifier

MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS = [5, 20, 60, 120]
DEFAULT_SAMPLE_STEP = 3  # one row every N trading days (smaller = more data, slower)

# Diversified large-cap training universe (deduped)
_DEFAULT_CORE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "ADBE", "CRM", "INTC", "AMD",
    "QCOM", "TXN", "AMAT", "MU", "AVGO", "LRCX", "KLAC", "SNPS", "CDNS", "IBM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "USB", "PNC", "COF",
    # Healthcare
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "BMY", "GILD", "AMGN", "BIIB",
    # Consumer
    "PG", "KO", "PEP", "WMT", "COST", "HD", "NKE", "MCD", "SBUX", "TGT",
    # Industrials
    "GE", "HON", "CAT", "DE", "UNP", "RTX", "LMT", "BA", "MMM", "UPS",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "HAL",
    # Materials & Utilities
    "LIN", "APD", "ECL", "SHW", "NEE", "DUK", "SO", "D", "AEP", "EXC",
    # Real estate & Other
    "O", "AMT", "PLD", "EQIX", "SPG", "WELL", "AVB", "EQR", "DLR", "PSA",
]

_DEFAULT_EXTRA = [
    "T", "VZ", "CMCSA", "DIS", "NFLX", "ORLY", "LOW", "TJX", "BKNG", "MAR",
    "CME", "ICE", "SCHW", "TFC", "AIG", "MET", "PRU", "ALL", "TRV", "AON",
    "ISRG", "SYK", "BSX", "MDT", "DHR", "TMO", "ABT", "CVS", "CI", "ELV",
    "NOW", "PANW", "CRWD", "ZS", "DDOG", "V", "MA", "AXP", "MCO", "SPGI",
    "ETN", "EMR", "PH", "ROK", "ITW", "FDX", "NSC", "CSX", "WM", "RSG",
    "FCX", "NEM", "AA", "DOW", "CTVA", "FANG", "HES", "DVN", "BKR", "FTI",
]

DEFAULT_TICKERS = list(dict.fromkeys(_DEFAULT_CORE + _DEFAULT_EXTRA))


def _fetch_spy_close(start_dt: datetime, end_dt: datetime) -> pd.Series | None:
    """Daily SPY close aligned to training window (market regime / crisis context)."""
    import yfinance as yf

    try:
        sh = yf.Ticker("SPY").history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if sh.empty or "Close" not in sh.columns:
            return None
        if sh.index.tz is not None:
            sh = sh.copy()
            sh.index = sh.index.tz_localize(None)
        return sh["Close"].astype(float)
    except Exception:
        return None


# ── data collection ────────────────────────────────────────────────────────────

def _spy_forward_return(
    spy_series: pd.Series | None, asof: pd.Timestamp, fwd_dt: datetime
) -> float | None:
    """Total return on SPY from last close on/before *asof* to first close on/after *fwd_dt*."""
    if spy_series is None or spy_series.empty:
        return None
    try:
        s_asof = spy_series.loc[:asof]
        if s_asof.empty:
            return None
        spy_now = float(s_asof.iloc[-1])
        fut = spy_series[spy_series.index >= fwd_dt]
        if fut.empty:
            return None
        spy_fut = float(fut.iloc[0])
        if spy_now == 0:
            return None
        return (spy_fut - spy_now) / spy_now
    except Exception:
        return None


def collect_training_data(
    tickers: list[str],
    lookback_years: int = 5,
    sample_step: int = DEFAULT_SAMPLE_STEP,
    *,
    label_mode: str = "raw",
    save_training_csv: str | None = None,
) -> pd.DataFrame:
    """
    Download historical OHLCV for each ticker and build training rows.
    Each row = technical features at date T + forward return labels.

    label_mode:
      raw          — target_up = 1 if stock forward return > 0
      alpha_vs_spy — target_up = 1 if (1+r_stock)/(1+r_spy) - 1 > 0 (requires SPY series)
    """
    import yfinance as yf

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_years * 365 + max(HORIZONS) + MIN_OHLCV_BARS + 30)

    all_rows: list[dict] = []

    spy_series = _fetch_spy_close(start_dt, end_dt)
    if spy_series is not None:
        print(f"SPY regime series: {len(spy_series)} days", flush=True)
    else:
        print("SPY unavailable — spy regime features zeroed.", flush=True)

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:>3}/{len(tickers)}] {ticker:<8}", end=" ", flush=True)
        try:
            hist = yf.Ticker(ticker).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                interval="1d",
            )
            if hist.empty:
                print("- no data")
                continue

            hist.index = hist.index.tz_localize(None)

            added_rows = _build_rows_from_hist(
                ticker, hist, sample_step, label_mode=label_mode, spy_series=spy_series
            )
            all_rows.extend(added_rows)
            print(f"-> {len(added_rows)} samples")

        except Exception as e:
            print(f"- error: {e}")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal training samples: {len(df):,}  across {df['ticker'].nunique()} tickers")
    if save_training_csv and not df.empty:
        try:
            df.to_csv(save_training_csv, index=False)
            print(f"Training table saved -> {save_training_csv}", flush=True)
        except OSError as e:
            print(f"Could not save training CSV: {e}", flush=True)
    return df


def _spy_forward_return_positional(
    spy_series: pd.Series | None,
    asof: pd.Timestamp,
    fwd_idx_date: pd.Timestamp,
) -> float | None:
    """SPY total return from the last close on/before ``asof`` to the close
    on ``fwd_idx_date`` (an actual SPY trading date). When ``fwd_idx_date``
    isn't in the SPY series, fall back to the next available bar."""
    if spy_series is None or spy_series.empty:
        return None
    try:
        s_asof = spy_series.loc[:asof]
        if s_asof.empty:
            return None
        spy_now = float(s_asof.iloc[-1])
        fut = spy_series[spy_series.index >= fwd_idx_date]
        if fut.empty:
            return None
        spy_fut = float(fut.iloc[0])
        if spy_now == 0:
            return None
        return (spy_fut - spy_now) / spy_now
    except Exception:
        return None


def _build_rows_from_hist(
    ticker: str,
    hist: pd.DataFrame,
    sample_step: int,
    *,
    label_mode: str,
    spy_series: pd.Series | None,
) -> list[dict]:
    """Shared row builder for yfinance and Dolt OHLCV frames.

    Forward labels are anchored on **trading-day positions** in the actual
    OHLCV frame, not calendar-day deltas. Earlier code did
    ``date + timedelta(days=int(h * 365 / 252))`` and then took the next
    available bar — across holiday weeks the realised horizon could drift
    from ``h`` to ``h+5`` trading days, inflating label noise.
    """
    max_fwd = int(max(HORIZONS) * 1.5)
    tradeable_dates = hist.index[MIN_OHLCV_BARS:-max_fwd:sample_step]
    if len(tradeable_dates) == 0:
        return []

    close = hist["Close"]
    n_total = len(hist)
    rows: list[dict] = []
    for date in tradeable_dates:
        feat = extract_historical_features(hist, date, spy_series)
        if feat is None:
            continue
        try:
            i_now = hist.index.get_loc(date)
        except KeyError:
            continue
        if isinstance(i_now, slice) or hasattr(i_now, "__iter__"):
            # Duplicate timestamps — take the last one to avoid leakage.
            i_now = int(np.where(hist.index == date)[0][-1])
        price_now = float(close.iloc[i_now])
        row = {"ticker": ticker, "date": date}
        row.update(feat)
        valid = True
        for h in HORIZONS:
            i_fwd = i_now + int(h)
            if i_fwd >= n_total:
                valid = False
                break
            p_future = float(close.iloc[i_fwd])
            ret = (p_future - price_now) / price_now
            row[f"return_{h}d"] = ret
            if label_mode == "alpha_vs_spy":
                fwd_idx_date = hist.index[i_fwd]
                r_spy = _spy_forward_return_positional(spy_series, date, fwd_idx_date)
                if r_spy is not None:
                    try:
                        excess = (1.0 + ret) / (1.0 + r_spy) - 1.0
                        row[f"target_up_{h}d"] = int(excess > 0.0)
                    except (ZeroDivisionError, ValueError, OverflowError):
                        row[f"target_up_{h}d"] = int(ret > 0.0)
                else:
                    row[f"target_up_{h}d"] = int(ret > 0.0)
            else:
                row[f"target_up_{h}d"] = int(ret > 0)
        if valid:
            rows.append(row)
    return rows


def collect_training_data_dolt_feather(
    tickers: list[str] | None,
    lookback_years: int = 5,
    sample_step: int = DEFAULT_SAMPLE_STEP,
    *,
    label_mode: str = "raw",
    save_training_csv: str | None = None,
    feather_path: str | Path | None = None,
    dolt_top: int | None = None,
) -> pd.DataFrame:
    """Build training rows from local Dolt feather cache (higher quality than yfinance)."""
    from data.dolt_source import (
        default_feather_path,
        load_ohlcv_feather,
        ticker_histories_from_feather,
        top_liquidity_tickers,
    )

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_years * 365 + max(HORIZONS) + MIN_OHLCV_BARS + 30)
    fpath = Path(feather_path) if feather_path else default_feather_path()

    if tickers is None and dolt_top:
        panel = load_ohlcv_feather(fpath)
        year = end_dt.year - 1
        tickers = top_liquidity_tickers(panel, year, top_n=dolt_top)
        print(f"Dolt universe: top {dolt_top} by {year} dollar volume -> {len(tickers)} tickers", flush=True)
    tickers = tickers or DEFAULT_TICKERS

    spy_series = _fetch_spy_close(start_dt, end_dt)
    print(f"Loading OHLCV from feather: {fpath}", flush=True)
    histories = ticker_histories_from_feather(tickers, start_dt, end_dt, fpath)

    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers, 1):
        hist = histories.get(ticker)
        if hist is None:
            print(f"[{i:>3}/{len(tickers)}] {ticker:<8} - not in feather", flush=True)
            continue
        added_rows = _build_rows_from_hist(
            ticker, hist, sample_step, label_mode=label_mode, spy_series=spy_series
        )
        all_rows.extend(added_rows)
        print(f"[{i:>3}/{len(tickers)}] {ticker:<8} -> {len(added_rows)} samples", flush=True)

    df = pd.DataFrame(all_rows)
    print(f"\nTotal training samples: {len(df):,}  across {df['ticker'].nunique()} tickers")
    if save_training_csv and not df.empty:
        try:
            df.to_csv(save_training_csv, index=False)
            print(f"Training table saved -> {save_training_csv}", flush=True)
        except OSError as e:
            print(f"Could not save training CSV: {e}", flush=True)
    return df


# ── training ───────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    lookback_years: int = 5,
    save: bool = True,
    sample_step: int = DEFAULT_SAMPLE_STEP,
    deep: bool = False,
    *,
    label_mode: str = "raw",
    calibrate: bool = True,
    save_training_csv: str | None = None,
    data_source: str = "yfinance",
    dolt_feather: str | Path | None = None,
    dolt_top: int | None = None,
    purged_cv: bool = True,
    optuna_tune: bool = False,
    optuna_trials: int = 40,
) -> dict:
    """
    Train LightGBM classifiers for each horizon.
    Returns dict {horizon: model}.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        return {}

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.isotonic import IsotonicRegression

    if df is None:
        print(f"Collecting training data (source={data_source})...")
        if data_source == "dolt-feather":
            df = collect_training_data_dolt_feather(
                tickers,
                lookback_years,
                sample_step,
                label_mode=label_mode,
                save_training_csv=save_training_csv,
                feather_path=dolt_feather,
                dolt_top=dolt_top,
            )
        else:
            if data_source != "yfinance":
                print(f"  Unknown data_source={data_source!r}, using yfinance.")
            df = collect_training_data(
                tickers or DEFAULT_TICKERS,
                lookback_years,
                sample_step,
                label_mode=label_mode,
                save_training_csv=save_training_csv,
            )

    if df.empty:
        print("No training data available.")
        return {}

    if deep:
        print("Deep mode: stronger LightGBM (more trees, slower fit).")

    feat_cols = TECH_FEATURES   # only technical — available historically
    models    = {}
    metrics   = {}

    print("\nTraining models...")

    if deep:
        lgbm_params = dict(
            n_estimators=1400,
            learning_rate=0.018,
            max_depth=9,
            num_leaves=96,
            min_child_samples=40,
            min_split_gain=0.002,
            subsample=0.82,
            colsample_bytree=0.82,
            reg_alpha=0.08,
            reg_lambda=0.12,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    else:
        lgbm_params = dict(
            n_estimators=800,
            learning_rate=0.025,
            max_depth=8,
            num_leaves=63,
            min_child_samples=25,
            min_split_gain=0.001,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

    for h in HORIZONS:
        target_col = f"target_up_{h}d"
        need_cols = feat_cols + [target_col, "date"]
        miss = [c for c in need_cols if c not in df.columns]
        if miss:
            print(f"  {h}d: skipped (missing columns {miss})")
            continue

        sub = df[need_cols].dropna().copy()
        sub["date"] = pd.to_datetime(sub["date"], utc=False)
        sub = sub.sort_values(["date", "ticker"] if "ticker" in sub.columns else ["date"])

        if len(sub) < 200:
            print(f"  {h}d: skipped (only {len(sub)} samples)")
            continue

        X_df = sub[feat_cols].reset_index(drop=True)
        y = sub[target_col].values.astype(int)
        dates = sub["date"].reset_index(drop=True)

        horizon_params = dict(lgbm_params)
        if optuna_tune:
            tr_sub, va_sub = chronological_holdout_split(sub, val_fraction=0.15, horizon_days=h)
            if len(va_sub) >= 100 and len(tr_sub) >= 200:
                tuned = tune_lgbm_classifier(
                    tr_sub[feat_cols].values,
                    tr_sub[target_col].values.astype(int),
                    va_sub[feat_cols].values,
                    va_sub[target_col].values.astype(int),
                    n_trials=optuna_trials,
                    n_estimators=1400 if deep else 1000,
                )
                if tuned:
                    horizon_params = {**lgbm_params, **tuned}
                    print(f"       Optuna tuned ({optuna_trials} trials)")

        cv_aucs: list[float] = []
        cv_briers: list[float] = []
        oof_pred = np.zeros(len(X_df), dtype=float)
        oof_filled = np.zeros(len(X_df), dtype=bool)

        if purged_cv:
            # Prefer the group-aware splitter when a ``ticker`` column is
            # available — single-name autocorrelation across the gap is the
            # usual leakage path on cross-sectional panels.
            if "ticker" in sub.columns:
                groups = sub["ticker"].reset_index(drop=True)
                split_iter = grouped_purged_time_series_splits(
                    dates, groups, n_splits=5, horizon_days=h
                )
            else:
                split_iter = purged_time_series_splits(dates, n_splits=5, horizon_days=h)
        else:
            split_iter = TimeSeriesSplit(n_splits=5).split(X_df)

        for train_idx, val_idx in split_iter:
            m = lgb.LGBMClassifier(**horizon_params)
            m.fit(X_df.iloc[train_idx], y[train_idx])
            prob = m.predict_proba(X_df.iloc[val_idx])[:, 1]
            oof_pred[val_idx] = prob
            oof_filled[val_idx] = True
            cv_aucs.append(roc_auc_score(y[val_idx], prob))
            cv_briers.append(brier_score_loss(y[val_idx], prob))

        if not cv_aucs:
            print(f"  {h}d: skipped (CV produced no valid folds)")
            continue

        mean_auc = float(np.mean(cv_aucs))
        std_auc = float(np.std(cv_aucs))
        mean_brier = float(np.mean(cv_briers))

        final = lgb.LGBMClassifier(**horizon_params)
        final.fit(X_df, y)

        models[h] = final
        metrics[h] = {
            "auc_mean": round(mean_auc, 4),
            "auc_std": round(std_auc, 4),
            "brier_mean": round(mean_brier, 5),
            "n_samples": len(sub),
            "n_tickers": int(df["ticker"].nunique()),
            "features": feat_cols,
        }
        print(
            f"  {h}d:  AUC {mean_auc:.3f} ± {std_auc:.3f}  "
            f"Brier {mean_brier:.4f}  n={len(sub):,}"
        )

        if save:
            import joblib

            path = MODELS_DIR / f"lgbm_{h}d.pkl"
            joblib.dump(final, path)
            print(f"       saved -> {path.name}")

            if calibrate and int(oof_filled.sum()) > 200:
                try:
                    mask = oof_filled
                    ir = IsotonicRegression(out_of_bounds="clip")
                    ir.fit(oof_pred[mask], y[mask])
                    cpath = MODELS_DIR / f"calibrator_{h}d.pkl"
                    joblib.dump(ir, cpath)
                    print(f"       calibrator -> {cpath.name}")
                except Exception as e:
                    print(f"       calibrator skipped: {e}")

    if save and metrics:
        has_cal = all((MODELS_DIR / f"calibrator_{h}d.pkl").exists() for h in metrics)
        meta = {
            "trained_at": datetime.now().isoformat(),
            "feature_cols": feat_cols,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "n_tech_features": len(feat_cols),
            "horizons": HORIZONS,
            "lookback_years": lookback_years,
            "sample_step": sample_step,
            "lgbm_deep": deep,
            "label_mode": label_mode,
            "data_source": data_source,
            "purged_cv": purged_cv,
            "optuna_tune": optuna_tune,
            "calibrated": bool(has_cal),
            "evaluation_report": "evaluation_report.json",
            "metrics": {str(k): v for k, v in metrics.items()},
        }
        (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"\nMetadata saved -> {MODELS_DIR / 'metadata.json'}")

    return models


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LightGBM projection models")
    parser.add_argument("--tickers",  nargs="+", default=None,
                        help="Override ticker list (default: built-in S&P 500 subset)")
    parser.add_argument("--lookback", type=int, default=5,
                        help="Years of historical data to use (default: 5)")
    parser.add_argument("--no-save",  action="store_true",
                        help="Train but don't save models to disk")
    parser.add_argument(
        "--sample-step",
        type=int,
        default=DEFAULT_SAMPLE_STEP,
        help=f"Subsampling stride along the trading calendar (default: {DEFAULT_SAMPLE_STEP}; use 1 for densest data)",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Stronger LightGBM (more trees, more regularization) — slower but often better fit",
    )
    parser.add_argument(
        "--label-mode",
        choices=("raw", "alpha_vs_spy"),
        default="raw",
        help="Classification target: raw stock direction vs market-relative (alpha vs SPY)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip isotonic calibration (OOF) model files",
    )
    parser.add_argument(
        "--save-training-csv",
        default=None,
        metavar="PATH",
        help="Write collected training rows to CSV (for evaluate.py / audits)",
    )
    parser.add_argument(
        "--data-source",
        choices=("yfinance", "dolt-feather"),
        default="yfinance",
        help="OHLCV source: yfinance (default) or local Dolt feather cache",
    )
    parser.add_argument(
        "--dolt-feather",
        default=None,
        metavar="PATH",
        help="Feather path (default: projection/data/cache/all_ohlcv_no_etfs.feather)",
    )
    parser.add_argument(
        "--dolt-top",
        type=int,
        default=None,
        metavar="N",
        help="With dolt-feather: train on top N tickers by prior-year dollar volume",
    )
    parser.add_argument(
        "--purged-cv",
        action="store_true",
        default=True,
        help="Use purge+embargo time-series CV (default: on)",
    )
    parser.add_argument(
        "--no-purged-cv",
        action="store_false",
        dest="purged_cv",
        help="Use plain TimeSeriesSplit instead of purged CV",
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Optuna hyperparameter search per horizon (slower, often better)",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=40,
        help="Optuna trials per horizon when --optuna is set",
    )
    args = parser.parse_args()

    tickers = args.tickers
    if tickers is None and args.data_source == "yfinance":
        tickers = DEFAULT_TICKERS
    n_msg = len(tickers) if tickers else f"auto (dolt-top={args.dolt_top})"
    print(f"Training on {n_msg} tickers, {args.lookback}Y lookback, sample_step={args.sample_step}")
    print(f"  label_mode={args.label_mode}  data_source={args.data_source}  purged_cv={args.purged_cv}")
    if args.deep:
        print("  --deep enabled")
    if args.optuna:
        print(f"  --optuna enabled ({args.optuna_trials} trials/horizon)")
    print(f"Output -> {MODELS_DIR}\n")

    models = train(
        tickers=tickers,
        lookback_years=args.lookback,
        save=not args.no_save,
        sample_step=args.sample_step,
        deep=args.deep,
        label_mode=args.label_mode,
        calibrate=not args.no_calibrate,
        save_training_csv=args.save_training_csv,
        data_source=args.data_source,
        dolt_feather=args.dolt_feather,
        dolt_top=args.dolt_top,
        purged_cv=args.purged_cv,
        optuna_tune=args.optuna,
        optuna_trials=args.optuna_trials,
    )

    if models:
        print(f"\nDone. {len(models)} model(s) trained.")
        print("Models are now picked up by projection_engine on the next run.")
    else:
        print("Training failed - check errors above.")


if __name__ == "__main__":
    main()
