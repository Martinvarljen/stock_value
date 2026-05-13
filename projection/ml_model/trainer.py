"""
trainer.py — Train LightGBM projection models on historical price data.

Trains binary classifiers for P(up 20d), P(up 60d), P(up 120d).
Uses extended OHLCV technical features (RSI, returns, vol, ATR, Bollinger %B,
MACD vs price, MA spread, intraday range, volume).

Run:
    cd Finance
    python projection/ml_model/trainer.py
    python projection/ml_model/trainer.py --tickers AAPL MSFT --lookback 8 --sample-step 1

Output:
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

from ml_model.features import TECH_FEATURES, extract_historical_features, MIN_OHLCV_BARS

MODELS_DIR = Path(__file__).parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS = [20, 60, 120]
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


# ── data collection ────────────────────────────────────────────────────────────

def collect_training_data(
    tickers: list[str],
    lookback_years: int = 5,
    sample_step: int = DEFAULT_SAMPLE_STEP,
) -> pd.DataFrame:
    """
    Download historical OHLCV for each ticker and build training rows.
    Each row = technical features at date T + forward return labels.
    """
    import yfinance as yf

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=lookback_years * 365 + max(HORIZONS) + MIN_OHLCV_BARS + 30)

    all_rows: list[dict] = []

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

            # Need enough look-forward room
            max_fwd = int(max(HORIZONS) * 1.5)
            tradeable_dates = hist.index[MIN_OHLCV_BARS:-max_fwd:sample_step]

            if len(tradeable_dates) == 0:
                print("- too short")
                continue

            close = hist["Close"]
            added = 0

            for date in tradeable_dates:
                feat = extract_historical_features(hist, date)
                if feat is None:
                    continue

                price_now = float(close.loc[:date].iloc[-1])
                row = {"ticker": ticker, "date": date}
                row.update(feat)

                valid = True
                for h in HORIZONS:
                    fwd_dt   = date + timedelta(days=int(h * 365 / 252))
                    future   = hist[hist.index >= fwd_dt]
                    if future.empty:
                        valid = False
                        break
                    p_future = float(future["Close"].iloc[0])
                    ret      = (p_future - price_now) / price_now
                    row[f"return_{h}d"]   = ret
                    row[f"target_up_{h}d"] = int(ret > 0)

                if valid:
                    all_rows.append(row)
                    added += 1

            print(f"-> {added} samples")

        except Exception as e:
            print(f"- error: {e}")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal training samples: {len(df):,}  across {df['ticker'].nunique()} tickers")
    return df


# ── training ───────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    lookback_years: int = 5,
    save: bool = True,
    sample_step: int = DEFAULT_SAMPLE_STEP,
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
    from sklearn.metrics import roc_auc_score

    if df is None:
        print("Collecting training data...")
        df = collect_training_data(tickers or DEFAULT_TICKERS, lookback_years, sample_step)

    if df.empty:
        print("No training data available.")
        return {}

    feat_cols = TECH_FEATURES   # only technical — available historically
    models    = {}
    metrics   = {}

    print("\nTraining models...")

    for h in HORIZONS:
        target_col = f"target_up_{h}d"
        sub = df[feat_cols + [target_col]].dropna()

        if len(sub) < 200:
            print(f"  {h}d: skipped (only {len(sub)} samples)")
            continue

        X_df = sub[feat_cols]
        y = sub[target_col].values.astype(int)

        # Time-series CV
        tscv = TimeSeriesSplit(n_splits=5)
        cv_aucs = []

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

        for train_idx, val_idx in tscv.split(X_df):
            m = lgb.LGBMClassifier(**lgbm_params)
            m.fit(X_df.iloc[train_idx], y[train_idx])
            prob = m.predict_proba(X_df.iloc[val_idx])[:, 1]
            cv_aucs.append(roc_auc_score(y[val_idx], prob))

        mean_auc = float(np.mean(cv_aucs))
        std_auc = float(np.std(cv_aucs))

        # Final model on all data
        final = lgb.LGBMClassifier(**lgbm_params)
        final.fit(X_df, y)

        models[h] = final
        metrics[h] = {
            "auc_mean":  round(mean_auc, 4),
            "auc_std":   round(std_auc, 4),
            "n_samples": len(sub),
            "n_tickers": int(df["ticker"].nunique()),
            "features":  feat_cols,
        }
        print(f"  {h}d:  AUC {mean_auc:.3f} ± {std_auc:.3f}   n={len(sub):,}")

        if save:
            import joblib
            path = MODELS_DIR / f"lgbm_{h}d.pkl"
            joblib.dump(final, path)
            print(f"       saved -> {path.name}")

    if save and metrics:
        meta = {
            "trained_at": datetime.now().isoformat(),
            "feature_cols": feat_cols,
            "horizons": HORIZONS,
            "lookback_years": lookback_years,
            "sample_step": sample_step,
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
    args = parser.parse_args()

    tickers = args.tickers or DEFAULT_TICKERS
    print(f"Training on {len(tickers)} tickers, {args.lookback}Y lookback, sample_step={args.sample_step}")
    print(f"Output -> {MODELS_DIR}\n")

    models = train(
        tickers=tickers,
        lookback_years=args.lookback,
        save=not args.no_save,
        sample_step=args.sample_step,
    )

    if models:
        print(f"\nDone. {len(models)} model(s) trained.")
        print("Run the dashboard to use them: streamlit run dashboard/app.py")
    else:
        print("Training failed - check errors above.")


if __name__ == "__main__":
    main()
