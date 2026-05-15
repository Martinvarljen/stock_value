# Research & ML pipeline notes

## Recommended data sources (yfinance vs vendors vs brokers)

**Two layers:** one stack for **research** (prices, fundamentals, screening), and your **broker only for execution**.

1. **Research & this app (replace or complement yfinance)**  
   Use **one paid market-data + fundamentals API** that covers the exchanges you trade (US + EU, etc.). Examples people commonly use in apps: **Polygon** or **Tiingo** for prices; **Financial Modeling Prep** or **EOD Historical Data** for fundamentals (some vendors bundle both). There is no perfect free replacement for serious DCF + broad global listings.

2. **Trading 212**  
   Use their **official Public API** only for what it is built for: **account, positions, orders, history** — not as your main historical fundamentals database. Docs: https://docs.trading212.com/api (beta; Invest / Stocks ISA).

3. **Practical rule**  
   **Signals and valuation from your data vendor (or yfinance as a rough, non-critical layer); orders and portfolio sync from Trading 212.** Do not mix broker rate limits and instrument metadata with bulk research pipelines.

---

## Dolt data (StockMarketTool-style, optional)

Higher-quality OHLCV than yfinance alone. Setup:

1. Install [Dolt](https://docs.dolthub.com/) and clone `post-no-preference/stocks` + `earnings` (see [StockMarketTool README](https://github.com/physicslifter/StockMarketTool)).
2. In that folder: `dolt sql-server`
3. `pip install mysql-connector-python`
4. `python projection/data/setup_dolt_cache.py`
5. Train: `python projection/ml_model/trainer.py --data-source dolt-feather --dolt-top 500 --purged-cv --optuna --label-mode alpha_vs_spy`

Feather default: `projection/data/cache/all_ohlcv_no_etfs.feather`. Override with `FINANCE_DOLT_FEATHER` or `--dolt-feather`.

**Purged CV** (default on) and **Optuna** (`--optuna`) follow StockMarketTool’s `Model.split_data` / `tune_params` ideas without changing live inference schema.

---

## Feature leakage (LightGBM training)

Training rows are built so that technical features at calendar date **T** use only
OHLCV history **up to and including** **T**. Forward labels use the first close on
or after **T + horizon** (calendar-day approximation). If you change bar alignment,
re-audit this invariant before trusting backtests.

## Labels

- **`raw` (default):** `target_up = 1` if stock forward total return > 0.
- **`alpha_vs_spy`:** `target_up = 1` if excess return vs SPY over the same window
  is positive: `(1+r_stock)/(1+r_spy) - 1 > 0`. Requires a valid SPY series for that
  row; otherwise falls back to raw.

Train with: `python projection/ml_model/trainer.py --label-mode alpha_vs_spy`

## Calibration

After time-series CV, out-of-fold predicted probabilities are fit with **sklearn**
`IsotonicRegression` and saved as `calibrator_{H}d.pkl`. Inference applies the
calibrator on top of the production `LGBMClassifier` probabilities. Retrain after
feature-schema changes.

## Walk-forward evaluation

`python projection/ml_model/evaluate.py` retrains a model per expanding calendar
train window and scores the next year (diagnostic; not identical to the shipped
all-data model). Output: `projection/ml_model/saved_models/evaluation_report.json`.

## Data quality (yfinance)

**yfinance** is convenient but imperfect (splits, restatements, survivorship, delay).
Use filings or a second vendor for material decisions.

## Not financial advice

This repository is for education and research. Models and DCF outputs are not
investment recommendations.

## Future extensions (not implemented in code)

- Sector-stratified or per-regime LightGBM heads when training rows carry a sector id.
- Formal conformal or split-conformal intervals instead of the heuristic P(up) band
  shown in the dashboard.
- Meta-learned blend weights (stacking) for ML vs rule vs news, fit on validation years.
- Optional dashboard authentication and rate limits for non-local deployments.
