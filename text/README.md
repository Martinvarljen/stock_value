# Trading strategy stack

Python-based equity trading research stack: point-in-time fundamentals,
LightGBM directional model, paper portfolio with auditable decision log,
and several backtests (vector, checkpoint, dynamic portfolio, market-neutral).

## Project layout

- `stock_analyzer/`: analysis engines (quality / financials / DCF valuation /
  growth / risk / red flags / sector / momentum / classification / technicals
  / market structure / candle patterns / Elliott / data validation)
- `projection/`: ML model + projection engine (LightGBM trainer, predictor,
  features, purged CV splits, optional FinBERT/Claude news sentiment)
- `backtesting/`: strategy backtests (checkpoint forward-return harness,
  vector engine, dynamic portfolio, market-neutral, walk-forward stat tests,
  resumable SQLite checkpoint cache)
- `portfolio/`: paper trading agent — stateless daily run, deterministic
  decision rules, append-only memory log with realised-outcome reflections
- `tests/`: unit tests (no network) for the strategy components

## Outputs

The stack is pure command-line / batch — every artifact is a static file:

| Producer                                | Artifact                                                         |
| --------------------------------------- | ---------------------------------------------------------------- |
| `backtesting/strategy_backtest.py`      | Console tier summary + `strategy_dynamic_{dcf,ml}_vs_spy_YYYYMMDD.html` (animated Plotly) |
| `backtesting/dynamic_portfolio_backtest.py` | Same animated HTML chart on its own                         |
| `backtesting/market_neutral_backtest.py` | Long/short HTML chart                                           |
| `portfolio/backtest.py` (offline agent) | `portfolio_agent_report_YYYYMMDD.html` (decision flow + equity curve), `portfolio_agent_trades_YYYYMMDD.json` |
| `portfolio/daily_run.py`                | `portfolio/data/decision_memory.md` (append-only audit log)     |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r text/requirements.txt
```

## Backtest

Tier-based classification against forward returns, top-100 universe by default:

```powershell
python backtesting\strategy_backtest.py                # DCF tiers
python backtesting\strategy_backtest.py --strategy ml  # ML quintiles
```

Outputs a console tier table plus an animated Plotly file
`strategy_dynamic_{dcf,ml}_vs_spy_YYYYMMDD.html`.

## Paper portfolio

```powershell
python portfolio\daily_run.py                          # uses top-100 universe
python portfolio\daily_run.py AAPL MSFT NVDA           # explicit tickers
```

Each run resolves yesterday's pending decisions against realised returns,
writes a deterministic reflection to `portfolio/data/decision_memory.md`,
then stores today's new decisions as pending.

## ML training

```powershell
python projection\ml_model\trainer.py
python projection\ml_model\trainer.py --purged-cv --optuna
```

## Engine integration status

Every analytical engine now pays rent — none are dashboard-only since the
Streamlit app was retired:

| Engine                          | Wired into                                                              |
| ------------------------------- | ----------------------------------------------------------------------- |
| `market_structure.py`           | ML features v4 (`ms_regime_up/down`, `ms_n_pivots_norm`, `ms_pivot_dist_norm`) |
| `candle_patterns.py`            | ML features v4 (`cand_bias_bull/bear`, `cand_body_pct`, wick pcts)      |
| `elliott_engine.py`             | ML features v4 (`ell_dir_up/down`, `ell_price_vs_fib_norm`) + `trade_setup` memory log |
| `ohlcv_validate.py`             | Daily-run skip rule + backtest checkpoint gate                          |
| `trade_setup_engine.py`         | Memory-log `setup_bias` + `watch_levels` extras                         |

The v4 feature schema is additive: the saved model from v3 still loads and
runs (predictor falls back to 0.0 for unknown columns). To actually train
on the new features, run `python projection/ml_model/trainer.py` — the
trainer reads `TECH_FEATURES` and picks up the additions automatically.

## Roadmap — next workstream

- Train a v4 LightGBM head and compare AUC / forward-return spread against
  the live v3 model on a held-out year.
- Record planned-vs-realised R-multiple in `portfolio.reflection.OutcomeContext`
  using the watch levels we now persist (true backtest of the rule-set).
- Sector-stratified or per-regime LightGBM heads (see `text/RESEARCH_NOTES.md`).
