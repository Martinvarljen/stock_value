# Legacy / research backtests

These live under `backtesting/` for **experiments**. They are **not** the same as:

- `portfolio/daily_run.py` (paper)
- `portfolio/backtest.py` (agent history)

See `text/STRATEGY_MAP.md` for the production lane.

## Files (research only)

| File | What it tests |
|------|----------------|
| `../dynamic_portfolio_backtest.py` | ML/dynamic portfolio vs SPY (Donchian breakout, optional rank mode) |
| `../strategy_backtest.py` | Classification tiers vs forward returns |
| `../vector_engine.py` | Vectorised backtest engine |
| `../market_neutral_backtest.py` | Market-neutral portfolio sim |
| `../run_vector_backtest.py` | CLI for vector engine |
| `../strategy_stat_tests.py` | Statistical tests on signals |

## Still “core” (used by the agent)

Do **not** treat these as legacy:

- `regime.py`, `ml_quant.py`, `performance_metrics.py`
- `yearly_top100_universe.py`, `sp500_pit_universe.py`
- `checkpoint_cache.py`, `build_yearly_top100_universe.py`

## Tools

| Tool | Simulator |
|------|-----------|
| `tools/run_threshold_sweep.py` | `dynamic_portfolio_backtest` (legacy) |
| `tools/run_agent_threshold_sweep.py` | `portfolio/backtest.py` (production) |

We keep legacy scripts for reference; new work on **paper strategy** should only touch `portfolio/`.
