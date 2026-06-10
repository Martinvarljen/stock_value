# Strategy map — which script is “the” strategy?

This repo has **one production strategy** and several **research backtests**.  
If you only paper-trade `research_ls`, use **only the production lane** below.

## Production lane (paper + truth)

| What | Command | Code |
|------|---------|------|
| **Daily paper trade** | `.\portfolio\run_daily.ps1` | `portfolio/daily_run.py` |
| **Historical test (same rules)** | `.\portfolio\run_backtest.ps1` | `portfolio/backtest.py` |
| **Forward OOS track vs SPY** | `.\portfolio\run_paper_report.ps1` | `portfolio/paper_oos.py` |
| **Config** | `portfolio/config.json` | profile `research_ls`, `universe_source: pit_filter` |
| **One-time / refresh setup** | `.\portfolio\setup_institutional.ps1` | ML baseline + OOS report |
| **OOS validation (freeze + test)** | `python tools/run_oos_validation.py` | See below |

**Rule:** If it is not `portfolio/daily_run.py` or `portfolio/backtest.py`, it is **not** your live book.

### OOS validation workflow

1. **Lock config** (snapshot of `config.json` → `portfolio/config.frozen.json`):
   `python tools/run_oos_validation.py --train-to 2022 --oos-from 2023 --oos-to 2026 --skip-is`
   Or after a sweep: `python tools/run_agent_threshold_sweep.py ... --write-frozen portfolio/config.frozen.json`
2. **OOS backtest only** uses frozen file: `python portfolio/backtest.py --from-year 2023 --to-year 2026 --frozen-config portfolio/config.frozen.json`
3. **Forward paper**: `python portfolio/daily_run.py` + `portfolio/data/paper_oos/report.md`
4. Reports include **calendar-year returns** (red flag if only 1–2 years beat SPY).

## Shared infrastructure (not separate strategies)

| Module | Role |
|--------|------|
| `stock_analyzer/` + `projection/` | Signals (ML `p_up_20d`, classification) |
| `portfolio/decisions.py` | Entries, exits, shorts, cover |
| `portfolio/broker.py` | Fills, 5× CFD, overnight, slippage |
| `backtesting/regime.py` | SPY 200d MA regime |
| `backtesting/yearly_top100_universe.py` | Universe lists |
| `backtesting/sp500_pit_universe.py` | PIT S&P filter (`pit_filter`) |

## Research / legacy lane (do not confuse with paper)

Documented in `backtesting/legacy/README.md`.

| Script | Purpose |
|--------|---------|
| `backtesting/dynamic_portfolio_backtest.py` | Separate ML sim (breakout / rank); **≠ agent** |
| `backtesting/strategy_backtest.py` | Tier/classification signal research |
| `backtesting/vector_engine.py` | Vectorised experiments |
| `backtesting/market_neutral_backtest.py` | Market-neutral experiment |
| `tools/run_threshold_sweep.py` | Threshold sweep on **dynamic** sim only (legacy) |

**Threshold tuning for the agent:** use `tools/run_agent_threshold_sweep.py` (calls `portfolio/backtest.py`).

## Reports

| File pattern | Source |
|--------------|--------|
| `portfolio_agent_report_*.html` | **Agent** backtest (`portfolio/backtest.py`) |
| `portfolio/data/paper_oos/report.md` | **Forward** paper OOS |
| `strategy_dynamic_*_vs_spy_*.html` | **Dynamic** sim (research) |

## Branches

| Branch | Note |
|--------|------|
| `main` | Production agent + `research_ls` |
| `mama-racunlanik-15-17.05` | Old Mac experiment; do not merge for paper |
