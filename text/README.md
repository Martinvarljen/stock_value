# Finance — stock analysis & paper trading agent

## Start here

**Which script do I run?** → read **[STRATEGY_MAP.md](STRATEGY_MAP.md)**

## Production (research_ls paper strategy)

```powershell
.\portfolio\run_daily.ps1              # daily paper (or Task Scheduler 22:30)
.\portfolio\run_backtest.ps1           # historical agent backtest vs SPY
.\portfolio\run_paper_report.ps1       # forward OOS paper report
.\portfolio\setup_institutional.ps1    # ML baseline + OOS report (occasional)
```

Config: `portfolio/config.json` (profile `research_ls`, PIT universe).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r text\requirements.txt
python backtesting\build_yearly_top100_universe.py --from 2023 --to 2025
```

## Research / legacy backtests

Not used for paper. See `backtesting/legacy/README.md`.

```powershell
python backtesting\strategy_backtest.py          # classification research
python backtesting\dynamic_portfolio_backtest.py # separate ML sim
```

## More docs

- `STOCK_ANALYZER_SPEC.md` — analysis stack
- `STRATEGY_MAP.md` — production vs research paths
