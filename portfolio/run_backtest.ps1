# Agent backtest vs SPY — research_ls (winning strategy; build universe cache first)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python portfolio/backtest.py --from-year 2019 --max-tickers 100 --signal-step 5 @args
