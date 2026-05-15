# Backtest daily agent vs SPY (build universe files first for full history)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python portfolio/backtest.py --from-year 2015 --max-tickers 100 --signal-step 5 @args
