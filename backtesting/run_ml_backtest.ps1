# Yearly top-100 strategy backtest using Dolt-trained ML (projection_engine).
# Produces tier stats Excel + animated dynamic portfolio HTML vs SPY.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python backtesting/strategy_backtest.py --yearly-top100 --universe-source pit --strategy ml --market-neutral
