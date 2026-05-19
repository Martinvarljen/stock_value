# One-time setup for 9/10 institutional checks: ML baseline + OOS report
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "Building ML feature baseline from daily snapshots..."
python tools/build_feature_baseline_from_snapshots.py

Write-Host "Refreshing paper OOS report..."
python -c "from portfolio.config_loader import load_config; from portfolio.paper_oos import write_oos_report; write_oos_report(load_config())"

Write-Host "Done. Re-run backtest with PIT universe:"
Write-Host "  .\portfolio\run_backtest.ps1"
