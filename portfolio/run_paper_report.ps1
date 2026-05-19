# Refresh forward OOS paper report (portfolio/data/paper_oos/report.md)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python -c "from portfolio.config_loader import load_config; from portfolio.paper_oos import write_oos_report; p=write_oos_report(load_config()); print('Wrote', p)"
