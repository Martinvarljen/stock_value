# Paper trade — research_ls (long+short 5x), same rules as winning agent backtest
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python portfolio/daily_run.py @args
