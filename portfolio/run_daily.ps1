# Paper trade — research_ls (long+short 5x), same rules as winning agent backtest
# Do not use $ErrorActionPreference Stop: Python writes warnings to stderr and PowerShell
# would treat them as failing errors when output is merged (scheduled task wrapper).
Set-Location $PSScriptRoot\..

$env:PYTHONUNBUFFERED = "1"
python portfolio/daily_run.py @args
exit $LASTEXITCODE
