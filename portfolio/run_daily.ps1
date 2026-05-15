# Morning paper-trading run (stateless; state in portfolio/data/)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python portfolio/daily_run.py @args
