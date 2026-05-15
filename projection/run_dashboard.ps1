# Double-click or: powershell -ExecutionPolicy Bypass -File .\run_dashboard.ps1
# Always starts from this script's folder so imports and ML paths resolve.
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot
$env:CI = "true"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Write-Host "Project: $ProjectRoot" -ForegroundColor Green
Write-Host "Open: http://127.0.0.1:8501" -ForegroundColor Cyan
python -m streamlit run (Join-Path $ProjectRoot "dashboard\app.py") @args
