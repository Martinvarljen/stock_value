# Scheduled wrapper: logs output, skips if today's audit already exists (no --force).
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "portfolio\data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$logFile = Join-Path $logDir "daily_run_$stamp.log"

Set-Location $repoRoot
try {
    & (Join-Path $repoRoot "portfolio\run_daily.ps1") *>&1 | Tee-Object -FilePath $logFile
    exit $LASTEXITCODE
} catch {
    $_ | Out-File -FilePath $logFile -Append -Encoding utf8
    exit 1
}
