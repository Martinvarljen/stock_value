# Scheduled wrapper: logs output, skips if today's audit already exists (no --force).
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$logDir = Join-Path $repoRoot "portfolio\data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$logFile = Join-Path $logDir "daily_run_$stamp.log"

Set-Location $repoRoot
$runner = Join-Path $repoRoot "portfolio\run_daily.ps1"

# Merge stdout+stderr to log; exit code comes from python only (not stderr warnings).
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $runner *>&1 | Tee-Object -FilePath $logFile
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        "daily_run exited with code $code" | Tee-Object -FilePath $logFile -Append
    }
    exit $code
} catch {
    $_ | Out-File -FilePath $logFile -Append -Encoding utf8
    exit 1
} finally {
    $ErrorActionPreference = $prevEap
}
