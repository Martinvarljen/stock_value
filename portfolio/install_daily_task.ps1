# Register Windows scheduled task: paper agent after US close (~22:30 local time, Mon–Fri).
# Run once:  powershell -ExecutionPolicy Bypass -File portfolio\install_daily_task.ps1
# Remove:   Unregister-ScheduledTask -TaskName "FinancePaperDaily" -Confirm:$false

$ErrorActionPreference = "Stop"
$taskName = "FinancePaperDaily"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $PSScriptRoot "run_daily_scheduled.ps1"

if (-not (Test-Path $runner)) {
    throw "Missing runner: $runner"
}

$ps = (Get-Command powershell.exe).Source
$arg = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""

$action = New-ScheduledTaskAction -Execute $ps -Argument $arg -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "22:30"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Paper trade: research_ls daily_run after US market close (local 22:30, Mon-Fri)."

Write-Host "Registered scheduled task: $taskName"
Write-Host "  When: Mon-Fri at 22:30 (PC local time; ~30 min after US cash close in summer from Slovenia)"
Write-Host "  Runs: $runner"
Write-Host "  Logs: portfolio\data\logs\"
Write-Host ""
Write-Host "Test now:  Start-ScheduledTask -TaskName $taskName"
Write-Host 'Remove:    Unregister-ScheduledTask -TaskName FinancePaperDaily -Confirm:$false'
