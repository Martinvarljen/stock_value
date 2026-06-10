#!/usr/bin/env pwsh
# Strategy-critical regression suite (no network required).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $py -m unittest `
  tests.test_strategy_regression `
  tests.test_daily_backtest_parity `
  tests.test_broker_costs_and_fills `
  tests.test_exit_policy `
  tests.test_decision_thresholds `
  tests.test_risk_limits `
  tests.test_ml_gates `
  tests.test_backtest_invariants `
  tests.test_trailing_stop `
  tests.test_atomic_io `
  tests.test_analyze_risk_fields `
  tests.test_risk_scaling `
  tests.test_cross_sectional `
  tests.test_leakage_test `
  -v

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "All strategy regression tests passed."
