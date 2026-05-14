# Long-history ML training (~30y data where Yahoo has it), LSTM + LightGBM deep.
# PC must stay awake. Logs timestamped under saved_models/.
# Schema v3: long-horizon returns, drawdown, vol stress, SPY regime + LSTM lookback 50.
$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logLstm = Join-Path $ProjectRoot "projection\ml_model\saved_models\candle_seq\train_best_lstm_$ts.log"
$logLgbm = Join-Path $ProjectRoot "projection\ml_model\saved_models\train_best_lgbm_$ts.log"
Write-Host "LSTM log: $logLstm" -ForegroundColor Cyan
Write-Host "LGBM log: $logLgbm" -ForegroundColor Cyan
python projection/ml_model/train_candle_sequence.py --all-tickers --lookback-years 28 --epochs 36 --sample-step 1 --batch-size 320 --lr 6e-4 *>&1 | Tee-Object -FilePath $logLstm
$lstmExit = $LASTEXITCODE
if ($lstmExit -ne 0) { Write-Host "LSTM failed exit $lstmExit (see log). Skipping LightGBM." -ForegroundColor Red; exit $lstmExit }
python projection/ml_model/trainer.py --lookback 28 --sample-step 1 --deep *>&1 | Tee-Object -FilePath $logLgbm
Write-Host "Done. Exit LGBM $LASTEXITCODE" -ForegroundColor Green
