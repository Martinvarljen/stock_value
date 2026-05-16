# Full ML strategy backtest (today's consolidated settings).
# - Yearly top-100 universe, month-end ML scores, P(up) rank entry (no breakout)
# - 20% stop, 40% take-profit, 130d max hold, SPY regime filter
# - Long P(up)>=52%, short P(up)<=48% (dynamic sim; was long-only before)
# - Animated NAV + trade journal + pipeline map + market-neutral L/S chart
#
# Outputs (tag tp40_m avoids overwriting older same-day runs):
#   strategy_dynamic_ml_tp40_m_vs_spy_YYYYMMDD.html
#   strategy_dynamic_ml_tp40_m_vs_spy_YYYYMMDD_{trades,pipeline}.html
#   strategy_dynamic_ml_tp40_m_vs_spy_YYYYMMDD_trades.json
#   market_neutral_ml_YYYYMMDD.html

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

python backtesting/strategy_backtest.py `
  --yearly-top100 `
  --strategy ml `
  --checkpoint-freq M `
  --lookback 5 `
  --take-profit 0.40 `
  --stop-loss 0.20 `
  --run-tag tp40_m `
  --market-neutral
