"""
backtesting  —  Strategy backtesting for the stock_analyzer pipeline.

Modules:
  strategy_backtest   — fundamental classification vs forward returns
  strategy_stat_tests — Donchian-style permutation / walk-forward Sharpe
  vector_engine       — vector backtest: signal @ close, next-open execution, bps costs
  performance_metrics — CAGR, Sharpe, drawdown, etc. from period returns
  run_vector_backtest — CLI wrapper over yfinance + vector_engine
"""
