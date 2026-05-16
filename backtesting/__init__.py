"""
backtesting — strategy backtesting for the trading pipeline.

Modules
-------
* ``dynamic_portfolio_backtest`` — production multi-asset event-driven
  simulator. ``run_dynamic`` is the entry point. Models t+1 open fills,
  per-leg commissions + slippage, short borrow, regime, ATR/vol-target
  sizing.
* ``strategy_backtest`` — checkpoint-style fundamental backtest (used
  for valuation-driven research; PIT fundamentals via the
  ``fundamentals_source`` argument on ``reconstruct_data_at``).
* ``performance_metrics`` — CAGR, Sharpe, Sortino, Probabilistic /
  Deflated Sharpe, t-stat, max drawdown.
* ``regime`` — SPY trend regime + abstain-on-unknown.
* ``ml_quant`` — calibrated/composite-aware quintile ranking.
* ``yearly_top100_universe`` — legacy universe builder (survivorship-biased).
* ``sp500_pit_universe`` — point-in-time S&P 500 membership reconstruction.
* ``vector_engine`` — DEPRECATED. Use ``dynamic_portfolio_backtest`` instead.
"""
