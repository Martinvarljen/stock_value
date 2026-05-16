"""Daily paper-trading agent: stateless runs, file-based memory.

This package intentionally exports nothing at import time so that ``import
portfolio`` stays cheap (no numpy/pandas/yfinance side effects). Import the
specific submodules you need:

  - ``portfolio.decision_schema`` — DecisionReport, parse_rating, render_decision
  - ``portfolio.memory_log``       — DecisionMemoryLog, MemoryEntry
  - ``portfolio.reflection``       — OutcomeContext, reflect_on_outcome
  - ``portfolio.decisions``        — Action, TickerDecision, decide_universe, ...
"""
