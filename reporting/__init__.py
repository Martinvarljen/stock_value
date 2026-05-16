"""
reporting/ — read-only report rendering and narrative generation.

This package contains code that produces HTML reports, weekly summaries,
flow maps, decision-trace diagnostics, and human-readable explanations
of analyses.

**Hot-path constraint** — modules in this package MUST NOT be imported by:

* ``portfolio/decisions.py``
* ``portfolio/daily_run.py``
* ``portfolio/broker.py``
* ``backtesting/dynamic_portfolio_backtest.py``
* ``backtesting/strategy_backtest.py`` (the per-checkpoint scoring loop)

Reporting code is allowed to be imported lazily by the pipeline only when
the caller explicitly opts in (e.g. ``include_explanation=True``). The
guard exists so a perf regression in narrative generation cannot silently
slow down the trading decision path.
"""

from __future__ import annotations
