"""
Decision thresholds — defaults + research sensitivity scaffolding.

The institutional audit flagged the four primary entry thresholds
(``min_p_up_long``, ``max_p_up_short``, ``long_quintile_min``,
``short_quintile_max``) as unjustified hand-tuned magic numbers. The fix
isn't to keep tuning them by feel — it's to (a) surface them as a
single documented config object so callers can override coherently and
(b) ship a sweep helper that runs a small grid of plausible values and
reports the joint sensitivity, so the team can *measure* how much P&L
depends on the chosen point.

This module deliberately does **not** auto-pick "best" thresholds. Doing
so on a single backtest is the textbook overfit pattern — pick the value
that happened to win on the in-sample data and ship it. The right
workflow is:

  1. Sweep on an in-sample window (e.g. 2018-2022).
  2. Pick a parameter neighborhood with broadly stable Sharpe / hit-rate.
  3. Lock those thresholds, re-run on a held-out OOS window (e.g. 2023+).
  4. Compare PSR/DSR with N=size_of_sweep trials so the OOS result is
     deflated correctly for the search.

Usage
-----

    from portfolio.decision_thresholds import (
        DEFAULT_DECISION_CFG, THRESHOLD_RANGES, sweep_decision_cfgs,
    )

    base = {**DEFAULT_DECISION_CFG, "lookback_years": 5}
    for cfg, label in sweep_decision_cfgs(base, THRESHOLD_RANGES):
        result = run_dynamic(**cfg)
        print(label, summarize(result))

Audit pointer: the absolute-vs-cohort fallback inside ``decide_ticker``
(see ``QUINTILE_MIN_COHORT`` in ``portfolio/decisions.py``) was added at
the same time as this module so the small-cohort case has a documented
path rather than a silent skip.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Iterator


# ── defaults documented in one place ─────────────────────────────────────────

DEFAULT_DECISION_CFG: dict[str, Any] = {
    # Entry probability gates ── calibrated p_up_20d from ML model.
    # 0.58 is approximately one-standard-deviation above the 0.50 base
    # rate when the model's calibrated reliability slope is ~1; pick
    # higher for more selective entry, lower for more turnover.
    "min_p_up_long": 0.60,
    "max_p_up_short": 0.38,

    # Quintile rank gates (when the day's scored cohort is large enough
    # for cross-sectional rank to be meaningful — see
    # ``QUINTILE_MIN_COHORT`` in ``decisions.py``).
    "long_quintile_min": 4,
    "short_quintile_max": 2,

    # Absolute-threshold buffers used when the cohort is too small for
    # quintile rank. The buffer adds tightness on top of the absolute
    # gates so small-cohort entries are stricter than full-rank entries.
    "min_p_up_long_abs_buffer": 0.04,
    "max_p_up_short_abs_buffer": 0.04,

    # Risk gates / sizing.
    "regime_filter": True,
    "long_entry_requires_bull_regime": True,
    "exit_long_when_regime_not_bull": True,
    "enable_short": False,
    "risk_limits_sector_on_margin": True,
    "risk_limits_beta_on_margin": True,
    "max_positions": 10,
    "position_frac": 0.10,
    "exit_p_up_long": 0.30,
    "score_exit_long_only_bear_regime": True,
    "min_hold_days_before_score_exit_long": 25,
    "exit_p_up_short": 0.48,
    "short_exit_p_up_relative_to_entry": True,
    "short_exit_p_up_delta": 0.10,
    "short_entry_requires_bear_regime": True,
    "cover_short_when_regime_not_bear": True,
    "min_hold_days_before_score_exit_short": 5,
    "stop_loss_pct": 0.20,
    "use_trailing_stop": True,
    "trailing_stop_pct": 0.12,
    "trail_activate_profit_pct": 0.10,
    "use_take_profit": False,
    "take_profit_pct": 0.20,
    "max_hold_days": 28,
    "estimated_hold_days": 20,

    # Costs (T212 CFD: 5× leverage long & short, overnight on exposure).
    "commission_bps": 0.0,
    "slippage_bps": 5.0,
    "cfd_leverage": 5.0,
    "short_leverage": 5.0,
    "long_leverage": 5.0,
    "overnight_interest_bps_annual": 400.0,
    "borrow_bps_annual": 400.0,

    # ATR / vol-target risk knobs (0 disables — see broker.apply_decisions).
    "atr_stop_mult": 0.0,
    "atr_tp_mult": 0.0,
    "atr_min_stop_pct": 0.04,
    "atr_max_stop_pct": 0.30,
    "vol_target_annual_pct": 0.0,
    "vol_size_floor": 0.25,
    "vol_size_cap": 2.0,
}


# ── recommended sweep ranges ─────────────────────────────────────────────────
#
# Two recipes are shipped:
#
# * ``THRESHOLD_RANGES`` — the **univariate** recipe, used by
#   ``univariate_sensitivity_sweep``. Each axis is varied independently
#   while the others stay at ``DEFAULT_DECISION_CFG``. Total trials =
#   ``sum(len(values) for values in axes.values())`` — typically 25-30
#   backtests, fast to run, and the right baseline diagnostic for "does
#   the strategy survive a small wiggle in this knob?".
#
# * ``JOINT_SWEEP_RANGES`` — a **deliberately tiny** cartesian recipe
#   for callers who want to study interactions between knobs. Three axes
#   at three values each is the upper bound here (27 trials). Anything
#   bigger is overfitting territory and the DSR with N=27 is already
#   sobering.

THRESHOLD_RANGES: dict[str, list[Any]] = {
    "min_p_up_long":     [0.54, 0.56, 0.58, 0.60, 0.62],
    "max_p_up_short":    [0.38, 0.40, 0.42, 0.44, 0.46],
    "long_quintile_min": [3, 4, 5],
    "short_quintile_max": [1, 2, 3],
    "stop_loss_pct":     [0.10, 0.15, 0.20, 0.25],
    "take_profit_pct":   [0.15, 0.20, 0.25, 0.30, 0.35],
    "position_frac":     [0.05, 0.08, 0.10, 0.12, 0.15],
}

JOINT_SWEEP_RANGES: dict[str, list[Any]] = {
    "min_p_up_long":   [0.56, 0.58, 0.60],
    "stop_loss_pct":   [0.15, 0.20, 0.25],
    "position_frac":   [0.08, 0.10, 0.12],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def sweep_decision_cfgs(
    base: dict[str, Any],
    axes: dict[str, list[Any]],
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield ``(cfg, label)`` pairs over the **cartesian product** of ``axes``.

    Use only on in-sample data; pair OOS results with
    ``deflated_sharpe_ratio(returns, n_trials=n_trials_for(axes))`` so
    selection bias is correctly deflated. For routine sensitivity checks
    prefer ``univariate_sensitivity_sweep``.
    """
    keys = list(axes.keys())
    grids = [axes[k] for k in keys]
    for combo in product(*grids):
        cfg = dict(base)
        parts: list[str] = []
        for k, v in zip(keys, combo):
            cfg[k] = v
            parts.append(f"{k}={v}")
        yield cfg, "__".join(parts)


def univariate_sensitivity_sweep(
    base: dict[str, Any],
    axes: dict[str, list[Any]],
) -> Iterator[tuple[dict[str, Any], str]]:
    """Vary ONE axis at a time while keeping the others at ``base``.

    Yields N total cfgs where ``N = sum(len(v) for v in axes.values())``.
    The yielded ``base`` value (e.g. ``min_p_up_long=0.58``) appears once
    per axis, so the caller can read univariate slope curves cleanly.
    """
    for axis, values in axes.items():
        for v in values:
            cfg = dict(base)
            cfg[axis] = v
            yield cfg, f"{axis}={v}"


def n_trials_for(axes: dict[str, list[Any]]) -> int:
    """Cartesian-product size — pass to ``deflated_sharpe_ratio`` after a
    cartesian sweep so the DSR correctly accounts for the search."""
    n = 1
    for v in axes.values():
        n *= max(1, len(v))
    return n


def n_univariate_trials_for(axes: dict[str, list[Any]]) -> int:
    """Total trials of ``univariate_sensitivity_sweep`` (``sum`` not
    ``product``). Use this for DSR when running univariate sweeps."""
    return sum(max(1, len(v)) for v in axes.values())
