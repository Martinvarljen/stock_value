"""
performance_metrics.py — Summary stats from an equity curve or period returns.

Research / reporting only (implementation brief 9.2). Not financial advice.

Adds risk-free aware Sharpe, Sortino, and the Bailey & Lopez de Prado
Probabilistic and Deflated Sharpe Ratios — institutional baselines for
deciding whether a Sharpe is statistically distinguishable from zero (or
from the best Sharpe across N tested strategies).

References
----------
* Bailey, D. H. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." Journal of Risk.
* Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  Journal of Portfolio Management.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def _annualized_return(total_return: float, n_periods: int, periods_per_year: float = 252.0) -> float:
    if n_periods < 2 or total_return <= -1:
        return float("nan")
    years = n_periods / periods_per_year
    if years <= 0:
        return float("nan")
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF using ``math.erf`` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── institutional Sharpe machinery ───────────────────────────────────────────

def t_stat_of_sharpe(sharpe_ann: float, n_periods: int, periods_per_year: float = 252.0) -> float:
    """Per Lo (2002): t-stat of an annualised Sharpe is ``SR * sqrt(years)``.

    A t-stat of ~2.0 is the conventional bar for "distinguishable from
    zero" at the 5% level under i.i.d. normal returns. Real returns are
    not i.i.d. or normal, so this is a **lower bound** on the t-stat the
    Probabilistic Sharpe Ratio actually produces.
    """
    if n_periods < 2 or periods_per_year <= 0:
        return float("nan")
    years = n_periods / periods_per_year
    return float(sharpe_ann) * math.sqrt(max(years, 0.0))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    *,
    sr_benchmark: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """PSR = Pr( true Sharpe > sr_benchmark | observed Sharpe ).

    Under non-normal returns (skew & kurtosis ≠ 0,3), the variance of the
    Sharpe estimator inflates. Returns a probability in [0, 1].

    ``sr_benchmark`` is annualised (e.g. 0.0 vs zero-Sharpe; 1.0 vs S&P).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        return float("nan")

    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 1e-12:
        return float("nan")

    sr_period = mu / sd
    sr_period_bench = float(sr_benchmark) / math.sqrt(periods_per_year)

    # Sample skewness and excess kurtosis (Fisher-Pearson, bias-uncorrected
    # forms — adequate for ``n`` typical of multi-year backtests).
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4) - 3.0)

    var_sr_per = (1.0 - skew * sr_period + (kurt_excess / 4.0) * sr_period ** 2) / (n - 1)
    if var_sr_per <= 0:
        return float("nan")
    psr_z = (sr_period - sr_period_bench) / math.sqrt(var_sr_per)
    return float(_normal_cdf(psr_z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    n_trials: int,
    sr_trial_var: float | None = None,
    periods_per_year: float = 252.0,
) -> float:
    """Bailey/Lopez de Prado DSR — the probability that the OBSERVED Sharpe
    survives selection-bias correction across ``n_trials`` candidate
    strategies.

    ``sr_trial_var`` is the variance of annualised Sharpes across trials.
    If unknown, pass ``None`` and we use a conservative default of 1.0
    (i.e. assume Sharpes across trials had std = 1.0).
    """
    if n_trials < 2:
        return probabilistic_sharpe_ratio(returns, periods_per_year=periods_per_year)

    var = 1.0 if sr_trial_var is None else max(float(sr_trial_var), 1e-9)
    # Expected max of N i.i.d. standard-normals ≈ Gumbel approximation.
    em_gamma = 0.5772156649015329  # Euler-Mascheroni
    # E[max] ≈ (1-γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N·e))
    from math import log
    def _qnorm(p: float) -> float:
        # Beasley-Springer-Moro inverse-normal — adequate accuracy here.
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        if p <= phigh:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
        q = math.sqrt(-2.0 * log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)

    n = max(int(n_trials), 2)
    e_max = ((1.0 - em_gamma) * _qnorm(1.0 - 1.0 / n)
             + em_gamma * _qnorm(1.0 - 1.0 / (n * math.e)))
    sr_zero = math.sqrt(var) * e_max  # in annualised units
    return probabilistic_sharpe_ratio(returns,
                                      sr_benchmark=sr_zero,
                                      periods_per_year=periods_per_year)


# ── main summary ─────────────────────────────────────────────────────────────

def summarize_backtest(
    period_returns: np.ndarray,
    equity: np.ndarray,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate_annual: float = 0.04,
    n_trials_for_dsr: int | None = None,
) -> dict[str, Any]:
    """
    Args
    ----
    period_returns: net strategy returns per bar (after costs); may contain NaN.
    equity: cumulative equity path (length = len(returns)+1 typical).
    risk_free_rate_annual: annualised risk-free rate to subtract for
        Sharpe / Sortino. ~4% in 2026; legacy default of 0 overstates
        excess Sharpe by ~0.3 on a typical strategy.
    n_trials_for_dsr: if provided, report Deflated Sharpe Ratio under the
        assumption that this many candidate strategies were tried before
        selecting this one.
    """
    r = np.asarray(period_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {"n_periods": 0, "error": "no_returns"}

    rf_per = float(risk_free_rate_annual) / periods_per_year
    excess = r - rf_per

    total_return = float(np.prod(1.0 + r) - 1.0)
    vol = float(np.std(r, ddof=1)) if n > 1 else 0.0
    mu_excess = float(np.mean(excess))
    mu = float(np.mean(r))
    sharpe = (mu_excess / vol) * math.sqrt(periods_per_year) if vol > 1e-12 else 0.0

    neg = excess[excess < 0]
    downside = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0
    sortino = (mu_excess / downside) * math.sqrt(periods_per_year) if downside > 1e-12 else 0.0

    gains = float(np.sum(r[r > 0]))
    losses = float(np.sum(r[r < 0]))
    profit_factor = gains / abs(losses) if losses < -1e-12 else float("inf")

    wins = int(np.sum(r > 0))
    win_rate = wins / n if n else 0.0

    eq = np.asarray(equity, dtype=float)
    mdd = max_drawdown(eq)
    cagr = _annualized_return(total_return, n, periods_per_year)
    calmar = cagr / abs(mdd) if mdd < -1e-6 and math.isfinite(cagr) else float("nan")

    t_stat = t_stat_of_sharpe(sharpe, n, periods_per_year)
    psr = probabilistic_sharpe_ratio(excess, sr_benchmark=0.0, periods_per_year=periods_per_year)
    dsr = (
        deflated_sharpe_ratio(excess, n_trials=int(n_trials_for_dsr),
                              periods_per_year=periods_per_year)
        if n_trials_for_dsr and int(n_trials_for_dsr) >= 2
        else None
    )

    return {
        "n_periods": n,
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6) if math.isfinite(cagr) else None,
        "vol_annual": round(vol * math.sqrt(periods_per_year), 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(mdd, 6),
        "calmar": round(calmar, 4) if math.isfinite(calmar) else None,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
        "avg_period_return": round(mu, 8),
        "risk_free_rate_annual": round(float(risk_free_rate_annual), 6),
        # Statistical significance — institutional decision baseline.
        "sharpe_t_stat": round(t_stat, 4) if math.isfinite(t_stat) else None,
        "psr_vs_zero": round(psr, 4) if math.isfinite(psr) else None,
        "dsr": round(dsr, 4) if dsr is not None and math.isfinite(dsr) else None,
        "n_trials_for_dsr": int(n_trials_for_dsr) if n_trials_for_dsr else None,
    }
