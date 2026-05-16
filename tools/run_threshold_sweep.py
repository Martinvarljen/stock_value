"""
run_threshold_sweep.py — In-sample / out-of-sample threshold sweep runner.

Why this script exists
======================
The audit flagged four primary entry thresholds (``min_p_up_long``,
``max_p_up_short``, ``long_quintile_min``, ``short_quintile_max``) plus
sizing/stop knobs as hand-tuned magic numbers. The sensitivity
scaffolding lives in ``portfolio.decision_thresholds``; this CLI
exercises it end-to-end:

  1. Run a univariate (or joint) sweep on an in-sample window. Record
     CAGR / Sharpe / PSR for each cfg.
  2. Pick the parameter neighborhood with broadly stable Sharpe (NOT
     the single highest Sharpe — that's overfit).
  3. Re-run the chosen cfg on a held-out OOS window. Report the
     **deflated** Sharpe accounting for the search size.

Output is a JSON report at ``out_dir/threshold_sweep.json`` plus a
human-readable Markdown summary at ``out_dir/threshold_sweep.md``.

Example
-------
::

    python tools/run_threshold_sweep.py \\
        --is-start 2018-01-01 --is-end 2022-12-31 \\
        --oos-start 2023-01-01 --oos-end 2024-12-31 \\
        --mode univariate \\
        --out-dir reports/sweeps/2025_q1 \\
        --top-n 100

This script imports ``backtesting.dynamic_portfolio_backtest.run_dynamic``
lazily so it remains usable for ``--help`` even in numpy-less sandboxes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.decision_thresholds import (  # noqa: E402
    DEFAULT_DECISION_CFG,
    JOINT_SWEEP_RANGES,
    THRESHOLD_RANGES,
    n_trials_for,
    n_univariate_trials_for,
    sweep_decision_cfgs,
    univariate_sensitivity_sweep,
)


@dataclass
class TrialResult:
    label: str
    cfg: dict[str, Any]
    cagr: float | None
    sharpe: float | None
    psr: float | None
    max_dd: float | None
    n_trades: int | None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cfg": self.cfg,
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "psr": self.psr,
            "max_dd": self.max_dd,
            "n_trades": self.n_trades,
            "error": self.error,
        }


def _run_one_backtest(
    cfg: dict[str, Any], *, start: date, end: date, top_n: int,
) -> dict[str, Any]:
    """Lazy wrapper around ``run_dynamic`` so this script imports clean."""
    from backtesting.dynamic_portfolio_backtest import run_dynamic
    return run_dynamic(
        start=start.isoformat(),
        end=end.isoformat(),
        top_n=top_n,
        **{k: v for k, v in cfg.items() if k in {
            "min_p_up_long", "max_p_up_short",
            "long_quintile_min", "short_quintile_max",
            "stop_loss_pct", "take_profit_pct", "position_frac",
            "max_positions", "max_hold_days",
            "commission_bps", "slippage_bps", "borrow_bps_annual",
            "atr_stop_mult", "atr_tp_mult",
            "vol_target_annual_pct",
            "regime_filter", "enable_short",
            "fill_at",
        }},
    )


def _summarize(result: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None, int | None]:
    """Pull (cagr, sharpe, psr, max_dd, n_trades) from a run_dynamic result.

    Tolerant of missing keys — different versions of the simulator have
    surfaced these in slightly different places. Returns ``None`` for
    anything we can't find.
    """
    summary = result.get("summary") or result.get("performance") or {}
    cagr = summary.get("cagr") or result.get("cagr")
    sharpe = summary.get("sharpe") or result.get("sharpe")
    psr = summary.get("psr") or summary.get("probabilistic_sharpe")
    max_dd = summary.get("max_drawdown") or result.get("max_drawdown")
    n_trades = (summary.get("n_trades") or result.get("n_trades")
                or len(result.get("trades", []) or []))
    return cagr, sharpe, psr, max_dd, n_trades


def run_sweep(
    *,
    base: dict[str, Any],
    axes: dict[str, list[Any]],
    mode: str,
    is_start: date,
    is_end: date,
    top_n: int,
) -> list[TrialResult]:
    if mode == "univariate":
        gen = univariate_sensitivity_sweep(base, axes)
    elif mode == "joint":
        gen = sweep_decision_cfgs(base, axes)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    out: list[TrialResult] = []
    for i, (cfg, label) in enumerate(gen, 1):
        print(f"[{i}] {label}", flush=True)
        try:
            result = _run_one_backtest(cfg, start=is_start, end=is_end, top_n=top_n)
            cagr, sharpe, psr, max_dd, n = _summarize(result)
            out.append(TrialResult(label=label, cfg=cfg, cagr=cagr, sharpe=sharpe,
                                   psr=psr, max_dd=max_dd, n_trades=n))
        except Exception as e:
            out.append(TrialResult(label=label, cfg=cfg, cagr=None, sharpe=None,
                                   psr=None, max_dd=None, n_trades=None, error=str(e)))
    return out


def pick_robust_cfg(results: list[TrialResult]) -> TrialResult | None:
    """Pick the cfg in the **top-quartile Sharpe** with the **smallest
    drawdown** — robust to noise and a defensible "neighborhood" choice.

    Avoids the textbook overfit of "highest Sharpe wins"; we want a
    parameter region that's broadly good rather than a single peak.
    """
    sharpes = [r.sharpe for r in results if r.sharpe is not None and r.error is None]
    if not sharpes:
        return None
    threshold = sorted(sharpes)[max(0, int(0.75 * len(sharpes)) - 1)]  # 75th pctile
    candidates = [r for r in results if r.sharpe is not None and r.sharpe >= threshold
                  and r.max_dd is not None and r.error is None]
    if not candidates:
        candidates = [r for r in results if r.sharpe is not None and r.error is None]
    # max_dd is signed-negative (e.g. -0.18); "smallest drawdown" means
    # closest to zero, i.e. *largest* signed value. Sort descending.
    candidates.sort(key=lambda r: r.max_dd if r.max_dd is not None else -1.0,
                    reverse=True)
    return candidates[0]


def deflated_sharpe(returns: list[float], n_trials: int) -> float | None:
    """Tiny shim around the proper DSR in ``performance_metrics`` so we
    don't import numpy/pandas at module level."""
    try:
        from backtesting.performance_metrics import deflated_sharpe_ratio
    except Exception:
        return None
    return deflated_sharpe_ratio(returns, n_trials=n_trials)


def write_reports(results: list[TrialResult], out_dir: Path,
                  *, n_trials: int, oos_block: dict[str, Any] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "threshold_sweep.json"
    md_path = out_dir / "threshold_sweep.md"
    json_path.write_text(json.dumps({
        "results": [r.to_json() for r in results],
        "n_trials": n_trials,
        "oos": oos_block or {},
    }, indent=2))

    lines = ["# Threshold sweep summary", ""]
    lines.append(f"Trials: {n_trials}")
    lines.append("")
    lines.append("| label | cagr | sharpe | psr | max_dd | n_trades |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: -(r.sharpe or -9.99)):
        if r.error:
            lines.append(f"| {r.label} | ERROR: {r.error} | | | | |")
            continue
        lines.append(
            f"| {r.label} | {r.cagr:.3f} | {r.sharpe:.3f} | "
            f"{r.psr or 0:.3f} | {r.max_dd or 0:.3f} | {r.n_trades or 0} |"
        )
    if oos_block:
        lines.append("")
        lines.append("## OOS validation")
        lines.append("```json")
        lines.append(json.dumps(oos_block, indent=2))
        lines.append("```")
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {json_path}\nWrote {md_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--is-start", required=True, type=date.fromisoformat)
    p.add_argument("--is-end", required=True, type=date.fromisoformat)
    p.add_argument("--oos-start", type=date.fromisoformat, default=None)
    p.add_argument("--oos-end", type=date.fromisoformat, default=None)
    p.add_argument("--mode", choices=["univariate", "joint"], default="univariate")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--out-dir", type=Path, default=Path("reports/sweeps"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = dict(DEFAULT_DECISION_CFG)
    if args.mode == "univariate":
        axes = THRESHOLD_RANGES
        n_trials = n_univariate_trials_for(axes)
    else:
        axes = JOINT_SWEEP_RANGES
        n_trials = n_trials_for(axes)
    print(f"Running {args.mode} sweep ({n_trials} trials) on IS "
          f"{args.is_start}..{args.is_end} top_n={args.top_n}")
    is_results = run_sweep(base=base, axes=axes, mode=args.mode,
                           is_start=args.is_start, is_end=args.is_end,
                           top_n=args.top_n)
    chosen = pick_robust_cfg(is_results)
    oos_block: dict[str, Any] | None = None
    if chosen and args.oos_start and args.oos_end:
        print(f"\nChosen IS robust cfg: {chosen.label}")
        print(f"Re-running OOS {args.oos_start}..{args.oos_end} ...")
        try:
            oos_result = _run_one_backtest(chosen.cfg, start=args.oos_start,
                                            end=args.oos_end, top_n=args.top_n)
            cagr, sharpe, psr, max_dd, n = _summarize(oos_result)
            returns = oos_result.get("daily_returns") or []
            dsr = deflated_sharpe(list(returns), n_trials)
            oos_block = {
                "label": chosen.label,
                "cfg": chosen.cfg,
                "cagr": cagr, "sharpe": sharpe, "psr": psr,
                "max_dd": max_dd, "n_trades": n,
                "deflated_sharpe": dsr,
                "n_trials_in_search": n_trials,
            }
        except Exception as e:
            oos_block = {"error": str(e), "label": chosen.label}
    write_reports(is_results, args.out_dir, n_trials=n_trials, oos_block=oos_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
