"""
Threshold sweep on the **production agent** (``portfolio/backtest.py``).

Same IS/OOS workflow as ``run_threshold_sweep.py``, but each trial runs
``research_ls`` rules via the agent backtest — not ``dynamic_portfolio_backtest``.

Example::

    python tools/run_agent_threshold_sweep.py \\
        --from-year 2018 --to-year 2022 \\
        --oos-from-year 2023 --oos-to-year 2024 \\
        --mode univariate \\
        --out-dir reports/sweeps/agent_2025 \\
        --max-tickers 50
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.config_loader import _deep_merge, load_config  # noqa: E402
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


def _summarize_agent(summary: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None, int | None]:
    rm = summary.get("risk_metrics") or {}
    return (
        summary.get("strategy_cagr"),
        rm.get("sharpe"),
        rm.get("psr_vs_zero"),
        summary.get("strategy_max_dd"),
        None,
    )


def _run_one_agent_backtest(
    trial_cfg: dict[str, Any],
    *,
    from_year: int,
    to_year: int,
    max_tickers: int,
    signal_step: int,
) -> dict[str, Any]:
    from portfolio.backtest import run_backtest

    merged = _deep_merge(load_config(), trial_cfg)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8",
    ) as tmp:
        json.dump(merged, tmp)
        frozen = Path(tmp.name)
    try:
        return run_backtest(
            from_year=from_year,
            to_year=to_year,
            max_tickers=max_tickers,
            signal_step=signal_step,
            out_html=None,
            out_json=None,
            out_flow_html=None,
            frozen_config_path=frozen,
            skip_invariants=True,
        )
    finally:
        frozen.unlink(missing_ok=True)


def run_sweep(
    *,
    base: dict[str, Any],
    axes: dict[str, list[Any]],
    mode: str,
    from_year: int,
    to_year: int,
    max_tickers: int,
    signal_step: int,
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
            summary = _run_one_agent_backtest(
                cfg, from_year=from_year, to_year=to_year,
                max_tickers=max_tickers, signal_step=signal_step,
            )
            cagr, sharpe, psr, max_dd, n = _summarize_agent(summary)
            out.append(TrialResult(label=label, cfg=cfg, cagr=cagr, sharpe=sharpe,
                                   psr=psr, max_dd=max_dd, n_trades=n))
        except Exception as e:
            out.append(TrialResult(label=label, cfg=cfg, cagr=None, sharpe=None,
                                   psr=None, max_dd=None, n_trades=None, error=str(e)))
    return out


def pick_robust_cfg(results: list[TrialResult]) -> TrialResult | None:
    sharpes = [r.sharpe for r in results if r.sharpe is not None and r.error is None]
    if not sharpes:
        return None
    threshold = sorted(sharpes)[max(0, int(0.75 * len(sharpes)) - 1)]
    candidates = [r for r in results if r.sharpe is not None and r.sharpe >= threshold
                  and r.max_dd is not None and r.error is None]
    if not candidates:
        candidates = [r for r in results if r.sharpe is not None and r.error is None]
    candidates.sort(key=lambda r: r.max_dd if r.max_dd is not None else -1.0, reverse=True)
    return candidates[0]


def write_reports(
    results: list[TrialResult],
    out_dir: Path,
    *,
    n_trials: int,
    oos_block: dict[str, Any] | None,
    simulator: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "agent_threshold_sweep.json"
    md_path = out_dir / "agent_threshold_sweep.md"
    json_path.write_text(json.dumps({
        "simulator": simulator,
        "results": [r.to_json() for r in results],
        "n_trials": n_trials,
        "oos": oos_block or {},
    }, indent=2))

    lines = [
        "# Agent threshold sweep (portfolio/backtest.py)",
        "",
        f"Simulator: **{simulator}**",
        f"Trials: {n_trials}",
        "",
        "| label | cagr | sharpe | psr | max_dd |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda x: -(x.sharpe or -9.99)):
        if r.error:
            lines.append(f"| {r.label} | ERROR: {r.error} | | | |")
            continue
        lines.append(
            f"| {r.label} | {r.cagr or 0:.3f} | {r.sharpe or 0:.3f} | "
            f"{r.psr or 0:.3f} | {r.max_dd or 0:.3f} |"
        )
    if oos_block:
        lines.extend(["", "## OOS validation", "```json", json.dumps(oos_block, indent=2), "```"])
    md_path.write_text("\n".join(lines))
    print(f"\nWrote {json_path}\nWrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-year", type=int, required=True, help="In-sample start year")
    p.add_argument("--to-year", type=int, required=True, help="In-sample end year")
    p.add_argument("--oos-from-year", type=int, default=None)
    p.add_argument("--oos-to-year", type=int, default=None)
    p.add_argument("--mode", choices=["univariate", "joint"], default="univariate")
    p.add_argument("--max-tickers", type=int, default=50, help="Cap universe (50 is faster)")
    p.add_argument("--signal-step", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=Path("reports/sweeps/agent"))
    args = p.parse_args(argv)

    base = dict(DEFAULT_DECISION_CFG)
    axes = THRESHOLD_RANGES if args.mode == "univariate" else JOINT_SWEEP_RANGES
    n_trials = n_univariate_trials_for(axes) if args.mode == "univariate" else n_trials_for(axes)

    print(f"Agent sweep ({n_trials} trials) IS {args.from_year}..{args.to_year} "
          f"max_tickers={args.max_tickers}")
    is_results = run_sweep(
        base=base, axes=axes, mode=args.mode,
        from_year=args.from_year, to_year=args.to_year,
        max_tickers=args.max_tickers, signal_step=args.signal_step,
    )
    chosen = pick_robust_cfg(is_results)
    oos_block = None
    if chosen and args.oos_from_year and args.oos_to_year:
        print(f"\nChosen: {chosen.label} — OOS {args.oos_from_year}..{args.oos_to_year}")
        try:
            summary = _run_one_agent_backtest(
                chosen.cfg,
                from_year=args.oos_from_year,
                to_year=args.oos_to_year,
                max_tickers=args.max_tickers,
                signal_step=args.signal_step,
            )
            cagr, sharpe, psr, max_dd, n = _summarize_agent(summary)
            oos_block = {
                "label": chosen.label,
                "cfg": chosen.cfg,
                "cagr": cagr,
                "sharpe": sharpe,
                "psr": psr,
                "max_dd": max_dd,
                "n_trials_in_search": n_trials,
            }
        except Exception as e:
            oos_block = {"error": str(e), "label": chosen.label}

    write_reports(
        is_results, args.out_dir, n_trials=n_trials, oos_block=oos_block,
        simulator="portfolio/backtest.py (research_ls)",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
