"""
run_cost_model_diff.py — Quantify the impact of the cost model on a backtest.

Runs the dynamic portfolio backtest twice over the same window —
once with costs zeroed out (the legacy / "free trade" world) and once
with the realistic defaults — then prints a side-by-side delta of
CAGR, Sharpe, max drawdown, hit rate, and n_trades.

The audit explicitly requested empirical verification of the cost
fixes; this CLI is the answer. Use it to:

  * confirm the cost model dents reported P&L by a sensible amount
    (~150-300bps annualised on a 25-day-hold quarterly book; if it's
    less than that something is mis-wired);
  * keep an artifact of the delta in the reports/ folder so future
    regressions in the cost path show up loud.

Example
-------
::

    python tools/run_cost_model_diff.py \\
        --start 2018-01-01 --end 2023-12-31 --top-n 100 \\
        --out reports/cost_model_diff_2018_2023.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_NO_COST = {"commission_bps": 0.0, "slippage_bps": 0.0, "borrow_bps_annual": 0.0}
_REALISTIC = {"commission_bps": 1.0, "slippage_bps": 2.0, "borrow_bps_annual": 50.0}


def _summary(result: dict) -> dict:
    s = result.get("summary") or result.get("performance") or {}
    return {
        "cagr": s.get("cagr") or result.get("cagr"),
        "sharpe": s.get("sharpe") or result.get("sharpe"),
        "max_drawdown": s.get("max_drawdown") or result.get("max_drawdown"),
        "n_trades": (s.get("n_trades") or result.get("n_trades")
                     or len(result.get("trades", []) or [])),
        "hit_rate": s.get("hit_rate") or result.get("hit_rate"),
        "psr": s.get("psr") or s.get("probabilistic_sharpe"),
    }


def _delta(no_cost: dict, with_cost: dict) -> dict:
    out = {}
    for k in ("cagr", "sharpe", "max_drawdown", "n_trades", "hit_rate", "psr"):
        a = no_cost.get(k)
        b = with_cost.get(k)
        if a is None or b is None:
            out[k] = None
            continue
        try:
            out[k] = b - a
        except TypeError:
            out[k] = None
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--out", type=Path, default=Path("reports/cost_model_diff.json"))
    args = p.parse_args(argv)

    from backtesting.dynamic_portfolio_backtest import run_dynamic

    print(f"Running NO-COST backtest {args.start}..{args.end} top_n={args.top_n}")
    no_cost = run_dynamic(
        start=args.start.isoformat(), end=args.end.isoformat(),
        top_n=args.top_n, **_NO_COST,
    )
    print(f"Running REALISTIC-COST backtest {args.start}..{args.end} top_n={args.top_n}")
    real = run_dynamic(
        start=args.start.isoformat(), end=args.end.isoformat(),
        top_n=args.top_n, **_REALISTIC,
    )

    payload = {
        "window": {"start": args.start.isoformat(), "end": args.end.isoformat(),
                   "top_n": args.top_n},
        "no_cost": _summary(no_cost),
        "with_cost": _summary(real),
        "delta": _delta(_summary(no_cost), _summary(real)),
        "cost_assumptions": _REALISTIC,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))

    print("\n--- Cost model impact ---")
    print(f"  CAGR  : {payload['no_cost']['cagr']:.3f} -> "
          f"{payload['with_cost']['cagr']:.3f}  "
          f"(Δ {payload['delta']['cagr']:+.3f})")
    print(f"  Sharpe: {payload['no_cost']['sharpe']:.3f} -> "
          f"{payload['with_cost']['sharpe']:.3f}  "
          f"(Δ {payload['delta']['sharpe']:+.3f})")
    if payload['no_cost']['max_drawdown'] is not None and payload['with_cost']['max_drawdown'] is not None:
        print(f"  MaxDD : {payload['no_cost']['max_drawdown']:.3f} -> "
              f"{payload['with_cost']['max_drawdown']:.3f}  "
              f"(Δ {payload['delta']['max_drawdown']:+.3f})")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
