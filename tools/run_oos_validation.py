#!/usr/bin/env python3
"""
OOS validation workflow for research_ls (portfolio/backtest.py).

Steps:
  1. Write ``portfolio/config.frozen.json`` (current config snapshot, or post-sweep cfg).
  2. Optional in-sample run (train window).
  3. Out-of-sample run with frozen config only.
  4. Markdown summary with yearly returns + forward paper OOS reminder.

Examples::

    # Lock today's config.json, OOS test 2023-2026 (recommended after full backtest tune):
    python tools/run_oos_validation.py --train-to 2022 --oos-from 2023 --oos-to 2026

    # Skip IS re-run (only OOS):
    python tools/run_oos_validation.py --skip-is --oos-from 2023 --oos-to 2026

Full threshold sweep (slow, many hours) — then merge chosen cfg manually or use sweep --write-frozen::
    python tools/run_agent_threshold_sweep.py --from-year 2019 --to-year 2022 \\
        --oos-from-year 2023 --oos-to-year 2024 --max-tickers 50 --write-frozen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.oos_validation import format_yearly_table, write_frozen_config  # noqa: E402


def _run_backtest(
    *,
    from_year: int,
    to_year: int,
    max_tickers: int,
    signal_step: int,
    frozen: Path | None,
    universe_source: str,
    out_html: Path | None,
    out_json: Path | None,
) -> dict:
    from portfolio.backtest import run_backtest

    return run_backtest(
        from_year=from_year,
        to_year=to_year,
        max_tickers=max_tickers,
        signal_step=signal_step,
        out_html=out_html,
        out_json=out_json,
        frozen_config_path=frozen,
        universe_source=universe_source,
        skip_invariants=False,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-from", type=int, default=2019)
    ap.add_argument("--train-to", type=int, default=2022)
    ap.add_argument("--oos-from", type=int, default=2023)
    ap.add_argument("--oos-to", type=int, default=2026)
    ap.add_argument("--max-tickers", type=int, default=100)
    ap.add_argument("--signal-step", type=int, default=5)
    ap.add_argument("--universe-source", default="pit_filter")
    ap.add_argument("--frozen-path", type=Path, default=_ROOT / "portfolio" / "config.frozen.json")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "reports" / "oos_validation")
    ap.add_argument("--skip-is", action="store_true", help="Skip in-sample backtest")
    ap.add_argument("--skip-freeze", action="store_true", help="Use existing frozen config")
    ap.add_argument(
        "--from-sweep-json",
        type=Path,
        default=None,
        help="Use cfg from agent_threshold_sweep.json oos block or top trial",
    )
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    extra: dict | None = None
    if args.from_sweep_json and args.from_sweep_json.is_file():
        data = json.loads(args.from_sweep_json.read_text(encoding="utf-8"))
        oos = data.get("oos") or {}
        extra = dict(oos.get("cfg") or {})
        extra["sweep_label"] = oos.get("label")
        extra["sweep_artifact"] = str(args.from_sweep_json)

    if not args.skip_freeze:
        write_frozen_config(
            args.frozen_path,
            train_through_year=args.train_to,
            oos_from_year=args.oos_from,
            extra=extra,
        )
        print(f"Wrote frozen config → {args.frozen_path}")
    elif not args.frozen_path.is_file():
        raise SystemExit(f"Missing frozen config: {args.frozen_path}")

    is_summary = None
    if not args.skip_is:
        print(f"\n=== In-sample ({args.train_from}–{args.train_to}) ===")
        is_summary = _run_backtest(
            from_year=args.train_from,
            to_year=args.train_to,
            max_tickers=args.max_tickers,
            signal_step=args.signal_step,
            frozen=None,
            universe_source=args.universe_source,
            out_html=out_dir / f"is_{args.train_from}_{args.train_to}.html",
            out_json=out_dir / f"is_{args.train_from}_{args.train_to}.json",
        )

    print(f"\n=== OOS ({args.oos_from}–{args.oos_to}) — frozen config only ===")
    oos_summary = _run_backtest(
        from_year=args.oos_from,
        to_year=args.oos_to,
        max_tickers=args.max_tickers,
        signal_step=args.signal_step,
        frozen=args.frozen_path,
        universe_source=args.universe_source,
        out_html=out_dir / f"oos_{args.oos_from}_{args.oos_to}.html",
        out_json=out_dir / f"oos_{args.oos_from}_{args.oos_to}.json",
    )

    md_lines = [
        "# OOS validation — research_ls",
        "",
        f"- Frozen config: `{args.frozen_path}`",
        f"- Train window: {args.train_from}–{args.train_to} (IS run "
        f"{'skipped' if args.skip_is else 'included'})",
        f"- OOS window: **{args.oos_from}–{args.oos_to}** (parameters locked)",
        "",
    ]

    if is_summary:
        md_lines.extend([
            "## In-sample (not used for final verdict if you only trust OOS)",
            "",
            f"- Strategy CAGR: {is_summary.get('strategy_cagr', 0):.1%}",
            f"- SPY CAGR: {is_summary.get('spy_cagr', 0):.1%}",
            f"- Beat SPY: {is_summary.get('beat_spy')}",
            "",
            format_yearly_table(is_summary.get("yearly") or []),
            "",
        ])

    md_lines.extend([
        "## Out-of-sample (primary)",
        "",
        f"- Strategy CAGR: {oos_summary.get('strategy_cagr', 0):.1%}",
        f"- SPY CAGR: {oos_summary.get('spy_cagr', 0):.1%}",
        f"- Strategy max DD: {oos_summary.get('strategy_max_dd', 0):.1%}",
        f"- Beat SPY: **{oos_summary.get('beat_spy')}**",
        "",
        format_yearly_table(oos_summary.get("yearly") or []),
        "",
        "## Forward test (6–12 months)",
        "",
        "Run daily paper and refresh OOS report:",
        "",
        "```powershell",
        "python portfolio/daily_run.py",
        "python -c \"from portfolio.config_loader import load_config; "
        "from portfolio.paper_oos import write_oos_report; "
        "print(write_oos_report(load_config()))\"",
        "```",
        "",
        "Set `paper_oos.oos_start_date` in config when you start the forward track.",
        "",
        f"HTML: `{out_dir / f'oos_{args.oos_from}_{args.oos_to}.html'}`",
    ])

    md_path = out_dir / "oos_validation.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nSummary → {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
