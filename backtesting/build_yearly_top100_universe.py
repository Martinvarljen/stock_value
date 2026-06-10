#!/usr/bin/env python3
"""Build cached yearly top-100 ticker lists (dollar-volume proxy). See yearly_top100_universe.py.

Either pass explicit years::

    python backtesting/build_yearly_top100_universe.py --from 2022 --to 2025

Or build the range needed for backtests that start in calendar year Y (uses prior-year volumes)::

    python backtesting/build_yearly_top100_universe.py --for-checkpoints-from-year 2023
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backtesting.yearly_top100_universe import (
    build_top_n_for_year,
    default_universe_cache_dir,
    normalize_universe_source,
    write_ticker_lines,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build yearly top-100 universe files for strategy_backtest.")
    ap.add_argument("--from", dest="y0", type=int, default=None, help="First calendar year for volume ranking (e.g. 2022)")
    ap.add_argument("--to", dest="y1", type=int, default=None, help="Last calendar year for volume ranking (e.g. 2025)")
    ap.add_argument(
        "--for-checkpoints-from-year",
        type=int,
        default=None,
        metavar="Y",
        help="Shorthand: build years [Y-1 .. prior calendar year] so checkpoints from year Y have lag files (full-year volumes only).",
    )
    ap.add_argument("--top", type=int, default=100, help="How many names to keep per year (default 100)")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default depends on --universe-source)",
    )
    ap.add_argument(
        "--universe-source",
        choices=("legacy", "pit"),
        default="pit",
        help="legacy=rank today's S&P 500; pit=rank PIT S&P pool from sp500_changes.csv (default: pit)",
    )
    args = ap.parse_args()
    uni_src = normalize_universe_source(args.universe_source)
    out_dir = args.out_dir or default_universe_cache_dir(_ROOT, uni_src)

    if args.for_checkpoints_from_year is not None:
        y0 = int(args.for_checkpoints_from_year) - 1
        y1 = datetime.today().year - 1
        if y1 < y0:
            y1 = y0
    elif args.y0 is not None and args.y1 is not None:
        y0, y1 = args.y0, args.y1
    else:
        raise SystemExit("Provide --from Y0 --to Y1, or --for-checkpoints-from-year Y (e.g. 2023).")

    if y0 > y1:
        raise SystemExit("--from must be <= --to")

    print(f"Universe source: {uni_src}  ->  {out_dir}\n")
    for year in range(y0, y1 + 1):
        print(f"\n=== Year {year} ===")
        tickers = build_top_n_for_year(year, universe_source=uni_src, top_n=args.top, verbose=True)
        path = out_dir / f"{year}.txt"
        write_ticker_lines(path, tickers)
        print(f"Wrote {len(tickers)} tickers -> {path}")


if __name__ == "__main__":
    main()
