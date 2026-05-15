"""
Build local feather cache from Dolt for Finance ML training.

Prerequisites:
  1. Clone Dolt DBs (see text/DOLT_SETUP.md or StockMarketTool README)
  2. Run `dolt sql-server` in the folder that contains stocks/ and earnings/

Usage:
    cd Finance
    python projection/data/setup_dolt_cache.py
    python projection/data/setup_dolt_cache.py --output projection/data/cache/all_ohlcv_no_etfs.feather
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from projection.data.dolt_source import (  # noqa: E402
    default_feather_path,
    dolt_available,
    export_ohlcv_from_dolt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Dolt OHLCV to feather cache")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output path (default: {default_feather_path()})",
    )
    parser.add_argument("--include-etfs", action="store_true", help="Do not strip ETFs")
    args = parser.parse_args()

    if not dolt_available():
        print(
            "ERROR: Cannot connect to Dolt MySQL at 127.0.0.1:3306.\n"
            "Start `dolt sql-server` in your post-no-preference data directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    dest = args.output or default_feather_path()
    export_ohlcv_from_dolt(dest, exclude_etfs=not args.include_etfs)
    print("\nNext: train with Dolt data:")
    print("  python projection/ml_model/trainer.py --data-source dolt-feather --purged-cv --optuna")


if __name__ == "__main__":
    main()
