"""
Backward-compatible entry point for the projection engine.

The implementation lives in ``projection/projection_engine.py``. This module
adds ``projection/`` to ``sys.path`` and re-exports the public API so scripts
that run with only ``stock_analyzer`` on the path can still import
``projection_engine`` when the repo layout is intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_proj = str(_root / "projection")
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from projection_engine import (  # noqa: E402  pylint: disable=wrong-import-position
    WEIGHTS,
    generate_projections,
    print_projections,
    _composite_score,
)

__all__ = [
    "WEIGHTS",
    "generate_projections",
    "print_projections",
    "_composite_score",
]
