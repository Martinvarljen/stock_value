"""
utils.py  —  Shared formatting and calculation helpers

Imported by all engine modules. Keep this file free of yfinance / pandas
dependencies so it remains fast to import.
"""

import numpy as np
from typing import Optional


# ── Formatters ────────────────────────────────────────────────────────────────

def _pct(v, decimals: int = 1) -> str:
    return f"{v:.{decimals}%}" if v is not None else "N/A"

def _bn(v) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    return f"{v:.0f}"

def _x(v, d: int = 2) -> str:
    return f"{v:.{d}f}x" if v is not None else "N/A"

def _num(v, d: int = 2) -> str:
    return f"{v:.{d}f}" if v is not None else "N/A"


# ── Threshold assessment ──────────────────────────────────────────────────────

def _assess(value, thresholds: list[tuple]) -> str:
    """
    thresholds: list of (min_value, label) sorted descending.
    Returns the label for the first threshold the value meets.
    """
    if value is None:
        return "No data"
    for min_val, label in thresholds:
        if value >= min_val:
            return label
    return thresholds[-1][1]


# ── List helpers ──────────────────────────────────────────────────────────────

def _valid(lst: list) -> list:
    return [v for v in lst if v is not None]

def _count_positive(lst: list) -> tuple[int, int]:
    """Returns (positive_count, total_non_None_count)."""
    valid = _valid(lst)
    return sum(1 for v in valid if v > 0), len(valid)

def _cv(lst: list) -> Optional[float]:
    """Coefficient of variation: std / |mean|. None if < 2 valid values."""
    valid = _valid(lst)
    if len(valid) < 2:
        return None
    m = np.mean(valid)
    if m == 0:
        return None
    return float(np.std(valid) / abs(m))


# ── Financial calculations ────────────────────────────────────────────────────

def capex_pct_for_valuation(data: dict, default: float = 0.03) -> float:
    """Net capex (capex − D&A) preferred for DCF; fallback to gross capex/revenue."""
    net = data.get("net_capex_pct_revenue")
    if net is not None:
        return float(net)
    gross = data.get("capex_pct_revenue")
    if gross is not None:
        return float(gross)
    return default


def _cagr(start: Optional[float], end: Optional[float], years: float) -> Optional[float]:
    """Compound annual growth rate. Returns None if inputs are invalid."""
    try:
        if start is None or end is None or years <= 0 or start <= 0:
            return None
        return (end / start) ** (1 / years) - 1
    except Exception:
        return None

def _cagr_from_list(lst: list) -> tuple[Optional[float], int]:
    """
    Derive CAGR from a time-ordered list (oldest → newest), skipping None gaps.
    Returns (cagr, years_spanned). Uses index distance so sparse lists are correct.
    """
    indexed = [(i, v) for i, v in enumerate(lst) if v is not None]
    if len(indexed) < 2:
        return None, 0
    i_start, v_start = indexed[0]
    i_end,   v_end   = indexed[-1]
    years = i_end - i_start
    return _cagr(v_start, v_end, years), years
