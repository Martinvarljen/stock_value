"""Lightweight integrity gates that drop tickers with broken inputs.

Kept stdlib-only and free of any numpy/pandas dependency so unit tests can
exercise the gate logic without pulling the full analysis stack into memory.

The OHLCV gate consumes the dict produced by
``stock_analyzer.ohlcv_validate.validate_ohlcv_from_data_dict``, which has
the shape ``{"ok": bool, "errors": [str, ...], "warnings": [...], "n_bars": int}``.
Tickers are dropped when ``ok`` is ``False`` — i.e. the validator flagged
at least one hard data-quality error (timestamps unsorted, OHLC body outside
high/low, all-NaN columns, etc.). Warnings (e.g. zero-range bars, missing
volume) are non-fatal and the ticker survives.
"""

from __future__ import annotations

from typing import Any, Iterable


# Fatal-only errors that should drop a ticker even if the validator's overall
# ``ok`` flag somehow stayed True. We do not include warning-bucketed flags
# such as ``zero_range_bars:*`` or ``volume_missing_or_misaligned_skipped``.
_FATAL_ERROR_PREFIXES: tuple[str, ...] = (
    "empty_frame",
    "missing_columns",
    "all_nan",
    "ohlc_length_mismatch",
    "timestamp_not_sorted",
    "duplicate_timestamp",
    "high_below_body",
    "low_above_body",
    "high_below_low",
    "negative_volume",
    "no_close_1y",
)


def is_ohlcv_ok(analysis: dict[str, Any]) -> tuple[bool, list[str]]:
    """Decide whether ``analysis`` has healthy enough OHLCV to trade on.

    Returns ``(ok, reasons)`` where ``reasons`` is the list of fatal flags
    that caused a rejection (empty when ``ok`` is True). Missing or ``None``
    ``ohlcv_quality`` is treated as healthy — the caller is responsible for
    upstream errors (the analysis dict itself carries an ``ok`` field that
    higher-level filters already inspect).
    """
    quality = analysis.get("ohlcv_quality")
    if not quality:
        return True, []

    errors = quality.get("errors") or []
    fatal = [
        e for e in errors
        if any(str(e).startswith(prefix) for prefix in _FATAL_ERROR_PREFIXES)
    ]
    if quality.get("ok") is False and not fatal:
        # Validator said False but no recognised fatal flag — keep conservative
        # behaviour and surface whatever errors it reported.
        fatal = list(errors)
    return (len(fatal) == 0), fatal


def filter_for_bad_ohlcv(
    analyses: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, list[str]]]]:
    """Split ``analyses`` into (kept, dropped).

    ``dropped`` is a list of ``(ticker, fatal_errors)`` tuples so the caller
    can log exactly why each ticker was rejected. Analyses missing a ticker
    field or already marked ``ok=False`` are passed through unchanged — the
    gate only rules on OHLCV health, not on data-fetch or fundamental errors.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[str, list[str]]] = []
    for a in analyses:
        if not a.get("ok"):
            kept.append(a)
            continue
        ok, fatal = is_ohlcv_ok(a)
        if ok:
            kept.append(a)
        else:
            dropped.append((str(a.get("ticker", "?")).upper(), fatal))
    return kept, dropped
