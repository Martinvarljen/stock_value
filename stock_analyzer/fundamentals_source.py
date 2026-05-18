"""
fundamentals_source.py — Pluggable point-in-time fundamentals layer.

Why this module exists
======================
``data_layer.collect_data`` historically pulled fundamentals from
yfinance, which serves the *latest restated* numbers rather than what
was originally reported on each filing date. For backtests of
valuation-driven signals (DCF fair value, ROIC, valuation upside,
red-flag triggers based on capex/sales ratios), reading restated
numbers is reading the future. Empirically that footgun inflates
strategy Sharpe by ~0.1-0.3 on fundamentals-heavy backtests.

This module replaces the ad-hoc yfinance call with a pluggable
``FundamentalsSource`` protocol. Three implementations ship:

* ``YfinanceRestatedSource`` — wraps the existing yfinance call. Marked
  unsafe for backtests; emits a one-time warning. This is the default
  for **live** trading where "what we'd consult today" is the right
  number.

* ``CSVPointInTimeSource`` — reads from a directory of per-ticker JSON
  files indexed by fiscal-period-end date. The format is documented in
  ``CSVPointInTimeSource.expected_schema()``. Use this when you've
  built your own PIT dump (e.g. by archiving 10-K/10-Q filings as they
  release) or when reconstituting Compustat / Sharadar exports.

* ``SimFinSource`` — adapter for SimFin's PIT-aware "as_reported" API.
  Requires a ``SIMFIN_API_KEY`` environment variable and the ``simfin``
  package (``pip install simfin``). The class is intentionally a thin
  stub — it raises ``FundamentalsSourceNotConfigured`` until you
  install credentials. This avoids paying $30/mo just to import the
  module.

Wiring
------
``data_layer.collect_data(ticker)`` accepts an optional
``fundamentals_source: FundamentalsSource`` argument and an ``as_of``
date. When both are provided, fundamentals come from the source for the
appropriate fiscal period; price/quote data still comes from yfinance.
The backtest reconstruction path
(``strategy_backtest.reconstruct_data_at``) is the primary caller.

Default behaviour is unchanged: when ``fundamentals_source`` is
``None`` we use the legacy yfinance path with the existing warning.

Adding new sources
------------------
Implement the ``FundamentalsSource`` protocol (one method:
``get_as_of(ticker, as_of)``). Register in ``get_fundamentals_source``
factory. Tests live in ``tests/test_fundamentals_source.py``.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol


# ── exceptions ────────────────────────────────────────────────────────────────

class FundamentalsSourceError(RuntimeError):
    """Base class for fundamentals-source errors that callers may want
    to handle gracefully (typically by logging + skipping the ticker)."""


class FundamentalsSourceNotConfigured(FundamentalsSourceError):
    """Source dependencies (API keys, packages, data files) are missing."""


# ── protocol ──────────────────────────────────────────────────────────────────

class FundamentalsSource(Protocol):
    """Returns the fundamental fields observable on or before ``as_of``.

    Output is a flat dict using the same field names ``data_layer``
    produces (revenue, operating_margin, fcf_5y, etc.). Missing fields
    should be ``None``, not raised — the scoring engines all handle
    ``None`` gracefully.

    Implementations MUST NOT return values that were known only after
    ``as_of``. Restated numbers, retroactive segment changes, and
    audited-correction trailing data are leakage and the ``Protocol``
    contract requires they be filtered out.
    """

    def name(self) -> str: ...

    def get_as_of(self, ticker: str, as_of: date) -> dict[str, Any]: ...

    def is_pit(self) -> bool:
        """True if outputs are guaranteed point-in-time. False for
        restated sources like yfinance."""


# ── yfinance restated (legacy / live default) ────────────────────────────────

_YFINANCE_PIT_WARNING = (
    "YfinanceRestatedSource: yfinance fundamentals are restated, not "
    "point-in-time. Backtests of valuation-driven signals overstate "
    "Sharpe by an estimated 0.1-0.3. Pass ``fundamentals_source=`` a "
    "PIT source for backtests; this default is intended for live trading."
)
_YFINANCE_WARNED = False


def _warn_yfinance_once() -> None:
    global _YFINANCE_WARNED
    if _YFINANCE_WARNED:
        return
    warnings.warn(_YFINANCE_PIT_WARNING, stacklevel=3)
    _YFINANCE_WARNED = True


@dataclass
class YfinanceRestatedSource:
    """Wraps the existing ``data_layer`` yfinance fetch.

    ``as_of`` is **ignored** — yfinance only returns latest-restated.
    Marked ``is_pit() == False`` so callers know what they're getting.
    """

    def name(self) -> str:
        return "yfinance_restated"

    def is_pit(self) -> bool:
        return False

    def get_as_of(self, ticker: str, as_of: date | None = None) -> dict[str, Any]:
        _warn_yfinance_once()
        # Lazy import — keeps this module free of pandas/numpy/yfinance at
        # import time. The legacy fetch lives in ``data_layer`` which is
        # heavyweight; we shell out only when the caller actually pulls.
        from data_layer import collect_data
        return collect_data(ticker)


# ── CSV / JSON point-in-time source ──────────────────────────────────────────

@dataclass
class CSVPointInTimeSource:
    """Reads per-ticker per-fiscal-period JSON files from disk.

    Directory layout
    ----------------
    ::

        <root>/
            AAPL/
                2018-09-29.json   # fiscal period end date
                2018-12-29.json
                ...
            MSFT/
                2018-06-30.json
                ...

    Each JSON file is a flat dict with the field names ``data_layer``
    produces. ``expected_schema()`` documents the required and optional
    keys. Files dated *after* ``as_of`` are ignored — we return the
    most recent file with ``date <= as_of - reporting_lag_days``.

    ``reporting_lag_days`` defaults to 90 — a conservative bound that
    matches the SEC's 90-day 10-K filing deadline. Set lower when
    you've validated actual filing dates from EDGAR.
    """

    root: Path
    reporting_lag_days: int = 90
    _index_cache: dict[str, list[tuple[date, Path]]] = field(
        default_factory=dict, init=False, repr=False,
    )

    def name(self) -> str:
        return f"csv_pit:{self.root}"

    def is_pit(self) -> bool:
        return True

    def _index(self, ticker: str) -> list[tuple[date, Path]]:
        if ticker in self._index_cache:
            return self._index_cache[ticker]
        ticker_dir = self.root / ticker.upper()
        out: list[tuple[date, Path]] = []
        if ticker_dir.is_dir():
            for f in sorted(ticker_dir.glob("*.json")):
                try:
                    d = date.fromisoformat(f.stem)
                except ValueError:
                    continue
                out.append((d, f))
            out.sort(key=lambda x: x[0])
        self._index_cache[ticker] = out
        return out

    def get_as_of(self, ticker: str, as_of: date) -> dict[str, Any]:
        from datetime import timedelta
        idx = self._index(ticker)
        if not idx:
            raise FundamentalsSourceNotConfigured(
                f"No CSVPointInTimeSource files for {ticker} under {self.root}. "
                f"Expected layout: {self.root}/{ticker}/<fiscal-period-end>.json"
            )
        cutoff = as_of - timedelta(days=int(self.reporting_lag_days))
        latest: tuple[date, Path] | None = None
        for fp_date, path in idx:
            if fp_date <= cutoff:
                latest = (fp_date, path)
            else:
                break
        if latest is None:
            return {
                "ticker": ticker,
                "error": f"no_pit_filing_before_{as_of.isoformat()}",
                "data_quality_score": 0,
            }
        try:
            payload = json.loads(latest[1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise FundamentalsSourceError(
                f"Failed to read {latest[1]}: {e}"
            ) from e
        payload.setdefault("ticker", ticker)
        payload["fiscal_period_end"] = latest[0].isoformat()
        payload["fundamentals_source"] = self.name()
        return payload

    @staticmethod
    def expected_schema() -> dict[str, str]:
        """Document the fields a PIT JSON should provide. Keys mirror
        those produced by ``data_layer.collect_data``; only ``revenue``,
        ``operating_margin``, ``net_debt``, and ``shares_outstanding``
        are strictly required for a DCF base case. Everything else
        improves accuracy."""
        return {
            "revenue": "trailing 12M revenue, native currency",
            "operating_margin": "EBIT / revenue (decimal)",
            "net_income_5y": "list of 5 prior fiscal years (oldest first)",
            "fcf_5y": "list of 5 prior fiscal years",
            "ebit_5y": "list",
            "revenue_5y": "list",
            "shares_5y": "list",
            "capex_5y": "list",
            "dividends_5y": "list",
            "sbc_5y": "list",
            "net_debt": "scalar",
            "net_debt_ebitda": "scalar",
            "shares_outstanding": "scalar",
            "effective_tax_rate": "decimal",
            "capex_pct_revenue": "decimal",
            "interest_coverage": "scalar",
            "interest_expense": "scalar",
            "total_debt": "scalar",
            "market_cap": "scalar (as of fiscal period end)",
            "beta": "scalar",
            "roic": "decimal",
            "fcf_yield": "decimal",
            "revenue_cagr_5y": "decimal",
            "fcf_cagr_5y": "decimal",
            "sector": "string (GICS sector)",
            "industry": "string",
        }


# ── SimFin stub ──────────────────────────────────────────────────────────────

@dataclass
class SimFinSource:
    """Adapter for SimFin's ``as_reported`` PIT API. Stubbed.

    Activation steps:

    1. ``pip install simfin`` (or ``simfin>=0.8`` for Python 3.10+).
    2. Set ``SIMFIN_API_KEY`` in env (free tier: 100 req/day, paid
       tier: $30/mo with PIT-aware historical data).
    3. Replace ``get_as_of`` body with the call below — schema mapping
       sketch is left in a comment so you don't have to figure it out
       cold.

    The class still imports cleanly without simfin so this module's
    other adapters keep working. ``get_as_of`` raises
    ``FundamentalsSourceNotConfigured`` until you do the wiring.
    """

    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("SIMFIN_API_KEY")

    def name(self) -> str:
        return "simfin_pit"

    def is_pit(self) -> bool:
        return True

    def get_as_of(self, ticker: str, as_of: date) -> dict[str, Any]:
        if not self.api_key:
            raise FundamentalsSourceNotConfigured(
                "SIMFIN_API_KEY missing. Either set the env var or pass "
                "api_key=... to SimFinSource. See module docstring for "
                "activation steps."
            )
        # ── implementation sketch ─────────────────────────────────────
        # import simfin as sf
        # sf.set_api_key(self.api_key)
        # sf.set_data_dir(<cache>)
        # df = sf.load_income(variant="quarterly", market="us")
        # row = (df.xs(ticker, level="Ticker")
        #          .loc[df.index.get_level_values("Report Date") <= as_of]
        #          .iloc[-1])
        # ... map SimFin fields to our schema (Revenue -> revenue,
        #     Operating Income (Loss) / Revenue -> operating_margin, etc.)
        # ... build 5y trailing arrays from prior rows
        # return mapped_dict
        raise FundamentalsSourceNotConfigured(
            "SimFinSource is a stub. Implement get_as_of() before use; "
            "see the implementation sketch in the module source."
        )


# ── factory ──────────────────────────────────────────────────────────────────

def get_fundamentals_source(name: str, **kwargs: Any) -> FundamentalsSource:
    """Resolve a source by short name.

    Known names:
      * ``yfinance_restated`` — legacy live default
      * ``csv_pit`` — requires ``root=<Path>``
      * ``simfin`` — requires API key (in env or via ``api_key=...``)
    """
    if name == "yfinance_restated":
        return YfinanceRestatedSource(**kwargs)
    if name == "csv_pit":
        if "root" not in kwargs:
            raise FundamentalsSourceNotConfigured(
                "csv_pit source requires root=<Path> (directory of "
                "per-ticker JSON files; see CSVPointInTimeSource docstring)"
            )
        kwargs["root"] = Path(kwargs["root"])
        return CSVPointInTimeSource(**kwargs)
    if name == "simfin":
        return SimFinSource(**kwargs)
    raise ValueError(
        f"Unknown fundamentals_source name: {name!r}. "
        f"Known: yfinance_restated, csv_pit, simfin."
    )
