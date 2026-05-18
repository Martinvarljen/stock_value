"""Append-only decision memory log.

Ported from primerjava's ``TradingMemoryLog`` (multi-agent LLM framework)
but specialised for a quant pipeline:

  - Every entry is grounded in *numbers* (ML score, p_up, regime, realised
    return, alpha vs SPY) — not LLM prose.
  - Reflections are produced by :mod:`portfolio.reflection`, which is
    deterministic, so the same outcome always yields the same log text.
    This fixes the "non-reproducible memory loop" risk we flagged in the
    primerjava review.
  - Tag-line schema is grep- and diff-friendly:

      ``[YYYY-MM-DD | TICKER | RATING | pending]``                       (pending)
      ``[YYYY-MM-DD | TICKER | RATING | +X.X% | +Y.Y% | Nd]``           (resolved)

  - Updates are atomic via tempfile + ``os.replace`` so a crashed daily run
    can never half-rewrite the log.

The log is a single markdown file (default ``portfolio/data/decision_memory.md``)
so it is easy to inspect by hand, ``git diff``-able, and trivially backed up.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from portfolio.decision_schema import DecisionReport, parse_rating, render_decision


# HTML comment: never appears inside a rendered decision, safe as separator.
_SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
_DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
_REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)


@dataclass
class MemoryEntry:
    """Parsed view of one memory-log entry."""

    date: str
    ticker: str
    rating: str
    pending: bool
    raw_return: str | None
    alpha_return: str | None
    holding_days: str | None
    decision_md: str
    reflection_md: str


class DecisionMemoryLog:
    """Append-only markdown log of decisions and their realised reflections.

    Two phases per entry:

    * **store** — writes a ``pending`` tag plus the rendered ``DecisionReport``
      block. Idempotent on ``(trade_date, ticker)``.
    * **resolve** — once the realised forward return is known (next day,
      next week, whatever the strategy's horizon is) the pending tag is
      replaced with the numeric outcome and a ``REFLECTION`` block is
      appended. Done in a single atomic write per call (``update_with_outcome``)
      or in one pass for many entries (``batch_update_with_outcomes``).
    """

    def __init__(self, path: str | Path, *, max_entries: int | None = None):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries

    @property
    def path(self) -> Path:
        return self._path

    # ── Write path ────────────────────────────────────────────────────

    def store_decision(self, report: DecisionReport) -> bool:
        """Append a pending entry for ``report``.

        Returns ``True`` if a new entry was written, ``False`` if an entry
        with the same ``(trade_date, ticker)`` already exists.
        """
        if self._path.exists():
            raw = self._path.read_text(encoding="utf-8")
            prefix = f"[{report.trade_date} | {report.ticker.upper()} |"
            for line in raw.splitlines():
                if line.startswith(prefix):
                    return False

        decision_md = render_decision(report)
        rating = parse_rating(decision_md, default=report.rating or "Hold")
        tag = f"[{report.trade_date} | {report.ticker.upper()} | {rating} | pending]"
        entry = f"{tag}\n\nDECISION:\n{decision_md}{_SEPARATOR}"

        with self._path.open("a", encoding="utf-8") as f:
            f.write(entry)
        return True

    # ── Read path ─────────────────────────────────────────────────────

    def load_entries(self) -> list[MemoryEntry]:
        if not self._path.exists():
            return []
        text = self._path.read_text(encoding="utf-8")
        out: list[MemoryEntry] = []
        for raw in text.split(_SEPARATOR):
            raw = raw.strip()
            if not raw:
                continue
            parsed = self._parse_entry(raw)
            if parsed:
                out.append(parsed)
        return out

    def get_pending_entries(self) -> list[MemoryEntry]:
        return [e for e in self.load_entries() if e.pending]

    def get_past_context(
        self,
        ticker: str,
        *,
        n_same: int = 3,
        n_cross: int = 3,
    ) -> str:
        """Return a markdown block summarising recent *resolved* lessons.

        Designed to be embedded into a ``DecisionReport.past_context`` field
        so each new entry self-documents what worked / failed last time.
        """
        entries = [e for e in self.load_entries() if not e.pending]
        if not entries:
            return ""

        ticker_u = ticker.upper()
        same: list[MemoryEntry] = []
        cross: list[MemoryEntry] = []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e.ticker.upper() == ticker_u and len(same) < n_same:
                same.append(e)
            elif e.ticker.upper() != ticker_u and len(cross) < n_cross:
                cross.append(e)

        parts: list[str] = []
        if same:
            parts.append(f"Past resolved decisions for {ticker_u}:")
            parts.extend(self._format_resolved_short(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_resolved_short(e) for e in cross)
        return "\n\n".join(parts)

    # ── Resolve path (atomic) ─────────────────────────────────────────

    def update_with_outcome(
        self,
        *,
        ticker: str,
        trade_date: str,
        raw_return: float,
        alpha_return: float,
        holding_days: int,
        reflection: str,
    ) -> bool:
        """Resolve a single pending entry. Returns ``True`` on success."""
        return self.batch_update_with_outcomes([
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "raw_return": raw_return,
                "alpha_return": alpha_return,
                "holding_days": holding_days,
                "reflection": reflection,
            }
        ]) > 0

    def batch_update_with_outcomes(self, updates: Iterable[dict]) -> int:
        """Resolve many pending entries in a single read + atomic write.

        Returns the number of entries actually updated.
        """
        updates = list(updates)
        if not self._path.exists() or not updates:
            return 0

        text = self._path.read_text(encoding="utf-8")
        blocks = text.split(_SEPARATOR)

        update_map = {
            (u["trade_date"], u["ticker"].upper()): u for u in updates
        }

        applied = 0
        new_blocks: list[str] = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                new_blocks.append(block)
                continue

            lines = stripped.splitlines()
            tag_line = lines[0].strip()

            matched = False
            for key, upd in list(update_map.items()):
                trade_date, ticker = key
                pending_prefix = f"[{trade_date} | {ticker} |"
                if (
                    tag_line.startswith(pending_prefix)
                    and tag_line.endswith("| pending]")
                ):
                    fields = [f.strip() for f in tag_line[1:-1].split("|")]
                    if len(fields) < 3:
                        continue
                    rating = fields[2]
                    raw_pct = f"{float(upd['raw_return']):+.1%}"
                    alpha_pct = f"{float(upd['alpha_return']):+.1%}"
                    new_tag = (
                        f"[{trade_date} | {ticker} | {rating}"
                        f" | {raw_pct} | {alpha_pct} | {int(upd['holding_days'])}d]"
                    )
                    rest = "\n".join(lines[1:])
                    new_blocks.append(
                        f"{new_tag}\n\n{rest.lstrip()}\n\nREFLECTION:\n{upd['reflection'].strip()}"
                    )
                    del update_map[key]
                    applied += 1
                    matched = True
                    break

            if not matched:
                new_blocks.append(block)

        if applied == 0:
            return 0

        new_blocks = self._apply_rotation(new_blocks)
        new_text = _SEPARATOR.join(new_blocks)

        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, self._path)
        return applied

    # ── Helpers ───────────────────────────────────────────────────────

    def _apply_rotation(self, blocks: list[str]) -> list[str]:
        if not self._max_entries or self._max_entries <= 0:
            return blocks

        decisions: list[tuple[str, bool]] = []
        for block in blocks:
            stripped = block.strip()
            if not stripped:
                decisions.append((block, False))
                continue
            tag_line = stripped.splitlines()[0].strip()
            is_resolved = (
                tag_line.startswith("[")
                and tag_line.endswith("]")
                and not tag_line.endswith("| pending]")
            )
            decisions.append((block, is_resolved))

        resolved_count = sum(1 for _, r in decisions if r)
        if resolved_count <= self._max_entries:
            return blocks

        to_drop = resolved_count - self._max_entries
        kept: list[str] = []
        for block, is_resolved in decisions:
            if is_resolved and to_drop > 0:
                to_drop -= 1
                continue
            kept.append(block)
        return kept

    def _parse_entry(self, raw: str) -> MemoryEntry | None:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag = lines[0].strip()
        if not (tag.startswith("[") and tag.endswith("]")):
            return None
        fields = [f.strip() for f in tag[1:-1].split("|")]
        if len(fields) < 4:
            return None
        pending = fields[3] == "pending"
        body = "\n".join(lines[1:]).strip()
        dec = _DECISION_RE.search(body)
        ref = _REFLECTION_RE.search(body)
        return MemoryEntry(
            date=fields[0],
            ticker=fields[1],
            rating=fields[2],
            pending=pending,
            raw_return=None if pending else fields[3],
            alpha_return=fields[4] if (not pending and len(fields) > 4) else None,
            holding_days=fields[5] if (not pending and len(fields) > 5) else None,
            decision_md=dec.group(1).strip() if dec else "",
            reflection_md=ref.group(1).strip() if ref else "",
        )

    @staticmethod
    def _format_resolved_short(e: MemoryEntry) -> str:
        raw = e.raw_return or "n/a"
        alpha = e.alpha_return or "n/a"
        hold = e.holding_days or "n/a"
        tag = f"[{e.date} | {e.ticker} | {e.rating} | raw {raw} | alpha {alpha} | {hold}]"
        if e.reflection_md:
            return f"{tag}\n{e.reflection_md}"
        snippet = e.decision_md[:240]
        suffix = "..." if len(e.decision_md) > 240 else ""
        return f"{tag}\n{snippet}{suffix}"
