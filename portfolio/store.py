"""Persistent portfolio state, trade ledger, and daily notes (each run reads/writes files only)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_DIR = Path(__file__).resolve().parent
DATA_DIR = PORTFOLIO_DIR / "data"
CONFIG_PATH = PORTFOLIO_DIR / "config.json"
STATE_PATH = DATA_DIR / "state.json"
LEDGER_PATH = DATA_DIR / "trade_ledger.jsonl"
NOTES_DIR = DATA_DIR / "daily_notes"
SNAPSHOTS_DIR = DATA_DIR / "daily_snapshots"
WEEKLY_DIR = DATA_DIR / "weekly"


def ensure_data_dirs() -> None:
    for d in (DATA_DIR, NOTES_DIR, SNAPSHOTS_DIR, WEEKLY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    if CONFIG_PATH.is_file():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def _today() -> date:
    return datetime.today().date()


@dataclass
class Position:
    ticker: str
    side: str  # "long" | "short"
    entry_date: str
    entry_price: float
    notional: float
    stop_price: float
    take_profit_price: float
    max_hold_days: int
    p_up_20d_at_entry: float | None = None
    entry_reason: str = ""

    def days_held(self, as_of: date | None = None) -> int:
        d0 = date.fromisoformat(self.entry_date)
        d1 = as_of or _today()
        return max(0, (d1 - d0).days)

    def estimated_days_remaining(self, default_horizon: int, as_of: date | None = None) -> int:
        held = self.days_held(as_of)
        by_horizon = max(0, default_horizon - held)
        by_max = max(0, self.max_hold_days - held)
        return min(by_horizon, by_max)


@dataclass
class PortfolioState:
    nav: float = 1.0
    cash: float = 1.0
    last_run_date: str | None = None
    positions: list[Position] = field(default_factory=list)

    def position_for(self, ticker: str) -> Position | None:
        t = ticker.upper()
        for p in self.positions:
            if p.ticker.upper() == t:
                return p
        return None

    def open_tickers(self) -> set[str]:
        return {p.ticker.upper() for p in self.positions}


def default_state(cfg: dict[str, Any]) -> PortfolioState:
    nav = float(cfg.get("paper_nav", 1.0))
    return PortfolioState(nav=nav, cash=nav, last_run_date=None, positions=[])


def load_state(cfg: dict[str, Any] | None = None) -> PortfolioState:
    ensure_data_dirs()
    if not STATE_PATH.is_file():
        return default_state(cfg or load_config())
    raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    positions = [Position(**p) for p in raw.get("positions", [])]
    return PortfolioState(
        nav=float(raw.get("nav", 1.0)),
        cash=float(raw.get("cash", 1.0)),
        last_run_date=raw.get("last_run_date"),
        positions=positions,
    )


def save_state(state: PortfolioState) -> None:
    ensure_data_dirs()
    payload = {
        "nav": state.nav,
        "cash": state.cash,
        "last_run_date": state.last_run_date,
        "positions": [asdict(p) for p in state.positions],
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_ledger(entry: dict[str, Any]) -> None:
    ensure_data_dirs()
    entry = {**entry, "logged_at": datetime.now().isoformat(timespec="seconds")}
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_ledger(since: date | None = None) -> list[dict[str, Any]]:
    if not LEDGER_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            d = row.get("date")
            if d and date.fromisoformat(str(d)[:10]) < since:
                continue
        rows.append(row)
    return rows


def notes_path(for_date: date) -> Path:
    return NOTES_DIR / f"{for_date.isoformat()}.json"


def write_daily_notes(for_date: date, payload: dict[str, Any]) -> Path:
    ensure_data_dirs()
    path = notes_path(for_date)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_daily_notes(for_date: date) -> dict[str, Any] | None:
    path = notes_path(for_date)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_snapshot(run_date: date, payload: dict[str, Any]) -> Path:
    ensure_data_dirs()
    path = SNAPSHOTS_DIR / f"{run_date.isoformat()}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def week_id(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def weekly_report_path(d: date | None = None) -> Path:
    return WEEKLY_DIR / f"{week_id(d or _today())}.md"
