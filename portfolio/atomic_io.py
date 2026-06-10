"""Atomic file writes for portfolio persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` via temp file + replace (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    default: Any = None,
) -> None:
    """JSON-serialize ``payload`` and atomically write to ``path``."""
    text = json.dumps(payload, indent=indent, default=default)
    atomic_write_text(path, text)
