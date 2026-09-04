"""
Trinetra AI - Phase 4: Analyst triage audit log.

Append-only JSONL on local disk. No database: the log must survive a container
restart, be readable with a text editor in an air-gapped SOC, and never block a
search request. One lock serializes writes across FastAPI's threadpool workers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()

VALID_VERDICTS = {"confirmed", "false_alarm"}


class AuditError(RuntimeError):
    """Audit log could not be written or read."""


def _path() -> Path:
    return settings.audit_log_path


def record(
    tile_id: str,
    verdict: str,
    *,
    query: str | None = None,
    analyst_note: str | None = None,
) -> dict:
    """Append one triage decision. Returns the written entry."""
    if verdict not in VALID_VERDICTS:
        raise AuditError(f"invalid verdict {verdict!r}; expected one of {VALID_VERDICTS}")

    entry = {
        "tile_id": tile_id,
        "verdict": verdict,
        "query": query,
        "analyst_note": analyst_note,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    line = json.dumps(entry, ensure_ascii=False)

    with _lock:
        try:
            path = _path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            raise AuditError(f"cannot write audit log at {_path()}: {exc}") from exc
    return entry


def read_all() -> list[dict]:
    """
    Every entry, oldest first. A corrupt line is skipped rather than fatal - a
    truncated write during a crash must not make the whole log unreadable.
    """
    path = _path()
    if not path.is_file():
        return []
    entries: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entries.append(json.loads(raw))
                except json.JSONDecodeError:
                    logger.warning("Skipping corrupt audit line %d", lineno)
    except OSError as exc:
        raise AuditError(f"cannot read audit log at {path}: {exc}") from exc
    return entries


def latest_verdicts() -> dict[str, dict]:
    """tile_id -> most recent entry. Later decisions supersede earlier ones."""
    out: dict[str, dict] = {}
    for entry in read_all():
        tid = entry.get("tile_id")
        if tid:
            out[tid] = entry
    return out


def count() -> int:
    return len(read_all())


def stats() -> dict:
    verdicts = latest_verdicts()
    confirmed = sum(1 for e in verdicts.values() if e.get("verdict") == "confirmed")
    return {
        "total_entries": count(),
        "tiles_triaged": len(verdicts),
        "confirmed": confirmed,
        "false_alarm": len(verdicts) - confirmed,
        "log_path": str(_path()),
    }
