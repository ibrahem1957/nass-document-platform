"""Strictly ordered write-once audit records ("the newest record wins").

Ordering by timestamp is unsafe: Windows clock ticks repeat within a burst, so an
approval and its revocation can carry the same `created_at`, and a random filename
suffix then decides which one "wins". Each record therefore claims the next
sequence number with exclusive file creation: `<prefix>_<NNNNNN>.json`. The
sequence is the order; two writers can never share one.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _sequence(path: Path, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d{{6}})\.json", path.name)
    return int(match.group(1)) if match else -1  # legacy timestamp-named records sort first


def write_sequenced(directory: Path, prefix: str, payload: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sequence = max((_sequence(path, prefix) for path in directory.glob(f"{prefix}_*.json")), default=0) + 1
    while True:
        path = directory / f"{prefix}_{sequence:06d}.json"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload | {"record_sequence": sequence}, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            return path
        except FileExistsError:
            sequence += 1


def ordered(directory: Path, prefix: str) -> list[tuple[Path, dict | None]]:
    """Records oldest first; an unreadable record is returned with payload None."""
    directory = Path(directory)
    found = []
    for path in directory.glob(f"{prefix}_*.json") if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        found.append((path, payload))
    found.sort(key=lambda item: (_sequence(item[0], prefix), (item[1] or {}).get("created_at") or "", item[0].name))
    return found
