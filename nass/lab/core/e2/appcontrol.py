"""Read-only Windows application-control diagnostics (no policy is read for writing,
changed, excluded or bypassed).

Smart App Control / App Control for Business block unsigned native modules by cloud
reputation, and a verdict can change over time: the same unchanged file can load one
day and be blocked the next. File hashes therefore cannot prove that a model will load.
This module provides the two checks that can:

- `recent_blocks`: Code Integrity events (3033/3077) for files under an environment,
  read with the built-in `wevtutil` tool;
- `native_import_probe`: import the runtime's native modules in a separate process
  (no model weights) and report exactly which import failed and why.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

CI_LOG = "Microsoft-Windows-CodeIntegrity/Operational"
_LOAD = re.compile(r"process \((?P<process>[^)]+)\) attempted to load (?P<file>\S+) that")
_POLICY = re.compile(r"Policy ID:(\{[0-9A-Fa-f-]+\})")
_EVENT_ID = re.compile(r"^\s*Event ID:\s*(\d+)", re.MULTILINE)


def _env_fragment(environment: str | Path) -> str:
    """`D:\\x\\venv` -> `\\x\\venv`: CI paths use volume devices, not drive letters."""
    text = str(Path(environment)).replace("/", "\\")
    return (text.split(":", 1)[1] if ":" in text[:3] else text).lower().rstrip("\\")


def parse_wevtutil_text(text: str, environment: str | Path | None = None) -> list[dict]:
    """Blocked files from `wevtutil qe ... /f:text` output, newest first, de-duplicated."""
    fragment = _env_fragment(environment) if environment else None
    found: list[dict] = []
    seen: set[str] = set()
    for chunk in re.split(r"(?m)^Event\[\d+\]", text):
        match = _LOAD.search(chunk)
        if not match:
            continue
        device_path = match.group("file")
        if fragment and fragment + "\\" not in device_path.lower():
            continue
        relative = device_path
        if fragment:
            relative = device_path[device_path.lower().index(fragment) + len(fragment) + 1 :]
        if relative.lower() in seen:
            continue
        seen.add(relative.lower())
        event = _EVENT_ID.search(chunk)
        policy = _POLICY.search(chunk)
        found.append(
            {
                "file": relative,
                "device_path": device_path,
                "process": match.group("process"),
                "event_id": int(event.group(1)) if event else None,
                "policy_id": policy.group(1) if policy else None,
            }
        )
    return found


def recent_blocks(
    within_seconds: float, environment: str | Path | None = None
) -> list[dict]:
    """Code Integrity blocks in the last `within_seconds` (Windows only, read-only)."""
    if sys.platform != "win32":
        return []
    window_ms = max(1000, int(within_seconds * 1000))
    query = f"*[System[(EventID=3033 or EventID=3077) and TimeCreated[timediff(@SystemTime) <= {window_ms}]]]"
    try:
        output = subprocess.run(
            ["wevtutil", "qe", CI_LOG, f"/q:{query}", "/f:text", "/rd:true", "/c:200"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_wevtutil_text(output, environment)


_PROBE = textwrap.dedent(
    """
    import importlib, json, sys, traceback
    results = []
    for name in json.loads(sys.argv[1]):
        try:
            importlib.import_module(name)
            results.append({"module": name, "ok": True})
        except BaseException as exc:
            results.append({
                "module": name, "ok": False, "type": type(exc).__name__,
                "message": str(exc)[:2000], "traceback": traceback.format_exc()[-6000:],
            })
            break
    print("PROBE_JSON=" + json.dumps(results))
    """
)


def native_import_probe(
    modules: list[str],
    *,
    python: str | None = None,
    environment: str | Path | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Import `modules` in order in a fresh, isolated interpreter (stops at the first
    failure, so a blocked module is attempted once). Loads no model weights."""
    started = time.monotonic()
    executable = python or sys.executable
    try:
        completed = subprocess.run(
            [executable, "-I", "-c", _PROBE, json.dumps(modules)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
        line = next(
            (item for item in completed.stdout.splitlines() if item.startswith("PROBE_JSON=")),
            None,
        )
        results = json.loads(line.split("=", 1)[1]) if line else []
        process_error = None if line else (completed.stderr or "no probe output")[-4000:]
    except subprocess.TimeoutExpired:
        results, process_error = [], f"probe timed out after {timeout_s:.0f}s"
    except OSError as exc:
        results, process_error = [], f"probe could not start: {exc}"
    failed = next((item for item in results if not item["ok"]), None)
    ok = process_error is None and failed is None and len(results) == len(modules)
    target = environment or Path(executable).resolve().parents[1]
    blocks: list[dict] = []
    # Code Integrity writes its event shortly after the failed load returns, so a
    # failure without an event yet is re-checked for a few seconds.
    for attempt in range(1 if ok else 7):
        blocks = recent_blocks(time.monotonic() - started + 5, target)
        if blocks or ok:
            break
        time.sleep(0.5)
    elapsed = time.monotonic() - started
    return {
        "ok": ok,
        "python": executable,
        "modules": modules,
        "results": results,
        "failed": failed,
        "process_error": process_error,
        "application_control_blocks": blocks,
        "elapsed_s": round(elapsed, 2),
    }
