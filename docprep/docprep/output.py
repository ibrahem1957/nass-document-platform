"""Write-once output folders: <root>/<run_id>__<UTC timestamp>/. Existing outputs are never overwritten."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__

ARTIFACTS = ("section_tree.json", "assignments.json", "structure_report.md", "structure_report.html")


class OutputExistsError(FileExistsError):
    """The target output folder already exists."""


def safe_name(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._") or "run"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(root: Path | str, run_id: str, contents: dict[str, str],
                  manifest_builder: Callable[[dict[str, str]], dict[str, Any]], *, stamp: str | None = None,
                  read_only: bool = True) -> Path:
    """Write every artifact into a temporary folder, then rename it to the final name in one step.
    The manifest is built last and receives the artifact hashes."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{safe_name(run_id)}__{stamp or timestamp()}"
    if target.exists():
        raise OutputExistsError(f"output folder already exists (never overwritten): {target}")
    staging = root / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        hashes = {}
        for name, text in contents.items():
            path = staging / name
            path.write_text(text, encoding="utf-8", newline="\n")
            hashes[name] = _sha256(path)
        manifest = manifest_builder(hashes)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                               encoding="utf-8", newline="\n")
        if target.exists():
            raise OutputExistsError(f"output folder already exists (never overwritten): {target}")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if read_only:
        for path in target.iterdir():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return target


def build_manifest(*, run_id: str, run_dir: str, input_hashes_before: dict[str, str | None],
                   input_hashes_after: dict[str, str | None], config: dict[str, Any], started_at: str,
                   finished_at: str, output_hashes: dict[str, str], findings: list[dict[str, Any]],
                   warnings: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 1, "kind": "docprep_manifest", "component": "Document Structure Builder",
        "docprep_version": __version__, "source_run_id": run_id, "source_run_dir": run_dir,
        "input_hashes": input_hashes_before, "input_hashes_after": input_hashes_after,
        "inputs_unchanged": input_hashes_before == input_hashes_after,
        "configuration": config, "detection_thresholds": config.get("thresholds"),
        "started_at": started_at, "finished_at": finished_at, "written_at": datetime.now(timezone.utc).isoformat(),
        "output_hashes": output_hashes,
        "validation_summary": {s: sum(1 for f in findings if f["severity"] == s) for s in ("ok", "warning", "error")},
        "warnings": warnings,
        "scope": "document structure only: no chunking, embeddings, retrieval or model inference",
    }
