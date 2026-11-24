"""Atomic, write-once, provider-first page artifacts for E2-D.

Each snapshot is self-contained and becomes visible only after every file, its
manifest and the completion marker have been fsynced, then the staging directory is
atomically renamed.  A crash can leave a ``.incomplete-*`` directory, but discovery
and reuse deliberately ignore it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.providers import ProviderSpec
from lab.core.models import RunResult

SCHEMA_VERSION = 1
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SNAPSHOT = re.compile(r"^snapshot-(\d{6})$")


class PageArtifactError(RuntimeError):
    pass


def _safe(value: str) -> str:
    cleaned = _SAFE.sub("-", str(value)).strip(".-")
    if not cleaned:
        raise PageArtifactError(f"Unusable artifact path component: {value!r}")
    return cleaned[:100]


def provider_folder(provider_id: str) -> str:
    value = provider_id.lower()
    if value.startswith("oxen-"):
        return "oxen"
    if value.startswith("baidu-"):
        return "baidu"
    if value.startswith("featherless-"):
        return "featherless"
    return _safe(value)


def page_directory(
    root: Path, provider_id: str, document_id: str, document_sha256: str, page: int
) -> Path:
    document = f"{_safe(document_id)}--{document_sha256[:12].lower()}"
    return Path(root) / "pages" / provider_folder(provider_id) / document / f"page-{page:04d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, payload: Any) -> None:
    _write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _status_label(state: str) -> str:
    return {
        job_store.COMPLETED: "completed",
        job_store.FAILED: "failed",
        job_store.ABANDONED: "failed",
        job_store.AWAITING_RESULT_HOST_APPROVAL: "paused",
        job_store.RESUMABLE_ERROR: "paused",
        job_store.RESUMING: "running",
        job_store.SUBMITTED: "running",
        job_store.SUBMISSION_UNCONFIRMED: "ambiguous",
        job_store.SUBMITTING: "running",
    }.get(state, state.lower())


def _response_from_run(run: RunResult | None, page: int) -> dict:
    if run is None:
        return {}
    return (((run.environment.get("e2") or {}).get("page_raw") or {}).get(str(page)) or {}).get(
        "response"
    ) or {}


def _next_snapshot(run_dir: Path) -> Path:
    numbers = [
        int(match.group(1))
        for path in run_dir.iterdir() if run_dir.is_dir()
        if (match := _SNAPSHOT.match(path.name))
    ] if run_dir.is_dir() else []
    return run_dir / f"snapshot-{max(numbers, default=0) + 1:06d}"


def _finalize_directory(staging: Path, final: Path) -> None:
    """Atomically publish a snapshot, tolerating short-lived Windows reader locks."""
    for attempt in range(25):
        try:
            staging.rename(final)
            return
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(0.01)


def save_page_snapshot(
    root: Path,
    spec: ProviderSpec,
    journal: job_store.JobJournal,
    *,
    state: str,
    outcome: dict,
    run: RunResult | None,
    artifact_context: dict[str, Any] | None = None,
) -> Path:
    """Persist one finished page-step atomically and return its final directory."""
    first = journal.first
    page = int(first["page"])
    document_id = str(first["document_id"])
    document_sha256 = str(first["document_sha256"])
    run_id = str(first.get("run_id") or journal.journal_id)
    transmission = first.get("transmission") or {}
    image = transmission.get("file") or {}
    run_dir = page_directory(root, spec.id, document_id, document_sha256, page) / _safe(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep staging names short for deep Windows workspaces (MAX_PATH policies are
    # still common even on systems with long-path-aware Python).
    staging = run_dir / f".incomplete-{uuid.uuid4().hex[:8]}"
    staging.mkdir(exist_ok=False)
    final = _next_snapshot(run_dir)
    response = _response_from_run(run, page)
    result = outcome.get("result") or {}
    raw_text = run.pages[0].text if run is not None and run.pages else str(result.get("text") or "")
    reusable = bool(state == job_store.COMPLETED and run is not None and run.status == "ok")
    prompt = (transmission.get("prompt") or {})
    rendered = transmission.get("rendering") or {}
    rendering_identity = {
        "dpi": rendered.get("dpi"),
        "renderer": rendered.get("renderer"),
        "width": image.get("width"),
        "height": image.get("height"),
        "sha256": rendered.get("sha256"),
    }
    reuse_identity = {
        "document_sha256": document_sha256,
        "page": page,
        "provider_id": spec.id,
        "provider_identity": spec.identity(),
        "model_id": spec.api_model_id,
        "prompt_id": spec.prompt_id,
        "prompt_sha256": prompt.get("sha256"),
        "rendering": rendering_identity,
        "image_sha256": image.get("sha256") or first.get("image_sha256"),
        "generation": dict(spec.generation),
        "protocol": spec.protocol,
        "api_implementation_version": (artifact_context or {}).get("api_implementation_version"),
        "input_policy": (artifact_context or {}).get("input_policy"),
        "batch_fingerprint": (artifact_context or {}).get("fingerprint"),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "provider_folder": provider_folder(spec.id),
        "provider_id": spec.id,
        "provider": spec.provider,
        "model_id": spec.api_model_id,
        "document_id": document_id,
        "document_sha256": document_sha256,
        "page": page,
        "page_image_sha256": reuse_identity["image_sha256"],
        "run_id": run_id,
        "batch_id": (artifact_context or {}).get("batch_id"),
        "batch_item_id": (artifact_context or {}).get("item_id"),
        "stage": (artifact_context or {}).get("stage"),
        "state": state,
        "status": _status_label(state),
        "reusable": reusable,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "reuse_identity": reuse_identity,
        "legacy_run_directory": first.get("run_directory"),
        "job_journal": str(journal.directory),
        "review_path_hint": str(Path(root) / "reviews" / spec.adapter_id / "<revision>" / run_id),
    }
    try:
        transmitted = Path(first.get("run_directory") or "") / "transmitted" / str(
            image.get("filename") or f"page_{page:04d}.png"
        )
        if transmitted.is_file():
            _write_bytes(staging / "page.png", transmitted.read_bytes())
        else:
            metadata["image_missing"] = True
            reusable = False
            metadata["reusable"] = False
        _write_json(staging / "metadata.json", metadata)
        _write_bytes(staging / "raw_text.txt", raw_text.encode("utf-8"))
        _write_json(staging / "raw_response.json", response or outcome)
        _write_json(staging / "structured_output.json", {
            "blocks": result.get("blocks"),
            "official_normalized": result.get("official_normalized"),
            "raw_pages": result.get("raw_pages"),
            "data_info": result.get("data_info"),
            "result_jsonl": result.get("jsonl_text"),
        })
        _write_json(staging / "request_response.json", {
            "transmission": transmission,
            "exchanges": result.get("exchanges") or outcome.get("exchanges") or [],
        })
        _write_json(staging / "status.json", {
            "state": state,
            "status": metadata["status"],
            "error": outcome.get("error"),
            "usage": run.usage.model_dump(mode="json") if run is not None and run.usage else None,
            "timestamps": {"started_at": first.get("started_at"), "saved_at": metadata["saved_at"]},
        })
        _write_json(staging / "job_journal.json", {
            "path": str(journal.directory), "events": journal.events(),
        })
        if run is not None:
            _write_json(staging / "run.json", run.model_dump(mode="json"))
        files = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        manifest = {"schema_version": SCHEMA_VERSION, "files_sha256": files}
        _write_json(staging / "manifest.json", manifest)
        manifest_sha256 = _sha256(staging / "manifest.json")
        _write_json(staging / "complete.json", {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "manifest_sha256": manifest_sha256,
            "reusable": metadata["reusable"],
            "state": state,
        })
        while final.exists():
            number = int(final.name.rsplit("-", 1)[1]) + 1
            final = run_dir / f"snapshot-{number:06d}"
        _finalize_directory(staging, final)
        return final
    except Exception:
        # Deliberately keep the staging directory. Its name makes the interruption
        # explicit, and discovery/reuse will never treat it as complete.
        raise


def verify_snapshot(path: Path, expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(path)
    problems: list[str] = []
    if not _SNAPSHOT.match(path.name):
        problems.append("not a finalized snapshot directory")
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"ok": False, "reusable": False, "problems": [f"missing/corrupt metadata: {exc}"],
                "path": str(path), "metadata": None}
    if not complete.get("complete"):
        problems.append("completion marker is false")
    if _sha256(path / "manifest.json") != complete.get("manifest_sha256"):
        problems.append("manifest hash mismatch")
    for relative, expected in (manifest.get("files_sha256") or {}).items():
        file = path / relative
        if not file.is_file():
            problems.append(f"missing {relative}")
        elif _sha256(file) != expected:
            problems.append(f"modified {relative}")
    required = {"page.png", "metadata.json", "raw_text.txt", "raw_response.json", "status.json",
                "job_journal.json", "request_response.json", "structured_output.json"}
    missing = sorted(required - set((manifest.get("files_sha256") or {})))
    if missing:
        problems.append("manifest lacks " + ", ".join(missing))
    if expected_identity is not None and metadata.get("reuse_identity") != expected_identity:
        actual_identity = metadata.get("reuse_identity") or {}
        differing = sorted(
            key for key in set(actual_identity) | set(expected_identity)
            if actual_identity.get(key) != expected_identity.get(key)
        )
        problems.append("reuse identity mismatch: " + ", ".join(differing))
    reusable = bool(not problems and complete.get("reusable") and metadata.get("reusable")
                    and metadata.get("state") == job_store.COMPLETED and (path / "run.json").is_file())
    return {"ok": not problems, "reusable": reusable, "problems": problems,
            "path": str(path), "metadata": metadata}


def snapshots_for_page(
    root: Path, provider_id: str, document_id: str, document_sha256: str, page: int
) -> list[dict[str, Any]]:
    base = page_directory(root, provider_id, document_id, document_sha256, page)
    found = []
    for path in sorted(base.glob("run_*/snapshot-*")) if base.is_dir() else []:
        found.append(verify_snapshot(path))
    return found


def latest_snapshot_for_run(
    root: Path, provider_id: str, run_id: str, *, include_incomplete: bool = True,
) -> dict[str, Any] | None:
    base = Path(root) / "pages" / provider_folder(provider_id)
    if not base.is_dir():
        return None
    run_dirs = list(base.glob(f"*/page-*/{_safe(run_id)}"))
    candidates = [
        path
        for run_dir in run_dirs
        for path in run_dir.iterdir()
        if path.is_dir() and (
            _SNAPSHOT.match(path.name)
            or (include_incomplete and path.name.startswith(".incomplete-"))
        )
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return verify_snapshot(latest)


def find_reusable_snapshot(
    root: Path, provider_id: str, document_id: str, document_sha256: str, page: int,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    for item in reversed(snapshots_for_page(root, provider_id, document_id, document_sha256, page)):
        checked = verify_snapshot(Path(item["path"]), expected_identity)
        if checked["reusable"]:
            return checked
    return None


__all__ = [
    "PageArtifactError", "find_reusable_snapshot", "latest_snapshot_for_run", "page_directory",
    "provider_folder", "save_page_snapshot", "snapshots_for_page", "verify_snapshot",
]
