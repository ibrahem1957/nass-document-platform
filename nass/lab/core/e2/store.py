"""Persistent, write-once evaluation library (local disk only).

Layout (root defaults to $AGENTICDOC_RESULTS_DIR):

    runs/<model_key>/<revision>/<run_id>/
        started.json          configuration, hardware, inputs (written first)
        pages/p0001.json      exact RAW page response as it arrives (crash-safe)
        run.json              the complete RAW RunResult
        derived/<variant>.json  post-processed runs (separate from RAW)
        summary.json          metrics with exact counts (RAW and derived)
        final.json            terminal status + SHA-256 of every file above
    index.jsonl               append-only, one line per finalized run
    reports/comparison_<UTC timestamp>.md

Nothing is ever overwritten or deleted: every file is created with mode "x".
A run directory without final.json is reported as `incomplete`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.core.models import RunResult

DEFAULT_RESULTS_ROOT = "D:/agenticdoc_e2/results"
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
_INDEX_THREAD_LOCK = threading.Lock()


class StoreError(RuntimeError):
    pass


def results_root() -> Path:
    return Path(os.environ.get("AGENTICDOC_RESULTS_DIR", DEFAULT_RESULTS_ROOT))


def _safe(value: str) -> str:
    cleaned = _SAFE.sub("_", value).strip("._")
    if not cleaned:
        raise StoreError(f"Unusable path component: {value!r}")
    if len(cleaned) <= 80:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f"{cleaned[:54]}--{digest}"


def _write_once(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise StoreError(f"Refusing to overwrite existing result file: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _locked_index(root: Path):
    """Cross-thread/process lock for the one shared append-only run index."""
    path = root / ".index.lock"
    with _INDEX_THREAD_LOCK, path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def model_key_for(run_or_adapter_id: str) -> str:
    return _safe(run_or_adapter_id)


class RunHandle:
    def __init__(self, store: "ResultStore", directory: Path, run_id: str):
        self.store = store
        self.directory = directory
        self.run_id = run_id

    def write_page(self, page: int, record: dict) -> None:
        _write_once(self.directory / "pages" / f"p{page:04d}.json", record)

    def finalize(
        self,
        run: RunResult,
        *,
        derived: list[RunResult] | None = None,
        summaries: list[dict] | None = None,
        language: str,
        status: str,
        notes: list[str] | None = None,
    ) -> Path:
        if run.run_id != self.run_id:
            raise StoreError("Run id does not match the reserved directory")
        _write_once(self.directory / "run.json", run.model_dump(mode="json"))
        for item in derived or []:
            _write_once(
                self.directory / "derived" / f"{_safe(item.variant)}.json", item.model_dump(mode="json")
            )
        _write_once(self.directory / "summary.json", {"summaries": summaries or []})
        files = {
            str(path.relative_to(self.directory)).replace("\\", "/"): _sha256(path)
            for path in sorted(self.directory.rglob("*"))
            if path.is_file()
        }
        final = {
            "run_id": run.run_id,
            "status": status,
            "language": language,
            "adapter_id": run.adapter.adapter_id,
            "model": run.adapter.model_id,
            "label": run.adapter.label,
            "pages": [page.page for page in run.pages],
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "notes": notes or [],
            "files_sha256": files,
        }
        _write_once(self.directory / "final.json", final)
        self.store._append_index(
            {key: final[key] for key in ("run_id", "status", "language", "adapter_id", "model", "label", "pages", "finalized_at")}
            | {"path": str(self.directory.relative_to(self.store.root)).replace("\\", "/")}
        )
        return self.directory


class ResultStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or results_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def begin(
        self, *, adapter_id: str, revision: str, run_id: str, started: dict
    ) -> RunHandle:
        revision_key = _safe(revision)
        directory = self.root / "runs" / model_key_for(adapter_id) / revision_key / _safe(run_id)
        if directory.exists():
            raise StoreError(f"Run directory already exists: {directory}")
        directory.mkdir(parents=True)
        _write_once(
            directory / "started.json",
            started
            | {
                "run_id": run_id,
                "storage_identity": {"revision": revision, "revision_key": revision_key},
            },
        )
        return RunHandle(self, directory, run_id)

    def reopen(self, directory: Path) -> RunHandle:
        """Continue a run that is not finalized yet (a resumable hosted API job)."""
        directory = Path(directory)
        if self.root not in directory.resolve().parents:
            raise StoreError(f"Run directory is outside the result store: {directory}")
        started = directory / "started.json"
        if not started.is_file():
            raise StoreError(f"No started.json in {directory}")
        if (directory / "final.json").exists():
            raise StoreError(f"Run is already finalized (write-once): {directory}")
        run_id = json.loads(started.read_text(encoding="utf-8"))["run_id"]
        return RunHandle(self, directory, run_id)

    def _append_index(self, record: dict) -> None:
        with _locked_index(self.root):
            with (self.root / "index.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def runs(self) -> list[dict]:
        """Every run directory, finalized or not (incomplete runs are kept visible)."""
        found: list[dict] = []
        base = self.root / "runs"
        if not base.is_dir():
            return found
        for started in sorted(base.glob("*/*/*/started.json")):
            directory = started.parent
            meta = json.loads(started.read_text(encoding="utf-8"))
            final_path = directory / "final.json"
            if final_path.is_file():
                final = json.loads(final_path.read_text(encoding="utf-8"))
                summary_path = directory / "summary.json"
                summaries = (
                    json.loads(summary_path.read_text(encoding="utf-8")).get("summaries", [])
                    if summary_path.is_file()
                    else []
                )
                found.append({"directory": str(directory), "started": meta, "final": final, "summaries": summaries})
            else:
                pages = sorted(path.name for path in (directory / "pages").glob("p*.json"))
                found.append(
                    {
                        "directory": str(directory),
                        "started": meta,
                        "final": {
                            "run_id": meta.get("run_id"),
                            "status": "incomplete",
                            "language": meta.get("language"),
                            "adapter_id": meta.get("adapter_id"),
                            "label": meta.get("label"),
                            "pages_saved": pages,
                        },
                        "summaries": [],
                    }
                )
        return found

    def write_report(self, name: str, text: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        # Windows clock ticks can repeat within a burst; a random suffix keeps names unique.
        path = self.root / "reports" / f"{_safe(name)}_{stamp}_{os.urandom(3).hex()}.md"
        _write_once(path, text)
        return path

    def write_manual_review(self, review: dict) -> Path:
        """Append one human page review; existing reviews are never replaced."""
        required = {
            "run_id",
            "page",
            "decision",
            "notes",
            "error_categories",
            "model_revision",
            "inference_config",
        }
        missing = sorted(required - set(review))
        if missing:
            raise StoreError(f"Manual review missing fields: {missing}")
        if review["decision"] not in {"PASS", "FAIL", "NEEDS_REVIEW"}:
            raise StoreError("Invalid manual review decision")
        directory = (
            self.root
            / "reviews"
            / _safe(str(review.get("adapter_id") or "unknown"))
            / _safe(str(review["model_revision"]))
            / _safe(str(review["run_id"]))
            / f"p{int(review['page']):04d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        # Windows clock ticks can repeat, so submission order is a per-page sequence
        # number claimed with exclusive creation (never a timestamp or random suffix).
        sequence = len(list(directory.glob("review_*.json"))) + 1
        while True:
            while any(directory.glob(f"review_{sequence:06d}_*.json")):
                sequence += 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = directory / f"review_{sequence:06d}_{stamp}.json"
            try:
                _write_once(
                    path,
                    review
                    | {
                        "schema_version": 1,
                        "review_type": "manual_user_review",
                        "automated_score": False,
                        "sequence": sequence,
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                return path
            except StoreError:
                if any(directory.glob(f"review_{sequence:06d}_*.json")):
                    sequence += 1
                    continue
                raise

    def manual_reviews(self, *, run_id: str | None = None, page: int | None = None) -> list[dict]:
        found: list[dict] = []
        base = self.root / "reviews"
        if not base.is_dir():
            return found
        for path in sorted(base.rglob("review_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if run_id is not None and payload.get("run_id") != run_id:
                continue
            if page is not None and payload.get("page") != page:
                continue
            found.append(payload | {"path": str(path)})
        found.sort(
            key=lambda item: (
                str(item.get("run_id")),
                int(item.get("page") or 0),
                int(item.get("sequence") or 0),
                item.get("submitted_at") or "",
            )
        )
        return found
