from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import AdapterError, CommandRunner, DocPrepAdapter, NassAdapter
from .config import StudioConfig
from .renderers import load_json, nass_markdown, prepared_markdown

JOB_PATTERN = re.compile(r"^job_[0-9a-f]{20}$")
FINAL_STATES = frozenset({"ready", "failed", "cancelled"})


class JobError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, data: bytes | str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        payload = data.encode("utf-8")
    elif isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    else:
        payload = data
    with path.open("xb") as stream:
        stream.write(payload)


def _safe_filename(value: str) -> str:
    name = Path(value.replace("\\", "/")).name
    cleaned = re.sub(r"[^A-Za-z0-9._\-\u0600-\u06ff ]+", "_", name).strip(" .")
    return cleaned[:180] or "document.pdf"


class JobManager:
    def __init__(self, config: StudioConfig, *, runner: CommandRunner | None = None,
                 nass_adapter: Any | None = None, docprep_adapter: Any | None = None) -> None:
        self.config = config
        self.runner = runner or CommandRunner()
        self.nass = nass_adapter or NassAdapter(config, self.runner)
        self.docprep = docprep_adapter or DocPrepAdapter(config, self.runner)
        self.config.jobs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._recover_interrupted_jobs()

    def _recover_interrupted_jobs(self) -> None:
        """Never silently resume a subprocess whose delivery state is unknown."""
        for folder in self.config.jobs_root.glob("job_*"):
            status = self._read_status(folder)
            if status and status.get("state") in {"queued", "running", "cancelling"}:
                stages = dict(status.get("stages", {}))
                stage = status.get("current_stage")
                if stage in stages and stages[stage] in {"pending", "running"}:
                    stages[stage] = "failed"
                self._update(
                    folder, state="failed", stages=stages,
                    error="Studio restarted during processing. The job was not retried automatically; "
                          "create an explicit retry as a new immutable job.",
                )

    def readiness(self) -> dict[str, Any]:
        return self.nass.readiness()

    def _job_dir(self, job_id: str) -> Path:
        if not JOB_PATTERN.fullmatch(job_id):
            raise JobError("invalid job id")
        path = (self.config.jobs_root / job_id).resolve()
        if path.parent != self.config.jobs_root.resolve():
            raise JobError("job path escapes the configured root")
        return path

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for folder in sorted(self.config.jobs_root.glob("job_*"), reverse=True):
            status = self._read_status(folder)
            if status:
                jobs.append(status)
        return jobs

    def status(self, job_id: str) -> dict[str, Any]:
        status = self._read_status(self._job_dir(job_id))
        if status is None:
            raise JobError("unknown job")
        return status

    @staticmethod
    def _read_status(job_dir: Path) -> dict[str, Any] | None:
        try:
            value = json.loads((job_dir / "processing_status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _update(self, job_dir: Path, **changes: Any) -> dict[str, Any]:
        with self._lock:
            status = self._read_status(job_dir)
            if status is None:
                raise JobError("job status is missing")
            status.update(changes)
            status["updated_at"] = _now()
            temporary = job_dir / f".processing_status.{secrets.token_hex(4)}.tmp"
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, job_dir / "processing_status.json")
            return status

    def create(self, pdf_bytes: bytes, original_filename: str, *, start: bool = True,
               retry_of: str | None = None) -> dict[str, Any]:
        if not pdf_bytes.startswith(b"%PDF-"):
            raise JobError("uploaded file is not a PDF")
        if len(pdf_bytes) > self.config.max_upload_mb * 1024 * 1024:
            raise JobError(f"PDF exceeds the {self.config.max_upload_mb} MB Studio limit")
        if len(pdf_bytes) < 8 or b"%%EOF" not in pdf_bytes[-4096:]:
            raise JobError("PDF is incomplete or malformed")
        filename = _safe_filename(original_filename)
        if not filename.lower().endswith(".pdf"):
            raise JobError("only .pdf uploads are accepted")
        while True:
            job_id = f"job_{secrets.token_hex(10)}"
            job_dir = self.config.jobs_root / job_id
            try:
                job_dir.mkdir(parents=True)
                break
            except FileExistsError:
                continue
        _write_once(job_dir / "original" / "source.pdf", pdf_bytes)
        status = {
            "schema_version": 1, "kind": "document_processing_studio_status", "job_id": job_id,
            "state": "queued", "current_stage": "pdf_received", "original_filename": filename,
            "created_at": _now(), "updated_at": _now(), "retry_of": retry_of,
            "stages": {
                "pdf_received": "completed", "nass_extraction": "pending",
                "docprep_structure": "pending", "docprep_relationships": "pending",
                "docprep_quality": "pending", "docprep_prepared": "pending",
                "document_ready": "pending",
            },
            "error": None, "nass_available": False, "docprep_available": False,
        }
        _write_once(job_dir / "processing_status.json", status)
        if start:
            self.start(job_id)
        return status

    def start(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        status = self.status(job_id)
        if status["state"] not in {"queued"}:
            raise JobError("only a queued job can be started")
        with self._lock:
            if job_id in self._threads:
                raise JobError("job is already running")
            cancel = threading.Event()
            thread = threading.Thread(target=self._process, args=(job_dir, cancel), daemon=True,
                                      name=f"studio-{job_id}")
            self._cancel[job_id] = cancel
            self._threads[job_id] = thread
            thread.start()

    def cancel(self, job_id: str) -> dict[str, Any]:
        job_dir = self._job_dir(job_id)
        with self._lock:
            event = self._cancel.get(job_id)
        if event is None:
            raise JobError("job is not running")
        event.set()
        self.runner.cancel(job_id)
        return self._update(job_dir, state="cancelling", error="Cancellation requested")

    def retry(self, job_id: str, *, start: bool = True) -> dict[str, Any]:
        source = self._job_dir(job_id) / "original" / "source.pdf"
        if not source.is_file():
            raise JobError("original PDF is unavailable")
        previous = self.status(job_id)
        return self.create(source.read_bytes(), previous["original_filename"],
                           start=start, retry_of=job_id)

    def _stage(self, job_dir: Path, stage: str) -> None:
        status = self._read_status(job_dir) or {}
        stages = dict(status.get("stages", {}))
        previous = status.get("current_stage")
        if previous in stages and stages[previous] == "running":
            stages[previous] = "completed"
        stages[stage] = "running"
        self._update(job_dir, state="running", current_stage=stage, stages=stages)

    def _complete_stage(self, job_dir: Path, stage: str) -> None:
        status = self._read_status(job_dir) or {}
        stages = dict(status.get("stages", {}))
        stages[stage] = "completed"
        self._update(job_dir, stages=stages)

    def _diagnostic(self, job_dir: Path, stage: str, exc: Exception) -> None:
        record = {"stage": stage, "created_at": _now(), "error_type": type(exc).__name__,
                  "message": str(exc)}
        if isinstance(exc, AdapterError):
            record.update(stdout=exc.stdout, stderr=exc.stderr)
        _write_once(job_dir / "diagnostics" / f"{stage}.json", record)

    def _process(self, job_dir: Path, cancel: threading.Event) -> None:
        job_id = job_dir.name
        stage = "nass_extraction"
        try:
            status = self.status(job_id)
            self._stage(job_dir, stage)
            nass_root = job_dir / "upstream" / "nass_results"
            nass_run = self.nass.run(
                job_id, job_dir / "original" / "source.pdf", status["original_filename"], nass_root,
                cancel,
            )
            self._snapshot_nass(job_dir, nass_run)
            self._complete_stage(job_dir, stage)
            self._update(job_dir, nass_available=True)
            if cancel.is_set():
                raise AdapterError("cancelled after Nass extraction")

            def on_docprep(name: str) -> None:
                nonlocal stage
                mapping = {
                    "structure": "docprep_structure", "relationships": "docprep_relationships",
                    "quality": "docprep_quality", "prepared": "docprep_prepared",
                }
                new_stage = mapping[name]
                if stage != "nass_extraction":
                    self._complete_stage(job_dir, stage)
                stage = new_stage
                self._stage(job_dir, stage)

            outputs = self.docprep.run_sequence(
                job_id, nass_run, job_dir / "upstream" / "docprep", cancel, on_docprep,
            )
            self._snapshot_docprep(job_dir, nass_run, outputs)
            self._complete_stage(job_dir, stage)
            self._stage(job_dir, "document_ready")
            self._complete_stage(job_dir, "document_ready")
            self._update(job_dir, state="ready", current_stage="document_ready",
                         docprep_available=True, error=None)
        except Exception as exc:  # noqa: BLE001
            self._diagnostic(job_dir, stage, exc)
            status = self._read_status(job_dir) or {}
            stages = dict(status.get("stages", {}))
            if stage in stages:
                stages[stage] = "cancelled" if cancel.is_set() else "failed"
            self._update(job_dir, state="cancelled" if cancel.is_set() else "failed",
                         current_stage=stage, stages=stages, error=str(exc))
        finally:
            with self._lock:
                self._threads.pop(job_id, None)
                self._cancel.pop(job_id, None)

    @staticmethod
    def _snapshot_nass(job_dir: Path, nass_run: Path) -> None:
        document_path = nass_run / "document.json"
        manifest_path = nass_run / "execution_manifest.json"
        document = load_json(document_path, "automatic_extraction_document")
        manifest = load_json(manifest_path, "automatic_extraction_manifest")
        if document.get("run_id") != nass_run.name or manifest.get("run_id") != nass_run.name:
            raise JobError("Nass run identity is inconsistent")
        _write_once(job_dir / "nass" / "document.json", document_path.read_bytes())
        _write_once(job_dir / "nass" / "extracted_content.md", nass_markdown(document))
        _write_once(job_dir / "nass" / "source_run_reference.json", {
            "run_id": nass_run.name, "run_dir": str(nass_run.resolve()),
            "document_sha256": _sha256(document_path), "manifest_sha256": _sha256(manifest_path),
        })

    @staticmethod
    def _snapshot_docprep(job_dir: Path, nass_run: Path, outputs: dict[str, Path]) -> None:
        prepared_dir = outputs["prepared"]
        prepared_path = prepared_dir / "prepared_document.json"
        validation_path = prepared_dir / "prepared_document_validation.json"
        prepared = load_json(prepared_path, "docprep_prepared_document")
        validation = load_json(validation_path, "docprep_prepared_document_validation")
        if (validation.get("summary") or {}).get("error") != 0:
            raise JobError("DocPrep prepared document has validation errors")
        _write_once(job_dir / "docprep" / "prepared_document.json", prepared_path.read_bytes())
        _write_once(job_dir / "docprep" / "organized_document.md", prepared_markdown(prepared))
        references = {
            "nass_run": str(nass_run.resolve()),
            "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
            "prepared_document_sha256": _sha256(prepared_path),
            "prepared_validation_sha256": _sha256(validation_path),
        }
        _write_once(job_dir / "docprep" / "source_run_references.json", references)

    def result_paths(self, job_id: str) -> dict[str, Path]:
        job_dir = self._job_dir(job_id)
        return {
            "pdf": job_dir / "original" / "source.pdf",
            "nass_json": job_dir / "nass" / "document.json",
            "nass_markdown": job_dir / "nass" / "extracted_content.md",
            "prepared_json": job_dir / "docprep" / "prepared_document.json",
            "prepared_markdown": job_dir / "docprep" / "organized_document.md",
        }
