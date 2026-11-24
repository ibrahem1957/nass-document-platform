"""Hosted API evaluation runner: one confirmed page per job, resumable, write-once.

A page is submitted to the provider at most once per job. Everything after the
Job ID exists (polling, a paused result-host review, expired result URLs, network
errors) continues from the persisted Job ID with `resume`, never by resubmitting.

Inputs to the provider are exactly one rendered page image and the provider's fixed
request fields. The runner has no Ground Truth, text-layer or prior-output parameter.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import multiprocessing
import platform
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lab import __version__
from lab.core.e2.api import documents
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts
from lab.core.e2.api.authorization import (
    AuthorizationError,
    _append_usage_unlocked,
    _check_authorization_unlocked,
    append_usage,
    check_authorization,
    provider_usage_lock,
)
from lab.core.e2.api.client import redact
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.worker import run_api_worker
from lab.core.e2.metrics_e2 import summarize_run
from lab.core.e2.prompts import get_prompt
from lab.core.e2.render import render_pdf_page
from lab.core.e2.runner import new_run_id
from lab.core.e2.store import ResultStore
from lab.core.metrics import UnverifiedGroundTruthError
from lab.core.models import (
    AdapterInfo,
    GroundTruth,
    LabDocument,
    PageResult,
    RunResult,
    TextBlock,
    UsageRecord,
)
from lab.core.settings import LabSettings
from lab.core.workspace import WorkspaceManager

JPEG_MAGIC = b"\xff\xd8\xff"  # JPEG start-of-image marker
API_IMPLEMENTATION_VERSION = "E2C-2"
INPUT_POLICY = (
    "one rendered page image (PNG) plus the provider's fixed request fields; no Ground "
    "Truth, text layer, other pages or previous model output"
)
NOT_SENT = [
    "Ground Truth (any version)",
    "native PDF text layer",
    "other pages or the PDF file",
    "previous model outputs and metrics",
    "file names other than the generic page name shown",
]
# Submission outcome unknown (the page may have been accepted): never resubmitted.
_UNCONFIRMED_CODES = frozenset(
    {"request_timeout", "network_error", "deadline_exceeded", "worker_exit", "worker_error", "client_exception"}
)

# executor(spec, credential, image_path, filename, job_id, result_hosts, on_event) -> outcome
Executor = Callable[..., dict]
_RENDER_LOCK = threading.Lock()
_RUN_ID = re.compile(r"^run_[0-9a-f]{18}$")


class APIRunRefused(PermissionError):
    pass


@dataclass
class APIJobOutcome:
    state: str
    journal: job_store.JobJournal
    run: RunResult | None
    directory: Path
    artifact_directory: Path | None = None


def verify_document_for_transmission(
    document: LabDocument, pdf_path: Path, *, root: Path | None = None, locked: bool = False,
) -> dict[str, Any]:
    """Exact-hash gate: the current bytes must equal the ingested hash, that hash must be
    an eligible registry entry, and it must carry a valid explicit authorization.

    Folders and filenames are irrelevant. `locked=True` means the caller already holds
    the document-authorization lock (the check that immediately precedes sending).
    """
    current = documents.file_sha256(pdf_path)
    if current != document.sha256:
        raise APIRunRefused(
            "The workspace PDF no longer matches its ingested SHA-256 (modified bytes); "
            "re-import it before any API request"
        )
    check = documents.authorization_state_unlocked if locked else documents.authorization_state
    state = check(current, root=root)
    if not state["allowed"]:
        raise APIRunRefused("; ".join(state["reasons"]))
    return state


def transmission_preview(
    spec: ProviderSpec, manager: WorkspaceManager, settings: LabSettings,
    workspace_id: str, document: LabDocument, pdf_path: Path, page: int,
    *, authorization_root: Path | None = None,
) -> dict[str, Any]:
    """Exactly what one submission would send. The image SHA-256 is what is confirmed."""
    document_state = verify_document_for_transmission(document, pdf_path, root=authorization_root)
    # Concurrent batch items render into the same workspace folder; serializing the
    # write+read keeps one request from hashing a PNG another thread is rewriting.
    with _RENDER_LOCK:
        rendered = render_pdf_page(
            pdf_path, page, manager.path(workspace_id) / "e2_images" / document.doc_id,
            dpi=settings.e2_render_dpi,
        )
        content = Path(rendered.path).read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    if spec.protocol == "openai_chat_completions_v1":
        prompt = get_prompt(spec.prompt_id or "")
        request = {
            "content_type": "application/json",
            "json_body": {
                "model": spec.api_model_id,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt.text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,<the PNG below, sha256 {sha}>"}},
                ]}],
                "stream": bool(spec.stream),
                **spec.generation,
            },
            "prompt": {"id": prompt.prompt_id, "sha256": prompt.sha256},
        }
        follow_up = ["none (synchronous; the transcription is returned in the same response)"]
    else:
        request = {
            "content_type": "multipart/form-data",
            "form_fields": {"model": spec.api_model_id, "optionalPayload": dict(spec.optional_payload)},
        }
        follow_up = [
            f"GET {spec.endpoint}/<jobId> (status polling and URL refresh, same token)",
            "GET <pre-signed result URL> (no token; host must be explicitly approved)",
        ]
    return {
        "provider": spec.provider,
        "provider_id": spec.id,
        "protocol": spec.protocol,
        "method": "POST",
        "endpoint": spec.endpoint,
        "headers": {"Authorization": f"Bearer <{spec.credential_label}: entered, never shown or stored>"},
        **request,
        "file": {
            "field": "file",
            "filename": f"page_{page:04d}.png",
            "content_type": "image/png",
            "bytes": len(content),
            "sha256": sha,
            "width": rendered.width,
            "height": rendered.height,
            "dpi": rendered.dpi,
            "source_page": page,
            "source_document_sha256": document.sha256,
        },
        "image_path": str(rendered.path),
        "follow_up_requests": follow_up,
        "outbound_hosts_allowed": sorted(spec.allowed_hosts),
        "not_sent": NOT_SENT,
        "rendering": {"renderer": rendered.renderer, "dpi": rendered.dpi, "sha256": rendered.sha256},
        "document_authorization": {
            "document": document_state["document"],
            "authorization_path": document_state["authorization"].get("path"),
            "reviewer": document_state["authorization"].get("reviewer"),
            "expires_at": document_state["authorization"].get("expires_at"),
        },
    }


def image_transmission_preview(
    spec: ProviderSpec, document: LabDocument, pdf_path: Path, image_path: Path, *, page: int, region_id: str,
    dpi: int | None = None, authorization_root: Path | None = None,
) -> dict[str, Any]:
    """Like `transmission_preview`, for one already rendered image region (a crop of one page).

    The same exact-hash document gate applies; only the crop's bytes are sent, never the page
    or the PDF. The request fields are the provider's unchanged registry fields."""
    document_state = verify_document_for_transmission(document, pdf_path, root=authorization_root)
    content = Path(image_path).read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    is_jpeg = content[:3] == JPEG_MAGIC
    mime, extension = ("image/jpeg", "jpg") if is_jpeg else ("image/png", "png")
    if spec.protocol == "openai_chat_completions_v1":
        prompt = get_prompt(spec.prompt_id or "")
        request = {
            "content_type": "application/json",
            "json_body": {"model": spec.api_model_id, "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt.text},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,<the {extension.upper()} below, sha256 {sha}>"}},
            ]}], "stream": bool(spec.stream), **spec.generation},
            "prompt": {"id": prompt.prompt_id, "sha256": prompt.sha256},
        }
        follow_up = ["none (synchronous; the transcription is returned in the same response)"]
    else:
        request = {"content_type": "multipart/form-data",
                   "form_fields": {"model": spec.api_model_id, "optionalPayload": dict(spec.optional_payload)}}
        follow_up = [f"GET {spec.endpoint}/<jobId> (status polling and URL refresh, same token)",
                     "GET <pre-signed result URL> (no token; host must be approved)"]
    safe_region = re.sub(r"[^A-Za-z0-9_-]", "_", region_id)[:40]
    return {
        "provider": spec.provider, "provider_id": spec.id, "protocol": spec.protocol, "method": "POST",
        "endpoint": spec.endpoint,
        "headers": {"Authorization": f"Bearer <{spec.credential_label}: entered, never shown or stored>"},
        **request,
        "file": {"field": "file", "filename": f"page_{page:04d}_{safe_region}.{extension}", "content_type": mime,
                 "bytes": len(content), "sha256": sha, "source_page": page, "region_id": region_id,
                 "source_document_sha256": document.sha256, "dpi": dpi, "kind": "image region crop"},
        "image_path": str(image_path), "follow_up_requests": follow_up,
        "outbound_hosts_allowed": sorted(spec.allowed_hosts),
        "not_sent": NOT_SENT + ["the rest of the page outside this region"],
        "rendering": {"renderer": "layout region crop", "dpi": dpi, "sha256": sha},
        "document_authorization": {
            "document": document_state["document"],
            "authorization_path": document_state["authorization"].get("path"),
            "reviewer": document_state["authorization"].get("reviewer"),
            "expires_at": document_state["authorization"].get("expires_at"),
        },
    }


def worker_deadline_s(spec: ProviderSpec) -> float:
    """Hard upper bound on one worker step (the parent kills the worker after it)."""
    limits = spec.limits
    return limits.request_timeout_s * 4 + limits.poll_timeout_s + 30


def resume_claim_stale_s(spec: ProviderSpec) -> float:
    """A RESUMING claim may be taken over only after its worker is certainly gone.

    A fixed 120 s threshold let a second session start a parallel resume while the
    first was still legitimately polling (up to the worker deadline), so both could
    try to finalize the same write-once run.
    """
    return max(job_store.CLAIM_STALE_S, worker_deadline_s(spec) + 60)


_SPAWN_MAIN_LOCK = threading.Lock()


@contextlib.contextmanager
def _neutral_main_module():
    """Start spawned workers without re-running the Streamlit page in the child.

    `spawn` re-imports the parent's `sys.modules['__main__']` in every child. Under
    Streamlit that is the page script being executed (a module without __spec__), so
    each API worker re-ran the whole page in a bare process (repeated "missing
    ScriptRunContext" warnings, slow start, page side effects). While the process is
    started, `__main__` names the worker module instead, which defines functions only.
    """
    with _SPAWN_MAIN_LOCK:
        previous = sys.modules.get("__main__")
        stub = types.ModuleType("__main__")
        stub.__spec__ = importlib.util.find_spec("lab.core.e2.api.worker")
        sys.modules["__main__"] = stub
        try:
            yield
        finally:
            if sys.modules.get("__main__") is stub and previous is not None:
                sys.modules["__main__"] = previous


def subprocess_executor(transport_path: str | None = None, deadline_s: float | None = None) -> Executor:
    """Run one job step in an isolated spawned process with an overall deadline."""

    def execute(spec, credential, image_path, filename, job_id, result_hosts, on_event) -> dict:
        deadline = deadline_s or worker_deadline_s(spec)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=run_api_worker,
            args=(child, spec.model_dump(mode="json"), str(image_path) if image_path else None,
                  filename, transport_path, job_id, sorted(result_hosts)),
            daemon=True,
        )
        with _neutral_main_module():
            process.start()
        child.close()
        known_job = job_id
        end = time.monotonic() + deadline
        try:
            parent.send(credential)
            while True:
                remaining = end - time.monotonic()
                if remaining <= 0 or not parent.poll(remaining):
                    return {"phase": "error", "ok": False, "job_id": known_job, "resumable": bool(known_job),
                            "error": {"code": "deadline_exceeded",
                                      "message": f"No result within {deadline:.0f}s; worker stopped"},
                            "exchanges": []}
                message = parent.recv()
                if message.get("phase") == "submitted":
                    known_job = message.get("job_id")
                    on_event(message)
                    continue
                return message
        except EOFError:
            return {"phase": "error", "ok": False, "job_id": known_job, "resumable": bool(known_job),
                    "error": {"code": "worker_exit", "message": f"API worker exited ({process.exitcode})"},
                    "exchanges": []}
        finally:
            parent.close()
            process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)

    return execute


class APIEvaluationRunner:
    def __init__(self, manager: WorkspaceManager, settings: LabSettings, store: ResultStore,
                 *, authorization_root: Path | None = None):
        self.manager = manager
        self.settings = settings
        self.store = store
        self.authorization_root = authorization_root or store.root

    def preview(self, spec, workspace_id, document, pdf_path, page) -> dict:
        return transmission_preview(spec, self.manager, self.settings, workspace_id, document, pdf_path, page,
                                    authorization_root=self.authorization_root)

    def journals(self, spec: ProviderSpec) -> list[job_store.JobJournal]:
        return job_store.journals(spec.id, self.authorization_root)

    def journal(self, spec: ProviderSpec, journal_id: str) -> job_store.JobJournal:
        journal = job_store.JobJournal(job_store.journal_dir(spec.id, journal_id, self.authorization_root))
        if not journal.events():
            raise APIRunRefused(f"No job journal {journal_id}")
        return journal

    def result_hosts(self, spec: ProviderSpec, journal_id: str | None = None) -> frozenset[str]:
        return job_store.approved_result_hosts(
            spec, self.authorization_root, journal_id=journal_id
        )

    # ------------------------------------------------------------------ submit

    def run(
        self, spec: ProviderSpec, workspace_id: str, document: LabDocument, pdf_path: Path, page: int,
        *, credential: str, confirmed_sha256: str, ground_truth: GroundTruth | None = None,
        executor: Executor | None = None, run_id: str | None = None,
        artifact_context: dict[str, Any] | None = None, preview: dict[str, Any] | None = None,
    ) -> APIJobOutcome:
        """Submit one confirmed page once.

        `preview` (optional) is an `image_transmission_preview` of one region crop: it is sent
        instead of the rendered page, under exactly the same gates.

        `run_id` lets a batch record its intent (with the journal id it will use)
        before calling, so a crash can always be traced to this exact journal.
        """
        if not spec.runnable:
            raise APIRunRefused(f"{spec.id} is {spec.status}; not implemented or not active")
        if run_id is not None and not _RUN_ID.match(run_id):
            raise APIRunRefused("Invalid pre-assigned run id")
        if document.kind != "pdf":
            raise APIRunRefused("Only rendered PDF pages can be sent")
        if not credential or not credential.strip():
            raise APIRunRefused(f"Enter the {spec.credential_label} for this session")
        if preview is None:
            preview = self.preview(spec, workspace_id, document, pdf_path, page)
        elif preview.get("provider_id") != spec.id or (preview.get("file") or {}).get("source_page") != page:
            raise APIRunRefused("The image preview does not belong to this provider and page")
        sha = preview["file"]["sha256"]
        if not confirmed_sha256 or confirmed_sha256 != sha:
            raise APIRunRefused("The confirmed image does not match the image that would be sent")
        run_id = run_id or new_run_id()
        # Quota check, same-image claim, journal creation and quota reservation are one
        # cross-process critical section.  Without this, two different pages could
        # both consume the final allowed slot after observing the same old ledger.
        with provider_usage_lock(spec.id, self.authorization_root), \
                provider_usage_lock(documents.DOCUMENT_LOCK_ID, self.authorization_root):
            state = _check_authorization_unlocked(spec, pages=1, root=self.authorization_root)
            if not state.allowed:
                raise APIRunRefused("; ".join(state.reasons))
            # Immediately before transmission, under the same lock that revocations take:
            # re-verify the document authorization and both byte streams.
            verify_document_for_transmission(document, pdf_path, root=self.authorization_root, locked=True)
            page_bytes = Path(preview["image_path"]).read_bytes()
            if hashlib.sha256(page_bytes).hexdigest() != sha:
                raise APIRunRefused("The rendered page image changed after confirmation; nothing was sent")
            if job_store.journal_dir(spec.id, run_id, self.authorization_root).exists():
                raise APIRunRefused(f"Run id {run_id} was already used; nothing was sent")
            try:
                job_store.claim_submission(spec.id, sha, run_id, self.authorization_root)
            except job_store.JobJournalError as exc:
                raise APIRunRefused(str(exc)) from exc

            started_at = datetime.now(timezone.utc)
            transmission = {key: value for key, value in preview.items() if key != "image_path"}
            handle = self.store.begin(
                adapter_id=spec.adapter_id,
                revision=f"{spec.api_model_id}__{started_at:%Y%m%d}",
                run_id=run_id,
                started={
                    "kind": "hosted_api", "language": "Manual", "adapter_id": spec.adapter_id,
                    "label": spec.label, "pipeline_mode": "hosted_api", "pages": [page],
                    "api_identity": spec.identity(), "transmission": transmission,
                    "authorization": state.to_dict(), "document_id": document.doc_id,
                    "document_sha256": document.sha256,
                    "ground_truth_version": ground_truth.version if ground_truth else None,
                    "artifact_context": artifact_context or {},
                    "job_journal": str(job_store.journal_dir(spec.id, run_id, self.authorization_root)),
                },
            )
            # The exact verified bytes become a write-once part of the run: the worker sends
            # this copy, so later changes to the workspace render cannot alter what is sent.
            transmitted = handle.directory / "transmitted" / preview["file"]["filename"]
            transmitted.parent.mkdir(parents=True, exist_ok=True)
            with transmitted.open("xb") as image_handle:
                image_handle.write(page_bytes)
            journal = job_store.JobJournal(job_store.journal_dir(spec.id, run_id, self.authorization_root))
            journal.append(
                job_store.SUBMITTING, provider_id=spec.id, api_model_id=spec.api_model_id,
                endpoint=spec.endpoint, identity=spec.identity(), page=page, image_sha256=sha,
                document_id=document.doc_id, document_sha256=document.sha256,
                run_id=run_id, run_directory=str(handle.directory), transmission=transmission,
                artifact_context=(artifact_context or {}) | {
                    "api_implementation_version": API_IMPLEMENTATION_VERSION,
                    "input_policy": INPUT_POLICY,
                },
                started_at=started_at.isoformat(),
            )
            # Reserved before the worker starts. Failures still count because the
            # provider may have received the bytes before a local timeout/crash.
            _append_usage_unlocked(
                spec,
                {"at": started_at.isoformat(), "run_id": run_id, "pages": 1,
                 "submitted": True, "image_sha256": sha},
                root=self.authorization_root,
            )

        def on_event(message: dict) -> None:
            if message.get("phase") == "submitted" and message.get("job_id"):
                journal.append(job_store.SUBMITTED, job_id=message["job_id"])

        clock = time.perf_counter()
        outcome = (executor or subprocess_executor())(
            spec, credential, transmitted, preview["file"]["filename"],
            None, frozenset(spec.result_hosts), on_event,
        )
        return self._conclude(spec, journal, handle, outcome, credential, ground_truth,
                              wall_ms=(time.perf_counter() - clock) * 1000, submitted_now=True)

    # ------------------------------------------------------------------ resume

    def resume(
        self, spec: ProviderSpec, journal_id: str, *, credential: str,
        ground_truth: GroundTruth | None = None, executor: Executor | None = None,
    ) -> APIJobOutcome:
        """Continue an existing job from its persisted Job ID. Never resubmits the page."""
        journal = self.journal(spec, journal_id)
        if spec.protocol == "openai_chat_completions_v1":
            raise APIRunRefused("Synchronous chat requests have no job to resume; abandon and resubmit deliberately")
        if not spec.runnable:
            raise APIRunRefused(f"{spec.id} is {spec.status}")
        if journal.first.get("endpoint") != spec.endpoint or journal.first.get("api_model_id") != spec.api_model_id:
            raise APIRunRefused("The provider endpoint/model changed since this job was submitted")
        if not credential or not credential.strip():
            raise APIRunRefused(f"Enter the {spec.credential_label} for this session")
        state = check_authorization(spec, pages=0, root=self.authorization_root)
        if not state.allowed:
            raise APIRunRefused("; ".join(state.reasons))
        # Claim a resume atomically. A fresh RESUMING event means another browser
        # session/process is already polling or downloading this same job.
        with provider_usage_lock(spec.id, self.authorization_root):
            current = journal.state
            if current == job_store.RESUMING:
                try:
                    claimed_at = datetime.fromisoformat(journal.latest()["at"])
                    age = (datetime.now(timezone.utc) - claimed_at).total_seconds()
                except (KeyError, ValueError):
                    age = 0.0
                if age < resume_claim_stale_s(spec):
                    raise APIRunRefused("This job is already being resumed by another request")
            elif current not in job_store.RESUMABLE:
                raise APIRunRefused(f"Job journal {journal_id} is {current}; it cannot be resumed")
            job_id = journal.job_id
            if not job_id:
                raise APIRunRefused("No Job ID was recorded; this job can only be abandoned")
            hosts = self.result_hosts(spec, journal_id)
            if current == job_store.AWAITING_RESULT_HOST_APPROVAL:
                waiting = journal.latest().get("result_url", {}).get("host")
                if waiting not in hosts:
                    raise APIRunRefused(f"Approve the result host {waiting!r} before resuming")
            handle = self.store.reopen(Path(journal.first["run_directory"]))
            journal.append(job_store.RESUMING, job_id=job_id, result_hosts=sorted(hosts))
        append_usage(spec, {"at": datetime.now(timezone.utc).isoformat(), "run_id": journal_id,
                            "pages": 0, "submitted": False, "resume_of_job": job_id},
                     root=self.authorization_root)
        clock = time.perf_counter()
        outcome = (executor or subprocess_executor())(
            spec, credential, None, None, job_id, hosts, lambda _message: None,
        )
        return self._conclude(spec, journal, handle, outcome, credential, ground_truth,
                              wall_ms=(time.perf_counter() - clock) * 1000, submitted_now=False)

    def abandon(self, spec: ProviderSpec, journal_id: str, *, by: str, reason: str) -> APIJobOutcome:
        journal = self.journal(spec, journal_id)
        if not journal.open:
            raise APIRunRefused(f"Job journal {journal_id} is already {journal.state}")
        if not by.strip() or not reason.strip():
            raise APIRunRefused("Give your name and a reason to abandon a job")
        handle = self.store.reopen(Path(journal.first["run_directory"]))
        outcome = {"phase": "error", "ok": False, "job_id": journal.job_id, "resumable": False,
                   "error": {"code": "abandoned_by_user", "message": f"Abandoned by {by.strip()}: {reason.strip()}"},
                   "exchanges": []}
        journal.append(job_store.ABANDONED, job_id=journal.job_id, by=by.strip(), reason=reason.strip())
        run = self._build_run(spec, journal, 0.0, outcome)  # ABANDONED is already the last event
        artifact = self._save_page_artifact(spec, journal, job_store.ABANDONED, outcome, run)
        handle.finalize(run, summaries=[], language="Manual", status="failed",
                        notes=["Hosted API job abandoned by the user; nothing was resubmitted."])
        return APIJobOutcome(job_store.ABANDONED, journal, run, handle.directory, artifact)

    # ---------------------------------------------------------------- outcomes

    def _conclude(self, spec, journal, handle, outcome, credential, ground_truth, *, wall_ms, submitted_now):
        outcome = redact(outcome, [credential.strip()])
        phase = outcome.get("phase")
        job_id = outcome.get("job_id") or journal.job_id
        if submitted_now and job_id and not journal.job_id:
            journal.append(job_store.SUBMITTED, job_id=job_id)  # executor without live events
        if phase == "awaiting_result_host":
            journal.append(job_store.AWAITING_RESULT_HOST_APPROVAL, job_id=job_id,
                           result_url=outcome.get("result_url") or {}, exchanges=outcome.get("exchanges", []))
            artifact = self._save_page_artifact(
                spec, journal, job_store.AWAITING_RESULT_HOST_APPROVAL, outcome, None
            )
            return APIJobOutcome(
                job_store.AWAITING_RESULT_HOST_APPROVAL, journal, None, handle.directory, artifact
            )
        if phase == "error":
            error = outcome.get("error") or {}
            if not job_id and error.get("code") in _UNCONFIRMED_CODES:
                journal.append(job_store.SUBMISSION_UNCONFIRMED, error=error, exchanges=outcome.get("exchanges", []))
                artifact = self._save_page_artifact(
                    spec, journal, job_store.SUBMISSION_UNCONFIRMED, outcome, None
                )
                return APIJobOutcome(
                    job_store.SUBMISSION_UNCONFIRMED, journal, None, handle.directory, artifact
                )
            if job_id and outcome.get("resumable"):
                journal.append(job_store.RESUMABLE_ERROR, job_id=job_id, error=error,
                               exchanges=outcome.get("exchanges", []))
                artifact = self._save_page_artifact(
                    spec, journal, job_store.RESUMABLE_ERROR, outcome, None
                )
                return APIJobOutcome(job_store.RESUMABLE_ERROR, journal, None, handle.directory, artifact)
        final_state = job_store.COMPLETED if outcome.get("ok") else job_store.FAILED
        run = self._build_run(spec, journal, wall_ms, outcome, final_state=final_state)
        first = journal.first
        page = int(first["page"])
        # The provider-page result is durable before legacy aggregate finalization.
        # If the process stops below, Stage 2 can still validate and reuse this atomic
        # page snapshot without waiting for the rest of the document or batch.
        journal.append(final_state, job_id=job_id, run_status=run.status,
                       result_url=(outcome.get("result") or {}).get("result_url"),
                       error=outcome.get("error"))
        artifact = self._save_page_artifact(spec, journal, final_state, outcome, run)
        handle.write_page(page, {
            "phase": "page", "page": page, "ok": bool(outcome.get("ok")), "text": run.pages[0].text,
            "raw": run.environment["e2"]["page_raw"][str(page)]["response"], "elapsed_ms": wall_ms,
            "provider": spec.provider, "api_model_id": spec.api_model_id, "job_id": job_id,
        })
        summaries: list[dict] = []
        if ground_truth is not None and outcome.get("ok"):
            try:
                summaries.append(summarize_run(run, ground_truth) | {"kind": "hosted API"})
            except UnverifiedGroundTruthError:
                pass
        status = "completed" if run.status == "ok" else "failed"
        handle.finalize(run, summaries=summaries, language="Manual", status=status,
                        notes=[f"Hosted API evaluation ({spec.classification}); job {job_id}; "
                               "not part of the air-gapped system."])
        append_usage(spec, {"at": datetime.now(timezone.utc).isoformat(), "run_id": journal.journal_id,
                            "pages": 1, "submitted": False, "outcome": run.status,
                            "cost_usd": run.usage.cost_usd if run.usage else None,
                            "cost_source": run.usage.cost_source if run.usage else "unknown"},
                     root=self.authorization_root)
        return APIJobOutcome(final_state, journal, run, handle.directory, artifact)

    def _save_page_artifact(self, spec, journal, state, outcome, run):
        context = dict(journal.first.get("artifact_context") or {})
        context.setdefault("api_implementation_version", API_IMPLEMENTATION_VERSION)
        context.setdefault("input_policy", INPUT_POLICY)
        return page_artifacts.save_page_snapshot(
            self.store.root, spec, journal, state=state, outcome=outcome, run=run,
            artifact_context=context,
        )

    def _build_run(self, spec, journal, wall_ms, outcome, *, final_state: str | None = None) -> RunResult:
        first = journal.first
        page = int(first["page"])
        transmission = first.get("transmission") or {}
        ok = bool(outcome.get("ok"))
        result = outcome.get("result") or {}
        error = outcome.get("error") or {}
        exchanges = [item for event in journal.events() for item in event.get("exchanges", [])]
        exchanges += result.get("exchanges") if ok else outcome.get("exchanges", [])
        blocks = [
            TextBlock(order=int(item["order"]), text=str(item.get("text") or ""),
                      bbox=tuple(item["bbox"]) if item.get("bbox") else None)
            for item in (result.get("blocks") or [])
        ]
        text = str(result.get("text") or "") if ok else ""
        page_result = PageResult(
            page=page, text=text, blocks=blocks or None,
            status=("ok" if text.strip() else "empty") if ok else "error",
            error=None if ok else f"{error.get('code')}: {error.get('message')}",
            elapsed_ms=wall_ms,
        )
        data_info = result.get("data_info") or {}
        usage = result.get("usage") or {}
        prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
        reported_cost = next(
            (
                source[key]
                for source in (usage, data_info)
                for key in ("cost", "costUsd", "price")
                if key in source
            ),
            None,
        )
        from_usage = spec.pricing.cost_from_usage(prompt_tokens, completion_tokens)
        bound = spec.pricing.compute_bound_usd()
        if isinstance(reported_cost, (int, float)) and not isinstance(reported_cost, bool):
            cost, cost_source, cost_basis = float(reported_cost), "response", "reported by the provider"
        elif bound is not None:
            # Compute-time billing without a usable reported charge: assume the worst case.
            cost, cost_source = bound, "estimated"
            cost_basis = ("compute-time bound (rate x max billable seconds); no usable USD cost reported"
                          + ("" if spec.pricing.price_verified else " (rate UNVERIFIED)"))
        elif from_usage is not None:
            cost, cost_source = from_usage, "estimated"
            cost_basis = ("provider-reported token usage x registry price"
                          + ("" if spec.pricing.price_verified else " (price UNVERIFIED)"))
        elif spec.pricing.estimated_usd_per_page >= 0:
            cost, cost_source, cost_basis = spec.pricing.estimated_usd_per_page, "estimated", "registry per-page estimate"
        else:
            cost, cost_source, cost_basis = None, "unknown", "unknown"
        truncated = result.get("truncated")
        response = {
            "job_id": outcome.get("job_id") or journal.job_id,
            "truncated_at_max_new_tokens": truncated if isinstance(truncated, bool) else None,
            "truncation_status": (
                f"reported (finish_reason={result.get('finish_reason')!r})"
                if isinstance(truncated, bool) else "unknown_not_reported_by_hosted_api"
            ),
            "finish_reason": result.get("finish_reason"),
            "reported_model": result.get("reported_model"),
            "reported_model_matches": result.get("reported_model_matches"),
            "usage": usage,
            "official_normalized": result.get("official_normalized"),
            "raw_pages": result.get("raw_pages"),
            "data_info": data_info,
            "result_jsonl": result.get("jsonl_text"),
            "result_url": result.get("result_url"),
        }
        started_at = datetime.fromisoformat(first.get("started_at") or first["at"])
        return RunResult(
            run_id=journal.journal_id, doc_id=first["document_id"], doc_sha256=first["document_sha256"],
            adapter=AdapterInfo(
                adapter_id=spec.adapter_id, task="ocr", label=spec.label, kind="api",
                library=spec.protocol, library_version=None,
                license="model Apache-2.0; provider terms of service not reviewed",
                model_id=f"{spec.provider}:{spec.api_model_id}",
                params=spec.identity() | {
                    "prompt_used_by_backend": spec.prompt_id is not None,
                    "prompt_sha256": get_prompt(spec.prompt_id).sha256 if spec.prompt_id else None,
                    "reported_model": result.get("reported_model"),
                },
                available=True, unavailable_reason=None, supports_blocks=True,
                implementation_version=API_IMPLEMENTATION_VERSION,
            ),
            variant="raw", pages=[page_result], page_mapping="exact",
            started_at=started_at, wall_ms=wall_ms, cpu_ms=None,
            usage=UsageRecord(
                provider=spec.provider, model=spec.api_model_id, engine=None,
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                pages_billed=1 if ok else None, cost_usd=cost, cost_source=cost_source,
                latency_ms=wall_ms, request_count=len(exchanges), retries=0,
            ),
            status="ok" if ok else "error",
            error=None if ok else page_result.error,
            cache_hit=False,
            environment={
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "lab_version": __version__,
                "api": {
                    "identity": spec.identity(),
                    "transmission": transmission,
                    "job_id": response["job_id"],
                    "job_journal": str(journal.directory),
                    "job_states": [event["state"] for event in journal.events()]
                    + ([final_state] if final_state else []),
                    "exchanges": exchanges,
                    "timings_ms": result.get("timings_ms"),
                    "cost": {"usd": cost, "source": cost_source, "basis": cost_basis,
                             "pricing_basis": spec.pricing.model_dump(),
                             "reported_raw": usage.get("cost_reported_raw", usage.get("cost")),
                             "reported_currency": usage.get("currency"),
                             "preflight_bound_usd": spec.limits.max_cost_usd_per_run,
                             # A charge above the preflight bound means the price assumption is wrong.
                             "exceeds_preflight_bound": bool(
                                 cost_source == "response" and cost is not None
                                 and cost > spec.limits.max_cost_usd_per_run + 1e-9)},
                },
                "e2": {
                    "is_mock": False,
                    "input_policy": INPUT_POLICY,
                    "pipeline_mode": "hosted_api",
                    "images": [(transmission.get("rendering") or {}) | {"page": page, "sha256": first.get("image_sha256")}],
                    "page_raw": {str(page): {"response": response, "elapsed_ms": wall_ms}},
                    "errors": [] if ok else [dict(error) | {"page": page}],
                },
            },
        )


__all__ = [
    "APIEvaluationRunner",
    "APIJobOutcome",
    "APIRunRefused",
    "AuthorizationError",
    "NOT_SENT",
    "subprocess_executor",
    "transmission_preview",
]
