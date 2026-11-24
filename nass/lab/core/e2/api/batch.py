"""E2-D two-stage hosted API batch evaluation, built on `APIEvaluationRunner`.

A batch is a fixed, write-once *plan* (exact page images by SHA-256, providers,
destinations, cost bounds) plus a separate, explicit, write-once *approval* of that
exact plan with per-provider spending and submission limits. The batch never
weakens a single-page gate: every page still goes through `APIEvaluationRunner.run`,
which re-checks document authorization, provider enablement and quota, the
confirmed image hash, duplicate claims, credential isolation and the network guard.

Layout (all write-once; nothing is overwritten or deleted):

    results/api_batches/<definition_id>/<batch_id>/
        plan.json                          the approved object (plan_sha256 inside)
        approvals/approval_*.json          approve / revoke records (newest wins)
        control/eNNNN_<RUN|PAUSE|STOP|...>.json
        items/<item_id>/eNNNN_<INTENT|REFUSED|OUTCOME|...>.json
        .executor.lock                     held (OS lock) while an execution runs

An item's journal id is assigned and recorded (INTENT) *before* the runner is
called, so after a crash every attempt is traceable to its job journal. No
attempt is ever retried automatically once anything may have been transmitted:
ambiguous submissions wait for the user (abandon, then an explicit retry).
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pypdfium2 as pdfium
from pydantic import Field

from lab.core.e2.api import documents
from lab.core.e2.api import host_policy
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts
from lab.core.e2.api.authorization import check_authorization
from lab.core.e2.api.client import redact
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.api.runner import (
    _RENDER_LOCK,
    API_IMPLEMENTATION_VERSION,
    INPUT_POLICY,
    APIEvaluationRunner,
    APIRunRefused,
)
from lab.core.e2.prompts import get_prompt
from lab.core.e2.render import render_pdf_page
from lab.core.e2.runner import new_run_id
from lab.core.models import ContractModel, GroundTruth, LabDocument
from lab.core.settings import LAB_DIR

DEFINITION_PATH = LAB_DIR / "config" / "e2d_traffic_evaluation.toml"
MAX_BATCH_BUDGET_USD = 5.0
MAX_APPROVAL_HOURS = 72
MAX_RETRY_ALLOWANCE = 5
BATCH_APPROVAL_STATEMENT = (
    "I approve sending exactly the page images listed in this stage plan (each identified "
    "by its SHA-256) to the listed providers and destinations, within the spending and "
    "submission limits I set here. This does not replace document authorization, provider "
    "enablement or result-host approval, which are still checked before every page."
)
RETRY_STATEMENT = (
    "I reviewed why this item failed or was abandoned and deliberately approve sending "
    "this same page image to the same provider once more."
)

# Item statuses (derived; never stored as mutable state)
PENDING = "PENDING"
IN_PROGRESS = "IN_PROGRESS"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
ABANDONED = "ABANDONED"
REFUSED = "REFUSED"  # refused before anything was sent
INTERRUPTED_BEFORE_SEND = "INTERRUPTED_BEFORE_SEND"  # crash before the journal existed
NEEDS_HOST_APPROVAL = "NEEDS_HOST_APPROVAL"
READY_TO_RESUME = "READY_TO_RESUME"  # Job ID known; resume polls/downloads, never resubmits
NEEDS_DECISION = "NEEDS_DECISION"  # submission outcome unknown: user must abandon first
RETRY_AUTHORIZED = "RETRY_AUTHORIZED"
REUSED = "REUSED"
BLOCKED = "BLOCKED"
INCOMPLETE_ARTIFACT = "INCOMPLETE_ARTIFACT"

SUBMITTABLE = frozenset({PENDING, REFUSED, INTERRUPTED_BEFORE_SEND, RETRY_AUTHORIZED})
NEEDS_USER = frozenset({
    NEEDS_HOST_APPROVAL, NEEDS_DECISION, FAILED, ABANDONED, BLOCKED, INCOMPLETE_ARTIFACT,
})
DONE = frozenset({COMPLETED, REUSED})

_ITEM_ID = re.compile(r"^[a-z0-9.-]+__p\d{4}__[a-z0-9.-]+$")
_RUN_ID_PATTERN = re.compile(r"^run_[0-9a-f]{18}$")


class BatchError(PermissionError):
    pass


# ------------------------------------------------------------------ definition


class EvaluationDocument(ContractModel):
    registry_id: str
    language: str
    stage1_pages: list[int] = Field(min_length=1)
    stage1_page_roles: dict[str, str] = Field(default_factory=dict)


class EvaluationDefinition(ContractModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$")
    title: str
    providers: list[str] = Field(min_length=1)  # the default selection
    # Providers the user may choose from when creating a plan (defaults to `providers`).
    provider_options: list[str] = Field(default_factory=list)
    max_concurrency: int = Field(ge=1, le=4)
    max_concurrency_per_provider: int = Field(ge=1, le=2)
    batch_deadline_minutes: int = Field(ge=1, le=480)
    ambiguous_stop_threshold: int = Field(ge=1, le=5)
    typical_prompt_tokens: int = Field(ge=0)
    typical_completion_tokens: int = Field(ge=0)
    typical_compute_seconds: float = Field(default=20.0, ge=0)
    documents: list[EvaluationDocument] = Field(min_length=1)

    @property
    def options(self) -> list[str]:
        return list(dict.fromkeys(self.provider_options or self.providers))

    def selection(self, providers: list[str] | None) -> list[str]:
        chosen = list(providers or self.providers)
        if not chosen or len(set(chosen)) != len(chosen):
            raise BatchError("Choose one or more distinct providers")
        outside = [item for item in chosen if item not in self.options]
        if outside:
            raise BatchError(f"Not a provider option of this evaluation: {outside}")
        return chosen


def load_definition(path: Path | None = None) -> tuple[EvaluationDefinition, str]:
    source = Path(path or DEFINITION_PATH)
    data = source.read_bytes()
    return EvaluationDefinition.model_validate(tomllib.loads(data.decode("utf-8"))), hashlib.sha256(data).hexdigest()


@dataclass
class SourceDocument:
    """A workspace document matched to a registry entry by exact SHA-256 only."""

    entry: documents.SyntheticDocument
    document: LabDocument
    pdf_path: Path
    language: str


def resolve_sources(
    definition: EvaluationDefinition, workspace_documents: list[tuple[LabDocument, Path]],
    registry: dict[str, documents.SyntheticDocument] | None = None,
) -> tuple[dict[str, SourceDocument], list[str]]:
    """Match the definition's registry entries to workspace PDFs by re-hashed bytes.

    Names and folders are ignored; an unrelated file can never be selected.
    """
    eligible = registry if registry is not None else documents.load_registry()
    by_id = {entry.id: entry for entry in eligible.values()}
    found: dict[str, SourceDocument] = {}
    problems: list[str] = []
    for wanted in definition.documents:
        entry = by_id.get(wanted.registry_id)
        if entry is None:
            problems.append(f"{wanted.registry_id}: not in the synthetic document registry")
            continue
        for document, path in workspace_documents:
            if document.kind != "pdf" or document.sha256 != entry.sha256:
                continue
            try:
                actual = documents.file_sha256(path)
            except OSError:
                continue
            if actual == entry.sha256 and Path(path).stat().st_size == entry.bytes \
                    and document.page_count == entry.page_count:
                found[entry.id] = SourceDocument(entry, document, Path(path), wanted.language)
                break
        else:
            problems.append(
                f"{wanted.registry_id}: no workspace PDF with SHA-256 {entry.sha256[:12]}… "
                f"({entry.bytes:,} bytes, {entry.page_count} pages); upload it on Documents"
            )
    return found, problems


# ---------------------------------------------------------------- utilities


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_once(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def batches_root(root: Path, definition_id: str) -> Path:
    return Path(root) / "api_batches" / definition_id


def item_id_for(registry_id: str, page: int, provider_id: str) -> str:
    value = f"{registry_id}__p{page:04d}__{provider_id}"
    if not _ITEM_ID.match(value):
        raise BatchError(f"Unusable item id {value!r}")
    return value


def max_cost_per_page(spec: ProviderSpec) -> float:
    """The enforced upper bound per submitted page (the runner's own per-run cap)."""
    return max(float(spec.limits.max_cost_usd_per_run), float(max(spec.pricing.estimated_usd_per_page, 0.0)))


def typical_cost_per_page(spec: ProviderSpec, definition: EvaluationDefinition) -> float | None:
    if spec.pricing.basis == "per_token":
        return spec.pricing.cost_from_usage(definition.typical_prompt_tokens, definition.typical_completion_tokens)
    if spec.pricing.basis == "per_compute_second":
        return float(spec.pricing.usd_per_compute_second or 0) * definition.typical_compute_seconds
    return spec.pricing.estimated_usd_per_page if spec.pricing.estimated_usd_per_page >= 0 else None


def _fingerprint(identity: dict, document_sha256: str | None, image_sha256: str | None,
                 prompt_sha256: str | None, rendering: dict, api_implementation_version: str | None,
                 input_policy: str | None) -> str:
    return hashlib.sha256(_canonical({
        "identity": identity,
        "document_sha256": document_sha256,
        "image_sha256": image_sha256,
        "prompt_sha256": prompt_sha256,
        "render": {"dpi": rendering.get("dpi"), "renderer": rendering.get("renderer"),
                   "width": rendering.get("width"), "height": rendering.get("height")},
        "api_implementation_version": api_implementation_version,
        "input_policy": input_policy,
    })).hexdigest()


def item_fingerprint(spec: ProviderSpec, document_sha256: str, image_sha256: str, rendering: dict) -> str:
    """Everything that must match for an earlier result to stand in for a new one."""
    return _fingerprint(
        spec.identity(), document_sha256, image_sha256,
        get_prompt(spec.prompt_id).sha256 if spec.prompt_id else None,
        rendering, API_IMPLEMENTATION_VERSION, INPUT_POLICY,
    )


def page_features(pdf_path: Path, page: int) -> dict[str, Any]:
    """Local, heuristic page description (never sent): text chars, images, vector paths."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pdf_page = pdf[page - 1]
        try:
            text_page = pdf_page.get_textpage()
            try:
                chars = text_page.count_chars()
            finally:
                text_page.close()
            counts = {"image": 0, "path": 0, "text": 0}
            for obj in pdf_page.get_objects():
                kind = {1: "text", 2: "path", 3: "image"}.get(getattr(obj, "type", 0))
                if kind:
                    counts[kind] += 1
        finally:
            pdf_page.close()
    finally:
        pdf.close()
    kinds = []
    if counts["image"]:
        kinds.append("images/charts")
    if counts["path"] >= 100:
        kinds.append("table-like vector drawing")
    if chars:
        kinds.append("text")
    return {"text_chars": chars, "image_objects": counts["image"], "path_objects": counts["path"],
            "heuristic_content": kinds or ["no text layer"]}


def text_layer(pdf_path: Path, page: int) -> str:
    """Library extraction of the PDF's own text layer (pypdfium2). Never sent; never
    Ground Truth; used only as a clearly labelled, unverified reference."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        pdf_page = pdf[page - 1]
        try:
            text_page = pdf_page.get_textpage()
            try:
                return text_page.get_text_range().replace("\r\n", "\n").replace("\r", "\n")
            finally:
                text_page.close()
        finally:
            pdf_page.close()
    finally:
        pdf.close()


# -------------------------------------------------------------------- planning


def render_page(manager, settings, workspace_id: str, source: SourceDocument, page: int):
    """Same folder and DPI as the runner's preview (so hashes agree), under the same
    lock, so a local render never rewrites a PNG while a submission is hashing it."""
    with _RENDER_LOCK:
        return render_pdf_page(
            source.pdf_path, page, manager.path(workspace_id) / "e2_images" / source.document.doc_id,
            dpi=settings.e2_render_dpi,
        )


def build_plan(
    definition: EvaluationDefinition, definition_sha256: str, stage: int, *,
    sources: dict[str, SourceDocument], specs: dict[str, ProviderSpec], manager, settings,
    workspace_id: str, root: Path, stage1_batch: Path | None = None, providers: list[str] | None = None,
) -> dict[str, Any]:
    """Plan one stage locally (render + hash; nothing is sent). Not yet persisted."""
    if stage not in (1, 2):
        raise BatchError("Stage must be 1 or 2")
    if stage == 1 and stage1_batch is not None:
        raise BatchError("Stage 1 has no prior stage")
    missing = [item.registry_id for item in definition.documents if item.registry_id not in sources]
    if missing:
        raise BatchError(f"Documents not available by exact hash: {missing}")
    selected = definition.selection(providers)
    missing_specs = [item for item in selected if item not in specs]
    if missing_specs:
        raise BatchError(f"Providers not in the registry: {missing_specs}")
    prior = None
    if stage1_batch is not None:
        prior = load_plan(stage1_batch)
        if prior["stage"] != 1 or prior["definition_id"] != definition.id:
            raise BatchError("The prior batch is not a Stage 1 plan of this evaluation")
    items: list[dict] = []
    for wanted in definition.documents:
        source = sources[wanted.registry_id]
        pages = list(wanted.stage1_pages) if stage == 1 else list(range(1, source.entry.page_count + 1))
        if any(page < 1 or page > source.entry.page_count for page in pages):
            raise BatchError(f"{wanted.registry_id}: page outside 1..{source.entry.page_count}")
        for page in pages:
            rendered = render_page(manager, settings, workspace_id, source, page)
            rendering = {"dpi": rendered.dpi, "renderer": rendered.renderer,
                         "width": rendered.width, "height": rendered.height,
                         "sha256": rendered.sha256}
            image_bytes = Path(rendered.path).stat().st_size
            features = page_features(source.pdf_path, page)
            for provider_id in selected:
                spec = specs[provider_id]
                item = {
                    "item_id": item_id_for(wanted.registry_id, page, provider_id),
                    "registry_id": wanted.registry_id,
                    "language": wanted.language,
                    "document_sha256": source.entry.sha256,
                    "doc_id": source.document.doc_id,
                    "page": page,
                    "page_role": wanted.stage1_page_roles.get(str(page)),
                    "page_features": features,
                    "provider_id": provider_id,
                    "destination": spec.endpoint,
                    "destination_hosts": sorted(spec.allowed_hosts),
                    "image_sha256": rendered.sha256,
                    "image_bytes": image_bytes,
                    "rendering": rendering,
                    "fingerprint": item_fingerprint(spec, source.entry.sha256, rendered.sha256, rendering),
                    "max_cost_usd": max_cost_per_page(spec),
                    "typical_cost_usd": typical_cost_per_page(spec, definition),
                    "action": "submit",
                    "reason": None,
                }
                if prior is not None:
                    _apply_reuse(item, spec, prior, stage1_batch, root)
                if item["action"] == "submit":
                    # Any earlier completed page result (manual or batch, any upload
                    # session) that matches exactly; never for blocked items.
                    _apply_saved_reuse(item, spec, root)
                items.append(item)
    providers = []
    for provider_id in selected:
        spec = specs[provider_id]
        mine = [item for item in items if item["provider_id"] == provider_id]
        submit = [item for item in mine if item["action"] == "submit"]
        typical = [item["typical_cost_usd"] for item in submit]
        providers.append({
            "provider_id": provider_id,
            "label": spec.label,
            "identity": spec.identity(),
            "destination": spec.endpoint,
            "destination_hosts": sorted(spec.allowed_hosts),
            "pricing": spec.pricing.model_dump(),
            "max_cost_per_page_usd": max_cost_per_page(spec),
            "typical_cost_per_page_usd": typical_cost_per_page(spec, definition),
            "submit_count": len(submit),
            "reuse_count": sum(item["action"] == "reuse" for item in mine),
            "blocked_count": sum(item["action"] == "blocked" for item in mine),
            "max_total_cost_usd": round(sum(item["max_cost_usd"] for item in submit), 6),
            "typical_total_cost_usd": None if any(value is None for value in typical) else round(sum(typical), 6),
        })
    stamp = _now().strftime("%Y%m%dT%H%M%S%fZ")
    plan = {
        "schema_version": 1,
        "batch_id": f"stage{stage}_{stamp}_{os.urandom(3).hex()}",
        "definition_id": definition.id,
        "definition_title": definition.title,
        "definition_sha256": definition_sha256,
        "stage": stage,
        "created_at": _now().isoformat(),
        "workspace_id": workspace_id,
        "prior_stage_batch": prior["batch_id"] if prior else None,
        "selected_providers": selected,
        "execution": {
            "max_concurrency": definition.max_concurrency,
            "max_concurrency_per_provider": definition.max_concurrency_per_provider,
            "batch_deadline_minutes": definition.batch_deadline_minutes,
            "ambiguous_stop_threshold": definition.ambiguous_stop_threshold,
        },
        "documents": [
            {"registry_id": source.entry.id, "sha256": source.entry.sha256, "doc_id": source.document.doc_id,
             "bytes": source.entry.bytes, "page_count": source.entry.page_count, "language": source.language,
             "title": source.entry.title, "provenance": source.entry.provenance}
            for source in (sources[item.registry_id] for item in definition.documents)
        ],
        "providers": providers,
        "items": items,
        "not_sent": [
            "Ground Truth (none exists for these documents)", "PDF text layer", "the PDF file",
            "other pages", "previous outputs or metrics",
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(_canonical(plan)).hexdigest()
    return plan


def _apply_reuse(item: dict, spec: ProviderSpec, prior: dict, prior_dir: Path, root: Path) -> None:
    """Reuse a Stage 1 result only when everything that defines it matches exactly."""
    earlier = next((other for other in prior["items"] if other["item_id"] == item["item_id"]), None)
    if earlier is None:
        return
    status = item_status(prior_dir, earlier, root)
    state = status["status"]
    if state in {NEEDS_HOST_APPROVAL, READY_TO_RESUME, NEEDS_DECISION, IN_PROGRESS}:
        item["action"] = "blocked"
        item["reason"] = f"Stage 1 job for this page is unresolved ({state}); resume or abandon it in Stage 1 first"
        return
    if state != COMPLETED:
        item["reason"] = f"Stage 1 result not reusable ({state}); planned for submission"
        return
    run_dir = Path(status["run_directory"] or "")
    problems = []
    if earlier.get("fingerprint") != item["fingerprint"]:
        problems.append("fingerprint (identity/prompt/settings/image) differs")
    artifact_path = status.get("artifact_directory")
    if artifact_path:
        expected = _artifact_identity(item, spec)
        integrity = page_artifacts.verify_snapshot(Path(artifact_path), expected)
        if not integrity["reusable"]:
            problems.append("stored page artifact is not reusable: " + "; ".join(integrity["problems"]))
        run_dir = Path(artifact_path)
    else:
        # Backward-compatible verification for results created before page artifacts.
        started = _read_json(run_dir / "started.json") or {}
        sent = (started.get("transmission") or {}).get("file") or {}
        if sent.get("sha256") != item["image_sha256"]:
            problems.append("transmitted image hash differs")
        if started.get("document_sha256") != item["document_sha256"]:
            problems.append("document hash differs")
        if started.get("api_identity") != spec.identity():
            problems.append("provider identity differs")
        integrity = verify_run_integrity(run_dir)
        if not integrity["ok"]:
            problems.append("stored run failed integrity: " + "; ".join(integrity["problems"]))
    run = _read_json(run_dir / "run.json") or {}
    if run.get("status") != "ok":
        problems.append(f"stored run status is {run.get('status')!r}")
    if problems:
        item["reason"] = "Stage 1 result not reused: " + "; ".join(problems)
        return
    item["action"] = "reuse"
    item["max_cost_usd"] = 0.0
    item["typical_cost_usd"] = 0.0
    item["reason"] = "Exact match with a completed Stage 1 result (identity, document, image, prompt, settings)"
    item["reuse_of"] = {"batch_id": prior["batch_id"], "item_id": earlier["item_id"],
                        "run_id": run.get("run_id"), "run_directory": str(run_dir),
                        "artifact_directory": artifact_path}


def check_saved_snapshot(snapshot: Path, item: dict, spec: ProviderSpec, root: Path) -> dict[str, Any]:
    """Strictly decide whether one saved page snapshot can stand in for `item`.

    Identified by exact document SHA-256 (upload-session document ids are ignored),
    page and provider; requires full integrity, a COMPLETED job journal and run, an
    exact match of every identity field, and the fingerprint recomputed from the
    snapshot's own recorded fields to equal the plan's fingerprint.
    """
    checked = page_artifacts.verify_snapshot(Path(snapshot))
    problems = list(checked["problems"])
    metadata = checked.get("metadata") or {}
    if not checked["reusable"] and not problems:
        problems.append(f"not a reusable completed result (state {metadata.get('state')!r})")
    if problems:
        return {"ok": False, "problems": problems, "metadata": metadata}
    if metadata.get("document_sha256") != item["document_sha256"]:
        problems.append("document SHA-256 differs")
    if metadata.get("page") != item["page"]:
        problems.append("page differs")
    if metadata.get("provider_id") != spec.id:
        problems.append("provider differs")
    saved = metadata.get("reuse_identity") or {}
    expected = _artifact_identity(item, spec)
    differing = sorted(
        key for key in set(saved) | set(expected)
        if key != "batch_fingerprint" and saved.get(key) != expected.get(key)
    )
    if differing:
        problems.append("identity differs: " + ", ".join(differing))
    recomputed = _fingerprint(
        saved.get("provider_identity") or {}, saved.get("document_sha256"), saved.get("image_sha256"),
        saved.get("prompt_sha256"), saved.get("rendering") or {},
        saved.get("api_implementation_version"), saved.get("input_policy"),
    )
    if recomputed != item["fingerprint"]:
        problems.append("fingerprint recomputed from the saved result differs")
    if saved.get("batch_fingerprint") not in (None, item["fingerprint"]):
        problems.append("saved batch fingerprint differs")
    run_id = str(metadata.get("run_id") or "")
    if not _RUN_ID_PATTERN.match(run_id):
        problems.append("unusable run id")
    else:
        journal = job_store.JobJournal(job_store.journal_dir(spec.id, run_id, root))
        if journal.state != job_store.COMPLETED:
            problems.append(f"job journal is {journal.state!r}, not COMPLETED")
    run = _read_json(Path(snapshot) / "run.json") or {}
    if run.get("status") != "ok":
        problems.append(f"stored run status is {run.get('status')!r}")
    complete = _read_json(Path(snapshot) / "complete.json") or {}
    return {"ok": not problems, "problems": problems, "metadata": metadata, "run": run,
            "manifest_sha256": complete.get("manifest_sha256")}


def find_saved_result(item: dict, spec: ProviderSpec, root: Path) -> tuple[dict | None, list[str]]:
    """The newest saved completed result that can stand in for `item` (or None),
    plus the reasons completed candidates were refused."""
    base = Path(root) / "pages" / page_artifacts.provider_folder(spec.id)
    pattern = f"*--{item['document_sha256'][:12].lower()}/page-{int(item['page']):04d}/run_*/snapshot-*"
    candidates = sorted(base.glob(pattern)) if base.is_dir() else []
    accepted = []
    rejected = []
    for snapshot in candidates:
        result = check_saved_snapshot(snapshot, item, spec, root)
        if result["ok"]:
            accepted.append((str(result["metadata"].get("saved_at") or ""), snapshot, result))
        elif (result.get("metadata") or {}).get("state") == job_store.COMPLETED:
            rejected.append(f"{snapshot.parent.name}: " + "; ".join(result["problems"]))
    if not accepted:
        return None, rejected
    _saved_at, snapshot, result = max(accepted, key=lambda entry: (entry[0], str(entry[1])))
    metadata = result["metadata"]
    return {
        "source": "saved_page_artifact",
        "batch_id": metadata.get("batch_id"), "item_id": metadata.get("batch_item_id"),
        "run_id": metadata.get("run_id"), "run_directory": str(snapshot),
        "artifact_directory": str(snapshot), "manifest_sha256": result["manifest_sha256"],
        "document_id_at_run": metadata.get("document_id"), "saved_at": metadata.get("saved_at"),
    }, rejected


def _apply_saved_reuse(item: dict, spec: ProviderSpec, root: Path) -> None:
    reuse_of, rejected = find_saved_result(item, spec, root)
    if reuse_of is None:
        open_jobs = job_store.open_journals_for_image(spec.id, item["image_sha256"], root)
        if open_jobs:
            # An unfinished job (e.g. awaiting host approval or with an unknown outcome)
            # for this exact page image: never plan a second submission for it.
            job = open_jobs[0]
            item["action"] = "blocked"
            item["reason"] = (f"An unfinished job exists for this page image (run {job.journal_id}, state "
                              f"{job.state}); resolve it in its own batch first")
            return
        if rejected:
            note = "Saved completed result not reused: " + " | ".join(rejected)
            item["reason"] = f"{item['reason']}; {note}" if item.get("reason") else note
        return
    item["action"] = "reuse"
    item["max_cost_usd"] = 0.0
    item["typical_cost_usd"] = 0.0
    item["reason"] = ("Exact match with a saved completed result (document SHA-256, page, image, rendering, "
                      "provider identity, prompt, settings); no new request")
    item["reuse_of"] = reuse_of


def _artifact_identity(item: dict, spec: ProviderSpec) -> dict[str, Any]:
    return {
        "document_sha256": item["document_sha256"],
        "page": item["page"],
        "provider_id": spec.id,
        "provider_identity": spec.identity(),
        "model_id": spec.api_model_id,
        "prompt_id": spec.prompt_id,
        "prompt_sha256": get_prompt(spec.prompt_id).sha256 if spec.prompt_id else None,
        "rendering": item["rendering"],
        "image_sha256": item["image_sha256"],
        "generation": dict(spec.generation),
        "protocol": spec.protocol,
        "api_implementation_version": API_IMPLEMENTATION_VERSION,
        "input_policy": INPUT_POLICY,
        "batch_fingerprint": item["fingerprint"],
    }


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_run_integrity(run_dir: Path) -> dict[str, Any]:
    final = _read_json(Path(run_dir) / "final.json")
    if not final:
        return {"ok": False, "problems": ["no final.json"]}
    problems = []
    for relative, expected in (final.get("files_sha256") or {}).items():
        path = Path(run_dir) / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif documents.file_sha256(path) != expected:
            problems.append(f"modified {relative}")
    return {"ok": not problems, "problems": problems}


def save_plan(plan: dict, root: Path) -> Path:
    expected = hashlib.sha256(_canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest()
    if expected != plan.get("plan_sha256"):
        raise BatchError("Plan content does not match its plan_sha256")
    directory = batches_root(root, plan["definition_id"]) / plan["batch_id"]
    directory.mkdir(parents=True, exist_ok=False)
    _write_once(directory / "plan.json", plan)
    return directory


def load_plan(batch_dir: Path) -> dict:
    plan = _read_json(Path(batch_dir) / "plan.json")
    if not plan:
        raise BatchError(f"No readable plan in {batch_dir}")
    expected = hashlib.sha256(_canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest()
    if expected != plan.get("plan_sha256"):
        raise BatchError("The stored plan was modified after creation (plan_sha256 mismatch)")
    return plan


def list_batches(root: Path, definition_id: str, *, stage: int | None = None) -> list[Path]:
    base = batches_root(root, definition_id)
    found = []
    for path in sorted(base.iterdir()) if base.is_dir() else []:
        plan = _read_json(path / "plan.json")
        if plan and (stage is None or plan.get("stage") == stage):
            found.append(path)
    found.sort(key=lambda path: (_read_json(path / "plan.json") or {}).get("created_at") or "")
    return found


# -------------------------------------------------------------------- approval


def approve_plan(
    batch_dir: Path, *, approved_by: str, statement: str, typed_plan_prefix: str,
    budgets_usd: dict[str, float], hours_valid: int, retry_allowance: int = 0,
) -> Path:
    plan = load_plan(batch_dir)
    if statement != BATCH_APPROVAL_STATEMENT:
        raise BatchError("The batch approval statement was not accepted")
    if not approved_by.strip():
        raise BatchError("Enter the name of the person approving this batch")
    if typed_plan_prefix.strip().lower() != plan["plan_sha256"][:12]:
        raise BatchError("Type the first 12 characters of the plan SHA-256 exactly")
    if not 1 <= int(hours_valid) <= MAX_APPROVAL_HOURS:
        raise BatchError(f"Validity must be 1..{MAX_APPROVAL_HOURS} hours")
    if not 0 <= int(retry_allowance) <= MAX_RETRY_ALLOWANCE:
        raise BatchError(f"Retry allowance must be 0..{MAX_RETRY_ALLOWANCE}")
    limits = {}
    for provider in plan["providers"]:
        if not provider["submit_count"]:
            continue
        budget = float(budgets_usd.get(provider["provider_id"], -1))
        if not 0 <= budget <= MAX_BATCH_BUDGET_USD:
            raise BatchError(f"Budget for {provider['provider_id']} must be 0..{MAX_BATCH_BUDGET_USD} USD")
        if budget < provider["max_cost_per_page_usd"]:
            raise BatchError(f"Budget for {provider['provider_id']} is below the cost bound of a single page")
        limits[provider["provider_id"]] = {
            "budget_usd": budget,
            "max_submissions": provider["submit_count"] + int(retry_allowance),
            "max_cost_per_page_usd": provider["max_cost_per_page_usd"],
        }
    if not limits:
        raise BatchError("This plan submits nothing; there is nothing to approve")
    now = _now()
    return write_sequenced(Path(batch_dir) / "approvals", "approval", {
        "schema_version": 1, "approved": True, "batch_id": plan["batch_id"], "stage": plan["stage"],
        "plan_sha256": plan["plan_sha256"], "approved_by": approved_by.strip(), "statement": statement,
        "limits": limits, "retry_allowance": int(retry_allowance),
        "items_to_submit": sum(item["action"] == "submit" for item in plan["items"]),
        "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=int(hours_valid))).isoformat(),
    })


def revoke_plan_approval(batch_dir: Path, *, revoked_by: str, reason: str) -> Path:
    if not revoked_by.strip() or not reason.strip():
        raise BatchError("Give your name and a reason to revoke")
    now = _now()
    return write_sequenced(Path(batch_dir) / "approvals", "approval", {
        "schema_version": 1, "approved": False, "revoked_by": revoked_by.strip(), "reason": reason.strip(),
        "created_at": now.isoformat(),
    })


def approval_state(batch_dir: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        plan = load_plan(batch_dir)
    except BatchError as exc:
        return {"allowed": False, "reasons": [str(exc)], "approval": None}
    records = ordered(Path(batch_dir) / "approvals", "approval")
    if any(payload is None for _path, payload in records):
        # An unreadable approval record fails closed (treated as a revocation).
        latest = {"approved": False, "corrupt": True}
    else:
        latest = (records[-1][1] | {"path": str(records[-1][0])}) if records else None
    if latest is None:
        reasons.append("This stage plan has not been approved")
    elif not latest.get("approved"):
        reasons.append("The batch approval was revoked" + (" (unreadable record)" if latest.get("corrupt") else ""))
    else:
        if latest.get("plan_sha256") != plan["plan_sha256"] or latest.get("batch_id") != plan["batch_id"]:
            reasons.append("The approval does not match this exact plan")
        if latest.get("statement") != BATCH_APPROVAL_STATEMENT:
            reasons.append("The approval lacks the required statement")
        try:
            if datetime.fromisoformat(latest["expires_at"]) <= _now():
                reasons.append("The batch approval has expired")
        except (KeyError, ValueError):
            reasons.append("The batch approval has no valid expiry")
    return {"allowed": not reasons, "reasons": reasons, "approval": latest}


# ---------------------------------------------------------------- control


CONTROL_ACTIONS = frozenset({"RUN", "PAUSE", "STOP"})


def _log(directory: Path) -> job_store.JobJournal:
    return job_store.JobJournal(directory)


def request_control(batch_dir: Path, action: str, *, by: str, note: str = "") -> dict:
    if action not in CONTROL_ACTIONS:
        raise BatchError(f"Unknown control action {action!r}")
    if not by.strip():
        raise BatchError("Enter your name")
    return _log(Path(batch_dir) / "control").append(action, by=by.strip(), note=note.strip())


def control_state(batch_dir: Path) -> str:
    events = [item for item in _log(Path(batch_dir) / "control").events() if item["state"] in CONTROL_ACTIONS]
    return events[-1]["state"] if events else "NOT_STARTED"


def _record(batch_dir: Path, kind: str, **fields) -> dict:
    return _log(Path(batch_dir) / "control").append(kind, **fields)


def control_events(batch_dir: Path) -> list[dict]:
    return _log(Path(batch_dir) / "control").events()


# ------------------------------------------------------------ executor lock


_ACTIVE_GUARD = threading.Lock()
_ACTIVE: set[str] = set()


@contextmanager
def executor_lock(batch_dir: Path):
    """Exclusive, non-blocking: one execution per batch across threads and processes.

    The OS releases the lock if the process dies, so a crash leaves no stale lock.
    """
    path = Path(batch_dir) / ".executor.lock"
    key = str(path.resolve()).lower()
    with _ACTIVE_GUARD:
        if key in _ACTIVE:
            raise BatchError("This batch is already executing")
        _ACTIVE.add(key)
    handle = None
    try:
        handle = path.open("a+b")
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BatchError("This batch is already executing in another process") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt  # noqa: PLC0415

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # noqa: PLC0415

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        if handle is not None:
            handle.close()
        with _ACTIVE_GUARD:
            _ACTIVE.discard(key)


def executor_active(batch_dir: Path) -> bool:
    try:
        with executor_lock(batch_dir):
            return False
    except BatchError:
        return True


# ------------------------------------------------------------- item status


def _item_log(batch_dir: Path, item_id: str) -> job_store.JobJournal:
    if not _ITEM_ID.match(item_id):
        raise BatchError(f"Unusable item id {item_id!r}")
    return _log(Path(batch_dir) / "items" / item_id)


def item_status(batch_dir: Path, item: dict, root: Path, *, active: bool | None = None) -> dict[str, Any]:
    """Derive an item's status from its write-once log and its job journal."""
    base = {"item_id": item["item_id"], "provider_id": item["provider_id"], "page": item["page"],
            "registry_id": item["registry_id"], "attempts": 0, "journal_id": None, "job_id": None,
            "run_directory": None, "awaiting_host": None, "detail": item.get("reason"), "cost_usd": None,
            "cost_source": None, "artifact_directory": None, "artifact_complete": False,
            "reusable_saved_result": False}
    if item["action"] == "reuse":
        reuse = item.get("reuse_of") or {}
        saved = reuse.get("artifact_directory") or reuse.get("run_directory")
        if reuse.get("artifact_directory"):
            # Re-verified every time: a reused snapshot that was altered or removed
            # after planning is never presented as a valid result.
            checked = page_artifacts.verify_snapshot(Path(reuse["artifact_directory"]))
            complete = _read_json(Path(reuse["artifact_directory"]) / "complete.json") or {}
            recorded = reuse.get("manifest_sha256")
            if not checked["reusable"] or (recorded and complete.get("manifest_sha256") != recorded):
                return base | {
                    "status": INCOMPLETE_ARTIFACT, "artifact_directory": saved,
                    "detail": "The reused saved result changed or is no longer valid ("
                              + ("; ".join(checked["problems"]) or "manifest changed")
                              + "); create a new plan",
                }
        return base | {
            "status": REUSED,
            "run_directory": reuse.get("run_directory"),
            "artifact_directory": saved,
            "artifact_complete": True,
            "reusable_saved_result": True,
            "cost_usd": 0.0,
            "cost_source": "reused (no new request)",
        }
    if item["action"] == "blocked":
        return base | {"status": BLOCKED}
    events = _item_log(batch_dir, item["item_id"]).events()
    intents = [event for event in events if event["state"] == "INTENT"]
    if not intents:
        skipped = [event for event in events if event["state"] == "SKIPPED"]
        return base | {"status": PENDING, "detail": skipped[-1]["reason"] if skipped else base["detail"]}
    last = intents[-1]
    after = [event for event in events if event["sequence"] > last["sequence"]]
    run_id = last["run_id"]
    base |= {"attempts": len(intents), "journal_id": run_id}
    retry = any(event["state"] == "RETRY_AUTHORIZED" for event in after)
    journal = job_store.JobJournal(job_store.journal_dir(item["provider_id"], run_id, root))
    if not journal.events():
        refused = next((event for event in after if event["state"] == "REFUSED"), None)
        if refused:
            return base | {"status": REFUSED, "detail": refused.get("reason")}
        if active is None:
            active = executor_active(batch_dir)
        if active:
            return base | {"status": IN_PROGRESS}
        # The journal's SUBMITTING record is written before the executor starts, so no
        # journal means the page was never handed to the provider.
        return base | {"status": INTERRUPTED_BEFORE_SEND,
                       "detail": "Interrupted before transmission (no job journal); safe to submit"}
    summary = journal.summary()
    state = summary["state"]
    first = journal.first
    base |= {"job_id": summary["job_id"], "run_directory": first.get("run_directory"),
             "journal_state": state}
    if active is None:
        active = executor_active(batch_dir)
    artifact = page_artifacts.latest_snapshot_for_run(
        root, item["provider_id"], run_id, include_incomplete=not active
    )
    if artifact is not None:
        base |= {
            "artifact_directory": artifact["path"],
            "artifact_complete": artifact["ok"],
            "reusable_saved_result": artifact["reusable"],
        }
    outcome = next((event for event in reversed(after) if event["state"] == "OUTCOME"), None)
    if outcome:
        base |= {"cost_usd": outcome.get("cost_usd"), "cost_source": outcome.get("cost_source")}
    if state == job_store.COMPLETED:
        if first.get("artifact_context") and (artifact is None or not artifact["ok"]):
            if active and artifact is None:
                return base | {
                    "status": IN_PROGRESS,
                    "detail": "Provider response completed; finalizing the atomic page artifact",
                }
            detail = "Atomic page artifact is missing" if artifact is None else (
                "Atomic page artifact is incomplete/corrupt: " + "; ".join(artifact["problems"])
            )
            return base | {"status": INCOMPLETE_ARTIFACT, "detail": detail}
        completed_dir = Path(artifact["path"]) if artifact and artifact["ok"] else Path(first["run_directory"])
        run = _read_json(completed_dir / "run.json") or {}
        usage = run.get("usage") or {}
        return base | {"status": COMPLETED, "run_directory": str(completed_dir),
                       "cost_usd": usage.get("cost_usd", base["cost_usd"]),
                       "cost_source": usage.get("cost_source", base["cost_source"])}
    if state in {job_store.FAILED, job_store.ABANDONED}:
        latest = journal.latest()
        detail = (latest.get("error") or {}).get("message") if isinstance(latest.get("error"), dict) else None
        status = RETRY_AUTHORIZED if retry else (FAILED if state == job_store.FAILED else ABANDONED)
        saved_dir = str(artifact["path"]) if artifact and artifact["ok"] else base["run_directory"]
        return base | {"status": status, "run_directory": saved_dir,
                       "detail": detail or latest.get("reason") or state}
    if state == job_store.AWAITING_RESULT_HOST_APPROVAL:
        spec_hosts = summary["awaiting_host"]
        # Sanitized when recorded (query values removed), safe to display for checking.
        result_url = journal.latest().get("result_url") or {}
        return base | {"status": NEEDS_HOST_APPROVAL, "awaiting_host": spec_hosts,
                       "result_url": result_url if isinstance(result_url, dict) else {},
                       "detail": f"Result is on host {spec_hosts!r}; approve it for this job, then resume"}
    if state == job_store.SUBMISSION_UNCONFIRMED:
        if active and journal.latest()["state"] == job_store.SUBMITTING:
            return base | {"status": IN_PROGRESS}
        error = journal.latest().get("error") or {}
        base = base | {"diagnostics": _unconfirmed_summary(error)}
        return base | {"status": NEEDS_DECISION, "detail": (
            "Submission outcome unknown (" + str(error.get("code") or "interrupted") + "): the provider may "
            "have received the page. Never retried automatically: abandon it explicitly, then authorize a retry "
            "if you want to send it again.")}
    if active:
        return base | {"status": IN_PROGRESS}
    return base | {"status": READY_TO_RESUME,
                   "detail": f"Job {summary['job_id']} is {state}; resume polls/downloads it (no resubmission)"}


def _unconfirmed_summary(error: dict) -> dict[str, Any]:
    """What is known about an ambiguous request (evidence for the user's decision)."""
    diagnostics = error.get("diagnostics") if isinstance(error.get("diagnostics"), dict) else {}
    stages = diagnostics.get("stages") or {}
    stalled = next((name for name, value in stages.items()
                    if isinstance(value, dict) and value.get("status") != "completed"), None)
    progress = diagnostics.get("stream_progress") or {}
    return {
        "code": error.get("code"),
        "message": error.get("message"),
        "stalled_stage": stalled,
        "stage_elapsed_ms": {name: value.get("elapsed_ms") for name, value in stages.items()
                             if isinstance(value, dict)},
        "upload_completed": (stages.get("sending_request") or {}).get("status") == "completed",
        "http_status": diagnostics.get("http_status"),
        "response_bytes": diagnostics.get("response_bytes"),
        "request_id_headers": diagnostics.get("request_id_headers") or {},
        "streamed_characters": progress.get("content_chars"),
    }


def batch_status(batch_dir: Path, root: Path) -> dict[str, Any]:
    plan = load_plan(batch_dir)
    active = executor_active(batch_dir)
    items = [item_status(batch_dir, item, root, active=active) for item in plan["items"]]
    counts: dict[str, int] = {}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    providers = {}
    for provider in plan["providers"]:
        mine = [item for item in items if item["provider_id"] == provider["provider_id"]]
        providers[provider["provider_id"]] = {
            "done": sum(item["status"] in DONE for item in mine),
            "total": len(mine),
            "spent_usd": round(sum(float(item["cost_usd"] or 0) for item in mine), 6),
            "committed_usd": round(_committed(plan, batch_dir, provider["provider_id"], root), 6),
            "submissions": _submissions(batch_dir, plan, provider["provider_id"], root),
        }
    total = len(items)
    done = sum(item["status"] in DONE for item in items)
    needs_user = [item for item in items if item["status"] in NEEDS_USER | {READY_TO_RESUME}]
    control = control_state(batch_dir)
    if active:
        overall = "RUNNING"
    elif done == total:
        overall = "COMPLETE"
    elif control in {"PAUSE", "STOP"}:
        overall = {"PAUSE": "PAUSED", "STOP": "STOPPED"}[control]
    elif needs_user:
        overall = "WAITING_FOR_USER"
    elif any(item["status"] != PENDING for item in items):
        overall = "INCOMPLETE"
    else:
        overall = "NOT_STARTED"
    return {"batch_id": plan["batch_id"], "stage": plan["stage"], "overall": overall, "active": active,
            "control": control, "counts": counts, "done": done, "total": total,
            "providers": providers, "items": items, "needs_user": needs_user}


def _sent_attempts(batch_dir: Path, plan: dict, provider_id: str, root: Path) -> list[dict]:
    """Attempts that reached a job journal: the page was (or may have been) sent.

    An attempt without a journal never reached the provider (the SUBMITTING record is
    written before the executor starts), so it is neither counted nor charged.
    """
    found = []
    for item in plan["items"]:
        if item["provider_id"] != provider_id or item["action"] != "submit":
            continue
        for event in _item_log(batch_dir, item["item_id"]).events():
            if event["state"] != "INTENT":
                continue
            directory = job_store.journal_dir(provider_id, event["run_id"], root)
            journal = job_store.JobJournal(directory)
            if journal.events():
                found.append({"item_id": item["item_id"], "run_id": event["run_id"], "journal": journal})
    return found


def _committed(plan: dict, batch_dir: Path, provider_id: str, root: Path) -> float:
    """Money spent or possibly spent: the provider-reported cost when the run has one,
    otherwise the per-page bound (an unfinished or failed request may still be billed)."""
    bound = next(p["max_cost_per_page_usd"] for p in plan["providers"] if p["provider_id"] == provider_id)
    total = 0.0
    for attempt in _sent_attempts(batch_dir, plan, provider_id, root):
        run_dir = attempt["journal"].first.get("run_directory")
        usage = ((_read_json(Path(run_dir) / "run.json") or {}).get("usage") or {}) if run_dir else {}
        cost = usage.get("cost_usd")
        total += float(cost) if isinstance(cost, (int, float)) and usage.get("cost_source") == "response" else bound
    return total


def _submissions(batch_dir: Path, plan: dict, provider_id: str, root: Path) -> int:
    return len(_sent_attempts(batch_dir, plan, provider_id, root))


def authorize_item_retry(batch_dir: Path, item_id: str, *, by: str, statement: str, root: Path) -> dict:
    plan = load_plan(batch_dir)
    item = next((other for other in plan["items"] if other["item_id"] == item_id), None)
    if item is None:
        raise BatchError(f"No item {item_id} in this plan")
    if statement != RETRY_STATEMENT:
        raise BatchError("The retry statement was not accepted")
    if not by.strip():
        raise BatchError("Enter your name")
    status = item_status(batch_dir, item, root)
    if status["status"] not in {FAILED, ABANDONED}:
        raise BatchError(f"Only a failed or abandoned item can be retried (this one is {status['status']})")
    return _item_log(batch_dir, item_id).append("RETRY_AUTHORIZED", by=by.strip(), after_journal=status["journal_id"])


def approve_batch_result_host(
    batch_dir: Path, spec: ProviderSpec, *, host: str, approved_by: str, statement: str, root: Path,
) -> list[Path]:
    """Approve one reported host for each paused job *of this batch* that reported it.

    Every approval is still a separate job-scoped, expiring record created by the
    existing `approve_result_host`; nothing becomes provider-wide.
    """
    plan = load_plan(batch_dir)
    value = job_store.validate_result_hostname(host)
    paths = []
    for item in plan["items"]:
        if item["provider_id"] != spec.id:
            continue
        status = item_status(batch_dir, item, root, active=False)
        if status["status"] != NEEDS_HOST_APPROVAL or status["awaiting_host"] != value:
            continue
        journal = job_store.JobJournal(job_store.journal_dir(spec.id, status["journal_id"], root))
        if value in job_store.approved_result_hosts(spec, root, journal_id=journal.journal_id):
            continue
        paths.append(job_store.approve_result_host(
            spec, host=value, approved_by=approved_by, statement=statement, journal=journal, root=root))
    if not paths:
        raise BatchError(f"No job in this batch is waiting for host {value!r}")
    return paths


def item_host_approved(spec: ProviderSpec, status: dict, root: Path) -> bool:
    """Whether this paused job's own reported host has a valid job-scoped approval."""
    if status.get("status") != NEEDS_HOST_APPROVAL or not status.get("journal_id"):
        return False
    hosts = job_store.approved_result_hosts(spec, root, journal_id=status["journal_id"])
    return status.get("awaiting_host") in hosts


def approve_item_result_host(
    batch_dir: Path, spec: ProviderSpec, item_id: str, *, host: str, job_id: str,
    approved_by: str, statement: str, root: Path,
) -> Path:
    """Approve the reported result host for exactly one paused job of this batch.

    The caller must repeat the hostname and the Job ID shown to the user; both must
    match what this job reported. The record is the existing job-scoped, expiring
    approval: it never applies to another job or to any other host.
    """
    plan = load_plan(batch_dir)
    item = next((entry for entry in plan["items"] if entry["item_id"] == item_id), None)
    if item is None or item["provider_id"] != spec.id:
        raise BatchError(f"Item {item_id!r} is not a {spec.id} item of this batch")
    status = item_status(batch_dir, item, root, active=executor_active(batch_dir))
    if status["status"] != NEEDS_HOST_APPROVAL:
        raise BatchError(f"Item {item_id} is {status['status']}, not waiting for a result-host approval")
    value = job_store.validate_result_hostname(host)
    if value != status["awaiting_host"]:
        raise BatchError(f"Type exactly the host this job reported: {status['awaiting_host']!r}")
    if str(job_id).strip() != str(status["job_id"] or ""):
        raise BatchError("The Job ID does not match the job waiting for approval")
    if item_host_approved(spec, status, root):
        raise BatchError(f"Host {value!r} is already approved for job {status['job_id']}; press Resume")
    journal = job_store.JobJournal(job_store.journal_dir(spec.id, status["journal_id"], root))
    return job_store.approve_result_host(
        spec, host=value, approved_by=approved_by, statement=statement, journal=journal, root=root)


# --------------------------------------------------------------- execution


@dataclass
class _Budget:
    budget_usd: float
    max_submissions: int
    committed_usd: float
    submissions: int
    inflight_usd: float = 0.0
    inflight: int = 0
    ambiguous_streak: int = 0
    stopped: str | None = None
    # Reasons that also forbid resuming already-submitted jobs (identity changed,
    # no credential). Submission stops (budget, ambiguity, credits) do not.
    resume_blocked: str | None = None


class BatchRunner:
    """Executes one approved stage plan. Only ever started by an explicit user action."""

    def __init__(self, runner: APIEvaluationRunner, specs: dict[str, ProviderSpec], *,
                 executor: Callable | None = None, clock: Callable[[], float] = time.monotonic):
        self.runner = runner
        self.specs = specs
        self.executor = executor
        self.clock = clock
        self.root = runner.authorization_root

    def execute(
        self, batch_dir: Path, *, credentials: dict[str, str], sources: dict[str, SourceDocument],
        ground_truth: GroundTruth | None = None, on_progress: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        batch_dir = Path(batch_dir)
        with executor_lock(batch_dir):
            plan = load_plan(batch_dir)
            # Batches of one evaluation run one at a time, so a page can never be in
            # flight in two batches (e.g. a staged continuation and its predecessor).
            others = [path for path in Path(batch_dir).parent.iterdir()
                      if path.is_dir() and path != batch_dir and (path / "plan.json").is_file()
                      and executor_active(path)]
            if others:
                raise BatchError("Another batch of this evaluation is executing (" + others[0].name
                                 + "); wait for it to finish or stop it first")
            approval = approval_state(batch_dir)
            if not approval["allowed"]:
                raise BatchError("Batch not approved: " + "; ".join(approval["reasons"]))
            if control_state(batch_dir) != "RUN":
                raise BatchError("Press Run (or Resume) first; the batch is " + control_state(batch_dir))
            for document in plan["documents"]:
                source = sources.get(document["registry_id"])
                if source is None or source.entry.sha256 != document["sha256"]:
                    raise BatchError(f"{document['registry_id']}: workspace document does not match the plan")
            limits = approval["approval"]["limits"]
            budgets: dict[str, _Budget] = {}
            for provider in plan["providers"]:
                provider_id = provider["provider_id"]
                spec = self.specs.get(provider_id)
                limit = limits.get(provider_id)
                if limit is None:
                    continue
                budget = _Budget(float(limit["budget_usd"]), int(limit["max_submissions"]),
                                 _committed(plan, batch_dir, provider_id, self.root),
                                 _submissions(batch_dir, plan, provider_id, self.root))
                if spec is None or spec.identity() != provider["identity"]:
                    budget.stopped = budget.resume_blocked = "provider registry identity changed after planning; re-plan"
                elif not (credentials.get(provider_id) or "").strip():
                    budget.stopped = budget.resume_blocked = f"no {spec.credential_label} entered for this session"
                budgets[provider_id] = budget
            _record(batch_dir, "EXECUTION_STARTED", providers={
                key: {"stopped": value.stopped, "committed_usd": value.committed_usd,
                      "submissions": value.submissions} for key, value in budgets.items()})
            deadline = self.clock() + plan["execution"]["batch_deadline_minutes"] * 60
            self._resume_phase(batch_dir, plan, budgets, credentials, ground_truth, deadline, on_progress)
            self._submit_phase(batch_dir, plan, budgets, credentials, sources, ground_truth, deadline, on_progress)
            if any(host_policy.policy_state(batch_dir, spec)["active"]
                   for provider_id, spec in self.specs.items() if provider_id in budgets):
                # Jobs that paused during this run: fetch those the policy approves
                # (polling/downloading only; nothing is resubmitted).
                self._resume_phase(batch_dir, plan, budgets, credentials, ground_truth, deadline, on_progress)
            _record(batch_dir, "EXECUTION_ENDED", providers={
                key: {"stopped": value.stopped, "committed_usd": round(value.committed_usd, 6),
                      "submissions": value.submissions} for key, value in budgets.items()})
        return batch_status(batch_dir, self.root)

    # A paused job continues from its Job ID; nothing is ever resubmitted here.
    def _resume_phase(self, batch_dir, plan, budgets, credentials, ground_truth, deadline, on_progress):
        batch_journals: set[str] | None = None
        for item in plan["items"]:
            if self._halted(batch_dir, deadline):
                return
            budget = budgets.get(item["provider_id"])
            # Resuming polls/downloads an existing Job ID and never resubmits, so a stop
            # on new submissions (budget, unknown outcomes, credits) does not block it.
            if budget is None or budget.resume_blocked:
                continue
            status = item_status(batch_dir, item, self.root, active=False)
            spec = self.specs[item["provider_id"]]
            if status["status"] == NEEDS_HOST_APPROVAL:
                hosts = job_store.approved_result_hosts(spec, self.root, journal_id=status["journal_id"])
                if status["awaiting_host"] not in hosts and host_policy.policy_state(batch_dir, spec)["active"]:
                    # Opt-in per-batch policy (OFF by default): creates the same job-scoped
                    # approval only for a provably eligible Baidu job; otherwise stays manual.
                    if batch_journals is None:
                        batch_journals = {
                            entry["journal_id"] for entry in (
                                item_status(batch_dir, other, self.root, active=False) for other in plan["items"])
                            if entry.get("journal_id")
                        }
                    journal = job_store.JobJournal(job_store.journal_dir(spec.id, status["journal_id"], self.root))
                    if host_policy.auto_approve(batch_dir, spec, journal, batch_journals=batch_journals,
                                                root=self.root):
                        hosts = job_store.approved_result_hosts(spec, self.root, journal_id=status["journal_id"])
                if status["awaiting_host"] not in hosts:
                    continue
            elif status["status"] != READY_TO_RESUME:
                continue
            log = _item_log(batch_dir, item["item_id"])
            try:
                outcome = self.runner.resume(spec, status["journal_id"], credential=credentials[spec.id],
                                             ground_truth=ground_truth, executor=self.executor)
            except APIRunRefused as exc:
                log.append("RESUME_REFUSED", run_id=status["journal_id"], reason=str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - recorded; the journal stays authoritative
                log.append("ERROR", run_id=status["journal_id"],
                           error=redact(f"{type(exc).__name__}: {exc}", [credentials[spec.id]]))
                continue
            log.append("OUTCOME", run_id=status["journal_id"], via="resume", job_state=outcome.state,
                       **_cost(outcome))
            if on_progress:
                on_progress({"item_id": item["item_id"], "state": outcome.state})

    def _halted(self, batch_dir: Path, deadline: float) -> bool:
        return control_state(batch_dir) != "RUN" or self.clock() >= deadline

    def _submit_phase(self, batch_dir, plan, budgets, credentials, sources, ground_truth, deadline, on_progress):
        queues: dict[str, queue.Queue] = {}
        for item in plan["items"]:
            if item["action"] != "submit" or item["provider_id"] not in budgets:
                continue
            if item_status(batch_dir, item, self.root, active=False)["status"] in SUBMITTABLE:
                queues.setdefault(item["provider_id"], queue.Queue()).put(item)
        gate = threading.Lock()
        slots = threading.BoundedSemaphore(plan["execution"]["max_concurrency"])

        def reserve(item: dict) -> str | None:
            with gate:
                budget = budgets[item["provider_id"]]
                if budget.stopped:
                    return budget.stopped
                if self._halted(batch_dir, deadline):
                    return "paused, stopped or past the batch deadline"
                if budget.committed_usd + budget.inflight_usd + item["max_cost_usd"] > budget.budget_usd + 1e-12:
                    budget.stopped = (f"batch budget {budget.budget_usd:.4f} USD would be exceeded "
                                      f"(committed {budget.committed_usd:.4f}, in flight {budget.inflight_usd:.4f})")
                    return budget.stopped
                if budget.submissions + budget.inflight + 1 > budget.max_submissions:
                    budget.stopped = f"approved submission limit {budget.max_submissions} reached"
                    return budget.stopped
                budget.inflight_usd += item["max_cost_usd"]
                budget.inflight += 1
                return None

        def settle(
            item: dict, *, sent: bool, cost: float | None, ambiguous: bool,
            stop_reason: str | None,
        ) -> None:
            with gate:
                budget = budgets[item["provider_id"]]
                budget.inflight_usd -= item["max_cost_usd"]
                budget.inflight -= 1
                if sent:
                    budget.submissions += 1
                    budget.committed_usd += item["max_cost_usd"] if cost is None else cost
                    if cost is not None and cost > item["max_cost_usd"] + 1e-9 and not budget.stopped:
                        # Reconciliation: the real charge broke the preflight assumption.
                        budget.stopped = (f"reported charge {cost:.6f} USD exceeded the per-page bound "
                                          f"{item['max_cost_usd']:.6f} USD; the price assumption is wrong")
                budget.ambiguous_streak = budget.ambiguous_streak + 1 if ambiguous else 0
                if budget.ambiguous_streak >= plan["execution"]["ambiguous_stop_threshold"] and not budget.stopped:
                    budget.stopped = (f"{budget.ambiguous_streak} consecutive submissions with unknown outcome; "
                                      "stopped submitting to this provider")
                if stop_reason and not budget.stopped:
                    budget.stopped = stop_reason

        def worker(provider_id: str) -> None:
            spec = self.specs[provider_id]
            work = queues[provider_id]
            while True:
                try:
                    item = work.get_nowait()
                except queue.Empty:
                    return
                with slots:
                    # Gates are checked only once a slot is held, so a Pause/Stop/deadline
                    # that arrived while this page waited for a slot is honoured.
                    reason = reserve(item)
                    if reason:
                        _item_log(batch_dir, item["item_id"]).append("SKIPPED", reason=reason)
                        continue
                    # Fail closed: if anything unexpected happens, treat the page as possibly
                    # sent (charged at its bound) for the rest of this execution.
                    sent, cost, ambiguous, stop_reason = True, None, True, None
                    try:
                        sent, cost, ambiguous, stop_reason = self._submit_one(
                            batch_dir, item, spec, credentials[provider_id], sources[item["registry_id"]],
                            plan["workspace_id"], ground_truth)
                    except Exception as exc:  # noqa: BLE001 - recorded; the thread must keep its accounting
                        _record(batch_dir, "WORKER_ERROR", item_id=item["item_id"], error=type(exc).__name__)
                    finally:
                        settle(
                            item, sent=sent, cost=cost, ambiguous=ambiguous,
                            stop_reason=stop_reason,
                        )
                if on_progress:
                    try:
                        on_progress({"item_id": item["item_id"]})
                    except Exception:  # noqa: BLE001 - a display callback never stops a batch
                        pass

        threads = [
            threading.Thread(target=worker, args=(provider_id,), name=f"batch-{provider_id}-{index}", daemon=True)
            for provider_id in queues
            for index in range(plan["execution"]["max_concurrency_per_provider"])
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def _submit_one(self, batch_dir, item, spec, credential, source, workspace_id, ground_truth):
        log = _item_log(batch_dir, item["item_id"])
        plan = load_plan(batch_dir)
        # Never repeat a completed request: if this exact page already has a valid
        # completed result (e.g. from a later staged batch), nothing is sent.
        saved, _rejected = find_saved_result(item, spec, self.root)
        if saved is not None:
            log.append("SKIPPED", reason=(
                f"Already completed in run {saved['run_id']}"
                + (f" (batch {saved['batch_id']})" if saved.get("batch_id") else "")
                + "; nothing sent. Create a new plan to reuse it."))
            return False, None, False, None
        run_id = new_run_id()
        log.append("INTENT", run_id=run_id, provider_id=spec.id, image_sha256=item["image_sha256"],
                   document_sha256=item["document_sha256"], page=item["page"])
        try:
            outcome = self.runner.run(
                spec, workspace_id, source.document, source.pdf_path, item["page"],
                credential=credential, confirmed_sha256=item["image_sha256"], ground_truth=ground_truth,
                executor=self.executor, run_id=run_id,
                artifact_context={
                    "batch_id": plan["batch_id"], "stage": plan["stage"],
                    "item_id": item["item_id"], "fingerprint": item["fingerprint"],
                },
            )
        except APIRunRefused as exc:
            log.append("REFUSED", run_id=run_id, reason=str(exc))
            return False, None, False, None
        except Exception as exc:  # noqa: BLE001 - the job journal stays authoritative
            log.append("ERROR", run_id=run_id, error=redact(f"{type(exc).__name__}: {exc}", [credential]))
            sent = bool(job_store.JobJournal(job_store.journal_dir(spec.id, run_id, self.root)).events())
            return sent, None, sent, None
        cost = _cost(outcome)
        log.append("OUTCOME", run_id=run_id, via="submit", job_state=outcome.state, **cost)
        reported = cost["cost_usd"] if cost["cost_source"] == "response" else None
        stop_reason = None
        error = outcome.run.error if outcome.run is not None else None
        if error and "payment_required" in error:
            stop_reason = "provider reported insufficient credits; stopped further submissions"
        return True, reported, outcome.state == job_store.SUBMISSION_UNCONFIRMED, stop_reason


def _cost(outcome) -> dict[str, Any]:
    usage = outcome.run.usage if outcome.run is not None else None
    return {"cost_usd": usage.cost_usd if usage else None,
            "cost_source": usage.cost_source if usage else "unknown",
            "run_status": outcome.run.status if outcome.run is not None else None,
            "run_directory": str(outcome.directory),
            "artifact_directory": str(outcome.artifact_directory) if outcome.artifact_directory else None}


# --------------------------------------------------------------- readiness


def stage_readiness(
    batch_dir: Path, *, specs: dict[str, ProviderSpec], sources: dict[str, SourceDocument],
    credentials_present: dict[str, bool], root: Path,
) -> list[dict[str, Any]]:
    """Every gate the user must satisfy before pressing Run, in plain terms."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        plan = load_plan(batch_dir)
    except BatchError as exc:
        add("Plan integrity", False, str(exc))
        return checks
    add("Plan integrity", True, f"plan_sha256 {plan['plan_sha256'][:16]}…")
    for document in plan["documents"]:
        source = sources.get(document["registry_id"])
        if source is None:
            add(f"Document {document['registry_id']}", False, "not in this workspace (by exact hash)")
            continue
        current = documents.file_sha256(source.pdf_path)
        add(f"Document bytes {document['registry_id']}", current == document["sha256"],
            f"SHA-256 {current[:16]}… (planned {document['sha256'][:16]}…)")
        state = documents.authorization_state(document["sha256"], root=root)
        add(f"Document authorization {document['registry_id']}", state["allowed"],
            "authorized until " + str(state["authorization"].get("expires_at")) if state["allowed"]
            else "; ".join(state["reasons"]))
    for provider in plan["providers"]:
        if not provider["submit_count"]:
            add(f"Provider {provider['provider_id']}", True, "nothing to submit (all reused/blocked)")
            continue
        spec = specs.get(provider["provider_id"])
        if spec is None:
            add(f"Provider {provider['provider_id']}", False, "not in the provider registry")
            continue
        add(f"Provider identity {spec.id}", spec.identity() == provider["identity"],
            "unchanged since planning" if spec.identity() == provider["identity"] else "changed; re-plan")
        # Quota is needed only for pages that would still be sent; jobs that were
        # already submitted resume from their Job ID and use no new page quota.
        need = sum(
            1 for item in plan["items"]
            if item["provider_id"] == spec.id and item["action"] == "submit"
            and item_status(batch_dir, item, root)["status"] in SUBMITTABLE
        )
        state = check_authorization(spec, pages=1 if need else 0, root=root)
        # Same policy as every other path: enablement budget, UTC day and cumulative cap.
        add(f"Provider enablement {spec.id}", state.allowed and state.remaining_pages >= need,
            ("; ".join(state.reasons) + "; " if state.reasons else "")
            + f"needs {need} pages; {state.describe()}")
        add(f"Credential {spec.id}", credentials_present.get(spec.id, False),
            f"{spec.credential_label} {'entered for this session' if credentials_present.get(spec.id) else 'missing'}")
    approval = approval_state(batch_dir)
    add("Batch approval", approval["allowed"], "; ".join(approval["reasons"]) or
        f"approved by {approval['approval'].get('approved_by')} until {approval['approval'].get('expires_at')}")
    add("Not already running", not executor_active(batch_dir), "no execution holds this batch")
    return checks


__all__ = [
    "BATCH_APPROVAL_STATEMENT",
    "BatchError",
    "BatchRunner",
    "RETRY_STATEMENT",
    "approval_state",
    "approve_batch_result_host",
    "approve_item_result_host",
    "approve_plan",
    "item_host_approved",
    "authorize_item_retry",
    "batch_status",
    "build_plan",
    "control_state",
    "item_status",
    "list_batches",
    "load_definition",
    "load_plan",
    "request_control",
    "resolve_sources",
    "save_plan",
    "stage_readiness",
    "text_layer",
]
