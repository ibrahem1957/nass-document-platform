"""Full-PDF hosted API run for the simple OCR/VLM Evaluation view (E2).

One run sends every page of ONE exact PDF to ONE hosted model, strictly one page
at a time, and saves the results in its own write-once folder:

    results/<model_folder>/<document_folder>/run_YYYYMMDD_NNN/
        metadata.json        model, provider, actual settings, prompt, document (written at start)
        plan.json            the exact page images (SHA-256 each) and request template
        approval.json        the ONE explicit approval covering every page in plan.json
        host_policy/         (Baidu only, optional) automatic result-host approval for this run
        attempts/            one intent record per submission attempt (+ refusals, local errors)
        page_NNN.txt         exact model text of a completed page (never modified)
        page_NNN.json        that page's record (status, timings, tokens, indicators, errors)
        full_text.txt        all pages in order, with a numbered separator  } written once
        report.html          summary, every page's text, errors, indicators } when the run
        results.json         everything above, structured (written last)    } is finalized

Nothing here weakens an existing gate. Every page still goes through
`APIEvaluationRunner.run`, which re-checks the document's exact-hash authorization,
the provider enablement and usage limits, the confirmed image hash, duplicate
claims, credential isolation and the network guard immediately before sending.
The per-page audit trail (job journals, `runs/`, `pages/`) is unchanged.

Safety rules specific to a full run:
- the approval names every page image by SHA-256 and expires; a changed plan,
  revoked document authorization or exceeded limit stops the run;
- pages are sent sequentially; a page whose submission is unconfirmed
  (SUBMISSION_UNCONFIRMED) is never resent automatically, and the run stops after
  `STOP_AFTER_CONSECUTIVE_PROBLEMS` consecutive failed/unconfirmed pages;
- a retry is a separate, explicit user action with its own statement;
- the API key is never written; Ground Truth is never read, sent or scored.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import platform
import re
import sys
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from lab import __version__
from lab.core.e2.api import documents, host_policy
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import check_authorization
from lab.core.e2.api.batch import BatchError, executor_active, executor_lock
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import write_sequenced
from lab.core.e2.api.runner import (
    API_IMPLEMENTATION_VERSION,
    INPUT_POLICY,
    NOT_SENT,
    APIEvaluationRunner,
    APIRunRefused,
)
from lab.core.e2.metrics_e2 import repetition_stats
from lab.core.e2.prompts import get_prompt
from lab.core.e2.runner import new_run_id
from lab.core.models import LabDocument, RunResult

FULL_PDF_VERSION = "E2-FULLPDF-1"
# The two models offered in the simple view, and the result folder for each.
MODEL_FOLDERS = {
    "oxen-qwen3-vl-4b-instruct": "qwen3_vl",
    "baidu-aistudio-paddleocr-vl-1.6": "paddleocr_vl",
}
APPROVAL_HOURS = 24
MAX_ATTEMPTS_PER_PAGE = 3
STOP_AFTER_CONSECUTIVE_PROBLEMS = 2

APPROVAL_STATEMENT = (
    "I approve sending, one page at a time, exactly the page images listed in this plan "
    "(each identified by its SHA-256) of this exact document to the provider and endpoint "
    "shown, for this run only. Each page is still checked against the document "
    "authorization, the provider enablement and the usage limits before it is sent, and "
    "nothing is resent automatically."
)
RETRY_STATEMENT = (
    "I deliberately approve sending this same page image to the same provider once more. "
    "If its earlier submission was unconfirmed, the provider may already have received "
    "(and charged for) it."
)
ABANDON_STATEMENT = (
    "I abandon the unfinished job(s) of this run listed above. Nothing is resent; their "
    "pages are recorded as not completed."
)

SUCCESS = "Success"
FAILED = "Failed"
UNCERTAIN = "Uncertain"
NOT_SENT_STATUS = "Not sent"

# Journal states that are not finished and must not be resubmitted automatically.
_PAUSED = frozenset({job_store.SUBMITTED, job_store.AWAITING_RESULT_HOST_APPROVAL,
                     job_store.RESUMABLE_ERROR, job_store.RESUMING})
_SAFE = re.compile(r"[^\w.-]+", re.UNICODE)
_RUN_NAME = re.compile(r"^run_(\d{8})_(\d{3,})$")

ProgressCallback = Callable[[str, int, int, "dict | None"], None]


class FullPDFError(PermissionError):
    pass


# ------------------------------------------------------------------ helpers


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _plan_digest(plan: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in plan.items() if k != "plan_sha256"})).hexdigest()


def _write_once_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
    return path


def _write_once_json(path: Path, payload: dict) -> Path:
    return _write_once_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def model_folder(spec: ProviderSpec) -> str:
    return MODEL_FOLDERS.get(spec.id) or (_SAFE.sub("_", spec.id).strip("._") or "model")


def document_folder_name(display_name: str) -> str:
    cleaned = _SAFE.sub("_", Path(display_name).stem).strip("._")
    return (cleaned or "document")[:60]


def page_name(page: int, suffix: str) -> str:
    return f"page_{page:03d}{suffix}"


def _document_dir(root: Path, spec: ProviderSpec, document: LabDocument) -> Path:
    """`<model>/<file stem>`; a different PDF with the same name gets a hash suffix."""
    base = Path(root) / model_folder(spec) / document_folder_name(document.display_name)
    if base.is_dir():
        for metadata in base.glob("run_*/metadata.json"):
            other = (_read_json(metadata) or {}).get("document", {}).get("sha256")
            if other and other != document.sha256:
                return base.with_name(f"{base.name}_{document.sha256[:8]}")
    return base


def _reserve_run_dir(doc_dir: Path) -> Path:
    doc_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y%m%d")
    used = [int(match.group(2)) for path in doc_dir.iterdir()
            if (match := _RUN_NAME.match(path.name)) and match.group(1) == day]
    sequence = max(used, default=0) + 1
    while True:
        path = doc_dir / f"run_{day}_{sequence:03d}"
        try:
            path.mkdir()  # exclusive: a run folder is never reused
            return path
        except FileExistsError:
            sequence += 1


# --------------------------------------------------------------------- plan


def build_plan(
    runner: APIEvaluationRunner, spec: ProviderSpec, workspace_id: str, document: LabDocument,
    pdf_path: Path, *, on_page: Callable[[int, int], None] | None = None, pages: list[int] | None = None,
) -> tuple[dict, dict[int, str]]:
    """Render every page locally and describe exactly what each request would send.

    Nothing is transmitted. Rendering goes through the runner's own preview, so the
    document must already be eligible and authorized by exact SHA-256.
    """
    if not spec.runnable:
        raise FullPDFError(f"{spec.id} is {spec.status}; no request can be sent")
    if document.kind != "pdf":
        raise FullPDFError("Only PDF documents can be run")
    pages: list[dict] = []
    images: dict[int, str] = {}
    first: dict | None = None
    selected = sorted(set(pages)) if pages else list(range(1, document.page_count + 1))
    if any(not 1 <= page <= document.page_count for page in selected):
        raise FullPDFError("A selected page is outside the document")
    for page in selected:
        preview = runner.preview(spec, workspace_id, document, pdf_path, page)
        item = preview["file"]
        pages.append({key: item[key] for key in ("filename", "bytes", "width", "height", "dpi")}
                     | {"page": page, "image_sha256": item["sha256"]})
        images[page] = preview["image_path"]
        first = first or preview
        if on_page is not None:
            on_page(page, len(selected))
    assert first is not None
    template = {key: value for key, value in first.items()
                if key not in {"file", "image_path", "document_authorization"}}
    template = json.loads(json.dumps(template, ensure_ascii=False).replace(
        first["file"]["sha256"], "<the page image SHA-256 listed in pages[]>"))
    prompt = get_prompt(spec.prompt_id) if spec.prompt_id else None
    per_page = spec.pricing.estimated_usd_per_page
    plan = {
        "schema_version": 1,
        "kind": "full_pdf_plan",
        "full_pdf_version": FULL_PDF_VERSION,
        "api_implementation_version": API_IMPLEMENTATION_VERSION,
        "provider_id": spec.id,
        "identity": spec.identity(),
        "endpoint": spec.endpoint,
        "document": {"sha256": document.sha256, "display_name": document.display_name,
                     "doc_id": document.doc_id, "page_count": document.page_count,
                     "bytes": document.size_bytes},
        "page_count": len(pages),
        "pages": pages,
        "request_template": template,
        "prompt": ({"id": prompt.prompt_id, "sha256": prompt.sha256, "text": prompt.text}
                   if prompt else None),
        "generation": dict(spec.generation),
        "optional_payload": dict(spec.optional_payload),
        "stream": bool(spec.stream),
        "outbound_hosts_allowed": sorted(spec.allowed_hosts),
        "auto_result_hosts": sorted(spec.auto_result_hosts) if host_policy.supported(spec) else [],
        "not_sent": list(NOT_SENT) + ["the API key is sent only as the Authorization header and never stored"],
        "input_policy": INPUT_POLICY,
        "sequential": True,
        "cost": {"basis": spec.pricing.basis, "price_verified": spec.pricing.price_verified,
                 "estimated_max_usd_per_page": per_page,
                 "estimated_max_usd_total": round(per_page * len(pages), 6) if per_page >= 0 else None,
                 "source": spec.pricing.source},
        "limits": spec.limits.model_dump(),
        "document_authorization": first["document_authorization"],
    }
    if pages:  # a page subset (Hybrid PDF Extraction); full-document plans are unchanged
        plan["selected_pages"] = selected
    plan["plan_sha256"] = _plan_digest(plan)
    return plan, images


def preflight(spec: ProviderSpec, plan: dict, *, root: Path, pages_to_send: int) -> dict[str, Any]:
    """Every gate that can be checked before the first page, including usage limits."""
    reasons: list[str] = []
    if plan.get("plan_sha256") != _plan_digest(plan):
        reasons.append("The plan was modified after it was prepared")
    if plan.get("provider_id") != spec.id or plan.get("endpoint") != spec.endpoint:
        reasons.append("The plan was prepared for a different provider or endpoint")
    state = check_authorization(spec, pages=1, root=root)
    reasons += state.reasons
    record = state.enablement or {}
    remaining = state.remaining_pages
    if record.get("enabled") and pages_to_send > remaining:
        reasons.append(f"Usage limit: {pages_to_send} page(s) to send but only {remaining} remain under the "
                       f"internal limits ({state.describe()})")
    document_state = documents.authorization_state((plan.get("document") or {}).get("sha256", ""), root=root)
    reasons += document_state["reasons"]
    return {"allowed": not reasons, "reasons": list(dict.fromkeys(reasons)), "remaining_pages": remaining,
            "pages_used_total": state.pages_used_total, "pages_used_today": state.pages_used_today,
            "usage": state.to_dict(),
            "enablement": record}


# ---------------------------------------------------------------- run setup


def create_run(
    root: Path, spec: ProviderSpec, document: LabDocument, plan: dict, *, approved_by: str,
    statement: str, trust_result_hosts: bool = False, render_dpi: int | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Write metadata, the plan and the single approval into a new write-once folder."""
    if statement != APPROVAL_STATEMENT:
        raise FullPDFError("The approval statement was not accepted")
    if not approved_by.strip():
        raise FullPDFError("Enter your name to approve the run")
    if (plan.get("document") or {}).get("sha256") != document.sha256:
        raise FullPDFError("The plan was prepared for a different document")
    checked = preflight(spec, plan, root=root, pages_to_send=int(plan["page_count"]))
    if not checked["allowed"]:
        raise FullPDFError("; ".join(checked["reasons"]))
    if run_dir is None:
        run_dir = _reserve_run_dir(_document_dir(root, spec, document))
    else:  # an explicit, new folder (e.g. a hybrid run's vlm/ subfolder); never reused
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=False)
    now = _now()
    _write_once_json(run_dir / "plan.json", plan)
    _write_once_json(run_dir / "approval.json", {
        "schema_version": 1, "approved": True, "approved_by": approved_by.strip(), "statement": statement,
        "plan_sha256": plan["plan_sha256"], "provider_id": spec.id, "endpoint": spec.endpoint,
        "document_sha256": document.sha256,
        "image_sha256": {str(item["page"]): item["image_sha256"] for item in plan["pages"]},
        "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=APPROVAL_HOURS)).isoformat(),
    })
    trusted = bool(trust_result_hosts and host_policy.supported(spec))
    _write_once_json(run_dir / "metadata.json", {
        "schema_version": 1,
        "kind": "full_pdf_run",
        "run_id": run_dir.name,
        "created_at": now.isoformat(),
        "folder": str(run_dir),
        "full_pdf_version": FULL_PDF_VERSION,
        "api_implementation_version": API_IMPLEMENTATION_VERSION,
        "lab_version": __version__,
        "model": {"label": spec.label, "api_model_id": spec.api_model_id,
                  "requested_model": spec.requested_model, "classification": spec.classification,
                  "reported_revision": spec.reported_revision,
                  "exact_revision_verified": spec.exact_revision_verified,
                  "local_counterpart": spec.local_counterpart},
        "provider": {"name": spec.provider, "provider_id": spec.id, "endpoint": spec.endpoint,
                     "protocol": spec.protocol, "data_residency": spec.data_residency,
                     "protocol_source": spec.protocol_source},
        "settings": {"prompt": plan.get("prompt"), "generation": dict(spec.generation),
                     "optional_payload": dict(spec.optional_payload), "stream": bool(spec.stream),
                     "render_dpi": render_dpi, "limits": spec.limits.model_dump(),
                     "pricing": spec.pricing.model_dump(), "sequential": True,
                     "max_attempts_per_page": MAX_ATTEMPTS_PER_PAGE,
                     "stop_after_consecutive_problems": STOP_AFTER_CONSECUTIVE_PROBLEMS,
                     "result_host_policy": ("automatic for listed Baidu result hosts (this run)" if trusted
                                            else "manual per job" if host_policy.supported(spec)
                                            else "not applicable (synchronous response)")},
        "document": dict(plan["document"]) | {"authorization": plan.get("document_authorization")},
        "plan_sha256": plan["plan_sha256"],
        "approval": {"approved_by": approved_by.strip(), "created_at": now.isoformat(),
                     "expires_at": (now + timedelta(hours=APPROVAL_HOURS)).isoformat()},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "ground_truth": "not used: never read, sent or scored; compare the saved text manually",
        "api_key": "never stored (kept only in the browser session's memory)",
    })
    if trusted:
        host_policy.enable_policy(run_dir, spec, enabled_by=approved_by, statement=host_policy.POLICY_STATEMENT,
                                  hours_valid=APPROVAL_HOURS)
    return run_dir


def load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    run_dir = Path(run_dir)
    metadata, plan, approval = (_read_json(run_dir / name) for name in ("metadata.json", "plan.json", "approval.json"))
    if not metadata or not plan or not approval:
        raise FullPDFError(f"Not a complete full-PDF run folder: {run_dir}")
    return metadata, plan, approval


def is_finalized(run_dir: Path) -> bool:
    return (Path(run_dir) / "results.json").is_file()


def approval_state(run_dir: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        _metadata, plan, approval = load_run(run_dir)
    except FullPDFError as exc:
        return {"valid": False, "reasons": [str(exc)], "approval": None}
    if not approval.get("approved") or approval.get("statement") != APPROVAL_STATEMENT:
        reasons.append("The run has no valid approval")
    if approval.get("plan_sha256") != plan.get("plan_sha256") or plan.get("plan_sha256") != _plan_digest(plan):
        reasons.append("The plan does not match its approval")
    try:
        if datetime.fromisoformat(approval["expires_at"]) <= _now():
            reasons.append("The run approval has expired; start a new run")
    except (KeyError, ValueError):
        reasons.append("The run approval has no valid expiry")
    if is_finalized(run_dir):
        reasons.append("The run is finalized (write-once); start a new run to send more pages")
    return {"valid": not reasons, "reasons": reasons, "approval": approval}


# ----------------------------------------------------------------- attempts


def _attempts_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "attempts"


def attempts(run_dir: Path, page: int) -> list[dict]:
    found = []
    for path in sorted(_attempts_dir(run_dir).glob(f"page_{page:03d}_attempt_[0-9][0-9].json")):
        record = _read_json(path) or {}
        for kind in ("refused", "local_error"):
            extra = _read_json(path.with_name(f"{path.stem}_{kind}.json"))
            if extra:
                record[kind] = extra
        found.append(record | {"_file": path.name})
    return found


def run_journal_ids(run_dir: Path) -> set[str]:
    return {str((_read_json(path) or {}).get("journal_id"))
            for path in _attempts_dir(run_dir).glob("page_*_attempt_[0-9][0-9].json")}


def _load_store_run(directory: str | None) -> RunResult | None:
    if not directory:
        return None
    path = Path(directory) / "run.json"
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _seconds_between(start: str | None, end: str | None) -> float | None:
    try:
        return round((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds(), 3)
    except (TypeError, ValueError):
        return None


def derive_page(run_dir: Path, spec: ProviderSpec, plan_page: dict, *, root: Path) -> dict[str, Any]:
    """The page's current record, from its write-once files and its job journal.

    A committed page (page_NNN.json) is returned as saved. Otherwise the record is
    derived from the latest attempt's journal, which is the source of truth.
    """
    page = int(plan_page["page"])
    committed = _read_json(Path(run_dir) / page_name(page, ".json"))
    if committed:
        return committed | {"committed": True}
    tried = attempts(run_dir, page)
    base: dict[str, Any] = {
        "page": page, "image_sha256": plan_page["image_sha256"], "attempts": len(tried),
        "status": NOT_SENT_STATUS, "status_reason": "not sent yet", "job_state": None,
        "journal_id": None, "job_id": None, "text": "", "committed": False,
    }
    if not tried:
        return base
    last = tried[-1]
    base["journal_id"] = last.get("journal_id")
    if last.get("refused"):
        return base | {"job_state": "REFUSED_BEFORE_SENDING",
                       "status_reason": f"refused before sending: {last['refused'].get('reason')}"}
    journal = job_store.JobJournal(job_store.journal_dir(spec.id, str(last.get("journal_id")), root))
    events = journal.events()
    if not events:
        reason = (f"local error before sending: {last['local_error'].get('message')}" if last.get("local_error")
                  else "interrupted before sending (no job journal exists, so nothing was transmitted)")
        return base | {"job_state": "NOT_TRANSMITTED", "status_reason": reason}
    state = journal.state
    first, latest = events[0], events[-1]
    run = _load_store_run(first.get("run_directory"))
    text = run.pages[0].text if run is not None and run.pages else ""
    response = ((((run.environment.get("e2") or {}).get("page_raw") or {}).get(str(page)) or {}).get("response")
                if run is not None else None) or {}
    usage = run.usage if run is not None else None
    prompt_tokens = usage.prompt_tokens if usage else None
    completion_tokens = usage.completion_tokens if usage else None
    truncated = response.get("truncated_at_max_new_tokens")
    repetition = repetition_stats(text)
    error = latest.get("error") if isinstance(latest.get("error"), dict) else None
    if run is not None and run.error and not error:
        error = {"code": "run_error", "message": run.error}
    record = base | {
        "job_state": state,
        "job_id": journal.job_id,
        "text": text,
        "chars": len(text),
        "started_at": first.get("started_at") or first.get("at"),
        "finished_at": latest.get("at"),
        "processing_seconds": _seconds_between(first.get("started_at") or first.get("at"), latest.get("at")),
        "request_wall_ms": run.wall_ms if run is not None else None,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens,
                   "total": (prompt_tokens + completion_tokens)
                   if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) else None,
                   "reported": prompt_tokens is not None or completion_tokens is not None},
        "finish_reason": response.get("finish_reason"),
        "truncated_at_token_limit": truncated if isinstance(truncated, bool) else None,
        "truncation_status": response.get("truncation_status") or "unknown",
        "reported_model": response.get("reported_model"),
        "cost": {"usd": usage.cost_usd if usage else None, "source": usage.cost_source if usage else "unknown"},
        "indicators": {"empty_output": not text.strip(), "truncated_at_token_limit": truncated is True,
                       "repetition": repetition},
        "error": error,
        "model_settings": {"provider_id": spec.id, "api_model_id": spec.api_model_id,
                           "prompt_id": spec.prompt_id,
                           "prompt_sha256": get_prompt(spec.prompt_id).sha256 if spec.prompt_id else None,
                           "generation": dict(spec.generation), "optional_payload": dict(spec.optional_payload)},
        "store_run_directory": first.get("run_directory"),
        "store_run_loaded": run is not None,
    }
    if state == job_store.COMPLETED and run is None:
        record["status"] = UNCERTAIN
        record["status_reason"] = ("the job completed but its run record (run.json) is missing or unreadable; "
                                   "nothing is committed and nothing is resent")
    elif state == job_store.COMPLETED:
        problems = [label for flag, label in (
            (not text.strip(), "the model returned no text"),
            (truncated is True, "output stopped at the token limit (may be incomplete)"),
            (repetition["flagged"], "repetition loop suspected (may be incomplete)"),
        ) if flag]
        record["status"] = UNCERTAIN if problems else SUCCESS
        record["status_reason"] = "; ".join(problems) or "completed"
    elif state in {job_store.FAILED, job_store.ABANDONED}:
        record["status"] = FAILED
        record["status_reason"] = ("abandoned by the user" if state == job_store.ABANDONED else "the provider job failed") + (
            f": {error.get('code')}: {error.get('message')}" if error else "")
    elif state == job_store.SUBMISSION_UNCONFIRMED:
        record["status"] = UNCERTAIN
        record["status_reason"] = ("submission unconfirmed (the provider may have received the page); it is never "
                                   "resent automatically") + (f": {error.get('code')}: {error.get('message')}" if error else "")
    elif state == job_store.AWAITING_RESULT_HOST_APPROVAL:
        host = (latest.get("result_url") or {}).get("host")
        record["status"] = UNCERTAIN
        record["status_reason"] = f"job finished; its result host {host!r} needs approval before download"
        record["awaiting_host"] = host
    elif state in _PAUSED:
        record["status"] = UNCERTAIN
        record["status_reason"] = f"job paused ({state}); resume it from its Job ID (no resubmission)" + (
            f": {error.get('code')}: {error.get('message')}" if error else "")
    else:  # SUBMITTING with a Job ID, or anything unexpected
        record["status"] = UNCERTAIN
        record["status_reason"] = f"job state {state}"
    return record


def _commit_page(run_dir: Path, record: dict) -> dict:
    """Write page_NNN.txt (exact text) then page_NNN.json (the commit marker), once."""
    page = int(record["page"])
    if (record.get("committed") or record.get("job_state") != job_store.COMPLETED
            or not record.get("store_run_loaded")):
        return record
    text_path = Path(run_dir) / page_name(page, ".txt")
    data = record["text"].encode("utf-8")
    if text_path.exists():
        if text_path.read_bytes() != data:
            raise FullPDFError(f"{text_path.name} exists with different content; nothing was overwritten")
    else:
        _write_once_bytes(text_path, data)
    saved = {key: value for key, value in record.items() if key not in {"text", "committed"}}
    saved |= {"text_file": text_path.name, "text_sha256": hashlib.sha256(data).hexdigest(),
              "committed_at": _now().isoformat()}
    _write_once_json(Path(run_dir) / page_name(page, ".json"), saved)
    return saved | {"text": record["text"], "committed": True}


def page_text(run_dir: Path, record: dict) -> str:
    if "text" in record:
        return str(record["text"])
    path = Path(run_dir) / str(record.get("text_file") or "")
    try:
        return path.read_bytes().decode("utf-8") if record.get("text_file") else ""
    except OSError:
        return ""


# ---------------------------------------------------------------- execution


@contextmanager
def _exclusive(run_dir: Path):
    """One execution per run folder across sessions and processes (OS file lock)."""
    manager = executor_lock(run_dir)
    try:
        manager.__enter__()
    except BatchError as exc:
        raise FullPDFError("This run is already executing in another session") from exc
    try:
        yield
    finally:
        manager.__exit__(None, None, None)


def running(run_dir: Path) -> bool:
    return executor_active(run_dir)


def _approve_known_host_for_run(run_dir, spec, journal, root) -> None:
    """Baidu: the run's single approval covers downloads from the registry's exact
    `auto_result_hosts` (or paddleocr-store-<n>.bj.bcebos.com), for HTTPS result URLs
    reported by a job journal of THIS run. Any other host stays blocked (manual)."""
    if not host_policy.supported(spec) or journal.journal_id not in run_journal_ids(run_dir):
        return
    _metadata, plan, approval = load_run(run_dir)
    if (approval.get("statement") != APPROVAL_STATEMENT or approval.get("plan_sha256") != plan.get("plan_sha256")
            or plan.get("plan_sha256") != _plan_digest(plan)):
        return
    first, latest = journal.first, journal.latest()
    if (first.get("provider_id") != spec.id or first.get("endpoint") != spec.endpoint
            or first.get("api_model_id") != spec.api_model_id or not journal.job_id):
        return
    url = latest.get("result_url") if isinstance(latest.get("result_url"), dict) else {}
    try:
        host = job_store.validate_result_hostname(str(url.get("host") or ""))
    except job_store.JobJournalError:
        return
    if (host != url.get("host") or url.get("scheme") != "https" or url.get("port") not in (None, 443)
            or not str(url.get("path") or "").startswith(host_policy.RESULT_PATH_PREFIX)):
        return
    listed = {item.lower() for item in spec.auto_result_hosts}
    if host not in listed and not re.fullmatch(host_policy.BAIDU_RESULT_HOST_PATTERN, host):
        return
    if host in job_store.approved_result_hosts(spec, root, journal_id=journal.journal_id):
        return
    via = {"mode": "full_pdf_run_approval", "run": Path(run_dir).name, "approval_file": str(Path(run_dir) / "approval.json"),
           "approved_by": approval.get("approved_by"), "basis": "listed Baidu result host" if host in listed
           else "matches paddleocr-store-<n>.bj.bcebos.com"}
    path = job_store.approve_result_host(
        spec, host=host, approved_by=f"full-PDF run approval ({approval.get('approved_by')})",
        statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal, root=root, via=via)
    # Separate, append-only record in the run folder; nothing existing is changed.
    write_sequenced(Path(run_dir) / "host_approvals", "host_approval", {
        "schema_version": 1, "page": first.get("page"), "journal_id": journal.journal_id, "job_id": journal.job_id,
        "host": host, "basis": via["basis"], "approval_record": str(path), "created_at": _now().isoformat()})


def _auto_resume(run_dir, runner, spec, journal_id, credential, root) -> None:
    """Baidu: approve a known result host under this run's approval, then download
    from the saved Job ID (never resubmits)."""
    journal = job_store.JobJournal(job_store.journal_dir(spec.id, journal_id, root))
    if not journal.events() or journal.state != job_store.AWAITING_RESULT_HOST_APPROVAL:
        return
    _approve_known_host_for_run(run_dir, spec, journal, root)
    host = (journal.latest().get("result_url") or {}).get("host")
    if host and host in runner.result_hosts(spec, journal_id):
        runner.resume(spec, journal_id, credential=credential, ground_truth=None)


def _send(run_dir, runner, spec, plan_page, *, workspace_id, document, pdf_path, credential, root,
          kind: str, by: str | None = None) -> tuple[dict, str | None]:
    """One submission attempt. Returns (record, refusal reason or None)."""
    page = int(plan_page["page"])
    number = len(attempts(run_dir, page)) + 1
    if number > MAX_ATTEMPTS_PER_PAGE:
        raise FullPDFError(f"Page {page} already has {MAX_ATTEMPTS_PER_PAGE} attempts")
    journal_id = new_run_id()
    intent = _attempts_dir(run_dir) / f"page_{page:03d}_attempt_{number:02d}.json"
    _write_once_json(intent, {"page": page, "attempt": number, "kind": kind, "by": by,
                              "journal_id": journal_id, "image_sha256": plan_page["image_sha256"],
                              "provider_id": spec.id, "created_at": _now().isoformat()})
    refusal = None
    try:
        outcome = runner.run(
            spec, workspace_id, document, pdf_path, page, credential=credential,
            confirmed_sha256=plan_page["image_sha256"], ground_truth=None, run_id=journal_id,
            artifact_context={"full_pdf_run": Path(run_dir).name, "full_pdf_folder": str(run_dir),
                              "attempt": number},
        )
        if outcome.state == job_store.AWAITING_RESULT_HOST_APPROVAL:
            try:
                _auto_resume(run_dir, runner, spec, journal_id, credential, root)
            except APIRunRefused:
                pass  # stays paused; shown to the user with its reason
    except APIRunRefused as exc:
        refusal = str(exc)
        _write_once_json(intent.with_name(f"{intent.stem}_refused.json"),
                         {"reason": refusal, "created_at": _now().isoformat()})
    except Exception as exc:  # noqa: BLE001 - recorded; the journal decides what was sent
        _write_once_json(intent.with_name(f"{intent.stem}_local_error.json"),
                         {"type": type(exc).__name__, "message": str(exc)[:2000],
                          "created_at": _now().isoformat()})
    record = _commit_page(run_dir, derive_page(run_dir, spec, plan_page, root=root))
    return record, refusal


def _sendable(record: dict) -> bool:
    return record["job_state"] in {None, "REFUSED_BEFORE_SENDING", "NOT_TRANSMITTED"}


def run_pending(
    run_dir: Path, runner: APIEvaluationRunner, spec: ProviderSpec, *, workspace_id: str,
    document: LabDocument, pdf_path: Path, credential: str, root: Path,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Send every page that was never transmitted, in page order, one at a time."""
    run_dir = Path(run_dir)
    _metadata, plan, _approval = load_run(run_dir)
    if plan.get("provider_id") != spec.id or (plan.get("document") or {}).get("sha256") != document.sha256:
        raise FullPDFError("This run belongs to a different provider or document")
    if not credential or not credential.strip():
        raise FullPDFError(f"Enter the {spec.credential_label}")
    stopped: str | None = None
    with _exclusive(run_dir):
        state = approval_state(run_dir)
        if not state["valid"]:
            raise FullPDFError("; ".join(state["reasons"]))
        total = len(plan["pages"])
        pending = [item for item in plan["pages"] if _sendable(derive_page(run_dir, spec, item, root=root))]
        checked = preflight(spec, plan, root=root, pages_to_send=len(pending))
        if pending and not checked["allowed"]:
            raise FullPDFError("; ".join(checked["reasons"]))
        problems = 0
        for item in plan["pages"]:
            record = derive_page(run_dir, spec, item, root=root)
            if not _sendable(record):
                if on_progress is not None:
                    on_progress("skipped", int(item["page"]), total, record)
                continue
            state = approval_state(run_dir)
            if not state["valid"]:
                stopped = "; ".join(state["reasons"])
                break
            if on_progress is not None:
                on_progress("sending", int(item["page"]), total, record)
            record, refusal = _send(run_dir, runner, spec, item, workspace_id=workspace_id, document=document,
                                    pdf_path=pdf_path, credential=credential, root=root, kind="run")
            if on_progress is not None:
                on_progress("done", int(item["page"]), total, record)
            if refusal is not None:
                stopped = f"Page {item['page']} was refused before sending: {refusal}. Later pages were not sent."
                break
            if record["job_state"] in {job_store.FAILED, job_store.SUBMISSION_UNCONFIRMED, job_store.RESUMABLE_ERROR,
                                       "NOT_TRANSMITTED"}:
                problems += 1
                if problems >= STOP_AFTER_CONSECUTIVE_PROBLEMS:
                    stopped = (f"Stopped after {problems} consecutive failed or unconfirmed pages (last: page "
                               f"{item['page']}). Check the provider, then continue the run; nothing is resent "
                               "automatically.")
                    break
            else:
                problems = 0
        summary = run_summary(run_dir, spec, root=root)
        if not stopped and summary["all_completed"] and not is_finalized(run_dir):
            _finalize_unlocked(run_dir, spec, root=root)
            summary = run_summary(run_dir, spec, root=root)
    return summary | {"stopped": stopped}


def resume_paused(
    run_dir: Path, runner: APIEvaluationRunner, spec: ProviderSpec, *, credential: str, root: Path,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Continue paused jobs (Baidu) from their Job IDs. Never resubmits a page."""
    run_dir = Path(run_dir)
    _metadata, plan, _approval = load_run(run_dir)
    if is_finalized(run_dir):
        raise FullPDFError("The run is finalized")
    messages: list[str] = []
    with _exclusive(run_dir):
        total = len(plan["pages"])
        for item in plan["pages"]:
            record = derive_page(run_dir, spec, item, root=root)
            if record["job_state"] not in _PAUSED or spec.protocol == "openai_chat_completions_v1":
                continue
            if on_progress is not None:
                on_progress("resuming", int(item["page"]), total, record)
            try:
                if record["job_state"] == job_store.AWAITING_RESULT_HOST_APPROVAL:
                    _auto_resume(run_dir, runner, spec, record["journal_id"], credential, root)
                    host = record.get("awaiting_host")
                    if host not in runner.result_hosts(spec, record["journal_id"]):
                        messages.append(f"Page {item['page']}: approve result host {host!r} first")
                        continue
                    if derive_page(run_dir, spec, item, root=root)["job_state"] == job_store.AWAITING_RESULT_HOST_APPROVAL:
                        runner.resume(spec, record["journal_id"], credential=credential, ground_truth=None)
                else:
                    runner.resume(spec, record["journal_id"], credential=credential, ground_truth=None)
            except APIRunRefused as exc:
                messages.append(f"Page {item['page']}: {exc}")
            record = _commit_page(run_dir, derive_page(run_dir, spec, item, root=root))
            if on_progress is not None:
                on_progress("done", int(item["page"]), total, record)
        summary = run_summary(run_dir, spec, root=root)
        if summary["all_completed"] and not is_finalized(run_dir):
            _finalize_unlocked(run_dir, spec, root=root)
            summary = run_summary(run_dir, spec, root=root)
    return summary | {"messages": messages}


def approve_page_host(run_dir: Path, spec: ProviderSpec, page: int, *, host: str, approved_by: str,
                      statement: str, root: Path) -> Path:
    """Manual, job-scoped approval of the exact result host a paused page reported."""
    _metadata, plan, _approval = load_run(run_dir)
    item = next((entry for entry in plan["pages"] if int(entry["page"]) == int(page)), None)
    if item is None:
        raise FullPDFError(f"Page {page} is not in this run")
    record = derive_page(run_dir, spec, item, root=root)
    if record["job_state"] != job_store.AWAITING_RESULT_HOST_APPROVAL:
        raise FullPDFError(f"Page {page} is not waiting for a result-host approval")
    journal = job_store.JobJournal(job_store.journal_dir(spec.id, record["journal_id"], root))
    try:
        return job_store.approve_result_host(spec, host=host, approved_by=approved_by, statement=statement,
                                             journal=journal, root=root)
    except job_store.JobJournalError as exc:
        raise FullPDFError(str(exc)) from exc


def retry_page(
    run_dir: Path, runner: APIEvaluationRunner, spec: ProviderSpec, page: int, *, workspace_id: str,
    document: LabDocument, pdf_path: Path, credential: str, root: Path, by: str, statement: str,
) -> dict[str, Any]:
    """Explicitly send one failed or unconfirmed page again (abandoning its open job first)."""
    if statement != RETRY_STATEMENT:
        raise FullPDFError("The retry statement was not accepted")
    if not by.strip():
        raise FullPDFError("Enter your name to retry a page")
    if not credential or not credential.strip():
        raise FullPDFError(f"Enter the {spec.credential_label}")
    run_dir = Path(run_dir)
    _metadata, plan, _approval = load_run(run_dir)
    item = next((entry for entry in plan["pages"] if int(entry["page"]) == int(page)), None)
    if item is None:
        raise FullPDFError(f"Page {page} is not in this run")
    with _exclusive(run_dir):
        state = approval_state(run_dir)
        if not state["valid"]:
            raise FullPDFError("; ".join(state["reasons"]))
        record = derive_page(run_dir, spec, item, root=root)
        job_state = record["job_state"]
        chat = spec.protocol == "openai_chat_completions_v1"
        if job_state == job_store.COMPLETED:
            raise FullPDFError(f"Page {page} is completed; it is never resent")
        if job_state in _PAUSED and not (chat and job_state == job_store.RESUMABLE_ERROR):
            raise FullPDFError(f"Page {page} has a paused job; resume it instead (no resubmission)")
        checked = preflight(spec, plan, root=root, pages_to_send=1)
        if not checked["allowed"]:
            raise FullPDFError("; ".join(checked["reasons"]))
        if job_state in {job_store.SUBMISSION_UNCONFIRMED, job_store.RESUMABLE_ERROR, job_store.SUBMITTING}:
            runner.abandon(spec, record["journal_id"], by=by,
                           reason=f"Explicit retry of page {page} from full-PDF run {run_dir.name}")
        record, refusal = _send(run_dir, runner, spec, item, workspace_id=workspace_id, document=document,
                                pdf_path=pdf_path, credential=credential, root=root, kind="retry", by=by.strip())
    return record | {"refusal": refusal}


# -------------------------------------------------------------- summaries


def run_summary(run_dir: Path, spec: ProviderSpec, *, root: Path) -> dict[str, Any]:
    metadata, plan, _approval = load_run(run_dir)
    records = [derive_page(run_dir, spec, item, root=root) for item in plan["pages"]]
    counts = {name: sum(1 for item in records if item["status"] == name)
              for name in (SUCCESS, UNCERTAIN, FAILED, NOT_SENT_STATUS)}
    open_jobs = [item for item in records if item["job_state"] in _PAUSED | {job_store.SUBMISSION_UNCONFIRMED,
                                                                              job_store.SUBMITTING}]
    return {
        "run_id": Path(run_dir).name, "folder": str(run_dir), "metadata": metadata, "records": records,
        "counts": counts, "total": len(records), "open_jobs": open_jobs,
        "all_completed": all(item["job_state"] == job_store.COMPLETED and item.get("store_run_loaded", True)
                             for item in records),
        "finalized": is_finalized(run_dir), "approval": approval_state(run_dir),
    }


def _results_payload(run_dir: Path, summary: dict, *, finalized: bool) -> dict:
    metadata = summary["metadata"]
    pages = []
    for record in summary["records"]:
        pages.append({key: value for key, value in record.items() if key != "committed"}
                     | {"text": page_text(run_dir, record)})
    seconds = [item.get("processing_seconds") for item in pages if isinstance(item.get("processing_seconds"), (int, float))]
    tokens = [item.get("tokens") or {} for item in pages]
    return {
        "schema_version": 1, "kind": "full_pdf_results", "run_id": summary["run_id"], "finalized": finalized,
        "generated_at": _now().isoformat(), "document": metadata.get("document"), "model": metadata.get("model"),
        "provider": metadata.get("provider"), "settings": metadata.get("settings"),
        "plan_sha256": metadata.get("plan_sha256"), "approval": metadata.get("approval"),
        "counts": summary["counts"], "page_count": summary["total"],
        "totals": {"processing_seconds": round(sum(seconds), 3) if seconds else None,
                   "prompt_tokens": sum(t["prompt"] for t in tokens if isinstance(t.get("prompt"), int)) or None,
                   "completion_tokens": sum(t["completion"] for t in tokens if isinstance(t.get("completion"), int)) or None},
        "quality_note": ("No accuracy is computed from model output. Indicators (empty output, token-limit stop, "
                         "repetition) flag pages to review; compare the text with Ground Truth manually."),
        "pages": pages,
    }


def _full_text(run_dir: Path, summary: dict) -> str:
    total = summary["total"]
    parts = []
    for record in summary["records"]:
        header = f"===== Page {record['page']} / {total} | {record['status']} ====="
        text = page_text(run_dir, record)
        if record["job_state"] != job_store.COMPLETED or not record.get("store_run_loaded", True):
            text = f"[no text: {record['status_reason']}]"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts) + "\n"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


_BADGE = {SUCCESS: "#1a7f37", UNCERTAIN: "#9a6700", FAILED: "#cf222e", NOT_SENT_STATUS: "#57606a"}


def _report_html(run_dir: Path, summary: dict, *, finalized: bool) -> str:
    metadata = summary["metadata"]
    document = metadata.get("document") or {}
    model = metadata.get("model") or {}
    provider = metadata.get("provider") or {}
    settings = metadata.get("settings") or {}
    counts = summary["counts"]
    rows = []
    sections = []
    for record in summary["records"]:
        tokens = record.get("tokens") or {}
        repetition = (record.get("indicators") or {}).get("repetition") or {}
        color = _BADGE.get(record["status"], "#57606a")
        rows.append(
            f"<tr><td>{record['page']}</td><td><span class='b' style='background:{color}'>{_esc(record['status'])}</span></td>"
            f"<td>{_esc(record.get('processing_seconds'))}</td><td>{_esc(tokens.get('prompt'))} / {_esc(tokens.get('completion'))}</td>"
            f"<td>{_esc(record.get('finish_reason'))}</td><td>{_esc(record.get('truncated_at_token_limit'))}</td>"
            f"<td>{'yes' if repetition.get('flagged') else 'no'}</td><td>{_esc(record.get('chars'))}</td>"
            f"<td>{_esc(record['status_reason'])}</td></tr>")
        error = record.get("error") or {}
        body = page_text(run_dir, record) if record["job_state"] == job_store.COMPLETED else ""
        sections.append(
            f"<section><h3>Page {record['page']} <span class='b' style='background:{color}'>{_esc(record['status'])}</span></h3>"
            f"<p class='m'>{_esc(record['status_reason'])} | job state {_esc(record.get('job_state'))} | attempts "
            f"{_esc(record.get('attempts'))} | time {_esc(record.get('processing_seconds'))} s | tokens "
            f"{_esc(tokens.get('prompt'))} in / {_esc(tokens.get('completion'))} out | finish reason "
            f"{_esc(record.get('finish_reason'))} | token-limit stop {_esc(record.get('truncated_at_token_limit'))} | "
            f"repetition {'flagged' if repetition.get('flagged') else 'not flagged'} | image SHA-256 "
            f"{_esc(record.get('image_sha256'))}</p>"
            + (f"<p class='e'>Error: {_esc(error.get('code'))}: {_esc(error.get('message'))}</p>" if error else "")
            + (f"<pre dir='auto'>{_esc(body)}</pre>" if body else "<p class='m'>No text saved for this page.</p>")
            + "</section>")
    prompt = settings.get("prompt") or {}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(summary['run_id'])} - {_esc(document.get('display_name'))}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",Tahoma,sans-serif;margin:24px;color:#1f2328;background:#fff}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f6f8fa}}.b{{color:#fff;border-radius:10px;padding:1px 8px;font-size:12px}}
pre{{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;padding:12px;font-size:14px;line-height:1.6}}
.m{{color:#57606a;font-size:12px}}.e{{color:#cf222e}}section{{margin-top:24px}}
</style></head><body>
<h1>Full-PDF OCR/VLM run {_esc(summary['run_id'])}</h1>
<p>{'Final report.' if finalized else '<strong>Provisional report: the run is not finalized yet.</strong>'}
Raw model output is shown unchanged. No accuracy is computed automatically; compare with Ground Truth manually.</p>
<table>
<tr><th>Document</th><td>{_esc(document.get('display_name'))} ({_esc(document.get('page_count'))} pages)<br><span class='m'>SHA-256 {_esc(document.get('sha256'))}</span></td></tr>
<tr><th>Model</th><td>{_esc(model.get('api_model_id'))} | {_esc(model.get('classification'))} | exact revision verified: {_esc(model.get('exact_revision_verified'))}</td></tr>
<tr><th>Provider</th><td>{_esc(provider.get('name'))} | {_esc(provider.get('endpoint'))}</td></tr>
<tr><th>Prompt</th><td>{_esc(prompt.get('id') or 'none (provider document-parsing endpoint)')} <span class='m'>{_esc(prompt.get('sha256'))}</span></td></tr>
<tr><th>Generation</th><td>{_esc(json.dumps(settings.get('generation') or {}, ensure_ascii=False))} | render DPI {_esc(settings.get('render_dpi'))}</td></tr>
<tr><th>Created</th><td>{_esc(metadata.get('created_at'))} | approved by {_esc((metadata.get('approval') or {}).get('approved_by'))}</td></tr>
<tr><th>Pages</th><td>Success {counts[SUCCESS]} | Uncertain {counts[UNCERTAIN]} | Failed {counts[FAILED]} | Not sent {counts[NOT_SENT_STATUS]}</td></tr>
</table>
<h2>Pages</h2>
<table><tr><th>Page</th><th>Status</th><th>Time (s)</th><th>Tokens in / out</th><th>Finish reason</th><th>Token-limit stop</th><th>Repetition</th><th>Chars</th><th>Reason</th></tr>
{''.join(rows)}</table>
{''.join(sections)}
</body></html>
"""


def outputs(run_dir: Path, spec: ProviderSpec, *, root: Path) -> dict[str, bytes]:
    """results.json, full_text.txt and report.html: the saved files when finalized,
    otherwise generated in memory (provisional, nothing written)."""
    run_dir = Path(run_dir)
    if is_finalized(run_dir):
        return {name: (run_dir / name).read_bytes() for name in ("results.json", "full_text.txt", "report.html")
                if (run_dir / name).is_file()}
    summary = run_summary(run_dir, spec, root=root)
    return {
        "results.json": json.dumps(_results_payload(run_dir, summary, finalized=False), ensure_ascii=False,
                                   indent=2).encode("utf-8"),
        "full_text.txt": _full_text(run_dir, summary).encode("utf-8"),
        "report.html": _report_html(run_dir, summary, finalized=False).encode("utf-8"),
    }


def finalize(run_dir: Path, spec: ProviderSpec, *, root: Path, runner: APIEvaluationRunner | None = None,
             abandon_by: str | None = None, statement: str | None = None) -> Path:
    """Write the final page records, full_text.txt, report.html and results.json once.

    Unfinished jobs must be abandoned explicitly (name + statement); nothing is resent.
    """
    with _exclusive(run_dir):
        return _finalize_unlocked(run_dir, spec, root=root, runner=runner, abandon_by=abandon_by,
                                  statement=statement)


def _finalize_unlocked(run_dir: Path, spec: ProviderSpec, *, root: Path, runner: APIEvaluationRunner | None = None,
                       abandon_by: str | None = None, statement: str | None = None) -> Path:
    run_dir = Path(run_dir)
    if is_finalized(run_dir):
        raise FullPDFError("The run is already finalized")
    summary = run_summary(run_dir, spec, root=root)
    if summary["open_jobs"]:
        if runner is None or not (abandon_by or "").strip() or statement != ABANDON_STATEMENT:
            raise FullPDFError(f"{len(summary['open_jobs'])} page(s) still have unfinished jobs; abandon them "
                               "explicitly (name and statement) or resolve them first")
        for record in summary["open_jobs"]:
            runner.abandon(spec, record["journal_id"], by=abandon_by,
                           reason=f"Finalizing full-PDF run {run_dir.name}")
        summary = run_summary(run_dir, spec, root=root)
    for record in summary["records"]:
        if record.get("committed"):
            continue
        if record["job_state"] == job_store.COMPLETED and record.get("store_run_loaded"):
            _commit_page(run_dir, record)
            continue
        saved = {key: value for key, value in record.items() if key not in {"text", "committed"}}
        _write_once_json(run_dir / page_name(int(record["page"]), ".json"),
                         saved | {"text_file": None, "committed_at": _now().isoformat()})
    summary = run_summary(run_dir, spec, root=root)
    _write_once_bytes(run_dir / "full_text.txt", _full_text(run_dir, summary).encode("utf-8"))
    _write_once_bytes(run_dir / "report.html", _report_html(run_dir, summary, finalized=True).encode("utf-8"))
    payload = _results_payload(run_dir, summary, finalized=True)
    payload["files_sha256"] = {
        str(path.relative_to(run_dir)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != ".executor.lock"
    }
    _write_once_json(run_dir / "results.json", payload)  # written last: its presence = finalized
    return run_dir


def zip_bytes(run_dir: Path, spec: ProviderSpec, *, root: Path) -> bytes:
    """The whole run folder as a ZIP (provisional outputs added in memory if not finalized)."""
    run_dir = Path(run_dir)
    buffer = io.BytesIO()
    generated = outputs(run_dir, spec, root=root)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != ".executor.lock":
                archive.write(path, f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
        if not is_finalized(run_dir):
            for name, data in generated.items():
                archive.writestr(f"{run_dir.name}/{name}", data)
    return buffer.getvalue()


def list_runs(root: Path) -> list[dict[str, Any]]:
    """Full-PDF runs under the model folders, newest first (read-only)."""
    found = []
    for folder in sorted(set(MODEL_FOLDERS.values())):
        base = Path(root) / folder
        if not base.is_dir():
            continue
        for metadata_path in base.glob("*/run_*/metadata.json"):
            metadata = _read_json(metadata_path)
            if not metadata or metadata.get("kind") != "full_pdf_run":
                continue
            run_dir = metadata_path.parent
            found.append({"run_dir": run_dir, "run_id": run_dir.name, "model_folder": folder,
                          "document_folder": run_dir.parent.name,
                          "provider_id": (metadata.get("provider") or {}).get("provider_id"),
                          "document": metadata.get("document") or {}, "created_at": metadata.get("created_at") or "",
                          "finalized": is_finalized(run_dir)})
    found.sort(key=lambda item: item["created_at"], reverse=True)
    return found


__all__ = [
    "ABANDON_STATEMENT", "APPROVAL_STATEMENT", "FAILED", "FullPDFError", "MODEL_FOLDERS", "NOT_SENT_STATUS",
    "RETRY_STATEMENT", "SUCCESS", "UNCERTAIN", "approval_state", "approve_page_host", "build_plan", "create_run",
    "derive_page", "finalize", "is_finalized", "list_runs", "load_run", "outputs", "preflight", "resume_paused",
    "retry_page", "run_pending", "run_summary", "running", "zip_bytes",
]
