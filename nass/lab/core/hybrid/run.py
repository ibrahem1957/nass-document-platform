"""Hybrid PDF Extraction runs: write-once folders, one approval, per-page combination.

    results/hybrid_pdf_extraction/run_YYYYMMDD_HHMMSS_xxxxxx/
        metadata.json  plan.json  approval.json (model runs)  library_raw.json  tables.json
        ground_truth.txt + ground_truth.json (optional; evaluation only)
        vlm/            hosted API: a full-PDF sub-run restricted to the selected pages
                        (full_pdf layout; job journals unchanged); Qari: plan, images, attempts
        page_NNN.txt  page_NNN.json  full_text.txt  vlm_raw.json  report.html
        evaluation.json (with ground truth)  results.json (written last = finalized)

Raw outputs are kept unchanged; `library_rejoined` and the final hybrid text are derived
results recorded separately with their source and reason. Ground Truth is read only by
`evaluate`, after the final text exists; it never influences a decision.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import secrets
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from lab import __version__
from lab.core.e2.api import documents, full_pdf
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.store import results_root
from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import layout as hybrid_layout
from lab.core.hybrid import qari, textcheck
from lab.local_pdf_extraction import diagnostics

HYBRID_VERSION = "1"
# Decision logic versions. "1": the original combiner; "2": category-B rules based on the
# relative single-letter improvement and genuine number conflicts. Runs created before
# version 2 keep version 1 until the user explicitly approves a recomputation.
DECISION_LOGIC_VERSION = "2"
REJOIN_MIN_RELATIVE_REDUCTION = 0.25
LAYOUT_STATEMENT = (
    "I reviewed the assembled page documents and their reading order. Save them as a new layout revision. "
    "It is derived from saved data only (no model or network call) and changes no existing file."
)
GT_SCOPES = {
    "text_layer": "text layer only (library text and tables; images and charts excluded)",
    "combined_page": "full combined page (text layer + images/charts)",
    "images": "images/charts only",
}  # single-letter ratio must drop by at least 25% (relative)
RECOMPUTE_STATEMENT = (
    "I reviewed the previous and proposed decisions. Apply the proposed decision revision to this open run. "
    "Only derived outputs change; raw library and model outputs, journals, authorizations and usage records "
    "are not touched, and nothing is sent anywhere."
)
FOLDER = "hybrid_pdf_extraction"
LIBRARY_ONLY = "library_only"
QARI = "qari"
OXEN = "oxen-qwen3-vl-4b-instruct"
BAIDU = "baidu-aistudio-paddleocr-vl-1.6"
MODEL_LABELS = {
    LIBRARY_ONLY: "Library Only (no model)",
    QARI: "Qari-OCR 0.4.0 (local GPU server / RunPod)",
    BAIDU: "PaddleOCR-VL 1.6 (Baidu AI Studio)",
    OXEN: "Qwen3-VL-4B-Instruct (Oxen.ai)",
}
APPROVAL_STATEMENT = (
    "I approve this hybrid run plan: only the listed page images (each identified by its SHA-256) are sent, "
    "one at a time, to the model and destination shown. Pages not listed, the library text, the PDF text layer "
    "and Ground Truth are never sent. Nothing is resent automatically."
)
APPROVAL_HOURS = 24

ACCEPTED, NEEDS_REVIEW, UNCERTAIN, FAILED = "Accepted", "Needs Review", "Uncertain", "Failed"
REJOINED_SOURCE = "library_rejoined (spacing guided by VLM)"


class HybridError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_once(path: Path, payload: dict | str | bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    elif isinstance(payload, str):
        data = payload.encode("utf-8", errors="surrogatepass")
    else:
        data = payload
    with path.open("xb") as handle:
        handle.write(data)
    return path


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def hybrid_root(root: Path | None = None) -> Path:
    return Path(root or results_root()) / FOLDER


def _new_run_dir(root: Path | None) -> Path:
    base = hybrid_root(root)
    base.mkdir(parents=True, exist_ok=True)
    while True:
        path = base / f"run_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


# ------------------------------------------------------------------- create


def create_run(
    *, root: Path | None, document, pdf_path: Path, analysis: dict, model: str, vlm_pages: dict[int, str],
    approved_by: str = "", statement: str | None = None, ground_truth: tuple[str, list[str]] | None = None,
    spec: ProviderSpec | None = None, provider_plan: dict | None = None,
    qari_endpoint: str | None = None, qari_pages: list[dict] | None = None, qari_model: str = qari.QARI_MODEL,
    ground_truth_scope: str = "text_layer",
) -> Path:
    """Write the run folder. Nothing is transmitted here."""
    if ground_truth_scope not in GT_SCOPES:
        raise HybridError(f"Unknown ground-truth scope {ground_truth_scope!r}")
    if model not in MODEL_LABELS:
        raise HybridError(f"Unknown model {model!r}")
    if analysis.get("page_count") != document.page_count:
        raise HybridError("The analysis does not match this PDF; analyze it again")
    if documents.file_sha256(pdf_path) != document.sha256:
        raise HybridError("The PDF bytes changed after upload")
    pages = sorted(vlm_pages) if model != LIBRARY_ONLY else []
    if model != LIBRARY_ONLY:
        if not pages:
            raise HybridError("No page is selected for the model; use Library Only instead")
        if statement != APPROVAL_STATEMENT or not approved_by.strip():
            raise HybridError("Enter your name and tick the run approval")
    if model in {OXEN, BAIDU}:
        if spec is None or spec.id != model or provider_plan is None:
            raise HybridError("Prepare the page images for the selected provider first")
        if sorted(provider_plan.get("selected_pages") or []) != pages:
            raise HybridError("The prepared images do not match the selected pages; prepare them again")
        checked = full_pdf.preflight(spec, provider_plan, root=Path(root or results_root()), pages_to_send=len(pages))
        if not checked["allowed"]:
            raise HybridError("; ".join(checked["reasons"]))
    if model == QARI:
        endpoint = qari.parse_endpoint(qari_endpoint or "")
        if sorted(item["page"] for item in qari_pages or []) != pages:
            raise HybridError("The prepared Qari images do not match the selected pages; prepare them again")
        if not endpoint["loopback"] and not documents.authorization_state(
                document.sha256, root=Path(root or results_root()))["allowed"]:
            raise HybridError("Sending to a remote server requires this document's SHA-256 authorization")

    run_dir = _new_run_dir(root)
    now = _now()
    library = {"version": hybrid_analysis.ANALYSIS_VERSION, "settings": analysis["settings"],
               "methods": analysis["methods"],
               "pages": [{key: value for key, value in page.items() if key != "tables"} for page in analysis["pages"]]}
    _write_once(run_dir / "library_raw.json", library)
    # Exact copy of the PDF (same SHA-256): layout revisions are recomputed from it offline.
    _write_once(run_dir / "source.pdf", Path(pdf_path).read_bytes())
    _write_once(run_dir / "tables.json", {"pages": {str(page["page"]): page["tables"] for page in analysis["pages"]},
                                          "method": analysis["methods"]["tables"]})
    plan = {
        "schema_version": 1, "kind": "hybrid_plan", "hybrid_version": HYBRID_VERSION, "model": model,
        "model_label": MODEL_LABELS[model],
        "document": {"sha256": document.sha256, "display_name": document.display_name,
                     "page_count": document.page_count, "bytes": document.size_bytes},
        "pages": [{"page": page["page"], "category": page["category"], "category_reason": page["category_reason"],
                   "send_to_model": page["page"] in pages, "send_reason": vlm_pages.get(page["page"]),
                   "image_warning": page["images"]["warning"]} for page in analysis["pages"]],
        "vlm_pages": pages,
    }
    _write_once(run_dir / "plan.json", plan)
    if model != LIBRARY_ONLY:
        _write_once(run_dir / "approval.json", {
            "approved": True, "approved_by": approved_by.strip(), "statement": statement, "model": model,
            "pages": pages, "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
            "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=APPROVAL_HOURS)).isoformat(),
        })
    if ground_truth is not None and ground_truth[0].strip():
        _write_once(run_dir / "ground_truth.txt", ground_truth[0])
        _write_once(run_dir / "ground_truth.json", {
            "sources": list(ground_truth[1]), "sha256": hashlib.sha256(ground_truth[0].encode("utf-8")).hexdigest(),
            "description": diagnostics.describe_ground_truth(ground_truth[0], document.page_count),
            "scope": ground_truth_scope, "scope_label": GT_SCOPES[ground_truth_scope],
            "use": "evaluation after processing only; never used for decisions or sent anywhere"})
    model_settings: dict[str, Any] = {}
    if model in {OXEN, BAIDU}:
        model_settings = {"identity": spec.identity(), "generation": dict(spec.generation),
                          "prompt": provider_plan.get("prompt"), "limits": spec.limits.model_dump()}
        full_pdf.create_run(Path(root or results_root()), spec, document, provider_plan, approved_by=approved_by,
                            statement=full_pdf.APPROVAL_STATEMENT, render_dpi=provider_plan["pages"][0]["dpi"],
                            run_dir=run_dir / "vlm")
    elif model == QARI:
        model_settings = qari.settings() | {"endpoint": qari.parse_endpoint(qari_endpoint)["display"],
                                            "served_model": qari_model}
        qari.create_plan(run_dir / "vlm", endpoint=qari.parse_endpoint(qari_endpoint), model=qari_model,
                         pages=qari_pages, document_sha256=document.sha256, approved_by=approved_by)
    _write_once(run_dir / "metadata.json", {
        "schema_version": 1, "kind": "hybrid_run", "run_id": run_dir.name, "created_at": now.isoformat(),
        "hybrid_version": HYBRID_VERSION, "lab_version": __version__, "folder": str(run_dir),
        "decision_logic_version": DECISION_LOGIC_VERSION,
        "document": plan["document"], "model": model, "model_label": MODEL_LABELS[model],
        "model_settings": model_settings, "library_settings": analysis["settings"],
        "ground_truth_attached": (run_dir / "ground_truth.txt").exists(),
        "ground_truth_files": list(ground_truth[1]) if ground_truth and ground_truth[0].strip() else [],
        "api_keys": "never stored (session memory only)",
    })
    return run_dir


def load(run_dir: Path) -> tuple[dict, dict, dict]:
    metadata, plan, library = (_read(Path(run_dir) / name) for name in ("metadata.json", "plan.json",
                                                                         "library_raw.json"))
    if not metadata or not plan or not library:
        raise HybridError(f"Not a complete hybrid run folder: {run_dir}")
    return metadata, plan, library


def is_finalized(run_dir: Path) -> bool:
    return (Path(run_dir) / "results.json").is_file()


def approval_valid(run_dir: Path) -> tuple[bool, str]:
    approval = _read(Path(run_dir) / "approval.json")
    if not approval or approval.get("statement") != APPROVAL_STATEMENT:
        return False, "the run has no valid approval"
    try:
        if datetime.fromisoformat(approval["expires_at"]) <= _now():
            return False, "the run approval has expired"
    except (KeyError, ValueError):
        return False, "the approval has no valid expiry"
    if is_finalized(run_dir):
        return False, "the run is finalized"
    return True, ""


# ------------------------------------------------------------------ execute

Progress = Callable[[str, int, int, "dict | None"], None]


def run_qari(run_dir: Path, *, token: str | None, root: Path | None, on_progress: Progress | None = None) -> dict:
    """Send every Qari page that was never transmitted, in order; never resends."""
    ok, reason = approval_valid(run_dir)
    if not ok:
        raise HybridError(reason)
    metadata, _plan, _library = load(run_dir)
    vlm_dir = Path(run_dir) / "vlm"
    plan = qari.load_plan(vlm_dir)
    total = len(plan["pages"])
    problems = 0
    stopped = None
    for index, item in enumerate(plan["pages"], start=1):
        record = qari.page_record(vlm_dir, item["page"])
        if record["job_state"] not in {None, qari.NOT_TRANSMITTED}:
            continue
        ok, reason = approval_valid(run_dir)
        if not ok:
            stopped = reason
            break
        authorized = documents.authorization_state(metadata["document"]["sha256"],
                                                   root=Path(root or results_root()))["allowed"]
        if on_progress:
            on_progress("sending", index, total, record)
        try:
            record = qari.send_page(vlm_dir, item["page"], token=token, document_authorized=authorized)
        except qari.QariError as exc:
            stopped = f"Page {item['page']} refused before sending: {exc}"
            break
        if on_progress:
            on_progress("done", index, total, record)
        if record["job_state"] in {qari.UNCONFIRMED, qari.FAILED, qari.NOT_TRANSMITTED}:
            problems += 1
            if problems >= full_pdf.STOP_AFTER_CONSECUTIVE_PROBLEMS:
                stopped = f"Stopped after {problems} consecutive failed or unconfirmed pages; nothing is resent."
                break
        else:
            problems = 0
    return {"stopped": stopped}


def retry_qari_page(run_dir: Path, page: int, *, token: str | None, root: Path | None, by: str,
                    statement: str) -> dict:
    if statement != full_pdf.RETRY_STATEMENT or not by.strip():
        raise HybridError("Enter your name and tick the retry statement")
    ok, reason = approval_valid(run_dir)
    if not ok:
        raise HybridError(reason)
    metadata, _plan, _library = load(run_dir)
    vlm_dir = Path(run_dir) / "vlm"
    record = qari.page_record(vlm_dir, page)
    if record["job_state"] == qari.SENT:
        raise HybridError(f"Page {page} completed; it is never resent")
    if record["attempts"] >= full_pdf.MAX_ATTEMPTS_PER_PAGE:
        raise HybridError(f"Page {page} already has {full_pdf.MAX_ATTEMPTS_PER_PAGE} attempts")
    authorized = documents.authorization_state(metadata["document"]["sha256"],
                                               root=Path(root or results_root()))["allowed"]
    return qari.send_page(vlm_dir, page, token=token, document_authorized=authorized, kind="retry", by=by.strip())


def vlm_records(run_dir: Path, metadata: dict, spec: ProviderSpec | None, root: Path | None) -> dict[int, dict]:
    """Per selected page: the model record (text unchanged) in one normalized shape."""
    vlm_dir = Path(run_dir) / "vlm"
    model = metadata["model"]
    found: dict[int, dict] = {}
    if model == QARI:
        for item in qari.load_plan(vlm_dir)["pages"]:
            found[item["page"]] = qari.page_record(vlm_dir, item["page"]) | {"image_sha256": item["image_sha256"]}
    elif model in {OXEN, BAIDU}:
        if spec is None:
            raise HybridError(f"Provider {model} is not available in providers_e2.toml")
        _meta, plan, _approval = full_pdf.load_run(vlm_dir)
        for item in plan["pages"]:
            record = full_pdf.derive_page(vlm_dir, spec, item, root=Path(root or results_root()))
            text = full_pdf.page_text(vlm_dir, record)
            found[item["page"]] = {key: value for key, value in record.items() if key != "committed"} | {"text": text}
    return found


def vlm_pending(records: dict[int, dict]) -> dict[str, list[int]]:
    unsent = [p for p, r in records.items() if r.get("job_state") in {None, "REFUSED_BEFORE_SENDING", "NOT_TRANSMITTED"}]
    open_ = [p for p, r in records.items() if r.get("job_state") in {
        "SUBMISSION_UNCONFIRMED", "SUBMITTED", "AWAITING_RESULT_HOST_APPROVAL", "RESUMABLE_ERROR", "RESUMING",
        "SUBMITTING"}]
    return {"unsent": sorted(unsent), "open": sorted(open_)}


# ------------------------------------------------------------------ combine


def _vlm_usable(record: dict | None) -> bool:
    return bool(record) and record.get("job_state") == "COMPLETED" and bool((record.get("text") or "").strip())


def combine_page(page: dict, record: dict | None, rule: dict | None,
                 logic_version: str = DECISION_LOGIC_VERSION) -> dict:
    """Final text, source, status and reasons for one page, by decision-logic version."""
    combiner = {"1": _combine_page_v1, "2": _combine_page_v2}.get(str(logic_version))
    if combiner is None:
        raise HybridError(f"Unknown decision logic version {logic_version!r}")
    return combiner(page, record, rule) | {"decision_logic_version": str(logic_version)}


def _combine_page_v1(page: dict, record: dict | None, rule: dict | None) -> dict:
    """Decision logic version 1 (kept unchanged for runs that have not been recomputed)."""
    library_text, library_source, library_reason = hybrid_analysis.library_choice(page)
    base_is_processed = library_source.startswith("pymupdf_processed")
    category = page["category"]
    reasons: list[str] = [page["category_reason"]]
    warnings: list[str] = [page["images"]["warning"]] if page["images"]["warning"] else []
    result: dict[str, Any] = {"page": page["page"], "category": category, "library_source": library_source,
                              "library_source_reason": library_reason,
                              "rejoined": None, "number_audit": None, "vlm_quality": None}
    vlm_text = (record or {}).get("text") or ""
    if record is not None:
        result["vlm_quality"] = textcheck.vlm_quality(vlm_text, finish_reason=record.get("finish_reason"),
                                                      truncated=record.get("truncated_at_token_limit"))
    if _vlm_usable(record):
        reliable = category != hybrid_analysis.CATEGORY_C
        result["number_audit"] = textcheck.number_audit(library_text, vlm_text, library_reliable=reliable)

    def final(text: str, source: str, status: str, why: str) -> dict:
        audit = result["number_audit"]
        if status == ACCEPTED and audit and audit["needs_review"]:
            status, why = NEEDS_REVIEW, why + "; numbers in the VLM output have no match in the library text"
        return result | {"final_text": text, "final_source": source, "status": status,
                         "reasons": reasons + [why], "warnings": warnings}

    if category == hybrid_analysis.CATEGORY_A:
        if record is not None:
            warnings.append("VLM output for this page is saved separately for review; it is not merged")
        return final(library_text, library_source, ACCEPTED, "library text accepted (quality checks passed)")

    if record is None:  # library only, or the page was not selected for the model
        if category == hybrid_analysis.CATEGORY_B:
            return final(library_text, library_source, NEEDS_REVIEW,
                         "words split by spaces; processed text used (logical order and numbers repaired), "
                         "no model output" if base_is_processed
                         else f"words split by spaces; raw text used ({library_reason}), no model output")
        has_text = bool(library_text.strip())
        return final(library_text, library_source, UNCERTAIN if has_text else FAILED,
                     "library text unreliable; no model output for this page" if has_text
                     else "no usable text layer and no model output")

    quality = result["vlm_quality"]
    if category == hybrid_analysis.CATEGORY_B and _vlm_usable(record) and not base_is_processed:
        reasons.append(f"library_rejoined not built: it is built only from the processed text ({library_reason})")
    if category == hybrid_analysis.CATEGORY_B and _vlm_usable(record) and base_is_processed:
        candidate = textcheck.rejoin(library_text, vlm_text)
        candidate["built_from"] = library_source
        candidate["built_from_reason"] = library_reason
        indicators = diagnostics.page_indicators(candidate["text"], rule)
        candidate["indicators"] = {key: indicators[key] for key in (
            "single_letter_arabic_ratio", "single_letter_arabic_tokens", "arabic_tokens", "needs_ocr",
            "needs_ocr_reason")}
        problems = []
        if not candidate["valid"]:
            problems.append("non-space characters or lines differ from the library text")
        if candidate["join_count"] == 0:
            problems.append("no join was supported by the VLM text")
        if indicators["needs_ocr"]:
            problems.append(f"still unreadable after rejoining ({indicators['needs_ocr_reason']})")
        candidate["accepted"] = not problems
        candidate["rejection_reasons"] = problems
        result["rejoined"] = candidate
        if not problems:
            return final(candidate["text"], REJOINED_SOURCE, ACCEPTED,
                         f"{candidate['join_count']} Arabic word join(s) confirmed by the VLM text; "
                         "characters unchanged")
        reasons.append("library_rejoined rejected: " + "; ".join(problems))

    if _vlm_usable(record):
        status = UNCERTAIN if quality["flagged"] or record.get("status") == "Uncertain" else NEEDS_REVIEW
        why = ("VLM output used as the primary candidate (not verified against a text layer)"
               + (f"; quality flags: {', '.join(quality['reasons'])}" if quality["flagged"] else ""))
        return final(vlm_text, f"vlm_raw ({record.get('status')})", status, why)
    # The model produced nothing usable: keep what exists, never invent text.
    state = record.get("job_state")
    has_text = bool(library_text.strip())
    return final(library_text, library_source, UNCERTAIN if has_text else FAILED,
                 f"no usable model output (job state {state}: {record.get('status_reason')}); "
                 + ("library text kept for review" if has_text else "no text available"))


def _combine_page_v2(page: dict, record: dict | None, rule: dict | None) -> dict:
    """Decision logic version 2. No text is invented; Ground Truth is never read here.

    Category B: `library_rejoined` (built only from the processed text) becomes the
    primary candidate when its non-space sequence equals the processed text, it has at
    least one VLM-confirmed join, and its single-letter ratio drops by at least
    REJOIN_MIN_RELATIVE_REDUCTION (relative). It is Accepted only below the Needs OCR
    threshold, otherwise Needs Review. If it fails and the VLM output has genuine number
    conflicts (page without raster images), the processed text stays primary.
    """
    library_text, library_source, library_reason = hybrid_analysis.library_choice(page)
    base_is_processed = library_source.startswith("pymupdf_processed")
    category = page["category"]
    has_images = bool(page["images"]["warning"])
    reasons: list[str] = [page["category_reason"]]
    warnings: list[str] = [page["images"]["warning"]] if has_images else []
    result: dict[str, Any] = {"page": page["page"], "category": category, "library_source": library_source,
                              "library_source_reason": library_reason,
                              "rejoined": None, "number_audit": None, "vlm_quality": None,
                              "alternatives": {}}
    vlm_text = (record or {}).get("text") or ""
    if record is not None:
        result["vlm_quality"] = textcheck.vlm_quality(vlm_text, finish_reason=record.get("finish_reason"),
                                                      truncated=record.get("truncated_at_token_limit"))
    genuine_conflicts = False
    if _vlm_usable(record):
        audit = textcheck.number_audit(library_text, vlm_text,
                                       library_reliable=category != hybrid_analysis.CATEGORY_C)
        audit["conflict_kind"] = ("may come from images" if has_images else "genuine") if audit["needs_review"] else None
        genuine_conflicts = bool(audit["needs_review"] and not has_images)
        audit["genuine_conflicts"] = genuine_conflicts
        result["number_audit"] = audit
        result["alternatives"]["vlm_raw"] = {"source": f"vlm_raw ({record.get('status')})",
                                             "note": "saved unchanged in vlm_raw.json"}

    def final(text: str, source: str, status: str, why: str) -> dict:
        audit = result["number_audit"]
        if status == ACCEPTED and audit and audit["needs_review"]:
            status = NEEDS_REVIEW
            why += ("; VLM numbers absent from the text layer (may come from images)" if has_images
                    else "; VLM numbers conflict with the text layer")
        return result | {"final_text": text, "final_source": source, "status": status,
                         "reasons": reasons + [why], "warnings": warnings}

    if category == hybrid_analysis.CATEGORY_A:
        if record is not None:
            warnings.append("VLM output for this page is saved separately for review; it is not merged")
        return final(library_text, library_source, ACCEPTED, "library text accepted (quality checks passed)")

    if record is None:
        if category == hybrid_analysis.CATEGORY_B:
            return final(library_text, library_source, NEEDS_REVIEW,
                         "words split by spaces; processed text used (logical order and numbers repaired), "
                         "no model output" if base_is_processed
                         else f"words split by spaces; raw text used ({library_reason}), no model output")
        has_text = bool(library_text.strip())
        return final(library_text, library_source, UNCERTAIN if has_text else FAILED,
                     "library text unreliable; no model output for this page" if has_text
                     else "no usable text layer and no model output")

    quality = result["vlm_quality"]
    if category == hybrid_analysis.CATEGORY_B and _vlm_usable(record):
        if not base_is_processed:
            reasons.append(f"library_rejoined not built: it is built only from the processed text ({library_reason})")
        else:
            candidate = textcheck.rejoin(library_text, vlm_text)
            before = diagnostics.single_letter_stats(library_text)["single_letter_arabic_ratio"]
            indicators = diagnostics.page_indicators(candidate["text"], rule)
            after = indicators["single_letter_arabic_ratio"]
            improvement = (before - after) / before if before else 0.0
            validation = {
                "non_space_sequence_identical_to_processed": candidate["non_space_sequence_identical"],
                "line_count_identical": candidate["line_count_identical"],
                "joins_confirmed_by_vlm": candidate["join_count"],
                "relative_single_letter_reduction": round(improvement, 4),
                "required_relative_reduction": REJOIN_MIN_RELATIVE_REDUCTION,
            }
            problems = []
            if not candidate["valid"]:
                problems.append("non-space characters or lines differ from the processed text")
            if candidate["join_count"] == 0:
                problems.append("no join was supported by the VLM text")
            if improvement < REJOIN_MIN_RELATIVE_REDUCTION:
                problems.append(f"single-letter ratio improved by {improvement:.0%} "
                                f"(< {REJOIN_MIN_RELATIVE_REDUCTION:.0%} required)")
            candidate.update(
                built_from=library_source, built_from_reason=library_reason,
                single_letter_ratio_before=before, single_letter_ratio_after=after,
                relative_improvement=round(improvement, 4), validation=validation,
                indicators={key: indicators[key] for key in ("single_letter_arabic_ratio", "single_letter_arabic_tokens",
                                                             "arabic_tokens", "needs_ocr", "needs_ocr_reason")},
                accepted=not problems, rejection_reasons=problems,
                reading_order_note="identical to pymupdf_processed by construction (only spaces removed)",
            )
            result["rejoined"] = candidate
            if not problems:
                still = bool(indicators["needs_ocr"])
                why = (f"{candidate['join_count']} VLM-confirmed join(s); characters identical to the processed "
                       f"text; single-letter ratio {before:.1%} -> {after:.1%} ({improvement:.0%} lower)"
                       + ("; still above the Needs OCR threshold" if still else ""))
                candidate["decision_reason"] = why
                return final(candidate["text"], REJOINED_SOURCE, NEEDS_REVIEW if still else ACCEPTED, why)
            candidate["decision_reason"] = "rejected: " + "; ".join(problems)
            reasons.append("library_rejoined rejected: " + "; ".join(problems))
        if genuine_conflicts and library_text.strip():
            return final(library_text, library_source, NEEDS_REVIEW, "VLM numbers conflict with the text layer")

    if _vlm_usable(record):
        status = UNCERTAIN if quality["flagged"] or record.get("status") == "Uncertain" else NEEDS_REVIEW
        why = ("VLM output used as the primary candidate (not verified against a text layer)"
               + (f"; quality flags: {', '.join(quality['reasons'])}" if quality["flagged"] else "")
               + ("; number conflicts may come from images" if has_images and result["number_audit"]["needs_review"]
                  else ""))
        return result | {"final_text": vlm_text, "final_source": f"vlm_raw ({record.get('status')})",
                         "status": status, "reasons": reasons + [why], "warnings": warnings}
    state = record.get("job_state")
    has_text = bool(library_text.strip())
    return final(library_text, library_source, UNCERTAIN if has_text else FAILED,
                 f"no usable model output (job state {state}: {record.get('status_reason')}); "
                 + ("library text kept for review" if has_text else "no text available"))


# ------------------------------------------------------- decision revisions


def _decisions_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "decisions"


def decision_revisions(run_dir: Path) -> list[dict]:
    return [payload | {"path": str(path)} for path, payload in ordered(_decisions_dir(run_dir), "revision") if payload]


def effective_logic_version(run_dir: Path, metadata: dict | None = None) -> str:
    """The latest approved revision's logic version; else the version the run was created with."""
    revisions = decision_revisions(run_dir)
    if revisions:
        return str(revisions[-1]["logic_version"])
    metadata = metadata or (_read(Path(run_dir) / "metadata.json") or {})
    return str(metadata.get("decision_logic_version", "1"))


def _decision_rows(state: dict) -> list[dict]:
    return [{"page": item["page"], "final_source": item["final_source"], "status": item["status"],
             "reason": item["reasons"][-1] if item["reasons"] else ""} for item in state["pages"]]


def preview_recompute(run_dir: Path, spec: ProviderSpec | None, root: Path | None,
                      logic_version: str = DECISION_LOGIC_VERSION) -> dict:
    """Previous vs proposed decisions, from saved data only (no extraction, model or network)."""
    if is_finalized(run_dir):
        raise HybridError("The run is finalized; its decisions cannot be recomputed")
    previous = summary(run_dir, spec, root)
    proposed = summary(run_dir, spec, root, logic_version=logic_version)
    rows = []
    for old, new in zip(_decision_rows(previous), _decision_rows(proposed)):
        rows.append({"page": old["page"], "previous": old, "proposed": new,
                     "changed": (old["final_source"], old["status"]) != (new["final_source"], new["status"])})
    return {"previous_logic_version": previous["decision_logic_version"], "proposed_logic_version": str(logic_version),
            "pages": rows, "changed_pages": [row["page"] for row in rows if row["changed"]],
            "previous_counts": previous["counts"], "proposed_counts": proposed["counts"]}


def apply_recompute(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, by: str, statement: str,
                    logic_version: str = DECISION_LOGIC_VERSION) -> Path:
    """Append an approved decision revision (write-once). Raw outputs are never touched."""
    if statement != RECOMPUTE_STATEMENT or not by.strip():
        raise HybridError("Enter your name and tick the confirmation to apply the recomputation")
    preview = preview_recompute(run_dir, spec, root, logic_version)
    revisions = decision_revisions(run_dir)
    return write_sequenced(_decisions_dir(run_dir), "revision", {
        "schema_version": 1, "kind": "hybrid_decision_revision", "logic_version": str(logic_version),
        "previous_logic_version": preview["previous_logic_version"],
        "previous_revision": Path(revisions[-1]["path"]).name if revisions else None,
        "approved_by": by.strip(), "statement": statement, "created_at": _now().isoformat(),
        "inputs": "saved library_raw.json, tables.json and saved model outputs only; no extraction, model or network",
        "previous_decisions": [row["previous"] for row in preview["pages"]],
        "new_decisions": [row["proposed"] for row in preview["pages"]],
        "changed_pages": preview["changed_pages"],
        "previous_counts": preview["previous_counts"], "new_counts": preview["proposed_counts"],
    })


# --------------------------------------------------------------- evaluation


def evaluate(ground_truth: str, texts: dict[str, dict[int, str]], page_count: int,
             scope: str = "text_layer") -> dict:
    """CER/WER per page and in total for each text source (after processing only).

    The scope states what the reference covers. Text-layer scores compare library text
    and tables only and are never overall extraction accuracy. Image/chart and combined
    page scores need image transcriptions (Phase 2); they are not computed in Phase 1.
    """
    parsed = diagnostics.parse_ground_truth(ground_truth)
    out: dict[str, Any] = {"normalization": {"none": "exact characters, including whitespace",
                                             "arabic_normalized": diagnostics.ARABIC_NORMALIZATION},
                           "level": "page" if isinstance(parsed, dict) else "document", "sources": {},
                           "scope": scope, "scope_label": GT_SCOPES.get(scope, scope)}
    if scope != "text_layer":
        out["not_computed"] = ("images and charts are not transcribed in Phase 1, so no "
                               f"{GT_SCOPES.get(scope, scope)} score can be computed")
        return out
    out["statement"] = ("Text-layer scope: library text and tables only; images and charts are excluded. "
                        "This is not an overall extraction accuracy.")
    for source, pages in texts.items():
        if isinstance(parsed, dict):
            per_page = {}
            totals = {mode: {"char_edits": 0.0, "ref_chars": 0, "word_edits": 0.0, "ref_words": 0}
                      for mode in ("none", "arabic_normalized")}
            for number, reference in sorted(parsed.items()):
                if not 1 <= number <= page_count or number not in pages:
                    continue
                rates = diagnostics.error_rates(reference, pages[number])
                per_page[str(number)] = rates
                for mode in totals:
                    item = rates[mode]
                    if item["cer"] is not None:
                        totals[mode]["char_edits"] += item["cer"] * item["reference_chars"]
                        totals[mode]["ref_chars"] += item["reference_chars"]
                    if item["wer"] is not None:
                        totals[mode]["word_edits"] += item["wer"] * item["reference_words"]
                        totals[mode]["ref_words"] += item["reference_words"]
            total = {mode: {"cer": (t["char_edits"] / t["ref_chars"]) if t["ref_chars"] else None,
                            "wer": (t["word_edits"] / t["ref_words"]) if t["ref_words"] else None,
                            "reference_chars": t["ref_chars"], "reference_words": t["ref_words"]}
                     for mode, t in totals.items()}
            out["sources"][source] = {"pages": per_page, "total": total, "pages_compared": len(per_page)}
        else:
            if len(pages) < page_count:
                out["sources"][source] = {"pages": {}, "total": None,
                                          "note": "document-level reference: this source does not cover every page"}
                continue
            full = "\n".join(pages[number] for number in sorted(pages))
            out["sources"][source] = {"pages": {}, "total": diagnostics.error_rates(ground_truth, full)}
    return out


# ----------------------------------------------------------------- finalize


def summary(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *,
            logic_version: str | None = None) -> dict:
    """Derived per-page decisions. Without `logic_version`, the run's latest approved
    decision revision is used (this is what Download ZIP and Finalize use)."""
    metadata, plan, library = load(run_dir)
    version = str(logic_version or effective_logic_version(run_dir, metadata))
    records = vlm_records(run_dir, metadata, spec, root) if metadata["model"] != LIBRARY_ONLY else {}
    rule = diagnostics.needs_ocr_settings(library["settings"])
    pages = []
    for page in library["pages"]:
        page_rule = None if (page.get("reorder") or {}).get("decision") == "mainly_ltr" else rule
        pages.append(combine_page(page, records.get(page["page"]), page_rule, version))
    counts = {name: sum(1 for item in pages if item["status"] == name)
              for name in (ACCEPTED, NEEDS_REVIEW, UNCERTAIN, FAILED)}
    return {"run_id": Path(run_dir).name, "metadata": metadata, "plan": plan, "library": library,
            "vlm": records, "pages": pages, "counts": counts, "pending": vlm_pending(records),
            "finalized": is_finalized(run_dir), "decision_logic_version": version,
            "decision_revision": (Path(decision_revisions(run_dir)[-1]["path"]).name
                                  if decision_revisions(run_dir) else None)}


def _texts_for_evaluation(state: dict) -> dict[str, dict[int, str]]:
    library = {page["page"]: page for page in state["library"]["pages"]}
    texts = {
        "pymupdf_raw": {n: p["raw_text"] for n, p in library.items()},
        "pymupdf_processed": {n: p["processed_text"] for n, p in library.items()},
        "final_hybrid": {p["page"]: p["final_text"] for p in state["pages"]},
    }
    vlm = {n: r["text"] for n, r in state["vlm"].items() if _vlm_usable(r)}
    if vlm:
        texts["vlm_raw"] = vlm
    rejoined = {p["page"]: p["rejoined"]["text"] for p in state["pages"] if p.get("rejoined")}
    if rejoined:
        texts["library_rejoined"] = rejoined
    return texts


def _full_text(state: dict) -> str:
    total = len(state["pages"])
    return "\n\n".join(
        f"===== Page {item['page']} / {total} | {item['status']} | source: {item['final_source']} =====\n"
        f"{item['final_text']}" for item in state["pages"]) + "\n"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"


def report_html(state: dict, evaluation: dict | None, *, finalized: bool, layout_docs: dict | None = None,
                layout_note: str | None = None) -> str:
    metadata = state["metadata"]
    rows, sections = [], []
    for item in state["pages"]:
        audit = item.get("number_audit") or {}
        rows.append(f"<tr><td>{item['page']}</td><td>{_esc(item['category'])}</td><td>{_esc(item['status'])}</td>"
                    f"<td>{_esc(item['final_source'])}</td><td>{_esc(len(audit.get('conflicts') or []))}</td>"
                    f"<td>{_esc('; '.join(item['warnings']))}</td></tr>")
        conflicts = "".join(
            f"<tr><td>{_esc(c['normalized'])}</td><td>{_esc(c['library_count'])}: {_esc(' '.join(c['library_originals']))}</td>"
            f"<td>{_esc(c['vlm_count'])}: {_esc(' '.join(c['vlm_originals']))}</td></tr>" for c in audit.get("conflicts") or [])
        sections.append(
            f"<section><h3>Page {item['page']} - {_esc(item['status'])}</h3>"
            f"<p class='m'>Source: {_esc(item['final_source'])}<br>{'<br>'.join(_esc(r) for r in item['reasons'])}</p>"
            + "".join(f"<p class='w'>{_esc(w)}</p>" for w in item["warnings"])
            + (f"<table><tr><th>Number</th><th>Library</th><th>VLM</th></tr>{conflicts}</table>" if conflicts else "")
            + f"<pre dir='auto'>{_esc(item['final_text'])}</pre></section>")
    evaluation_html = ""
    if evaluation:
        evaluation_rows = "".join(
            f"<tr><td>{_esc(source)}</td><td>{_pct(((data.get('total') or {}).get('none') or {}).get('cer'))}</td>"
            f"<td>{_pct(((data.get('total') or {}).get('none') or {}).get('wer'))}</td>"
            f"<td>{_pct(((data.get('total') or {}).get('arabic_normalized') or {}).get('cer'))}</td>"
            f"<td>{_pct(((data.get('total') or {}).get('arabic_normalized') or {}).get('wer'))}</td></tr>"
            for source, data in evaluation["sources"].items())
        evaluation_html = (f"<h2>Evaluation against Ground Truth ({_esc(evaluation['level'])} level; scope: "
                           f"{_esc(evaluation.get('scope_label'))})</h2>"
                           + f"<p class='m'>{_esc(evaluation.get('statement') or evaluation.get('not_computed'))}</p>"
                           + f"<table><tr><th>Source</th><th>CER</th><th>WER</th><th>CER (normalized)</th>"
                           f"<th>WER (normalized)</th></tr>{evaluation_rows}</table>"
                           f"<p class='m'>Normalized: {_esc(diagnostics.ARABIC_NORMALIZATION)}</p>")
    counts = state["counts"]
    if layout_docs:
        label = (f"approved layout revision {layout_docs['revision']}" if layout_docs.get("approved")
                 else "assembled now; no layout revision approved yet")
        documents_html = (f"<h2>Page documents ({_esc(label)})</h2>"
                          + "".join(f"<h3>Page {page['page']}</h3><pre dir='auto'>{_esc(page['markdown'])}</pre>"
                                    for page in layout_docs["pages"]))
    else:
        documents_html = f"<h2>Page documents</h2><p class='m'>{_esc(layout_note or 'not available')}</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{_esc(state['run_id'])}</title>
<style>body{{font-family:system-ui,Segoe UI,Tahoma,sans-serif;margin:24px;color:#1f2328;background:#fff}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0}}td,th{{border:1px solid #d0d7de;padding:4px 8px;text-align:left}}
pre{{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;padding:12px}}.m{{color:#57606a;font-size:12px}}
.w{{color:#9a6700}}</style></head><body>
<h1>Hybrid PDF Extraction {_esc(state['run_id'])}</h1>
<p>{'Final report.' if finalized else '<b>Provisional report (run not finalized).</b>'} Raw outputs are saved unchanged;
the final text per page is a derived result with its source and reason. API success is not proof of accuracy.</p>
<p>Document: {_esc(metadata['document']['display_name'])} ({metadata['document']['page_count']} pages, SHA-256
{_esc(metadata['document']['sha256'])})<br>Model: {_esc(metadata['model_label'])}<br>
Accepted {counts[ACCEPTED]} | Needs Review {counts[NEEDS_REVIEW]} | Uncertain {counts[UNCERTAIN]} | Failed {counts[FAILED]}</p>
<table><tr><th>Page</th><th>Category</th><th>Status</th><th>Final source</th><th>Number conflicts</th><th>Warnings</th></tr>
{''.join(rows)}</table>{evaluation_html}{documents_html}{''.join(sections)}</body></html>"""


# ---------------------------------------------------------- layout stage


def _layout_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "layout"


def layout_revisions(run_dir: Path) -> list[dict]:
    return [payload | {"path": str(path)} for path, payload in ordered(_layout_dir(run_dir), "revision") if payload]


def _sha_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def layout_pdf(run_dir: Path, metadata: dict, pdf_path: Path | None = None) -> tuple[Path, str]:
    """The PDF for layout geometry: the run's own copy, else the given file. SHA-256 must match."""
    expected = metadata["document"]["sha256"]
    own = Path(run_dir) / "source.pdf"
    if own.is_file() and _sha_file(own) == expected:
        return own, "run copy (source.pdf)"
    if pdf_path is not None and Path(pdf_path).is_file() and documents.file_sha256(Path(pdf_path)) == expected:
        return Path(pdf_path), "workspace PDF verified by SHA-256"
    raise HybridError("The layout needs the PDF whose SHA-256 matches this run (select the same PDF)")


def compute_layout(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, pdf_path: Path | None = None,
                   state: dict | None = None, transcriptions: dict[str, dict] | None = None) -> dict:
    """Assemble every page in memory. Local only: no model, no network, nothing written."""
    import pymupdf

    run_dir = Path(run_dir)
    state = state or summary(run_dir, spec, root)
    library_pages = {page["page"]: page for page in state["library"]["pages"]}
    reorder_config = state["library"]["settings"].get("reorder") or {}
    tables = (_read(run_dir / "tables.json") or {}).get("pages") or {}
    pdf, pdf_source = layout_pdf(run_dir, state["metadata"], pdf_path)
    pages, crops = [], {}
    with pymupdf.open(pdf) as document:
        for decision in state["pages"]:
            number = decision["page"]
            elements, page_crops = hybrid_layout.extract_page_elements(
                document[number - 1], number, reorder_config=reorder_config, tables_doc=tables.get(str(number)))
            vlm_text = (state["vlm"].get(number) or {}).get("text") or ""
            assembled = hybrid_layout.assemble_page(
                elements, decision=decision, library_page=library_pages[number], tables_doc=tables.get(str(number)),
                vlm_text=vlm_text, reorder_config=reorder_config,
                previous_tables_doc=tables.get(str(number - 1)), transcriptions=transcriptions)
            pages.append({"page": number, **assembled,
                          "elements": {key: value for key, value in elements.items() if key != "settings"}})
            crops.update(page_crops)
    full = "\n".join(f"--- Page {page['page']} ---\n\n{page['markdown']}" for page in pages)
    return {
        "layout_version": hybrid_layout.LAYOUT_VERSION, "pages": pages, "full_document": full, "crops": crops,
        "settings": hybrid_layout.DEFAULTS,
        "sources": {"library_raw.json": _sha_file(run_dir / "library_raw.json"),
                    "tables.json": _sha_file(run_dir / "tables.json"),
                    "pdf": {"source": pdf_source, "sha256": state["metadata"]["document"]["sha256"]},
                    "decision_logic_version": state["decision_logic_version"],
                    "decision_revision": state["decision_revision"],
                    "vlm_text_sha256": {str(n): hashlib.sha256((r.get("text") or "").encode("utf-8")).hexdigest()
                                        for n, r in sorted(state["vlm"].items())}},
    }


def _layout_report(computed: dict, title: str) -> str:
    sections = "".join(f"<section><h2>Page {page['page']}</h2><pre dir='auto'>{_esc(page['markdown'])}</pre>"
                       + (f"<p class='w'>Needs review: {_esc('; '.join(page['record']['needs_review']))}</p>"
                          if page["record"]["needs_review"] else "") + "</section>" for page in computed["pages"])
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(title)}</title><style>"
            "body{font-family:system-ui,Segoe UI,Tahoma,sans-serif;margin:24px;background:#fff;color:#1f2328}"
            "pre{white-space:pre-wrap;background:#f6f8fa;border:1px solid #d0d7de;padding:12px}.w{color:#9a6700}"
            f"</style></head><body><h1>{_esc(title)}</h1>{sections}</body></html>")


def preview_layout(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *,
                   pdf_path: Path | None = None) -> dict:
    computed = compute_layout(run_dir, spec, root, pdf_path=pdf_path)
    revisions = layout_revisions(run_dir)
    previous = revisions[-1] if revisions else None
    computed["previous_revision"] = Path(previous["path"]).name if previous else None
    computed["previous_pages"] = {str(item["page"]): item for item in (previous or {}).get("pages", [])}
    return computed


def apply_layout(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, by: str, statement: str,
                 pdf_path: Path | None = None, kind: str = "recompute") -> Path:
    """Write a new, append-only layout revision (also allowed on finalized runs: nothing
    existing is changed; the revision lives only under layout/)."""
    if statement != LAYOUT_STATEMENT or not by.strip():
        raise HybridError("Enter your name and tick the confirmation to save the layout revision")
    run_dir = Path(run_dir)
    computed = compute_layout(run_dir, spec, root, pdf_path=pdf_path)
    base = _layout_dir(run_dir)
    base.mkdir(parents=True, exist_ok=True)
    sequence = len(layout_revisions(run_dir)) + 1
    while True:
        folder = base / f"rev_{sequence:06d}"
        try:
            folder.mkdir()
            break
        except FileExistsError:
            sequence += 1
    files = {}
    for page in computed["pages"]:
        files[f"page_{page['page']:03d}.md"] = page["markdown"].encode("utf-8", errors="surrogatepass")
        files[f"page_{page['page']:03d}.layout.json"] = json.dumps(
            {"record": page["record"], "elements": page["elements"], "plain_text_layer": page["plain_text_layer"]},
            ensure_ascii=False, indent=2
        ).encode("utf-8", errors="surrogatepass")
    files["full_document.md"] = computed["full_document"].encode("utf-8", errors="surrogatepass")
    files["report_layout.html"] = _layout_report(computed, f"{run_dir.name} layout revision {sequence}").encode(
        "utf-8", errors="surrogatepass")
    for name, data in computed["crops"].items():
        files[f"figures/{name}"] = data
    for name, data in files.items():
        _write_once(folder / name, data)
    _write_once(base / f"revision_{sequence:06d}.json", {
        "schema_version": 1, "kind": "hybrid_layout_revision", "kind_detail": kind, "revision": sequence,
        "folder": folder.name, "layout_version": computed["layout_version"], "created_at": _now().isoformat(),
        "approved_by": by.strip(), "statement": statement, "settings": computed["settings"],
        "sources": computed["sources"], "run_finalized_at_creation": is_finalized(run_dir),
        "pages": [{"page": page["page"], "order": page["record"].get("order"),
                   "fallback_reason": page["record"]["fallback_reason"],
                   "needs_review": page["record"]["needs_review"],
                   "validation": page["record"]["validation"]} for page in computed["pages"]],
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    })
    return folder


def latest_layout(run_dir: Path) -> dict | None:
    """The approved layout documents of the latest revision, read from disk."""
    revisions = layout_revisions(run_dir)
    if not revisions:
        return None
    latest = revisions[-1]
    folder = _layout_dir(run_dir) / latest["folder"]
    pages = []
    for item in latest["pages"]:
        md = folder / f"page_{item['page']:03d}.md"
        record = _read(folder / f"page_{item['page']:03d}.layout.json") or {}
        pages.append({"page": item["page"], "markdown": md.read_text(encoding="utf-8") if md.exists() else "",
                      "record": record.get("record") or item,
                      "plain_text_layer": record.get("plain_text_layer") or ""})
    full = folder / "full_document.md"
    return {"revision": latest["revision"], "folder": str(folder), "pages": pages,
            "full_document": full.read_text(encoding="utf-8") if full.exists() else "", "approved": True,
            "kind": latest.get("kind_detail"), "approved_by": latest.get("approved_by"),
            "layout_version": latest.get("layout_version"), "created_at": latest.get("created_at")}


def _layout_plain(computed_or_latest: dict | None) -> dict[int, str]:
    if not computed_or_latest:
        return {}
    out = {}
    for page in computed_or_latest["pages"]:
        if "plain_text_layer" in page:
            out[page["page"]] = page["plain_text_layer"]
    return out


def outputs(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, finalized: bool = False,
            pdf_path: Path | None = None) -> dict[str, bytes]:
    from lab.core.hybrid import enrich as hybrid_enrich  # noqa: PLC0415 - avoid an import cycle

    state = summary(run_dir, spec, root)
    layout_docs = latest_layout(run_dir)
    layout_note = None
    layout_plain: dict[int, str] = _layout_plain(layout_docs)
    enriched = hybrid_enrich.latest_enriched(run_dir)
    if enriched is not None:  # the approved enrichment revision is the current page document
        layout_docs = enriched
    if layout_docs is None:
        try:
            computed = compute_layout(run_dir, spec, root, pdf_path=pdf_path, state=state)
            layout_docs = {"revision": None, "pages": computed["pages"], "full_document": computed["full_document"],
                           "approved": False}
            layout_plain = _layout_plain(computed)
        except HybridError as exc:
            layout_note = str(exc)
    gt_path = Path(run_dir) / "ground_truth.txt"
    gt_meta = _read(Path(run_dir) / "ground_truth.json") or {}
    evaluation = None
    if gt_path.exists():
        texts = _texts_for_evaluation(state)
        if layout_plain:
            texts["layout_text_layer"] = layout_plain
        evaluation = evaluate(gt_path.read_text(encoding="utf-8"), texts,
                              state["metadata"]["document"]["page_count"], scope=gt_meta.get("scope", "text_layer"))
        if "scope" not in gt_meta:
            evaluation["scope_note"] = "scope not recorded for this ground truth; treated as text layer only"
    files: dict[str, bytes] = {}
    if layout_docs:
        for page in layout_docs["pages"]:
            files[f"page_{page['page']:03d}.md"] = page["markdown"].encode("utf-8", errors="surrogatepass")
        files["full_document.md"] = layout_docs["full_document"].encode("utf-8", errors="surrogatepass")
    for item in state["pages"]:
        files[f"page_{item['page']:03d}.txt"] = item["final_text"].encode("utf-8", errors="surrogatepass")
        files[f"page_{item['page']:03d}.json"] = json.dumps(
            {key: value for key, value in item.items() if key != "final_text"}
            | {"vlm_job_state": (state["vlm"].get(item["page"]) or {}).get("job_state")},
            ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    files["full_text.txt"] = _full_text(state).encode("utf-8", errors="surrogatepass")
    if state["vlm"]:
        files["vlm_raw.json"] = json.dumps({"model": state["metadata"]["model"], "pages": {
            str(page): record for page, record in sorted(state["vlm"].items())}},
            ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    if evaluation is not None:
        files["evaluation.json"] = json.dumps(evaluation, ensure_ascii=False, indent=2).encode("utf-8")
    files["report.html"] = report_html(state, evaluation, finalized=finalized, layout_docs=layout_docs,
                                       layout_note=layout_note).encode("utf-8", errors="surrogatepass")
    files["results.json"] = json.dumps({
        "schema_version": 1, "kind": "hybrid_results", "run_id": state["run_id"], "finalized": finalized,
        "generated_at": _now().isoformat(), "document": state["metadata"]["document"],
        "model": state["metadata"]["model"], "counts": state["counts"], "vlm_pending": state["pending"],
        "decision_logic_version": state["decision_logic_version"], "decision_revision": state["decision_revision"],
        "evaluation_total": {source: data.get("total") for source, data in (evaluation or {}).get("sources", {}).items()},
        "ground_truth": {"attached": gt_path.exists(),
                         "files": (_read(Path(run_dir) / "ground_truth.json") or {}).get("sources", [])},
        "layout": {"revision": (layout_docs or {}).get("revision"), "approved": bool((layout_docs or {}).get("approved")),
                   "note": layout_note or ("page_NNN.md / full_document.md from the approved layout revision"
                                           if (layout_docs or {}).get("approved") else
                                           "page_NNN.md / full_document.md assembled now (no layout revision approved yet)")},
        "evaluation_scope": (evaluation or {}).get("scope_label"),
        "pages": state["pages"],
        "note": "Ground Truth was used only for this evaluation, after processing.",
    }, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    return files


def finalize(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, pdf_path: Path | None = None,
             by: str = "") -> Path:
    """Write the derived files once (results.json last). Open model jobs stay as they are.

    Without an approved layout revision, an initial one is saved first (recorded as created
    by finalize), so page_NNN.md / full_document.md come from an approved revision."""
    run_dir = Path(run_dir)
    if is_finalized(run_dir):
        raise HybridError("The run is already finalized")
    if not layout_revisions(run_dir):
        try:
            apply_layout(run_dir, spec, root, by=by.strip() or "finalize (initial layout)",
                         statement=LAYOUT_STATEMENT, pdf_path=pdf_path, kind="initial layout at finalize")
        except HybridError:
            pass  # no matching PDF: finalized without page documents (recorded in results.json)
    files = outputs(run_dir, spec, root, finalized=True, pdf_path=pdf_path)
    results = files.pop("results.json")
    for name, data in files.items():
        _write_once(run_dir / name, data)
    _write_once(run_dir / "results.json", results)
    return run_dir


def attach_ground_truth(run_dir: Path, text: str, sources: list[str], spec: ProviderSpec | None,
                        root: Path | None, *, scope: str = "text_layer") -> Path:
    """Attach Ground Truth once (evaluation only). For a finalized run, evaluation.json is
    written now if it does not exist yet; nothing else changes."""
    run_dir = Path(run_dir)
    if (run_dir / "ground_truth.txt").exists():
        raise HybridError("Ground truth is already attached to this run (write-once)")
    if not text.strip():
        raise HybridError("The ground-truth file is empty")
    metadata, _plan, _library = load(run_dir)
    _write_once(run_dir / "ground_truth.txt", text)
    _write_once(run_dir / "ground_truth.json", {
        "sources": list(sources), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "attached_at": _now().isoformat(), "attached_after_run_start": True,
        "scope": scope, "scope_label": GT_SCOPES.get(scope, scope),
        "description": diagnostics.describe_ground_truth(text, metadata["document"]["page_count"]),
        "use": "evaluation after processing only; never used for decisions or sent anywhere"})
    if is_finalized(run_dir) and not (run_dir / "evaluation.json").exists():
        state = summary(run_dir, spec, root)
        _write_once(run_dir / "evaluation.json", evaluate(text, _texts_for_evaluation(state),
                                                          metadata["document"]["page_count"], scope=scope))
    return run_dir


def zip_bytes(run_dir: Path, spec: ProviderSpec | None, root: Path | None, *, pdf_path: Path | None = None) -> bytes:
    run_dir = Path(run_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.name != ".executor.lock":
                archive.write(path, f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}")
        if not is_finalized(run_dir):
            for name, data in outputs(run_dir, spec, root, finalized=False, pdf_path=pdf_path).items():
                archive.writestr(f"{run_dir.name}/provisional/{name}", data)
        from lab.core.e2.api.providers import load_providers  # noqa: PLC0415
        from lab.core.hybrid import enrich as hybrid_enrich  # noqa: PLC0415

        # Phase 2 provider outputs are stored in the shared result store (journals, runs/, pages/);
        # copies are added to the ZIP for review (read-only; nothing is written or duplicated on disk).
        provider_specs = {item.id: item for item in load_providers()}
        for set_dir in hybrid_enrich.region_sets(run_dir):
            try:
                for name, data in hybrid_enrich.outputs_export(set_dir, provider_specs.get, Path(root or results_root())).items():
                    archive.writestr(f"{run_dir.name}/{name}", data)
            except hybrid_enrich.EnrichError as exc:
                archive.writestr(f"{run_dir.name}/phase2_outputs/{set_dir.name}/EXPORT_ERROR.txt", str(exc))
        enriched = hybrid_enrich.latest_enriched(run_dir)
        if enriched:
            archive.writestr(f"{run_dir.name}/LATEST_DOCUMENTS.txt",
                             f"Current approved page documents: {enriched['revision']} "
                             f"(enrichment/documents/{Path(enriched['folder']).name}).\n"
                             "Layout revisions and files written at finalization stay as they were.\n")
        latest = latest_layout(run_dir)
        if latest:
            archive.writestr(f"{run_dir.name}/LATEST_LAYOUT.txt",
                             f"Current approved layout revision: {latest['revision']} ({Path(latest['folder']).name}).\n"
                             "Its page_NNN.md and full_document.md are in that folder. Files written at "
                             "finalization stay as they were (write-once).\n")
    return buffer.getvalue()


def list_runs(root: Path | None = None) -> list[dict]:
    base = hybrid_root(root)
    found = []
    for path in sorted(base.glob("run_*/metadata.json"), reverse=True) if base.is_dir() else []:
        metadata = _read(path)
        if metadata and metadata.get("kind") == "hybrid_run":
            found.append({"run_dir": path.parent, "run_id": path.parent.name, "metadata": metadata,
                          "finalized": is_finalized(path.parent)})
    return found
