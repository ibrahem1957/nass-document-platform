"""Validation of an automatic extraction. Flags only: nothing is corrected, converted or replaced.

Per OCR job (from the service's own attempt records): empty output, loop-stopped, timed out, token
limit reached, retried (and whether the retry still failed), service error / unconfirmed / pending.
Per region: incomplete-region suspicion (border cuts another block), pending review, removed duplicate.
Per text: number formats and units via `hybrid.numbers.validate` (%, ‰, separators, U+2212, axis-like
sequences) and Phase 1/v3 layout review flags carried over.
"""

from __future__ import annotations

from typing import Any

from lab.core.auto import digits
from lab.core.hybrid import numbers

VALIDATION_VERSION = "1"
ERROR, WARNING, INFO = "error", "warning", "info"


def job_flags(job: dict, result: dict | None) -> list[dict]:
    """Flags for one OCR job from its latest outcome."""
    flags: list[dict] = []
    where = {"job_id": job["job_id"], "page": job["page"], "region_id": job.get("region_id")}
    if result is None:
        return [where | {"severity": ERROR, "code": "not_processed", "message": "OCR not done yet"}]
    state = result.get("state")
    if state != "completed":
        severity = ERROR
        message = {"unconfirmed": "request sent but no response; not resent automatically",
                   "not_transmitted": "Qari service unavailable; nothing was sent",
                   "service_error": "the Qari service returned an error"}.get(state, f"state {state}")
        return [where | {"severity": severity, "code": f"ocr_{state}", "message": message,
                         "detail": result.get("error")}]
    response = result.get("response") or {}
    final = response.get("final_flags") or {}
    text = response.get("final_text") or ""
    if not text.strip():
        flags.append(where | {"severity": ERROR, "code": "empty_ocr", "message": "the OCR output is empty"})
    for name, message in (("loop_stopped", "the final attempt was stopped by the loop detector (text may be cut)"),
                          ("timed_out", "the final attempt reached the time limit (text may be incomplete)"),
                          ("hit_limit", "the final attempt reached max_new_tokens (text may be incomplete)")):
        if final.get(name):
            flags.append(where | {"severity": WARNING, "code": name, "message": message})
    if response.get("retried"):
        first = (response.get("attempts") or [{}])[0]
        reasons = [n for n in ("loop_stopped", "hit_limit", "timed_out") if first.get(n)]
        flags.append(where | {"severity": INFO, "code": "retried",
                              "message": "attempt 1 " + "/".join(reasons) + "; retried once with repetition_penalty"})
    if result.get("source") == "offline_import":
        flags.append(where | {"severity": INFO, "code": "offline_import",
                              "message": "result imported from an offline Qari run (hash-checked)"})
    return flags


def region_flags(region: dict) -> list[dict]:
    flags = []
    where = {"page": region["page"], "region_id": region.get("region_id")}
    if region.get("suspected_missing_content"):
        flags.append(where | {"severity": WARNING, "code": "incomplete_region",
                              "message": "the region border cuts another block; content may be missing",
                              "detail": {"cut_blocks": region["suspected_missing_content"],
                                         "proposed_expansion_pt": region.get("proposed_expansion_pt")}})
    if region.get("decision") == "review" and not region.get("processed_on_request"):
        flags.append(where | {"severity": WARNING, "code": "pending_review",
                              "message": "region not processed (" + "; ".join(region.get("reasons") or []) + ")"})
    if region.get("decision") in {"auto_flagged", "auto_review"}:
        flags.append(where | {"severity": INFO, "code": "processed_needs_review",
                              "message": "; ".join(r for r in region.get("reasons") or [] if "Needs Review" in r)})
    if region.get("title_ambiguity"):
        flags.append(where | {"severity": WARNING, "code": "ambiguous_title_unattached",
                              "message": "processed without attaching a nearby title/caption: "
                                         + "; ".join(region["title_ambiguity"])})
    if region.get("decision") == "removed":
        flags.append(where | {"severity": INFO, "code": "duplicate_removed",
                              "message": "; ".join(region.get("reasons") or [])})
    return flags


def text_flags(element: dict) -> tuple[dict, list[dict]]:
    """Number/unit validation of one element's content."""
    result = numbers.validate(element.get("content") or "")
    flags = []
    if result["needs_review"]:
        flags.append({"severity": WARNING, "code": "number_format", "page": element["page"],
                      "element_id": element["id"], "message": "; ".join(result["flags"][:8])})
    if str(element.get("source") or "").startswith("qari"):  # OCR outputs only: digit-shape checks
        result["digit_checks"] = digits.static_flags(element.get("content") or "")
        for item in result["digit_checks"]:
            flags.append({"severity": item["severity"], "code": item["code"], "page": element["page"],
                          "element_id": element["id"], "message": item["message"]})
    return result, flags


def report(elements: list[dict], jobs: list[dict], results: dict[str, dict | None], regions: list[dict],
           page_flags: list[dict]) -> dict[str, Any]:
    flags: list[dict] = list(page_flags)
    for job in jobs:
        flags += job_flags(job, results.get(job["job_id"]))
    for region in regions:
        flags += region_flags(region)
    per_element = {}
    for element in elements:
        if element["type"] in {"text", "heading", "table", "figure", "page_text"} and element.get("content"):
            result, found = text_flags(element)
            per_element[element["id"]] = {"data_values": result["data_values"], "flags": result["flags"],
                                          "axis_candidates": result["axis_candidates"],
                                          "digit_checks": result.get("digit_checks") or []}
            flags += found
            needs = [f for f in found if f["severity"] != INFO]
            element["validation_state"] = "needs_review" if needs else element.get("validation_state", "ok")
    by_severity = {name: sum(1 for f in flags if f["severity"] == name) for name in (ERROR, WARNING, INFO)}
    return {"validation_version": VALIDATION_VERSION, "numbers_version": numbers.VALIDATION_VERSION,
            "summary": by_severity, "status": "errors" if by_severity[ERROR] else
            ("needs_review" if by_severity[WARNING] else "ok"), "flags": flags, "numbers": per_element,
            "note": "Validation only: no text, number or unit was corrected; chart values are never estimated."}
