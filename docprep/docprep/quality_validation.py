"""Integrity validation for Component 3 outputs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .quality_inputs import QualityInputs, rehash_quality_inputs
from .quality_model import QualityResult


def _check(code: str, ok: bool, message: str, items: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "severity": "ok" if ok else "error", "message": message,
            "items": items or [], "item_count": len(items or [])}


def _forbidden_correction_keys(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            current = f"{path}.{key}" if path else str(key)
            if key in {"corrected_text", "replacement_text", "inferred_value", "corrected_value"}:
                found.append(current)
            found.extend(_forbidden_correction_keys(child, current))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_correction_keys(child, f"{path}[{index}]"))
    return found


def validate_quality(result: QualityResult, inputs: QualityInputs) -> list[dict[str, Any]]:
    validations = []
    known_ids = set(result.nodes)
    unresolved = sorted({source_id for finding in result.findings for source_id in finding.source_ids
                         if source_id not in known_ids})
    validations.append(_check("finding_sources_resolve", not unresolved,
                              "every finding source resolves" if not unresolved else "unresolved finding sources", unresolved))

    invalid_offsets = []
    for finding in result.findings:
        for offset in finding.offsets:
            source = result.nodes.get(str(offset.get("source_id")))
            start, end = offset.get("start"), offset.get("end")
            if not source or not isinstance(source.get("text"), str) or not isinstance(start, int) or \
                    not isinstance(end, int) or not (0 <= start <= end <= len(source["text"])) or \
                    source["text"][start:end] != offset.get("text"):
                invalid_offsets.append(finding.finding_id)
    validations.append(_check("character_offsets_valid", not invalid_offsets,
                              "all character offsets reproduce exact source text" if not invalid_offsets else
                              "invalid finding offsets", sorted(set(invalid_offsets))))

    ids = [finding.finding_id for finding in result.findings]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    validations.append(_check("finding_ids_unique", not duplicates,
                              "finding IDs are unique" if not duplicates else "duplicate finding IDs", duplicates))

    correction_paths = []
    for finding in result.findings:
        correction_paths.extend(f"{finding.finding_id}:{path}" for path in _forbidden_correction_keys(finding.evidence))
    validations.append(_check("no_unsupported_corrections", not correction_paths,
                              "findings contain no corrected or inferred replacement values" if not correction_paths else
                              "findings contain prohibited correction fields", correction_paths))

    text_changed, state_changed = [], []
    for element in inputs.run.elements:
        node = result.nodes.get(element.element_id, {})
        if node.get("text") != (element.text or ""):
            text_changed.append(element.element_id)
        if (node.get("provenance") or {}).get("validation_state") != element.validation_state:
            state_changed.append(element.element_id)
    validations.append(_check("source_text_unchanged", not text_changed,
                              "all source text is unchanged" if not text_changed else "source text changed", text_changed))
    validations.append(_check("validation_states_unchanged", not state_changed,
                              "all Nass validation states are unchanged" if not state_changed else
                              "validation states changed", state_changed))

    current_hashes = rehash_quality_inputs(inputs)
    hashes_ok = result.hashes_before == result.hashes_after == current_hashes
    validations.append(_check("input_hashes_unchanged", hashes_ok,
                              "source, Structure and Relationship hashes are unchanged" if hashes_ok else
                              "one or more input hashes changed"))

    source_ids = {element.element_id for element in inputs.run.elements}
    inventory_ids = [record.source_id for record in result.inventory]
    inventory_errors = sorted((source_ids - set(inventory_ids)) |
                              {value for value, count in Counter(inventory_ids).items() if count != 1})
    validations.append(_check("inventory_complete", not inventory_errors,
                              "quality inventory covers every original element exactly once" if not inventory_errors else
                              "quality inventory is incomplete or duplicated", inventory_errors))

    unsupported = [finding.finding_id for finding in result.findings
                   if not finding.evidence and not finding.limitations]
    validations.append(_check("evidence_or_limitation_present", not unsupported,
                              "every finding has evidence or an explicit limitation" if not unsupported else
                              "findings lack evidence and limitations", unsupported))

    missing_context = [finding.finding_id for finding in result.findings
                       if not finding.page_numbers and not finding.related_ids and not finding.limitations]
    validations.append(_check("page_or_context_present", not missing_context,
                              "findings identify pages, related context or a limitation" if not missing_context else
                              "findings lack page and related context", missing_context))

    skipped_errors = [record.source_id for record in result.inventory
                      if any(not reason for reason in record.skipped_checks.values())]
    validations.append(_check("skipped_checks_explained", not skipped_errors,
                              "all skipped checks have explicit reasons" if not skipped_errors else
                              "skipped checks lack reasons", skipped_errors))
    return validations


def validation_summary(validations: list[dict[str, Any]]) -> dict[str, int]:
    return {severity: sum(item["severity"] == severity for item in validations) for severity in ("ok", "error")}

