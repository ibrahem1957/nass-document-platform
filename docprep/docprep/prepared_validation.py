"""Integrity checks for the unified prepared document."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .prepared_inputs import PreparedInputs, rehash_prepared_inputs
from .prepared_model import PreparedResult


def _check(code: str, ok: bool, message: str, items: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "severity": "ok" if ok else "error", "message": message,
            "items": items or [], "item_count": len(items or [])}


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _cycle(nodes: dict[str, list[str]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    found: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            found.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        for child in nodes.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in nodes:
        visit(node)
    return sorted(found)


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "prepared_document.schema.json"


def _schema_check(document: dict[str, Any]) -> dict[str, Any]:
    """Full JSON Schema validation. Missing schema or validator is an error, never a silent pass."""
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        return _check("json_schema_valid", False, "jsonschema is not installed: the prepared document could not be "
                      "validated against its schema")
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _check("json_schema_valid", False, f"prepared document schema unavailable: {exc}")
    errors = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda item: list(item.absolute_path))
    items = [f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message[:200]}"
             for error in errors[:50]]
    return _check("json_schema_valid", not errors,
                  f"document validates against {SCHEMA_PATH.name}" if not errors else
                  f"{len(errors)} schema violation(s) against {SCHEMA_PATH.name}", items)


def validate_prepared(result: PreparedResult, inputs: PreparedInputs) -> list[dict[str, Any]]:
    document = result.document
    validations = []
    source_by_id = {item.element_id: item for item in inputs.run.elements}
    elements = document.get("elements") or []
    element_ids = [str(item.get("element_id")) for item in elements]
    inventory_errors = sorted((set(source_by_id) - set(element_ids)) | (set(element_ids) - set(source_by_id)) |
                              set(_duplicates(element_ids)))
    validations.append(_check("source_inventory_exact", not inventory_errors,
                              "every Nass element appears exactly once" if not inventory_errors else
                              "source inventory is missing, extra, or duplicated", inventory_errors))

    text_errors = [element_id for element_id, source in source_by_id.items()
                   if next((item.get("text") for item in elements if item.get("element_id") == element_id), object())
                   != source.text]
    validations.append(_check("source_text_verbatim", not text_errors,
                              "all source strings are byte-for-byte equal as UTF-8 text" if not text_errors else
                              "source text was lost or rewritten", text_errors))

    state_errors = [element_id for element_id, source in source_by_id.items()
                    if next((item.get("validation_state") for item in elements
                             if item.get("element_id") == element_id), object()) != source.validation_state]
    validations.append(_check("validation_states_preserved", not state_errors,
                              "all Nass validation states are preserved" if not state_errors else
                              "Nass validation states changed", state_errors))

    sections = document.get("sections") or []
    section_ids = [str(item.get("section_id")) for item in sections]
    section_set = set(section_ids)
    section_ref_errors = []
    for section in sections:
        section_id = str(section.get("section_id"))
        parent = section.get("parent")
        if parent is not None and str(parent) not in section_set:
            section_ref_errors.append(f"{section_id}:parent:{parent}")
        section_ref_errors.extend(f"{section_id}:child:{child}" for child in section.get("children") or []
                                  if str(child) not in section_set)
    for element in elements:
        section_ref_errors.extend(f"{element.get('element_id')}:section:{section_id}"
                                  for section_id in element.get("section_ids") or [] if section_id not in section_set)
    validations.append(_check("section_references_resolve", not section_ref_errors,
                              "all section references resolve" if not section_ref_errors else
                              "broken section references", sorted(section_ref_errors)))
    section_cycles = _cycle({str(item.get("section_id")): [str(value) for value in item.get("children") or []]
                             for item in sections})
    validations.append(_check("section_hierarchy_acyclic", not section_cycles,
                              "section hierarchy is acyclic" if not section_cycles else
                              "section hierarchy contains a cycle", section_cycles))

    nodes = document.get("nodes") or []
    node_ids = [str(item.get("node_id")) for item in nodes]
    known_nodes = set(node_ids) | set(element_ids) | section_set | {
        str(item.get("list_id")) for item in document.get("lists") or []
    } | {str(item.get("logical_table_id")) for item in document.get("logical_tables") or []}
    relationships = document.get("relationships") or []
    relation_ids = [str(item.get("rel_id")) for item in relationships]
    relation_errors = [f"{item.get('rel_id')}:{side}:{item.get(side)}" for item in relationships
                       for side in ("source_id", "target_id") if str(item.get(side)) not in known_nodes]
    validations.append(_check("relationship_references_resolve", not relation_errors,
                              "all relationship endpoints resolve" if not relation_errors else
                              "broken relationship endpoints", sorted(relation_errors)))

    findings = (document.get("quality") or {}).get("findings") or []
    finding_ids = [str(item.get("finding_id")) for item in findings]
    finding_errors = []
    for finding in findings:
        finding_id = str(finding.get("finding_id"))
        finding_errors.extend(f"{finding_id}:source:{value}" for value in finding.get("source_ids") or []
                              if str(value) not in known_nodes)
        finding_errors.extend(f"{finding_id}:related:{value}" for value in finding.get("related_ids") or []
                              if str(value) not in known_nodes)
        finding_errors.extend(f"{finding_id}:relationship:{value}"
                              for value in finding.get("relationship_ids") or [] if str(value) not in relation_ids)
    validations.append(_check("quality_references_resolve", not finding_errors,
                              "all quality references resolve" if not finding_errors else
                              "broken quality references", sorted(finding_errors)))

    reliability_errors = []
    for item in [*elements, *(document.get("visuals") or [])]:
        item_id = str(item.get("element_id") or item.get("visual_id"))
        if item.get("evidence_reliability") not in {"unassessed", "high", "medium", "low"} or \
                not isinstance(item.get("rag_flags"), list) or \
                not isinstance(item.get("evidence_reliability_reasons"), list):
            reliability_errors.append(item_id)
    validations.append(_check("rag_evidence_contract_valid", not reliability_errors,
                              "all elements and visuals expose valid RAG evidence fields" if not reliability_errors else
                              "invalid RAG evidence fields", sorted(reliability_errors)))

    known_missing_errors = []
    known_missing_relations = {str(item.get("target_id")) for item in relationships
                               if item.get("type") == "known_missing"}
    for visual in document.get("visuals") or []:
        if visual.get("visual_type") != "known_missing":
            continue
        visual_id = str(visual.get("visual_id"))
        if visual_id not in known_missing_relations or not isinstance(visual.get("source_record"), dict) or \
                "known_missing" not in (visual.get("rag_flags") or []):
            known_missing_errors.append(visual_id)
    validations.append(_check("known_missing_traceable", not known_missing_errors,
                              "known-missing visuals retain an explicit source record and relationship"
                              if not known_missing_errors else "known-missing visuals lack explicit traceability",
                              sorted(known_missing_errors)))

    audit_errors = []
    for index, item in enumerate(document.get("evidence_audit") or []):
        for key in ("visual_id", "source_id", "target_id"):
            value = item.get(key)
            if value is not None and str(value) not in known_nodes:
                audit_errors.append(f"{index}:{key}:{value}")
    validations.append(_check("evidence_audit_references_resolve", not audit_errors,
                              "all evidence-audit references resolve" if not audit_errors else
                              "evidence audit contains broken references", sorted(audit_errors)))

    span_errors = []
    for span in (document.get("derived_spans") or {}).get("structure") or []:
        parent_id = str(span.get("element_id"))
        source = source_by_id.get(parent_id)
        start, end = span.get("start"), span.get("end")
        if not source or not isinstance(source.text, str) or not isinstance(start, int) or not isinstance(end, int) or \
                not (0 <= start <= end <= len(source.text)) or source.text[start:end] != span.get("text"):
            span_errors.append(str(span.get("span_id")))
    for span in (document.get("derived_spans") or {}).get("relationships") or []:
        parent_id = str(span.get("parent_element_id"))
        source = source_by_id.get(parent_id)
        start, end = span.get("start"), span.get("end")
        if not source or not isinstance(source.text, str) or not isinstance(start, int) or not isinstance(end, int) or \
                not (0 <= start <= end <= len(source.text)) or source.text[start:end] != span.get("text"):
            span_errors.append(str(span.get("span_id")))
    validations.append(_check("derived_span_offsets_valid", not span_errors,
                              "all derived spans reproduce exact source intervals" if not span_errors else
                              "invalid derived span offsets", sorted(span_errors)))

    page_count = document.get("page_count")
    page_errors = []
    for group in (elements, sections, document.get("visuals") or []):
        for item in group:
            values = []
            if isinstance(item.get("page"), int):
                values.append(item["page"])
            values.extend(value for value in (item.get("pages") or []) if isinstance(value, int))
            values.extend(value for value in (item.get("page_start"), item.get("page_end")) if isinstance(value, int))
            if any(value < 1 or not isinstance(page_count, int) or value > page_count for value in values):
                page_errors.append(str(item.get("element_id") or item.get("section_id") or item.get("visual_id")))
    validations.append(_check("page_numbers_valid", not page_errors,
                              "all page references are within the source document" if not page_errors else
                              "invalid page references", sorted(set(page_errors))))

    list_errors, list_cycles = [], []
    for list_record in document.get("lists") or []:
        view = list_record.get("relationship_view") or {}
        items = view.get("items") or []
        ids = {str(item.get("node_id") or item.get("unit_id")) for item in items}
        graph: dict[str, list[str]] = defaultdict(list)
        for item in items:
            item_id = str(item.get("node_id") or item.get("unit_id"))
            parent = item.get("parent_id")
            if parent:
                if str(parent) not in ids:
                    list_errors.append(f"{list_record.get('list_id')}:{item_id}:parent:{parent}")
                else:
                    graph[str(parent)].append(item_id)
            graph.setdefault(item_id, [])
        list_cycles.extend(f"{list_record.get('list_id')}:{value}" for value in _cycle(graph))
    validations.append(_check("list_references_resolve", not list_errors,
                              "all nested-list references resolve" if not list_errors else
                              "broken nested-list references", sorted(list_errors)))
    validations.append(_check("list_hierarchy_acyclic", not list_cycles,
                              "list hierarchies are acyclic" if not list_cycles else
                              "list hierarchy contains a cycle", sorted(list_cycles)))

    table_errors = []
    for table in document.get("logical_tables") or []:
        parts = table.get("source_part_ids") or []
        if any(part not in source_by_id or source_by_id[part].element_type != "table" for part in parts):
            table_errors.append(str(table.get("logical_table_id")))
            continue
        pages = [source_by_id[part].page for part in parts]
        if pages != sorted(pages) or pages != table.get("pages"):
            table_errors.append(str(table.get("logical_table_id")))
    validations.append(_check("logical_tables_valid", not table_errors,
                              "logical table parts are valid and ordered" if not table_errors else
                              "invalid or unordered logical table parts", sorted(table_errors)))

    duplicate_errors = []
    for namespace, values in (("elements", element_ids), ("sections", section_ids), ("nodes", node_ids),
                              ("relationships", relation_ids), ("findings", finding_ids),
                              ("lists", [str(item.get("list_id")) for item in document.get("lists") or []]),
                              ("logical_tables", [str(item.get("logical_table_id"))
                                                  for item in document.get("logical_tables") or []])):
        duplicate_errors.extend(f"{namespace}:{value}" for value in _duplicates(values))
    validations.append(_check("ids_unique_within_namespaces", not duplicate_errors,
                              "IDs are unique within every namespace" if not duplicate_errors else
                              "duplicate IDs within a namespace", sorted(duplicate_errors)))

    metadata = document.get("metadata") or {}
    unknown_ok = metadata.get("author") is None and (
        metadata.get("language") is None or metadata.get("language_status") == "recorded"
    )
    validations.append(_check("unknown_metadata_explicit", unknown_ok,
                              "unknown author/language values are null and not inferred" if unknown_ok else
                              "metadata contains an unsupported inference"))

    validations.append(_schema_check(document))

    current_hashes = rehash_prepared_inputs(inputs)
    hashes_ok = result.input_hashes_before == result.input_hashes_after == current_hashes
    validations.append(_check("input_hashes_unchanged", hashes_ok,
                              "all upstream input hashes are unchanged" if hashes_ok else
                              "one or more upstream inputs changed"))
    return validations


def validation_summary(validations: list[dict[str, Any]]) -> dict[str, int]:
    return {severity: sum(item["severity"] == severity for item in validations) for severity in ("ok", "error")}
