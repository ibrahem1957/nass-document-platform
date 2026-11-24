"""Deterministic assembly of the self-contained prepared document contract."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from typing import Any

from . import __version__
from .prepared_config import PreparedConfig
from .prepared_inputs import PreparedInputs, rehash_prepared_inputs
from .prepared_model import PreparedResult
from .quality_policy import RAG_FLAG_BY_SUBTYPE, affects_content_reliability, finding_reason, rag_flag

# Kept for compatibility; the single source of truth is quality_policy.RAG_FLAG_BY_SUBTYPE / rag_flag().
RAG_FLAG_BY_FINDING = RAG_FLAG_BY_SUBTYPE
TABLE_CONSUMER_RULE = (
    "cell-level questions (a value for a given row and column) may be answered from this table only when "
    "table_structure is 'source_cells'; when it is 'none' the text is a flattened transcription whose "
    "row/column associations are unknown and must not be used as verified cell evidence"
)


def _table_structure(element: dict[str, Any] | None, visual: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable cell-structure state of a table (never fabricated).

    'source_cells': the source run recorded row/column cells (e.g. a native text-layer table or a parsed
    Markdown/HTML table). 'none': only a flattened transcription exists, so row/column associations are unknown."""
    source = (element or {}).get("source_extraction") or {}
    cells = source.get("cells")
    flattened = "table_cells_unknown" in set((element or {}).get("rag_flags") or []) | set(visual.get("rag_flags") or [])
    has_cells = isinstance(cells, list) and len(cells) > 0 and not flattened
    return {
        "table_structure": "source_cells" if has_cells else "none",
        "cell_level_evidence_allowed": has_cells,
        "table_structure_basis": (source.get("cells_note") or "source cells recorded") if has_cells else (
            "no row/column cells were recorded for this table" + (f": {source.get('cells_note')}"
                                                                  if source.get("cells_note") else "")),
        "table_consumer_rule": TABLE_CONSUMER_RULE,
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _nested_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = {str(item.get("node_id") or item.get("unit_id")): deepcopy(item) for item in items}
    children: dict[str, list[str]] = defaultdict(list)
    roots = []
    for node_id, item in records.items():
        parent = item.get("parent_id")
        if parent and str(parent) in records:
            children[str(parent)].append(node_id)
        else:
            roots.append(node_id)

    def visit(node_id: str, trail: frozenset[str]) -> dict[str, Any]:
        item = deepcopy(records[node_id])
        if node_id in trail:
            item["children"] = []
            item["cycle_detected"] = True
            return item
        item["children"] = [visit(child, trail | {node_id}) for child in children.get(node_id, [])]
        return item

    return [visit(node_id, frozenset()) for node_id in roots]


def _metadata(inputs: PreparedInputs, sections: list[dict[str, Any]], nodes: dict[str, dict[str, Any]],
              relationships: list[dict[str, Any]]) -> dict[str, Any]:
    source = inputs.run.document.get("document") or {}
    title_section = next((item for item in sections if item.get("level") == 0), None)
    metadata_relations = [item for item in relationships if item.get("type") == "document_metadata"]
    annotations = []
    for relation in metadata_relations:
        annotation = nodes.get(str(relation.get("source_id")))
        if annotation:
            annotations.append({"node": deepcopy(annotation), "relationship_id": relation.get("rel_id")})
    language = source.get("language") if isinstance(source.get("language"), str) else None
    return {
        "title": title_section.get("title") if title_section else None,
        "title_source_section_id": title_section.get("section_id") if title_section else None,
        "author": None,
        "author_status": "unavailable; metadata annotations are preserved but not reinterpreted",
        "source_filename": source.get("display_name"),
        "source_run_id": inputs.run.run_id,
        "language": language,
        "language_status": "recorded" if language else "unavailable; not inferred by Component 4",
        "page_count": inputs.run.page_count,
        "annotations": annotations,
    }


def build_prepared_document(inputs: PreparedInputs, config: PreparedConfig | None = None) -> PreparedResult:
    config = config or PreparedConfig()
    assignments = deepcopy(inputs.assignments.get("assignments") or [])
    assignments.sort(key=lambda item: (tuple(item.get("order") or [0, 0]), str(item.get("unit_id") or "")))
    assignment_by_element: dict[str, list[dict[str, Any]]] = defaultdict(list)
    assignment_by_section: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        assignment_by_element[str(assignment.get("element_id"))].append(assignment)
        if assignment.get("section_id"):
            assignment_by_section[str(assignment["section_id"])].append(assignment)

    relationships = deepcopy(inputs.relationships.get("relationships") or [])
    nodes = deepcopy(inputs.relationships.get("nodes") or [])
    node_by_id = {str(item.get("node_id")): item for item in nodes}
    relation_by_node: dict[str, list[str]] = defaultdict(list)
    for relation in relationships:
        relation_id = str(relation.get("rel_id"))
        relation_by_node[str(relation.get("source_id"))].append(relation_id)
        relation_by_node[str(relation.get("target_id"))].append(relation_id)

    findings = deepcopy(inputs.quality_findings.get("findings") or [])
    source_states = {item.element_id: item.validation_state for item in inputs.run.elements}
    for finding in findings:
        finding["source_validation_states"] = {
            str(source_id): source_states[str(source_id)] for source_id in finding.get("source_ids") or []
            if str(source_id) in source_states
        }
    finding_by_source: dict[str, list[str]] = defaultdict(list)
    for finding in findings:
        for source_id in finding.get("source_ids") or []:
            finding_by_source[str(source_id)].append(str(finding.get("finding_id")))
    quality_inventory = {
        str(item.get("source_id")): item for item in deepcopy(inputs.quality_inventory.get("inventory") or [])
    }

    def evidence_signals(source_id: str, inventory: dict[str, Any] | None) -> tuple[list[str], str, list[str]]:
        flags = list((inventory or {}).get("rag_flags") or [])
        reasons = list((inventory or {}).get("evidence_reliability_reasons") or [])
        reliability = str((inventory or {}).get("evidence_reliability") or "unassessed")
        relevant = [finding for finding in findings if source_id in {
            str(value) for value in finding.get("source_ids") or []
        }]
        actionable_findings = []
        for finding in relevant:
            category, subtype = str(finding.get("category") or ""), str(finding.get("subtype") or "")
            confidence, evidence = str(finding.get("confidence") or ""), finding.get("evidence") or {}
            flag = rag_flag(category, subtype, confidence, evidence)
            if flag:
                flags.append(flag)
                reasons.append(finding_reason(finding))
            if not affects_content_reliability(category, subtype, confidence, evidence):
                continue
            actionable_findings.append(finding)
            if finding.get("status") in {"open", "needs_review"}:
                flags.append("needs_review")
                reasons.append("requires review: " + finding_reason(finding))
        flags = sorted(set(flags))
        reasons = list(dict.fromkeys(reasons))
        if reliability == "unassessed" and flags:
            severe = any(finding.get("severity") in {"critical", "high"} for finding in actionable_findings)
            reliability = "low" if severe or "known_missing" in flags else "medium"
        if reliability not in {"unassessed", "high", "medium", "low"}:
            reliability = "unassessed"
            reasons.append("unsupported upstream reliability value was not promoted")
        if flags and not reasons:
            reasons.append(f"quality inventory recorded rag_flags: {', '.join(flags)}")
        if reliability != "unassessed" and not reasons:
            reasons.append(f"quality inventory recorded evidence_reliability={reliability}")
        return flags, reliability, reasons

    elements = []
    for source in inputs.run.elements:
        raw = source.raw
        source_metadata = {key: deepcopy(value) for key, value in raw.items()
                           if key not in {"id", "page", "reading_order", "type", "bbox_pt", "content",
                                          "validation_state"}}
        inventory = quality_inventory.get(source.element_id)
        rag_flags, evidence_reliability, reliability_reasons = evidence_signals(source.element_id, inventory)
        elements.append({
            "element_id": source.element_id,
            "original_type": source.element_type,
            "text": source.text,
            "page": source.page,
            "reading_order": source.reading_order,
            "document_order": source.order,
            "bbox_pt": deepcopy(source.bbox),
            "geometry_reliability": (inventory or {}).get("geometry_reliability", "unavailable"),
            "validation_state": source.validation_state,
            "source_extraction": source_metadata,
            "structure_assignments": deepcopy(assignment_by_element.get(source.element_id, [])),
            "section_ids": _unique([str(item.get("section_id")) for item in
                                    assignment_by_element.get(source.element_id, []) if item.get("section_id")]),
            "roles": _unique([str(item.get("role")) for item in
                              assignment_by_element.get(source.element_id, []) if item.get("role")]),
            "relationship_ids": sorted(set(relation_by_node.get(source.element_id, []))),
            "quality_finding_ids": sorted(set(finding_by_source.get(source.element_id, []))),
            "quality_inventory": deepcopy(inventory),
            "rag_flags": rag_flags,
            "evidence_reliability": evidence_reliability,
            "evidence_reliability_reasons": reliability_reasons,
        })

    sections = []
    for original in inputs.section_tree.get("sections") or []:
        section = deepcopy(original)
        section_id = str(section.get("section_id"))
        section_assignments = assignment_by_section.get(section_id, [])
        section["unit_ids_in_reading_order"] = [str(item.get("unit_id")) for item in section_assignments]
        section["element_ids_in_reading_order"] = _unique(
            [str(item.get("element_id")) for item in section_assignments]
        )
        section["assignment_confidences"] = {
            str(item.get("unit_id")): item.get("confidence") for item in section_assignments
        }
        section["quality_finding_ids"] = sorted(set(finding_by_source.get(section_id, [])))
        sections.append(section)

    structure_lists = {str(item.get("list_id")): item for item in inputs.section_tree.get("lists") or []}
    relationship_lists = {str(item.get("list_id")): item for item in inputs.lists.get("nodes") or []}
    original_lists = {str(item.get("list_id")): item for item in inputs.lists.get("original_lists") or []}
    lists = []
    for list_id in sorted(set(structure_lists) | set(relationship_lists) | set(original_lists)):
        relationship_view = deepcopy(relationship_lists.get(list_id))
        items = (relationship_view or {}).get("items") or (structure_lists.get(list_id) or {}).get("items") or []
        item_ids = [str(item.get("node_id") or item.get("unit_id")) for item in items]
        lists.append({
            "list_id": list_id,
            "structure": deepcopy(structure_lists.get(list_id)),
            "relationship_view": relationship_view,
            "nested_items": _nested_items(items),
            "item_ids": item_ids,
            "relationship_ids": sorted(set(relation_by_node.get(list_id, [])) |
                                       {rel for item_id in item_ids for rel in relation_by_node.get(item_id, [])}),
            "quality_finding_ids": sorted(set(finding_by_source.get(list_id, []))),
        })

    visuals = []
    element_by_id = {item["element_id"]: item for item in elements}
    for original in inputs.visuals.get("visuals") or []:
        visual = deepcopy(original)
        visual_id = str(visual.get("visual_id"))
        element = element_by_id.get(visual_id)
        visual["original_content"] = element.get("text") if element else None
        visual["relationship_ids"] = sorted(set(relation_by_node.get(visual_id, [])))
        visual["quality_finding_ids"] = sorted(set(finding_by_source.get(visual_id, [])))
        visual["layout_provenance"] = deepcopy((node_by_id.get(visual_id) or {}).get("provenance"))
        inventory = quality_inventory.get(visual_id)
        flags, reliability, reasons = evidence_signals(visual_id, inventory)
        visual["rag_flags"] = sorted(set(visual.get("rag_flags") or []) | set(flags))
        upstream_reliability = str(visual.get("evidence_reliability") or "unassessed")
        visual["evidence_reliability"] = (
            reliability if upstream_reliability == "unassessed" and reliability != "unassessed"
            else upstream_reliability
        )
        visual["evidence_reliability_reasons"] = list(dict.fromkeys(
            [*(visual.get("evidence_reliability_reasons") or []), *reasons]
        ))
        if visual["evidence_reliability"] != "unassessed" and not visual["evidence_reliability_reasons"]:
            visual["evidence_reliability_reasons"] = [
                f"upstream visual inventory recorded evidence_reliability={visual['evidence_reliability']}"
            ]
        if visual.get("visual_type") == "table":
            visual.update(_table_structure(element, visual))
        if visual.get("visual_type") == "known_missing":
            visual["rag_flags"] = sorted(set(visual["rag_flags"]) | {"known_missing"})
            visual["evidence_reliability"] = "low"
            if not visual["evidence_reliability_reasons"]:
                visual["evidence_reliability_reasons"] = ["upstream explicitly recorded missing visual content"]
        visuals.append(visual)

    logical_tables = []
    for node in nodes:
        if node.get("node_type") != "logical_table":
            continue
        logical_id = str(node.get("node_id"))
        provenance = deepcopy(node.get("provenance") or {})
        part_ids = [str(value) for value in provenance.get("ordered_parts") or []]
        logical_tables.append({
            "logical_table_id": logical_id,
            "source_part_ids": part_ids,
            "pages": [element_by_id[value]["page"] for value in part_ids if value in element_by_id],
            "repeated_header_between_parts": provenance.get("repeated_header_between_parts") or [],
            "pair_evidence": provenance.get("pair_evidence") or [],
            "relationship_ids": sorted(set(relation_by_node.get(logical_id, []))),
            "caption_relationship_ids": [str(item.get("rel_id")) for item in relationships
                                         if item.get("type") == "caption_of"
                                         and item.get("target_id") in {logical_id, *part_ids}],
            "quality_finding_ids": sorted(set(finding_by_source.get(logical_id, [])) |
                                          {finding for part in part_ids
                                           for finding in finding_by_source.get(part, [])}),
            "provenance": provenance,
        })

    metadata = _metadata(inputs, sections, node_by_id, relationships)
    element_index = {item["element_id"]: index for index, item in enumerate(elements)}
    section_index = {str(item.get("section_id")): index for index, item in enumerate(sections)}
    visual_ids = {str(item.get("visual_id")) for item in visuals}
    indexes = {
        "element_by_id": element_index,
        "section_by_id": section_index,
        "list_by_id": {item["list_id"]: index for index, item in enumerate(lists)},
        "logical_table_by_id": {item["logical_table_id"]: index for index, item in enumerate(logical_tables)},
        "visual_by_id": {str(item.get("visual_id")): index for index, item in enumerate(visuals)},
        "known_missing_visuals": [index for index, item in enumerate(visuals)
                                  if item.get("visual_type") == "known_missing"],
        "elements_by_section": {section_id: [element_index[value] for value in
                                             _unique([str(item.get("element_id")) for item in records])
                                             if value in element_index]
                                for section_id, records in sorted(assignment_by_section.items())},
        "visuals_by_describing_source": {
            source_id: sorted(set(targets)) for source_id, targets in sorted(
                ((source_id, [str(item.get("target_id")) for item in relationships
                              if item.get("type") == "describes" and str(item.get("source_id")) == source_id
                              and str(item.get("target_id")) in visual_ids])
                 for source_id in {str(item.get("source_id")) for item in relationships
                                   if item.get("type") == "describes"}),
            )
        },
        "quality_findings_by_element": {key: sorted(set(value)) for key, value in sorted(finding_by_source.items())
                                        if key in element_index},
        "relationships_by_node": {key: sorted(set(value)) for key, value in sorted(relation_by_node.items())},
    }
    configuration = config.to_dict()
    document = {
        "schema_version": config.schema_version,
        "kind": "docprep_prepared_document",
        "document_id": f"document:{inputs.run.run_id}",
        "source_run_id": inputs.run.run_id,
        "source_document": deepcopy(inputs.run.document.get("document") or {}),
        "language": metadata["language"],
        "page_count": inputs.run.page_count,
        "metadata": metadata,
        "pages": deepcopy(inputs.run.document.get("pages") or []),
        "sections": sections,
        "elements": elements,
        "derived_spans": {
            "structure": [deepcopy(span) for item in inputs.assignments.get("source_inventory") or []
                          for span in item.get("derived_spans") or []],
            "relationships": deepcopy(inputs.lists.get("derived_spans") or []),
        },
        "lists": lists,
        "visuals": visuals,
        "logical_tables": logical_tables,
        "nodes": nodes,
        "relationships": relationships,
        "evidence_audit": (
            deepcopy(inputs.relationships.get("evidence_audit") or [])
            + deepcopy(inputs.quality_findings.get("evidence_audit") or [])
        ),
        "quality": {
            "findings": findings,
            "inventory": deepcopy(inputs.quality_inventory.get("inventory") or []),
            "summary": deepcopy(inputs.quality_summary),
            "validation": deepcopy(inputs.quality_findings.get("validation") or {}),
        },
        "indexes": indexes,
        "provenance": {
            "component": "Unified Prepared Document Builder",
            "docprep_version": __version__,
            "output_schema_version": config.schema_version,
            "processing_configuration": configuration,
            "processing_configuration_sha256": _canonical_hash(configuration),
            "input_directories": {"source": str(inputs.run.run_dir), "structure": str(inputs.structure_dir),
                                  "relations": str(inputs.relations_dir), "quality": str(inputs.quality_dir)},
            "input_sha256": deepcopy(inputs.hashes),
            "upstream_manifests": {
                "structure": deepcopy(inputs.structure_manifest),
                "relationships": deepcopy(inputs.relation_manifest),
                "quality": deepcopy(inputs.quality_manifest),
            },
            "content_policy": "verbatim upstream content; no correction, deletion, deduplication or inference",
        },
    }
    return PreparedResult(inputs.run.run_id, document, deepcopy(inputs.hashes), rehash_prepared_inputs(inputs))
