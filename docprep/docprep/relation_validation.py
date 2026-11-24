"""Structural integrity checks for relationship outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .relation_model import RelationResult


def _finding(severity: str, code: str, message: str, items: list[str] | None = None) -> dict[str, Any]:
    return {"severity": severity, "code": code, "message": message, "items": items or [],
            "item_count": len(items or [])}


def _cycle_nodes(edges: list[tuple[str, str]]) -> list[str]:
    graph: dict[str, list[str]] = defaultdict(list)
    for child, parent in edges:
        graph[parent].append(child)
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cycles.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        for child in graph.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)
    return sorted(cycles)


def validate_relationships(result: RelationResult) -> list[dict[str, Any]]:
    findings = list(result.findings)
    references = []
    self_links = []
    for relation in result.relationships:
        if relation.source_id not in result.nodes or relation.target_id not in result.nodes:
            references.append(relation.rel_id)
        if relation.source_id == relation.target_id:
            self_links.append(relation.rel_id)
    findings.append(_finding("error" if references else "ok", "references_resolve",
                             "every relationship source and target resolves" if not references else
                             "relationships contain unresolved references", references))
    findings.append(_finding("error" if self_links else "ok", "no_self_links",
                             "no relationship links a node to itself" if not self_links else
                             "self-referential relationships found", self_links))

    bad_offsets = []
    for node in result.nodes.values():
        if node.node_type == "known_missing_visual":
            valid_placeholder = (
                node.text is None and node.source_id is None and node.start is None and node.end is None
                and isinstance(node.provenance, dict) and bool(node.provenance.get("record"))
                and "missing_source_content" in node.review_flags
            )
            if not valid_placeholder:
                bad_offsets.append(node.node_id)
            continue
        derived_from_other_node = node.source_id is not None and node.source_id != node.node_id
        if not derived_from_other_node and node.start is None and node.end is None:
            continue
        source = result.nodes.get(node.source_id or "")
        if source is None or source.text is None or type(node.start) is not int or type(node.end) is not int or not (
            0 <= node.start <= node.end <= len(source.text)
        ) or source.text[node.start:node.end] != (node.text or ""):
            bad_offsets.append(node.node_id)
    findings.append(_finding("error" if bad_offsets else "ok", "derived_offsets_valid",
                             "all derived offsets map to verbatim source text" if not bad_offsets else
                             "derived nodes have invalid or non-verbatim offsets", bad_offsets))

    caption_counts = Counter(
        relation.source_id for relation in result.relationships if relation.type == "caption_of"
    )
    duplicate_captions = sorted(source for source, count in caption_counts.items() if count > 1)
    findings.append(_finding("error" if duplicate_captions else "ok", "one_primary_caption_target",
                             "each caption has at most one primary target" if not duplicate_captions else
                             "captions have multiple primary targets", duplicate_captions))

    original_visuals = {
        node.node_id for node in result.nodes.values() if node.node_type in {"figure", "table"}
        and node.source_id == node.node_id
    }
    inventory = [str(item.get("visual_id")) for item in result.visuals]
    visual_errors = sorted((original_visuals - set(inventory)) | {item for item, count in Counter(inventory).items() if count != 1})
    findings.append(_finding("error" if visual_errors else "ok", "visual_inventory_exact",
                             "every original figure/table occurs once in visuals.json" if not visual_errors else
                             "visual inventory is missing or duplicates source visuals", visual_errors))

    logical_errors = []
    for node in result.nodes.values():
        if node.node_type != "logical_table":
            continue
        parts = node.provenance.get("ordered_parts") or []
        pages = [result.nodes[part].page for part in parts if part in result.nodes]
        if len(parts) != len(set(parts)) or len(pages) != len(parts) or pages != sorted(pages):
            logical_errors.append(node.node_id)
    findings.append(_finding("error" if logical_errors else "ok", "logical_tables_ordered",
                             "logical table parts are ordered and unique" if not logical_errors else
                             "logical table parts are invalid", logical_errors))

    list_edges = [(relation.source_id, relation.target_id) for relation in result.relationships
                  if relation.type == "list_nesting"]
    cycles = _cycle_nodes(list_edges)
    findings.append(_finding("error" if cycles else "ok", "list_relationships_acyclic",
                             "list nesting is acyclic" if not cycles else "list nesting contains cycles", cycles))

    state_errors = []
    for item in result.visuals:
        node = result.nodes.get(str(item.get("visual_id")))
        if node and item.get("validation_state") != node.provenance.get("validation_state"):
            state_errors.append(node.node_id)
    findings.append(_finding("error" if state_errors else "ok", "validation_states_preserved",
                             "source validation states are preserved" if not state_errors else
                             "source validation state changed", state_errors))

    inputs_unchanged = result.source_hashes_before == result.source_hashes_after
    structure_unchanged = result.structure_hashes_before == result.structure_hashes_after
    findings.append(_finding("error" if not inputs_unchanged else "ok", "source_inputs_unchanged",
                             "Nass source inputs are unchanged" if inputs_unchanged else
                             "one or more Nass source inputs changed"))
    findings.append(_finding("error" if not structure_unchanged else "ok", "structure_inputs_unchanged",
                             "Document Structure outputs are unchanged" if structure_unchanged else
                             "one or more Document Structure outputs changed"))

    ambiguous_unflagged = [
        relation.rel_id for relation in result.relationships
        if relation.conflicts and not relation.review_recommended
    ]
    findings.append(_finding("error" if ambiguous_unflagged else "ok", "ambiguities_flagged",
                             "ambiguous relationships carry conflicts and review flags" if not ambiguous_unflagged else
                             "ambiguous relationships lack review flags", ambiguous_unflagged))
    return findings


def validation_summary(findings: list[dict[str, Any]]) -> dict[str, int]:
    return {severity: sum(1 for item in findings if item.get("severity") == severity)
            for severity in ("ok", "warning", "error")}

