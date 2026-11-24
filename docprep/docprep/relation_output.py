"""Write-once serialization for Component 2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .output import write_outputs
from .relation_model import RelationResult
from .relation_reports import render_relations_html, render_relations_markdown
from .relation_validation import validate_relationships, validation_summary

DEFAULT_RELATION_OUTPUT_ROOT = Path(r"D:\agenticdoc_e2\results\document_relationships")


def relationship_json(result: RelationResult, findings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "docprep_content_relationships",
        "docprep_version": __version__,
        "source_run_id": result.run_id,
        "source_run_dir": result.source_run_dir,
        "structure_output_dir": result.structure_dir,
        "nodes": [result.nodes[key].to_dict() for key in sorted(result.nodes)],
        "relationships": result.relationship_dicts(),
        "evidence_audit": result.evidence_audit,
        "validation": {"summary": validation_summary(findings), "findings": findings},
    }


def write_relation_outputs(result: RelationResult, *, out_root: Path | str = DEFAULT_RELATION_OUTPUT_ROOT,
                           stamp: str | None = None, read_only: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    findings = validate_relationships(result)
    rel_data = relationship_json(result, findings)
    visuals_data = {
        "schema_version": 1, "kind": "docprep_visual_inventory", "source_run_id": result.run_id,
        "visuals": result.visuals,
        "evidence_audit": [item for item in result.evidence_audit if item.get("candidate_type") == "visual_title"],
    }
    lists_data = {
        "schema_version": 1, "kind": "docprep_list_relationships", "source_run_id": result.run_id,
        **result.lists,
    }
    contents = {
        "relationships.json": json.dumps(rel_data, ensure_ascii=False, indent=2),
        "visuals.json": json.dumps(visuals_data, ensure_ascii=False, indent=2),
        "lists.json": json.dumps(lists_data, ensure_ascii=False, indent=2),
        "relations_report.md": render_relations_markdown(result, findings),
        "relations_report.html": render_relations_html(result, findings),
    }

    def manifest(output_hashes: dict[str, str]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "docprep_relationship_manifest",
            "component": "Content Relationship Builder",
            "docprep_version": __version__,
            "source_run_id": result.run_id,
            "source_run_dir": result.source_run_dir,
            "structure_output_dir": result.structure_dir,
            "source_hashes_before": result.source_hashes_before,
            "source_hashes_after": result.source_hashes_after,
            "structure_hashes_before": result.structure_hashes_before,
            "structure_hashes_after": result.structure_hashes_after,
            "inputs_unchanged": result.source_hashes_before == result.source_hashes_after
            and result.structure_hashes_before == result.structure_hashes_after,
            "output_hashes": output_hashes,
            "validation_summary": validation_summary(findings),
            "warnings": result.warnings,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "scope": "content relationships only; no OCR, rewriting, chunking, embeddings, retrieval or inference",
            "determinism": "logical identifiers and relationships are source-derived; folder timestamp is not semantic",
        }

    target = write_outputs(out_root, result.run_id, contents, manifest, stamp=stamp, read_only=read_only)
    return target, findings

