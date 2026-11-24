"""Component 4 synthetic contract tests. Written for later execution; NOT executed by the implementer."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from docprep.inputs import InputError
from docprep.output import OutputExistsError
from docprep.prepared_builder import build_prepared_document
from docprep.prepared_config import PreparedConfig
from docprep.prepared_inputs import load_prepared_inputs
from docprep.prepared_output import write_prepared_outputs
from docprep.prepared_reports import render_prepared_html
from docprep.prepared_validation import validate_prepared

from .helpers import el, write_run


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _fixture(tmp_path: Path):
    elements = [
        el("h", 1, 0, "عنوان تجريبي", type="heading", bbox=[10, 10, 200, 30]),
        el("p", 1, 1, "فقرة عربية exact 42%", bbox=[10, 40, 300, 80]),
        el("li", 1, 2, "1. أصل", type="text", bbox=[20, 90, 200, 110]),
        el("child", 1, 3, "أ. فرع", type="text", bbox=[40, 115, 220, 135]),
        el("cap", 1, 4, "Figure 1: Synthetic", bbox=[10, 140, 250, 160]),
        el("fig", 1, 5, "42%", type="figure", bbox=[10, 170, 300, 350], state="needs_review"),
        el("t1", 1, 6, "| A |\n| 1 |", type="table", bbox=[10, 360, 300, 500]),
        el("t2", 2, 0, "| A |\n| 2 |", type="table", bbox=[10, 40, 300, 180]),
        el("comp", 2, 1, "panel", type="figure", bbox=[320, 40, 500, 180], state="included_in_composite"),
    ]
    run = write_run(tmp_path, "prepared_synthetic", elements, page_count=2, directions={1: "rtl", 2: "ltr"})
    structure = tmp_path / "structure"
    relations = tmp_path / "relations"
    quality = tmp_path / "quality"
    structure.mkdir(); relations.mkdir(); quality.mkdir()
    source_inventory = []
    assignments = []
    for index, item in enumerate(elements):
        span = []
        if item["id"] == "p":
            span = [{"span_id": "p_span", "element_id": "p", "start": 0, "end": 9,
                     "text": item["content"][0:9], "cut_reason": "synthetic", "role_hint": "paragraph",
                     "support": "text", "basis": ["synthetic"], "geometry": None,
                     "uncertain": False, "notes": []}]
        source_inventory.append({"element_id": item["id"], "page": item["page"], "type": item["type"],
                                 "reading_order": item["reading_order"], "document_order": index,
                                 "bbox_pt": item.get("bbox_pt"), "validation_state": item["validation_state"],
                                 "source": item["source"], "text": item["content"],
                                 "represented_by": [item["id"]], "derived_spans": span})
        section = "s0" if item["id"] == "h" else "s1"
        role = "heading" if item["id"] == "h" else "list_item" if item["id"] in {"li", "child"} else \
            item["type"] if item["type"] in {"figure", "table"} else "paragraph"
        assignments.append({"unit_id": item["id"], "element_id": item["id"], "span_id": None,
                            "page": item["page"], "order": [index, 0], "role": role,
                            "original_type": item["type"], "section_id": section,
                            "list_id": "L1" if item["id"] in {"li", "child"} else None,
                            "confidence": "high", "reasons": ["synthetic"], "supporting": [],
                            "conflicting": [], "deterministic": True, "review_recommended": False,
                            "validation_state": item["validation_state"], "derived_span_uncertain": None,
                            "role_uncertain": False, "semantic_check": None, "text": item["content"],
                            "bbox_pt": item.get("bbox_pt"), "source_run_id": "prepared_synthetic",
                            "typography": None, "layout": None})
    sections = [
        {"section_id": "s0", "title": "عنوان تجريبي", "title_parts": ["عنوان تجريبي"], "level": 0,
         "raw_level": 0, "level_name": "document title", "level_notes": [], "parent": None,
         "children": ["s1"], "heading_candidate_id": "c0", "heading_unit_ids": ["h"],
         "heading_element_ids": ["h"], "page_start": 1, "page_end": 1, "content_unit_ids": [],
         "list_nodes": [], "confidence": {"heading_detection": "high", "heading_level": "high"},
         "review_recommended": False, "evidence": {}, "flags": []},
        {"section_id": "s1", "title": "Body", "title_parts": ["Body"], "level": 1, "raw_level": 1,
         "level_name": "major section", "level_notes": [], "parent": "s0", "children": [],
         "heading_candidate_id": "c1", "heading_unit_ids": [], "heading_element_ids": [],
         "page_start": 1, "page_end": 2, "content_unit_ids": [item["id"] for item in elements[1:]],
         "list_nodes": ["L1"], "confidence": {"heading_detection": "medium", "heading_level": "medium"},
         "review_recommended": True, "evidence": {}, "flags": ["synthetic body"]},
    ]
    structure_list = {"list_id": "L1", "section_id": "s1", "intro_unit_id": "p",
                      "items": [{"unit_id": "li", "element_id": "li", "span_id": None, "start": None,
                                 "end": None, "page": 1, "marker": "1.", "marker_position": "start",
                                 "continuation_unit_ids": []},
                                {"unit_id": "child", "element_id": "child", "span_id": None, "start": None,
                                 "end": None, "page": 1, "marker": "أ.", "marker_position": "start",
                                 "continuation_unit_ids": []}],
                      "unit_ids": ["li", "child"], "possible_continuations": [], "pages": [1],
                      "page_start": 1, "page_end": 1, "crosses_pages": False}
    _write(structure / "section_tree.json", {"schema_version": 1, "kind": "docprep_section_tree",
                                              "docprep_version": "0.1.0", "run_id": "prepared_synthetic",
                                              "source_run_dir": str(run), "document": {}, "outline": [],
                                              "sections": sections, "lists": [structure_list], "unassigned": [],
                                              "heading_candidates": [], "style_levels": [],
                                              "layout_heading_diagnostics": []})
    _write(structure / "assignments.json", {"schema_version": 1, "kind": "docprep_assignments",
                                             "docprep_version": "0.1.0", "run_id": "prepared_synthetic",
                                             "source_inventory": source_inventory, "assignments": assignments})
    _write(structure / "manifest.json", {"schema_version": 1, "kind": "docprep_manifest",
                                          "component": "Document Structure Builder", "docprep_version": "0.1.0",
                                          "source_run_id": "prepared_synthetic", "inputs_unchanged": True})
    nodes = [{"node_id": item["id"], "node_type": item["type"], "page": item["page"],
              "text": item["content"], "source_id": item["id"], "start": None, "end": None,
              "provenance": {"validation_state": item["validation_state"]}, "review_flags": []}
             for item in elements]
    nodes += [
        {"node_id": "s0", "node_type": "section", "page": 1, "text": "عنوان تجريبي",
         "source_id": None, "start": None, "end": None, "provenance": {}, "review_flags": []},
        {"node_id": "s1", "node_type": "section", "page": 1, "text": "Body", "source_id": None,
         "start": None, "end": None, "provenance": {}, "review_flags": []},
        {"node_id": "L1", "node_type": "list", "page": 1, "text": None, "source_id": None,
         "start": None, "end": None, "provenance": {"section_id": "s1"}, "review_flags": []},
        {"node_id": "logical", "node_type": "logical_table", "page": 1, "text": None, "source_id": None,
         "start": None, "end": None,
         "provenance": {"ordered_parts": ["t1", "t2"], "repeated_header_between_parts": [True],
                        "pair_evidence": [{"matching_column_count": True}]}, "review_flags": []},
    ]
    def rel(rel_id, kind, source, target, evidence=None, confidence="high"):
        return {"rel_id": rel_id, "type": kind, "source_id": source, "target_id": target,
                "evidence": evidence or {}, "confidence": confidence, "method": "synthetic", "conflicts": [],
                "source_type": None, "target_type": None, "source_page": 1, "target_page": 1,
                "review_recommended": confidence != "high", "limitations": [],
                "source_validation_state": None, "target_validation_state": None}
    relationships = [
        rel("r_cap", "caption_of", "cap", "fig"),
        rel("r_desc", "describes", "p", "fig", {"numeric_matches": [{"normalized_value": "42"}]}),
        rel("r_t1", "table_continuation", "t1", "logical", {"part_index": 0}),
        rel("r_t2", "table_continuation", "t2", "logical", {"part_index": 1}),
        rel("r_nested", "list_nesting", "child", "li"),
        rel("r_merged", "merged_list_items", "child", "li"),
        rel("r_intro", "introduces_list", "p", "L1"),
        rel("r_comp", "composite_part", "comp", "fig", confidence="medium"),
    ]
    _write(relations / "relationships.json", {"schema_version": 1, "kind": "docprep_content_relationships",
                                               "docprep_version": "0.1.0",
                                               "source_run_id": "prepared_synthetic", "source_run_dir": str(run),
                                               "structure_output_dir": str(structure), "nodes": nodes,
                                               "relationships": relationships,
                                               "validation": {"summary": {}, "findings": []}})
    visuals = [{"visual_id": item["id"], "visual_type": item["type"], "page": item["page"],
                "section_id": "s1", "validation_state": item["validation_state"],
                "caption": "cap" if item["id"] == "fig" else None, "internal_title": None,
                "description_paragraphs": ["p"] if item["id"] == "fig" else [],
                "matched_numeric_evidence": [{"normalized_value": "42"}] if item["id"] == "fig" else [],
                "leaked_text_overlaps": [], "composite": {"parent": "fig" if item["id"] == "comp" else None,
                                                            "parts": ["comp"] if item["id"] == "fig" else [],
                                                            "unresolved": False},
                "logical_table_id": "logical" if item["id"] in {"t1", "t2"} else None,
                "review_flags": []} for item in elements if item["type"] in {"figure", "table"}]
    _write(relations / "visuals.json", {"schema_version": 1, "kind": "docprep_visual_inventory",
                                         "source_run_id": "prepared_synthetic", "visuals": visuals})
    list_node = {"list_id": "L1", "section_id": "s1", "intro_unit_id": "p",
                 "items": [{"node_id": "li", "element_id": "li", "page": 1, "order": 0, "marker": "1.",
                            "marker_family": "number", "indent": 0, "level": 0, "parent_id": None,
                            "text": "1. أصل", "evidence": ["synthetic"]},
                           {"node_id": "child", "element_id": "child", "page": 1, "order": 1,
                            "marker": "أ.", "marker_family": "letter", "indent": 1, "level": 1,
                            "parent_id": "li", "text": "أ. فرع", "evidence": ["synthetic"]}],
                 "pages": [1], "crosses_pages": False, "possible_continuations": []}
    _write(relations / "lists.json", {"schema_version": 1, "kind": "docprep_list_relationships",
                                       "source_run_id": "prepared_synthetic", "original_lists": [structure_list],
                                       "nodes": [list_node], "relationships": ["r_nested", "r_merged", "r_intro"],
                                       "derived_spans": [{"span_id": "list_span", "parent_element_id": "li",
                                                          "start": 0, "end": 6, "text": "1. أصل",
                                                          "page": 1, "marker": "1.",
                                                          "marker_position": "start", "confidence": "high"}],
                                       "unresolved_ambiguities": []})
    _write(relations / "manifest.json", {"schema_version": 1, "kind": "docprep_relationship_manifest",
                                          "component": "Content Relationship Builder", "docprep_version": "0.1.0",
                                          "source_run_id": "prepared_synthetic", "inputs_unchanged": True})
    finding = {"finding_id": "q1", "category": "figure_chart_quality", "subtype": "synthetic_review",
               "severity": "medium", "confidence": "high", "source_ids": ["fig"], "page_numbers": [1],
               "evidence": {"saved": True}, "explanation": "synthetic review evidence",
               "suggested_review_action": "review", "status": "needs_review", "offsets": [],
               "related_ids": [], "relationship_ids": ["r_desc"], "limitations": [], "method": "synthetic",
               "priority_score": 55, "priority_rank": 1}
    inventory = [{"source_id": item["id"], "page": item["page"], "element_type": item["type"],
                  "validation_state": item["validation_state"], "structure_assignment": assignments[index],
                  "relationship_ids": [], "categories_attempted": [], "skipped_checks": {},
                  "geometry_reliability": "precise", "finding_ids": ["q1"] if item["id"] == "fig" else []}
                 for index, item in enumerate(elements)]
    _write(quality / "quality_findings.json", {"schema_version": 1, "kind": "docprep_quality_findings",
                                                "docprep_version": "0.1.0", "source_run_id": "prepared_synthetic",
                                                "findings": [finding], "review_sources": {},
                                                "validation": {"summary": {"ok": 1, "error": 0}, "checks": []}})
    _write(quality / "quality_summary.json", {"schema_version": 1, "kind": "docprep_quality_summary",
                                               "source_run_id": "prepared_synthetic",
                                               "document": {"finding_count": 1}, "pages": {}, "sections": {},
                                               "element_types": {}, "inspection_coverage": {},
                                               "validation_summary": {"ok": 1, "error": 0}})
    _write(quality / "quality_inventory.json", {"schema_version": 1, "kind": "docprep_quality_inventory",
                                                 "source_run_id": "prepared_synthetic", "inventory": inventory})
    _write(quality / "quality_manifest.json", {"schema_version": 1, "kind": "docprep_quality_manifest",
                                                "component": "Document Quality & Validation Layer",
                                                "docprep_version": "0.1.0", "source_run_id": "prepared_synthetic",
                                                "inputs_unchanged": True})
    return run, structure, relations, quality, elements


def _build(tmp_path: Path):
    run, structure, relations, quality, elements = _fixture(tmp_path)
    inputs = load_prepared_inputs(run, structure, relations, quality)
    result = build_prepared_document(inputs, PreparedConfig(read_only_files=False))
    result.validation = validate_prepared(result, inputs)
    return result, inputs, run, structure, relations, quality, elements


def _errors(result):
    return [item for item in result.validation if item["severity"] == "error"]


def test_complete_source_inventory_and_ids(tmp_path):
    result, *_rest, elements = _build(tmp_path)
    assert [item["element_id"] for item in result.document["elements"]] == [item["id"] for item in elements]


def test_source_strings_are_byte_identical_and_arabic_is_preserved(tmp_path):
    result, *_rest, elements = _build(tmp_path)
    expected = {item["id"]: item["content"] for item in elements}
    assert {item["element_id"]: item["text"] for item in result.document["elements"]} == expected
    assert result.document["elements"][0]["text"].encode("utf-8") == "عنوان تجريبي".encode()


def test_reading_order_and_section_hierarchy(tmp_path):
    result, *_ = _build(tmp_path)
    assert [item["reading_order"] for item in result.document["elements"][:3]] == [0, 1, 2]
    sections = {item["section_id"]: item for item in result.document["sections"]}
    assert sections["s0"]["children"] == ["s1"] and sections["s1"]["parent"] == "s0"


def test_nested_lists_and_merged_list_relationships(tmp_path):
    result, *_ = _build(tmp_path)
    prepared_list = result.document["lists"][0]
    assert prepared_list["nested_items"][0]["children"][0]["node_id"] == "child"
    assert any(item["type"] == "merged_list_items" for item in result.document["relationships"])


def test_derived_spans_preserve_exact_offsets(tmp_path):
    result, *_ = _build(tmp_path)
    assert not [item for item in result.validation if item["code"] == "derived_span_offsets_valid"
                and item["severity"] == "error"]


def test_visual_caption_description_and_numeric_evidence(tmp_path):
    result, *_ = _build(tmp_path)
    visual = next(item for item in result.document["visuals"] if item["visual_id"] == "fig")
    assert visual["caption"] == "cap" and visual["description_paragraphs"] == ["p"]
    assert visual["matched_numeric_evidence"] == [{"normalized_value": "42"}]


def test_unlinked_visual_is_retained(tmp_path):
    result, *_ = _build(tmp_path)
    assert any(item["visual_id"] == "comp" for item in result.document["visuals"])


def test_multi_page_logical_table_has_one_ordered_node(tmp_path):
    result, *_ = _build(tmp_path)
    assert len(result.document["logical_tables"]) == 1
    assert result.document["logical_tables"][0]["source_part_ids"] == ["t1", "t2"]
    assert result.document["logical_tables"][0]["pages"] == [1, 2]


def test_quality_finding_attaches_to_element(tmp_path):
    result, *_ = _build(tmp_path)
    figure = next(item for item in result.document["elements"] if item["element_id"] == "fig")
    assert figure["quality_finding_ids"] == ["q1"]


def test_validation_states_needs_review_and_composite_are_preserved(tmp_path):
    result, *_ = _build(tmp_path)
    states = {item["element_id"]: item["validation_state"] for item in result.document["elements"]}
    assert states["fig"] == "needs_review" and states["comp"] == "included_in_composite"


def test_unknown_metadata_is_explicit_not_inferred(tmp_path):
    result, *_ = _build(tmp_path)
    assert result.document["metadata"]["author"] is None
    assert result.document["metadata"]["language"] is None


def test_missing_optional_source_fields_are_accepted(tmp_path):
    run, structure, relations, quality, *_ = _fixture(tmp_path)
    document = json.loads((run / "document.json").read_text(encoding="utf-8"))
    document["elements"][0].pop("model", None); document["elements"][0].pop("provenance", None)
    _write(run / "document.json", document)
    inputs = load_prepared_inputs(run, structure, relations, quality)
    assert build_prepared_document(inputs).document["elements"][0]["source_extraction"]["source"]


def test_incompatible_upstream_schema_version_fails_clearly(tmp_path):
    run, structure, relations, quality, *_ = _fixture(tmp_path)
    tree = json.loads((structure / "section_tree.json").read_text(encoding="utf-8")); tree["schema_version"] = 2
    _write(structure / "section_tree.json", tree)
    with pytest.raises(InputError, match="incompatible section_tree schema_version"):
        load_prepared_inputs(run, structure, relations, quality)


def test_mismatched_recorded_lineage_fails_clearly(tmp_path):
    run, structure, relations, quality, *_ = _fixture(tmp_path)
    manifest = json.loads((quality / "quality_manifest.json").read_text(encoding="utf-8"))
    manifest["relationship_output_dir"] = str(tmp_path / "different_relationship_output")
    _write(quality / "quality_manifest.json", manifest)
    with pytest.raises(InputError, match="Quality Relationships lineage mismatch"):
        load_prepared_inputs(run, structure, relations, quality)


def test_broken_section_reference_is_reported(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    result.document["sections"][1]["parent"] = "missing"
    result.validation = validate_prepared(result, inputs)
    assert "section_references_resolve" in {item["code"] for item in _errors(result)}


def test_broken_relationship_reference_is_reported(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    result.document["relationships"][0]["target_id"] = "missing"
    result.validation = validate_prepared(result, inputs)
    assert "relationship_references_resolve" in {item["code"] for item in _errors(result)}


def test_broken_quality_reference_is_reported(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    result.document["quality"]["findings"][0]["source_ids"] = ["missing"]
    result.validation = validate_prepared(result, inputs)
    assert "quality_references_resolve" in {item["code"] for item in _errors(result)}


def test_invalid_span_offset_is_reported(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    result.document["derived_spans"]["structure"][0]["end"] = 999
    result.validation = validate_prepared(result, inputs)
    assert "derived_span_offsets_valid" in {item["code"] for item in _errors(result)}


def test_duplicate_ids_are_reported(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    result.document["relationships"].append(deepcopy(result.document["relationships"][0]))
    result.validation = validate_prepared(result, inputs)
    assert "ids_unique_within_namespaces" in {item["code"] for item in _errors(result)}


def test_source_inputs_remain_immutable(tmp_path):
    result, inputs, run, structure, relations, quality, *_ = _build(tmp_path)
    paths = [*run.glob("*.json"), *structure.glob("*.json"), *relations.glob("*.json"), *quality.glob("*.json")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write_prepared_outputs(result, inputs, out_root=tmp_path / "out", stamp="one",
                           config=PreparedConfig(read_only_files=False))
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_atomic_write_once_output(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    kwargs = {"out_root": tmp_path / "out", "stamp": "same", "config": PreparedConfig(read_only_files=False)}
    write_prepared_outputs(result, inputs, **kwargs)
    with pytest.raises(OutputExistsError):
        write_prepared_outputs(result, inputs, **kwargs)


def test_required_output_files_are_complete(tmp_path):
    result, inputs, *_ = _build(tmp_path)
    output, _ = write_prepared_outputs(result, inputs, out_root=tmp_path / "out", stamp="complete",
                                       config=PreparedConfig(read_only_files=False))
    assert {path.name for path in output.iterdir()} == {
        "prepared_document.json", "prepared_document_manifest.json", "prepared_document_validation.json",
        "prepared_document_report.md", "prepared_document_report.html",
    }


def test_machine_schema_declares_every_required_top_level_field(tmp_path):
    result, *_ = _build(tmp_path)
    schema_path = Path(__file__).parents[2] / "schemas" / "prepared_document.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert set(schema["required"]) == set(result.document)
    assert result.document["schema_version"] == schema["properties"]["schema_version"]["const"]


def test_html_escapes_source_derived_text(tmp_path):
    result, *_ = _build(tmp_path)
    result.document["sections"][0]["title"] = "<script>alert(1)</script>"
    html = render_prepared_html(result)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_logical_contents_are_deterministic(tmp_path):
    _result, inputs, *_ = _build(tmp_path)
    first = build_prepared_document(inputs, PreparedConfig(read_only_files=False)).document
    second = build_prepared_document(inputs, PreparedConfig(read_only_files=False)).document
    assert first == second


def test_navigation_indexes_resolve_expected_objects(tmp_path):
    result, *_ = _build(tmp_path)
    indexes = result.document["indexes"]
    assert result.document["elements"][indexes["element_by_id"]["fig"]]["element_id"] == "fig"
    assert result.document["sections"][indexes["section_by_id"]["s1"]]["section_id"] == "s1"
    assert indexes["visuals_by_describing_source"]["p"] == ["fig"]


def test_valid_fixture_has_no_integrity_errors(tmp_path):
    result, *_ = _build(tmp_path)
    assert not _errors(result)


def test_schema_1_1_adds_evidence_audit_without_changing_source_content(tmp_path):
    result, *_rest, elements = _build(tmp_path)
    assert result.document["schema_version"] == "1.1.0"
    assert result.document["evidence_audit"] == []
    assert [item["element_id"] for item in result.document["elements"]] == [item["id"] for item in elements]
    assert [item["text"] for item in result.document["elements"]] == [item["content"] for item in elements]


def test_rag_flags_and_reliability_propagate_without_automatic_high_rating(tmp_path):
    result, *_ = _build(tmp_path)
    elements = {item["element_id"]: item for item in result.document["elements"]}
    assert elements["fig"]["evidence_reliability"] == "medium"
    assert "needs_review" in elements["fig"]["rag_flags"]
    assert elements["h"]["evidence_reliability"] == "unassessed"
    assert elements["h"]["rag_flags"] == []


def test_relationship_only_review_remains_attached_without_content_downgrade(tmp_path):
    _result, inputs, *_ = _build(tmp_path)
    finding = {
        "finding_id": "q_relationship", "category": "relationship_quality",
        "subtype": "uncertain_existing_relationship", "severity": "medium", "confidence": "high",
        "source_ids": ["t1"], "page_numbers": [1], "evidence": {"relationship_id": "r_t1"},
        "explanation": "relationship evidence needs review", "suggested_review_action": "review link",
        "status": "needs_review", "offsets": [], "related_ids": [], "relationship_ids": ["r_t1"],
        "limitations": [], "method": "synthetic",
    }
    inputs.quality_findings["findings"].append(finding)
    document = build_prepared_document(inputs, PreparedConfig(read_only_files=False)).document
    element = next(item for item in document["elements"] if item["element_id"] == "t1")
    visual = next(item for item in document["visuals"] if item["visual_id"] == "t1")
    assert element["rag_flags"] == [] and element["evidence_reliability"] == "unassessed"
    assert visual["rag_flags"] == [] and visual["evidence_reliability"] == "unassessed"
    assert "q_relationship" in element["quality_finding_ids"]
    assert "q_relationship" in visual["quality_finding_ids"]
    assert any(item["rel_id"] == "r_t1" for item in document["relationships"])
    assert visual["original_content"] == "| A |\n| 1 |"


def test_prepared_historical_downgrade_without_reason_gets_traceable_origin(tmp_path):
    _result, inputs, *_ = _build(tmp_path)
    inventory = next(item for item in inputs.quality_inventory["inventory"] if item["source_id"] == "p")
    inventory.update({"rag_flags": ["needs_review"], "evidence_reliability": "medium",
                      "evidence_reliability_reasons": []})
    document = build_prepared_document(inputs, PreparedConfig(read_only_files=False)).document
    element = next(item for item in document["elements"] if item["element_id"] == "p")
    assert element["evidence_reliability"] == "medium"
    assert element["evidence_reliability_reasons"]
    assert "quality inventory recorded rag_flags" in element["evidence_reliability_reasons"][0]


def test_known_missing_visual_is_indexed_and_traceable_to_quality_evidence(tmp_path):
    _result, inputs, *_ = _build(tmp_path)
    missing_id = "known_missing_synthetic"
    inputs.relationships["nodes"].append({
        "node_id": "document:prepared_synthetic", "node_type": "document", "page": None, "text": None,
        "source_id": None, "start": None, "end": None, "provenance": {}, "review_flags": [],
    })
    inputs.relationships["nodes"].append({
        "node_id": missing_id, "node_type": "known_missing_visual", "page": 2, "text": None,
        "source_id": None, "start": None, "end": None,
        "provenance": {"source": "execution_manifest.skipped_visuals", "record": {"status": "skipped"}},
        "review_flags": ["missing_source_content"],
    })
    inputs.relationships["relationships"].append({
        "rel_id": "r_missing", "type": "known_missing", "source_id": "document:prepared_synthetic",
        "target_id": missing_id, "evidence": {"page": 2, "reason": "explicit saved image skip"},
        "confidence": "high", "method": "explicit_saved_metadata", "conflicts": [],
        "source_type": "document", "target_type": "known_missing_visual", "source_page": None,
        "target_page": 2, "review_recommended": True, "limitations": [],
        "source_validation_state": None, "target_validation_state": None,
    })
    inputs.visuals["visuals"].append({
        "visual_id": missing_id, "visual_type": "known_missing", "page": 2, "bbox_pt": None,
        "section_id": None, "validation_state": None, "reason": "explicit saved image skip",
        "source_record": {"status": "skipped"}, "source_record_path": "execution_manifest.skipped_visuals",
        "caption": None, "internal_title": None, "description_paragraphs": [],
        "matched_numeric_evidence": [], "leaked_text_overlaps": [],
        "composite": {"parent": None, "parts": [], "unresolved": False}, "logical_table_id": None,
        "review_flags": ["known_missing"], "rag_flags": ["known_missing"],
        "evidence_reliability": "low", "evidence_reliability_reasons": ["explicit upstream skip"],
    })
    inputs.quality_findings["findings"].append({
        "finding_id": "q_missing", "category": "source_extraction", "subtype": "recorded_known_missing",
        "severity": "medium", "confidence": "high", "source_ids": [missing_id], "page_numbers": [2],
        "evidence": {"relationship_id": "r_missing"}, "explanation": "explicit missing content",
        "suggested_review_action": "preserve placeholder", "status": "needs_review", "offsets": [],
        "related_ids": [], "relationship_ids": ["r_missing"], "limitations": [], "method": "synthetic",
    })
    document = build_prepared_document(inputs, PreparedConfig(read_only_files=False)).document
    index = document["indexes"]["visual_by_id"][missing_id]
    visual = document["visuals"][index]
    assert visual["visual_type"] == "known_missing" and visual["original_content"] is None
    assert visual["rag_flags"] == ["known_missing", "needs_review"]
    assert index in document["indexes"]["known_missing_visuals"]
