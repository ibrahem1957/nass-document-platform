"""Synthetic specification for Component 3. Written for later execution; NOT executed by the implementer."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from docprep.evaluate_quality import evaluate_review, make_template
from docprep.output import OutputExistsError
from docprep.quality_builder import build_quality
from docprep.quality_config import QualityConfig
from docprep.quality_inputs import load_quality_inputs
from docprep.quality_output import write_quality_outputs
from docprep.quality_reports import render_quality_html
from docprep.quality_validation import validate_quality

from .helpers import el, write_run


def _structure(root: Path, run_id: str, elements: list[dict], *, assignments: dict[str, dict] | None = None,
               lists: list[dict] | None = None, heading_candidates: list[dict] | None = None,
               sections: list[dict] | None = None) -> Path:
    folder = root / "structure"
    folder.mkdir()
    overrides = assignments or {}
    inventory = []
    records = []
    for index, item in enumerate(elements):
        inventory.append({"element_id": item["id"], "page": item["page"], "type": item["type"],
                          "validation_state": item.get("validation_state"), "text": item.get("content"),
                          "bbox_pt": item.get("bbox_pt"), "represented_by": [item["id"]], "derived_spans": []})
        record = {"unit_id": item["id"], "element_id": item["id"], "page": item["page"],
                  "order": [index, 0], "role": "figure" if item["type"] == "figure" else
                  "table" if item["type"] == "table" else "paragraph", "section_id": "s1",
                  "confidence": "high", "review_recommended": False, "text": item.get("content")}
        record.update(overrides.get(item["id"], {}))
        records.append(record)
    (folder / "assignments.json").write_text(json.dumps({"run_id": run_id, "source_inventory": inventory,
                                                          "assignments": records}), encoding="utf-8")
    (folder / "section_tree.json").write_text(json.dumps({
        "run_id": run_id, "sections": sections or [{"section_id": "s1", "title": "Synthetic section",
                                                       "page_start": 1, "page_end": max(e["page"] for e in elements),
                                                       "heading_element_ids": [], "flags": []}],
        "lists": lists or [], "heading_candidates": heading_candidates or [],
        "layout_heading_diagnostics": [],
    }), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({"source_run_id": run_id}), encoding="utf-8")
    return folder


def _relations(root: Path, run_id: str, elements: list[dict], *, relationships: list[dict] | None = None,
               findings: list[dict] | None = None, logical_nodes: list[dict] | None = None,
               visuals: list[dict] | None = None, lists: list[dict] | None = None) -> Path:
    folder = root / "relations"
    folder.mkdir()
    nodes = [{"node_id": item["id"], "node_type": item["type"], "page": item["page"],
              "text": item.get("content") or "", "source_id": item["id"], "start": None, "end": None,
              "provenance": {"validation_state": item.get("validation_state")}, "review_flags": []}
             for item in elements]
    nodes.extend(logical_nodes or [])
    (folder / "relationships.json").write_text(json.dumps({
        "source_run_id": run_id, "nodes": nodes, "relationships": relationships or [],
        "validation": {"summary": {}, "findings": findings or []},
    }), encoding="utf-8")
    visual_data = visuals if visuals is not None else [
        {"visual_id": item["id"], "visual_type": item["type"], "page": item["page"],
         "validation_state": item.get("validation_state"), "composite": {"unresolved": False}}
        for item in elements if item["type"] in {"figure", "table"}
    ]
    (folder / "visuals.json").write_text(json.dumps({"source_run_id": run_id, "visuals": visual_data}),
                                         encoding="utf-8")
    (folder / "lists.json").write_text(json.dumps({"source_run_id": run_id, "nodes": lists or []}),
                                       encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({"source_run_id": run_id}), encoding="utf-8")
    return folder


def _build(tmp_path: Path, elements: list[dict], *, validation: dict | None = None,
           manifest_updates: dict | None = None, assignments: dict[str, dict] | None = None,
           structure_lists: list[dict] | None = None, heading_candidates: list[dict] | None = None,
           sections: list[dict] | None = None, relationships: list[dict] | None = None,
           relationship_findings: list[dict] | None = None, logical_nodes: list[dict] | None = None,
           visuals: list[dict] | None = None):
    run_id = "synthetic_quality"
    directions = {item["page"]: "rtl" for item in elements if any("\u0600" <= char <= "\u06ff"
                                                                   for char in str(item.get("content") or ""))}
    run = write_run(tmp_path, run_id, elements, validation=validation, directions=directions)
    if manifest_updates:
        path = run / "execution_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(manifest_updates)
        path.write_text(json.dumps(data), encoding="utf-8")
    structure = _structure(tmp_path, run_id, elements, assignments=assignments, lists=structure_lists,
                           heading_candidates=heading_candidates, sections=sections)
    relations = _relations(tmp_path, run_id, elements, relationships=relationships,
                           findings=relationship_findings, logical_nodes=logical_nodes, visuals=visuals)
    inputs = load_quality_inputs(run, structure, relations)
    result = build_quality(inputs, QualityConfig(read_only_files=False))
    result.validation = validate_quality(result, inputs)
    return result, inputs, run, structure, relations


def _find(result, subtype: str):
    return [item for item in result.findings if item.subtype == subtype]


def test_needs_review_propagation(tmp_path):
    result, *_ = _build(tmp_path, [el("f", 1, 0, "chart", type="figure", state="needs_review")])
    finding = _find(result, "nass_needs_review")[0]
    assert finding.source_ids == ["f"] and finding.status == "needs_review"


def test_low_recorded_ocr_confidence(tmp_path):
    result, *_ = _build(tmp_path, [el("t", 1, 0, "text", ocr_confidence=0.42)])
    assert _find(result, "low_recorded_ocr_confidence")[0].evidence["recorded_value"] == 0.42


def test_ambiguous_reading_order_saved_flag(tmp_path):
    validation = {"flags": [{"page": 1, "code": "reading_order_approximate", "message": "ambiguous order"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "text")], validation=validation)
    assert _find(result, "ambiguous_reading_order")


def test_page_reading_order_warning_is_structural_not_source_text_corruption(tmp_path):
    validation = {"flags": [{"page": 1, "code": "reading_order_approximate", "message": "no paragraph boxes"}]}
    result, *_ = _build(tmp_path, [el("body", 1, 0, "preserved text")], validation=validation)
    finding = _find(result, "ambiguous_reading_order")[0]
    assert finding.source_ids == ["body"]
    record = next(item for item in result.inventory if item.source_id == "body")
    assert record.reliability_dimensions["structural_placement"]["state"] == "uncertain"
    assert record.reliability_dimensions["source_text_reliability"]["state"] == "no_finding"


def test_region_warning_scopes_only_to_matching_saved_job(tmp_path):
    validation = {"flags": [{
        "page": 1, "region_id": "p1_R1", "code": "incomplete_region",
        "message": "the region border cuts another block; content may be missing",
    }]}
    elements = [
        el("body", 1, 0, "good page text", type="page_text", provenance={"job_id": "p001_page"}),
        el("outer", 1, 1, "complete outer visual", type="figure", provenance={"job_id": "p1_R0"}),
        el("region", 1, 2, "", type="figure", state="pending_review", provenance={"job_id": "p1_R1"}),
    ]
    result, *_ = _build(tmp_path, elements, validation=validation)
    finding = _find(result, "incomplete_region_coverage")[0]
    assert finding.source_ids == ["region"]
    assert finding.evidence["saved_flag"]["region_id"] == "p1_R1"
    body = next(item for item in result.inventory if item.source_id == "body")
    outer = next(item for item in result.inventory if item.source_id == "outer")
    assert finding.finding_id not in body.finding_ids
    assert finding.finding_id not in outer.finding_ids


def test_unresolved_region_id_does_not_downgrade_the_whole_page(tmp_path):
    validation = {"flags": [{
        "page": 1, "region_id": "p1_missing", "code": "incomplete_region", "message": "saved warning",
    }]}
    result, *_ = _build(tmp_path, [el("body", 1, 0, "good text")], validation=validation)
    finding = _find(result, "incomplete_region_coverage")[0]
    assert finding.source_ids == []
    assert finding.limitations == ["flag has no directly identified source element"]


def test_known_missing_visual_requires_recorded_metadata(tmp_path):
    validation = {"flags": [{"page": 1, "code": "missing_visual", "message": "visual missing explicitly"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "text")], validation=validation)
    assert _find(result, "missing_extraction")


def test_nonfatal_layout_warning_is_not_an_extraction_failure(tmp_path):
    validation = {"flags": [{"page": 1, "code": "layout_review", "message": "borderline region skipped"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "preserved source")], validation=validation)
    assert _find(result, "nonfatal_layout_warning")
    assert not _find(result, "recorded_processing_failure")


def test_optional_layout_verification_failure_is_distinguished(tmp_path):
    validation = {"flags": [{"page": 1, "code": "layout_review", "message": "structure check failed"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "preserved source")], validation=validation)
    assert _find(result, "optional_layout_verification_failure")[0].status == "unverified"
    assert not _find(result, "recorded_processing_failure")


def test_explicit_recorded_extraction_failure_remains_detectable(tmp_path):
    validation = {"flags": [{"page": 1, "code": "processing_error", "message": "page extraction failed"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "preserved source")], validation=validation)
    assert _find(result, "recorded_processing_failure")[0].status == "open"


def test_optional_execution_verification_failure_is_not_extraction_failure(tmp_path):
    manifest = {"layout_verification": {"page": 1, "status": "failed", "reason": "check unavailable"}}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "preserved source")], manifest_updates=manifest)
    assert _find(result, "optional_verification_failure")
    assert not _find(result, "recorded_processing_failure")


def test_saved_execution_skip_is_reported_without_inventing_failure(tmp_path):
    manifest = {"page_jobs": [{"page": 1, "status": "skipped", "reason": "saved synthetic skip"}]}
    result, *_ = _build(tmp_path, [el("t", 1, 0, "preserved source")], manifest_updates=manifest)
    finding = _find(result, "recorded_processing_skip")[0]
    assert finding.source_ids == ["t"]
    assert finding.evidence["reason"] == "saved synthetic skip"
    assert not _find(result, "recorded_processing_failure")


def test_no_fabricated_missing_visual(tmp_path):
    result, *_ = _build(tmp_path, [el("t", 1, 0, "short page")])
    assert not _find(result, "missing_extraction")


def test_malformed_table_operand_pattern_is_general(tmp_path):
    result, *_ = _build(tmp_path, [el("tab", 1, 0, "| A | B |\n| --- | --- |\n| x | 120 + = |",
                                                    type="table")])
    finding = _find(result, "unverified_anomaly")[0]
    assert finding.evidence["verbatim"].strip() == "120 + ="
    assert finding.severity == "medium" and finding.evidence["origin_classification"] == "unverified"


@pytest.mark.parametrize("text,subtype", [
    ("Ù†Øµ ØºÙŠØ± Ø³Ù„ÙŠÙ…", "mojibake_sequence"),
    ("word word word", "repeated_token_run"),
])
def test_arabic_and_english_ocr_noise(tmp_path, text, subtype):
    result, *_ = _build(tmp_path, [el("t", 1, 0, text)])
    assert _find(result, subtype)


@pytest.mark.parametrize("text", ["https://example.test/a_b", "report_v2.4.json", "user@example.test"])
def test_legitimate_technical_text_not_flagged(tmp_path, text):
    result, *_ = _build(tmp_path, [el("t", 1, 0, text)])
    assert not [item for item in result.findings if item.category == "ocr_corruption"]


def _describes(source="p", target="f", matches=None):
    return {"rel_id": "r1", "type": "describes", "source_id": source, "target_id": target,
            "source_page": 1, "target_page": 1, "confidence": "high", "review_recommended": False,
            "evidence": {"numeric_matches": matches or []}, "conflicts": [], "limitations": []}


def test_arabic_indic_decimal_and_percentage_agreement(tmp_path):
    match = {"source_raw": "٤٫٣٣٪", "target_raw": "4.33%", "normalized_value": "4.33",
             "dimension": "percent", "source_offsets": [5, 10], "target_offsets": [0, 5]}
    result, *_ = _build(tmp_path, [el("p", 1, 0, "بلغ ٤٫٣٣٪"), el("f", 1, 1, "4.33%", type="figure")],
                        relationships=[_describes(matches=[match])])
    assert _find(result, "numeric_agreement")[0].evidence["count"] == 1


def test_numeric_mismatch_requires_aligned_metric_and_unit(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "Monthly injury rate was 4.33%"),
                                   el("f", 1, 1, "Monthly injury rate 3.33%", type="figure")],
                        relationships=[_describes()])
    assert _find(result, "numeric_conflict")


def test_unrelated_numbers_do_not_create_mismatch(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "Passenger count was 120"),
                                   el("f", 1, 1, "Waiting duration was 15", type="figure")],
                        relationships=[_describes()])
    assert not _find(result, "numeric_conflict")


def test_percentage_and_scalar_are_incompatible(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "Monthly injury rate 5%"),
                                   el("f", 1, 1, "Monthly injury rate 5", type="figure")],
                        relationships=[_describes()])
    assert not _find(result, "numeric_conflict")


def test_existing_leaked_text_becomes_finding(tmp_path):
    relation = {"rel_id": "leak1", "type": "leaked_text", "source_id": "p", "target_id": "f",
                "source_page": 1, "target_page": 1, "confidence": "high",
                "evidence": {"target_offsets": [0, 30], "normalized_overlap_ratio": 1.0}}
    result, *_ = _build(tmp_path, [el("p", 1, 0, "duplicate long synthetic paragraph text"),
                                   el("f", 1, 1, "duplicate long synthetic paragraph text", type="figure")],
                        relationships=[relation])
    assert _find(result, "recorded_leaked_text")


def test_repeated_logical_table_header_is_not_duplicate(tmp_path):
    elements = [el("t1", 1, 0, "| A | B |\n| --- | --- |", type="table"),
                el("t2", 2, 0, "| A | B |\n| --- | --- |", type="table")]
    logical = {"node_id": "logical", "node_type": "logical_table", "page": 1, "text": None,
               "source_id": None, "start": None, "end": None,
               "provenance": {"ordered_parts": ["t1", "t2"], "repeated_header_between_parts": [True]}}
    result, *_ = _build(tmp_path, elements, logical_nodes=[logical])
    assert _find(result, "confirmed_repeated_continuation_header")
    assert not _find(result, "near_duplicate_visual_text")


def test_low_confidence_structure_assignment(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "body")],
                        assignments={"p": {"confidence": "low", "review_recommended": True}})
    assert _find(result, "uncertain_assignments")


def test_unlinked_figure_is_not_called_extraction_failure(tmp_path):
    result, *_ = _build(tmp_path, [el("f", 1, 0, "synthetic chart labels", type="figure")])
    finding = _find(result, "visual_without_caption_or_description")[0]
    assert finding.severity == "info" and finding.status == "unverified"
    assert not _find(result, "missing_extraction")
    visual = next(item for item in result.inventory if item.source_id == "f")
    assert visual.rag_flags == [] and visual.evidence_reliability == "unassessed"


@pytest.mark.parametrize("visual_text", ["مؤشرات مرورية ١٠ ٢٠", "Synthetic traffic indicators 10 20"])
def test_low_confidence_co_location_is_expected_and_does_not_downgrade_content(tmp_path, visual_text):
    relationship = {
        "rel_id": "co1", "type": "co_located_in_section", "source_id": "p", "target_id": "f",
        "source_page": 1, "target_page": 1, "confidence": "low", "review_recommended": True,
        "evidence": {"section_id": "s1"}, "conflicts": [],
        "limitations": ["co-location does not establish explanation, support, or factual agreement"],
    }
    original = json.loads(json.dumps(relationship))
    result, inputs, *_ = _build(
        tmp_path, [el("p", 1, 0, "Nearby paragraph"), el("f", 1, 1, visual_text, type="figure")],
        relationships=[relationship],
    )
    assert not _find(result, "uncertain_existing_relationship")
    assert _find(result, "visual_without_caption_or_description")
    visual = next(item for item in result.inventory if item.source_id == "f")
    assert visual.rag_flags == []
    assert visual.evidence_reliability == "unassessed"
    assert visual.evidence_reliability_reasons == []
    assert inputs.relationships["relationships"][0] == original


def test_uncertain_describes_remains_reviewable_without_downgrading_visual_content(tmp_path):
    relationship = {
        "rel_id": "d1", "type": "describes", "source_id": "p", "target_id": "f",
        "source_page": 1, "target_page": 1, "confidence": "low", "review_recommended": True,
        "evidence": {}, "conflicts": [{"target_id": "other"}], "limitations": ["weak semantic support"],
    }
    result, *_ = _build(
        tmp_path, [el("p", 1, 0, "Possible explanation"),
                   el("f", 1, 1, "Synthetic visual labels", type="figure")],
        relationships=[relationship],
    )
    finding = _find(result, "uncertain_existing_relationship")[0]
    assert finding.status == "needs_review" and finding.relationship_ids == ["d1"]
    visual = next(item for item in result.inventory if item.source_id == "f")
    assert visual.rag_flags == [] and visual.evidence_reliability == "unassessed"
    assert "d1" in visual.relationship_ids and finding.finding_id in visual.finding_ids


def test_verified_numeric_conflict_downgrades_content_with_traceable_reason(tmp_path):
    elements = [el("p", 1, 0, "Monthly injury rate was 4.33%"),
                el("f", 1, 1, "Monthly injury rate was 3.33%", type="figure")]
    result, *_ = _build(tmp_path, elements, relationships=[_describes()])
    assert _find(result, "numeric_conflict")
    visual = next(item for item in result.inventory if item.source_id == "f")
    assert visual.rag_flags == ["needs_review"]
    assert visual.evidence_reliability == "low"
    assert visual.evidence_reliability_reasons
    assert any("numeric_conflict" in reason for reason in visual.evidence_reliability_reasons)


def test_relationship_and_content_findings_remain_separate_when_concurrent(tmp_path):
    relationship = _describes()
    relationship.update({"confidence": "low", "review_recommended": True,
                         "limitations": ["semantic link needs review"]})
    elements = [el("p", 1, 0, "Monthly injury rate was 4.33%"),
                el("f", 1, 1, "Monthly injury rate was 3.33%", type="figure")]
    result, *_ = _build(tmp_path, elements, relationships=[relationship])
    assert _find(result, "uncertain_existing_relationship")
    assert _find(result, "numeric_conflict")
    visual = next(item for item in result.inventory if item.source_id == "f")
    assert visual.evidence_reliability == "low"
    assert any("numeric_conflict" in reason for reason in visual.evidence_reliability_reasons)
    assert not any("uncertain_existing_relationship" in reason for reason in visual.evidence_reliability_reasons)


def test_every_quality_reliability_downgrade_has_a_nonempty_reason(tmp_path):
    elements = [el("p", 1, 0, "Monthly injury rate was 4.33%"),
                el("f", 1, 1, "Monthly injury rate was 3.33%", type="figure", state="needs_review")]
    result, *_ = _build(tmp_path, elements, relationships=[_describes()])
    downgraded = [item for item in result.inventory if item.evidence_reliability != "unassessed"]
    assert downgraded
    assert all(item.evidence_reliability_reasons for item in downgraded)


def test_unresolved_caption_from_component2(tmp_path):
    finding = {"severity": "warning", "code": "ambiguous_caption_target", "items": ["c"],
               "message": "caption has competing targets"}
    result, *_ = _build(tmp_path, [el("c", 1, 0, "Figure 1")], relationship_findings=[finding])
    assert _find(result, "ambiguous_caption_target")


def test_relationship_finding_without_source_keeps_saved_page_context(tmp_path):
    finding = {"severity": "warning", "code": "unresolved_caption", "items": [],
               "message": "saved warning", "evidence": {"page": 1}}
    result, *_ = _build(tmp_path, [el("p", 1, 0, "body")], relationship_findings=[finding])
    result_finding = _find(result, "unresolved_caption")[0]
    assert result_finding.page_numbers == [1]
    assert result_finding.limitations


def test_table_continuation_repeated_header(tmp_path):
    elements = [el("t1", 1, 0, "| A | B |", type="table"), el("t2", 2, 0, "| A | B |", type="table")]
    logical = {"node_id": "logical", "node_type": "logical_table", "page": 1, "text": None,
               "source_id": None, "start": None, "end": None,
               "provenance": {"ordered_parts": ["t1", "t2"], "repeated_header_between_parts": [True]}}
    result, *_ = _build(tmp_path, elements, logical_nodes=[logical])
    assert _find(result, "confirmed_repeated_continuation_header")[0].source_ids == ["t1", "t2"]


def test_incorrect_markdown_column_count(tmp_path):
    result, *_ = _build(tmp_path, [el("t", 1, 0, "| A | B |\n| 1 | 2 | 3 |", type="table")])
    assert _find(result, "inconsistent_markdown_columns")


def test_flattened_scanned_table(tmp_path):
    result, *_ = _build(tmp_path, [el("t", 1, 0, "A B C 1 2 3", type="table", source="qari_crop")])
    assert _find(result, "flattened_unstructured_table")


def test_sparse_chart_ocr_is_unverified(tmp_path):
    result, *_ = _build(tmp_path, [el("f", 1, 0, "7", type="figure")])
    finding = _find(result, "sparse_visual_ocr")[0]
    assert finding.status == "unverified" and finding.limitations


def test_empty_pending_composite_part_is_not_misclassified_as_sparse_ocr(tmp_path):
    elements = [
        el("parent", 1, 0, "complete outer visual", type="figure"),
        el("part", 1, 1, "", type="figure", state="pending_review"),
    ]
    relation = {
        "rel_id": "rel_composite", "type": "composite_part", "source_id": "part", "target_id": "parent",
        "confidence": "high", "method": "saved_layout_containment_decision", "evidence": {},
        "review_recommended": False, "limitations": [],
    }
    result, *_ = _build(tmp_path, elements, relationships=[relation])
    assert not [item for item in _find(result, "sparse_visual_ocr") if "part" in item.source_ids]
    part = next(item for item in result.inventory if item.source_id == "part")
    assert "explicit saved composite relationship" in part.skipped_checks["sparse_visual_ocr"]
    assert _find(result, "nass_pending_review")


def test_cross_page_paragraph_continuation(tmp_path):
    elements = [el("a", 1, 0, "This sentence continues", bbox=[20, 720, 500, 780]),
                el("b", 2, 0, "on the next synthetic page.", bbox=[20, 20, 500, 60])]
    result, *_ = _build(tmp_path, elements)
    finding = _find(result, "possible_paragraph_continuation")[0]
    assert finding.evidence["previous_geometry_reliability"] == "precise"
    assert finding.evidence["next_geometry_reliability"] == "precise"


def test_full_page_ocr_bbox_does_not_trigger_cross_page_continuity(tmp_path):
    elements = [el("a", 1, 0, "This text may continue", type="page_text", bbox=[0, 0, 612, 792]),
                el("b", 2, 0, "but has page-level geometry.", type="page_text", bbox=[0, 0, 612, 792])]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "possible_paragraph_continuation")
    assert all(item.geometry_reliability == "page_level_only" for item in result.inventory)
    assert all("insufficient paragraph geometry" in item.skipped_checks["cross_page_continuity"]
               for item in result.inventory)


def test_missing_geometry_skips_continuity_without_false_positive(tmp_path):
    elements = [el("a", 1, 0, "This text may continue", bbox=None),
                el("b", 2, 0, "but has no geometry.", bbox=None)]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "possible_paragraph_continuation")
    assert all(item.geometry_reliability == "unavailable" for item in result.inventory)
    assert all("cross_page_continuity" in item.skipped_checks for item in result.inventory)


def test_geometry_reliability_is_recorded_for_precise_approximate_and_page_level(tmp_path):
    elements = [el("precise", 1, 0, "precise", bbox=[20, 20, 200, 50]),
                el("approx", 1, 1, "approximate", type="page_text", bbox=[0, 0, 612, 792]),
                el("page", 1, 2, "page", type="page_text", bbox=[0, 0, 612, 792])]
    assignments = {"approx": {"span_id": "approx_span", "unit_id": "approx_span",
                                "bbox_pt": [20, 80, 200, 110]}}
    result, *_ = _build(tmp_path, elements, assignments=assignments)
    reliability = {item.source_id: item.geometry_reliability for item in result.inventory}
    assert reliability == {"precise": "precise", "approx": "approximate", "page": "page_level_only"}


def test_no_invented_text_or_coordinates(tmp_path):
    result, *_ = _build(tmp_path, [el("f", 1, 0, "", type="figure", bbox=None)])
    assert all("corrected_text" not in json.dumps(item.evidence) for item in result.findings)
    assert all(not item.offsets for item in result.findings)


def test_quality_inventory_is_complete(tmp_path):
    elements = [el("a", 1, 0, "A"), el("b", 1, 1, "B", type="figure")]
    result, *_ = _build(tmp_path, elements)
    assert {item.source_id for item in result.inventory} == {"a", "b"}


def test_scorecards_include_clean_pages_with_explicit_denominators(tmp_path):
    result, *_ = _build(tmp_path, [el("a", 1, 0, "ordinary text"), el("b", 2, 0, "more text")])
    assert set(result.summary["pages"]) == {"1", "2"}
    assert result.summary["pages"]["2"]["elements_inspected"] == 1
    assert result.summary["pages"]["2"]["finding_count"] == 0


def test_human_review_fields_initially_blank(tmp_path):
    result, inputs, *_ = _build(tmp_path, [el("f", 1, 0, "7", type="figure")])
    output, _ = write_quality_outputs(result, inputs, out_root=tmp_path / "quality", stamp="fixed",
                                      config=QualityConfig(read_only_files=False))
    template_dir = make_template(output, {1}, stamp="fixed")
    with (template_dir / "quality_review_template.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(item["human_verdict"] == "" and item["reviewer_note"] == "" for item in rows)


def test_human_review_partial_coverage(tmp_path):
    result, inputs, *_ = _build(tmp_path, [el("f", 1, 0, "7", type="figure")])
    output, _ = write_quality_outputs(result, inputs, out_root=tmp_path / "quality", stamp="fixed",
                                      config=QualityConfig(read_only_files=False))
    template_dir = make_template(output, {1}, stamp="fixed")
    review = template_dir / "quality_review_template.csv"
    rows = list(csv.DictReader(review.open(encoding="utf-8-sig", newline="")))
    rows[0]["human_verdict"] = "confirmed_issue"
    with review.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    evaluation = evaluate_review(output, review, stamp="fixed")
    summary = json.loads((evaluation / "quality_review_evaluation.json").read_text(encoding="utf-8"))
    assert summary["counts"]["confirmed_issue"] == 1 and summary["counts"]["not_reviewed"] == len(rows) - 1


def test_source_inputs_are_immutable(tmp_path):
    result, inputs, run, structure, relations = _build(tmp_path, [el("p", 1, 0, "body")])
    paths = [run / "document.json", run / "validation_report.json", run / "execution_manifest.json",
             structure / "assignments.json", structure / "section_tree.json", structure / "manifest.json",
             relations / "relationships.json", relations / "visuals.json", relations / "lists.json",
             relations / "manifest.json"]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write_quality_outputs(result, inputs, out_root=tmp_path / "quality", stamp="one",
                          config=QualityConfig(read_only_files=False))
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_write_once_quality_folder(tmp_path):
    result, inputs, *_ = _build(tmp_path, [el("p", 1, 0, "body")])
    kwargs = {"out_root": tmp_path / "quality", "stamp": "same", "config": QualityConfig(read_only_files=False)}
    write_quality_outputs(result, inputs, **kwargs)
    with pytest.raises(OutputExistsError):
        write_quality_outputs(result, inputs, **kwargs)


def test_quality_output_persists_complete_evidence_audit(tmp_path):
    result, inputs, *_ = _build(tmp_path, [el("p", 1, 0, "Clean mixed GIS text.")])
    output, _ = write_quality_outputs(
        result, inputs, out_root=tmp_path / "quality", stamp="audit",
        config=QualityConfig(read_only_files=False),
    )
    saved = json.loads((output / "quality_findings.json").read_text(encoding="utf-8"))
    assert saved["evidence_audit"] == result.evidence_audit
    assert any(item["check"] == "bidi_order_suspect" and item["status"] == "checked"
               for item in saved["evidence_audit"])


def test_finding_ids_are_stable(tmp_path):
    first, *_ = _build(tmp_path / "a", [el("f", 1, 0, "7", type="figure")])
    second, *_ = _build(tmp_path / "b", [el("f", 1, 0, "7", type="figure")])
    assert [item.finding_id for item in first.findings] == [item.finding_id for item in second.findings]


def test_html_escapes_source_text(tmp_path):
    result, *_ = _build(tmp_path, [el("f", 1, 0, "<script>alert(1)</script>", type="figure")])
    html = render_quality_html(result)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html


def test_skipped_checks_have_reasons(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, None)])
    record = result.inventory[0]
    assert record.skipped_checks and all(record.skipped_checks.values())


def test_suspicious_bidi_line_sequence_is_flagged_but_clean_mixed_text_is_not(tmp_path):
    suspicious = "تشير النتائج إلى\nGIS\nو في التحليل التالي"
    result, *_ = _build(tmp_path, [el("p", 1, 0, suspicious)])
    finding = _find(result, "bidi_order_suspect")[0]
    start, end = finding.evidence["offsets"]
    assert suspicious[start:end] == "GIS"
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean, *_ = _build(clean_root, [el("p", 1, 0, "يستخدم التقرير نظام GIS لتحليل المواقع.")])
    assert not _find(clean, "bidi_order_suspect")


def test_embedded_ltr_clause_break_is_flagged_and_every_element_is_audited(tmp_path):
    elements = [
        el("broken", 1, 0, "تشير الحقول إلى\nوجود نقاط داخل GIS أن جميع السجلات تتبع المنطقة\nالمعتمدة."),
        el("agent", 1, 1, "سجل الموظف\nAgent 25 أعلى متوسط بـ\n70.92 حالة."),
        el("models", 1, 2, "استُخدمت Decision Tree و Random Forest للمقارنة."),
    ]
    result, *_ = _build(tmp_path, elements)
    assert [item.source_ids for item in _find(result, "bidi_order_suspect")] == [["broken"]]
    audit = [item for item in result.evidence_audit if item["check"] == "bidi_order_suspect"]
    assert {item["source_id"] for item in audit} == {"broken", "agent", "models"}
    assert next(item for item in audit if item["source_id"] == "broken")["status"] == "flagged"
    assert all(item["status"] == "checked" for item in audit if item["source_id"] != "broken")


def test_saved_chart_classification_without_values_is_flagged_but_illustration_is_not(tmp_path):
    elements = [
        el("chart", 1, 0, "Speed Delay Weather", type="figure", classification="bar_chart"),
        el("illustration", 1, 1, "Road safety illustration", type="figure", classification="illustration"),
    ]
    result, *_ = _build(tmp_path, elements)
    findings = _find(result, "chart_without_values")
    assert [item.source_ids for item in findings] == [["chart"]]


def test_verified_visual_title_and_labels_without_numbers_is_flagged_with_scope_audit(tmp_path):
    elements = [
        el("chart", 1, 0, "Weekly distribution weekday weekend", type="figure"),
        el("map", 1, 1, "North South EASTERN PROVINCE", type="figure"),
        el("code", 1, 2, "RandomForestClassifier.fit(X_train) dtype: int64", type="figure"),
        el("part", 1, 3, "", type="figure", state="included_in_composite"),
        el("table", 1, 4, "| label | value |", type="table"),
    ]
    visuals = [
        {"visual_id": "chart", "visual_type": "figure", "page": 1,
         "internal_title": {"node_id": "title1", "text": "Weekly distribution", "offsets": [0, 19]},
         "composite": {"parent": None, "parts": [], "unresolved": False}},
        *[{"visual_id": item["id"], "visual_type": item["type"], "page": 1,
           "internal_title": None, "composite": {"parent": "parent" if item["id"] == "part" else None,
                                                    "parts": [], "unresolved": False}}
          for item in elements[1:]],
    ]
    result, *_ = _build(tmp_path, elements, visuals=visuals)
    assert [item.source_ids for item in _find(result, "chart_without_values")] == [["chart"]]
    audit = {item["source_id"]: item for item in result.evidence_audit
             if item["check"] == "chart_without_values"}
    assert audit["chart"]["status"] == "flagged"
    assert "map-like" in audit["map"]["reason"]
    assert "code" in audit["code"]["reason"]
    assert "composite" in audit["part"]["reason"]
    assert "table-quality" in audit["table"]["reason"]
    vocabulary_audit_ids = {item["source_id"] for item in result.evidence_audit
                            if item["check"] == "ocr_vocabulary_noise"}
    assert vocabulary_audit_ids == {"chart", "map", "code", "part", "table"}


def test_ocr_vocabulary_neighbor_support_and_rejected_token_audit(tmp_path):
    visual = "انشغال عن القفادة ومؤشر غريب"
    elements = [
        el("body", 1, 0, "برز الانشغال عن القيادة في التحليل، وتكررت القيادة في الملخص."),
        el("fig", 1, 1, visual, type="figure", source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build(tmp_path, elements)
    finding = next(item for item in _find(result, "ocr_vocabulary_noise")
                   if item.evidence["original_token"] == "القفادة")
    assert finding.evidence["candidate_body_token"] == "القياده"
    audit = [item for item in result.evidence_audit
             if item["check"] == "ocr_vocabulary_noise" and item["source_id"] == "fig"]
    assert any(item["status"] == "flagged" and item["evidence"]["token"] == "القفادة" for item in audit)
    assert any(item["evidence"]["token"] == "غريب" and item["status"] != "flagged" for item in audit)
    assert result.nodes["fig"]["text"] == visual


def test_arabic_singular_plural_spelling_is_not_accused_as_ocr_noise(tmp_path):
    elements = [
        el("body", 1, 0, "توضح معالجة الشكاوي ومتابعة الشكاوي في النظام."),
        el("fig", 1, 1, "مؤشر معالجة الشكوى والمتابعة", type="figure",
           source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not [item for item in _find(result, "ocr_vocabulary_noise")
                if item.evidence["original_token"] == "الشكوى"]


def test_large_edit_distance_needs_two_exact_context_anchors(tmp_path):
    elements = [
        el("body", 1, 0, "توضح معالجة الشكاوى الشكل النهائي، وتكرر الشكل في الصفحة."),
        el("fig", 1, 1, "مؤشر معالجة الشكوى الإغلاق", type="figure",
           source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not [item for item in _find(result, "ocr_vocabulary_noise")
                if item.evidence["original_token"] == "الإغلاق"]


def test_ocr_vocabulary_strong_context_allows_three_edits_and_single_anchor_needs_frequency(tmp_path):
    visual = "التنبؤ المبكر بلحظالية وجود إصابات حسب نوع الطروق ريفي"
    elements = [
        el("body1", 1, 0, "التنبؤ المبكر باحتمالية وجود إصابات حسب نوع الطرق وحالة الطرق."),
        el("body2", 1, 1, "تؤثر الطرق في النتائج وتختلف الطرق حسب الموقع."),
        el("fig", 1, 2, visual, type="figure", source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build(tmp_path, elements)
    findings = {item.evidence["original_token"]: item for item in _find(result, "ocr_vocabulary_noise")}
    assert findings["بلحظالية"].evidence["candidate_body_token"] == "باحتماليه"
    assert findings["الطروق"].evidence["candidate_body_token"] == "الطرق"
    assert result.nodes["fig"]["text"] == visual


def test_ocr_vocabulary_noise_uses_repeated_body_vocabulary_and_preserves_token(tmp_path):
    elements = [
        el("p1", 1, 0, "speed delay weather importance"),
        el("p2", 1, 1, "speed delay weather importance"),
        el("fig", 1, 2, "speed delai weather", type="figure", source="qari_ocr", state="needs_review"),
    ]
    result, *_ = _build(tmp_path, elements)
    finding = _find(result, "ocr_vocabulary_noise")[0]
    assert finding.evidence["original_token"] == "delai"
    assert finding.evidence["candidate_body_token"] == "delay"
    assert result.nodes["fig"]["text"] == "speed delai weather"


def test_ocr_typo_supported_by_aligned_heading_has_traceable_evidence(tmp_path):
    elements = [
        el("heading", 1, 0, "Safety delay patterns"),
        el("fig", 1, 1, "Safety delai patterns 10 20", type="figure", source="qari_ocr"),
    ]
    sections = [{"section_id": "s1", "title": "Safety delay patterns", "page_start": 1, "page_end": 1,
                 "heading_element_ids": ["heading"], "flags": []}]
    result, *_ = _build(tmp_path, elements, sections=sections)
    finding = _find(result, "ocr_vocabulary_noise")[0]
    assert finding.evidence["original_token"] == "delai"
    assert finding.evidence["candidate_body_token"] == "delay"
    assert finding.evidence["support_methods"] == ["aligned_section_heading"]
    assert finding.evidence["supporting_source_ids"] == ["heading"]
    assert finding.evidence["evidence_strength"] == "strong"
    assert finding.confidence == "high" and finding.status == "needs_review"


def test_repeated_matching_local_phrases_are_required_not_global_frequency(tmp_path):
    supported = [
        el("p1", 1, 0, "speed delay weather importance"),
        el("p2", 1, 1, "speed delay weather changes"),
        el("fig", 1, 2, "speed delai weather", type="figure", source="qari_ocr"),
    ]
    result, *_ = _build(tmp_path, supported)
    finding = _find(result, "ocr_vocabulary_noise")[0]
    assert finding.evidence["support_methods"] == ["repeated_matching_local_phrase"]
    assert finding.evidence["supporting_source_ids"] == ["p1", "p2"]

    weak_root = tmp_path / "weak"
    weak_root.mkdir()
    weak = [
        el("p1", 1, 0, "delay analysis speed overview"),
        el("p2", 1, 1, "delay report weather metrics"),
        el("fig", 1, 2, "speed delai weather", type="figure", source="qari_ocr"),
    ]
    weak_result, *_ = _build(weak_root, weak)
    assert not _find(weak_result, "ocr_vocabulary_noise")


def test_two_valid_small_distance_words_are_not_flagged_when_both_are_attested(tmp_path):
    elements = [
        el("p1", 1, 0, "the trial period remains valid"),
        el("p2", 1, 1, "the trail period remains mapped"),
        el("fig", 1, 2, "the trial period remains", type="figure", source="qari_ocr"),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "ocr_vocabulary_noise")


@pytest.mark.parametrize(("heading", "visual"), [
    ("Service reports remain stable", "Service report remain stable"),
    ("سلامة المركبات ضمن الخطة", "سلامة المركبة ضمن الخطة"),
    ("مراقبة الانحراف ضمن المسار", "مراقبة انحراف ضمن المسار"),
    ("تقارير تجاوزت الحدود المسجلة", "تقارير تجاوز الحدود المسجلة"),
])
def test_morphology_and_definite_article_variants_are_not_ocr_noise(tmp_path, heading, visual):
    elements = [el("heading", 1, 0, heading),
                el("fig", 1, 1, visual, type="figure", source="qari_ocr")]
    sections = [{"section_id": "s1", "title": heading, "page_start": 1, "page_end": 1,
                 "heading_element_ids": ["heading"], "flags": []}]
    result, *_ = _build(tmp_path, elements, sections=sections)
    assert not _find(result, "ocr_vocabulary_noise")


def test_ambiguous_aligned_candidate_corrections_are_rejected(tmp_path):
    elements = [
        el("p1", 1, 0, "speed bold weather value"),
        el("p2", 1, 1, "speed bold weather trend"),
        el("p3", 1, 2, "speed gold weather value"),
        el("p4", 1, 3, "speed gold weather trend"),
        el("fig", 1, 4, "speed sold weather", type="figure", source="qari_ocr"),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "ocr_vocabulary_noise")


def test_missing_independent_vocabulary_evidence_does_not_change_rag_reliability(tmp_path):
    elements = [
        el("p", 1, 0, "general speed and weather discussion"),
        el("fig", 1, 1, "speed delai weather", type="figure", source="qari_ocr"),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "ocr_vocabulary_noise")
    visual = next(item for item in result.inventory if item.source_id == "fig")
    assert visual.rag_flags == []
    assert visual.evidence_reliability == "unassessed"


def test_arabic_ocr_typo_offsets_are_exact_and_source_text_is_unchanged(tmp_path):
    heading = "سلامة القيادة المرورية"
    visual_text = "مؤشر سلامة القفادة المرورية العام"
    elements = [el("heading", 1, 0, heading),
                el("fig", 1, 1, visual_text, type="figure", source="qari_ocr")]
    sections = [{"section_id": "s1", "title": heading, "page_start": 1, "page_end": 1,
                 "heading_element_ids": ["heading"], "flags": []}]
    result, *_ = _build(tmp_path, elements, sections=sections)
    finding = _find(result, "ocr_vocabulary_noise")[0]
    offset = finding.offsets[0]
    assert visual_text[offset["start"]:offset["end"]] == "القفادة"
    assert result.nodes["fig"]["text"] == visual_text
    assert finding.evidence["candidate_body_token"] == "القياده"


def test_strong_ocr_evidence_sets_proportionate_flags_and_keeps_upstream_review(tmp_path):
    elements = [
        el("heading", 1, 0, "Safety delay patterns"),
        el("fig", 1, 1, "Safety delai patterns", type="figure", source="qari_ocr", state="needs_review"),
    ]
    sections = [{"section_id": "s1", "title": "Safety delay patterns", "page_start": 1, "page_end": 1,
                 "heading_element_ids": ["heading"], "flags": []}]
    result, *_ = _build(tmp_path, elements, sections=sections)
    visual = next(item for item in result.inventory if item.source_id == "fig")
    assert visual.rag_flags == ["needs_review", "ocr_noise"]
    assert visual.evidence_reliability == "medium"
    assert any("upstream validation_state" in reason for reason in visual.evidence_reliability_reasons)
    assert any("ocr_vocabulary_noise" in reason for reason in visual.evidence_reliability_reasons)


def test_vocabulary_detection_preserves_numeric_title_and_describes_evidence(tmp_path):
    match = {"source_raw": "10%", "target_raw": "10%", "normalized_value": "10",
             "dimension": "percent", "source_offsets": [20, 23], "target_offsets": [22, 25]}
    elements = [
        el("heading", 1, 0, "Safety delay patterns"),
        el("p", 1, 1, "The documented value is 10%."),
        el("fig", 1, 2, "Safety delai patterns 10%", type="figure", source="qari_ocr"),
    ]
    sections = [{"section_id": "s1", "title": "Safety delay patterns", "page_start": 1, "page_end": 1,
                 "heading_element_ids": ["heading"], "flags": []}]
    visual = {"visual_id": "fig", "visual_type": "figure", "page": 1,
              "internal_title": {"text": "Safety delai patterns", "offsets": [0, 21]},
              "caption": None, "composite": {"unresolved": False}}
    result, inputs, *_ = _build(
        tmp_path, elements, sections=sections, relationships=[_describes(source="p", target="fig", matches=[match])],
        visuals=[visual],
    )
    assert _find(result, "ocr_vocabulary_noise")
    assert _find(result, "numeric_agreement")
    inventory = next(item for item in result.inventory if item.source_id == "fig")
    assert inventory.relationship_ids == ["r1"]
    assert inputs.visuals["visuals"][0]["internal_title"] == visual["internal_title"]
    assert result.nodes["fig"]["text"] == elements[2]["content"]


def test_legitimate_uncommon_visual_word_is_not_automatically_ocr_noise(tmp_path):
    elements = [
        el("p1", 1, 0, "speed delay weather importance"),
        el("p2", 1, 1, "speed delay weather importance"),
        el("fig", 1, 2, "speed QGIS weather", type="figure", source="qari_ocr", state="needs_review"),
    ]
    result, *_ = _build(tmp_path, elements)
    assert not _find(result, "ocr_vocabulary_noise")


def test_source_content_anomaly_requires_explicit_origin_provenance(tmp_path):
    supported, *_ = _build(tmp_path, [el(
        "t", 1, 0, "| value |\n| 29433 + = |", type="table",
        provenance={"anomaly_origin": "source_document"},
    )])
    assert _find(supported, "source_content_anomaly")
    assert not _find(supported, "unverified_anomaly")
    unknown_root = tmp_path / "unknown"
    unknown_root.mkdir()
    unknown, *_ = _build(unknown_root, [el("t", 1, 0, "| value |\n| 29433 + = |", type="table")])
    assert _find(unknown, "unverified_anomaly")
    assert not _find(unknown, "source_content_anomaly")


def test_known_missing_relationship_creates_quality_finding(tmp_path):
    relationship = {
        "rel_id": "missing-rel", "type": "known_missing", "source_id": "document:synthetic_quality",
        "target_id": "missing-visual", "evidence": {"page": 1, "reason": "explicit image skip"},
        "confidence": "high", "method": "explicit_saved_metadata", "conflicts": [],
        "review_recommended": True, "limitations": [],
    }
    node = {"node_id": "missing-visual", "node_type": "known_missing_visual", "page": 1,
            "text": None, "source_id": None, "start": None, "end": None,
            "provenance": {"source": "execution_manifest", "record": {"status": "skipped"}}}
    document_node = {"node_id": "document:synthetic_quality", "node_type": "document", "page": None,
                     "text": None, "source_id": None, "start": None, "end": None, "provenance": {}}
    result, *_ = _build(tmp_path, [el("p", 1, 0, "text")], relationships=[relationship],
                        logical_nodes=[document_node, node])
    finding = _find(result, "recorded_known_missing")[0]
    assert finding.source_ids == ["missing-visual"] and finding.status == "needs_review"


def test_rag_flags_default_unassessed_and_propagate_only_explicit_concerns(tmp_path):
    result, *_ = _build(tmp_path, [
        el("clean", 1, 0, "ordinary clean text"),
        el("review", 1, 1, "review text", state="needs_review"),
    ])
    inventory = {item.source_id: item for item in result.inventory}
    assert inventory["clean"].evidence_reliability == "unassessed"
    assert inventory["clean"].rag_flags == []
    assert inventory["review"].evidence_reliability == "medium"
    assert "needs_review" in inventory["review"].rag_flags

