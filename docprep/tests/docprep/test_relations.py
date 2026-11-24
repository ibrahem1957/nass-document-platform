"""Synthetic regression specification for Component 2.

These tests intentionally contain no real document text or files.  They are written
for the user to execute; implementation work must not execute them automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from docprep.evaluate_relations import make_template
from docprep.output import OutputExistsError
from docprep.relation_builder import build_relationships
from docprep.relation_inputs import load_relationship_inputs
from docprep.relation_model import RelationNode
from docprep.relation_output import relationship_json, write_relation_outputs
from docprep.relation_text import match_numeric_tokens
from docprep.relation_validation import validate_relationships

from .helpers import el, write_run


def _structure(root: Path, run_id: str, elements: list[dict], *, roles: dict[str, str] | None = None,
               sections: dict[str, str] | None = None, lists: list[dict] | None = None,
               section_titles: dict[str, str] | None = None,
               section_heading_ids: dict[str, list[str]] | None = None,
               derived_spans: dict[str, list[dict]] | None = None) -> Path:
    folder = root / f"{run_id}_structure"
    folder.mkdir()
    roles, sections = roles or {}, sections or {}
    derived_spans = derived_spans or {}
    inventory = [{
        "element_id": item["id"], "page": item["page"], "type": item["type"],
        "reading_order": item.get("reading_order"), "document_order": index,
        "bbox_pt": item.get("bbox_pt"), "validation_state": item.get("validation_state"),
        "source": item.get("source"), "text": item.get("content"), "represented_by": [item["id"]],
        "derived_spans": derived_spans.get(item["id"], []),
    } for index, item in enumerate(elements)]
    assignments = [{
        "unit_id": item["id"], "element_id": item["id"], "span_id": None, "page": item["page"],
        "order": [index, 0], "role": roles.get(item["id"], "figure" if item["type"] == "figure" else
                                                 "table" if item["type"] == "table" else "paragraph"),
        "original_type": item["type"], "section_id": sections.get(item["id"]), "list_id": None,
        "confidence": "high", "reasons": [], "supporting": [], "conflicting": [], "deterministic": True,
        "review_recommended": False, "validation_state": item.get("validation_state"),
        "derived_span_uncertain": None, "text": item.get("content"), "bbox_pt": item.get("bbox_pt"),
    } for index, item in enumerate(elements)]
    section_ids = sorted(set(sections.values()))
    section_titles, section_heading_ids = section_titles or {}, section_heading_ids or {}
    section_data = [{
        "section_id": section_id, "title": section_titles.get(section_id, "Synthetic traffic statistics"),
        "page_start": 1, "page_end": max(item["page"] for item in elements),
        "heading_element_ids": section_heading_ids.get(section_id, []),
    } for section_id in section_ids]
    (folder / "assignments.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "source_inventory": inventory, "assignments": assignments,
    }), encoding="utf-8")
    (folder / "section_tree.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run_id, "sections": section_data, "lists": lists or [],
    }), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps({"source_run_id": run_id}), encoding="utf-8")
    return folder


def _build(tmp_path: Path, elements: list[dict], *, roles: dict[str, str] | None = None,
           sections: dict[str, str] | None = None, lists: list[dict] | None = None,
           validation: dict | None = None, layout: dict | None = None, manifest_updates: dict | None = None,
           section_titles: dict[str, str] | None = None,
           section_heading_ids: dict[str, list[str]] | None = None,
           derived_spans: dict[str, list[dict]] | None = None):
    run_id = "synthetic_relations"
    directions = {item["page"]: "rtl" for item in elements if any("\u0600" <= char <= "\u06ff"
                                                                   for char in str(item.get("content") or ""))}
    run = write_run(tmp_path, run_id, elements, validation=validation, layout=layout, directions=directions)
    if manifest_updates:
        manifest_path = run / "execution_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(manifest_updates)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    structure = _structure(
        tmp_path, run_id, elements, roles=roles, sections=sections, lists=lists,
        section_titles=section_titles, section_heading_ids=section_heading_ids, derived_spans=derived_spans,
    )
    return build_relationships(load_relationship_inputs(run, structure)), run, structure


def _types(result, kind: str):
    return [relation for relation in result.relationships if relation.type == kind]


@pytest.mark.parametrize("caption_y,visual_y", [(80, 120), (300, 120)])
def test_caption_above_or_below_figure(tmp_path, caption_y, visual_y):
    elements = [
        el("cap", 1, 0, "Figure 3 — synthetic totals", bbox=[100, caption_y, 300, caption_y + 20]),
        el("fig", 1, 1, "Synthetic chart\n10 20", type="figure", bbox=[80, visual_y, 400, visual_y + 150]),
    ]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"}, sections={"cap": "s1", "fig": "s1"})
    assert [(item.source_id, item.target_id) for item in _types(result, "caption_of")] == [("cap", "fig")]


def test_competing_visuals_are_conflicted_not_arbitrary(tmp_path):
    elements = [
        el("cap", 1, 0, "Figure 1", bbox=[100, 100, 200, 120]),
        el("a", 1, 1, "A", type="figure", bbox=[90, 130, 210, 230]),
        el("b", 1, 2, "B", type="figure", bbox=[90, 130, 210, 230]),
    ]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"})
    assert not _types(result, "caption_of")
    assert any(item["code"] == "ambiguous_caption_target" for item in result.findings)


@pytest.mark.parametrize("caption", ["الشكل ٣: نتائج تركيبية", "Figure 3: Synthetic results"])
def test_arabic_and_english_figure_captions(tmp_path, caption):
    elements = [el("cap", 1, 0, caption, bbox=[20, 20, 250, 40]),
                el("fig", 1, 1, "chart", type="figure", bbox=[20, 45, 250, 200])]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"})
    assert _types(result, "caption_of")[0].target_id == "fig"


def test_table_caption_targets_table(tmp_path):
    elements = [el("cap", 1, 0, "Table 2", bbox=[20, 20, 100, 40]),
                el("table", 1, 1, "A\tB\n1\t2", type="table", bbox=[20, 45, 300, 200])]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"})
    assert _types(result, "caption_of")[0].target_id == "table"


@pytest.mark.parametrize("source_text,source_type", [
    ("Chart OCR text Figure 4: payments 41%", "figure"),
    ("This paragraph discusses Figure 4 and Figure 5 in context.", "text"),
    ("ملخص يجمع المؤشرات في شكل واحد.", "text"),
])
def test_embedded_reference_or_visual_ocr_is_not_a_caption_source(tmp_path, source_text, source_type):
    elements = [el("source", 1, 0, source_text, type=source_type, bbox=[20, 20, 200, 80]),
                el("target", 1, 1, "chart", type="figure", bbox=[20, 90, 200, 200])]
    result, _, _ = _build(tmp_path, elements)
    assert not _types(result, "caption_of")


def test_inside_title_and_matching_section(tmp_path):
    elements = [el("fig", 1, 0, "Synthetic traffic statistics\n10 20 30", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"fig": "s1"})
    title = _types(result, "title_inside_visual")[0]
    assert title.evidence["verbatim"] == "Synthetic traffic statistics"
    assert _types(result, "titled_by_section")


def test_visual_title_matches_explicit_caption_with_auditable_offsets(tmp_path):
    elements = [
        el("cap", 1, 0, "Figure 1: Feature Importance", bbox=[20, 20, 250, 40]),
        el("fig", 1, 1, "Feature Importance\nSpeed Delay Weather", type="figure", bbox=[20, 50, 300, 240]),
    ]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"},
                           sections={"cap": "s1", "fig": "s1"})
    title = next(item for item in result.visuals if item["visual_id"] == "fig")["internal_title"]
    assert title["text"] == "Feature Importance"
    assert "normalized_explicit_caption_match" in title["detection_methods"]
    assert result.nodes[title["node_id"]].text == elements[1]["content"][title["offsets"][0]:title["offsets"][1]]


def test_visual_without_supported_title_remains_null_with_rejection_reason(tmp_path):
    result, _, _ = _build(tmp_path, [el(
        "fig", 1, 0, "This is a complete explanatory sentence.\nAnother ordinary sentence follows.", type="figure",
    )])
    visual = result.visuals[0]
    assert visual["internal_title"] is None
    assert visual["internal_title_evaluation"]["status"] == "rejected"
    assert any(item["status"] == "rejected" for item in result.evidence_audit)


@pytest.mark.parametrize(("title", "labels"), [
    ("توزيع المؤشرات حسب الفترة", "الفئة الأولى الفئة الثانية ١٠ ٢٠"),
    ("Regional indicators by period", "North South 10 20"),
])
def test_flattened_ocr_title_matches_heading_without_consuming_labels(tmp_path, title, labels):
    original = f"  {title} {labels}"
    elements = [el("heading", 1, 0, title), el("fig", 1, 1, original, type="figure", state="needs_review")]
    result, run, _ = _build(
        tmp_path, elements, sections={"heading": "s1", "fig": "s1"},
        section_titles={"s1": title}, section_heading_ids={"s1": ["heading"]},
    )
    visual = next(item for item in result.visuals if item["visual_id"] == "fig")
    detected = visual["internal_title"]
    assert detected["text"] == title
    assert detected["offsets"] == [2, 2 + len(title)]
    assert original[slice(*detected["offsets"])] == title
    assert detected["detection_methods"] == ["prefix_heading_match"]
    assert detected["source_evidence"][0]["source_id"] == "heading"
    assert detected["source_evidence"][0]["similarity"] == 1.0
    assert visual["rag_flags"] == ["needs_review"]
    assert result.nodes["fig"].text == original
    saved = json.loads((run / "document.json").read_text(encoding="utf-8"))
    assert next(item for item in saved["elements"] if item["id"] == "fig")["content"] == original


def test_saved_layout_title_attachment_plus_independent_page_span_supports_title(tmp_path):
    title = "أداء المسارات خلال الأسبوع"
    page_text = f"{title} الشكل 4: أداء المسارات"
    visual_text = f"{title} التأخير الركاب الرحلات 3.2 412"
    attachment = {"block_index": 7, "label": "figure_title", "position": "above", "gap_pt": 6.0,
                  "bbox_pt": [300, 40, 500, 60]}
    elements = [
        el("page", 1, 0, page_text, type="page_text"),
        el("table", 1, 1, visual_text, type="table", provenance={"title_attachments": attachment}),
    ]
    spans = {"page": [{
        "span_id": "page_span_1", "element_id": "page", "start": 0, "end": len(title),
        "text": title, "role_hint": "paragraph", "uncertain": False,
    }]}
    layout = {"pages": [{"page": 1, "boxes": [{
        "block_index": 7, "label": "figure_title", "bbox_pt": attachment["bbox_pt"], "score": 0.9,
    }], "selection": {"regions": [], "others": []}}]}
    result, _, _ = _build(tmp_path, elements, derived_spans=spans, layout=layout)
    visual = next(item for item in result.visuals if item["visual_id"] == "table")
    assert visual["internal_title"]["text"] == title
    evidence = visual["internal_title"]["source_evidence"][0]
    assert evidence["detection_method"] == "prefix_layout_title_match"
    assert evidence["span_id"] == "page_span_1"
    assert evidence["layout_title_attachment"] == attachment
    assert result.nodes["table"].text == visual_text
    assert not any(item["code"] == "unresolved_layout_figure_title" for item in result.findings)


def test_layout_title_attachment_without_matching_independent_span_is_rejected(tmp_path):
    attachment = {"block_index": 4, "label": "figure_title", "position": "above"}
    elements = [
        el("page", 1, 0, "Figure 8: unrelated dashboard"),
        el("fig", 1, 1, "Narrative sentence. Revenue 12 Cost 8", type="figure",
           provenance={"title_attachments": attachment}),
    ]
    spans = {"page": [{
        "span_id": "page_span", "element_id": "page", "start": 0, "end": 29,
        "text": "Figure 8: unrelated dashboard", "role_hint": "caption", "uncertain": False,
    }]}
    result, _, _ = _build(tmp_path, elements, derived_spans=spans)
    visual = next(item for item in result.visuals if item["visual_id"] == "fig")
    assert visual["internal_title"] is None
    assert "no strong prefix match" in visual["internal_title_evaluation"]["reason"]


def test_flattened_ocr_title_accepts_strong_near_exact_heading_match(tmp_path):
    heading = "Regional incident trends by month"
    ocr_title = "Regional incident trend by month"
    result, _, _ = _build(
        tmp_path, [el("fig", 1, 0, f"{ocr_title} North South 10 20", type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": heading},
    )
    title = result.visuals[0]["internal_title"]
    assert title["text"] == ocr_title
    assert title["source_evidence"][0]["match_kind"] == "near_exact"
    assert title["source_evidence"][0]["similarity"] >= 0.72


def test_flattened_title_accepts_one_internal_content_token_edit_but_not_two(tmp_path):
    heading = "أعداد البلاغات وأبرز خمسة مسببات حسب الفترة"
    accepted_text = "أعداد البلاغات وأبرز خمسة مسببات للبلاغات حسب الفترة سبب البلاغ تأخير"
    accepted, _, _ = _build(
        tmp_path / "accepted", [el("fig", 1, 0, accepted_text, type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": heading},
    )
    title = accepted.visuals[0]["internal_title"]
    assert title["text"] == "أعداد البلاغات وأبرز خمسة مسببات للبلاغات حسب الفترة"
    assert title["detection_methods"] == ["prefix_heading_match_token_edit"]
    assert title["source_evidence"][0]["match_kind"] == "token_edit"
    assert title["source_evidence"][0]["boundary_rule"] == "single_internal_content_token_edit"

    rejected_text = "أعداد البلاغات وأبرز خمسة مسببات مهمة للبلاغات حسب الفترة سبب البلاغ"
    rejected, _, _ = _build(
        tmp_path / "rejected", [el("fig", 1, 0, rejected_text, type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": heading},
    )
    assert rejected.visuals[0]["internal_title"] is None


@pytest.mark.parametrize(("heading", "ocr_title", "labels"), [
    ("3) توزيع البلاغات المهمة حسب الفترة:", "توزيع البلاغات المهمة حسب الفترة", "وجود إصابات ١٠ ٢٠"),
    ("٤. توزيع المؤشرات بين أيام العمل والعطلة:", "توزيع المؤشرات بين أيام العمل والعطلة", "وجود تأخير ١٢ ١٨"),
    ("5) العلاقة بين فئة الطريق وأسباب البلاغ:", "العلاقة بين فئة الطريق وأسباب البلاغ", "نوع الطريق رئيسي فرعي"),
    ("6. Service incidents by reporting period:", "Service incidents by reporting period", "Injury status 10 20"),
])
def test_structural_heading_number_does_not_pull_first_chart_label(tmp_path, heading, ocr_title, labels):
    original = f"{ocr_title} {labels}"
    result, _, _ = _build(
        tmp_path, [el("heading", 1, 0, heading), el("fig", 1, 1, original, type="figure")],
        sections={"heading": "s1", "fig": "s1"}, section_titles={"s1": heading},
        section_heading_ids={"s1": ["heading"]},
    )
    title = next(item for item in result.visuals if item["visual_id"] == "fig")["internal_title"]
    assert title["text"] == ocr_title
    assert original[slice(*title["offsets"])] == ocr_title
    evidence = title["source_evidence"][0]
    assert evidence["match_kind"] == "exact_normalized"
    assert evidence["boundary_rule"] == "shortest_exact_prefix"
    assert evidence["comparison_reference"] == ocr_title


def test_missing_title_word_is_not_replaced_by_following_chart_label(tmp_path):
    result, _, _ = _build(
        tmp_path, [el("fig", 1, 0, "Regional incident trends North South 10 20", type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": "Regional incident trends by month"},
    )
    visual = result.visuals[0]
    assert visual["internal_title"] is None
    assert "no strong prefix match" in visual["internal_title_evaluation"]["reason"]


def test_near_exact_boundary_allows_spelling_substitution_but_not_extra_label(tmp_path):
    ocr_title = "Monthly incident severty distribution"
    original = f"  {ocr_title} Critical Minor 14 22"
    result, _, _ = _build(
        tmp_path, [el("fig", 1, 0, original, type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": "Monthly incident severity distribution"},
    )
    title = result.visuals[0]["internal_title"]
    assert title["text"] == ocr_title
    assert title["offsets"] == [2, 2 + len(ocr_title)]
    assert title["source_evidence"][0]["match_kind"] == "near_exact"
    assert title["source_evidence"][0]["boundary_rule"] == "equal_token_count_single_spelling_substitution"
    assert len([item for item in title["source_evidence"][0]["token_alignment"]
                if item["candidate"] != item["reference"]]) == 1


def test_flattened_ocr_title_matches_explicit_caption(tmp_path):
    elements = [
        el("cap", 1, 0, "Figure 7: Service delay distribution", bbox=[20, 20, 300, 40]),
        el("fig", 1, 1, "Service delay distribution Morning Evening 12 18", type="figure",
           bbox=[20, 45, 320, 220]),
    ]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"})
    title = next(item for item in result.visuals if item["visual_id"] == "fig")["internal_title"]
    assert title["text"] == "Service delay distribution"
    assert title["detection_methods"] == ["prefix_caption_match"]
    assert title["source_evidence"][0]["source_id"] == "cap"


def test_flattened_ocr_without_independent_title_support_is_rejected(tmp_path):
    result, _, _ = _build(tmp_path, [el(
        "fig", 1, 0, "Regional indicators North South 10 20", type="figure",
    )])
    visual = result.visuals[0]
    assert visual["internal_title"] is None
    assert "no strong prefix match" in visual["internal_title_evaluation"]["reason"]


def test_conflicting_heading_and_caption_prefixes_are_rejected(tmp_path):
    elements = [
        el("heading", 1, 0, "Regional trends"),
        el("cap", 1, 1, "Figure 4: Regional trends by month", bbox=[20, 40, 300, 60]),
        el("fig", 1, 2, "Regional trends by month North South 10 20", type="figure",
           bbox=[20, 65, 320, 220]),
    ]
    result, _, _ = _build(
        tmp_path, elements, roles={"cap": "caption"},
        sections={"heading": "s1", "cap": "s1", "fig": "s1"},
        section_titles={"s1": "Regional trends"}, section_heading_ids={"s1": ["heading"]},
    )
    visual = next(item for item in result.visuals if item["visual_id"] == "fig")
    assert visual["internal_title"] is None
    assert "conflicting title evidence" in visual["internal_title_evaluation"]["reason"]
    assert len(visual["internal_title_evaluation"]["candidates"]) == 2


def test_multiple_single_word_kpi_labels_do_not_become_one_internal_title(tmp_path):
    result, _, _ = _build(
        tmp_path, [el("fig", 1, 0, "Revenue Profit Margin 12 18 25", type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": "Revenue"},
    )
    assert result.visuals[0]["internal_title"] is None


def test_generic_topic_overlap_is_not_a_flattened_internal_title(tmp_path):
    result, _, _ = _build(
        tmp_path, [el("fig", 1, 0, "Traffic statistics North South 10 20", type="figure")],
        sections={"fig": "s1"}, section_titles={"s1": "Traffic statistics"},
    )
    assert result.visuals[0]["internal_title"] is None


def test_flattened_title_preserves_existing_caption_description_and_numeric_evidence(tmp_path):
    elements = [
        el("heading", 1, 0, "Synthetic regional trend"),
        el("cap", 1, 1, "Figure 2: Synthetic regional trend", bbox=[20, 40, 300, 60]),
        el("fig", 1, 2, "Synthetic regional trend North South 10% 20%", type="figure",
           bbox=[20, 65, 320, 220]),
        el("p", 1, 3, "As shown in Figure 2, the values are 10% and 20%."),
    ]
    result, _, _ = _build(
        tmp_path, elements, roles={"cap": "caption"},
        sections={item["id"]: "s1" for item in elements},
        section_titles={"s1": "Synthetic regional trend"}, section_heading_ids={"s1": ["heading"]},
    )
    visual = next(item for item in result.visuals if item["visual_id"] == "fig")
    assert visual["internal_title"] is not None
    assert _types(result, "caption_of")
    assert any(item.source_id == "p" and item.target_id == "fig" for item in _types(result, "describes"))
    assert visual["matched_numeric_evidence"]
    assert visual["rag_flags"] == []
    assert visual["evidence_reliability"] == "unassessed"


def test_numeric_equivalence_preserves_scale_and_original_forms():
    for left, right in (("4K incidents", "4000 incidents"), ("4 آلاف حادث", "4000 حادث"),
                        ("four thousand incidents", "4000 incidents")):
        matches = match_numeric_tokens(left, right)
        assert matches[0]["normalized_value"] == "4000"
        assert matches[0]["source_raw"] in left and matches[0]["target_raw"] in right


def test_rounded_percentage_match_records_method_and_tolerance():
    match = match_numeric_tokens("approximately 3%", "3.01%")[0]
    assert match["comparison_method"] == "rounded_percent"
    assert match["rounding"]["decimal_places"] == 0 and match["tolerance"] == "0.5"


def test_explicit_arabic_and_english_time_expressions_match():
    match = match_numeric_tokens("Peak time 05 PM", "وقت الذروة الساعة الخامسة مساءً")[0]
    assert match["normalized_value"] == "17:00" and match["dimension"] == "time_of_day"


def test_single_rounded_numeric_match_is_rejected_and_audited(tmp_path):
    elements = [el("p", 1, 0, "The chart shows approximately 3%."),
                el("fig", 1, 1, "Rate\n3.01%", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")
    assert any(item.get("reason") == "fewer than two independent matching values"
               for item in result.evidence_audit)


def test_named_visual_reference_creates_describes_without_topic_only_matching(tmp_path):
    elements = [
        el("fig", 1, 0, "Feature Importance\nSpeed Delay Weather", type="figure"),
        el("p", 2, 0, "The Feature Importance chart explains the ranked variables."),
    ]
    result, _, _ = _build(tmp_path, elements, sections={"fig": "s1", "p": "s1"})
    relation = _types(result, "describes")[0]
    assert relation.evidence["rule"] == "explicit_or_contextual_named_reference"
    assert relation.evidence["named_references"][0]["original_reference"] == "Feature Importance"


def test_weak_colocation_is_low_confidence_and_never_describes(tmp_path):
    elements = [el("p", 1, 0, "General introductory material without a visual reference."),
                el("fig", 1, 1, "Unrelated Labels\nNorth South", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")
    weak = _types(result, "co_located_in_section")
    assert len(weak) == 1 and weak[0].confidence == "low"
    assert "does not establish explanation" in weak[0].limitations[0]


def test_weak_colocation_redirects_empty_composite_part_to_parent(tmp_path):
    elements = [
        el("parent", 1, 0, "Composite chart labels", type="figure", bbox=[20, 40, 300, 220]),
        el("part", 1, 5, "", type="figure", bbox=[40, 80, 120, 130],
           state="included_in_composite", provenance={"parent_element_id": "parent"}),
        el("p", 1, 4, "Nearby explanatory material without a semantic visual reference."),
    ]
    result, _, _ = _build(tmp_path, elements, sections={item["id"]: "s1" for item in elements})
    weak = _types(result, "co_located_in_section")
    assert len(weak) == 1 and weak[0].target_id == "parent"
    assert weak[0].evidence["redirected_from_composite_part"] == "part"
    assert not any(item.target_id == "part" for item in weak)
    assert any(item.source_id == "part" and item.target_id == "parent"
               for item in _types(result, "composite_part"))


def test_explicit_skip_creates_known_missing_visual_but_no_inference_without_record(tmp_path):
    elements = [el("p", 1, 0, "Synthetic text")]
    skip_record = {"page": 1, "status": "skipped", "type": "image",
                   "reason": "image skipped because the saved region was too small"}
    result, _, _ = _build(tmp_path, elements, manifest_updates={
        "skipped_visuals": [skip_record],
    })
    missing = [item for item in result.visuals if item["visual_type"] == "known_missing"]
    assert len(missing) == 1 and missing[0]["page"] == 1 and missing[0]["bbox_pt"] is None
    node = result.nodes[missing[0]["visual_id"]]
    assert node.text is None and node.source_id is None
    assert node.start is None and node.end is None
    assert node.provenance == {
        "source": "execution_manifest.skipped_visuals", "record": skip_record, "bbox_pt": None,
    }
    assert node.review_flags == ("missing_source_content",)
    findings = validate_relationships(result)
    offsets = next(item for item in findings if item["code"] == "derived_offsets_valid")
    assert offsets["severity"] == "ok" and offsets["items"] == []
    serialized = next(item for item in relationship_json(result, findings)["nodes"]
                      if item["node_id"] == node.node_id)
    assert serialized["start"] is None and serialized["end"] is None
    assert serialized["provenance"]["record"] == skip_record
    assert serialized["review_flags"] == ["missing_source_content"]
    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean, _, _ = _build(clean_root, elements)
    assert not [item for item in clean.visuals if item["visual_type"] == "known_missing"]


def test_malformed_known_missing_positional_fields_fail_validation(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "Synthetic text")], manifest_updates={
        "skipped_visuals": [{"page": 1, "status": "skipped", "type": "image", "reason": "saved skip"}],
    })
    node = next(value for value in result.nodes.values() if value.node_type == "known_missing_visual")
    result.nodes[node.node_id] = RelationNode(
        node.node_id, "known_missing_visual", node.page, None, None,
        {"source": "validation_report", "record": {"status": "skipped"}},
        ("missing_source_content",),
    )
    findings = validate_relationships(result)
    offsets = next(item for item in findings if item["code"] == "derived_offsets_valid")
    assert offsets["severity"] == "error" and offsets["items"] == [node.node_id]


def test_real_text_derived_node_offsets_remain_strict(tmp_path):
    result, _, _ = _build(tmp_path, [el("fig", 1, 0, "Synthetic visual title\n10 20", type="figure")],
                           sections={"fig": "s1"}, section_titles={"s1": "Synthetic visual title"})
    title = next(value for value in result.nodes.values() if value.node_type == "visual_title_span")
    result.nodes[title.node_id] = RelationNode(
        node_id=title.node_id, node_type=title.node_type, page=title.page, text=title.text,
        source_id=title.source_id, start=title.start, end=(title.end or 0) + 1,
        provenance=title.provenance, review_flags=title.review_flags,
    )
    findings = validate_relationships(result)
    offsets = next(item for item in findings if item["code"] == "derived_offsets_valid")
    assert offsets["severity"] == "error" and offsets["items"] == [title.node_id]


def test_known_missing_validation_does_not_mutate_relationships_or_visuals(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "Synthetic text")], manifest_updates={
        "missing_visuals": [{"page": 1, "reason": "missing_visual: explicit saved skip", "region_id": "r1"}],
    })
    relationships_before = result.relationship_dicts()
    visuals_before = json.loads(json.dumps(result.visuals, ensure_ascii=False))
    findings = validate_relationships(result)
    assert next(item for item in findings if item["code"] == "derived_offsets_valid")["severity"] == "ok"
    assert result.relationship_dicts() == relationships_before
    assert result.visuals == visuals_before
    known = _types(result, "known_missing")
    assert len(known) == 1 and known[0].evidence["source_record"]["region_id"] == "r1"


@pytest.mark.parametrize("table_text", [
    "| Column A | Column B |\n| --- | --- |\n| 1 | 2 |",
    "_(table continued from page 1; header repeated from that page)_\n\n| A | B |\n| --- | --- |",
])
def test_table_headers_and_continuation_markers_are_not_inside_titles(tmp_path, table_text):
    result, _, _ = _build(tmp_path, [el("table", 1, 0, table_text, type="table")])
    assert not _types(result, "title_inside_visual")


def test_caption_leakage_is_recorded_without_deletion(tmp_path):
    phrase = "Figure 8 shows the complete synthetic monthly totals."
    elements = [el("p", 1, 0, phrase), el("fig", 1, 1, f"{phrase}\n10 20", type="figure")]
    result, _, _ = _build(tmp_path, elements)
    leak = _types(result, "leaked_text")[0]
    assert leak.evidence["target_offsets"] == [0, len(phrase)]
    assert result.nodes["fig"].text.startswith(phrase)


@pytest.mark.parametrize("paragraph,visual", [
    ("القيم ١٬٢٠٠ ريال و٣٫٥٪", "١٬٢٠٠ ريال\n٣٫٥٪"),
    ("Values are 12K riders and 3M visits", "12,000 riders\n3,000,000 visits"),
])
def test_two_compatible_numeric_values_describe_visual(tmp_path, paragraph, visual):
    elements = [el("p", 1, 0, paragraph), el("fig", 1, 1, visual, type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    relation = _types(result, "describes")[0]
    assert len(relation.evidence["numeric_matches"]) >= 2


def test_unlabeled_chart_values_require_compatible_context_labels(tmp_path):
    paragraph = "Average cases were 44.38 cases and duration was 5.96 minutes; the maximum was 159 minutes."
    visual = "Average weekly cases and inspection duration 159 5.96 44.38"
    elements = [el("p", 1, 0, paragraph), el("fig", 1, 1, visual, type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    relation = _types(result, "describes")[0]
    assert len(relation.evidence["numeric_matches"]) == 3
    assert all(item["unit_compatibility"].startswith("context_labels")
               for item in relation.evidence["numeric_matches"])


def test_chart_series_label_after_number_is_not_treated_as_a_unit(tmp_path):
    paragraph = "Agent 25 averaged 70.92 cases and Agent 26 averaged 66.26 cases."
    visual = "Average weekly cases Agent 25 70.92 Agent 26 66.26"
    elements = [el("p", 1, 0, paragraph), el("fig", 1, 1, visual, type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    matches = _types(result, "describes")[0].evidence["numeric_matches"]
    assert [item["normalized_value"] for item in matches] == ["70.92", "66.26"]


def test_explicit_reference_describes_numbered_figure(tmp_path):
    elements = [el("cap", 1, 0, "Figure 2", bbox=[20, 20, 100, 40]),
                el("fig", 1, 1, "Synthetic chart", type="figure", bbox=[20, 50, 200, 200]),
                el("p", 1, 2, "As shown in Figure 2, the synthetic trend changes.")]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"},
                           sections={"cap": "s1", "fig": "s1", "p": "s1"})
    assert any(item.source_id == "p" and item.target_id == "fig" for item in _types(result, "describes"))


def test_per_mille_is_not_equivalent_to_percent(tmp_path):
    elements = [el("p", 1, 0, "Rates are 5‰ and 7‰"),
                el("fig", 1, 1, "5%\n7%", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")


def test_percent_and_scalar_with_same_digits_are_not_equivalent(tmp_path):
    elements = [el("p", 1, 0, "Values 10% and 20%"), el("fig", 1, 1, "10 items\n20 items", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")


def test_topic_words_alone_do_not_describe_visual(tmp_path):
    elements = [el("p", 1, 0, "Traffic statistics changed during the synthetic year."),
                el("fig", 1, 1, "Traffic statistics\n10", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")
    assert any(item.get("reason") == "title contains only generic topic terms"
               for item in result.evidence_audit)


def test_numeric_agreement_does_not_link_distant_pages_in_same_section(tmp_path):
    elements = [el("p", 1, 0, "Values 3 and 4"),
                el("fig", 5, 0, "Chart values 3 and 4", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not _types(result, "describes")
    assert not any(item["code"] == "ambiguous_description_target" for item in result.findings)


def test_ambiguous_numeric_visuals_remain_unlinked(tmp_path):
    elements = [el("p", 1, 0, "The values are 10% and 20%."),
                el("a", 1, 1, "10%\n20%", type="figure"), el("b", 1, 2, "10%\n20%", type="figure")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "a": "s1", "b": "s1"})
    assert not _types(result, "describes")
    assert any(item["code"] == "ambiguous_description_target" for item in result.findings)


def test_table_continuation_with_repeated_headers(tmp_path):
    elements = [el("t1", 1, 0, "A\tB\n1\t2", type="table"),
                el("t2", 2, 0, "A\tB\n3\t4", type="table")]
    result, _, _ = _build(tmp_path, elements, sections={"t1": "s1", "t2": "s1"})
    assert len(_types(result, "table_continuation")) == 2
    logical = [node for node in result.nodes.values() if node.node_type == "logical_table"]
    assert logical[0].provenance["ordered_parts"] == ["t1", "t2"]


def test_table_continuation_supports_more_than_two_pages(tmp_path):
    elements = [el("t1", 1, 0, "A\tB\n1\t2", type="table"),
                el("t2", 2, 0, "A\tB\n3\t4", type="table"),
                el("t3", 3, 0, "A\tB\n5\t6", type="table")]
    result, _, _ = _build(tmp_path, elements, sections={"t1": "s1", "t2": "s1", "t3": "s1"})
    logical = [node for node in result.nodes.values() if node.node_type == "logical_table"]
    assert logical[0].provenance["ordered_parts"] == ["t1", "t2", "t3"]
    assert len(_types(result, "table_continuation")) == 3


def test_markdown_table_continuation_recognizes_repeated_header_after_note(tmp_path):
    header = "| A | B |\n| --- | --- |"
    elements = [el("t1", 1, 0, f"{header}\n| 1 | 2 |", type="table"),
                el("t2", 2, 0, f"_(table continued from page 1; header repeated)_\n\n{header}\n| 3 | 4 |",
                   type="table")]
    result, _, _ = _build(tmp_path, elements, sections={"t1": "s1", "t2": "s1"})
    logical = next(node for node in result.nodes.values() if node.node_type == "logical_table")
    assert logical.provenance["repeated_header_between_parts"] == [True]
    assert logical.provenance["pair_evidence"][0]["column_counts"] == [2, 2]


def test_similar_adjacent_tables_remain_separate_without_enough_evidence(tmp_path):
    elements = [el("t1", 1, 0, "A\tB\n1\t2", type="table"),
                el("t2", 2, 0, "C\tD\n3\t4", type="table")]
    result, _, _ = _build(tmp_path, elements)
    assert not _types(result, "table_continuation")


def test_composite_parent_from_source_provenance(tmp_path):
    elements = [el("parent", 1, 0, "composite", type="figure"),
                el("part", 1, 1, "part", type="figure", state="included_in_composite",
                   provenance={"composite_parent_id": "parent"})]
    result, _, _ = _build(tmp_path, elements)
    assert _types(result, "composite_part")[0].target_id == "parent"


def test_pending_empty_visual_uses_saved_layout_containment_without_inventing_text(tmp_path):
    decision = {"type": "containment", "decision": "contained; outer preferred", "contained_in": 2}
    elements = [
        el("parent", 1, 0, "outer visual content", type="figure", provenance={"job_id": "p001_R0"}),
        el("part", 1, 1, "", type="figure", state="pending_review",
           provenance={"job_id": "p001_R1", "decisions": [decision]}),
    ]
    layout = {"pages": [{"page": 1, "selection": {"regions": [
        {"region_id": "p1_R0", "job_id": "p001_R0", "block_index": 2, "label": "chart"},
        {"region_id": "p1_R1", "job_id": "p001_R1", "block_index": 5, "label": "chart"},
    ], "others": []}}]}
    result, _, _ = _build(tmp_path, elements, layout=layout)
    relation = _types(result, "composite_part")[0]
    assert (relation.source_id, relation.target_id) == ("part", "parent")
    assert relation.method == "saved_layout_containment_decision"
    assert relation.evidence["saved_decision"] == decision
    assert result.nodes["part"].text == ""
    assert "source_pending_review" in result.nodes["part"].review_flags
    part = next(item for item in result.visuals if item["visual_id"] == "part")
    assert part["composite"]["parent"] == "parent"
    assert part["rag_flags"] == ["needs_review"]


def test_saved_split_container_links_inner_visuals_to_empty_parent(tmp_path):
    split = {"type": "containment", "decision": "split into inner visuals", "inner": [3, 4]}
    elements = [
        el("container", 1, 0, "", type="figure", state="pending_review",
           provenance={"job_id": "p1_R0", "decisions": split}),
        el("left", 1, 1, "left", type="figure", provenance={"job_id": "p1_R1"}),
        el("right", 1, 2, "right", type="figure", provenance={"job_id": "p1_R2"}),
    ]
    layout = {"pages": [{"page": 1, "selection": {"regions": [
        {"region_id": "p1_R0", "block_index": 1, "label": "image"},
        {"region_id": "p1_R1", "block_index": 3, "label": "image"},
        {"region_id": "p1_R2", "block_index": 4, "label": "image"},
    ], "others": []}}]}
    result, _, _ = _build(tmp_path, elements, layout=layout)
    pairs = {(item.source_id, item.target_id) for item in _types(result, "composite_part")}
    assert pairs == {("left", "container"), ("right", "container")}
    assert result.nodes["container"].text == ""


def test_unresolved_composite_parent_is_review_finding(tmp_path):
    elements = [el("part", 1, 0, "part", type="figure", state="included_in_composite")]
    result, _, _ = _build(tmp_path, elements)
    assert not _types(result, "composite_part")
    assert any(item["code"] == "unresolved_composite_parent" for item in result.findings)


def test_aside_adjacent_to_map_uses_geometry(tmp_path):
    elements = [el("aside", 1, 0, "Note", bbox=[10, 100, 70, 180]),
                el("map", 1, 1, "map", type="figure", bbox=[75, 90, 350, 240])]
    result, _, _ = _build(tmp_path, elements, roles={"aside": "aside_label"},
                           sections={"aside": "s1", "map": "s1"})
    assert _types(result, "aside_of")[0].target_id == "map"


@pytest.mark.parametrize("markers", [("• Parent:", "  - Child"), ("1. Parent:", "  a. Child")])
def test_nested_bullet_and_numbered_lists(tmp_path, markers):
    elements = [el("i1", 1, 0, markers[0]), el("i2", 1, 1, markers[1])]
    saved_list = {"list_id": "l001", "section_id": "s1", "intro_unit_id": None,
                  "items": [{"unit_id": "i1", "element_id": "i1", "page": 1, "marker": markers[0].split()[0]},
                            {"unit_id": "i2", "element_id": "i2", "page": 1, "marker": markers[1].split()[0]}],
                  "pages": [1], "crosses_pages": False, "possible_continuations": []}
    result, _, _ = _build(tmp_path, elements, roles={"i1": "list_item", "i2": "list_item"},
                           sections={"i1": "s1", "i2": "s1"}, lists=[saved_list])
    assert _types(result, "list_nesting")


def test_multiple_list_nesting_levels_and_intro_are_preserved(tmp_path):
    elements = [el("intro", 1, 0, "Examples:"), el("i1", 1, 1, "• Parent:"),
                el("i2", 1, 2, "  - Child:"), el("i3", 1, 3, "    1. Grandchild")]
    saved_list = {"list_id": "l001", "section_id": "s1", "intro_unit_id": "intro",
                  "items": [{"unit_id": "i1", "element_id": "i1", "page": 1, "marker": "•"},
                            {"unit_id": "i2", "element_id": "i2", "page": 1, "marker": "-"},
                            {"unit_id": "i3", "element_id": "i3", "page": 1, "marker": "1."}],
                  "pages": [1], "crosses_pages": False, "possible_continuations": []}
    result, _, _ = _build(tmp_path, elements, roles={"intro": "list_intro", "i1": "list_item",
                                                      "i2": "list_item", "i3": "list_item"},
                           sections={item["id"]: "s1" for item in elements}, lists=[saved_list])
    assert len(_types(result, "list_nesting")) == 2
    assert _types(result, "introduces_list")[0].source_id == "intro"


def test_rtl_marker_change_colon_and_geometry_support_nesting(tmp_path):
    elements = [el("parent", 1, 0, "• أمثلة:", bbox=[350, 100, 520, 120]),
                el("child1", 1, 1, "- العنصر الأول", bbox=[280, 130, 485, 150]),
                el("child2", 1, 2, "- العنصر الثاني", bbox=[275, 160, 485, 180]),
                el("next_root", 2, 0, "• عنصر رئيسي جديد", bbox=[350, 100, 520, 120])]
    saved_list = {"list_id": "l001", "section_id": "s1", "intro_unit_id": None,
                  "items": [{"unit_id": "parent", "element_id": "parent", "page": 1, "marker": "•"},
                            {"unit_id": "child1", "element_id": "child1", "page": 1, "marker": "-"},
                            {"unit_id": "child2", "element_id": "child2", "page": 1, "marker": "-"},
                            {"unit_id": "next_root", "element_id": "next_root", "page": 2, "marker": "•"}],
                  "pages": [1, 2], "crosses_pages": True, "possible_continuations": []}
    result, _, _ = _build(tmp_path, elements, roles={item["id"]: "list_item" for item in elements},
                           sections={item["id"]: "s1" for item in elements}, lists=[saved_list])
    relations = _types(result, "list_nesting")
    assert {(item.source_id, item.target_id) for item in relations} == {
        ("child1", "parent"), ("child2", "parent"),
    }


def test_list_crossing_pages_is_preserved(tmp_path):
    elements = [el("i1", 1, 0, "• First"), el("i2", 2, 0, "• Second")]
    saved_list = {"list_id": "l001", "section_id": None, "intro_unit_id": None,
                  "items": [{"unit_id": "i1", "element_id": "i1", "page": 1, "marker": "•"},
                            {"unit_id": "i2", "element_id": "i2", "page": 2, "marker": "•"}],
                  "pages": [1, 2], "crosses_pages": True, "possible_continuations": []}
    result, _, _ = _build(tmp_path, elements, roles={"i1": "list_item", "i2": "list_item"}, lists=[saved_list])
    assert result.lists["nodes"][0]["crosses_pages"] is True


def test_merged_rtl_end_markers_create_verbatim_spans(tmp_path):
    text = "البند الأول -\nالبند الثاني -"
    result, _, _ = _build(tmp_path, [el("merged", 1, 0, text)], roles={"merged": "list_item"})
    spans = result.lists["derived_spans"]
    assert len(spans) == 2
    assert all(text[item["start"]:item["end"]] == item["text"] for item in spans)


def test_hyphenated_prose_is_not_split(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "A well-tested end-to-end synthetic sentence.")])
    assert not result.lists["derived_spans"]


def test_author_credit_metadata_near_document_end(tmp_path):
    elements = [el("body", 1, 0, "Synthetic body"), el("credit", 2, 0, "Prepared by: Test Author")]
    result, _, _ = _build(tmp_path, elements)
    assert _types(result, "document_metadata")[0].source_id.startswith("metadata_")


def test_scanned_figure_title_label_is_caption_signal(tmp_path):
    elements = [el("title", 1, 0, "Synthetic chart", bbox=[20, 20, 200, 40], label="figure_title"),
                el("chart", 1, 1, "chart", type="figure", bbox=[20, 50, 300, 250])]
    result, _, _ = _build(tmp_path, elements)
    assert _types(result, "caption_of")


def test_explicit_missing_visual_metadata_creates_placeholder(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "body")], manifest_updates={
        "missing_visuals": [{"page": 1, "reason": "missing_visual: crop was explicitly skipped", "region_id": "r1"}],
    })
    assert _types(result, "known_missing")


def test_no_missing_visual_is_fabricated(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "body")])
    assert not _types(result, "known_missing")


def test_needs_review_state_is_preserved(tmp_path):
    result, _, _ = _build(tmp_path, [el("fig", 1, 0, "chart", type="figure", state="needs_review")])
    assert result.visuals[0]["validation_state"] == "needs_review"
    assert "source_needs_review" in result.nodes["fig"].review_flags


def test_relationship_confidence_is_capped_when_source_needs_review(tmp_path):
    elements = [el("p", 1, 0, "Values are 10% and 20%."),
                el("fig", 1, 1, "10%\n20%", type="figure", state="needs_review")]
    result, _, _ = _build(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    relation = _types(result, "describes")[0]
    assert relation.confidence == "medium"
    assert relation.review_recommended is True


def test_review_template_human_fields_are_blank(tmp_path):
    elements = [el("cap", 1, 0, "Figure 1", bbox=[20, 20, 100, 40]),
                el("fig", 1, 1, "chart", type="figure", bbox=[20, 50, 200, 200])]
    result, _, _ = _build(tmp_path, elements, roles={"cap": "caption"})
    output, _ = write_relation_outputs(result, out_root=tmp_path / "outputs", stamp="fixed", read_only=False)
    template = make_template(output, {1}, stamp="fixed")
    with template.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["human_verdict"] == "" and row["human_note"] == "" for row in rows)


def test_input_hashes_remain_unchanged(tmp_path):
    elements = [el("p", 1, 0, "Synthetic body")]
    result, run, structure = _build(tmp_path, elements)
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [run / "document.json", structure / "assignments.json"]}
    write_relation_outputs(result, out_root=tmp_path / "outputs", stamp="one", read_only=False)
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_write_once_output_directory(tmp_path):
    result, _, _ = _build(tmp_path, [el("p", 1, 0, "Synthetic body")])
    write_relation_outputs(result, out_root=tmp_path / "outputs", stamp="same", read_only=False)
    with pytest.raises(OutputExistsError):
        write_relation_outputs(result, out_root=tmp_path / "outputs", stamp="same", read_only=False)


def test_new_relationships_and_evidence_audit_are_deterministic_and_stably_ordered(tmp_path):
    elements = [el("p", 1, 0, "Feature Importance chart values are 4K and 3%."),
                el("fig", 1, 1, "Feature Importance\n4000 and 3.01%", type="figure")]
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir(); second_root.mkdir()
    first, _, _ = _build(first_root, elements, sections={"p": "s1", "fig": "s1"})
    second, _, _ = _build(second_root, elements, sections={"p": "s1", "fig": "s1"})
    assert first.relationship_dicts() == second.relationship_dicts()
    assert first.evidence_audit == second.evidence_audit
    assert first.relationships == sorted(
        first.relationships,
        key=lambda item: (item.type, item.source_page or 0, item.source_id, item.target_id),
    )

