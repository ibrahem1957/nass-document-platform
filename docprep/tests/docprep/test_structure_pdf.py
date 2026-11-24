"""Typography-driven structure on synthetic PDFs generated with PyMuPDF (English, base-14 fonts)."""

from __future__ import annotations

import pytest

from docprep.pipeline import build_structure

from .helpers import BODY, body_lines, make_config, pdf_run

pytest.importorskip("pymupdf")


def _cand(result, unit_id):
    return next(c for c in result.candidates if unit_id in c.unit_ids)


def _section_of(result, unit_id):
    return next(a.section_id for a in result.tree.assignments if a.unit_id == unit_id)


def test_font_size_title_and_three_levels(tmp_path):
    lines = ([{"id": "t", "page": 1, "y": 80, "text": "Annual Operations Review", "size": 24},
              {"id": "h1", "page": 1, "y": 120, "text": "Introduction", "size": 18}]
             + body_lines("a", 1, 145)
             + [{"id": "h2", "page": 1, "y": 210, "text": "Scope of the Data", "size": 14}]
             + body_lines("b", 1, 232)
             + [{"id": "h3", "page": 1, "y": 300, "text": "Network Changes", "size": 18}]
             + body_lines("c", 1, 325))
    result = build_structure(pdf_run(tmp_path, "run_sizes", lines), make_config())
    assert result.body.size == 11
    t, h1, h2, h3 = (_cand(result, u) for u in ("t", "h1", "h2", "h3"))
    assert all(c.status == "promoted" for c in (t, h1, h2, h3))
    assert t.is_document_title and t.level == 0
    assert h1.level == 1 and h3.level == 1 and h2.level == 2
    assert h1.level_confidence == "high"
    sections = {s.heading_unit_ids[0]: s for s in result.tree.sections}
    assert sections["h2"].parent == sections["h1"].section_id
    assert sections["h1"].parent == sections["t"].section_id
    assert _section_of(result, "c0") == sections["h3"].section_id
    assert any(s.name == "font_size_large" for s in h1.signals)
    assert not [f for f in result.findings if f["severity"] == "error"]


def test_heading_colour_separates_classes_and_same_size_bold_is_a_heading(tmp_path):
    lines = (body_lines("a", 1, 80, 2)
             + [{"id": "hA", "page": 1, "y": 130, "text": "Operations", "size": 11, "bold": True, "color": (0.8, 0, 0)}]
             + body_lines("b", 1, 150, 2)
             + [{"id": "hB", "page": 1, "y": 210, "text": "Daily service", "size": 11, "bold": True}]
             + body_lines("c", 1, 230, 2)
             + [{"id": "hA2", "page": 1, "y": 290, "text": "Maintenance", "size": 11, "bold": True,
                 "color": (0.8, 0, 0)}]
             + body_lines("d", 1, 310, 2))
    result = build_structure(pdf_run(tmp_path, "run_colour", lines), make_config())
    hA, hB, hA2 = (_cand(result, u) for u in ("hA", "hB", "hA2"))
    assert all(c.status == "promoted" for c in (hA, hB, hA2))
    assert any(s.name == "distinct_color" for s in hA.signals) and any(s.name == "bold" for s in hB.signals)
    assert hA.level == 1 and hA2.level == 1 and hB.level == 2
    assert hB.level_confidence == "medium"  # weight/colour only, same size
    sections = {s.heading_unit_ids[0]: s for s in result.tree.sections}
    assert sections["hB"].parent == sections["hA"].section_id and sections["hA2"].parent is None


def test_bold_colon_heading_is_promoted_from_text(tmp_path):
    lines = (body_lines("a", 1, 80, 2)
             + [{"id": "lab", "page": 1, "y": 130, "text": "Data description:", "size": 11, "bold": True}]
             + [{"id": "b", "page": 1, "y": 150, "text": BODY[1], "size": 11}])
    result = build_structure(pdf_run(tmp_path, "run_bold_colon", lines), make_config())
    lab = _cand(result, "lab")
    assert lab.status == "promoted"
    assert next(u for u in result.units if u.unit_id == "lab").element_type == "text"  # promoted from Nass 'text'
    assert _section_of(result, "b") == result.tree.sections[0].section_id


def test_wrapped_two_line_heading_is_merged(tmp_path):
    lines = (body_lines("a", 1, 80, 2)
             + [{"id": "w1", "page": 1, "y": 150, "text": "A Very Long Section Heading That", "size": 16},
                {"id": "w2", "page": 1, "y": 169.2, "text": "Continues On A Second Line", "size": 16}]
             + body_lines("b", 1, 195, 3))
    result = build_structure(pdf_run(tmp_path, "run_wrapped", lines), make_config())
    merged = _cand(result, "w1")
    assert merged.unit_ids == ["w1", "w2"] and merged.status == "promoted"
    assert merged.parts == ["A Very Long Section Heading That", "Continues On A Second Line"]
    assert any(s.name == "merged_wrapped_lines" for s in merged.signals)
    w2 = next(a for a in result.tree.assignments if a.unit_id == "w2")
    assert w2.role == "heading"


def test_conflicting_typography_and_semantic_evidence(tmp_path):
    lines = ([{"id": "h", "page": 1, "y": 100, "text": "Summary", "size": 18, "bold": True},
              {"id": "c", "page": 1, "y": 125, "text": "and more.", "size": 11}]
             + body_lines("b", 1, 150, 3))
    result = build_structure(pdf_run(tmp_path, "run_conflict", lines), make_config())
    h = _cand(result, "h")
    assert h.status == "promoted"
    assert h.semantic and h.semantic["conflicting"]
    assert any(s.startswith("semantic_relation") for s in h.conflicts) and h.review


def test_no_typography_on_image_only_pages(tmp_path):
    lines = [{"id": "h", "page": 1, "y": 100, "text": "Overview", "size": 18}] + body_lines("b", 1, 130, 2)
    result = build_structure(pdf_run(tmp_path, "run_image_only", lines, routes={1: "SCANNED"}), make_config())
    assert all(u.typography is None for u in result.units)
    assert result.body.size is None
