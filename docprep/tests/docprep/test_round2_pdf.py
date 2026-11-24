"""Round 2 regressions on synthetic PDFs (PyMuPDF base-14 fonts): body-style bonus, level normalization,
separate confidences and fake bold. Written for the user to run; not executed by the implementer."""

from __future__ import annotations

import pytest

from docprep.pipeline import build_structure

from .helpers import BODY, body_lines, make_config, pdf_run

pytest.importorskip("pymupdf")


def _c(result, unit_id):
    return next(c for c in result.candidates if unit_id in c.unit_ids)


def _sections(result):
    return {s.heading_unit_ids[0]: s for s in result.tree.sections}


def levels_run(tmp_path):
    lines = ([{"id": "A", "page": 1, "y": 70, "text": "Approach", "size": 16}] + body_lines("a", 1, 92, 2)
             + [{"id": "M", "page": 1, "y": 150, "text": "Methods", "size": 14}] + body_lines("m", 1, 170, 2)
             + [{"id": "S1", "page": 1, "y": 225, "text": "1) Data collection:", "size": 11}]
             + body_lines("s", 1, 245, 2)
             + [{"id": "R", "page": 1, "y": 300, "text": "Results", "size": 16}] + body_lines("r", 1, 322, 2)
             + [{"id": "N1", "page": 1, "y": 380, "text": "1) Time indicators:", "size": 11}]
             + body_lines("n", 1, 400, 2)
             + [{"id": "N2", "page": 1, "y": 455, "text": "2) Location indicators:", "size": 11}]
             + body_lines("k", 1, 475, 2))
    return build_structure(pdf_run(tmp_path, "run_levels", lines), make_config())


def test_unsupported_level_gap_is_normalized_and_genuine_nesting_kept(tmp_path):
    result = levels_run(tmp_path)
    s = _sections(result)
    assert (s["A"].level, s["M"].level, s["S1"].level) == (1, 2, 3)   # genuine nesting preserved
    assert s["S1"].raw_level == 3 and not s["S1"].level_notes
    assert s["N1"].level == 2 and s["N1"].raw_level == 3 and s["N1"].parent == s["R"].section_id
    assert any("normalized from raw level 3 to 2" in n for n in s["N1"].level_notes)
    assert s["N2"].level == 2 and s["N2"].parent == s["R"].section_id


def test_body_style_headings_get_no_style_bonus_and_no_high_level_confidence(tmp_path):
    result = levels_run(tmp_path)
    for uid in ("S1", "N1", "N2"):
        c = _c(result, uid)
        assert c.status == "promoted"                                  # numbering keeps them detectable
        names = {sig.name for sig in c.signals}
        assert "body_style" in names and "style_consistency" not in names
        assert c.level_confidence in ("medium", "low")                 # visually like body text
    assert _c(result, "A").level_confidence == "high"


def test_three_confidences_are_reported_separately(tmp_path):
    result = levels_run(tmp_path)
    section = _sections(result)["N1"]
    candidate = _c(result, "N1")
    assert candidate.confidence in ("high", "medium", "low") and candidate.level_confidence in ("medium", "low")
    content = [a for a in result.tree.assignments if a.section_id == section.section_id and a.role != "heading"]
    assert content and all(a.confidence in ("high", "medium", "low") for a in content)
    from docprep.tree import section_to_dict
    data = section_to_dict(section, candidate, {n.list_id: n for n in result.tree.lists})
    assert set(data["confidence"]) == {"heading_detection", "heading_level"}


def test_fill_and_stroke_heading_is_detected_as_fake_bold(tmp_path):
    lines = (body_lines("a", 1, 80, 3)
             + [{"id": "h", "page": 1, "y": 150, "text": "1) Overview:", "size": 11, "fake_bold": True}]
             + body_lines("b", 1, 170, 3))
    result = build_structure(pdf_run(tmp_path, "run_fake_bold", lines), make_config())
    units = {u.unit_id: u for u in result.units}
    assert units["h"].typography.fake_bold_status == "fill_and_stroke"
    assert units["a0"].typography.fake_bold_status == "none"
    h = _c(result, "h")
    assert any(s.name == "fake_bold" and 0 < s.contribution <= 0.05 for s in h.signals)
    assert h.status == "promoted"  # numbering and colon are the main evidence


def test_fill_and_stroke_list_introduction_is_not_promoted(tmp_path):
    lines = (body_lines("a", 1, 80, 3)
             + [{"id": "intro", "page": 1, "y": 150, "text": "These records include the following fields:",
                 "size": 11, "fake_bold": True},
                {"id": "li1", "page": 1, "y": 168, "text": "• Date of the trip.", "size": 11, "x": 90},
                {"id": "li2", "page": 1, "y": 186, "text": "• Line and station.", "size": 11, "x": 90}]
             + body_lines("b", 1, 210, 2))
    result = build_structure(pdf_run(tmp_path, "run_fake_bold_intro", lines), make_config())
    intro = _c(result, "intro")
    assert intro.status != "promoted"
    assert any(r.startswith("list introduction") for r in intro.exclusions)


def test_genuine_colon_heading_followed_by_a_list_remains_a_heading(tmp_path):
    lines = (body_lines("a", 1, 80, 3)
             + [{"id": "h", "page": 1, "y": 150, "text": "Data description:", "size": 15},
                {"id": "li1", "page": 1, "y": 172, "text": "• Date of the trip.", "size": 11, "x": 90},
                {"id": "li2", "page": 1, "y": 190, "text": "• Line and station.", "size": 11, "x": 90}]
             + [{"id": "b", "page": 1, "y": 215, "text": BODY[2], "size": 11}])
    result = build_structure(pdf_run(tmp_path, "run_heading_list", lines), make_config())
    h = _c(result, "h")
    assert h.status == "promoted" and not h.exclusions
    li1 = next(a for a in result.tree.assignments if a.unit_id == "li1")
    assert li1.section_id == _sections(result)["h"].section_id and li1.role == "list_item"
