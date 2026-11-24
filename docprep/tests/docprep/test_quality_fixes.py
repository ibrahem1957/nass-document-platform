"""Regression tests for the systemic quality fixes (heading styles, hierarchy, OCR loops, script-aware
vocabulary, missing chart numbers, reliability dimensions, bidi bracket order, numeric qualifiers,
cross-script visual titles). All content is synthetic."""

from __future__ import annotations

import json

import pytest

from docprep.pipeline import build_structure
from docprep.quality_builder import QualityBuilder
from docprep.relation_text import match_numeric_tokens

from .helpers import body_lines, el, make_config, pdf_run, write_run
from .test_quality import _build, _find
from .test_relations import _build as _build_relations

GREEN = (0.0, 0.69, 0.31)
BLUE = (0.31, 0.51, 0.74)


def _cand(result, unit_id):
    return next(c for c in result.candidates if unit_id in c.unit_ids)


def _sections(result):
    return {s.heading_unit_ids[0]: s for s in result.tree.sections}


# ----------------------------------------------------------------------------------------- structure

def test_repeated_distinctive_style_promotes_body_size_coloured_headings(tmp_path):
    lines = ([{"id": "h1", "page": 1, "y": 80, "text": "Service Overview", "size": 12, "color": GREEN}]
             + body_lines("a", 1, 100)
             + [{"id": "h2", "page": 1, "y": 200, "text": "Network Notes", "size": 12, "color": GREEN}]
             + body_lines("b", 1, 220)
             + [{"id": "h3", "page": 2, "y": 80, "text": "Maintenance Plan", "size": 12, "color": GREEN}]
             + body_lines("c", 2, 100))
    result = build_structure(pdf_run(tmp_path, "run_repeated", lines, page_count=2), make_config())
    for unit_id in ("h1", "h2", "h3"):
        candidate = _cand(result, unit_id)
        assert candidate.status == "promoted"
        assert any(s.name == "repeated_heading_style" for s in candidate.signals)
        assert not any(s.name == "style_consistency" for s in candidate.signals)


def test_coloured_style_shared_with_body_sentences_gets_no_repeated_style_bonus(tmp_path):
    lines = []
    y = 80
    for page in (1, 2):
        for index in range(2):
            lines.append({"id": f"h{page}{index}", "page": page, "y": y, "text": f"Network Notes {page}{index}",
                          "size": 12, "color": GREEN})
            lines.append({"id": f"s{page}{index}", "page": page, "y": y + 20,
                          "text": "This coloured sentence is ordinary emphasised body text.", "size": 12,
                          "color": GREEN})
            lines += body_lines(f"b{page}{index}", page, y + 40, 2)
            y += 110
        y = 80
    result = build_structure(pdf_run(tmp_path, "run_shared_colour", lines, page_count=2), make_config())
    assert not any(s.name == "repeated_heading_style" for c in result.candidates for s in c.signals)


def test_numbering_marker_alone_on_first_line_is_not_a_second_line(tmp_path):
    run = write_run(tmp_path, "run_marker", [
        el("h", 1, 0, "1. \nTime Analysis Indicators"), el("p", 1, 1, "An ordinary sentence follows here."),
    ])
    result = build_structure(run, make_config())
    unit = next(u for u in result.units if u.unit_id == "h")
    assert unit.features["marker_only_first_line"] is True and unit.features["n_lines"] == 1
    other = next(u for u in result.units if u.unit_id == "p")
    assert other.features["marker_only_first_line"] is False


def test_numbered_heading_run_nests_under_unnumbered_same_style_heading(tmp_path):
    lines = ([{"id": "top", "page": 1, "y": 70, "text": "Operations Report", "size": 16, "color": GREEN}]
             + body_lines("a", 1, 90, 2)
             + [{"id": "rec", "page": 1, "y": 140, "text": "Recommendations", "size": 12, "color": GREEN},
                {"id": "r1", "page": 1, "y": 160, "text": "1. Strengthen Vehicle Maintenance", "size": 12,
                 "color": GREEN}]
             + body_lines("b", 1, 180, 2)
             + [{"id": "r2", "page": 1, "y": 230, "text": "2. Redistribute Field Teams", "size": 12,
                 "color": GREEN}]
             + body_lines("c", 1, 250, 2)
             + [{"id": "r3", "page": 2, "y": 80, "text": "3. Improve Secondary Roads", "size": 12, "color": GREEN}]
             + body_lines("d", 2, 100, 2))
    result = build_structure(pdf_run(tmp_path, "run_numbered_run", lines, page_count=2), make_config())
    rec = _cand(result, "rec")
    assert rec.status == "promoted"
    assert any(s.name == "introduces_numbered_sequence" for s in rec.signals)
    assert not any(s.name == "followed_by_same_style_candidate" for s in rec.signals)
    sections = _sections(result)
    for unit_id in ("r1", "r2", "r3"):
        assert _cand(result, unit_id).level == rec.level + 1
        assert sections[unit_id].parent == sections["rec"].section_id


def test_body_style_list_introduction_before_numbered_items_stays_protected(tmp_path):
    lines = ([{"id": "h", "page": 1, "y": 70, "text": "Methodology", "size": 16}]
             + [{"id": "intro", "page": 1, "y": 100, "text": "The stages of work included:", "size": 11},
                {"id": "i1", "page": 1, "y": 118, "text": "1. Cleaning the data and handling gaps.", "size": 11},
                {"id": "i2", "page": 1, "y": 136, "text": "2. Creating derived analytical variables.", "size": 11}]
             + body_lines("a", 1, 170, 3))
    result = build_structure(pdf_run(tmp_path, "run_list_intro", lines), make_config())
    intro = _cand(result, "intro")
    assert intro.status != "promoted"
    assert not any(s.name == "introduces_numbered_sequence" for s in intro.signals)
    assert all(_cand(result, u).status != "promoted" for u in ("i1", "i2"))


def test_title_tie_and_top_level_placement_after_title(tmp_path):
    lines = ([{"id": "t", "page": 1, "y": 70, "text": "Analysis of Network Factors", "size": 16, "color": GREEN},
              {"id": "intro", "page": 1, "y": 100, "text": "Introduction", "size": 14, "color": GREEN}]
             + body_lines("a", 1, 120)
             + [{"id": "res", "page": 2, "y": 70, "text": "Analysis Results", "size": 16, "color": GREEN}]
             + body_lines("b", 2, 90)
             + [{"id": "ai", "page": 2, "y": 170, "text": "Prediction System", "size": 16, "color": GREEN}]
             + body_lines("c", 2, 190)
             + [{"id": "sum", "page": 3, "y": 70, "text": "Summary", "size": 14, "color": GREEN}]
             + body_lines("d", 3, 90))
    result = build_structure(pdf_run(tmp_path, "run_title_tie", lines, page_count=3), make_config())
    title = _cand(result, "t")
    assert title.is_document_title and title.level == 0
    assert any("shared with later headings" in basis for basis in title.level_basis)
    sections = _sections(result)
    for unit_id in ("intro", "res", "ai", "sum"):
        assert sections[unit_id].level == 1
        assert sections[unit_id].parent == sections["t"].section_id


def test_colour_only_class_that_never_interleaves_shares_the_level(tmp_path):
    lines = ([{"id": "h_intro", "page": 1, "y": 70, "text": "Introduction", "size": 14, "color": GREEN}]
             + body_lines("a", 1, 90)
             + [{"id": "h_sum", "page": 1, "y": 170, "text": "Summary", "size": 14, "color": GREEN}]
             + body_lines("b", 1, 190)
             + [{"id": "att", "page": 2, "y": 70, "text": "Attachments", "size": 14, "color": BLUE}]
             + body_lines("c", 2, 90))
    result = build_structure(pdf_run(tmp_path, "run_colour_peer", lines, page_count=2), make_config())
    a2, att = _cand(result, "h_sum"), _cand(result, "att")
    assert att.status == "promoted" and att.level == a2.level
    assert any("differs by colour only" in basis for basis in att.level_basis)
    sections = _sections(result)
    assert sections["att"].parent != sections["h_sum"].section_id


def test_numbering_continuity_keeps_sibling_measured_one_point_smaller(tmp_path):
    lines = ([{"id": "top", "page": 1, "y": 70, "text": "Analysis Results", "size": 16, "color": GREEN}]
             + body_lines("a", 1, 90, 2))
    y = 140
    for number, size in ((1, 12), (2, 12), (3, 11), (4, 12)):
        lines.append({"id": f"n{number}", "page": 1, "y": y, "text": f"{number}. Indicator Group {number}",
                      "size": size, "color": GREEN})
        lines += body_lines(f"b{number}", 1, y + 20, 2)
        y += 80
    result = build_structure(pdf_run(tmp_path, "run_continuity", lines), make_config())
    levels = {u: _cand(result, u).level for u in ("n1", "n2", "n3", "n4")}
    assert len(set(levels.values())) == 1, levels
    assert any("numbering continuity" in basis for basis in _cand(result, "n3").level_basis)


# ----------------------------------------------------------------------------------------- quality

def test_uncertain_heading_audit_is_explicit_when_confidence_is_null(tmp_path):
    candidate = {"candidate_id": "c0005", "unit_ids": ["h"], "page": 1, "text": "Data Description",
                 "status": "rejected", "score": 0.43, "confidence": None, "ambiguous": True, "level": None,
                 "level_confidence": None, "level_ambiguous": False, "exclusions": [], "notes": [],
                 "signals": [{"name": "distinct_color", "contribution": 0.1, "detail": "colour", "category": "t"}]}
    result, *_ = _build(tmp_path, [el("h", 1, 0, "Data Description"), el("p", 1, 1, "Body text.")],
                        heading_candidates=[candidate])
    evidence = _find(result, "uncertain_heading_or_level")[0].evidence
    assert evidence["decision_status"] == "rejected" and evidence["score"] == 0.43
    assert evidence["detection_confidence"] is None
    assert evidence["detection_confidence_meaning"].startswith("not applicable")
    assert evidence["signals"] and evidence["uncertainty_reasons"]
    assert evidence["source_ids"] == ["h"] and evidence["text"] == "Data Description"


@pytest.mark.parametrize("phrase", ["يوجد إصابة يوجد إصابات لا", "with injury without injury"])
def test_ocr_repetition_loop_is_detected_without_changing_text_or_state(tmp_path, phrase):
    text = "Distribution title " + " ".join([phrase] * 4)
    result, inputs, *_ = _build(tmp_path, [el("f", 1, 0, text, type="figure")])
    finding = _find(result, "ocr_repetition_loop")[0]
    run = finding.evidence["runs"][0]
    assert run["repeat_count"] >= 3 and run["phrase_tokens"] >= 3
    assert text[run["start"]:run["end"]] == finding.offsets[0]["text"]
    assert finding.status == "needs_review" and finding.severity == "medium"
    record = next(item for item in result.inventory if item.source_id == "f")
    assert "ocr_repetition_loop" in record.rag_flags and record.evidence_reliability == "medium"
    assert record.validation_state == "ok" and inputs.run.elements[0].text == text


@pytest.mark.parametrize(("loop_stopped", "severity"), [(True, "high"), (False, "medium")])
def test_upstream_loop_flag_strengthens_repetition_finding(tmp_path, monkeypatch, loop_stopped, severity):
    from . import test_quality

    original = test_quality.write_run

    def write_with_raw(*args, **kwargs):
        run = original(*args, **kwargs)
        folder = run / "raw" / "job1"
        folder.mkdir(parents=True)
        (folder / "attempt_01.result.json").write_text(json.dumps({"response": {
            "retried": True, "final_attempt": 2, "final_flags": {"loop_stopped": loop_stopped, "hit_limit": False},
            "attempts": [{"loop_stopped": True}, {"loop_stopped": loop_stopped}]}}), encoding="utf-8")
        return run

    monkeypatch.setattr(test_quality, "write_run", write_with_raw)
    text = "Title " + " ".join(["alpha beta gamma"] * 3)
    result, *_ = _build(tmp_path, [el("f", 1, 0, text, type="figure", crop={"file": "crops/job1.png"})])
    finding = _find(result, "ocr_repetition_loop")[0]
    assert finding.severity == severity
    assert finding.evidence["upstream_ocr_record"]["record"] == "raw/job1/attempt_01.result.json"
    assert ("upstream_final_loop_stopped" in finding.evidence["signals"]) is loop_stopped
    record = next(item for item in result.inventory if item.source_id == "f")
    assert record.evidence_reliability == ("low" if loop_stopped else "medium")


@pytest.mark.parametrize(("text", "kind"), [
    ("North South East\nNorth South East\nNorth South East", "figure"),
    ("| North | South | East |\n| North | South | East |\n| North | South | East |", "table"),
    ("Low Medium High Low Medium High", "figure"),
])
def test_repetition_rule_avoids_legends_tables_and_two_repeats(tmp_path, text, kind):
    result, *_ = _build(tmp_path, [el("f", 1, 0, text, type=kind)])
    assert not _find(result, "ocr_repetition_loop")


def test_arabic_ocr_in_english_document_is_unverifiable_not_accused(tmp_path):
    english = "The weekly inspection workload is described in this synthetic English paragraph."
    result, *_ = _build(tmp_path, [el("p", 1, 0, english),
                                   el("f", 1, 1, "عدد حائلات الفحص الأسبوعية للمفتش", type="figure",
                                      source="qari_crop", model={"label": "ocr"})])
    note = _find(result, "unverifiable_script_vocabulary")[0]
    assert note.evidence["script"] == "arabic" and note.evidence["other_script_text_used"] is False
    assert note.severity == "info" and note.status == "unverified"
    assert not _find(result, "ocr_vocabulary_noise")
    record = next(item for item in result.inventory if item.source_id == "f")
    assert "ocr_noise" not in record.rag_flags


def test_same_script_body_text_suppresses_unverifiable_note(tmp_path):
    sentence = "تحليل الحوادث المرورية وأنماطها في المدينة يساعد الفرق الميدانية على الاستجابة "
    body = [el(f"p{i}", 1, i, sentence * 3) for i in range(8)]
    result, *_ = _build(tmp_path, body + [el("f", 1, 9, "توزع الحوادث حسب الفترة", type="figure",
                                              source="qari_crop", model={"label": "ocr"})])
    assert not _find(result, "unverifiable_script_vocabulary")
    assert not _find(result, "ocr_vocabulary_noise")


@pytest.mark.parametrize(("text", "state", "expected"), [
    ("عدد الحالات الأسبوعية للمفتش", "ok", True),
    ("Number of weekly cases per inspector", "ok", True),
    ("Number of weekly cases 10 20 30", "ok", False),
    ("خريطة المناطق الرئيسية في المدينة", "ok", False),
    ("عدد الحالات الأسبوعية للمفتش", "included_in_composite", False),
])
def test_probable_missing_numeric_evidence_needs_measure_terms(tmp_path, text, state, expected):
    result, *_ = _build(tmp_path, [el("f", 1, 0, text, type="figure", state=state)])
    found = _find(result, "probable_missing_numeric_evidence")
    assert bool(found) is expected
    if expected:
        assert found[0].evidence["numeric_token_count"] == 0 and found[0].status == "unverified"
        record = next(item for item in result.inventory if item.source_id == "f")
        assert "probable_missing_numeric_evidence" in record.rag_flags


def test_structural_uncertainty_does_not_lower_evidence_reliability(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "Ordinary body text."),
                                   el("q", 1, 1, "Reviewed text.", state="needs_review")],
                        assignments={"p": {"confidence": "low", "review_recommended": True,
                                           "conflicting": ["follows a near-miss heading candidate"]}})
    finding = _find(result, "uncertain_assignments")[0]
    assert finding.evidence["assignments"][0]["reasons"] == ["follows a near-miss heading candidate"]
    p = next(item for item in result.inventory if item.source_id == "p")
    assert p.evidence_reliability == "unassessed" and p.rag_flags == []
    assert p.reliability_dimensions["structural_placement"]["state"] == "uncertain"
    assert p.reliability_dimensions["source_text_reliability"]["state"] == "no_finding"
    assert p.reliability_dimensions["retrieval_readiness"]["state"] == "content_unflagged_placement_or_links_uncertain"
    q = next(item for item in result.inventory if item.source_id == "q")
    assert q.evidence_reliability == "medium" and "needs_review" in q.rag_flags
    assert any("needs_review" in reason for reason in q.evidence_reliability_reasons)


@pytest.mark.parametrize("text", ["خط العرض )GIS( بالدرجات", "يستخدم نظام GIS في تحليل المواقع"])
def test_bidi_bracket_order_requires_layout_evidence(tmp_path, text):
    result, *_ = _build(tmp_path, [el("t", 1, 0, text, bbox=[40, 40, 560, 80])])
    assert not _find(result, "bidi_order_suspect")


def test_bidi_bracket_order_with_pdf_glyph_geometry(tmp_path, monkeypatch):
    chars = [("ع", 520, 50, 528, 62), ("ر", 510, 50, 518, 62), ("(", 500, 50, 503, 62), ("G", 480, 50, 485, 62),
             ("I", 485, 50, 488, 62), ("S", 488, 50, 494, 62), (")", 476, 50, 479, 62)]
    monkeypatch.setattr(QualityBuilder, "_pdf_page_chars", lambda self, page: chars)
    result, *_ = _build(tmp_path, [el("t", 1, 0, "عر )GIS(", bbox=[470, 45, 530, 65])])
    finding = _find(result, "bidi_order_suspect")[0]
    assert finding.evidence["layout_evidence"]["rtl_context"] == "arabic_on_same_pdf_line"
    assert finding.offsets[0]["text"] == ")GIS("


def test_numeric_qualifiers_are_not_exact_equality():
    left = "Injuries exceeded 4,000 cases, more than 3% of accidents, taking about 5.96 minutes each."
    right = "cases 4K share 3.01% duration minutes 5.96"
    matches = {item["source_raw"]: item for item in match_numeric_tokens(left, right)}
    assert matches["3%"]["match_semantics"] == "bound_consistent"
    assert matches["3%"]["equality_asserted"] is False
    assert matches["3%"]["source_qualifier"]["relation"] == "greater_than"
    assert matches["4,000 cases"]["bound_checks"][0]["compared_display_half_width"] == "500"
    assert matches["5.96 minutes"]["match_semantics"] == "approximate_consistent"
    assert matches["3%"]["comparison_method"] == "rounded_percent"  # unchanged comparison evidence


def test_contradicted_bound_is_not_supporting_evidence_and_arabic_qualifiers():
    assert match_numeric_tokens("more than 3% of accidents", "share 2.9%") == []
    arabic = match_numeric_tokens("بنسبة تجاوزت 3% من الحوادث", "النسبة 3.01%")
    assert arabic and arabic[0]["source_qualifier"]["raw"] == "تجاوزت"
    plain = match_numeric_tokens("recorded 49.58% of cases", "49.58%")
    assert plain[0]["match_semantics"] == "exact_equality" and plain[0]["equality_asserted"] is True


def test_cross_script_title_evidence_keeps_title_null_with_limitation(tmp_path):
    elements = [el("heading", 1, 0, "Weekly Workload per Inspector"),
                el("fig", 1, 1, "عدد حالات الفحص الأسبوعية للمفتش", type="figure")]
    result, _, _ = _build_relations(tmp_path, elements, sections={"heading": "s1", "fig": "s1"},
                                    section_titles={"s1": "Weekly Workload per Inspector"},
                                    section_heading_ids={"s1": ["heading"]})
    visual = next(item for item in result.visuals if item["visual_id"] == "fig")
    assert visual["internal_title"] is None
    evaluation = visual["internal_title_evaluation"]
    assert "another script" in evaluation["reason"]
    assert evaluation["limitation"]["kind"] == "cross_script_title_evidence"


def test_same_script_title_rejection_reason_is_unchanged(tmp_path):
    elements = [el("heading", 1, 0, "Weekly Workload per Inspector"),
                el("fig", 1, 1, "Regional legend North South", type="figure")]
    result, _, _ = _build_relations(tmp_path, elements, sections={"heading": "s1", "fig": "s1"},
                                    section_titles={"s1": "Weekly Workload per Inspector"},
                                    section_heading_ids={"s1": ["heading"]})
    evaluation = next(item for item in result.visuals if item["visual_id"] == "fig")["internal_title_evaluation"]
    assert evaluation["reason"] == "single-line visual has no strong prefix match to independent title evidence"
    assert "limitation" not in evaluation


def test_json_roundtrip_of_new_inventory_dimensions(tmp_path):
    result, *_ = _build(tmp_path, [el("p", 1, 0, "Body text.")])
    record = next(item for item in result.inventory if item.source_id == "p").to_dict()
    assert set(record["reliability_dimensions"]) == {
        "source_text_reliability", "structural_placement", "relationship_confidence", "evidence_completeness",
        "retrieval_readiness"}
    json.dumps(record)


# ------------------------------------------------------------------- final structure / numeric corrections

def _parallel_run_lines(rec_item_x: float):
    """A report whose 'Results' section holds a numbered run directly under a 14pt heading, and whose later
    12pt 'Proposals' heading introduces a second run of the same style (12pt, indent rec_item_x)."""
    lines = [{"id": "t", "page": 1, "y": 70, "text": "Synthetic Network Review", "size": 16, "color": GREEN},
             {"id": "intro", "page": 1, "y": 100, "text": "Introduction", "size": 14, "color": GREEN}]
    lines += body_lines("a", 1, 120, 2)
    lines.append({"id": "res", "page": 1, "y": 180, "text": "Results", "size": 14, "color": GREEN})
    y = 200
    for number in (1, 2, 3):
        lines.append({"id": f"n{number}", "page": 1, "y": y, "text": f"{number}. Indicator Group {number}",
                      "size": 12, "color": GREEN, "x": 100})
        lines += body_lines(f"b{number}", 1, y + 20, 2)
        y += 70
    lines.append({"id": "sys", "page": 2, "y": 70, "text": "Prediction Tool", "size": 14, "color": GREEN})
    lines += body_lines("c", 2, 90, 2)
    lines.append({"id": "how", "page": 2, "y": 150, "text": "How It Works", "size": 12, "color": GREEN})
    lines += body_lines("d", 2, 170, 2)
    lines.append({"id": "rec", "page": 2, "y": 230, "text": "Proposals", "size": 12, "color": GREEN})
    y = 250
    for number in (1, 2, 3):
        lines.append({"id": f"r{number}", "page": 2, "y": y, "text": f"{number}. Proposal Area {number}",
                      "size": 12, "color": GREEN, "x": rec_item_x})
        lines += body_lines(f"e{number}", 2, y + 20, 2)
        y += 70
    lines.append({"id": "sum", "page": 3, "y": 70, "text": "Summary", "size": 14, "color": GREEN})
    lines += body_lines("f", 3, 90, 2)
    return lines


def test_parallel_numbered_runs_lift_same_class_introducer_to_major_section(tmp_path):
    result = build_structure(pdf_run(tmp_path, "run_parallel", _parallel_run_lines(100), page_count=3),
                             make_config())
    sections = _sections(result)
    rec, res = _cand(result, "rec"), _cand(result, "res")
    assert rec.level == res.level == sections["sys"].level
    assert sections["rec"].parent == sections["t"].section_id  # an independent major section
    assert sections["how"].parent == sections["sys"].section_id  # same-class sibling stays nested
    for number in (1, 2, 3):
        assert sections[f"r{number}"].parent == sections["rec"].section_id
        assert _cand(result, f"r{number}").level == _cand(result, f"n{number}").level
    assert rec.level_confidence == "low" and rec.level_conflicts
    assert any("parallel numbered runs" in basis for basis in rec.level_basis)


def test_parallel_run_rule_needs_matching_indent(tmp_path):
    result = build_structure(pdf_run(tmp_path, "run_parallel_indent", _parallel_run_lines(72), page_count=3),
                             make_config())
    sections = _sections(result)
    assert sections["rec"].parent == sections["sys"].section_id  # no evidence: typographic placement kept
    assert not any("parallel numbered runs" in basis for basis in _cand(result, "rec").level_basis)


def test_heading_audit_separates_support_conflict_absent_signals_and_parent(tmp_path):
    def cand(cid, unit, score, names, conf="medium"):
        return {"candidate_id": cid, "unit_ids": [unit], "page": 1, "text": unit, "status": "promoted",
                "score": score, "confidence": conf, "ambiguous": conf == "low", "level": 2,
                "level_confidence": "medium", "level_ambiguous": False, "exclusions": [], "notes": [],
                "style_key": [12.0, False, "#00b050"], "level_basis": ["typographic class"],
                "level_conflicts": [], "conflicts": [],
                "signals": [{"name": n, "contribution": c, "detail": n, "category": "t"} for n, c in names]}
    candidates = [cand("c1", "h1", 0.6, [("distinct_color", 0.1), ("fake_bold", 0.05)]),
                  cand("c2", "h2", 0.6, [("distinct_color", 0.1), ("fake_bold", 0.05)]),
                  cand("c3", "h3", 0.45, [("distinct_color", 0.1), ("followed_by_same_style_candidate", -0.05)],
                       conf="low")]
    sections = [{"section_id": "s1", "title": "Parent Heading", "level": 1, "parent": None,
                 "heading_candidate_id": "c0", "heading_unit_ids": ["p"], "content_unit_ids": []},
                {"section_id": "s2", "title": "h3", "level": 2, "raw_level": 2, "parent": "s1",
                 "heading_candidate_id": "c3", "heading_unit_ids": ["h3"], "content_unit_ids": [],
                 "level_notes": []}]
    elements = [el("p", 1, 0, "Parent Heading"), el("h1", 1, 1, "h1"), el("h2", 1, 2, "h2"), el("h3", 1, 3, "h3")]
    result, *_ = _build(tmp_path, elements, heading_candidates=candidates, sections=sections)
    evidence = next(f.evidence for f in _find(result, "uncertain_heading_or_level")
                    if f.evidence["candidate_id"] == "c3")
    assert any(item.startswith("distinct_color") for item in evidence["supporting_signals"])
    assert any(item.startswith("followed_by_same_style_candidate") for item in evidence["conflicting_signals"])
    assert any(item.startswith("fake_bold: present on 2 of 2") for item in evidence["absent_peer_signals"])
    assert "not evaluated" in evidence["confidence_explanation"]  # synthetic manifest records no thresholds
    assert evidence["placement"]["parent_section_id"] == "s1"
    assert evidence["placement"]["parent_title"] == "Parent Heading"


def test_numeric_agreement_distinguishes_proof_from_insufficient_precision():
    left = "Injuries exceeded 4,000 cases, more than 3% of accidents, taking about 5.96 minutes, at 5:00 PM."
    right = "cases 4K share 3.01% duration minutes 5.96 peak 5:00 PM"
    matches = {item["source_raw"]: item for item in match_numeric_tokens(left, right)}
    four_k = matches["4,000 cases"]
    assert four_k["numeric_agreement"] == "inequality_consistent_insufficient_precision"
    assert four_k["evidential_status"] == "inconclusive" and four_k["equality_asserted"] is False
    assert four_k["bound_checks"][0]["verdict"] == "consistent_not_proved"
    assert "3500" in four_k["bound_checks"][0]["reason"] and "4500" in four_k["bound_checks"][0]["reason"]
    assert matches["3%"]["numeric_agreement"] == "inequality_proved"
    assert matches["3%"]["evidential_status"] == "supports" and matches["3%"]["bound_checks"][0]["proved"]
    assert matches["5.96 minutes"]["numeric_agreement"] == "approximate_agreement"
    assert matches["5.96 minutes"]["equality_asserted"] is False
    assert matches["5:00 PM"]["numeric_agreement"] == "exact_agreement"


def test_open_bound_against_signed_label_is_unverifiable():
    match = match_numeric_tokens("vehicles aged 20+ years", "age group +20")[0]
    assert match["numeric_agreement"] == "inequality_unverifiable_qualified_value"
    assert match["evidential_status"] == "inconclusive" and match["equality_asserted"] is False
    arabic = match_numeric_tokens("بلغ عدد الحالات أكثر من 4 آلاف حالة", "عدد الحالات 4K")
    assert arabic and arabic[0]["evidential_status"] == "inconclusive"


def test_inconclusive_numeric_matches_do_not_count_toward_describes(tmp_path):
    paragraph = "Injuries exceeded 4,000 cases while 49.58% of cases occurred in the evening."
    visual = "cases 4K share 49.58%"
    elements = [el("p", 1, 0, paragraph), el("fig", 1, 1, visual, type="figure")]
    result, _, _ = _build_relations(tmp_path, elements, sections={"p": "s1", "fig": "s1"})
    assert not [r for r in result.relationships if r.type == "describes"]
    audit = [item for item in result.evidence_audit if item.get("target_id") == "fig"
             and item.get("status") == "rejected" and item.get("rule") == "numeric_only"]
    assert audit and "inconclusive" in audit[0]["reason"]
    # With a second conclusive value the relationship holds, and the inconclusive match stays recorded.
    elements = [el("p", 1, 0, paragraph + " The rate was 1.37%."),
                el("fig", 1, 1, visual + " rate 1.37%", type="figure")]
    result, _, _ = _build_relations(tmp_path / "b", elements, sections={"p": "s1", "fig": "s1"})
    relation = next(r for r in result.relationships if r.type == "describes")
    assert relation.evidence["conclusive_numeric_match_count"] == 2
    assert relation.evidence["inconclusive_numeric_match_count"] == 1
