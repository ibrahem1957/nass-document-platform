"""Validation rules detect corrupted structures (findings are reported, never fixed silently)."""

from __future__ import annotations

import copy

from docprep.pipeline import build_structure
from docprep.tree import TreeResult
from docprep.validation import validate

from .helpers import el, make_config, write_run

TEXT = "Intro sentence one. 2) Second topic: Details about the second topic follow here. More details."


def built(tmp_path):
    elements = [el("p1_e0", 1, 0, TEXT, type="page_text", bbox=[0, 0, 612, 792], source="qari_full_page"),
                el("p2_e0", 2, 0, "Overview", type="heading", bbox=[72, 80, 200, 96]),
                el("p2_e1", 2, 1, "A synthetic paragraph about the overview of this test document.",
                   bbox=[72, 100, 540, 114])]
    run = write_run(tmp_path, "run_valid", elements, routes={1: "SCANNED", 2: "DIGITAL"})
    return build_structure(run, make_config())


def check(result, tree=None, spans=None, hashes_after=None):
    return {f["code"]: f["severity"] for f in validate(
        result.inputs.elements, result.units, spans if spans is not None else result.spans, result.candidates,
        tree or result.tree, result.inputs.page_count, result.inputs.input_hashes,
        hashes_after if hashes_after is not None else result.hashes_after)}


def test_clean_result_has_no_errors(tmp_path):
    result = built(tmp_path)
    assert all(severity == "ok" for severity in check(result).values())


def test_duplicate_assignment_is_reported(tmp_path):
    result = built(tmp_path)
    tree = TreeResult(result.tree.sections, result.tree.lists, result.tree.assignments + [result.tree.assignments[0]], [])
    assert check(result, tree=tree)["assignment_uniqueness"] == "error"


def test_lost_span_text_is_reported(tmp_path):
    result = built(tmp_path)
    spans = copy.deepcopy(result.spans)
    spans["p1_e0"][-1].text = spans["p1_e0"][-1].text[:-1]
    spans["p1_e0"][-1].end -= 1
    assert check(result, spans=spans)["span_coverage"] == "error"


def test_cycle_in_hierarchy_is_reported(tmp_path):
    result = built(tmp_path)
    sections = copy.deepcopy(result.tree.sections)
    assert len(sections) >= 2
    sections[0].parent, sections[1].parent = sections[1].section_id, sections[0].section_id
    tree = TreeResult(sections, result.tree.lists, result.tree.assignments, [])
    assert check(result, tree=tree)["hierarchy_acyclic"] == "error"


def test_unmarked_low_confidence_assignment_is_reported(tmp_path):
    result = built(tmp_path)
    assignments = copy.deepcopy(result.tree.assignments)
    target = next(a for a in assignments if a.section_id and a.role != "heading")
    target.confidence, target.review_recommended = "low", False
    tree = TreeResult(result.tree.sections, result.tree.lists, assignments, [])
    assert check(result, tree=tree)["ambiguity_marked"] == "error"


def test_changed_input_is_reported(tmp_path):
    result = built(tmp_path)
    changed = dict(result.hashes_after, **{"document.json": "0" * 64})
    assert check(result, hashes_after=changed)["inputs_unchanged"] == "error"


def test_section_heading_must_be_source_text(tmp_path):
    result = built(tmp_path)
    sections = copy.deepcopy(result.tree.sections)
    sections[0].title_parts = ["Invented heading"]
    tree = TreeResult(sections, result.tree.lists, result.tree.assignments, [])
    assert check(result, tree=tree)["section_headings"] == "error"
