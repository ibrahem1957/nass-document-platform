"""Round 3 regressions: semantic-warning fix, human review template and v2 evaluation, scanned heading
diagnostics. Synthetic data only. Written for the user to run; NOT EXECUTED by the implementer."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os

import pytest

from docprep.eval import REFERENCE_DIR
from docprep.evaluate import (
    EvaluationError,
    load_human_reference,
    load_structure,
    make_template,
    run,
)
from docprep.output import OutputExistsError
from docprep.pipeline import build_structure
from docprep.semantic import content_terms
from docprep.structure import run as structure_run
from docprep.tree import common_terms

from .helpers import el, make_config, write_run

LONG = "This paragraph explains the topic of the section in enough words to read as ordinary body text."
V1_SHA256 = "890c3b18cae080fa91bd7c968796673ceb0605b4c52e7543800382a911910954"

PRIVATE_V1 = pytest.mark.skipif(
    not (REFERENCE_DIR / "traffic_report_review_v1.json").is_file(),
    reason="private frozen v1 reference (built from a user-owned PDF) is not distributed with this repository; place it in docprep/docprep/eval/references/ to run this test",
)


def tree_hash(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _a(result):
    return {a.unit_id: a for a in result.tree.assignments}


# ------------------------------------------------------------------ semantic warnings

def generic_heading_run(tmp_path, name="run_generic"):
    elements = [
        el("h1", 1, 0, "Introduction", type="heading", bbox=[72, 70, 200, 86]),
        el("p1", 1, 1, "This report studies traffic accidents in the city and their causes over the year.",
           bbox=[72, 90, 540, 120]),
        el("h2", 1, 2, "2) Report questions about traffic accidents:", bbox=[72, 130, 360, 146]),
        el("p2", 1, 3, "The questions concern the time, the place and the severity of accidents.", bbox=[72, 150, 540, 180]),
    ]
    return write_run(tmp_path, name, elements)


def test_generic_single_word_heading_gives_no_false_warning(tmp_path):
    result = build_structure(generic_heading_run(tmp_path), make_config())
    p1 = _a(result)["p1"]
    assert not any(c.startswith("semantic warning") for c in p1.conflicting)
    assert p1.semantic_check["status"] == "skipped" and "own heading has" in p1.semantic_check["reason"]


def test_skipped_check_keeps_the_assignment_confidence(tmp_path):
    with_check = _a(build_structure(generic_heading_run(tmp_path, "run_a"), make_config()))["p1"]
    without = _a(build_structure(generic_heading_run(tmp_path, "run_b"),
                                 make_config(semantic={"assignment_warnings": False})))["p1"]
    assert with_check.confidence == without.confidence and with_check.deterministic == without.deterministic


def topic_run(tmp_path):
    elements = [
        el("h1", 1, 0, "1) Fleet maintenance plan:", bbox=[72, 70, 300, 86]),
        el("pa", 1, 1, "Network ridership figures were reviewed during the year.", bbox=[72, 90, 540, 120]),
        el("h2", 1, 2, "2) Network ridership indicators:", bbox=[72, 130, 320, 146]),
        el("pb", 1, 3, "Network ridership indicators rose in every month of the period.", bbox=[72, 150, 540, 180]),
        el("h3", 1, 4, "3) Network ridership forecast:", bbox=[72, 190, 320, 206]),
        el("pc", 1, 5, "Network ridership forecast values are shown in the next table.", bbox=[72, 210, 540, 240]),
    ]
    return write_run(tmp_path, "run_topic", elements)


def test_document_wide_topic_terms_are_excluded(tmp_path):
    result = build_structure(topic_run(tmp_path), make_config())
    units = {u.unit_id: u for u in result.units}
    common, _freq, n = common_terms(result.tree.sections, result.tree.assignments, units, make_config())
    assert n == 3 and {"network", "ridership"} <= common
    pa = _a(result)["pa"]
    # Without filtering "pa" would share network + ridership with the next heading and nothing with its own.
    assert not any(c.startswith("semantic warning") for c in pa.conflicting)
    assert pa.semantic_check["status"] == "skipped"


def test_stopwords_are_filtered_in_arabic_and_english():
    assert content_terms("the analysis of the data and their sources") == content_terms("analysis data sources")
    assert not content_terms("في هذه على التي من إلى")
    assert content_terms("في الدراسة") == content_terms("الدراسة")


def test_genuine_semantic_warning_flags_without_moving(tmp_path):
    elements = [
        el("h5", 1, 0, "5) Operational indicators:", bbox=[72, 70, 260, 86]),
        el("p1", 1, 1, "The branch handled most cases within the target time during the year.", bbox=[72, 90, 540, 120]),
        el("p2", 1, 2, "Inspector workload varies widely between inspectors and their weekly cases.",
           bbox=[72, 124, 540, 154]),
        el("h6", 1, 3, "6) Inspector workload and efficiency:", bbox=[72, 170, 320, 186]),
        el("p3", 1, 4, "The chart compares cases and processing time for every inspector.", bbox=[72, 190, 540, 220]),
    ]
    result = build_structure(write_run(tmp_path, "run_flag", elements), make_config())
    sections = {s.heading_unit_ids[0]: s for s in result.tree.sections}
    p2 = _a(result)["p2"]
    assert p2.section_id == sections["h5"].section_id
    assert p2.semantic_check["status"] == "flagged" and p2.semantic_check["shared_with_next_heading"]


# ------------------------------------------------------------------ human review template

def review_run(tmp_path):
    elements = [
        el("h", 1, 0, "Results", type="heading", bbox=[72, 70, 200, 86]),
        el("p", 1, 1, LONG, bbox=[72, 90, 540, 120]),
        el("intro", 1, 2, "These records include the following fields:", bbox=[72, 130, 360, 144]),
        el("li1", 1, 3, "• Date of the trip.", bbox=[90, 148, 300, 160]),
        el("li2", 1, 4, "• Line and station.", bbox=[90, 164, 300, 176]),
        el("h2", 2, 0, "Annex", type="heading", bbox=[72, 70, 200, 86]),
        el("q", 2, 1, LONG, bbox=[72, 90, 540, 120]),
    ]
    run_dir = write_run(tmp_path, "run_review", elements, page_count=2)
    out, _ = structure_run(run_dir, out_root=tmp_path / "results" / "document_structure", stamp="20260101T000000Z")
    return run_dir, out


def read_template_csv(folder):
    text = (folder / "review_template.csv").read_bytes().decode("utf-8")
    assert text.startswith("﻿")
    return list(csv.DictReader(io.StringIO(text[1:])))


def write_review(path, rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_text("﻿" + buffer.getvalue(), encoding="utf-8")
    return path


def test_template_has_blank_human_columns_and_no_overwrite(tmp_path):
    _run_dir, out = review_run(tmp_path)
    before = tree_hash(out)
    folder = make_template(out, {1}, stamp="20260101T000001Z")
    assert folder.parent.name == "document_structure_review"
    assert sorted(p.name for p in folder.iterdir()) == ["README_review.md", "manifest.json", "review_template.csv",
                                                        "review_template.json"]
    rows = read_template_csv(folder)
    assert rows and all(r["human_verdict"] == r["human_level"] == r["human_note"] == "" for r in rows)
    by_unit = {r["unit_ids"]: r for r in rows}
    assert by_unit["h"]["tool_verdict"] == "heading" and by_unit["h"]["source_text"] == "Results"
    assert by_unit["intro"]["tool_role"] == "list_intro"
    assert "li1" not in by_unit and all(r["page"] == "1" for r in rows)  # plain list items and page 2 omitted
    meta = json.loads((folder / "review_template.json").read_text(encoding="utf-8"))
    assert meta["pages"] == [1] and all(r["human_verdict"] == "" for r in meta["rows"])
    assert os.access(folder / "review_template.csv", os.W_OK)  # meant to be edited
    assert tree_hash(out) == before
    with pytest.raises(OutputExistsError):
        make_template(out, {1}, stamp="20260101T000001Z")


def test_valid_partial_human_review_is_evaluated_separately(tmp_path):
    _run_dir, out = review_run(tmp_path)
    rows = read_template_csv(make_template(out, {1}, stamp="t1"))
    for r in rows:
        if r["unit_ids"] == "h":
            r["human_verdict"], r["human_level"] = "heading", "1"
    (tmp_path / "human").mkdir()
    review = write_review(tmp_path / "human" / "review.csv", rows)
    reference, coverage = load_human_reference(review, load_structure(out))
    assert coverage["reviewed"] == 1 and coverage["unreviewed"] == len(rows) - 1
    assert reference["provenance"]["human_verified"] is True and len(reference["headings"]) == 1
    target = run(out, None, human_reference=review, out_root=tmp_path / "evals", stamp="e1")
    result = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
    assert result["human"]["metrics"]["headings"]["true_headings_detected"] == 1
    assert result["human"]["metrics"]["evaluated_confirmed_cases"] == 1
    assert "partial review" in (target / "evaluation.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("change,message", [
    ({"human_verdict": "maybe"}, "invalid human_verdict"),
    ({"human_verdict": "not_heading", "human_level": "1"}, "only allowed with human_verdict 'heading'"),
    ({"human_verdict": "heading", "human_level": "x"}, "must be an integer"),
    ({"human_verdict": "heading", "source_text": "Edited text"}, "must stay verbatim"),
    ({"human_level": "2"}, "without a human_verdict"),
])
def test_invalid_human_judgments_are_rejected(tmp_path, change, message):
    _run_dir, out = review_run(tmp_path)
    rows = read_template_csv(make_template(out, {1}, stamp="t2"))
    rows[0].update(change)
    (tmp_path / "human").mkdir()
    review = write_review(tmp_path / "human" / "bad.csv", rows)
    with pytest.raises(EvaluationError, match=message):
        load_human_reference(review, load_structure(out))


def test_uncertain_human_label_is_not_counted(tmp_path):
    _run_dir, out = review_run(tmp_path)
    rows = read_template_csv(make_template(out, {1}, stamp="t3"))
    for r in rows:
        r["human_verdict"] = "uncertain"
    (tmp_path / "human").mkdir()
    _reference, coverage = load_human_reference(write_review(tmp_path / "human" / "u.csv", rows), load_structure(out))
    assert coverage["uncertain"] == len(rows) and coverage["confirmed"] == 0
    target = run(out, None, human_reference=tmp_path / "human" / "u.csv", out_root=tmp_path / "evals", stamp="e2")
    m = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))["human"]["metrics"]
    assert m["evaluated_confirmed_cases"] == 0 and m["headings"]["precision"] is None


def test_human_labels_override_v1_only_where_labelled(tmp_path):
    _run_dir, out = review_run(tmp_path)
    rows = read_template_csv(make_template(out, {1}, stamp="t4"))
    for r in rows:
        if r["unit_ids"] == "h":
            r["human_verdict"], r["human_level"] = "heading", "1"
    (tmp_path / "human").mkdir()
    review = write_review(tmp_path / "human" / "r.csv", rows)
    (tmp_path / "v1").mkdir()
    v1 = tmp_path / "v1" / "v1.json"
    v1.write_text(json.dumps({
        "kind": "docprep_structure_reference", "reference_id": "synthetic_v1", "frozen": False,
        "source": {"run_id": "run_review"}, "provenance": {"human_verified": False},
        "headings": [
            {"source_element_ids": ["h"], "text": "Results", "page": 1, "expected_verdict": "not_heading",
             "status": "confirmed"},
            {"source_element_ids": ["intro"], "text": "x", "page": 1, "expected_verdict": "not_heading",
             "status": "confirmed"}],
        "assignments": [], "continuations": []}), encoding="utf-8")
    v1_hash = hashlib.sha256(v1.read_bytes()).hexdigest()
    target = run(out, v1, human_reference=review, out_root=tmp_path / "evals", stamp="e3")
    result = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
    comp = result["combined_headings_composition"]
    assert comp == {"from_human": 1, "from_v1": 1, "replaced_v1_labels": 1}
    combined = {c["key"]: c for c in result["combined_headings"]["cases"]}
    assert combined["heading:h"]["expected"] == "heading" and combined["heading:intro"]["source"] == "v1"
    assert [d["key"] for d in result["disagreements"]] == ["heading:h"]
    assert result["current"]["metrics"]["headings"]["false_promotions"] == 1   # v1 view kept separately
    assert hashlib.sha256(v1.read_bytes()).hexdigest() == v1_hash


@PRIVATE_V1
def test_frozen_v1_reference_is_still_unchanged():
    path = REFERENCE_DIR / "traffic_report_review_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == V1_SHA256


def test_inputs_and_previous_outputs_are_not_modified(tmp_path):
    run_dir, out = review_run(tmp_path)
    run_before, out_before = tree_hash(run_dir), tree_hash(out)
    rows = read_template_csv(make_template(out, {1, 2}, stamp="t5"))
    (tmp_path / "human").mkdir()
    review = write_review(tmp_path / "human" / "r.csv", rows)
    run(out, None, human_reference=review, out_root=tmp_path / "evals", stamp="e4")
    assert tree_hash(run_dir) == run_before and tree_hash(out) == out_before


# ------------------------------------------------------------------ scanned heading diagnostics

def scanned_layout(boxes):
    return {"pages": [{"page": 1, "route": "SCANNED", "page_image": {"px": [1700, 2200], "bbox_pt": [0.0, 0.0, 612.0, 792.0]},
                       "boxes": boxes}]}


def box(label, px, index, score=0.8):
    return {"label": label, "bbox_px": px, "score": score, "cls_id": 0, "block_index": index, "warnings": []}


def scanned_run(tmp_path, name, text, blocks, boxes, extra=()):
    elements = [el("p1_e0", 1, 0, text, type="page_text", bbox=[0, 0, 612, 792], source="qari_full_page",
                   body_segment={"segment": 0, "text_blocks": blocks})] + list(extra)
    return write_run(tmp_path, name, elements, routes={1: "SCANNED"}, layout=scanned_layout(boxes))


def test_paragraph_title_matched_to_a_derived_span_by_order_and_role(tmp_path):
    run_dir = scanned_run(tmp_path, "run_diag_seq", "Fleet summary: The fleet grew during the year.", [5, 1],
                          [box("paragraph_title", [300, 200, 1400, 260], 5), box("text", [200, 280, 1500, 600], 1)])
    result = build_structure(run_dir, make_config())
    (d,) = result.layout_diagnostics
    assert d.layout_label == "paragraph_title" and d.association == "approximate" and d.match_confidence == "medium"
    assert d.span_id == "p1_e0_span_001" and d.extracted_text == "Fleet summary: "
    assert d.became_heading and d.section_id
    assert result.units[0].bbox is None  # no coordinates invented for the span itself


def test_single_block_segment_is_a_verified_association(tmp_path):
    run_dir = scanned_run(tmp_path, "run_diag_one", "Network overview", [3],
                          [box("paragraph_title", [300, 200, 1400, 260], 3)])
    (d,) = build_structure(run_dir, make_config()).layout_diagnostics
    assert d.association == "verified" and d.element_id == "p1_e0" and d.extracted_text == "Network overview"


def test_title_region_without_a_reliable_match(tmp_path):
    figure = el("p1_e1", 1, 1, "chart text", type="figure", bbox=[100.0, 300.0, 500.0, 600.0], source="qari_crop")
    run_dir = scanned_run(tmp_path, "run_diag_none", "Body text of the page. Another sentence here.", [1],
                          [box("text", [200, 280, 1500, 600], 1),
                           box("paragraph_title", [400, 900, 1200, 960], 7),     # inside the figure crop
                           box("doc_title", [300, 100, 1400, 150], 9)],          # referenced by nothing
                          extra=[figure])
    diags = {d.block_index: d for d in build_structure(run_dir, make_config()).layout_diagnostics}
    assert diags[7].association == "none" and "inside figure p1_e1" in diags[7].reason_unmatched
    assert diags[9].association == "none" and "no extracted element references" in diags[9].reason_unmatched
    assert not diags[7].became_heading and diags[7].element_id is None


def test_diagnostics_appear_in_the_reports(tmp_path):
    run_dir = scanned_run(tmp_path, "run_diag_report", "Fleet summary: The fleet grew during the year.", [5, 1],
                          [box("paragraph_title", [300, 200, 1400, 260], 5), box("text", [200, 280, 1500, 600], 1)])
    out, _ = structure_run(run_dir, out_root=tmp_path / "out", stamp="20260101T000000Z")
    tree = json.loads((out / "section_tree.json").read_text(encoding="utf-8"))
    assert tree["layout_heading_diagnostics"][0]["association"] == "approximate"
    assert "Layout heading regions" in (out / "structure_report.md").read_text(encoding="utf-8")
    assert "Layout heading regions" in (out / "structure_report.html").read_text(encoding="utf-8")
    assert all(not os.access(p, os.W_OK) for p in out.iterdir())  # structure outputs stay write-once/read-only
