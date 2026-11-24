"""Evaluation framework: frozen reference, confirmed-only metrics, write-once outputs. Synthetic structure
outputs are written directly as JSON (no pipeline run). Written for the user to run; not executed."""

from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from docprep.eval import REFERENCE_DIR
from docprep.evaluate import (
    EvaluationError,
    evaluate_structure,
    load_reference,
    load_structure,
    main,
    run,
)
from docprep.output import OutputExistsError
from docprep.structure import run as structure_run

from .helpers import el, write_run

V1 = REFERENCE_DIR / "traffic_report_review_v1.json"
V1_SHA256 = "890c3b18cae080fa91bd7c968796673ceb0605b4c52e7543800382a911910954"

PRIVATE_V1 = pytest.mark.skipif(
    not (REFERENCE_DIR / "traffic_report_review_v1.json").is_file(),
    reason="private frozen v1 reference (built from a user-owned PDF) is not distributed with this repository; place it in docprep/docprep/eval/references/ to run this test",
)


def tree_hash(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------ frozen reference

@PRIVATE_V1
def test_frozen_reference_is_unchanged():
    assert hashlib.sha256(V1.read_bytes()).hexdigest() == V1_SHA256
    recorded = (REFERENCE_DIR / "SHA256SUMS").read_text(encoding="utf-8").split()
    assert recorded == [V1_SHA256, V1.name]


@PRIVATE_V1
def test_frozen_reference_content_and_provenance():
    reference, digest = load_reference(V1)
    assert digest == V1_SHA256 and reference["frozen"] is True
    assert reference["provenance"]["human_verified"] is False
    assert len(reference["headings"]) == 44 and len(reference["assignments"]) == 81
    assert sum(h["status"] == "uncertain" for h in reference["headings"]) == 2
    assert sum(a["status"] == "uncertain" for a in reference["assignments"]) == 8
    assert sum(h["expected_verdict"] == "heading" for h in reference["headings"]) == 18
    statuses = {tuple(c["pages"]): c["status"] for c in reference["continuations"]}
    assert statuses[(13, 14)] == "unverified" and statuses[(16, 17, 18)] == "unverified"
    hist = reference["historical_aggregate_metrics"]
    assert hist["heading_decisions"] == {"total": 44, "correct": 39, "incorrect": 3, "uncertain": 2}
    assert hist["assignments"] == {"total": 81, "correct": 59, "incorrect": 14, "uncertain": 8}


@PRIVATE_V1
def test_modified_frozen_reference_is_refused(tmp_path):
    folder = tmp_path / "refs"
    folder.mkdir()
    shutil.copy(V1, folder / V1.name)
    shutil.copy(REFERENCE_DIR / "SHA256SUMS", folder / "SHA256SUMS")
    data = json.loads((folder / V1.name).read_text(encoding="utf-8"))
    data["headings"][0]["expected_verdict"] = "not_heading"
    (folder / V1.name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(EvaluationError, match="was modified"):
        load_reference(folder / V1.name)


# ------------------------------------------------------------------ synthetic outputs

def make_output(root, name, *, promoted=("h1",), sections=None, assignments=None, run_id="run_x"):
    folder = root / name
    folder.mkdir(parents=True)
    sections = sections or [{"section_id": "s001", "title": "Heading one", "level": 1, "parent": None,
                             "heading_element_ids": ["h1"]}]
    elements = ["h1", "h2", "p1", "p2", "p3"]
    assignments = assignments or [
        {"unit_id": "h1", "element_id": "h1", "section_id": "s001", "role": "heading", "page": 1},
        {"unit_id": "h2", "element_id": "h2", "section_id": "s001", "role": "paragraph", "page": 1},
        {"unit_id": "p1", "element_id": "p1", "section_id": "s001", "role": "paragraph", "page": 1},
        {"unit_id": "p2", "element_id": "p2", "section_id": None, "role": "paragraph", "page": 1},
        {"unit_id": "p3", "element_id": "p3", "section_id": "s001", "role": "paragraph", "page": 2},
    ]
    candidates = [{"unit_ids": [e], "status": "promoted" if e in promoted else "rejected"} for e in ("h1", "h2")]
    (folder / "section_tree.json").write_text(json.dumps({"run_id": run_id, "sections": sections,
                                                          "heading_candidates": candidates}), encoding="utf-8")
    (folder / "assignments.json").write_text(json.dumps({
        "source_inventory": [{"element_id": e} for e in elements], "assignments": assignments}), encoding="utf-8")
    return folder


def make_reference(root, *, headings=None, assignments=None, continuations=None, run_id="run_x"):
    """Written to <root>/refs/ref.json: the reference folder must not contain the evaluation output root."""
    folder = root / "refs"
    folder.mkdir(exist_ok=True)
    path = folder / "ref.json"
    path.write_text(json.dumps({
        "kind": "docprep_structure_reference", "reference_id": "synthetic", "frozen": False,
        "source": {"run_id": run_id}, "provenance": {"method": "synthetic test", "human_verified": False},
        "headings": headings if headings is not None else [
            {"source_element_ids": ["h1"], "text": "Heading one", "page": 1, "expected_verdict": "heading",
             "expected_level": 1, "status": "confirmed"},
            {"source_element_ids": ["h2"], "text": "Heading two", "page": 1, "expected_verdict": "heading",
             "expected_level": 1, "status": "uncertain"}],
        "assignments": assignments if assignments is not None else [
            {"element_id": "p1", "expected_parent_heading_element_id": "h1", "status": "confirmed"},
            {"element_id": "p2", "expected_parent_heading_element_id": "h1", "status": "uncertain"}],
        "continuations": continuations if continuations is not None else [
            {"pages": [1, 2], "expected_parent_heading_element_id": "h1", "continuing_element_ids": ["p3"],
             "status": "confirmed"}],
        "historical_aggregate_metrics": {"note": "synthetic"}}), encoding="utf-8")
    return path


def test_uncertain_cases_are_excluded_from_metrics(tmp_path):
    view = load_structure(make_output(tmp_path, "out"))
    reference, _ = load_reference(make_reference(tmp_path))
    result = evaluate_structure(view, reference)
    m = result["metrics"]
    # h2 (uncertain, a missed heading) and p2 (uncertain, unassigned) do not count either way.
    assert m["headings"]["true_headings_detected"] == 1 and m["headings"]["missed_headings"] == 0
    assert m["headings"]["recall"] == 1.0 and m["headings"]["recall_denominator"] == 1
    assert m["assignments"] == {"correct": 1, "wrong": 0, "denominator": 1, "uncertain_listed_separately": 1}
    assert m["continuations"]["correct"] == 1
    uncertain = [c for c in result["cases"] if c["status"] == "uncertain"]
    assert {c["key"] for c in uncertain} == {"heading:h2", "assignment:p2"}
    assert m["unassigned_elements_in_output"] == 1


def test_incomplete_reference_gives_no_misleading_metrics(tmp_path):
    view = load_structure(make_output(tmp_path, "out"))
    reference, _ = load_reference(make_reference(tmp_path, headings=[
        {"source_element_ids": ["missing"], "text": "?", "page": 3, "expected_verdict": "heading", "status": "confirmed"}],
        assignments=[], continuations=[{"pages": [16, 18], "expected_parent_heading_element_id": None,
                                        "continuing_element_ids": [], "status": "unverified"}]))
    result = evaluate_structure(view, reference)
    m = result["metrics"]
    assert m["headings"]["precision"] is None and m["headings"]["recall"] is None
    assert m["not_evaluable"] == 2 and m["evaluated_confirmed_cases"] == 0
    from docprep.evaluate import render_markdown
    text = render_markdown({"structure_run_id": "run_x", "structure_out_dir": "x", "reference_file": "r",
                            "reference_sha256": "0" * 64, "reference_provenance": {}, "current": result})
    assert "n/a (no confirmed cases)" in text


def test_evaluation_is_write_once_and_leaves_inputs_unchanged(tmp_path):
    out = make_output(tmp_path / "results" / "document_structure", "run_x__1")
    base = make_output(tmp_path / "results" / "document_structure", "run_x__0", promoted=("h1", "h2"))
    ref = make_reference(tmp_path)
    before = {p: tree_hash(p) for p in (out, base)}
    ref_hash = hashlib.sha256(ref.read_bytes()).hexdigest()
    target = run(out, ref, baseline_dir=base, stamp="20260101T000000Z")
    assert target.parent.resolve() == (tmp_path / "results" / "document_structure_evaluation").resolve()
    assert sorted(p.name for p in target.iterdir()) == ["evaluation.json", "evaluation.md", "manifest.json"]
    assert all(not os.access(p, os.W_OK) for p in target.iterdir())
    assert {p: tree_hash(p) for p in (out, base)} == before
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == ref_hash
    with pytest.raises(OutputExistsError):
        run(out, ref, baseline_dir=base, stamp="20260101T000000Z")


def test_output_root_inside_an_input_is_refused(tmp_path):
    out = make_output(tmp_path, "out")
    with pytest.raises(EvaluationError, match="inside an input folder"):
        run(out, make_reference(tmp_path), out_root=out / "eval")


def test_regressions_and_fixes_are_reported(tmp_path):
    base = make_output(tmp_path, "base")
    current = make_output(tmp_path, "cur", sections=[
        {"section_id": "s001", "title": "Heading one", "level": 2, "parent": None, "heading_element_ids": ["h1"]}],
        assignments=[
            {"unit_id": "h1", "element_id": "h1", "section_id": "s001", "role": "heading", "page": 1},
            {"unit_id": "h2", "element_id": "h2", "section_id": "s001", "role": "paragraph", "page": 1},
            {"unit_id": "p1", "element_id": "p1", "section_id": None, "role": "paragraph", "page": 1},
            {"unit_id": "p2", "element_id": "p2", "section_id": None, "role": "paragraph", "page": 1},
            {"unit_id": "p3", "element_id": "p3", "section_id": "s001", "role": "paragraph", "page": 2}])
    target = run(current, make_reference(tmp_path), baseline_dir=base, out_root=tmp_path / "evals", stamp="1")
    result = json.loads((target / "evaluation.json").read_text(encoding="utf-8"))
    assert "assignment:p1" in result["comparison"]["regressed"]
    assert "heading:h1:level" in result["comparison"]["regressed"]


def test_run_id_mismatch_is_refused(tmp_path, capsys):
    out = make_output(tmp_path, "out", run_id="run_other")
    assert main([str(out), "--reference", str(make_reference(tmp_path)), "--out", str(tmp_path / "e")]) == 2
    assert "reference is for run_x" in capsys.readouterr().err


def test_structure_runs_never_touch_an_earlier_output(tmp_path):
    run_dir = write_run(tmp_path, "run_hist", [el("h", 1, 0, "Overview", type="heading", bbox=[72, 80, 200, 96]),
                                               el("b", 1, 1, "A short synthetic paragraph about the overview.",
                                                  bbox=[72, 100, 540, 114])])
    first, _ = structure_run(run_dir, out_root=tmp_path / "out", stamp="20260101T000000Z")
    snapshot = tree_hash(first)
    second, _ = structure_run(run_dir, out_root=tmp_path / "out", stamp="20260101T000001Z")
    assert second != first and tree_hash(first) == snapshot
