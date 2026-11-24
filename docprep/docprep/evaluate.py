"""Independent evaluation of a structure output against frozen and human references.

Evaluate:
  python -m docprep.evaluate <structure_out_dir> [--reference <v1.json>] [--human-reference <review.csv|json>]
                             [--baseline <previous_out_dir>] [--out <root>]
Make an editable human review template (blank human columns):
  python -m docprep.evaluate --make-template <structure_out_dir> --pages 1,2,13 [--out <root>]

Rules:
- only confirmed, individually labelled cases enter the metrics; uncertain, unverified and unreviewed cases are
  listed separately and never counted as correct;
- every metric states its evaluated subset and denominator; a metric without confirmed cases is "n/a";
- tool-reviewed (v1) and human-reviewed labels are reported separately; in the combined heading view a human
  label replaces a v1 label only for the exact same element(s) or span, and only when the human row is labelled;
- references are read-only; a frozen reference whose SHA-256 differs from its SHA256SUMS entry is refused;
- nothing is inferred from the implementation under evaluation; human columns are never pre-filled;
- outputs are write-once, outside the structure output, baseline and reference folders.

Exit codes: 0 done, 2 input error, 3 output folder exists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .output import OutputExistsError, write_outputs

FURNITURE_ROLES = {"header", "footer", "page_number"}
HUMAN_VERDICTS = ("heading", "not_heading", "excluded", "uncertain")
TEMPLATE_COLUMNS = ["row_id", "element_id", "span_id", "page", "source_text", "tool_verdict", "tool_role",
                    "tool_level", "tool_reason", "human_verdict", "human_level", "human_note", "candidate_id",
                    "unit_ids"]
HUMAN_COLUMNS = ("human_verdict", "human_level", "human_note")
MAX_LEVEL = 9


class EvaluationError(ValueError):
    """Invalid reference, review file or structure output."""


# ------------------------------------------------------------------ loading

def load_reference(path: Path | str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file():
        raise EvaluationError(f"reference not found: {path}")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    try:
        reference = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"reference is not valid JSON: {exc}") from exc
    if not isinstance(reference, dict) or reference.get("kind") != "docprep_structure_reference":
        raise EvaluationError("not a docprep structure reference")
    sums = path.parent / "SHA256SUMS"
    if reference.get("frozen") and sums.is_file():
        recorded = {}
        for line in sums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                recorded[parts[1].lstrip("*")] = parts[0]
        if path.name in recorded and recorded[path.name] != digest:
            raise EvaluationError(f"frozen reference {path.name} was modified (SHA-256 {digest} differs from "
                                  "SHA256SUMS); corrections must be a new versioned file")
    for key in ("headings", "assignments", "continuations"):
        if not isinstance(reference.get(key), list):
            raise EvaluationError(f"reference has no '{key}' list")
    return reference, digest


@dataclass
class StructureView:
    out_dir: Path
    run_id: str | None
    elements: set[str]
    unit_element: dict[str, str]
    sections: dict[str, dict[str, Any]]
    section_of_heading_element: dict[str, str]
    candidate_status: dict[str, str]                      # element id -> best candidate status
    assignments_by_element: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    unassigned: list[dict[str, Any]] = field(default_factory=list)
    candidate_status_by_unit: dict[str, str] = field(default_factory=dict)
    section_of_heading_unit: dict[str, str] = field(default_factory=dict)
    unit_text: dict[str, str] = field(default_factory=dict)
    unit_page: dict[str, int] = field(default_factory=dict)
    unit_role: dict[str, str] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    section_tree_sha256: str | None = None


def load_structure(out_dir: Path | str) -> StructureView:
    out_dir = Path(out_dir)
    try:
        tree_bytes = (out_dir / "section_tree.json").read_bytes()
        tree = json.loads(tree_bytes.decode("utf-8"))
        assigned = json.loads((out_dir / "assignments.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"incomplete structure output {out_dir}: {exc.filename} missing") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvaluationError(f"invalid JSON in {out_dir}: {exc}") from exc
    unit_element = {a["unit_id"]: a["element_id"] for a in assigned["assignments"]}
    elements = {e["element_id"] for e in assigned.get("source_inventory", [])} or set(unit_element.values())
    sections = {s["section_id"]: s for s in tree["sections"]}
    by_heading: dict[str, str] = {}
    by_heading_unit: dict[str, str] = {}
    for s in tree["sections"]:
        for element_id in s.get("heading_element_ids") or []:
            by_heading.setdefault(element_id, s["section_id"])
        for unit_id in s.get("heading_unit_ids") or []:
            by_heading_unit.setdefault(unit_id, s["section_id"])
    rank = {"promoted": 3, "rejected": 2, "excluded": 1}
    status: dict[str, str] = {}
    status_unit: dict[str, str] = {}
    for c in tree.get("heading_candidates", []):
        for unit_id in c["unit_ids"]:
            element_id = unit_element.get(unit_id, unit_id)
            if rank.get(c["status"], 0) > rank.get(status.get(element_id, ""), 0):
                status[element_id] = c["status"]
            status_unit[unit_id] = c["status"]
    by_element: dict[str, list[dict[str, Any]]] = {}
    for a in assigned["assignments"]:
        by_element.setdefault(a["element_id"], []).append(a)
    unassigned = [a for a in assigned["assignments"] if a["section_id"] is None and a["role"] not in FURNITURE_ROLES]
    return StructureView(
        out_dir, tree.get("run_id"), elements, unit_element, sections, by_heading, status, by_element, unassigned,
        candidate_status_by_unit=status_unit, section_of_heading_unit=by_heading_unit,
        unit_text={a["unit_id"]: a.get("text") for a in assigned["assignments"]},
        unit_page={a["unit_id"]: a.get("page") for a in assigned["assignments"]},
        unit_role={a["unit_id"]: a.get("role") for a in assigned["assignments"]},
        candidates=tree.get("heading_candidates", []), section_tree_sha256=hashlib.sha256(tree_bytes).hexdigest())


# ------------------------------------------------------------------ evaluation

def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def heading_key(h: dict[str, Any]) -> str:
    return "heading:" + (h.get("span_id") or "+".join(h["source_element_ids"]))


def _assignment_outcome(view: StructureView, element_ids: list[str], expected_parent: str) -> tuple[bool, str]:
    records = [a for e in element_ids for a in view.assignments_by_element.get(e, [])]
    if not records:
        return False, "no assignment record for the element"
    wrong = []
    for a in records:
        if a["section_id"] is None:
            wrong.append(f"{a['unit_id']} unassigned")
            continue
        heading_elements = view.sections[a["section_id"]].get("heading_element_ids") or []
        if expected_parent not in heading_elements:
            wrong.append(f"{a['unit_id']} in {a['section_id']} «{view.sections[a['section_id']]['title'][:60]}»")
    return (not wrong), ("; ".join(wrong) if wrong else "in the expected section")


def evaluate_structure(view: StructureView, reference: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    tp = fp = fn = tn = 0
    level_ok = level_bad = 0
    for h in reference["headings"]:
        ids = h["source_element_ids"]
        span_id = h.get("span_id") or None
        case = {"key": heading_key(h), "kind": "heading", "page": h.get("page"), "text": h.get("text"),
                "status": h["status"], "expected": h["expected_verdict"], "expected_level": h.get("expected_level"),
                "source": h.get("source", reference.get("reference_id"))}
        if span_id and span_id not in view.unit_element or any(i not in view.elements for i in ids):
            case.update(outcome="not_evaluable", correct=None, detail="source element or span missing from the output")
            cases.append(case)
            continue
        if span_id:
            predicted = view.candidate_status_by_unit.get(span_id, "not_a_candidate")
            section_id = view.section_of_heading_unit.get(span_id)
        else:
            predicted = view.candidate_status.get(ids[0], "not_a_candidate")
            section_id = view.section_of_heading_element.get(ids[0])
        is_heading = predicted == "promoted"
        expected_heading = h["expected_verdict"] == "heading"
        case.update(predicted=predicted, correct=is_heading == expected_heading)
        if expected_heading and is_heading:
            case["outcome"] = "true_heading_detected"
        elif expected_heading:
            case["outcome"] = "missed_heading"
        elif is_heading:
            case["outcome"] = "false_promotion"
        else:
            case["outcome"] = "correct_non_heading"
        if is_heading and section_id is not None:
            section = view.sections[section_id]
            case["predicted_level"] = section.get("level")
            if expected_heading and h.get("expected_level") is not None:
                case["level_correct"] = section.get("level") == h["expected_level"]
            parent = section.get("parent")
            if expected_heading and h.get("expected_parent_heading_element_id"):
                parent_heads = view.sections[parent].get("heading_element_ids", []) if parent else []
                case["parent_correct"] = h["expected_parent_heading_element_id"] in parent_heads
        if h["status"] == "confirmed":
            tp += case["outcome"] == "true_heading_detected"
            fp += case["outcome"] == "false_promotion"
            fn += case["outcome"] == "missed_heading"
            tn += case["outcome"] == "correct_non_heading"
            if "level_correct" in case:
                level_ok += case["level_correct"]
                level_bad += not case["level_correct"]
        cases.append(case)
    a_ok = a_bad = 0
    for a in reference["assignments"]:
        case = {"key": f"assignment:{a['element_id']}", "kind": "assignment", "page": a.get("page"),
                "status": a["status"], "expected_parent": a.get("expected_parent_heading_element_id")}
        if a["element_id"] not in view.elements:
            case.update(outcome="not_evaluable", correct=None, detail="element missing from the output")
        elif not a.get("expected_parent_heading_element_id"):
            case.update(outcome="no_expected_parent", correct=None, detail="reference gives no expected parent")
        else:
            ok, detail = _assignment_outcome(view, [a["element_id"]], a["expected_parent_heading_element_id"])
            case.update(outcome="correct" if ok else "wrong_section", correct=ok, detail=detail)
            if a["status"] == "confirmed":
                a_ok += ok
                a_bad += not ok
        cases.append(case)
    c_ok = c_bad = 0
    for c in reference["continuations"]:
        case = {"key": "continuation:" + "-".join(str(p) for p in c["pages"]), "kind": "continuation",
                "pages": c["pages"], "status": c["status"]}
        if not c.get("continuing_element_ids") or not c.get("expected_parent_heading_element_id"):
            case.update(outcome="not_evaluable", correct=None, detail="no individually identified elements")
        elif any(e not in view.elements for e in c["continuing_element_ids"]):
            case.update(outcome="not_evaluable", correct=None, detail="element missing from the output")
        else:
            ok, detail = _assignment_outcome(view, c["continuing_element_ids"], c["expected_parent_heading_element_id"])
            case.update(outcome="correct" if ok else "broken", correct=ok, detail=detail)
            if c["status"] == "confirmed":
                c_ok += ok
                c_bad += not ok
        cases.append(case)

    confirmed = [c for c in cases if c["status"] == "confirmed" and c.get("correct") is not None]
    by_status: dict[str, int] = {}
    for c in cases:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
    metrics = {
        "scope": "confirmed, individually labelled reference cases only; uncertain and unverified cases excluded",
        "evaluated_confirmed_cases": len(confirmed),
        "cases_by_reference_status": by_status,
        "not_evaluable": sum(1 for c in cases if c.get("outcome") == "not_evaluable"),
        "headings": {"true_headings_detected": tp, "false_promotions": fp, "missed_headings": fn,
                     "correct_non_headings": tn,
                     "precision": _ratio(tp, tp + fp), "precision_denominator": tp + fp,
                     "recall": _ratio(tp, tp + fn), "recall_denominator": tp + fn},
        "heading_levels": {"correct": level_ok, "incorrect": level_bad, "denominator": level_ok + level_bad,
                           "note": "detected confirmed true headings with a documented expected level"},
        "assignments": {"correct": a_ok, "wrong": a_bad, "denominator": a_ok + a_bad,
                        "uncertain_listed_separately": sum(1 for c in cases if c["kind"] == "assignment"
                                                           and c["status"] != "confirmed")},
        "continuations": {"correct": c_ok, "broken": c_bad, "denominator": c_ok + c_bad,
                          "unverified_or_uncertain": sum(1 for c in cases if c["kind"] == "continuation"
                                                         and c["status"] != "confirmed")},
        "unassigned_elements_in_output": len(view.unassigned),
    }
    return {"cases": cases, "metrics": metrics,
            "unassigned": [{"unit_id": a["unit_id"], "page": a["page"], "role": a["role"],
                            "reasons": a.get("reasons")} for a in view.unassigned]}


def compare(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    base = {c["key"]: c for c in baseline["cases"]}
    changes = {"fixed": [], "regressed": [], "unchanged_correct": [], "unchanged_wrong": [], "not_comparable": []}
    for case in current["cases"]:
        if case["status"] != "confirmed":
            continue
        old = base.get(case["key"])
        if old is None or case.get("correct") is None or old.get("correct") is None:
            changes["not_comparable"].append(case["key"])
        elif old["correct"] and not case["correct"]:
            changes["regressed"].append(case["key"])
        elif not old["correct"] and case["correct"]:
            changes["fixed"].append(case["key"])
        elif case["correct"]:
            changes["unchanged_correct"].append(case["key"])
        else:
            changes["unchanged_wrong"].append(case["key"])
        if old is not None and case["kind"] == "heading" and "level_correct" in case and "level_correct" in old \
                and old["level_correct"] != case["level_correct"]:
            changes["fixed" if case["level_correct"] else "regressed"].append(case["key"] + ":level")
    return {"counts": {k: len(v) for k, v in changes.items()}, **changes}


# ------------------------------------------------------------------ human review template

def _section_level(view: StructureView, unit_id: str) -> int | None:
    section_id = view.section_of_heading_unit.get(unit_id)
    return view.sections[section_id].get("level") if section_id else None


def template_rows(view: StructureView, pages: set[int]) -> list[dict[str, Any]]:
    """One row per heading candidate (promoted, rejected, and excluded except plain list items and page
    furniture), plus every list_intro and aside_label unit not already covered, on the selected pages.
    The human columns are always blank: nothing is copied from DocPrep or from any reference."""
    rows: list[dict[str, Any]] = []
    covered: set[str] = set()
    skip_prefixes = ("list item", "page furniture", "page number")
    for c in view.candidates:
        units = c["unit_ids"]
        page = view.unit_page.get(units[0])
        if page not in pages:
            continue
        reasons = c.get("exclusions") or []
        roles = {view.unit_role.get(u) for u in units}
        if c["status"] == "excluded" and reasons and all(r.startswith(skip_prefixes) or r.startswith("this overrides")
                                                         for r in reasons) and not roles & {"list_intro", "aside_label"}:
            continue
        element_ids = list(dict.fromkeys(view.unit_element[u] for u in units))
        span_id = units[0] if len(units) == 1 and units[0] != view.unit_element[units[0]] else ""
        verdict = {"promoted": "heading", "rejected": "not_heading", "excluded": "excluded"}[c["status"]]
        level = _section_level(view, units[0]) if c["status"] == "promoted" else None
        rows.append({"element_id": "+".join(element_ids), "span_id": span_id, "page": page,
                     "source_text": "\n".join(view.unit_text[u] for u in units), "tool_verdict": verdict,
                     "tool_role": view.unit_role.get(units[0]) or "", "tool_level": "" if level is None else level,
                     "tool_reason": "; ".join(reasons) if reasons else f"score {c.get('score')}",
                     "human_verdict": "", "human_level": "", "human_note": "",
                     "candidate_id": c.get("candidate_id", ""), "unit_ids": "+".join(units)})
        covered.update(units)
    for unit_id, role in view.unit_role.items():
        if role in ("list_intro", "aside_label") and unit_id not in covered and view.unit_page.get(unit_id) in pages:
            element_id = view.unit_element[unit_id]
            rows.append({"element_id": element_id, "span_id": unit_id if unit_id != element_id else "",
                         "page": view.unit_page[unit_id], "source_text": view.unit_text[unit_id],
                         "tool_verdict": "not_heading", "tool_role": role, "tool_level": "",
                         "tool_reason": f"derived role {role}", "human_verdict": "", "human_level": "",
                         "human_note": "", "candidate_id": "", "unit_ids": unit_id})
    rows.sort(key=lambda r: (r["page"], r["unit_ids"]))
    for number, row in enumerate(rows, start=1):
        row["row_id"] = f"r{number:04d}"
    return rows


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TEMPLATE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in TEMPLATE_COLUMNS})
    return "﻿" + buffer.getvalue()  # BOM so spreadsheet programs open Arabic text as UTF-8


INSTRUCTIONS = """# Human structure review

Fill only the three human columns; leave every other column exactly as it is (source text and identifiers are
checked against the structure output when the review is loaded).

- human_verdict: heading | not_heading | excluded | uncertain   (blank = not reviewed)
- human_level:   relative heading level for verdict "heading" (0 document title, 1 major section, 2 subsection,
                 ...); leave blank otherwise
- human_note:    free text

Judge the original document (the PDF), not the tool's columns. Save the completed file under a new name, for
example human_reference_v2.csv, and evaluate with:

    python -m docprep.evaluate <structure_out_dir> --reference docprep/eval/references/traffic_report_review_v1.json \\
        --human-reference <completed file>

Blank rows stay unreviewed and never replace v1 judgments. This template contains no page images.
"""


def make_template(structure_dir: Path, pages: set[int], *, out_root: Path | None = None,
                  stamp: str | None = None) -> Path:
    view = load_structure(structure_dir)
    if not pages:
        raise EvaluationError("--pages must list at least one page")
    rows = template_rows(view, pages)
    meta = {"schema_version": 1, "kind": "docprep_human_review_template", "template_version": 1,
            "source_run_id": view.run_id, "structure_out_dir": str(structure_dir),
            "section_tree_sha256": view.section_tree_sha256, "pages": sorted(pages),
            "allowed_human_verdicts": list(HUMAN_VERDICTS), "created_at": datetime.now(timezone.utc).isoformat(),
            "columns": TEMPLATE_COLUMNS, "rows": rows}
    root = Path(out_root) if out_root else Path(structure_dir).resolve().parent.parent / "document_structure_review"
    if _inside(root, Path(structure_dir)):
        raise EvaluationError("the template output root must not be inside the structure output")
    contents = {"review_template.csv": _csv_text(rows),
                "review_template.json": json.dumps(meta, ensure_ascii=False, indent=2),
                "README_review.md": INSTRUCTIONS}

    def manifest(hashes: dict[str, str]) -> dict[str, Any]:
        return {"schema_version": 1, "kind": "docprep_review_template_manifest", "docprep_version": __version__,
                "structure_out_dir": str(structure_dir), "section_tree_sha256": view.section_tree_sha256,
                "pages": sorted(pages), "rows": len(rows), "output_hashes": hashes}

    # The template files stay writable (they are meant to be edited); the folder itself is never reused.
    return write_outputs(root, f"review__{Path(structure_dir).name}", contents, manifest, stamp=stamp,
                         read_only=False)


# ------------------------------------------------------------------ human reference

def _read_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"human reference not found: {path}")
    text = path.read_bytes().decode("utf-8-sig")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"human reference is not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
            raise EvaluationError("human reference JSON must contain a 'rows' list")
        return data["rows"], {k: v for k, v in data.items() if k != "rows"}
    reader = csv.DictReader(io.StringIO(text))
    missing = [c for c in ("element_id", "span_id", "page", "source_text", *HUMAN_COLUMNS) if c not in (reader.fieldnames or [])]
    if missing:
        raise EvaluationError(f"human reference CSV lacks columns: {missing}")
    return list(reader), {}


def load_human_reference(path: Path | str, view: StructureView) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a completed review and turn its labelled rows into a reference (headings only).
    Returns (reference, coverage)."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    rows, meta = _read_rows(path)
    if meta.get("source_run_id") and view.run_id and meta["source_run_id"] != view.run_id:
        raise EvaluationError(f"the review is for {meta['source_run_id']}, the structure output is for {view.run_id}")
    errors: list[str] = []
    headings: list[dict[str, Any]] = []
    reviewed = unreviewed = 0
    pages: set[int] = set()
    for number, row in enumerate(rows, start=2 if path.suffix.lower() != ".json" else 1):
        label = f"row {row.get('row_id') or number}"
        verdict = str(row.get("human_verdict") or "").strip()
        level_text = str(row.get("human_level") if row.get("human_level") is not None else "").strip()
        element_ids = [e for e in str(row.get("element_id") or "").split("+") if e]
        span_id = str(row.get("span_id") or "").strip()
        source_text = row.get("source_text")
        if not element_ids:
            errors.append(f"{label}: no element_id")
            continue
        unknown = [e for e in element_ids if e not in view.elements] + ([span_id] if span_id and span_id not in view.unit_element else [])
        if unknown:
            errors.append(f"{label}: unknown source identifier(s) {unknown}")
            continue
        unit_ids = [u for u in str(row.get("unit_ids") or "").split("+") if u] or ([span_id] if span_id else element_ids)
        expected_text = "\n".join(view.unit_text.get(u) or "" for u in unit_ids)
        if source_text is None or str(source_text) != expected_text:
            errors.append(f"{label}: source_text differs from the structure output (it must stay verbatim)")
            continue
        if not verdict:
            if level_text:
                errors.append(f"{label}: human_level given without a human_verdict")
            unreviewed += 1
            continue
        if verdict not in HUMAN_VERDICTS:
            errors.append(f"{label}: invalid human_verdict '{verdict}' (allowed: {', '.join(HUMAN_VERDICTS)})")
            continue
        level = None
        if level_text:
            if verdict != "heading":
                errors.append(f"{label}: human_level is only allowed with human_verdict 'heading'")
                continue
            if not level_text.isdigit() or not 0 <= int(level_text) <= MAX_LEVEL:
                errors.append(f"{label}: human_level must be an integer 0..{MAX_LEVEL}, got '{level_text}'")
                continue
            level = int(level_text)
        try:
            page = int(row.get("page"))
        except (TypeError, ValueError):
            errors.append(f"{label}: invalid page")
            continue
        reviewed += 1
        pages.add(page)
        headings.append({"source_element_ids": element_ids, "span_id": span_id or None, "page": page,
                         "text": source_text, "expected_verdict": verdict if verdict != "uncertain" else "uncertain",
                         "expected_level": level, "status": "uncertain" if verdict == "uncertain" else "confirmed",
                         "explanation": str(row.get("human_note") or ""), "source": "human"})
    if errors:
        raise EvaluationError("invalid human reference:\n  " + "\n  ".join(errors[:50]))
    coverage = {"rows": len(rows), "reviewed": reviewed, "unreviewed": unreviewed,
                "confirmed": sum(1 for h in headings if h["status"] == "confirmed"),
                "uncertain": sum(1 for h in headings if h["status"] == "uncertain"),
                "pages_with_reviewed_rows": sorted(pages), "template_pages": meta.get("pages"),
                "note": "partial review: metrics cover the reviewed rows only, never the whole document"}
    reference = {"kind": "docprep_structure_reference", "reference_id": f"human:{path.name}", "frozen": False,
                 "provenance": {"method": "human review of a DocPrep review template", "human_verified": True,
                                "file": str(path), "sha256": digest},
                 "headings": headings, "assignments": [], "continuations": []}
    return reference, coverage


def combine_headings(v1: dict[str, Any] | None, human: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Headings only: labelled human rows replace v1 labels with the exact same key; everything else from v1."""
    human_by_key = {heading_key(h): h for h in human["headings"]}
    merged = [h for h in human["headings"]]
    from_v1 = 0
    for h in (v1 or {}).get("headings", []):
        if heading_key(h) not in human_by_key:
            merged.append(dict(h, source="v1"))
            from_v1 += 1
    composition = {"from_human": len(human["headings"]), "from_v1": from_v1,
                   "replaced_v1_labels": sum(1 for h in (v1 or {}).get("headings", []) if heading_key(h) in human_by_key)}
    return ({"kind": "docprep_structure_reference", "reference_id": "combined_headings", "headings": merged,
             "assignments": [], "continuations": []}, composition)


def disagreements(v1: dict[str, Any] | None, human: dict[str, Any]) -> list[dict[str, Any]]:
    def klass(verdict: str) -> str:
        return {"heading": "heading", "not_heading": "non_heading", "excluded": "non_heading"}.get(verdict, verdict)

    out = []
    v1_by_key = {heading_key(h): h for h in (v1 or {}).get("headings", [])}
    for h in human["headings"]:
        old = v1_by_key.get(heading_key(h))
        if old is None:
            continue
        diff = []
        if klass(old["expected_verdict"]) != klass(h["expected_verdict"]) or old["status"] != h["status"]:
            diff.append(f"verdict v1 {old['expected_verdict']} ({old['status']}) vs human {h['expected_verdict']} "
                        f"({h['status']})")
        if h["expected_verdict"] == "heading" and old.get("expected_level") is not None and \
                h.get("expected_level") is not None and old["expected_level"] != h["expected_level"]:
            diff.append(f"level v1 {old['expected_level']} vs human {h['expected_level']}")
        if diff:
            out.append({"key": heading_key(h), "page": h.get("page"), "text": h.get("text"), "differences": diff})
    return out


# ------------------------------------------------------------------ report

def _fmt(value: float | None, den: int) -> str:
    return "n/a (no confirmed cases)" if value is None else f"{value:.3f} ({den} cases)"


def _metric_lines(title: str, evaluation: dict[str, Any]) -> list[str]:
    m = evaluation["metrics"]
    h, lv, a, c = m["headings"], m["heading_levels"], m["assignments"], m["continuations"]
    return [f"## {title}", "",
            (f"- Scope: {m['scope']}. Evaluated confirmed cases: {m['evaluated_confirmed_cases']}; not evaluable: "
             f"{m['not_evaluable']}; cases by status: {m['cases_by_reference_status']}."), "",
            "| Metric | Value |", "| --- | --- |",
            f"| Heading precision | {_fmt(h['precision'], h['precision_denominator'])} |",
            f"| Heading recall | {_fmt(h['recall'], h['recall_denominator'])} |",
            f"| True headings detected | {h['true_headings_detected']} |",
            f"| False heading promotions | {h['false_promotions']} |",
            f"| Missed headings | {h['missed_headings']} |",
            f"| Heading levels correct / incorrect | {lv['correct']} / {lv['incorrect']} (of {lv['denominator']}) |",
            f"| Assignments correct / wrong | {a['correct']} / {a['wrong']} (of {a['denominator']}) |",
            f"| Cross-page continuations correct / broken | {c['correct']} / {c['broken']} (of {c['denominator']}) |",
            f"| Unassigned elements in the output | {m['unassigned_elements_in_output']} |", ""]


def _case_lines(evaluation: dict[str, Any]) -> list[str]:
    cases = evaluation["cases"]
    wrong = [x for x in cases if x["status"] == "confirmed" and x.get("correct") is False]
    lines = ["### Confirmed cases judged wrong", ""]
    lines += [f"- `{x['key']}` ({x.get('outcome')}): {x.get('detail', '')} {(x.get('text') or '')[:80]}"
              for x in wrong] or ["_None._"]
    level_wrong = [x for x in cases if x.get("level_correct") is False and x["status"] == "confirmed"]
    lines += ["", "### Confirmed headings with a wrong level", ""]
    lines += [f"- `{x['key']}`: expected {x['expected_level']}, got {x.get('predicted_level')}" for x in level_wrong] \
        or ["_None._"]
    others = [x for x in cases if x["status"] != "confirmed"]
    lines += ["", "### Uncertain and unverified cases (not counted as correct or wrong)", ""]
    lines += [f"- `{x['key']}` [{x['status']}]: {x.get('outcome')} — {x.get('detail', '')}" for x in others] \
        or ["_None._"]
    return lines + [""]


def render_markdown(result: dict[str, Any]) -> str:
    lines = [f"# Structure evaluation: {result.get('structure_run_id')}", "",
             f"- Structure output: `{result.get('structure_out_dir')}`"]
    if result.get("current") is not None:
        lines.append(f"- v1 reference (tool-reviewed): `{result.get('reference_file')}` (SHA-256 "
                     f"`{str(result.get('reference_sha256'))[:16]}…`, provenance: {result.get('reference_provenance')})")
    if result.get("human") is not None:
        lines.append(f"- Human reference: `{result['human_reference_file']}` (provenance: "
                     f"{result['human_reference_provenance']}); coverage: {result['human_coverage']}")
    lines.append("")
    if result.get("current") is not None:
        lines += _metric_lines("v1 reference (tool-reviewed, not human-verified)", result["current"])
        lines += _case_lines(result["current"])
    if result.get("human") is not None:
        lines += _metric_lines("Human reference (human-reviewed rows only; partial review)", result["human"])
        lines += _case_lines(result["human"])
    if result.get("combined_headings") is not None:
        comp = result["combined_headings_composition"]
        lines += _metric_lines(f"Combined heading view (human precedence; {comp['from_human']} human + "
                               f"{comp['from_v1']} v1 labels; {comp['replaced_v1_labels']} v1 labels replaced)",
                               result["combined_headings"])
    if result.get("disagreements") is not None:
        lines += ["## Disagreements between v1 and human labels", ""]
        lines += [f"- `{d['key']}` (p{d['page']}): {'; '.join(d['differences'])}" for d in result["disagreements"]] \
            or ["_None._"]
        lines.append("")
    for name, title in (("comparison", "v1"), ("human_comparison", "human")):
        cmp = result.get(name)
        if not cmp:
            continue
        lines += [f"## Before / after on the {title} reference (baseline `{result['baseline_out_dir']}`)", "",
                  f"- Counts: {cmp['counts']}", "", "### Regressions (correct before, wrong now)", ""]
        lines += [f"- `{k}`" for k in cmp["regressed"]] or ["_None._"]
        lines += ["", "### Fixed (wrong before, correct now)", ""]
        lines += [f"- `{k}`" for k in cmp["fixed"]] or ["_None._"]
        lines.append("")
    if result.get("current") is not None and result.get("baseline") is not None:
        bm, m = result["baseline"]["metrics"], result["current"]["metrics"]
        lines += ["| v1 metric | Baseline | Current |", "| --- | --- | --- |",
                  f"| False promotions | {bm['headings']['false_promotions']} | {m['headings']['false_promotions']} |",
                  f"| Missed headings | {bm['headings']['missed_headings']} | {m['headings']['missed_headings']} |",
                  f"| Levels correct | {bm['heading_levels']['correct']} | {m['heading_levels']['correct']} |",
                  f"| Assignments wrong | {bm['assignments']['wrong']} | {m['assignments']['wrong']} |",
                  f"| Continuations broken | {bm['continuations']['broken']} | {m['continuations']['broken']} |", ""]
    hist = result.get("historical_aggregate_metrics")
    if hist:
        lines += ["## Historical aggregate metrics (from the v1 reference; not recomputed)", "",
                  f"- {json.dumps(hist, ensure_ascii=False)}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ CLI

def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def run(structure_dir: Path, reference_path: Path | None = None, *, human_reference: Path | None = None,
        baseline_dir: Path | None = None, out_root: Path | None = None, stamp: str | None = None) -> Path:
    if reference_path is None and human_reference is None:
        raise EvaluationError("give --reference and/or --human-reference")
    view = load_structure(structure_dir)
    result: dict[str, Any] = {
        "schema_version": 2, "kind": "docprep_structure_evaluation", "docprep_version": __version__,
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "structure_out_dir": str(structure_dir),
        "structure_run_id": view.run_id}
    reference = None
    if reference_path is not None:
        reference, digest = load_reference(reference_path)
        expected_run = (reference.get("source") or {}).get("run_id")
        if expected_run and view.run_id and expected_run != view.run_id:
            raise EvaluationError(f"the reference is for {expected_run}, the structure output is for {view.run_id}")
        result.update(reference_file=str(reference_path), reference_sha256=digest,
                      reference_id=reference.get("reference_id"), reference_provenance=reference.get("provenance"),
                      current=evaluate_structure(view, reference),
                      historical_aggregate_metrics=reference.get("historical_aggregate_metrics"))
    human = None
    if human_reference is not None:
        human, coverage = load_human_reference(human_reference, view)
        combined, composition = combine_headings(reference, human)
        result.update(human_reference_file=str(human_reference), human_reference_provenance=human["provenance"],
                      human_coverage=coverage, human=evaluate_structure(view, human),
                      combined_headings=evaluate_structure(view, combined),
                      combined_headings_composition=composition, disagreements=disagreements(reference, human))
    if baseline_dir is not None:
        base_view = load_structure(baseline_dir)
        result["baseline_out_dir"] = str(baseline_dir)
        if reference is not None:
            result["baseline"] = evaluate_structure(base_view, reference)
            result["comparison"] = compare(result["current"], result["baseline"])
        if human is not None:
            result["human_baseline"] = evaluate_structure(base_view, human)
            result["human_comparison"] = compare(result["human"], result["human_baseline"])
    root = Path(out_root) if out_root else Path(structure_dir).resolve().parent.parent / "document_structure_evaluation"
    protected = [Path(structure_dir)] + ([Path(reference_path).parent] if reference_path else []) \
        + ([Path(human_reference).parent] if human_reference else []) + ([Path(baseline_dir)] if baseline_dir else [])
    for folder in protected:
        if _inside(root, folder):
            raise EvaluationError(f"evaluation output root {root} is inside an input folder ({folder})")
    contents = {"evaluation.json": json.dumps(result, ensure_ascii=False, indent=2),
                "evaluation.md": render_markdown(result)}

    def manifest(hashes: dict[str, str]) -> dict[str, Any]:
        return {"schema_version": 2, "kind": "docprep_evaluation_manifest", "docprep_version": __version__,
                "reference_sha256": result.get("reference_sha256"),
                "human_reference_sha256": (human or {}).get("provenance", {}).get("sha256"),
                "structure_out_dir": str(structure_dir),
                "baseline_out_dir": str(baseline_dir) if baseline_dir else None, "output_hashes": hashes}

    return write_outputs(root, f"eval__{Path(structure_dir).name}", contents, manifest, stamp=stamp)


def _pages(text: str) -> set[int]:
    try:
        return {int(p) for p in text.split(",") if p.strip()}
    except ValueError as exc:
        raise EvaluationError(f"--pages must be a comma-separated list of page numbers: {text}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m docprep.evaluate",
                                     description="Evaluate a structure output, or make a human review template.")
    parser.add_argument("structure_out_dir", type=Path, nargs="?")
    parser.add_argument("--reference", type=Path, default=None, help="frozen v1 (tool-reviewed) reference")
    parser.add_argument("--human-reference", type=Path, default=None, help="completed human review (CSV or JSON)")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--make-template", type=Path, default=None, metavar="STRUCTURE_OUT_DIR",
                        help="write an editable human review template for this structure output")
    parser.add_argument("--pages", type=str, default=None, help="pages for the template, e.g. 1,2,13")
    parser.add_argument("--out", type=Path, default=None, help="output root")
    args = parser.parse_args(argv)
    try:
        if args.make_template is not None:
            if not args.pages:
                raise EvaluationError("--make-template needs --pages")
            target = make_template(args.make_template, _pages(args.pages), out_root=args.out)
        else:
            if args.structure_out_dir is None:
                raise EvaluationError("give a structure output folder (or --make-template)")
            target = run(args.structure_out_dir, args.reference, human_reference=args.human_reference,
                         baseline_dir=args.baseline, out_root=args.out)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OutputExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(f"written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
