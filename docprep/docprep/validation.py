"""Consistency checks on the built structure. Findings are reported, never silently fixed."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .model import DerivedSpan, HeadingCandidate, SourceElement, Unit
from .segmentation import check_partition
from .tree import TreeResult


def _finding(code: str, severity: str, message: str, items: list[str] | None = None) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "items": (items or [])[:50],
            "item_count": len(items or [])}


def validate(elements: list[SourceElement], units: list[Unit], spans: dict[str, list[DerivedSpan]],
             candidates: list[HeadingCandidate], tree: TreeResult, page_count: int,
             hashes_before: dict[str, str | None], hashes_after: dict[str, str | None]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    element_ids = [e.element_id for e in elements]
    element_set = set(element_ids)

    # 1. Every original element is represented exactly once (as itself, or through its derived spans).
    duplicated = [i for i, n in Counter(element_ids).items() if n > 1]
    represented = {u.element_id for u in units}
    missing = sorted(element_set - represented)
    whole = Counter(u.element_id for u in units if u.span is None)
    double = sorted(i for i in element_set if whole[i] and i in spans)
    if duplicated or missing or double or any(n > 1 for n in whole.values()):
        findings.append(_finding("source_inventory", "error", "original elements not represented exactly once",
                                 duplicated + missing + double))
    else:
        findings.append(_finding("source_inventory", "ok", f"all {len(element_set)} original elements represented "
                                                           "exactly once"))
    # 2. Every derived span belongs to an existing element.
    orphans = sorted(s.span_id for lst in spans.values() for s in lst if s.element_id not in element_set)
    findings.append(_finding("span_parents", "error" if orphans else "ok",
                             "derived spans without an existing parent element" if orphans else
                             "every derived span maps to an existing element", orphans))
    # 3. Segmentation neither loses nor duplicates text.
    by_id = {e.element_id: e for e in elements}
    problems = []
    for element_id, span_list in spans.items():
        element = by_id.get(element_id)
        if element is None:
            continue
        problems += [f"{element_id}: {p}" for p in check_partition(element.text or "", span_list)]
    findings.append(_finding("span_coverage", "error" if problems else "ok",
                             "segmentation lost or duplicated text" if problems else
                             f"{sum(len(v) for v in spans.values())} derived spans exactly partition the text of "
                             f"{len(spans)} element(s)", problems))
    # 4. One assignment per assignable unit.
    unit_ids = [u.unit_id for u in units]
    assigned = Counter(a.unit_id for a in tree.assignments)
    dup_assign = sorted(i for i, n in assigned.items() if n > 1)
    unassigned_units = sorted(set(unit_ids) - set(assigned))
    unknown = sorted(set(assigned) - set(unit_ids))
    bad = dup_assign + unassigned_units + unknown
    findings.append(_finding("assignment_uniqueness", "error" if bad else "ok",
                             "assignable units without exactly one assignment record" if bad else
                             f"{len(unit_ids)} assignable units, one assignment record each", bad))
    # 5. Unique section ids.
    sids = [s.section_id for s in tree.sections]
    dup_sections = [i for i, n in Counter(sids).items() if n > 1]
    findings.append(_finding("section_ids", "error" if dup_sections else "ok",
                             "duplicate section ids" if dup_sections else f"{len(sids)} unique section ids",
                             dup_sections))
    # 6. Every section has a real, source-grounded heading.
    unit_by_id = {u.unit_id: u for u in units}
    cand_by_id = {c.candidate_id: c for c in candidates}
    ungrounded = []
    for section in tree.sections:
        candidate = cand_by_id.get(section.heading_candidate_id)
        if candidate is None or candidate.status != "promoted" or not section.heading_unit_ids:
            ungrounded.append(section.section_id)
            continue
        if any(uid not in unit_by_id for uid in section.heading_unit_ids):
            ungrounded.append(section.section_id)
            continue
        if section.title_parts != [unit_by_id[uid].text for uid in section.heading_unit_ids]:
            ungrounded.append(section.section_id)
    findings.append(_finding("section_headings", "error" if ungrounded else "ok",
                             "sections without a verbatim source heading" if ungrounded else
                             "every section heading is verbatim source text", ungrounded))
    # 7. Acyclic parent-child relations, consistent both ways.
    parent = {s.section_id: s.parent for s in tree.sections}
    cyclic = []
    for sid in parent:
        seen, node = set(), sid
        while node is not None:
            if node in seen:
                cyclic.append(sid)
                break
            seen.add(node)
            node = parent.get(node)
    inconsistent = [s.section_id for s in tree.sections for c in s.children if parent.get(c) != s.section_id]
    bad = sorted(set(cyclic + inconsistent))
    findings.append(_finding("hierarchy_acyclic", "error" if bad else "ok",
                             "cyclic or inconsistent parent/child links" if bad else "section hierarchy is acyclic",
                             bad))
    # 8. Reading order preserved.
    orders = [tuple(a.order) for a in tree.assignments]
    out_of_order = [tree.assignments[i].unit_id for i in range(1, len(orders)) if orders[i] < orders[i - 1]]
    findings.append(_finding("reading_order", "error" if out_of_order else "ok",
                             "assignments not in the original reading order" if out_of_order else
                             "assignments follow the original Nass reading order", out_of_order))
    # 9/10. Page references valid; multi-page sections consistent.
    bad_pages = [a.unit_id for a in tree.assignments if not (1 <= a.page <= max(page_count, 1))]
    span_issues = []
    for section in tree.sections:
        member_pages = [unit_by_id[uid].page for uid in section.heading_unit_ids + section.content_unit_ids]
        member_pages += [a.page for a in tree.assignments if a.section_id == section.section_id]
        if member_pages and (min(member_pages) != section.page_start or max(member_pages) != section.page_end):
            span_issues.append(section.section_id)
    findings.append(_finding("page_references", "error" if bad_pages else "ok",
                             "invalid page references" if bad_pages else "all page references are valid", bad_pages))
    multi = [s.section_id for s in tree.sections if s.page_end > s.page_start]
    findings.append(_finding("multi_page_sections", "error" if span_issues else "ok",
                             "section page range does not match its content" if span_issues else
                             f"{len(multi)} section(s) span several pages; page ranges match their content",
                             span_issues))
    # 11. Ambiguous assignments explicitly marked.
    unmarked = [a.unit_id for a in tree.assignments
                if a.confidence == "low" and not (a.review_recommended and (a.reasons or a.conflicting))]
    findings.append(_finding("ambiguity_marked", "error" if unmarked else "ok",
                             "low-confidence assignments without a review mark" if unmarked else
                             "every low-confidence assignment is marked for review", unmarked))
    # 12. Inputs unchanged.
    changed = sorted(k for k in hashes_before if hashes_before.get(k) != hashes_after.get(k))
    findings.append(_finding("inputs_unchanged", "error" if changed else "ok",
                             "input files changed during processing" if changed else
                             "no input file was modified (SHA-256 before and after)", changed))
    return findings
