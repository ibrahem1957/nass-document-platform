"""Section tree, list nodes and one assignment per assignable unit."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import Config
from .model import HeadingCandidate, Unit
from .semantic import content_terms

CONF_ORDER = {"low": 0, "medium": 1, "high": 2}
LEVEL_NAMES = {0: "document title", 1: "major section", 2: "subsection", 3: "nested subsection"}
FURNITURE_ROLES = ("header", "footer", "page_number")


def _cap(conf: str, cap: str) -> str:
    return conf if CONF_ORDER[conf] <= CONF_ORDER[cap] else cap


def _down(conf: str) -> str:
    return {"high": "medium", "medium": "low", "low": "low"}[conf]


@dataclass
class Section:
    section_id: str
    title: str
    title_parts: list[str]
    level: int                         # normalized level (parent level + 1)
    raw_level: int                     # level from the heading's style / numbering class
    heading_candidate_id: str
    heading_unit_ids: list[str]
    heading_element_ids: list[str]
    parent: str | None
    children: list[str] = field(default_factory=list)
    content_unit_ids: list[str] = field(default_factory=list)
    list_ids: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    level_notes: list[str] = field(default_factory=list)

    @property
    def page_start(self) -> int:
        return min(self.pages)

    @property
    def page_end(self) -> int:
        return max(self.pages)


@dataclass
class ListItem:
    unit_id: str
    element_id: str
    span_id: str | None
    start: int | None
    end: int | None
    page: int
    marker: str
    marker_position: str | None
    continuation_unit_ids: list[str] = field(default_factory=list)


@dataclass
class ListNode:
    list_id: str
    section_id: str | None
    items: list[ListItem] = field(default_factory=list)
    intro_unit_id: str | None = None
    possible_continuations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def unit_ids(self) -> list[str]:
        out = []
        for item in self.items:
            out.append(item.unit_id)
            out.extend(item.continuation_unit_ids)
        return out

    @property
    def pages(self) -> list[int]:
        return sorted({item.page for item in self.items})

    def to_dict(self) -> dict[str, Any]:
        return {"list_id": self.list_id, "section_id": self.section_id, "intro_unit_id": self.intro_unit_id,
                "items": [asdict(i) for i in self.items], "unit_ids": self.unit_ids,
                "possible_continuations": self.possible_continuations, "pages": self.pages,
                "page_start": min(self.pages), "page_end": max(self.pages), "crosses_pages": len(self.pages) > 1}


@dataclass
class Assignment:
    unit_id: str
    element_id: str
    span_id: str | None
    page: int
    order: list[int]
    role: str
    original_type: str
    section_id: str | None
    list_id: str | None
    confidence: str | None
    reasons: list[str]
    supporting: list[str]
    conflicting: list[str]
    deterministic: bool
    review_recommended: bool
    validation_state: str | None
    derived_span_uncertain: bool | None
    role_uncertain: bool = False
    semantic_check: dict[str, Any] | None = None   # skipped | passed | flagged, with the reason or the evidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TreeResult:
    sections: list[Section]
    lists: list[ListNode]
    assignments: list[Assignment]
    warnings: list[str]


def content_role(unit: Unit, numbered_run: bool) -> str:
    if unit.furniture is not None:
        return unit.furniture.kind
    if unit.element_type in ("figure", "table"):
        return unit.element_type
    if not unit.clean:
        return "empty"
    f = unit.features
    if f.get("caption") or (unit.span is not None and unit.span.role_hint == "caption") or \
            (unit.layout is not None and unit.layout.role == "caption"):
        return "caption"
    if f.get("aside"):
        return "aside_label" if f["aside"]["kind"] == "label" else "aside_text"
    if f.get("list_intro"):
        return "list_intro"
    if f.get("list_marker") or (numbered_run and f.get("numbering")):
        return "list_item"
    if f.get("page_number"):
        return "page_number"
    return "paragraph"


def page_reading_order_flags(validation_report: dict[str, Any] | None) -> dict[int, list[tuple[str, str]]]:
    """Per page: (strength, message) for Nass flags about reading order. 'ambiguous' -> low, otherwise medium."""
    flags: dict[int, list[tuple[str, str]]] = {}
    for flag in (validation_report or {}).get("flags") or []:
        if not isinstance(flag, dict) or not isinstance(flag.get("page"), int):
            continue
        message = str(flag.get("message") or "")
        code = str(flag.get("code") or "")
        if "reading_order" in code or "reading order" in message:
            strength = "low" if "ambiguous" in message.lower() else "medium"
            flags.setdefault(flag["page"], []).append((strength, f"{code}: {message}"))
    return flags


def _continues_item(prev: Unit, cur: Unit, cfg: Config) -> tuple[bool | None, str]:
    """Is `cur` a wrapped continuation line of list item line `prev`? True (linked), None (text suggests it but
    geometry cannot confirm: keep separate, flag), False (no)."""
    pf, cf = prev.features, cur.features
    if cf.get("list_marker") or cf.get("numbering") or cf.get("caption") or not cur.clean:
        return False, ""
    if pf.get("terminal_punctuation") or pf.get("ends_colon"):
        return False, ""
    if cf.get("n_words", 0) > cfg.lists.continuation_max_words:
        return False, ""
    if prev.bbox is None or cur.bbox is None or prev.page != cur.page:
        why = "different page" if prev.page != cur.page else "no geometry for derived text"
        return None, f"text suggests a wrapped continuation of {prev.unit_id}, but it cannot be confirmed ({why})"
    line_height = (prev.bbox[3] - prev.bbox[1]) / max(1, pf.get("n_lines", 1))
    gap = cur.bbox[1] - prev.bbox[3]
    overlap = min(prev.bbox[2], cur.bbox[2]) - max(prev.bbox[0], cur.bbox[0])
    narrower = max(1.0, min(prev.bbox[2] - prev.bbox[0], cur.bbox[2] - cur.bbox[0]))
    if gap <= cfg.lists.continuation_max_gap_lines * line_height and overlap >= cfg.lists.continuation_min_x_overlap * narrower:
        return True, (f"wrapped continuation of {prev.unit_id}: unfinished item, short line, gap {gap:.1f}pt "
                      f"<= {cfg.lists.continuation_max_gap_lines} line ({line_height:.1f}pt), horizontal overlap")
    return False, ""


def build_tree(units: list[Unit], candidates: list[HeadingCandidate], validation_report: dict[str, Any] | None,
               cfg: Config) -> TreeResult:
    warnings: list[str] = []
    by_unit: dict[str, HeadingCandidate] = {}
    for c in candidates:
        for uid in c.unit_ids:
            by_unit[uid] = c
    heading_units = {uid for c in candidates if c.status == "promoted" for uid in c.unit_ids}
    order_flags = page_reading_order_flags(validation_report)

    # Numbered items that are not headings form lists when two or more are consecutive.
    numbered_runs: set[str] = set()
    run: list[str] = []
    for unit in units:
        if unit.furniture is not None:
            continue
        if unit.unit_id not in heading_units and unit.features.get("numbering") and unit.clean:
            run.append(unit.unit_id)
        else:
            if len(run) >= 2:
                numbered_runs.update(run)
            run = []
    if len(run) >= 2:
        numbered_runs.update(run)

    sections: list[Section] = []
    lists: list[ListNode] = []
    assignments: list[Assignment] = []
    stack: list[Section] = []
    section_by_id: dict[str, Section] = {}
    section_by_candidate: dict[str, Section] = {}
    candidate_by_id = {c.candidate_id: c for c in candidates}
    unit_by_id = {u.unit_id: u for u in units}
    last_page: dict[str, int] = {}
    bottom_heading: dict[str, int] = {}
    ambiguous_after: dict[str, HeadingCandidate] = {}
    current_list: ListNode | None = None
    last_item_unit: Unit | None = None      # last line (item or continuation) of the current list
    pending_intro: str | None = None
    previous_content: str | None = None
    no_headings = not heading_units
    content_on_page: dict[int, list[str]] = {}
    for unit in units:
        if unit.furniture is None and (unit.clean or unit.element_type in ("figure", "table")):
            content_on_page.setdefault(unit.page, []).append(unit.unit_id)

    for unit in units:
        candidate = by_unit.get(unit.unit_id)
        base = {"unit_id": unit.unit_id, "element_id": unit.element_id, "span_id": unit.span_id, "page": unit.page,
                "order": list(unit.order), "original_type": unit.element_type,
                "validation_state": unit.validation_state,
                "derived_span_uncertain": unit.span.uncertain if unit.span else None}
        # ---- headings
        if candidate is not None and candidate.status == "promoted":
            if unit.unit_id != candidate.unit_ids[0]:
                section = section_by_candidate[candidate.candidate_id]
                section.pages.append(unit.page)
                assignments.append(Assignment(**base, role="heading", section_id=section.section_id, list_id=None,
                                               confidence="high", reasons=[(f"continuation line of the heading of "
                                                                           f"{section.section_id}")],
                                              supporting=[], conflicting=[], deterministic=True,
                                              review_recommended=False))
                previous_content = unit.unit_id
                continue
            current_list, last_item_unit, pending_intro = None, None, None
            raw = candidate.level if candidate.level is not None else 1
            while stack and stack[-1].raw_level >= raw:
                stack.pop()
            parent = stack[-1] if stack else None
            level = parent.level + 1 if parent is not None else (0 if candidate.is_document_title else 1)
            notes = []
            if level != raw:
                notes.append(f"normalized from raw level {raw} to {level}: "
                             + (f"no heading between it and its parent {parent.section_id} (raw level {parent.raw_level})"
                                if parent is not None else "top-level section"))
                if candidate.level_confidence == "high":
                    candidate.level_confidence = "medium"
                    notes.append("level confidence capped at medium because the gap was collapsed")
                parent_candidate = candidate_by_id.get(parent.heading_candidate_id) if parent else None
                if parent_candidate is not None and parent_candidate.level_confidence == "low":
                    candidate.level_ambiguous = True
                    notes.append("parent heading's own level is uncertain: the parent may be wrong")
            candidate.raw_level, candidate.level = raw, level
            candidate.level_basis.extend(notes)
            section = Section(section_id=f"s{len(sections) + 1:03d}", title=candidate.text,
                              title_parts=candidate.parts, level=level, raw_level=raw,
                              heading_candidate_id=candidate.candidate_id, heading_unit_ids=list(candidate.unit_ids),
                              heading_element_ids=sorted({unit_by_id[x].element_id for x in candidate.unit_ids}),
                              parent=parent.section_id if parent else None, pages=[unit.page], level_notes=notes)
            if section.parent:
                section_by_id[section.parent].children.append(section.section_id)
            sections.append(section)
            section_by_id[section.section_id] = section
            section_by_candidate[candidate.candidate_id] = section
            stack.append(section)
            last_page[section.section_id] = unit.page
            page_items = content_on_page.get(unit.page, [])
            if page_items and page_items[-1] == candidate.unit_ids[-1]:
                bottom_heading[section.section_id] = unit.page
                section.flags.append(f"heading at the bottom of page {unit.page}")
            ambiguous_after.pop(section.section_id, None)
            assignments.append(Assignment(**base, role="heading", section_id=section.section_id, list_id=None,
                                          confidence="high", reasons=[f"heading of {section.section_id}"],
                                          supporting=[], conflicting=[], deterministic=True,
                                          review_recommended=False))
            previous_content = unit.unit_id
            continue
        role = content_role(unit, unit.unit_id in numbered_runs)
        # ---- page furniture and page numbers never interrupt a section or a list
        if role in FURNITURE_ROLES:
            assignments.append(Assignment(**base, role=role, section_id=None, list_id=None, confidence=None,
                                          reasons=[f"page furniture ({unit.furniture.reason})" if unit.furniture
                                                   else "page number"],
                                          supporting=[], conflicting=[], deterministic=True,
                                          review_recommended=False))
            continue
        if not stack:
            reason = "no heading was detected in the document" if no_headings else \
                "precedes the first detected heading"
            assignments.append(Assignment(**base, role=role, section_id=None, list_id=None, confidence=None,
                                          reasons=[reason], supporting=[], conflicting=[], deterministic=True,
                                          review_recommended=role not in ("empty",)))
            previous_content = unit.unit_id
            continue
        section = stack[-1]
        reasons = [f"follows the heading of {section.section_id} in reading order"]
        supporting: list[str] = []
        conflicting: list[str] = []
        deterministic = True
        role_uncertain = False
        heading = candidate_by_id[section.heading_candidate_id]
        confidence = _cap("high", heading.confidence or "low")
        if heading.confidence != "high":
            reasons.append(f"section heading detection confidence is {heading.confidence}")
        if candidate is not None and candidate.status == "rejected" and candidate.ambiguous:
            ambiguous_after[section.section_id] = candidate
            conflicting.append(f"this unit was a near-miss heading candidate ({candidate.candidate_id}, score "
                               f"{candidate.score:.2f})")
        if section.section_id in ambiguous_after and ambiguous_after[section.section_id].unit_ids[0] != unit.unit_id:
            near = ambiguous_after[section.section_id]
            confidence = "low"
            deterministic = False
            conflicting.append(f"follows a near-miss heading candidate '{near.text}' ({near.candidate_id}) that was "
                               "not promoted; it may start its own section")
        previous_page = last_page.get(section.section_id, unit.page)
        if unit.page > previous_page:
            if bottom_heading.get(section.section_id) == previous_page and \
                    section.content_unit_ids == [] and not section.list_ids:
                supporting.append(f"first content after a heading at the bottom of page {previous_page}")
            else:
                supporting.append(f"section continues from page {previous_page}")
        last_page[section.section_id] = max(previous_page, unit.page)
        if unit.span is not None:
            if unit.span.uncertain:
                confidence = _cap(confidence, "medium")
                deterministic = False
                conflicting.append("derived span with uncertain segmentation")
            supporting.append(f"derived span of {unit.element_id} ({unit.span.support} support)")
        for strength, message in order_flags.get(unit.page, []):
            confidence = _cap(confidence, strength)
            deterministic = False
            conflicting.append(f"Nass reading-order flag on page {unit.page}: {message}")
        if unit.validation_state and unit.validation_state != "ok":
            reasons.append(f"Nass validation state '{unit.validation_state}' preserved (OCR confidence is "
                           "independent of this assignment)")
        list_id = None
        # ---- wrapped continuation of the previous list item
        if role == "paragraph" and current_list is not None and last_item_unit is not None \
                and previous_content == last_item_unit.unit_id:
            linked, why = _continues_item(last_item_unit, unit, cfg)
            if linked:
                role = "list_item_continuation"
                current_list.items[-1].continuation_unit_ids.append(unit.unit_id)
                list_id = current_list.list_id
                last_item_unit = unit
                supporting.append(why)
            elif linked is None:
                role_uncertain = True
                current_list.possible_continuations.append({"unit_id": unit.unit_id,
                                                            "of_item": current_list.items[-1].unit_id,
                                                            "reason": why})
                conflicting.append(why + "; kept as a separate paragraph")
        if role == "list_item":
            if current_list is None or current_list.section_id != section.section_id:
                current_list = ListNode(list_id=f"L{len(lists) + 1:03d}", section_id=section.section_id)
                if pending_intro is not None and previous_content == pending_intro:
                    current_list.intro_unit_id = pending_intro
                lists.append(current_list)
                section.list_ids.append(current_list.list_id)
            elif current_list.items and unit.page > current_list.items[-1].page:
                supporting.append(f"list continues from page {current_list.items[-1].page}")
            current_list.items.append(ListItem(
                unit_id=unit.unit_id, element_id=unit.element_id, span_id=unit.span_id,
                start=unit.span.start if unit.span else None, end=unit.span.end if unit.span else None,
                page=unit.page, marker=unit.features.get("list_marker") or (unit.features.get("numbering") or {})
                .get("label") or "", marker_position=unit.features.get("marker_position") or (
                    "start" if unit.features.get("numbering") else None)))
            list_id = current_list.list_id
            last_item_unit = unit
            pending_intro = None
        elif role != "list_item_continuation":
            if role not in ("caption",):
                current_list, last_item_unit = None, None
            pending_intro = unit.unit_id if role == "list_intro" else None
            section.content_unit_ids.append(unit.unit_id)
        if role == "list_intro":
            reasons.append("list introduction: stays in the enclosing section; "
                           + "; ".join(unit.features["list_intro"]["cues"]))
        if role in ("aside_label", "aside_text"):
            role_uncertain = True
            reasons.append("callout or side box (layout-derived, uncertain); not part of the heading hierarchy: "
                           + unit.features["aside"]["reason"])
        section.pages.append(unit.page)
        assignments.append(Assignment(**base, role=role, section_id=section.section_id, list_id=list_id,
                                      confidence=confidence, reasons=reasons, supporting=supporting,
                                      conflicting=conflicting, deterministic=deterministic,
                                      review_recommended=confidence == "low" or role_uncertain,
                                      role_uncertain=role_uncertain))
        previous_content = unit.unit_id

    for section in sections:
        if not section.content_unit_ids and not section.list_ids:
            if section.children:
                section.flags.append("no direct content: its content is in child sections")
            else:
                section.flags.append("no content under this heading")
                candidate_by_id[section.heading_candidate_id].review = True
    if cfg.semantic.enabled and cfg.semantic.verifier != "none" and cfg.semantic.assignment_warnings:
        _semantic_warnings(sections, assignments, unit_by_id, candidate_by_id, cfg)
    if no_headings:
        warnings.append("no heading was promoted; every content unit is unassigned")
    return TreeResult(sections=sections, lists=lists, assignments=assignments, warnings=warnings)


def common_terms(sections: list[Section], assignments: list[Assignment], units: dict[str, Unit],
                 cfg: Config) -> tuple[set[str], dict[str, int], int]:
    """Document-wide terms: present (after stopword filtering and light stemming) in at least
    common_term_section_share of the sections, counting each section's title and its own content once."""
    section_terms: dict[str, set[str]] = {s.section_id: set(content_terms(s.title)) for s in sections}
    for a in assignments:
        if a.section_id is not None and a.role not in FURNITURE_ROLES and a.role != "heading":
            section_terms[a.section_id] |= content_terms(units[a.unit_id].text)
    n = len(sections)
    frequency: dict[str, int] = {}
    for terms in section_terms.values():
        for term in terms:
            frequency[term] = frequency.get(term, 0) + 1
    if n < cfg.semantic.min_sections_for_df:
        return set(), frequency, n
    # At least two sections, and at least the configured share: with few sections a term found in one section
    # would otherwise already pass the share.
    needed = max(2, math.ceil(cfg.semantic.common_term_section_share * n))
    common = {t for t, c in frequency.items() if c >= needed}
    return common, frequency, n


def _semantic_warnings(sections: list[Section], assignments: list[Assignment], units: dict[str, Unit],
                       candidates: dict[str, HeadingCandidate], cfg: Config) -> None:
    """A paragraph that shares meaningful terms with the NEXT heading and none with its own is flagged and its
    confidence lowered one step. It is never moved: only explicit structural evidence changes the structure.

    Document-wide common terms are ignored. The check is skipped (confidence untouched, reason recorded) when
    either heading has too few meaningful words, or when the next heading is itself uncertain (an unconfirmed
    structure is never used as evidence)."""
    s = cfg.semantic
    common, _frequency, n_sections = common_terms(sections, assignments, units, cfg)
    order = [x.section_id for x in sections]
    titles = {x.section_id: x.title for x in sections}
    heading_of = {x.section_id: x.heading_candidate_id for x in sections}

    def meaningful(text: str) -> set[str]:
        return content_terms(text) - common

    for a in assignments:
        if a.section_id is None or a.role not in ("paragraph", "aside_text") or a.confidence is None:
            continue
        index = order.index(a.section_id)
        if index + 1 >= len(order):
            a.semantic_check = {"status": "skipped", "reason": "last section: there is no following heading"}
            continue
        nxt_id = order[index + 1]
        own_terms, next_terms = meaningful(titles[a.section_id]), meaningful(titles[nxt_id])
        if len(own_terms) < s.min_heading_terms:
            a.semantic_check = {"status": "skipped", "reason": f"own heading has {len(own_terms)} meaningful "
                                f"content word(s) (< {s.min_heading_terms}); a generic heading cannot be compared"}
            continue
        if len(next_terms) < s.min_heading_terms:
            a.semantic_check = {"status": "skipped", "reason": f"next heading has {len(next_terms)} meaningful "
                                f"content word(s) (< {s.min_heading_terms})"}
            continue
        next_candidate = candidates.get(heading_of[nxt_id])
        if s.skip_when_next_heading_uncertain and next_candidate is not None and next_candidate.confidence == "low":
            a.semantic_check = {"status": "skipped", "reason": "the next heading's detection confidence is low; an "
                                                               "unconfirmed structure is not used as evidence"}
            continue
        terms = meaningful(units[a.unit_id].text)
        shared_next = sorted(terms & next_terms)
        shared_own = sorted(terms & own_terms)
        detail = {"shared_with_own_heading": shared_own, "shared_with_next_heading": shared_next,
                  "next_section": nxt_id, "ignored_common_terms": len(common), "sections_counted": n_sections}
        if len(shared_next) >= s.assignment_min_shared_terms and not shared_own:
            a.confidence = _down(a.confidence)
            a.deterministic = False
            a.conflicting.append(f"semantic warning: shares {len(shared_next)} term(s) ({', '.join(shared_next[:5])}) "
                                 f"with the next heading '{titles[nxt_id]}' ({nxt_id}) and none with its own heading; "
                                 "not moved (structure changes need structural evidence)")
            a.review_recommended = True  # a semantic warning always recommends a manual look
            a.semantic_check = {"status": "flagged", **detail}
        else:
            a.semantic_check = {"status": "passed", **detail}


def section_to_dict(section: Section, candidate: HeadingCandidate, lists: dict[str, ListNode]) -> dict[str, Any]:
    return {
        "section_id": section.section_id, "title": section.title, "title_parts": section.title_parts,
        "level": section.level, "raw_level": section.raw_level,
        "level_name": LEVEL_NAMES.get(section.level, f"level {section.level}"), "level_notes": section.level_notes,
        "parent": section.parent, "children": section.children,
        "heading_candidate_id": section.heading_candidate_id, "heading_unit_ids": section.heading_unit_ids,
        "heading_element_ids": section.heading_element_ids, "page_start": section.page_start,
        "page_end": section.page_end, "content_unit_ids": section.content_unit_ids,
        "list_nodes": [lists[i].to_dict() for i in section.list_ids],
        "confidence": {"heading_detection": candidate.confidence, "heading_level": candidate.level_confidence},
        "review_recommended": candidate.review,
        "evidence": {"score": round(candidate.score, 4), "signals": [s.to_dict() for s in candidate.signals],
                     "conflicts": candidate.conflicts, "semantic": candidate.semantic,
                     "level_basis": candidate.level_basis, "level_conflicts": candidate.level_conflicts,
                     "level_ambiguous": candidate.level_ambiguous},
        "flags": section.flags,
    }
