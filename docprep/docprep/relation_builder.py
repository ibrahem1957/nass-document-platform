"""Deterministic, evidence-first relationship construction."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from typing import Any

from .relation_inputs import SOURCE_FILES, RelationshipInputs, hash_existing
from .relation_model import RelationNode, RelationResult, Relationship
from .relation_text import (
    CAPTION_RE,
    EXPLICIT_REF_RE,
    NON_VALUE_DIMENSIONS,
    caption_signature,
    content_terms,
    lexical_match,
    list_markers,
    marker_family,
    match_numeric_tokens,
    normalize_text,
    numeric_tokens,
)

CONFIDENCE = ((0.82, "high"), (0.62, "medium"), (0.0, "low"))
VISUAL_TYPES = {"figure", "table"}
TEXT_ROLES = {"paragraph", "text", "page_text", "list_intro", "list_item", "aside_text"}
TITLE_SIMILARITY_THRESHOLD = 0.72
NAMED_REFERENCE_SIMILARITY_THRESHOLD = 0.78
# Saved title-region geometry limits (the same limits Nass uses to attach titles; recorded in every candidate).
CAPTION_REGION_MAX_GAP_PT = 30.0
CAPTION_REGION_MIN_OVERLAP = 0.5
CAPTION_REGION_DECISION_MARGIN_PT = 5.0
# A caption span must agree with its visual's own OCR on at least this many distinctive words and share.
CAPTION_SPAN_MIN_SHARED_TERMS = 2
CAPTION_SPAN_MIN_COVERAGE = 0.6
# Crop-margin leakage: layout blocks that can carry caption text, and caption labels anywhere in a crop's OCR.
MARGIN_TITLE_LABELS = frozenset({"figure_title", "figure_caption", "table_caption", "chart_caption"})
MARGIN_TEXT_LABELS = frozenset({"text", "paragraph_title", "doc_title", "content", "reference", "footnote",
                                "vision_footnote", "abstract"})
CAPTION_LABEL_ANY_RE = re.compile(CAPTION_RE.pattern.replace("^", r"(?<![\w])", 1))
GENERIC_VISUAL_TERMS = {
    "traffic", "statistics", "data", "results", "analysis", "report", "summary", "chart", "figure", "table",
    "المرور", "الاحصاءات", "البيانات", "النتائج", "التحليل", "التقرير", "الملخص", "الشكل", "الجدول",
}
TITLE_TOKEN_RE = re.compile(r"[0-9A-Za-z\u0600-\u06ff]+")
STRUCTURAL_NUMBERING_RE = re.compile(
    r"^\s*\(?[0-9٠-٩۰-۹]+(?:\s*\.\s*[0-9٠-٩۰-۹]+)*\)?\s*(?:[.)]|[-–—:：])\s*"
)
METADATA_RE = re.compile(
    r"(?i)(?:^|\n)\s*(?:author|prepared\s+by|reviewed\s+by|إعداد|اعداد|عمل\s+وتحليل)\s*[:：\-–]?\s*\S+"
)


def _stable_id(prefix: str, *parts: object) -> str:
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _confidence(score: float, *, cap: str | None = None) -> str:
    value = next(label for floor, label in CONFIDENCE if score >= floor)
    if cap and {"low": 0, "medium": 1, "high": 2}[value] > {"low": 0, "medium": 1, "high": 2}[cap]:
        return cap
    return value


def _caption_title_text(text: str) -> str:
    """Return the lexical title portion of an associated caption, if it has one."""
    clean = text.strip()
    match = CAPTION_RE.match(clean)
    if not match:
        return clean
    return clean[match.end():].lstrip(" \t:：.-–—")


def _comparison_title(text: str) -> str:
    """Remove heading structure for comparison without changing stored source text."""
    return STRUCTURAL_NUMBERING_RE.sub("", text, count=1).rstrip(" \t:：.;،-–—")


def _valid_prefix_shape(candidate: str) -> bool:
    candidate_terms = content_terms(candidate)
    if not (2 <= len(candidate_terms) <= 20) or not (candidate_terms - GENERIC_VISUAL_TERMS):
        return False
    if len(candidate) > 180 or caption_signature(candidate):
        return False
    return sum(char.isdigit() for char in candidate) <= max(3, len(candidate) // 3)


def _leading_title_match(text: str, reference: str) -> dict[str, Any] | None:
    """Match one independently sourced title to a verbatim OCR prefix.

    Exact normalized matches are searched at every real word boundary. A near
    match normally uses the reference's normalized token count. One internal
    inserted or deleted content token is also allowed when every remaining token
    matches in order; an edge token is never consumed because it may be a chart label.
    """
    start = len(text) - len(text.lstrip())
    source = text[start:]
    comparison_reference = _comparison_title(reference)
    normalized_reference = normalize_text(comparison_reference)
    reference_tokens = normalized_reference.split()
    if len(reference_tokens) < 2:
        return None
    source_tokens = list(TITLE_TOKEN_RE.finditer(source))
    if len(source_tokens) < 2:
        return None

    exact_match = None
    for count in range(2, min(20, len(source_tokens)) + 1):
        end = start + source_tokens[count - 1].end()
        candidate = text[start:end]
        if _valid_prefix_shape(candidate) and normalize_text(candidate) == normalized_reference:
            exact_match = (candidate, end)
            break

    token_alignment: list[dict[str, Any]] = []
    if exact_match:
        candidate, end = exact_match
        match_kind = "exact_normalized"
    else:
        matched_candidate = None
        counts = [len(reference_tokens)]
        counts.extend(value for value in (len(reference_tokens) - 1, len(reference_tokens) + 1) if value >= 2)
        for count in counts:
            if len(source_tokens) < count:
                continue
            candidate_end = start + source_tokens[count - 1].end()
            possible = text[start:candidate_end]
            if not _valid_prefix_shape(possible):
                continue
            candidate_tokens = normalize_text(possible).split()
            matcher = SequenceMatcher(None, candidate_tokens, reference_tokens, autojunk=False)
            edits = [item for item in matcher.get_opcodes() if item[0] != "equal"]
            alignment = []
            if len(candidate_tokens) == len(reference_tokens):
                replacements = 0
                valid = True
                for index, (candidate_token, reference_token) in enumerate(
                    zip(candidate_tokens, reference_tokens, strict=True)
                ):
                    token_similarity = SequenceMatcher(
                        None, candidate_token, reference_token, autojunk=False,
                    ).ratio()
                    alignment.append({
                        "position": index, "candidate": candidate_token,
                        "reference": reference_token, "similarity": round(token_similarity, 4),
                    })
                    if candidate_token != reference_token:
                        replacements += 1
                        if token_similarity < 0.80:
                            valid = False
                kind = "near_exact" if valid and replacements == 1 else None
            else:
                kind = None
                if len(edits) == 1:
                    tag, left_start, left_end, right_start, right_end = edits[0]
                    inserted = tag == "delete" and left_end - left_start == 1 and right_end == right_start
                    deleted = tag == "insert" and right_end - right_start == 1 and left_end == left_start
                    edit_position = left_start if inserted else right_start
                    edited_token = (candidate_tokens[left_start] if inserted else
                                    reference_tokens[right_start] if deleted else "")
                    internal = 0 < edit_position < min(len(candidate_tokens), len(reference_tokens))
                    content_token = bool(re.search(r"[A-Za-z\u0600-\u06ff]", edited_token))
                    if (inserted or deleted) and internal and content_token:
                        kind = "token_edit"
                        alignment = [{
                            "operation": "candidate_insertion" if inserted else "candidate_deletion",
                            "position": edit_position,
                            "token": edited_token,
                        }]
            if kind:
                matched_candidate = possible, candidate_end, kind, alignment
                break
        if matched_candidate is None:
            return None
        candidate, end, match_kind, token_alignment = matched_candidate

    candidate_terms = content_terms(candidate)
    reference_terms = content_terms(comparison_reference)
    candidate_normalized = normalize_text(candidate)
    sequence = SequenceMatcher(None, candidate_normalized, normalized_reference, autojunk=False).ratio()
    lexical, shared = lexical_match(candidate, comparison_reference)
    similarity = max(sequence, lexical)
    coverage = len(shared) / max(1, max(len(candidate_terms), len(reference_terms)))
    if match_kind in {"near_exact", "token_edit"} and similarity < TITLE_SIMILARITY_THRESHOLD:
        return None
    # Numeric content is accepted only when it is explicitly identical in the source title.
    if any(char.isdigit() for char in candidate_normalized) and match_kind != "exact_normalized":
        return None
    return {
        "text": candidate,
        "offsets": [start, end],
        "similarity": round(similarity, 4),
        "sequence_similarity": round(sequence, 4),
        "lexical_similarity": round(lexical, 4),
        "shared_terms": shared,
        "term_coverage": round(coverage, 4),
        "match_kind": match_kind,
        "reference_text": reference,
        "comparison_reference": comparison_reference,
        "boundary_rule": (
            "shortest_exact_prefix" if match_kind == "exact_normalized" else
            "single_internal_content_token_edit" if match_kind == "token_edit" else
            "equal_token_count_single_spelling_substitution"
        ),
        "token_alignment": token_alignment,
    }


def _bbox_distance(source: list[float] | None, target: list[float] | None) -> tuple[float | None, float | None]:
    if not source or not target:
        return None, None
    vertical = max(0.0, max(source[1], target[1]) - min(source[3], target[3]))
    horizontal_overlap = max(0.0, min(source[2], target[2]) - max(source[0], target[0]))
    width = max(1.0, min(source[2] - source[0], target[2] - target[0]))
    return vertical, horizontal_overlap / width


def _conclusive_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Numeric matches that may count as evidence. A match whose precision cannot establish a stated bound, or
    whose qualifier is ambiguous, is inconclusive: it stays in the evidence but is not counted as proof."""
    return [item for item in matches if item.get("evidential_status", "supports") != "inconclusive"]


def _dominant_script(text: str) -> str | None:
    arabic = len(re.findall(r"[ء-ي]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic + latin == 0:
        return None
    if arabic >= 0.8 * (arabic + latin):
        return "arabic"
    if latin >= 0.8 * (arabic + latin):
        return "latin"
    return "mixed"


def _cross_script_limitation(visual_text: str, sources: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Every independent title source is in a different single script than the visual OCR."""
    visual_script = _dominant_script(visual_text)
    scripts = {str(item.get("source_id")): _dominant_script(str(item.get("text") or "")) for item in sources}
    if visual_script in (None, "mixed") or not scripts or any(
            value in (None, "mixed", visual_script) for value in scripts.values()):
        return None
    return {"visual_script": visual_script, "evidence_scripts": scripts}


@dataclass(frozen=True)
class ElementContext:
    element_id: str
    page: int
    order: int
    element_type: str
    text: str
    bbox: list[float] | None
    validation_state: str | None
    raw: dict[str, Any]
    section_id: str | None
    role: str | None


class Builder:
    def __init__(self, inputs: RelationshipInputs):
        self.inputs = inputs
        self.nodes: dict[str, RelationNode] = {}
        self.relationships: list[Relationship] = []
        self.findings: list[dict[str, Any]] = []
        self.evidence_audit: list[dict[str, Any]] = []
        self.warnings = list(inputs.run.warnings)
        self.assignments = list(inputs.assignments.get("assignments") or [])
        self.assignment_by_unit = {str(item.get("unit_id")): item for item in self.assignments}
        self.sections = {str(item.get("section_id")): item for item in inputs.section_tree.get("sections") or []}
        self.element_section = self._element_sections()
        self.elements = [self._context(item) for item in inputs.run.elements]
        self.by_id = {item.element_id: item for item in self.elements}
        self.source_inventory = list(inputs.assignments.get("source_inventory") or [])
        self.visuals: dict[str, dict[str, Any]] = {}
        self.list_output: dict[str, Any] = {
            "original_lists": [], "nodes": [], "relationships": [], "derived_spans": [],
            "unresolved_ambiguities": [],
        }
        self._make_base_nodes()

    @staticmethod
    def _provenance(element: ElementContext) -> dict[str, Any]:
        value = element.raw.get("provenance")
        return value if isinstance(value, dict) else {}

    def _layout_regions_by_block(self) -> dict[tuple[int, int], dict[str, Any]]:
        """Index saved Nass selection regions without inferring new geometry."""
        indexed: dict[tuple[int, int], dict[str, Any]] = {}
        for page in (self.inputs.run.layout or {}).get("pages") or []:
            if not isinstance(page, dict) or not isinstance(page.get("page"), int):
                continue
            selection = page.get("selection") if isinstance(page.get("selection"), dict) else {}
            for region in [*(selection.get("regions") or []), *(selection.get("others") or [])]:
                if (isinstance(region, dict) and isinstance(region.get("block_index"), int)
                        and region.get("region_id")):
                    indexed.setdefault((page["page"], region["block_index"]), region)
        return indexed

    def _element_by_region_id(self) -> dict[str, ElementContext]:
        indexed = {}
        for element in self.elements:
            region_id = self._provenance(element).get("job_id")
            if region_id:
                indexed[str(region_id)] = element
        return indexed

    def _layout_span_title_sources(self, visual: ElementContext) -> list[dict[str, Any]]:
        """Independent page-text spans usable only when Nass attached a title region to the visual."""
        provenance = self._provenance(visual)
        attachments = provenance.get("title_attachments")
        if isinstance(attachments, dict):
            attachments = [attachments]
        attachments = [item for item in attachments or [] if isinstance(item, dict)]
        if len(attachments) != 1 or provenance.get("title_ambiguity"):
            return []
        attachment = attachments[0]
        contested = [record for record in self._margin_leakage(visual)["records"]
                     if record.get("block_index") == attachment.get("block_index")
                     and record.get("reason_code") == "attached_title_with_competing_visual"]
        if contested:
            # The attached title region is also next to another visual: its text inside this crop may be that
            # visual's caption, so it is not independent support for this visual's title.
            self.evidence_audit.append({
                "candidate_type": "visual_title", "status": "rejected", "visual_id": visual.element_id,
                "page": visual.page, "reason": "saved title attachment is contested by another visual; "
                "possible caption leakage from the crop margin", "evidence": {"margin_leakage": contested},
            })
            return []
        sources = []
        for item in self.source_inventory:
            if item.get("page") != visual.page or str(item.get("element_id")) == visual.element_id:
                continue
            for span in item.get("derived_spans") or []:
                if not isinstance(span, dict) or span.get("uncertain") is True:
                    continue
                role = str(span.get("role_hint") or "")
                if role not in {"paragraph", "caption"}:
                    continue
                raw_text = str(span.get("text") or "").strip()
                reference = _caption_title_text(raw_text) if role == "caption" else raw_text
                if role == "paragraph" and re.search(r"[.!?؟…]\s*$", reference):
                    continue
                if not reference:
                    continue
                sources.append({
                    "text": reference,
                    "source_id": str(item.get("element_id")),
                    "source_ids": [str(item.get("element_id"))],
                    "span_id": str(span.get("span_id")),
                    "source_kind": "layout_attached_page_span",
                    "detection_method": "prefix_layout_title_match",
                    "layout_title_attachment": attachment,
                })
        return sources

    def _layout_title_regions(self) -> list[dict[str, Any]]:
        regions: list[dict[str, Any]] = []
        seen: set[tuple[int, tuple[float, ...]]] = set()
        for page in (self.inputs.run.layout or {}).get("pages") or []:
            if not isinstance(page, dict) or not isinstance(page.get("page"), int):
                continue
            candidates = list(page.get("boxes") or [])
            selection = page.get("selection") if isinstance(page.get("selection"), dict) else {}
            candidates.extend(selection.get("regions") or [])
            candidates.extend(selection.get("others") or [])
            for region in candidates:
                if not isinstance(region, dict) or region.get("label") != "figure_title":
                    continue
                bbox = region.get("bbox_pt") or region.get("bbox_pt_detected")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                key = page["page"], tuple(round(float(value), 3) for value in bbox)
                if key in seen:
                    continue
                seen.add(key)
                regions.append({"page": page["page"], "bbox_pt": list(key[1]),
                                "score": region.get("score"), "title_decision": region.get("title_decision")})
        return regions

    def _matching_title_region(self, element: ElementContext) -> dict[str, Any] | None:
        if not element.bbox:
            return None
        best = None
        best_overlap = 0.0
        area = max(1.0, (element.bbox[2] - element.bbox[0]) * (element.bbox[3] - element.bbox[1]))
        for region in self._layout_title_regions():
            if region["page"] != element.page:
                continue
            box = region["bbox_pt"]
            width = max(0.0, min(element.bbox[2], box[2]) - max(element.bbox[0], box[0]))
            height = max(0.0, min(element.bbox[3], box[3]) - max(element.bbox[1], box[1]))
            overlap = width * height / area
            if overlap > best_overlap:
                best, best_overlap = region, overlap
        if best and best_overlap >= 0.50:
            return {**best, "element_overlap_ratio": round(best_overlap, 4)}
        return None

    def _element_sections(self) -> dict[str, str | None]:
        grouped: dict[str, list[str | None]] = defaultdict(list)
        for item in self.assignments:
            grouped[str(item.get("element_id"))].append(item.get("section_id"))
        out = {}
        for element_id, values in grouped.items():
            nonempty = [str(value) for value in values if value]
            out[element_id] = Counter(nonempty).most_common(1)[0][0] if nonempty else None
        return out

    def _context(self, source) -> ElementContext:
        assignments = [item for item in self.assignments if item.get("element_id") == source.element_id]
        whole = next((item for item in assignments if item.get("unit_id") == source.element_id), None)
        chosen = whole or (assignments[0] if assignments else {})
        return ElementContext(
            source.element_id, source.page, source.order, source.element_type, source.text or "", source.bbox,
            source.validation_state, source.raw, self.element_section.get(source.element_id), chosen.get("role"),
        )

    def _make_base_nodes(self) -> None:
        document_id = f"document:{self.inputs.run.run_id}"
        self.nodes[document_id] = RelationNode(
            node_id=document_id, node_type="document", page=None, text=None, source_id=None,
        )
        for element in self.elements:
            self.nodes[element.element_id] = RelationNode(
                node_id=element.element_id, node_type=element.element_type, page=element.page,
                text=element.text, source_id=element.element_id,
                provenance={"validation_state": element.validation_state, "section_id": element.section_id},
                review_flags=(f"source_{element.validation_state}",)
                if element.validation_state in {"needs_review", "pending_review"} else (),
            )
            if element.element_type in VISUAL_TYPES:
                self.visuals[element.element_id] = {
                    "visual_id": element.element_id,
                    "visual_type": element.element_type,
                    "page": element.page,
                    "section_id": element.section_id,
                    "validation_state": element.validation_state,
                    "caption": None,
                    "internal_title": None,
                    "internal_title_evaluation": {
                        "status": "unassessed" if element.text else "rejected",
                        "reason": "title detection has not run" if element.text else
                                  "visual has no extracted text from which to establish a title",
                        "candidates": [],
                    },
                    "description_paragraphs": [],
                    "matched_numeric_evidence": [],
                    "leaked_text_overlaps": [],
                    "composite": {"parent": None, "parts": [], "unresolved": False},
                    "logical_table_id": None,
                    "review_flags": ([f"source_{element.validation_state}"]
                                     if element.validation_state in {"needs_review", "pending_review"} else []),
                    "rag_flags": (["needs_review"]
                                  if element.validation_state in {"needs_review", "pending_review"} else []),
                    "evidence_reliability": "unassessed",
                    "evidence_reliability_reasons": [],
                }
        for section_id, section in self.sections.items():
            self.nodes[section_id] = RelationNode(
                node_id=section_id, node_type="section", page=section.get("page_start"),
                text=str(section.get("title") or ""), source_id=None,
                provenance={"heading_element_ids": section.get("heading_element_ids") or []},
            )
        for original in self.inputs.section_tree.get("lists") or []:
            list_id = str(original.get("list_id"))
            self.nodes[list_id] = RelationNode(
                node_id=list_id, node_type="list", page=original.get("page_start"), text=None, source_id=None,
                provenance={"component": "document_structure", "section_id": original.get("section_id")},
            )

    def add_node(self, node: RelationNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing and existing != node:
            raise ValueError(f"derived node id collision: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_relation(
        self, relation_type: str, source_id: str, target_id: str, evidence: dict[str, Any], score: float,
        method: str, *, conflicts: list[dict[str, Any]] | None = None, limitations: list[str] | None = None,
        cap: str | None = None, review: bool = False,
    ) -> Relationship:
        rel_id = _stable_id("rel", relation_type, source_id, target_id, evidence)
        source, target = self.nodes[source_id], self.nodes[target_id]
        source_state = source.provenance.get("validation_state")
        target_state = target.provenance.get("validation_state")
        uncertain_states = {"needs_review", "pending_review"}
        state_uncertain = relation_type != "composite_part" and (
            source_state in uncertain_states or target_state in uncertain_states
        )
        effective_cap = "medium" if state_uncertain and cap != "low" else cap
        effective_limitations = list(limitations or [])
        if state_uncertain:
            effective_limitations.append("confidence capped because a source element requires Nass review")
        relation = Relationship(
            rel_id, relation_type, source_id, target_id, evidence, _confidence(score, cap=effective_cap), method,
            conflicts or [], source.node_type, target.node_type, source.page, target.page,
            review or bool(conflicts) or score < 0.62 or state_uncertain, effective_limitations,
            source_validation_state=source_state,
            target_validation_state=target_state,
        )
        self.relationships.append(relation)
        return relation

    def _visual_candidates(self, source: ElementContext, kind: str | None = None) -> list[ElementContext]:
        return [
            item for item in self.elements
            if item.page == source.page and item.element_type in VISUAL_TYPES and item.element_id != source.element_id
            and (kind is None or item.element_type == kind)
        ]

    def _title_region_associations(self) -> list[dict[str, Any]]:
        """Measured geometry between each saved figure_title region and the saved visual regions of its page.

        Uses Nass's structured `title_association` when the run has it; older runs are measured here from the
        saved boxes with the same limits. Geometry only: no title text is assigned to a region."""
        out: list[dict[str, Any]] = []
        region_elements = self._element_by_region_id()
        for page in (self.inputs.run.layout or {}).get("pages") or []:
            if not isinstance(page, dict) or not isinstance(page.get("page"), int):
                continue
            selection = page.get("selection") if isinstance(page.get("selection"), dict) else {}
            visuals = []
            for region in selection.get("regions") or []:
                if not isinstance(region, dict) or not isinstance(region.get("bbox_pt"), list):
                    continue
                element = next((region_elements[str(value)] for value in (region.get("job_id"), region.get("region_id"))
                                if value and str(value) in region_elements), None)
                box = region.get("bbox_pt_without_title") or region.get("bbox_pt")
                visuals.append({"block_index": region.get("block_index"), "element_id": element.element_id if element else None,
                                "bbox_pt": [float(v) for v in box],
                                "attached_title_blocks": [item.get("block_index") for item in region.get("title_attachments") or []
                                                          if isinstance(item, dict)]})
            titles = [item for item in (selection.get("others") or []) + (selection.get("attached_titles") or [])
                      if isinstance(item, dict) and item.get("label") == "figure_title"
                      and isinstance(item.get("bbox_pt"), list) and len(item["bbox_pt"]) == 4]
            for title in titles:
                tx0, ty0, tx1, ty1 = (float(v) for v in title["bbox_pt"])
                saved = title.get("title_association") if isinstance(title.get("title_association"), dict) else None
                relations = []
                for visual in visuals:
                    gx0, gy0, gx1, gy1 = visual["bbox_pt"]
                    overlap = max(0.0, min(tx1, gx1) - max(tx0, gx0)) / max(tx1 - tx0, 1e-6)
                    if overlap < CAPTION_REGION_MIN_OVERLAP:
                        continue
                    for position, gap, boundary_ok in (
                        ("title_above_visual", gy0 - ty1, ty0 <= gy0 + CAPTION_REGION_MAX_GAP_PT),
                        ("title_below_visual", ty0 - gy1, ty1 >= gy1 - CAPTION_REGION_MAX_GAP_PT),
                    ):
                        if -CAPTION_REGION_MAX_GAP_PT <= gap <= CAPTION_REGION_MAX_GAP_PT and boundary_ok:
                            relations.append({"block_index": visual["block_index"], "element_id": visual["element_id"],
                                              "position": position, "gap_pt": round(gap, 2),
                                              "horizontal_overlap_of_title": round(overlap, 3)})
                attached_to = next((visual["element_id"] for visual in visuals
                                    if title.get("block_index") in visual["attached_title_blocks"]), None)
                out.append({
                    "page": page["page"], "block_index": title.get("block_index"),
                    "bbox_pt": [round(v, 3) for v in (tx0, ty0, tx1, ty1)],
                    "relations": relations, "attached_to_element_id": attached_to,
                    "ambiguous": str(title.get("title_status") or "") == "ambiguous"
                    or (saved or {}).get("status") == "ambiguous",
                    "source": "nass_title_association" if saved else "measured_by_docprep_from_saved_boxes",
                    "saved_decision": title.get("title_decision"),
                })
        return out

    def _composite_member(self, element: ElementContext) -> bool:
        decisions = self._provenance(element).get("decisions")
        decisions = [decisions] if isinstance(decisions, dict) else decisions or []
        return element.validation_state == "included_in_composite" or any(
            isinstance(item, dict) and item.get("type") == "containment" for item in decisions)

    def build_span_captions(self) -> set[tuple[int, tuple[float, ...]]]:
        """caption_of from a caption-shaped derived span (text offsets known, coordinates unknown).

        Accepted only when independent evidence converges on one visual: (1) a caption pattern; (2) the caption's
        distinctive words agree with exactly one visual's own OCR (no other visual shares any of them); (3) no
        other visual's crop OCR contains the caption verbatim; (4) the visual is not a composite member and has no
        saved title ambiguity; (5) a saved figure_title region that is not attached to another visual sits next to
        that visual within the layout limits, and is nearer to it than to any other visual by the decision margin.
        The span-to-region correspondence itself is never measured (no paragraph coordinates), so confidence is
        capped at medium and review is recommended. Anything short of this is recorded as an unresolved candidate."""
        used: set[tuple[int, tuple[float, ...]]] = set()
        associations = self._title_region_associations()
        by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in self.source_inventory:
            owner = self.by_id.get(str(item.get("element_id")))
            if owner is None or owner.element_type in VISUAL_TYPES:
                continue
            for span in item.get("derived_spans") or []:
                if (isinstance(span, dict) and span.get("role_hint") == "caption" and span.get("uncertain") is not True
                        and caption_signature(str(span.get("text") or ""))):
                    by_page[owner.page].append({**span, "_owner": owner})
        for page, spans in sorted(by_page.items()):
            visuals = [item for item in self.elements if item.page == page and item.element_type in VISUAL_TYPES
                       and item.text.strip() and not self.visuals.get(item.element_id, {}).get("caption")]
            assessed = []
            for span in spans:
                text = str(span.get("text") or "").strip()
                title_terms = content_terms(_caption_title_text(text))
                verbatim = normalize_text(text)
                per_visual = []
                for visual in visuals:
                    visual_text = normalize_text(visual.text)
                    contains = bool(verbatim) and verbatim in visual_text
                    shared = sorted(title_terms & content_terms(visual_text.replace(verbatim, " ")))
                    per_visual.append({"visual_id": visual.element_id, "shared_terms": shared,
                                       "coverage": round(len(shared) / max(1, len(title_terms)), 3),
                                       "contains_caption_verbatim": contains})
                per_visual.sort(key=lambda item: (-len(item["shared_terms"]), -item["coverage"], item["visual_id"]))
                assessed.append((span, text, title_terms, per_visual))
            for span, text, title_terms, per_visual in assessed:
                failed: list[str] = []
                best = per_visual[0] if per_visual else None
                if best is None or len(best["shared_terms"]) < CAPTION_SPAN_MIN_SHARED_TERMS \
                        or best["coverage"] < CAPTION_SPAN_MIN_COVERAGE:
                    failed.append("caption words do not agree distinctively with any visual's own OCR")
                if best and any(item["shared_terms"] for item in per_visual[1:]):
                    failed.append("caption words also agree with another visual on the page")
                if best and any(item["contains_caption_verbatim"] for item in per_visual
                                if item["visual_id"] != best["visual_id"]):
                    failed.append("another visual's crop OCR contains this caption verbatim (it lies inside that crop)")
                if best and any(other is not span and other_visuals and other_visuals[0]["visual_id"] == best["visual_id"]
                                and len(other_visuals[0]["shared_terms"]) >= CAPTION_SPAN_MIN_SHARED_TERMS
                                for other, _t, _terms, other_visuals in assessed):
                    failed.append("another caption span agrees with the same visual")
                target = self.by_id.get(best["visual_id"]) if best else None
                if target is not None and self._composite_member(target):
                    failed.append("target is part of a saved composite/containment decision")
                if target is not None and self._provenance(target).get("title_ambiguity"):
                    failed.append("Nass recorded a title/caption ambiguity for the target visual")
                region = None
                if target is not None:
                    near = []
                    for association in associations:
                        if association["page"] != page or association["ambiguous"]:
                            continue
                        if association["attached_to_element_id"] not in (None, target.element_id):
                            continue
                        mine = [rel for rel in association["relations"] if rel["element_id"] == target.element_id]
                        others = [rel for rel in association["relations"] if rel["element_id"] != target.element_id]
                        if not mine or association["attached_to_element_id"] == target.element_id:
                            continue
                        my_gap = min(max(rel["gap_pt"], 0.0) for rel in mine)
                        if any(max(rel["gap_pt"], 0.0) < my_gap + CAPTION_REGION_DECISION_MARGIN_PT for rel in others):
                            continue
                        near.append(association)
                    if len(near) == 1:
                        region = near[0]
                    else:
                        failed.append("no single unattached saved figure_title region sits unambiguously next to "
                                      "the target visual" if not near else
                                      "several unattached figure_title regions sit next to the target visual")
                evidence = {
                    "span_id": span.get("span_id"), "source_element_id": span["_owner"].element_id,
                    "offsets": [span.get("start"), span.get("end")], "verbatim": text,
                    "caption_pattern": list(caption_signature(text) or ()),
                    "caption_title_terms": sorted(title_terms), "visual_text_agreement": per_visual,
                    "layout_title_region": region, "span_geometry": span.get("geometry"),
                    "span_geometry_used": False,
                }
                if failed or target is None or region is None:
                    self.findings.append({
                        "severity": "info", "code": "unresolved_caption_candidate", "items": [span["_owner"].element_id],
                        "message": "caption span was not linked: " + "; ".join(failed),
                        "evidence": {**evidence, "failed_checks": failed},
                    })
                    self.evidence_audit.append({
                        "candidate_type": "caption_of", "status": "unresolved", "source_id": span["_owner"].element_id,
                        "span_id": span.get("span_id"), "target_id": best["visual_id"] if best else None,
                        "page": page, "reason": "; ".join(failed), "evidence": evidence,
                    })
                    continue
                node_id = _stable_id("caption_span", span.get("span_id"), text)
                self.add_node(RelationNode(
                    node_id=node_id, node_type="caption_span", page=page, text=str(span.get("text")),  # verbatim
                    source_id=span["_owner"].element_id, start=span.get("start"), end=span.get("end"),
                    provenance={"verbatim": True, "span_id": span.get("span_id"), "geometry": None,
                                "validation_state": span["_owner"].validation_state},
                ))
                relation = self.add_relation(
                    "caption_of", node_id, target.element_id, evidence, 0.70,
                    "caption_span_text_agreement+layout_title_region", cap="medium", review=True,
                    limitations=["the caption span has no page coordinates (the OCR returns none); its correspondence "
                                 "to the layout title region is inferred, not measured",
                                 "full-page OCR geometry is a page container and was not used as caption position"],
                )
                used.add((region["page"], tuple(region["bbox_pt"])))
                self.visuals[target.element_id]["caption"] = {
                    "source_id": node_id, "text": text, "relationship_id": relation.rel_id,
                    "confidence": relation.confidence,
                }
                self.evidence_audit.append({
                    "candidate_type": "caption_of", "status": "accepted", "source_id": node_id,
                    "span_id": span.get("span_id"), "target_id": target.element_id, "page": page,
                    "reason": "caption pattern, unique visual text agreement and an unambiguous saved title region "
                              "converge on one visual", "evidence": evidence,
                })
        return used

    def _margin_leakage(self, visual: ElementContext) -> dict[str, Any]:
        """Possible caption/text leakage into a scanned visual crop through its margin.

        Uses Nass `dedup.suspected_margin_leakage` when the run has it. Older runs are measured here from the
        saved layout (block, region and crop pixels): geometric risk only. Nothing is ever confirmed and the
        visual's text is never changed."""
        cache = self.__dict__.setdefault("_leakage_cache", {})
        if visual.element_id in cache:
            return cache[visual.element_id]
        dedup = visual.raw.get("dedup") if isinstance(visual.raw.get("dedup"), dict) else {}
        if isinstance(dedup.get("suspected_margin_leakage"), list):
            result = {"source": "nass_dedup", "records": [item for item in dedup["suspected_margin_leakage"]
                                                          if isinstance(item, dict)]}
            cache[visual.element_id] = result
            return result
        result = {"source": "derived_by_docprep_from_saved_layout", "records": []}
        cache[visual.element_id] = result
        crop = (visual.raw.get("crop") or {}).get("pixel_box") if isinstance(visual.raw.get("crop"), dict) else None
        job_id = self._provenance(visual).get("job_id")
        page = next((item for item in (self.inputs.run.layout or {}).get("pages") or []
                     if isinstance(item, dict) and item.get("page") == visual.page), None)
        selection = (page or {}).get("selection") if isinstance((page or {}).get("selection"), dict) else {}
        region = next((item for item in selection.get("regions") or [] if isinstance(item, dict) and job_id
                       and str(job_id) in (str(item.get("job_id")), str(item.get("region_id")))), None)
        if not (isinstance(crop, list) and len(crop) == 4 and region and isinstance(region.get("bbox_px"), list)):
            result["source"] = "unavailable: no saved crop or region pixels"
            return result
        detected = [float(v) for v in region["bbox_px"]]
        crop = [float(v) for v in crop]
        own = {item.get("block_index") for item in region.get("title_attachments") or [] if isinstance(item, dict)}
        associations = {(item["page"], item["block_index"]): item for item in self._title_region_associations()}
        labels = [{"text": m.group(0).strip(), "offsets": [m.start(), m.end()]}
                  for m in CAPTION_LABEL_ANY_RE.finditer(visual.text or "")]

        def area(box: list[float]) -> float:
            return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

        def inter(a: list[float], b: list[float]) -> float:
            return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])

        for block in (selection.get("others") or []) + (selection.get("attached_titles") or []):
            label = block.get("label") if isinstance(block, dict) else None
            box = block.get("bbox_px") if isinstance(block, dict) else None
            if label not in MARGIN_TITLE_LABELS | MARGIN_TEXT_LABELS or not (isinstance(box, list) and len(box) == 4):
                continue
            box = [float(v) for v in box]
            if area(box) <= 0 or inter(box, crop) <= 0 or inter(box, detected) / area(box) > 0.10:
                continue
            association = associations.get((visual.page, block.get("block_index")))
            competing = sorted({rel.get("element_id") for rel in (association or {}).get("relations") or []
                                if rel.get("element_id") and rel.get("element_id") != visual.element_id})
            if label not in MARGIN_TITLE_LABELS:
                reason = "text_block_in_crop_margin"
            elif block.get("block_index") in own:
                if not competing:
                    continue  # this visual's own unambiguous attached title
                reason = "attached_title_with_competing_visual"
            elif str(block.get("title_status") or "") == "ambiguous":
                reason = "ambiguous_title_in_crop_margin"
            elif str(block.get("title_status") or "").startswith("attached to block"):
                reason = "other_visual_title_in_crop_margin"
            else:
                reason = "unattached_title_in_crop_margin"
            result["records"].append({
                "reason_code": reason, "block_index": block.get("block_index"), "block_label": label,
                "block_bbox_px": box, "visual_detected_bbox_px": detected, "crop_pixel_box": crop, "units": "px",
                "block_share_inside_crop": round(inter(box, crop) / area(box), 3),
                "competing_visual_ids": competing, "crop_caption_labels": labels,
                "label_attribution": "unattributed",
                "evidence_level": "geometric_risk_with_caption_label_in_crop_text" if labels else "geometric_risk_only",
                "confirmed": False,
            })
        return result

    def report_margin_leakage(self) -> None:
        """Annotate visuals with possible crop-margin leakage; review is recommended when the block is contested,
        ambiguous or belongs to another visual. Text, titles and captions are not changed here."""
        review_reasons = {"attached_title_with_competing_visual", "ambiguous_title_in_crop_margin",
                          "other_visual_title_in_crop_margin"}
        for visual in (item for item in self.elements if item.element_type in VISUAL_TYPES and item.text.strip()):
            leakage = self._margin_leakage(visual)
            if not leakage["records"]:
                continue
            review = any(item.get("reason_code") in review_reasons for item in leakage["records"])
            self.visuals[visual.element_id]["margin_leakage_risk"] = {**leakage, "review_recommended": review}
            self.findings.append({
                "severity": "warning" if review else "info", "code": "possible_caption_margin_leakage",
                "items": [visual.element_id],
                "message": "layout title/text blocks outside the visual lie inside its margin-expanded crop; its OCR "
                           "may contain neighbouring caption text (not confirmed; nothing was removed)",
                "evidence": {"source": leakage["source"], "records": leakage["records"],
                             "review_recommended": review},
            })

    @staticmethod
    def _page_level_geometry(element: ElementContext) -> bool:
        """The element's box is a page container (full-page OCR), not the position of any sentence in it."""
        return element.element_type == "page_text" or str(element.raw.get("source") or "") == "qari_full_page"

    def build_captions(self) -> None:
        used_layout_regions: set[tuple[int, tuple[float, ...]]] = set()
        for caption in self.elements:
            if self._page_level_geometry(caption):
                # A whole page of OCR is never one caption; its caption spans are handled by build_span_captions
                # without using the page box as sentence-level geometry.
                continue
            signature = caption_signature(caption.text)
            saved_role = caption.role == "caption"
            layout_region = self._matching_title_region(caption)
            layout_role = str(caption.raw.get("label") or "") == "figure_title" or layout_region is not None
            if caption.element_type in VISUAL_TYPES and not saved_role and not layout_role:
                continue
            if not signature and not saved_role and not layout_role:
                continue
            if layout_region:
                used_layout_regions.add((layout_region["page"], tuple(layout_region["bbox_pt"])))
            kind = signature[0] if signature else None
            candidates = []
            for visual in self._visual_candidates(caption, kind):
                distance, overlap = _bbox_distance(caption.bbox, visual.bbox)
                score = 0.42
                evidence: dict[str, Any] = {
                    "caption_pattern": list(signature) if signature else None,
                    "saved_structure_role": caption.role,
                    "saved_layout_label": caption.raw.get("label"),
                    "saved_figure_title_region": layout_region,
                    "same_page": True,
                    "same_section": caption.section_id is not None and caption.section_id == visual.section_id,
                    "reading_order_distance": abs(caption.order - visual.order),
                }
                if caption.section_id and caption.section_id == visual.section_id:
                    score += 0.12
                score += max(0.0, 0.12 - 0.025 * abs(caption.order - visual.order))
                if distance is not None:
                    score += max(0.0, 0.24 - distance / 400)
                    score += 0.07 * min(1.0, overlap or 0.0)
                    evidence.update({"vertical_gap_pt": round(distance, 3), "horizontal_overlap_ratio": round(overlap or 0, 4)})
                if layout_role:
                    score += 0.08
                candidates.append((score, visual, evidence))
            candidates.sort(key=lambda item: (-item[0], item[1].order, item[1].element_id))
            if not candidates:
                self.findings.append(self._finding("warning", "unlinked_caption", caption.element_id,
                                                   "caption has no compatible visual on its page"))
                continue
            top = candidates[0]
            if top[0] < 0.55:
                self.findings.append({
                    "severity": "warning", "code": "weak_caption_target", "items": [caption.element_id],
                    "message": "caption candidate was not linked because its spatial/order support is too weak",
                    "evidence": {"best_target": top[1].element_id, "score": round(top[0], 4)},
                })
                continue
            alternatives = [
                {"target_id": item[1].element_id, "score": round(item[0], 4), "reason": item[2]}
                for item in candidates[1:] if top[0] - item[0] <= 0.12
            ]
            if len(candidates) > 1 and top[0] - candidates[1][0] < 0.03:
                self.findings.append({
                    "severity": "warning", "code": "ambiguous_caption_target", "items": [caption.element_id],
                    "message": "caption was not assigned because two visual targets have equivalent support",
                    "conflicts": [{"target_id": item[1].element_id, "score": round(item[0], 4)} for item in candidates[:3]],
                })
                continue
            relation = self.add_relation(
                "caption_of", caption.element_id, top[1].element_id, top[2], top[0], "caption_pattern+layout",
                conflicts=alternatives, cap=None if caption.bbox and top[1].bbox else "medium",
                review=bool(alternatives), limitations=[] if caption.bbox else ["no source geometry; reading order only"],
            )
            self.visuals[top[1].element_id]["caption"] = {
                "source_id": caption.element_id, "text": caption.text, "relationship_id": relation.rel_id,
                "confidence": relation.confidence,
            }
        used_layout_regions |= self.build_span_captions()
        associations = {(item["page"], tuple(item["bbox_pt"])): item for item in self._title_region_associations()}
        for region in self._layout_title_regions():
            key = region["page"], tuple(region["bbox_pt"])
            if key not in used_layout_regions:
                association = associations.get(key)
                self.findings.append({
                    "severity": "warning", "code": "unresolved_layout_figure_title", "items": [],
                    "message": "saved figure_title region has no source text element with verifiable coordinates; "
                    "no caption text or relationship was fabricated",
                    "evidence": {**region, **({"measured_visual_relations": association["relations"],
                                               "association_source": association["source"]}
                                              if association else {})},
                })

    def _single_line_title_match(self, visual: ElementContext) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
        sources: list[dict[str, Any]] = []
        if visual.section_id and visual.section_id in self.sections:
            section = self.sections[visual.section_id]
            heading_ids = [str(value) for value in section.get("heading_element_ids") or [] if value]
            sources.append({
                "text": str(section.get("title") or ""),
                "source_id": heading_ids[0] if len(heading_ids) == 1 else visual.section_id,
                "source_ids": heading_ids or [visual.section_id],
                "source_kind": "section_heading",
                "detection_method": "prefix_heading_match",
            })
        caption = self.visuals[visual.element_id].get("caption") or {}
        caption_text = _caption_title_text(str(caption.get("text") or ""))
        if caption_text:
            sources.append({
                "text": caption_text,
                "source_id": str(caption.get("source_id")),
                "source_ids": [str(caption.get("source_id"))],
                "source_kind": "explicit_caption",
                "detection_method": "prefix_caption_match",
            })
        for key in ("internal_title", "visual_title", "title"):
            value = visual.raw.get(key)
            if isinstance(value, str) and value.strip() and visual.raw.get(f"{key}_verified") is True:
                sources.append({
                    "text": value,
                    "source_id": visual.element_id,
                    "source_ids": [visual.element_id],
                    "source_kind": f"verified_{key}_metadata",
                    "detection_method": "prefix_metadata_match",
                })
        sources.extend(self._layout_span_title_sources(visual))

        candidates: list[dict[str, Any]] = []
        for source in sources:
            matched = _leading_title_match(visual.text, source["text"])
            if matched:
                source_values = {key: value for key, value in source.items() if key != "text"}
                if matched["match_kind"] == "token_edit":
                    source_values["detection_method"] = (
                        "prefix_heading_match_token_edit" if source["source_kind"] == "section_heading" else
                        f"{source['detection_method']}_token_edit"
                    )
                candidates.append({**matched, **source_values})
        if not candidates:
            limitation = _cross_script_limitation(visual.text, sources)
            if limitation:
                return None, (f"single-line visual: all independent title evidence is in another script "
                              f"({', '.join(sorted(set(limitation['evidence_scripts'].values())))}) than the visual "
                              f"OCR ({limitation['visual_script']}); a literal prefix match is impossible and "
                              "translation is not used as evidence"), []
            return None, "single-line visual has no strong prefix match to independent title evidence", []

        distinct = {(tuple(item["offsets"]), normalize_text(item["text"])) for item in candidates}
        if len(distinct) > 1:
            return None, "conflicting title evidence identifies different OCR prefixes", candidates

        candidates.sort(key=lambda item: (-item["similarity"], item["source_kind"], item["source_id"]))
        accepted = dict(candidates[0])
        accepted["support"] = [{
            "source_id": item["source_id"],
            "source_ids": item["source_ids"],
            "source_kind": item["source_kind"],
            "detection_method": item["detection_method"],
            "similarity": item["similarity"],
            "sequence_similarity": item["sequence_similarity"],
            "lexical_similarity": item["lexical_similarity"],
            "term_coverage": item["term_coverage"],
            "match_kind": item["match_kind"],
            "shared_terms": item["shared_terms"],
            "reference_text": item["reference_text"],
            "comparison_reference": item["comparison_reference"],
            "boundary_rule": item["boundary_rule"],
            "token_alignment": item["token_alignment"],
            **({"span_id": item["span_id"]} if item.get("span_id") else {}),
            **({"layout_title_attachment": item["layout_title_attachment"]}
               if item.get("layout_title_attachment") else {}),
        } for item in candidates]
        return accepted, "supported by an unambiguous independent prefix match", candidates

    def build_internal_titles(self) -> None:
        for visual in (item for item in self.elements if item.element_type in VISUAL_TYPES and item.text):
            lines = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"[^\r\n]+", visual.text)]
            evaluation = self.visuals[visual.element_id]["internal_title_evaluation"]
            prefix_match = None
            if len(lines) < 2:
                prefix_match, reason, prefix_candidates = self._single_line_title_match(visual)
                if prefix_match is None:
                    evaluation.update({"status": "rejected", "reason": reason, "candidates": prefix_candidates})
                    if "another script" in reason:
                        evaluation["limitation"] = {
                            "kind": "cross_script_title_evidence",
                            "effect": "internal_title stays null; no title is inferred from translation, topic "
                                      "or legend labels",
                        }
                    self.evidence_audit.append({
                        "candidate_type": "visual_title", "status": "rejected", "visual_id": visual.element_id,
                        "page": visual.page, "section_id": visual.section_id,
                        "reason": reason, "candidate": None, "candidates": prefix_candidates,
                    })
                    continue
                start, end = prefix_match["offsets"]
                text = visual.text[start:end]
            else:
                text, start, end = lines[0]
            clean = text.strip()
            terms = content_terms(clean)
            structural_table_line = clean.startswith("|") and clean.endswith("|")
            continuation_marker = bool(re.search(
                r"(?i)(?:table\s+continued|continued\s+from\s+page|تابع\s+الجدول|استكمال\s+الجدول)", clean
            ))
            rejection = None
            if not (2 <= len(terms) <= 20) or caption_signature(clean) or len(clean) > 180 or \
                    structural_table_line or continuation_marker:
                rejection = "leading line failed title shape, caption, table-header, or continuation gates"
            elif sum(ch.isdigit() for ch in clean) > max(3, len(clean) // 3):
                rejection = "leading line is dominated by numeric content"
            if rejection:
                evaluation.update({"status": "rejected", "reason": rejection,
                                   "candidates": [{"text": clean, "offsets": [start, end]}]})
                self.evidence_audit.append({
                    "candidate_type": "visual_title", "status": "rejected", "visual_id": visual.element_id,
                    "page": visual.page, "section_id": visual.section_id, "candidate": clean,
                    "offsets": [start, end], "reason": rejection,
                })
                continue
            start += len(text) - len(text.lstrip())
            end = start + len(clean)
            prefix_support = prefix_match.get("support", []) if prefix_match else []
            supported_attachments = [
                item["layout_title_attachment"] for item in prefix_support if item.get("layout_title_attachment")
            ]
            for attachment in supported_attachments:
                attachment_bbox = attachment.get("bbox_pt")
                self.findings = [finding for finding in self.findings if not (
                    finding.get("code") == "unresolved_layout_figure_title"
                    and (finding.get("evidence") or {}).get("page") == visual.page
                    and (finding.get("evidence") or {}).get("bbox_pt") == attachment_bbox
                )]
            evidence_references = [visual.element_id] + [
                source_id for item in prefix_support for source_id in item["source_ids"]
            ]
            evidence_references = list(dict.fromkeys(evidence_references))
            methods: list[str] = list(dict.fromkeys(
                item["detection_method"] for item in prefix_support
            ))
            similarities: dict[str, float] = {
                item["source_kind"]: item["similarity"] for item in prefix_support
            }
            heading = ""
            if visual.section_id and visual.section_id in self.nodes:
                heading = self.nodes[visual.section_id].text or ""
            if len(lines) >= 2 and heading:
                lexical, shared = lexical_match(clean, heading)
                sequence = SequenceMatcher(None, normalize_text(clean), normalize_text(heading), autojunk=False).ratio()
                similarities["section_heading"] = round(max(lexical, sequence), 4)
                if len(shared) >= 2 and max(lexical, sequence) >= TITLE_SIMILARITY_THRESHOLD:
                    methods.append("normalized_section_heading_match")
                    evidence_references.append(visual.section_id)
            caption = self.visuals[visual.element_id].get("caption") or {}
            caption_text = str(caption.get("text") or "")
            if len(lines) >= 2 and caption_text:
                lexical, shared = lexical_match(clean, caption_text)
                sequence = SequenceMatcher(None, normalize_text(clean), normalize_text(caption_text),
                                           autojunk=False).ratio()
                similarity = max(lexical, sequence)
                similarities["explicit_caption"] = round(similarity, 4)
                if len(shared) >= 2 and (similarity >= TITLE_SIMILARITY_THRESHOLD or
                                         normalize_text(clean) in normalize_text(caption_text)):
                    methods.append("normalized_explicit_caption_match")
                    evidence_references.append(str(caption.get("source_id")))
            upstream_title = next((visual.raw.get(key) for key in ("internal_title", "visual_title", "title")
                                   if isinstance(visual.raw.get(key), str) and visual.raw.get(key).strip()), None)
            if len(lines) >= 2 and upstream_title:
                sequence = SequenceMatcher(None, normalize_text(clean), normalize_text(upstream_title),
                                           autojunk=False).ratio()
                similarities["upstream_metadata"] = round(sequence, 4)
                if sequence >= 0.90:
                    methods.append("explicit_upstream_metadata")
            following = " ".join(value[0].strip() for value in lines[1:3])
            structural = (
                2 <= len(terms) <= 12 and len(clean) <= 120
                and not re.search(r"[.!?؟…]\s*$", clean)
                and bool(re.search(r"[0-9٠-٩۰-۹%٪]", following) or len(content_terms(following)) >= 2)
            )
            if len(lines) >= 2 and structural:
                methods.append("leading_title_line_structure")
            if not methods:
                reason = "leading line lacks structural, caption, section-heading, or upstream title support"
                evaluation.update({"status": "rejected", "reason": reason,
                                   "candidates": [{"text": clean, "offsets": [start, end],
                                                   "similarities": similarities}]})
                self.evidence_audit.append({
                    "candidate_type": "visual_title", "status": "rejected", "visual_id": visual.element_id,
                    "page": visual.page, "section_id": visual.section_id, "candidate": clean,
                    "offsets": [start, end], "normalized_candidate": normalize_text(clean),
                    "similarities": similarities, "reason": reason,
                })
                continue
            node_id = _stable_id("visual_title", visual.element_id, start, end, clean)
            self.add_node(RelationNode(
                node_id=node_id, node_type="visual_title_span", page=visual.page, text=clean,
                source_id=visual.element_id, start=start, end=end,
                provenance={"verbatim": True, "detection_methods": methods,
                            "evidence_references": evidence_references,
                            "similarities": similarities, "source_evidence": prefix_support},
            ))
            confidence_score = 0.88 if any(method != "leading_title_line_structure" for method in methods) else 0.72
            relation = self.add_relation(
                "title_inside_visual", node_id, visual.element_id,
                {"offsets": [start, end], "verbatim": clean, "remaining_lines": len(lines) - 1,
                 "detection_methods": methods, "similarities": similarities,
                  "evidence_references": evidence_references, "source_evidence": prefix_support},
                confidence_score, "evidence_supported_leading_title",
            )
            self.visuals[visual.element_id]["internal_title"] = {
                "node_id": node_id, "text": clean, "offsets": [start, end], "relationship_id": relation.rel_id,
                "source_id": visual.element_id, "detection_methods": methods,
                "similarity_scores": similarities, "confidence": relation.confidence,
                "evidence_references": evidence_references, "source_evidence": prefix_support,
            }
            evaluation.update({"status": "accepted", "reason": "supported by deterministic title evidence",
                               "candidates": [{"text": clean, "offsets": [start, end],
                                               "methods": methods, "similarities": similarities}]})
            self.evidence_audit.append({
                "candidate_type": "visual_title", "status": "accepted", "visual_id": visual.element_id,
                "page": visual.page, "section_id": visual.section_id, "candidate": clean,
                "offsets": [start, end], "normalized_candidate": normalize_text(clean), "rule": methods,
                "confidence": relation.confidence, "similarities": similarities,
                "evidence_references": evidence_references, "source_evidence": prefix_support,
            })
            if heading:
                ratio, shared = lexical_match(clean, heading)
                sequence = SequenceMatcher(None, normalize_text(clean), normalize_text(heading), autojunk=False).ratio()
                similarity = max(ratio, sequence)
                if len(shared) >= 2 and similarity >= TITLE_SIMILARITY_THRESHOLD:
                    self.add_relation(
                        "titled_by_section", node_id, visual.section_id,
                        {"lexical_jaccard": round(ratio, 4), "sequence_similarity": round(sequence, 4),
                         "effective_similarity": round(similarity, 4), "threshold": TITLE_SIMILARITY_THRESHOLD,
                         "shared_terms": shared},
                        min(0.95, 0.58 + similarity / 2), "normalized_title_heading_match",
                    )

    def build_leakage(self) -> None:
        text_elements = [item for item in self.elements if item.element_type not in VISUAL_TYPES and len(item.text.strip()) >= 20]
        for visual in (item for item in self.elements if item.element_type in VISUAL_TYPES and item.text):
            visual_normal = normalize_text(visual.text)
            if len(visual_normal) < 20:
                continue
            for source in text_elements:
                if source.page != visual.page:
                    continue
                source_normal = normalize_text(source.text)
                if len(source_normal) < 20:
                    continue
                exact = source_normal in visual_normal or visual_normal in source_normal
                ratio = min(len(source_normal), len(visual_normal)) / max(len(source_normal), len(visual_normal)) if exact else \
                    SequenceMatcher(None, source_normal, visual_normal, autojunk=False).ratio()
                if (not exact and ratio < 0.72) or (exact and min(len(source_normal), len(visual_normal)) < 20):
                    continue
                offsets = None
                raw_index = visual.text.casefold().find(source.text.strip().casefold())
                if raw_index >= 0:
                    offsets = [raw_index, raw_index + len(source.text.strip())]
                evidence = {
                    "match_kind": "exact" if exact else "approximate",
                    "normalized_overlap_ratio": round(ratio, 4),
                    "target_offsets": offsets,
                    "source_length": len(source.text),
                    "may_be_incidental": not exact or ratio < 0.85,
                }
                relation = self.add_relation(
                    "leaked_text", source.element_id, visual.element_id, evidence,
                    0.88 if exact and offsets else 0.64, "normalized_overlap",
                    cap=None if offsets else "medium", review=not exact,
                    limitations=[] if offsets else ["normalized overlap has no exact raw character offsets"],
                )
                self.visuals[visual.element_id]["leaked_text_overlaps"].append({
                    "source_id": source.element_id, "relationship_id": relation.rel_id, **evidence,
                })

    def _caption_targets(self) -> dict[tuple[str, str], str]:
        output = {}
        for relation in self.relationships:
            if relation.type != "caption_of":
                continue
            source = self.by_id.get(relation.source_id)
            signature = caption_signature(source.text) if source else None
            if signature and signature[1]:
                output[signature] = relation.target_id
        return output

    @staticmethod
    def _numeric_conflicts(left: str, right: str, matches: list[dict[str, object]]) -> list[dict[str, Any]]:
        matched = {(str(item.get("source_raw")), str(item.get("target_raw"))) for item in matches}
        conflicts = []
        for source in numeric_tokens(left):
            source_terms = content_terms(left[max(0, source.start - 60):source.start])
            for target in numeric_tokens(right):
                if (source.raw, target.raw) in matched or source.dimension != target.dimension or \
                        source.dimension in {"scalar"} | NON_VALUE_DIMENSIONS or source.value == target.value:
                    continue
                target_terms = content_terms(right[max(0, target.start - 60):target.start])
                shared = sorted(source_terms & target_terms)
                if len(shared) < 2:
                    continue
                conflicts.append({
                    "source_raw": source.raw, "source_offsets": [source.start, source.end],
                    "source_normalized": source.value, "target_raw": target.raw,
                    "target_offsets": [target.start, target.end], "target_normalized": target.value,
                    "dimension": source.dimension, "shared_metric_terms": shared,
                })
        return conflicts

    def _named_visual_reference(self, paragraph: ElementContext, visual: ElementContext) -> dict[str, Any] | None:
        visual_record = self.visuals[visual.element_id]
        sources = []
        title = visual_record.get("internal_title") or {}
        if title.get("text"):
            sources.append(("internal_title", str(title["text"]), title.get("node_id")))
        caption = visual_record.get("caption") or {}
        if caption.get("text"):
            sources.append(("explicit_caption", str(caption["text"]), caption.get("source_id")))
        paragraph_normal = normalize_text(paragraph.text)
        contextual_cue = bool(re.search(
            r"(?i)\b(?:figure|fig|chart|plot|table|shows|depicts|illustrates)\b|"
            r"(?:الشكل|الرسم|المخطط|الجدول|يوضح|يبين)", paragraph.text
        ))
        best = None
        rejected = None
        for source_kind, text, source_id in sources:
            reference_normal = normalize_text(text)
            terms = content_terms(text)
            if len(terms) < 2:
                continue
            distinctive_terms = terms - GENERIC_VISUAL_TERMS
            exact = reference_normal in paragraph_normal
            paragraph_terms = content_terms(paragraph.text)
            shared = sorted(terms & paragraph_terms)
            coverage = len(shared) / max(1, len(terms))
            partial_score = 0.68 + min(0.20, coverage / 5)
            accepted = bool(distinctive_terms) and (exact or (
                contextual_cue and len(shared) >= 2 and coverage >= 0.67
                                 and partial_score >= NAMED_REFERENCE_SIMILARITY_THRESHOLD)
            )
            if len(shared) >= 2 and not accepted:
                reason = ("title contains only generic topic terms" if not distinctive_terms else
                          "partial title terms lack an explicit visual cue or sufficient coverage")
                candidate = {"accepted": False, "source": source_kind, "source_id": source_id,
                             "original_reference": text, "normalized_reference": reference_normal,
                             "shared_terms": shared, "title_term_coverage": round(coverage, 4),
                             "contextual_visual_cue": contextual_cue, "reason": reason}
                if rejected is None or coverage > rejected[0]:
                    rejected = coverage, candidate
            if not accepted:
                continue
            candidate = {
                "accepted": True,
                "source": source_kind, "source_id": source_id, "original_reference": text,
                "normalized_reference": reference_normal, "exact_normalized_phrase": exact,
                "shared_distinctive_terms": shared, "title_term_coverage": round(coverage, 4),
                "distinctive_title_terms": sorted(distinctive_terms),
                "contextual_visual_cue": contextual_cue,
            }
            score = 0.92 if exact else partial_score
            if best is None or score > best[0]:
                best = score, candidate
        if best:
            return {"score": best[0], **best[1]}
        return rejected[1] if rejected else None

    def build_descriptions(self) -> None:
        caption_targets = self._caption_targets()
        paragraphs = [item for item in self.elements if item.role in TEXT_ROLES and len(item.text.strip()) >= 12]
        for paragraph in paragraphs:
            candidates: dict[str, dict[str, Any]] = {}
            for reference in EXPLICIT_REF_RE.finditer(paragraph.text):
                kind = "table" if "table" in reference.group("kind").casefold() or "جدول" in reference.group("kind") else "figure"
                number = (reference.group("number") or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
                target = caption_targets.get((kind, number)) if number and not number.startswith("التالي") else None
                if target:
                    candidates.setdefault(target, {"explicit_references": [], "named_references": [],
                                                   "numeric_matches": [], "numeric_conflicts": []})[
                        "explicit_references"
                    ].append({"verbatim": reference.group(0), "offsets": [reference.start(), reference.end()]})
                elif number.startswith("التالي"):
                    nearby = [item for item in self.elements if item.element_type == kind and item.section_id == paragraph.section_id]
                    if len(nearby) == 1:
                        candidates.setdefault(nearby[0].element_id, {
                            "explicit_references": [], "named_references": [], "numeric_matches": [],
                            "numeric_conflicts": [],
                        })[
                            "explicit_references"
                        ].append({"verbatim": reference.group(0), "offsets": [reference.start(), reference.end()],
                                  "relative_reference": True})
            for visual in self.elements:
                if visual.element_type not in VISUAL_TYPES or not visual.text:
                    continue
                page_distance = abs(visual.page - paragraph.page)
                same_section = bool(visual.section_id and visual.section_id == paragraph.section_id)
                if page_distance > 1 or (not same_section and visual.page != paragraph.page):
                    continue
                named = self._named_visual_reference(paragraph, visual)
                if named and named.get("accepted"):
                    candidates.setdefault(visual.element_id, {
                        "explicit_references": [], "named_references": [], "numeric_matches": [],
                        "numeric_conflicts": [],
                    })["named_references"].append(named)
                elif named:
                    self.evidence_audit.append({
                        "candidate_type": "describes", "status": "rejected", "source_id": paragraph.element_id,
                        "target_id": visual.element_id, "page": paragraph.page,
                        "section_id": paragraph.section_id, "rule": "named_reference",
                        "evidence": named, "reason": named["reason"],
                    })
                matches = match_numeric_tokens(paragraph.text, visual.text)
                conflicts = self._numeric_conflicts(paragraph.text, visual.text, matches)
                conclusive = _conclusive_matches(matches)
                if len(conclusive) >= 2:
                    record = candidates.setdefault(visual.element_id, {
                        "explicit_references": [], "named_references": [], "numeric_matches": [],
                        "numeric_conflicts": [],
                    })
                    record["numeric_matches"] = matches
                    record["numeric_conflicts"] = conflicts
                    record["conclusive_numeric_match_count"] = len(conclusive)
                    record["inconclusive_numeric_match_count"] = len(matches) - len(conclusive)
                elif matches:
                    inconclusive = len(matches) - len(conclusive)
                    self.evidence_audit.append({
                        "candidate_type": "describes", "status": "rejected", "source_id": paragraph.element_id,
                        "target_id": visual.element_id, "page": paragraph.page,
                        "section_id": paragraph.section_id, "rule": "numeric_only",
                        "numeric_matches": matches,
                        "reason": "fewer than two independent matching values"
                                  + (f" that are conclusive ({inconclusive} inconclusive match(es) are recorded but "
                                     "not counted)" if inconclusive else ""),
                    })
            scored = []
            for target_id, evidence in candidates.items():
                target = self.by_id[target_id]
                if evidence["numeric_conflicts"] and not evidence["explicit_references"] and not evidence["named_references"]:
                    self.evidence_audit.append({
                        "candidate_type": "describes", "status": "rejected", "source_id": paragraph.element_id,
                        "target_id": target_id, "page": paragraph.page, "section_id": paragraph.section_id,
                        "rule": "numeric_only", "numeric_matches": evidence["numeric_matches"],
                        "numeric_conflicts": evidence["numeric_conflicts"],
                        "reason": "conflicting aligned numerical evidence",
                    })
                    continue
                if evidence["explicit_references"]:
                    score, rule = 0.90, "explicit_figure_or_table_reference"
                elif evidence["named_references"]:
                    score = max(float(item["score"]) for item in evidence["named_references"])
                    rule = "explicit_or_contextual_named_reference"
                elif len(_conclusive_matches(evidence["numeric_matches"])) >= 2:
                    score = 0.60 + min(0.20, 0.05 * len(_conclusive_matches(evidence["numeric_matches"])))
                    rule = "two_or_more_compatible_numbers"
                else:
                    continue
                if target.section_id and target.section_id == paragraph.section_id:
                    score += 0.08
                if target.page == paragraph.page:
                    score += 0.04
                evidence.update({
                    "same_section": target.section_id == paragraph.section_id,
                    "page_distance": abs(target.page - paragraph.page),
                    "section_id": target.section_id or paragraph.section_id,
                    "source_page": paragraph.page, "target_page": target.page, "rule": rule,
                })
                scored.append((score, target, evidence))
            scored.sort(key=lambda item: (-item[0], abs(item[1].order - paragraph.order), item[1].element_id))
            if not scored:
                continue
            best = scored[0]
            competitors = [
                {"target_id": item[1].element_id, "score": round(item[0], 4), "evidence": item[2]}
                for item in scored[1:] if best[0] - item[0] <= 0.08
            ]
            # Without an explicit reference, equivalent support for another visual makes the target arbitrary:
            # numeric-only ties, and named-reference ties where a competitor is named just as strongly.
            named_tie = bool(best[2]["named_references"]) and any(
                item["evidence"].get("named_references") for item in competitors
            )
            if competitors and not best[2]["explicit_references"] and (
                    not best[2]["named_references"] or named_tie):
                self.findings.append({
                    "severity": "warning", "code": "ambiguous_description_target", "items": [paragraph.element_id],
                    "message": ("the paragraph names competing visuals with equivalent support" if named_tie else
                                "numeric evidence supports competing visuals")
                               + "; no describes relationship was emitted",
                    "conflicts": [{"target_id": best[1].element_id, "score": round(best[0], 4)}, *competitors],
                })
                self.evidence_audit.append({
                    "candidate_type": "describes", "status": "rejected", "source_id": paragraph.element_id,
                    "target_id": best[1].element_id, "page": paragraph.page,
                    "section_id": paragraph.section_id, "rule": best[2]["rule"],
                    "evidence": best[2], "reason": "competing visual candidates have equivalent support",
                })
                continue
            relation = self.add_relation(
                "describes", paragraph.element_id, best[1].element_id, best[2], best[0],
                "explicit_reference_or_numeric_agreement", conflicts=competitors, review=bool(competitors),
            )
            self.visuals[best[1].element_id]["description_paragraphs"].append({
                "source_id": paragraph.element_id, "relationship_id": relation.rel_id,
                "confidence": relation.confidence,
            })
            self.visuals[best[1].element_id]["matched_numeric_evidence"].extend(best[2]["numeric_matches"])
            self.evidence_audit.append({
                "candidate_type": "describes", "status": "accepted", "source_id": paragraph.element_id,
                "target_id": best[1].element_id, "page": paragraph.page,
                "target_page": best[1].page, "section_id": paragraph.section_id,
                "rule": best[2]["rule"], "confidence": relation.confidence,
                "evidence": best[2], "reason": "candidate met the deterministic evidence rule",
            })

        described = {relation.target_id for relation in self.relationships if relation.type == "describes"}
        weak_targets: set[str] = set()
        weak_visuals = [item for item in self.elements if item.element_type in VISUAL_TYPES
                        and item.element_id not in described and item.section_id]
        weak_visuals.sort(key=lambda item: (
            not bool((self.visuals.get(item.element_id, {}).get("composite") or {}).get("parent")), item.order,
        ))
        for visual in weak_visuals:
            target = visual
            redirected_from = None
            parent_id = (self.visuals.get(visual.element_id, {}).get("composite") or {}).get("parent")
            # Redirect only a part whose content is carried by its parent (included or textless). A text-bearing
            # visual that Nass split out of an outer region keeps its own weak link.
            subsumed = visual.validation_state == "included_in_composite" or not visual.text.strip()
            if parent_id and subsumed:
                parent = self.by_id.get(str(parent_id)) if parent_id else None
                if parent is None:
                    self.evidence_audit.append({
                        "candidate_type": "co_located_in_section", "status": "rejected",
                        "target_id": visual.element_id, "page": visual.page,
                        "reason": "included composite part has no resolved composite parent",
                    })
                    continue
                target = parent
                redirected_from = visual.element_id
            if target.element_id in described or target.element_id in weak_targets:
                continue
            proximity_source = visual if redirected_from else target
            nearby = [item for item in paragraphs if item.section_id == target.section_id
                      and item.page == target.page and abs(item.order - proximity_source.order) <= 2]
            if not nearby:
                continue
            paragraph = min(nearby, key=lambda item: (abs(item.order - proximity_source.order), item.element_id))
            relation = self.add_relation(
                "co_located_in_section", paragraph.element_id, target.element_id,
                {"section_id": target.section_id, "page": target.page,
                 "reading_order_distance": abs(paragraph.order - proximity_source.order),
                 "redirected_from_composite_part": redirected_from,
                 "semantic_assertion": "none; this relation records weak structural proximity only"},
                0.20, "same_section_same_page_adjacency", cap="low", review=True,
                limitations=["co-location does not establish explanation, support, or factual agreement"],
            )
            self.evidence_audit.append({
                "candidate_type": "co_located_in_section", "status": "accepted",
                "source_id": paragraph.element_id, "target_id": target.element_id,
                "page": target.page, "section_id": target.section_id,
                "redirected_from_composite_part": redirected_from,
                "relationship_id": relation.rel_id,
                "reason": "weak proximity relation targets the composite parent when applicable",
            })
            weak_targets.add(target.element_id)

    @staticmethod
    def _table_shape(table: ElementContext) -> tuple[int | None, tuple[str, ...]]:
        markdown_rows = [line.strip() for line in table.text.splitlines()
                         if line.strip().startswith("|") and line.strip().endswith("|")]
        data_rows = [line for line in markdown_rows if not re.fullmatch(r"[|:\-–—\s]+", line)]
        if data_rows:
            header = tuple(normalize_text(value) for value in data_rows[0].strip("|").split("|"))
            return len(header), header
        cells = table.raw.get("cells")
        if isinstance(cells, list) and cells:
            rows = [row for row in cells if isinstance(row, list)]
            if rows:
                return max(len(row) for row in rows), tuple(normalize_text(str(value)) for value in rows[0])
            positioned = [cell for cell in cells if isinstance(cell, dict)
                          and isinstance(cell.get("row"), int) and isinstance(cell.get("col"), int)]
            if positioned:
                first_row = min(cell["row"] for cell in positioned)
                header_cells = sorted((cell for cell in positioned if cell["row"] == first_row),
                                      key=lambda cell: cell["col"])
                columns = max(cell["col"] for cell in positioned) + 1
                return columns, tuple(normalize_text(str(cell.get("text") or "")) for cell in header_cells)
        lines = [line.strip() for line in table.text.splitlines() if line.strip()]
        if lines and any("\t" in line or "|" in line for line in lines):
            separator = "\t" if "\t" in lines[0] else "|"
            header = tuple(normalize_text(value) for value in lines[0].split(separator))
            return len(header), header
        return None, ()

    def build_table_continuations(self) -> None:
        tables = sorted((item for item in self.elements if item.element_type == "table"), key=lambda item: item.order)
        supported: list[tuple[ElementContext, ElementContext, float, dict[str, Any]]] = []
        for left, right in pairwise(tables):
            if right.page != left.page + 1:
                continue
            left_cols, left_header = self._table_shape(left)
            right_cols, right_header = self._table_shape(right)
            source_marker = any(
                "continu" in str(value).casefold() or "تابع" in str(value)
                for value in (left.raw.get("reasons"), right.raw.get("reasons"), left.raw.get("provenance"),
                              right.raw.get("provenance"))
            )
            repeated_header = bool(left_header and left_header == right_header)
            column_match = left_cols is not None and left_cols == right_cols
            same_section = left.section_id is not None and left.section_id == right.section_id
            score = 0.22 + (0.38 if source_marker else 0) + (0.22 if repeated_header else 0) + \
                (0.12 if column_match else 0) + (0.06 if same_section else 0)
            evidence = {
                "consecutive_pages": True, "source_continuation_marker": source_marker,
                "repeated_header": repeated_header, "matching_column_count": column_match,
                "column_counts": [left_cols, right_cols], "same_section": same_section,
            }
            if score < 0.62:
                if repeated_header or column_match or source_marker:
                    self.findings.append({
                        "severity": "warning", "code": "possible_table_continuation",
                        "items": [left.element_id, right.element_id],
                        "message": "adjacent tables share some continuation evidence but remain separate",
                        "evidence": evidence,
                    })
                continue
            supported.append((left, right, score, evidence))
        chains: list[tuple[list[ElementContext], list[tuple[float, dict[str, Any]]]]] = []
        for left, right, score, evidence in supported:
            if chains and chains[-1][0][-1].element_id == left.element_id:
                chains[-1][0].append(right)
                chains[-1][1].append((score, evidence))
            else:
                chains.append(([left, right], [(score, evidence)]))
        for parts, links in chains:
            part_ids = [part.element_id for part in parts]
            logical_id = _stable_id("logical_table", *part_ids)
            repeated = [evidence["repeated_header"] for _, evidence in links]
            self.add_node(RelationNode(
                node_id=logical_id, node_type="logical_table", page=parts[0].page, text=None, source_id=None,
                provenance={"ordered_parts": part_ids, "repeated_header_between_parts": repeated,
                            "pair_evidence": [evidence for _, evidence in links]},
            ))
            score = min(item[0] for item in links)
            for index, table in enumerate(parts):
                adjacent = []
                if index > 0:
                    adjacent.append(links[index - 1][1])
                if index < len(links):
                    adjacent.append(links[index][1])
                self.add_relation(
                    "table_continuation", table.element_id, logical_id,
                    {"part_index": index, "part_count": len(parts), "adjacent_pair_evidence": adjacent,
                     "header_repeated_before": repeated[index - 1] if index > 0 else False}, score,
                    "saved_marker+headers+columns+page_adjacency",
                )
                self.visuals[table.element_id]["logical_table_id"] = logical_id

    def build_composites(self) -> None:
        linked_parts: set[str] = set()

        def link(part: ElementContext, parent_id: str, method: str, evidence: dict[str, Any], score: float) -> None:
            if part.element_id in linked_parts or parent_id == part.element_id:
                return
            relation = self.add_relation("composite_part", part.element_id, parent_id, evidence, score, method)
            self.visuals[part.element_id]["composite"]["parent"] = parent_id
            self.visuals[parent_id]["composite"]["parts"].append({
                "source_id": part.element_id, "relationship_id": relation.rel_id,
            })
            linked_parts.add(part.element_id)

        region_elements = self._element_by_region_id()
        blocks = self._layout_regions_by_block()
        by_block: dict[tuple[int, int], ElementContext] = {}
        for key, region in blocks.items():
            element = next((region_elements.get(str(value)) for value in (
                region.get("job_id"), region.get("region_id"),
            ) if value and str(value) in region_elements), None)
            if element:
                by_block[key] = element

        # Nass records containment decisions even when a skipped/review region remains a first-class source element.
        # Reuse those decisions; do not infer that empty text means failed OCR.
        for part in self.elements:
            if part.element_type not in VISUAL_TYPES:
                continue
            decisions = self._provenance(part).get("decisions")
            if isinstance(decisions, dict):
                decisions = [decisions]
            for decision in decisions or []:
                if not isinstance(decision, dict) or decision.get("type") != "containment":
                    continue
                parent_block = decision.get("contained_in")
                if isinstance(parent_block, int) and "contained" in str(decision.get("decision") or ""):
                    parent = by_block.get((part.page, parent_block))
                    if parent:
                        link(part, parent.element_id, "saved_layout_containment_decision", {
                            "validation_state": part.validation_state,
                            "saved_decision": decision,
                            "parent_block_index": parent_block,
                        }, 0.92)
                if decision.get("decision") == "split into inner visuals":
                    for inner_block in decision.get("inner") or []:
                        child = by_block.get((part.page, inner_block)) if isinstance(inner_block, int) else None
                        if child:
                            link(child, part.element_id, "saved_layout_split_decision", {
                                "validation_state": child.validation_state,
                                "saved_decision": decision,
                                "child_block_index": inner_block,
                            }, 0.92)

        for part in (item for item in self.elements if item.validation_state == "included_in_composite"):
            if part.element_id in linked_parts:
                continue
            provenance = part.raw.get("provenance") if isinstance(part.raw.get("provenance"), dict) else {}
            parent_id = next((
                str(value) for value in (
                    part.raw.get("composite_parent_id"), provenance.get("composite_parent_id"),
                    provenance.get("parent_element_id"),
                ) if value and str(value) in self.visuals
            ), None)
            method = "source_provenance"
            if not parent_id and part.bbox:
                containers = []
                for candidate in self.elements:
                    if candidate.element_type not in VISUAL_TYPES or candidate.element_id == part.element_id or \
                            candidate.page != part.page or not candidate.bbox:
                        continue
                    inside = part.bbox[0] >= candidate.bbox[0] and part.bbox[1] >= candidate.bbox[1] and \
                        part.bbox[2] <= candidate.bbox[2] and part.bbox[3] <= candidate.bbox[3]
                    if inside:
                        area = (candidate.bbox[2] - candidate.bbox[0]) * (candidate.bbox[3] - candidate.bbox[1])
                        containers.append((area, candidate.element_id))
                if len(containers) == 1:
                    parent_id, method = containers[0][1], "unambiguous_layout_containment"
            if parent_id:
                link(part, parent_id, method, {"validation_state": part.validation_state, "basis": method},
                     0.92 if method == "source_provenance" else 0.78)
            else:
                self.visuals[part.element_id]["composite"]["unresolved"] = True
                self.visuals[part.element_id]["review_flags"].append("composite_parent_unresolved")
                self.findings.append(self._finding(
                    "warning", "unresolved_composite_parent", part.element_id,
                    "source is included_in_composite but no provenance or unique containment identifies its parent",
                ))

    def build_asides(self) -> None:
        asides = [item for item in self.elements if item.role in {"aside_label", "aside_text"}]
        for aside in asides:
            candidates = []
            for target in self.elements:
                if target.element_id == aside.element_id or target.page != aside.page or target.role in {
                    "aside_label", "aside_text"
                }:
                    continue
                if target.element_type not in VISUAL_TYPES and target.role != "paragraph":
                    continue
                if self._page_level_geometry(aside) or self._page_level_geometry(target):
                    continue  # a page container box says nothing about where a sentence sits
                distance, overlap = _bbox_distance(aside.bbox, target.bbox)
                if distance is None or overlap is None:
                    continue
                horizontal_gap = max(0.0, max(aside.bbox[0], target.bbox[0]) - min(aside.bbox[2], target.bbox[2]))
                vertical_overlap = max(0.0, min(aside.bbox[3], target.bbox[3]) - max(aside.bbox[1], target.bbox[1]))
                aside_height = max(1.0, aside.bbox[3] - aside.bbox[1])
                shared = vertical_overlap / aside_height
                score = 0.42 + 0.28 * min(1, shared) + max(0, 0.20 - horizontal_gap / 500)
                if aside.section_id and aside.section_id == target.section_id:
                    score += 0.08
                candidates.append((score, target, horizontal_gap, shared))
            candidates.sort(key=lambda item: (-item[0], item[2], item[1].element_id))
            if not candidates or candidates[0][0] < 0.62:
                self.findings.append(self._finding("warning", "unlinked_aside", aside.element_id,
                                                   "aside has no geometry-supported nearby target"))
                continue
            top = candidates[0]
            conflicts = [
                {"target_id": item[1].element_id, "score": round(item[0], 4)}
                for item in candidates[1:] if top[0] - item[0] <= 0.08
            ]
            self.add_relation(
                "aside_of", aside.element_id, top[1].element_id,
                {"horizontal_gap_pt": round(top[2], 3), "vertical_overlap_ratio": round(top[3], 4),
                 "same_section": aside.section_id == top[1].section_id},
                top[0], "geometry+section", conflicts=conflicts, review=bool(conflicts),
            )

    def _derived_list_spans(self) -> None:
        for element in self.elements:
            markers = list_markers(element.text)
            if len(markers) < 2:
                continue
            leading = [marker for marker in markers if marker["position"] == "leading"]
            trailing = [marker for marker in markers if marker["position"] == "trailing"]
            if len(leading) < 2 and len(trailing) < 2:
                continue
            anchors = leading if len(leading) >= 2 else trailing
            for index, marker in enumerate(anchors):
                if marker["position"] == "leading":
                    start = int(marker["start"])
                    end = int(anchors[index + 1]["start"]) if index + 1 < len(anchors) else len(element.text)
                else:
                    previous_end = int(anchors[index - 1]["end"]) if index else 0
                    start = previous_end
                    end = int(marker["end"])
                while start < end and element.text[start].isspace():
                    start += 1
                while end > start and element.text[end - 1].isspace():
                    end -= 1
                if end - start < 3:
                    continue
                substring = element.text[start:end]
                span_id = _stable_id("list_span", element.element_id, start, end, substring)
                self.add_node(RelationNode(
                    node_id=span_id, node_type="derived_list_span", page=element.page, text=substring,
                    source_id=element.element_id, start=start, end=end,
                    provenance={"marker": marker["marker"], "marker_position": marker["position"],
                                "verbatim": True},
                ))
                record = {
                    "span_id": span_id, "parent_element_id": element.element_id, "start": start, "end": end,
                    "text": substring, "page": element.page, "marker": marker["marker"],
                    "marker_position": marker["position"], "confidence": "high",
                }
                self.list_output["derived_spans"].append(record)
                relation = self.add_relation(
                    "merged_list_items", span_id, element.element_id,
                    {"offsets": [start, end], "marker": marker["marker"],
                     "marker_position": marker["position"], "verbatim": substring},
                    0.90, "multiple_supported_markers_with_exact_offsets",
                )
                self.list_output["relationships"].append(relation.rel_id)

    def build_lists(self) -> None:
        self._derived_list_spans()
        originals = self.inputs.section_tree.get("lists") or []
        self.list_output["original_lists"] = originals
        for original in originals:
            items = original.get("items") or []
            list_id = str(original.get("list_id"))
            intro_unit_id = original.get("intro_unit_id")
            if intro_unit_id:
                intro_assignment = self.assignment_by_unit.get(str(intro_unit_id), {})
                intro_source = str(intro_unit_id) if str(intro_unit_id) in self.nodes else str(
                    intro_assignment.get("element_id") or ""
                )
                if intro_source in self.nodes and list_id in self.nodes:
                    relation = self.add_relation(
                        "introduces_list", intro_source, list_id,
                        {"saved_structure_intro_unit_id": intro_unit_id}, 0.88,
                        "saved_document_structure_list_intro",
                    )
                    self.list_output["relationships"].append(relation.rel_id)
            tree_nodes = []
            last_at_level: dict[int, str] = {}
            marker_at_level: dict[int, str] = {}
            previous = None
            for index, item in enumerate(items):
                unit_id = str(item.get("unit_id"))
                assignment = self.assignment_by_unit.get(unit_id, {})
                element_id = str(item.get("element_id") or assignment.get("element_id") or unit_id)
                text = str(assignment.get("text") or self.by_id.get(element_id, ElementContext(
                    "", 0, 0, "", "", None, None, {}, None, None
                )).text)
                marker = str(item.get("marker") or "")
                markers = list_markers(text)
                indent = int(markers[0]["indent"]) if markers else 0
                family = marker_family(marker) if marker else "unknown"
                context = self.by_id.get(element_id)
                level = 0
                evidence = ["saved_structure_list"]
                if previous:
                    page_direction = str(self.inputs.run.pages.get(int(item.get("page") or 0), {})
                                         .get("direction") or "").casefold()
                    geometry_indents = None
                    if context and context.bbox and previous["bbox"] and context.page == previous["page"]:
                        if page_direction == "rtl":
                            geometry_indents = context.bbox[2] < previous["bbox"][2] - 8
                        else:
                            geometry_indents = context.bbox[0] > previous["bbox"][0] + 8
                    ancestor_level = next((candidate_level for candidate_level, ancestor_marker
                                           in sorted(marker_at_level.items())
                                           if candidate_level < previous["level"] and marker == ancestor_marker), None)
                    if ancestor_level is not None:
                        level = ancestor_level
                        evidence.append("marker_matches_ancestor_level")
                    elif indent > previous["indent"]:
                        level = previous["level"] + 1
                        evidence.append("greater_indentation")
                    elif marker != previous["marker"] and previous["text"].rstrip().endswith((':', '：')) and \
                            geometry_indents is not False:
                        level = previous["level"] + 1
                        evidence.extend(["marker_change", "parent_ends_colon"])
                        evidence.append("geometry_indents_child" if geometry_indents else "geometry_unavailable")
                    else:
                        level = previous["level"]
                parent_id = last_at_level.get(level - 1) if level > 0 else None
                node_id = unit_id if unit_id in self.nodes else element_id
                if node_id not in self.nodes:
                    continue
                tree_node = {
                    "node_id": node_id, "element_id": element_id, "page": item.get("page"), "order": index,
                    "marker": marker, "marker_family": family, "indent": indent, "level": level,
                    "parent_id": parent_id, "text": text, "evidence": evidence,
                }
                tree_nodes.append(tree_node)
                if parent_id:
                    relation = self.add_relation(
                        "list_nesting", node_id, parent_id,
                        {"list_id": list_id, "level": level, "marker": marker, "basis": evidence},
                        0.80 if "greater_indentation" in evidence else 0.68,
                        "saved_list+indentation+context",
                    )
                    self.list_output["relationships"].append(relation.rel_id)
                last_at_level[level] = node_id
                marker_at_level[level] = marker
                for deeper in [value for value in last_at_level if value > level]:
                    del last_at_level[deeper]
                    marker_at_level.pop(deeper, None)
                previous = {"indent": indent, "family": family, "marker": marker, "level": level, "text": text,
                            "bbox": context.bbox if context else None, "page": context.page if context else None}
            self.list_output["nodes"].append({
                "list_id": list_id, "section_id": original.get("section_id"),
                "intro_unit_id": intro_unit_id, "items": tree_nodes,
                "pages": original.get("pages") or [], "crosses_pages": bool(original.get("crosses_pages")),
                "possible_continuations": original.get("possible_continuations") or [],
            })

    def build_metadata(self) -> None:
        page_count = self.inputs.run.page_count
        document_id = f"document:{self.inputs.run.run_id}"
        for element in self.elements:
            match = METADATA_RE.search(element.text)
            near_end = element.page >= max(1, page_count - 1)
            footer = bool(element.bbox and element.bbox[1] >= 0.80 * max(element.bbox[3], 792))
            if not match or (not near_end and not footer and element.role != "footer"):
                continue
            start, end = match.span()
            while start < end and element.text[start].isspace():
                start += 1
            text = element.text[start:end]
            node_id = _stable_id("metadata", element.element_id, start, end, text)
            self.add_node(RelationNode(
                node_id=node_id, node_type="document_metadata_annotation", page=element.page, text=text,
                source_id=element.element_id, start=start, end=end,
                provenance={"verbatim": True, "near_document_end": near_end, "footer_geometry": footer},
            ))
            self.add_relation(
                "document_metadata", node_id, document_id,
                {"offsets": [start, end], "verbatim": text, "near_document_end": near_end,
                 "footer_geometry": footer, "saved_role": element.role},
                0.74 + (0.10 if near_end else 0) + (0.08 if footer or element.role == "footer" else 0),
                "anchored_label+structural_context",
            )

    def build_known_missing(self) -> None:
        document_id = f"document:{self.inputs.run.run_id}"
        sources: list[tuple[str, dict[str, Any]]] = []

        def collect(origin: str, value: Any) -> None:
            if isinstance(value, dict):
                status = str(value.get("status") or value.get("action") or "").casefold()
                description = " ".join(str(value.get(key) or "") for key in (
                    "code", "kind", "type", "label", "message", "reason",
                )).casefold()
                visual_record = any(term in description for term in (
                    "visual", "image", "figure", "chart", "صورة", "شكل", "رسم",
                ))
                explicit_skip = status in {"skipped", "dropped", "unavailable"} or any(
                    term in description for term in (
                        "missing_visual", "missing visual", "skipped_visual", "skipped visual",
                        "visual skipped", "image skipped", "dropped image", "dropped visual",
                    )
                )
                if visual_record and explicit_skip:
                    sources.append((origin, value))
                for key, child in value.items():
                    collect(f"{origin}.{key}", child)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect(f"{origin}[{index}]", child)

        for flag in (self.inputs.run.validation_report or {}).get("flags") or []:
            if isinstance(flag, dict):
                sources.append(("validation_report", flag))
        manifest = self.inputs.run.execution_manifest or {}
        for key in ("missing_visuals", "skipped_visuals"):
            for item in manifest.get(key) or []:
                if isinstance(item, dict):
                    sources.append((f"execution_manifest.{key}", item))
        collect("execution_manifest", self.inputs.run.execution_manifest or {})
        collect("layout", self.inputs.run.layout or {})
        unique_sources = {}
        for origin, record in sources:
            key = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
            unique_sources.setdefault(key, (origin, record))
        for origin, record in unique_sources.values():
            code = str(record.get("code") or "").casefold()
            message = str(record.get("message") or record.get("reason") or "")
            status = str(record.get("status") or record.get("action") or "").casefold()
            kind = " ".join(str(record.get(key) or "") for key in ("kind", "type", "label")).casefold()
            explicit_status_record = status in {"skipped", "dropped", "unavailable"} and any(
                token in kind for token in ("visual", "image", "figure", "chart", "صورة", "شكل", "رسم")
            )
            if not explicit_status_record and not any(token in f"{code} {message.casefold()}" for token in (
                "missing_visual", "missing visual", "visual_missing", "skipped_visual", "skipped visual",
                "visual skipped", "image skipped", "dropped image", "dropped visual",
            )):
                continue
            page = record.get("page") if isinstance(record.get("page"), int) else None
            node_id = _stable_id("known_missing", origin, page, record)
            if node_id in self.nodes:
                continue
            bbox = record.get("bbox_pt") or record.get("bbox")
            bbox = bbox if isinstance(bbox, list) and len(bbox) == 4 else None
            self.add_node(RelationNode(
                node_id=node_id, node_type="known_missing_visual", page=page, text=None, source_id=None,
                start=None, end=None,
                provenance={"source": origin, "record": record, "bbox_pt": bbox},
                review_flags=("missing_source_content",),
            ))
            relation = self.add_relation(
                "known_missing", document_id, node_id,
                {"page": page, "reason": message, "source": origin,
                 "layout_region": record.get("region_id") or bbox, "source_record": record},
                0.88, "explicit_saved_metadata", review=True,
                limitations=["placeholder records saved evidence only; no missing text or visual is reconstructed"],
            )
            self.visuals[node_id] = {
                "visual_id": node_id, "visual_type": "known_missing", "page": page,
                "bbox_pt": bbox, "section_id": None, "validation_state": None,
                "reason": message or "upstream-recorded skip/drop", "source_record": record,
                "source_record_path": origin, "relationship_ids": [relation.rel_id],
                "caption": None, "internal_title": None, "description_paragraphs": [],
                "matched_numeric_evidence": [], "leaked_text_overlaps": [],
                "composite": {"parent": None, "parts": [], "unresolved": False},
                "logical_table_id": None, "review_flags": ["known_missing"],
                "rag_flags": ["known_missing"], "evidence_reliability": "low",
                "evidence_reliability_reasons": ["upstream explicitly recorded skipped or dropped visual content"],
            }

    @staticmethod
    def _finding(severity: str, code: str, item: str, message: str) -> dict[str, Any]:
        return {"severity": severity, "code": code, "items": [item], "message": message}

    def build(self) -> RelationResult:
        self.build_captions()
        self.report_margin_leakage()
        self.build_internal_titles()
        self.build_leakage()
        self.build_composites()
        self.build_descriptions()
        self.build_table_continuations()
        self.build_asides()
        self.build_lists()
        self.build_metadata()
        self.build_known_missing()
        self.relationships.sort(key=lambda item: (item.type, item.source_page or 0, item.source_id, item.target_id))
        return RelationResult(
            self.inputs.run.run_id, str(self.inputs.run.run_dir), str(self.inputs.structure_dir), self.nodes,
            self.relationships, sorted(self.visuals.values(), key=lambda item: (item.get("page") or 0,
                                                                                item["visual_id"])),
            self.list_output, self.evidence_audit, self.findings, self.warnings,
            self.inputs.source_hashes, self.inputs.structure_hashes,
            hash_existing(self.inputs.run.run_dir, SOURCE_FILES), hash_existing(self.inputs.structure_dir, (
                "section_tree.json", "assignments.json", "manifest.json",
            )),
        )


def build_relationships(inputs: RelationshipInputs) -> RelationResult:
    """Build relationships without writing files or reading the source PDF."""
    return Builder(inputs).build()

