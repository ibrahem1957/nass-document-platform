"""Diagnostics for saved PP-DocLayout heading regions (doc_title, paragraph_title, ...) on scanned pages.

For every heading region the report shows which extracted text it corresponds to, how that association was
established, and whether it became a section heading. Three association levels:

- verified: the layout block and the text are linked one to one (an element whose own box lies in the region,
  or a page-OCR segment made from this single layout block);
- approximate: the region is part of a multi-block segment and a derived span was matched to it by order and
  role, or it is the only heading-like span of that segment;
- none: no reliable link (explicit reason given). An unmatched layout title is never treated as a heading.

No coordinates are invented for OCR-derived spans: only geometry recorded by segmentation is reported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .config import Config
from .layout import HEADING_LABELS, LayoutBox, role_of
from .model import HeadingCandidate, Unit


@dataclass
class HeadingRegionDiagnostic:
    page: int
    layout_label: str
    layout_bbox_pt: list[float]
    layout_score: float | None
    block_index: int | None
    association: str                 # verified | approximate | none
    match_confidence: str | None     # high | medium | low | None
    element_id: str | None
    span_id: str | None
    unit_id: str | None
    extracted_text: str | None
    evidence: list[str]
    reason_unmatched: str | None
    became_heading: bool
    section_id: str | None
    candidate_status: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _containment(a, b) -> float:
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    return inter / max(1e-9, min(_area(a), _area(b)))


def layout_heading_diagnostics(units: list[Unit], layout_boxes: dict[int, list[LayoutBox]],
                               candidates: list[HeadingCandidate], section_of_unit: dict[str, str],
                               cfg: Config) -> list[HeadingRegionDiagnostic]:
    candidate_of = {uid: c for c in candidates for uid in c.unit_ids}
    out: list[HeadingRegionDiagnostic] = []
    for page, boxes in sorted(layout_boxes.items()):
        page_units = [u for u in units if u.page == page]
        for box in boxes:
            if box.label not in HEADING_LABELS:
                continue
            result = _associate(box, page_units, cfg)
            unit = result.pop("unit")
            candidate = candidate_of.get(unit.unit_id) if unit is not None else None
            became = candidate is not None and candidate.status == "promoted"
            out.append(HeadingRegionDiagnostic(
                page=page, layout_label=box.label, layout_bbox_pt=[round(v, 2) for v in box.bbox],
                layout_score=box.score, block_index=box.block_index,
                element_id=unit.element_id if unit is not None else None,
                span_id=unit.span_id if unit is not None else None, unit_id=unit.unit_id if unit is not None else None,
                extracted_text=unit.text if unit is not None else None, became_heading=became,
                section_id=section_of_unit.get(unit.unit_id) if (unit is not None and became) else None,
                candidate_status=candidate.status if candidate is not None else None, **result))
    return out


def _associate(box: LayoutBox, page_units: list[Unit], cfg: Config) -> dict[str, Any]:
    none = {"association": "none", "match_confidence": None, "evidence": [], "unit": None}
    # 1. A whole element whose own (non page-wide) box lies in the region: verified by geometry.
    best, best_c = None, 0.0
    for unit in page_units:
        if unit.span is not None or unit.bbox is None or not unit.clean:
            continue
        if unit.element_type == "page_text":
            continue  # page-wide OCR text has no paragraph geometry
        c = _containment(unit.bbox, box.bbox)
        if c > best_c:
            best, best_c = unit, c
    if best is not None and best_c >= cfg.layout.min_overlap and best.element_type not in ("figure", "table"):
        level = "verified" if best_c >= 0.8 else "approximate"
        return {"association": level, "match_confidence": "high" if level == "verified" else "medium",
                "evidence": [f"element {best.element_id} box lies in the region (containment {best_c:.2f})"],
                "reason_unmatched": None, "unit": best}
    # 2. A page-OCR segment that Nass built from this layout block (body_segment.text_blocks).
    seg_units = [u for u in page_units if u.span is not None and box.block_index is not None
                 and box.block_index in ((u.element.raw.get("body_segment") or {}).get("text_blocks") or [])]
    if seg_units:
        blocks = (seg_units[0].element.raw.get("body_segment") or {}).get("text_blocks") or []
        element_id = seg_units[0].element_id
        if len(blocks) == 1:
            heading_like = [u for u in seg_units if u.span.role_hint == "heading"]
            unit = heading_like[0] if len(heading_like) == 1 else seg_units[0]
            return {"association": "verified", "match_confidence": "high" if len(seg_units) == 1 else "medium",
                    "evidence": [(f"{element_id} is the page OCR of exactly this layout block (Nass body_segment); "
                                  f"{len(seg_units)} derived span(s); reported span {unit.span_id}")],
                    "reason_unmatched": None, "unit": unit}
        by_geometry = [u for u in seg_units if (u.span.geometry or {}).get("block_index") == box.block_index
                       and "bbox_pt" in (u.span.geometry or {})]
        if len(by_geometry) == 1:
            u = by_geometry[0]
            return {"association": "approximate", "match_confidence": "medium",
                    "evidence": [(f"span {u.span_id} matched to block {box.block_index} by order and role "
                                  f"({u.span.geometry.get('precision')})")], "reason_unmatched": None, "unit": u}
        heading_like = [u for u in seg_units if u.span.role_hint == "heading"]
        if len(heading_like) == 1:
            u = heading_like[0]
            return {"association": "approximate", "match_confidence": "low",
                    "evidence": [(f"segment {element_id} contains this {box.label} block and exactly one heading-like "
                                  f"span ({u.span_id}, {u.span.cut_reason}); position inside the text is not recorded")],
                    "reason_unmatched": None, "unit": u}
        why = (f"segment {element_id} contains this block among {len(blocks)} blocks, but "
               + ("no derived span looks like a heading" if not heading_like else
                  f"{len(heading_like)} spans look like headings") + "; the title cannot be located in the OCR text")
        return none | {"reason_unmatched": why, "evidence": [f"layout block {box.block_index} is in segment {element_id}"]}
    # 3. Not referenced by any extracted text.
    inside = next((u for u in page_units if u.span is None and u.bbox and u.element_type in ("figure", "table")
                   and _containment(box.bbox, u.bbox) >= 0.8), None)
    if inside is not None:
        return none | {"reason_unmatched": f"the region lies inside {inside.element_type} {inside.element_id}; its text "
                                           "(if any) is part of that crop's OCR"}
    referenced = any(box.block_index is not None and box.block_index in
                     ((u.element.raw.get("body_segment") or {}).get("text_blocks") or []) for u in page_units)
    if not referenced:
        return none | {"reason_unmatched": "no extracted element references this layout block (it may have been "
                                           "masked, skipped or merged into another region)"}
    return none | {"reason_unmatched": "no reliable association found"}


__all__ = ["HeadingRegionDiagnostic", "layout_heading_diagnostics", "role_of"]
