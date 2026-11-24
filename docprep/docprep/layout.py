"""Saved PP-DocLayout results (plan/layout.json). Read only; nothing is re-detected."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import LayoutConfig

HEADING_LABELS = {"doc_title": "doc_title", "title": "doc_title", "paragraph_title": "paragraph_title",
                  "section_title": "paragraph_title", "content_title": "paragraph_title",
                  "abstract_title": "paragraph_title"}
CAPTION_LABELS = {"figure_title", "table_title", "chart_title", "figure_caption", "table_caption", "image_caption",
                  "caption", "figure_table_chart_title"}
FURNITURE_LABELS = {"header", "footer", "number", "page_number", "header_image", "footer_image"}
BODY_LABELS = {"text", "abstract", "content", "paragraph", "reference", "reference_content", "list", "footnote",
               "aside_text", "vertical_text"}
VISUAL_LABELS = {"chart", "image", "table", "figure", "seal", "formula", "algorithm"}


def role_of(label: str | None) -> str:
    if label is None:
        return "unknown"
    if label in HEADING_LABELS:
        return "heading"
    if label in CAPTION_LABELS:
        return "caption"
    if label in FURNITURE_LABELS:
        return "furniture"
    if label in BODY_LABELS:
        return "body"
    if label in VISUAL_LABELS:
        return "visual"
    return "unknown"


@dataclass
class LayoutBox:
    page: int
    label: str
    score: float | None
    bbox: list[float]
    block_index: int | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayoutMatch:
    label: str
    role: str
    score: float | None
    containment: float
    block_index: int | None
    ambiguous: bool
    alternatives: list[dict[str, Any]]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def load_layout(data: dict[str, Any] | None) -> tuple[dict[int, list[LayoutBox]], dict[int, tuple[float, float]],
                                                       list[str]]:
    """Boxes per page in PDF points, page sizes from the saved page image metadata, and warnings."""
    boxes: dict[int, list[LayoutBox]] = {}
    sizes: dict[int, tuple[float, float]] = {}
    warnings: list[str] = []
    if not data:
        return boxes, sizes, warnings
    pages = data.get("pages")
    if not isinstance(pages, list):
        return boxes, sizes, ["plan/layout.json has no 'pages' list; layout signals unavailable"]
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("page"), int):
            continue
        number = page["page"]
        image = page.get("page_image") or {}
        px, bbox_pt = image.get("px"), image.get("bbox_pt")
        if isinstance(bbox_pt, list) and len(bbox_pt) == 4:
            sizes[number] = (float(bbox_pt[2]) - float(bbox_pt[0]), float(bbox_pt[3]) - float(bbox_pt[1]))
        raw_boxes = page.get("boxes")
        if not raw_boxes:
            continue
        if not (isinstance(px, list) and len(px) == 2 and isinstance(bbox_pt, list) and len(bbox_pt) == 4 and px[0]
                and px[1]):
            warnings.append(f"page {number}: layout boxes without page image geometry; ignored")
            continue
        sx = (float(bbox_pt[2]) - float(bbox_pt[0])) / float(px[0])
        sy = (float(bbox_pt[3]) - float(bbox_pt[1])) / float(px[1])
        out = []
        for raw in raw_boxes:
            if not isinstance(raw, dict) or not isinstance(raw.get("label"), str):
                continue
            b = raw.get("bbox_px")
            if not (isinstance(b, list) and len(b) == 4):
                continue
            pt = [float(bbox_pt[0]) + b[0] * sx, float(bbox_pt[1]) + b[1] * sy,
                  float(bbox_pt[0]) + b[2] * sx, float(bbox_pt[1]) + b[3] * sy]
            score = raw.get("score")
            out.append(LayoutBox(page=number, label=raw["label"],
                                 score=float(score) if isinstance(score, (int, float)) else None, bbox=pt,
                                 block_index=raw.get("block_index") if isinstance(raw.get("block_index"), int) else None,
                                 warnings=[str(w) for w in raw.get("warnings") or []]))
        boxes[number] = out
    return boxes, sizes, warnings


def match_bbox(bbox: list[float] | None, boxes: list[LayoutBox], cfg: LayoutConfig) -> LayoutMatch | None:
    """Best box by containment (intersection / smaller area). Ambiguous when another box with a different
    role is almost as good. No match below cfg.min_overlap."""
    if not bbox or _area(bbox) <= 0 or not boxes:
        return None
    scored = []
    for box in boxes:
        inter = _intersection(bbox, box.bbox)
        if inter <= 0:
            continue
        containment = inter / max(1e-9, min(_area(bbox), _area(box.bbox)))
        scored.append((containment, box.score or 0.0, box))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], -item[1]))
    best_containment, _, best = scored[0]
    if best_containment < cfg.min_overlap:
        return None
    alternatives = [{"label": box.label, "score": box.score, "containment": round(c, 3), "block_index": box.block_index}
                    for c, _, box in scored[1:4]]
    ambiguous = any(c >= best_containment - cfg.ambiguity_delta and role_of(box.label) != role_of(best.label)
                    for c, _, box in scored[1:])
    note = "overlapping layout regions with different roles" if ambiguous else "single best region"
    return LayoutMatch(label=best.label, role=role_of(best.label), score=best.score,
                       containment=round(best_containment, 3), block_index=best.block_index, ambiguous=ambiguous,
                       alternatives=alternatives, note=note)


def segment_boxes(element_raw: dict[str, Any], page_boxes: list[LayoutBox]) -> list[LayoutBox]:
    """Layout boxes that Nass recorded as the text blocks of a page-OCR segment (body_segment.text_blocks),
    in the recorded order. Empty when Nass recorded no mapping."""
    segment = element_raw.get("body_segment")
    if not isinstance(segment, dict):
        return []
    indices = segment.get("text_blocks")
    if not isinstance(indices, list):
        return []
    by_index = {box.block_index: box for box in page_boxes if box.block_index is not None}
    return [by_index[i] for i in indices if i in by_index]
