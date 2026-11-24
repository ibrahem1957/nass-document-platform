"""Assignable units (elements, or derived spans of segmented elements) and their per-unit features."""

from __future__ import annotations

import re
import statistics
from typing import Any

from .config import Config
from .layout import LayoutBox, segment_boxes
from .model import DerivedSpan, SourceElement, Unit
from .segmentation import segment_text
from .text_patterns import (
    bullet_marker,
    caption_match,
    ends_with_colon,
    ends_with_terminal,
    is_axis_like,
    is_fragment,
    is_numeric_label,
    is_page_number,
    list_marker,
    numbering,
    script_of,
    words,
)

MARKER_ONLY_LINE_RE = re.compile(
    r"(?:\(?[0-9٠-٩۰-۹]{1,3}(?:\.[0-9٠-٩۰-۹]{1,3})*[.)\-–]?|\(?[A-Za-z][.)]|[IVXLCDM]{1,6}[.)])"
)


def should_segment(element: SourceElement, cfg: Config) -> bool:
    seg = cfg.segmentation
    return bool(seg.enabled and element.text and element.text.strip()
                and (element.element_type in seg.element_types or (element.source or "") in seg.sources))


def build_units(elements: list[SourceElement], pages: dict[int, dict[str, Any]],
                layout_boxes: dict[int, list[LayoutBox]], cfg: Config) -> tuple[list[Unit], dict[str, list[DerivedSpan]]]:
    units: list[Unit] = []
    spans_by_element: dict[str, list[DerivedSpan]] = {}
    for element in elements:
        page = pages.get(element.page) or {}
        route, direction = str(page.get("route") or "UNKNOWN"), page.get("direction")
        if should_segment(element, cfg):
            boxes = segment_boxes(element.raw, layout_boxes.get(element.page, [])) if cfg.layout.enabled else []
            spans = segment_text(element.element_id, element.text or "", boxes, cfg.segmentation)
            if spans:
                spans_by_element[element.element_id] = spans
                for index, span in enumerate(spans):
                    units.append(Unit(unit_id=span.span_id, element=element, span=span, order=(element.order, index),
                                      text=span.text, page=element.page, route=route, direction=direction))
                continue
        units.append(Unit(unit_id=element.element_id, element=element, span=None, order=(element.order, 0),
                          text=element.text or "", page=element.page, route=route, direction=direction))
    return units, spans_by_element


def _alignment(bbox: list[float], page_width: float, direction: str | None) -> dict[str, Any]:
    x0, x1 = bbox[0], bbox[2]
    left, right = x0 / page_width, (page_width - x1) / page_width
    width = (x1 - x0) / page_width
    center_offset = abs((x0 + x1) / 2 - page_width / 2) / page_width
    if width > 0.75:
        alignment = "full"
    elif center_offset < 0.05 and abs(left - right) < 0.08:
        alignment = "center"
    elif right < left:
        alignment = "right"
    else:
        alignment = "left"
    start_side = "right" if direction == "rtl" else "left"
    return {"alignment": alignment, "start_aligned": alignment == start_side, "width_share": round(width, 3),
            "left_margin_share": round(left, 3), "right_margin_share": round(right, 3)}


def compute_features(units: list[Unit], page_sizes: dict[int, tuple[float, float]], cfg: Config) -> None:
    p = cfg.patterns
    by_page: dict[int, list[Unit]] = {}
    for unit in units:
        by_page.setdefault(unit.page, []).append(unit)
    for unit in units:
        text = unit.clean
        n = numbering(text) if text else None
        caption, caption_reason = caption_match(text, p.short_words) if text else (False, "")
        lines = [line for line in unit.text.splitlines() if line.strip()]
        # A numbering marker alone on the first line ("1." then a line break, as Word list numbering with a tab
        # often extracts) is part of the same heading line, not a second line of text.
        marker_only_line = bool(n and len(lines) > 1 and MARKER_ONLY_LINE_RE.fullmatch(lines[0].strip()))
        n_lines = (len(lines) - 1 if marker_only_line else len(lines)) if lines else (1 if text else 0)
        marker, marker_position = list_marker(text, unit.direction, cfg.lists.trailing_marker_max_words) \
            if text else (None, None)
        unit.features = {
            "list_marker": marker, "marker_position": marker_position, "marker_only_first_line": marker_only_line,
            "n_chars": len(text), "n_words": len(words(text)), "n_lines": n_lines,
            "ends_colon": ends_with_colon(text), "terminal_punctuation": ends_with_terminal(text),
            "numbering": n.to_dict() if n else None, "bullet": bullet_marker(text), "caption": caption,
            "caption_reason": caption_reason, "page_number": bool(text) and is_page_number(text),
            "numeric_label": bool(text) and is_numeric_label(text), "axis_like": bool(text) and is_axis_like(text),
            "fragment": bool(text) and is_fragment(text, p.fragment_max_chars), "script": script_of(text),
            "geometry": None,
        }
    for page, page_units in by_page.items():
        size = page_sizes.get(page)
        boxed = [u for u in page_units if u.bbox is not None]
        if size is None or not boxed:
            continue
        width, height = size
        gaps = []
        previous = None
        for unit in boxed:
            if unit.bbox[3] - unit.bbox[1] > 0.6 * height:
                continue  # page-wide boxes carry no paragraph-level position
            geometry = {"top_share": round(unit.bbox[1] / height, 3), "bottom_share": round(unit.bbox[3] / height, 3),
                        "height_pt": round(unit.bbox[3] - unit.bbox[1], 2), "gap_before_pt": None, "gap_after_pt": None}
            geometry.update(_alignment(unit.bbox, width, unit.direction))
            if previous is not None:
                gap = unit.bbox[1] - previous.bbox[3]
                geometry["gap_before_pt"] = round(gap, 2)
                previous.features["geometry"]["gap_after_pt"] = round(gap, 2)
                if gap > 0:
                    gaps.append(gap)
            unit.features["geometry"] = geometry
            previous = unit
        median_gap = statistics.median(gaps) if gaps else None
        for unit in boxed:
            if unit.features.get("geometry") is not None:
                unit.features["geometry"]["page_median_gap_pt"] = round(median_gap, 2) if median_gap else None
