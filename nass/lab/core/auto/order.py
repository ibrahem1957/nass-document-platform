"""Scanned-page reading order: place the full-page body text among the visuals using PP-DocLayout boxes.

The masked-page OCR returns ONE text for the page; Qari gives no coordinates for its paragraphs. The
layout model gives text-block boxes and visual boxes. All boxes are put in reading order (rows top to
bottom, right-to-left on Arabic pages); consecutive text blocks form a text SEGMENT, and visuals sit
between segments. The body text is then split into units and assigned to segments, deterministically:

  page_text_first                    no text blocks detected: body first, then visuals in box order
  single_text_segment                all text blocks are contiguous: exact (no visual between them)
  paragraphs_aligned_to_text_blocks  the number of OCR paragraphs equals the number of text blocks:
                                     paragraph i is placed at text block i
  lines_aligned_to_text_blocks       the same with OCR lines
  proportional_to_text_block_area    otherwise: units are distributed over the segments in order, in
                                     proportion to each segment's text-block area. Order is preserved,
                                     but a unit near a boundary may land one segment early or late
                                     (recorded as a limitation; never presented as exact)
No paragraph coordinates are invented: only the segment a unit belongs to is decided.
"""

from __future__ import annotations

import re

from lab.core.auto.regions import reading_order

ORDER_VERSION = "1"
LIMITATIONS = {
    "page_text_first": "no text blocks were detected; the body text is placed before the visuals",
    "single_text_segment": None,
    "paragraphs_aligned_to_text_blocks": "assumes the OCR paragraph order matches the text-block order",
    "lines_aligned_to_text_blocks": "assumes the OCR line order matches the text-block order",
    "proportional_to_text_block_area": "approximate: text near a figure boundary may be placed one segment "
                                       "early or late (Qari returns no paragraph coordinates)",
    "no_page_text": "the page OCR is not available",
}


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def units(text: str) -> tuple[list[str], list[str]]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return paragraphs, lines


def place_body(body_text: str | None, visuals: list[dict], text_blocks: list[dict], *, rtl: bool) -> dict:
    """Sequence of {"kind": "body", "text", "segment", "text_blocks"} and {"kind": "visual", "item"}."""
    wrapped_visuals = [{"bbox_pt": v["bbox_pt"], "_kind": "visual", "_ref": v} for v in visuals]
    if body_text is None:
        return {"sequence": [{"kind": "visual", "item": w["_ref"]} for w in reading_order(wrapped_visuals, rtl=rtl)],
                "method": "no_page_text", "limitation": LIMITATIONS["no_page_text"], "segments": 0, "units": 0}
    if not text_blocks:
        sequence = [{"kind": "body", "text": body_text.strip(), "segment": 0, "text_blocks": []}] if body_text.strip() else []
        sequence += [{"kind": "visual", "item": w["_ref"]} for w in reading_order(wrapped_visuals, rtl=rtl)]
        return {"sequence": sequence, "method": "page_text_first", "limitation": LIMITATIONS["page_text_first"],
                "segments": 1, "units": 1}
    wrapped_text = [{"bbox_pt": t["bbox_pt"], "_kind": "text", "_ref": t} for t in text_blocks]
    skeleton: list[dict] = []
    for item in reading_order(wrapped_visuals + wrapped_text, rtl=rtl):
        if item["_kind"] == "text":
            if skeleton and skeleton[-1]["kind"] == "segment":
                skeleton[-1]["blocks"].append(item["_ref"])
            else:
                skeleton.append({"kind": "segment", "blocks": [item["_ref"]]})
        else:
            skeleton.append({"kind": "visual", "item": item["_ref"]})
    segments = [s for s in skeleton if s["kind"] == "segment"]
    for index, segment in enumerate(segments):
        segment["index"], segment["units"] = index, []
    paragraphs, lines = units(body_text)
    blocks = [(segment, block) for segment in segments for block in segment["blocks"]]
    joiner = "\n\n"
    if len(segments) == 1:
        method = "single_text_segment"
        segments[0]["units"] = paragraphs
    elif paragraphs and len(paragraphs) == len(blocks):
        method = "paragraphs_aligned_to_text_blocks"
        for (segment, _block), unit in zip(blocks, paragraphs):
            segment["units"].append(unit)
    elif lines and len(lines) == len(blocks):
        method, joiner = "lines_aligned_to_text_blocks", "\n"
        for (segment, _block), unit in zip(blocks, lines):
            segment["units"].append(unit)
    else:
        method = "proportional_to_text_block_area"
        chosen = paragraphs if len(paragraphs) >= len(segments) else lines
        joiner = "\n\n" if chosen is paragraphs else "\n"
        weights = [sum(_area(b["bbox_pt"]) for b in s["blocks"]) or 1.0 for s in segments]
        total_weight, bounds, running = sum(weights), [], 0.0
        for weight in weights:
            running += weight
            bounds.append(running / total_weight)
        total_chars = sum(len(u) for u in chosen) or 1
        position, current = 0, 0
        for unit in chosen:
            middle = (position + len(unit) / 2) / total_chars
            while current < len(segments) - 1 and middle > bounds[current]:
                current += 1  # monotonic: units never move backwards
            segments[current]["units"].append(unit)
            position += len(unit)
    sequence = []
    for slot in skeleton:
        if slot["kind"] == "visual":
            sequence.append({"kind": "visual", "item": slot["item"]})
        elif slot["units"]:
            sequence.append({"kind": "body", "text": joiner.join(slot["units"]), "segment": slot["index"],
                             "text_blocks": [b.get("block_index") for b in slot["blocks"]]})
    return {"sequence": sequence, "method": method, "limitation": LIMITATIONS[method], "segments": len(segments),
            "units": sum(len(s["units"]) for s in segments)}
