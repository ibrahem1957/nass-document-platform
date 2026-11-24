"""Visual masking for full-page body-text OCR on scanned pages and embedded scans.

Full-page Qari transcribes body text AND the visuals, while every automatically processed visual is
also transcribed from its own crop. Ordered-text de-duplication cannot reliably remove the page copy
when the two transcriptions put the same values in a different order. Masking prevents the duplicate
at the source: the page copy sent for body-text OCR has every automatically processed visual painted
solid white, so the visual's content comes only from its (unmasked) crop.

Rules (policy MASK_POLICY):
- the original page image is never changed; masking works on a copy;
- the mask box is the visual's DETECTED box (PP-DocLayout pixels), intersected with the final region
  box so that a foreign caption trimmed off the top stays visible; the 60 px crop margin is never
  applied, and an attached title outside the detected box is not masked;
- only regions processed automatically (auto / auto_flagged) are masked; review-queue regions are not
  (they are not transcribed from a crop, so their content must stay in the page OCR);
- identical boxes are masked once and a box fully inside another is recorded as covered; overlapping
  boxes are each filled exactly (their union), so no mask is ever enlarged;
- text blocks (titles, captions, paragraphs) that intersect a mask are reported: that text may be
  missing from the page OCR and must be recovered from the visual's crop OCR.
"""

from __future__ import annotations

from typing import Any

from lab.core.auto import regions as auto_regions
from lab.core.hybrid import layout_regions as lr

MASK_POLICY = "white_fill_detected_box"
MASK_VERSION = "1"
FILL = (255, 255, 255)
TEXT_OVERLAP_REPORT = 0.05  # report a text block when >= 5% of it lies under a mask


def _clamp_box(box: list[float], image_px: list[int]) -> list[int] | None:
    """Integer pixel box covering exactly the detected area (rounded to the nearest pixel edge)."""
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(image_px[0]), x1), min(int(image_px[1]), y1)
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def _inside(inner: list[int], outer: list[int]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def mask_boxes(selection: dict) -> list[dict]:
    """Mask rectangles (image pixels) for the automatically processed regions of one selection."""
    image_px, image_bbox_pt = selection["image_px"], selection["image_bbox_pt"]
    masks: list[dict] = []
    for region in selection["regions"]:
        if region.get("decision") not in auto_regions.PROCESSED or not region.get("region_id"):
            continue
        detected = [float(v) for v in region["bbox_px"]]
        final = lr.pt_to_image_px(region["box_pt_before_margin"], image_bbox_pt, image_px)  # after caption trim
        box = [max(detected[0], final[0]), max(detected[1], final[1]),
               min(detected[2], final[2]), min(detected[3], final[3])]
        source = "detected box ∩ final region box (no crop margin)"
        if box[2] <= box[0] or box[3] <= box[1]:
            box, source = detected, "detected box (no crop margin)"
        pixels = _clamp_box(box, image_px)
        if pixels is None:
            continue
        entry = {"region_id": region["region_id"], "label": region["label"], "box_px": pixels,
                 "box_pt": lr.px_to_pt(pixels, image_px, image_bbox_pt), "source": source,
                 "detected_box_px": [round(v, 1) for v in detected], "crop_margin_applied": False}
        duplicate = next((m for m in masks if m["box_px"] == pixels), None)
        covered = next((m for m in masks if _inside(pixels, m["box_px"])), None)
        if duplicate is not None or covered is not None:
            entry["covered_by"] = (duplicate or covered)["region_id"]  # masked once only
        masks.append(entry)
    return masks


def apply_masks(image, masks: list[dict]):
    """A NEW image with every mask filled white; `image` itself is not modified."""
    from PIL import ImageDraw  # noqa: PLC0415

    masked = image.copy()
    draw = ImageDraw.Draw(masked)
    for mask in masks:
        if mask.get("covered_by"):
            continue
        x0, y0, x1, y1 = mask["box_px"]
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=FILL)  # PIL rectangles include the end pixel
    return masked


def text_under_masks(masks: list[dict], selection: dict) -> list[dict]:
    """Text blocks (incl. attached titles) that a mask may hide from the page OCR."""
    image_px, image_bbox_pt = selection["image_px"], selection["image_bbox_pt"]
    blocks = [dict(b, attached=False) for b in selection.get("text_blocks") or []]
    blocks += [dict(b, attached=True) for b in selection.get("attached_titles") or []]
    found = []
    for block in blocks:
        box = [float(v) for v in block.get("bbox_px") or lr.pt_to_image_px(block["bbox_pt"], image_bbox_pt, image_px)]
        area = max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))
        for mask in masks:
            if mask.get("covered_by"):
                continue
            m = mask["box_px"]
            inter = max(0.0, min(box[2], m[2]) - max(box[0], m[0])) * max(0.0, min(box[3], m[3]) - max(box[1], m[1]))
            if inter / area >= TEXT_OVERLAP_REPORT:
                found.append({"block_index": block.get("block_index"), "label": block["label"],
                              "attached_title": block["attached"], "region_id": mask["region_id"],
                              "share_masked": round(inter / area, 3),
                              "note": "this text may be missing from the page OCR; it is expected in the crop "
                                      f"OCR of {mask['region_id']}"})
    return found


def plan(selection: dict, image) -> tuple[Any, dict]:
    """(masked copy or None, masking record) for one page image and its region selection."""
    masks = mask_boxes(selection)
    review_ids = [r["region_id"] for r in selection["regions"] if r.get("decision") == auto_regions.REVIEW]
    record: dict[str, Any] = {
        "policy": MASK_POLICY, "version": MASK_VERSION, "applied": bool(masks), "fill": list(FILL),
        "image_px": list(selection["image_px"]), "image_bbox_pt": list(selection["image_bbox_pt"]),
        "masked_region_ids": [m["region_id"] for m in masks], "masks": masks,
        "not_masked_review_region_ids": review_ids,
        "text_under_masks": text_under_masks(masks, selection) if masks else [],
        "rule": "detected box only (no 60 px crop margin); auto and auto_flagged regions only",
    }
    return (apply_masks(image, masks) if masks else None), record


def masked_fraction(image, box: list[int]) -> float:
    """Share of pure-white pixels inside a box (used by tests and audits)."""
    crop = image.convert("RGB").crop(tuple(box))
    pixels = list(crop.getdata())
    return sum(1 for p in pixels if tuple(p) == FILL) / max(1, len(pixels))

