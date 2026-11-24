"""Layout detection and automatic region selection (scanned pages and embedded scans on MIXED pages).

Detection: local PP-DocLayout (PaddleOCR `LayoutDetection`, model loaded once, under the outbound
network guard) on the page image's own pixels; no deskew, unwarping, orientation correction or resize.

Selection reuses the proven rules of `hybrid.layout_regions` unchanged: `attach_titles` (titles
overlapping the top boundary attached in full, captions of a previous figure not attached),
`classify` (exact duplicates removed and recorded; containment >= 90% prefers the complete outer
visual or splits into distinct inner visuals; partial overlaps uncertain; small boxes not
preselected), `trim_foreign_captions` and `flag_cut_content` (proposed expansion, never applied
automatically). Each region then gets an automatic decision:

  auto          processed automatically (selected, no review flag)
  auto_flagged  processed automatically AND listed in the review queue: the outer visual was preferred
                over unclear inner boxes, its border may cut another block, or a nearby title/caption
                is ambiguous (the title stays unattached; the ambiguity is reported, never a reason to
                skip an otherwise complete visual). Runs created before this change use "auto_review".
  review        not processed until a reviewer clicks "Process now" (uncertain / small / contained /
                container / policy review_all). Never discarded: listed with image, box and reasons.
  removed       exact duplicate of another box (recorded; processing it would duplicate content)
"""

from __future__ import annotations

from typing import Any

from lab.core.hybrid import layout_regions as lr

TEXT_LABELS = frozenset({"text", "paragraph_title", "doc_title", "abstract", "content", "reference",
                         "reference_content", "footnote", "aside_text", "number", "header", "footer",
                         "vision_footnote", "formula", "formula_number", "algorithm"})
AUTO, AUTO_FLAGGED, REVIEW, REMOVED = "auto", "auto_flagged", "review", "removed"
AUTO_REVIEW = "auto_review"  # legacy name of auto_flagged (runs created before masking); read only
PROCESSED = frozenset({AUTO, AUTO_FLAGGED, AUTO_REVIEW})
FLAGGED = frozenset({AUTO_FLAGGED, AUTO_REVIEW})
ROW_OVERLAP = 0.5


class LayoutDetector:
    """Local PP-DocLayout. `predictor` may be injected (tests); otherwise loaded lazily, once."""

    def __init__(self, *, model_name: str, model_dir: str | None, device: str | None,
                 threshold: float | None, predictor: Any = None) -> None:
        self.model_name, self.model_dir, self.device, self.threshold = model_name, model_dir, device, threshold
        self._predictor = predictor
        self.injected = predictor is not None

    def setup(self) -> dict:
        if self.injected:
            return {"ok": True, "problems": [], "model_name": self.model_name, "injected": True}
        return lr.doclayout_setup(self.model_dir, self.model_name)

    def describe(self) -> dict:
        setup = self.setup()
        return {"model_name": self.model_name, "model_dir": self.model_dir, "device": self.device,
                "threshold": self.threshold, "versions": setup.get("versions"), "labels": setup.get("labels"),
                "model_files": setup.get("model_files"), "injected_predictor": self.injected,
                "preprocessing": "none (no deskew, unwarping, orientation correction or resize)"}

    def detect(self, image) -> list[dict]:
        """Boxes in the pixel space of `image` (PIL RGB)."""
        import numpy as np  # noqa: PLC0415

        from lab.core.e2.airgap import outbound_network_guard  # noqa: PLC0415

        with outbound_network_guard():
            if self._predictor is None:
                setup = self.setup()
                if not setup["ok"]:
                    raise lr.LayoutSourceError("Local PP-DocLayout is not ready: " + "; ".join(setup["problems"]))
                self._predictor = lr._load_predictor(self.model_name, setup["model_dir"], self.device, self.threshold)
            array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()  # BGR, as PaddleX expects
            results = list(self._predictor.predict(array, batch_size=1))
        boxes = []
        for result in results:
            for index, box in enumerate(lr._result_boxes(result)):
                coordinate = [float(v) for v in (box.get("coordinate") or [])][:4]
                if len(coordinate) != 4:
                    continue
                warnings = []
                if (min(coordinate) < -lr.BOUNDS_TOLERANCE_PX or coordinate[2] > image.width + lr.BOUNDS_TOLERANCE_PX
                        or coordinate[3] > image.height + lr.BOUNDS_TOLERANCE_PX):
                    warnings.append(f"box {coordinate} lies outside the {image.width}x{image.height} px image")
                if coordinate[2] <= coordinate[0] or coordinate[3] <= coordinate[1]:
                    warnings.append("empty or inverted box")
                score, cls_id = box.get("score"), box.get("cls_id")
                boxes.append({"label": str(box.get("label") or "unknown"), "bbox_px": coordinate,
                              "score": float(score) if score is not None else None,  # numpy -> JSON-safe
                              "cls_id": int(cls_id) if cls_id is not None else None, "block_index": len(boxes),
                              "warnings": warnings})
        return boxes


def reading_order(items: list[dict], *, rtl: bool) -> list[dict]:
    """Rows (vertical overlap >= 50% of the shorter box) top to bottom; within a row right-to-left
    for Arabic pages, left-to-right otherwise. Uses `bbox_pt`."""
    rows: list[list[dict]] = []
    for item in sorted(items, key=lambda i: (i["bbox_pt"][1], i["bbox_pt"][0])):
        y0, y1 = item["bbox_pt"][1], item["bbox_pt"][3]
        for row in rows:
            ry0, ry1 = min(r["bbox_pt"][1] for r in row), max(r["bbox_pt"][3] for r in row)
            overlap = max(0.0, min(y1, ry1) - max(y0, ry0))
            if overlap >= ROW_OVERLAP * max(1e-6, min(y1 - y0, ry1 - ry0)):
                row.append(item)
                break
        else:
            rows.append([item])
    ordered = []
    for row in rows:
        ordered += sorted(row, key=lambda i: -i["bbox_pt"][2] if rtl else i["bbox_pt"][0])
    return ordered


def select_regions(page_number: int, boxes: list[dict], *, image_px: list[int], image_bbox_pt: list[float],
                   labels: tuple[str, ...] | list[str], margin_px: int, attach_titles: bool = True,
                   captions_below: bool = False, policy: str = "confident_only", id_prefix: str = "R") -> dict:
    """Regions (with pixel boxes incl. margin) and the other blocks, with every decision recorded."""
    items = []
    for box in boxes:
        detected = lr.px_to_pt(box["bbox_px"], image_px, image_bbox_pt)
        items.append(dict(box, page=page_number, bbox_pt=list(detected), bbox_pt_detected=detected))
    targets_and_titles = [i for i in items if i["label"] in labels or i["label"] in lr.TITLE_LABELS | lr.CAPTION_LABELS]
    lr.attach_titles(targets_and_titles, captions_below=captions_below, attach=attach_titles)
    attached = [i for i in items if str(i.get("title_status", "")).startswith("attached")]
    attached_ids = {id(i) for i in attached}
    chosen, others = [], []
    for item in items:
        if id(item) in attached_ids:
            continue
        if item.get("reasons_extra"):
            if item["label"] in labels:
                # Title/caption ambiguity of a visual: reported, but not a box warning (a warning makes
                # classify() mark the box uncertain, which would skip an otherwise complete visual).
                item["title_ambiguity"] = list(item["reasons_extra"])
            else:
                item["warnings"] = list(item.get("warnings") or []) + item["reasons_extra"]
        (chosen if item["label"] in labels else others).append(item)
    classified = lr.classify(chosen)
    live = [i for i in classified if i["status"] != "duplicate removed"]
    not_attached = [i for i in items if id(i) not in attached_ids]
    margin_pt = margin_px * (image_bbox_pt[3] - image_bbox_pt[1]) / float(image_px[1])
    lr.trim_foreign_captions(live, not_attached, margin_pt)
    lr.flag_cut_content(live, not_attached)
    regions = []
    number = 0
    for item in classified:
        if item["status"] == "duplicate removed":
            regions.append(dict(item, region_id=None, decision=REMOVED, pixel_box=None))
            continue
        region_id = f"p{page_number}_{id_prefix}{number}"
        number += 1
        for note in item.get("title_ambiguity") or []:
            item["reasons"].append(note + " (title left unattached)")
        before_margin = lr.pt_to_image_px(item["bbox_pt"], image_bbox_pt, image_px)
        pixel_box = lr.with_margin(before_margin, margin_px, image_px)
        needs_review = any("Needs Review" in str(reason) for reason in item["reasons"])
        if policy == "review_all":
            decision, why = REVIEW, "selection policy review_all"
        elif item["status"] == "selected":
            decision = AUTO_FLAGGED if needs_review else AUTO
            why = "selected; processed automatically" + ("; flagged for review" if needs_review else "")
        else:
            decision, why = REVIEW, f"status {item['status']}: not processed automatically (Process now available)"
        regions.append(dict(item, region_id=region_id, decision=decision, decision_reason=why,
                            box_pt_before_margin=list(item["bbox_pt"]),
                            box_px_before_margin=[round(v, 1) for v in before_margin], pixel_box=pixel_box,
                            margin_px=margin_px))
    # Safety net: two automatically processed regions must never overlap (no duplicate processing).
    for _first, second in overlapping_auto(regions):
        region = next(r for r in regions if r["region_id"] == second)
        if region["decision"] in PROCESSED:
            region["decision"] = REVIEW
            region["decision_reason"] = f"overlaps automatically processed region {_first}; moved to review"
            region["reasons"].append(f"Needs Review: overlaps {_first} after title/caption adjustments")
    return {"page": page_number, "regions": regions,
            "others": [dict(o) for o in others], "attached_titles": [dict(a) for a in attached],
            "image_px": list(image_px), "image_bbox_pt": list(image_bbox_pt), "labels": list(labels),
            "margin_px": margin_px, "policy": policy,
            "text_blocks": [dict(o) for o in others if o["label"] in TEXT_LABELS or o["label"] in lr.TITLE_LABELS]}


def overlapping_auto(regions: list[dict]) -> list[tuple[str, str]]:
    """Pairs of automatically processed regions that overlap (should be empty: no duplicate processing)."""
    auto = [r for r in regions if r["decision"] in PROCESSED]
    pairs = []
    for i, a in enumerate(auto):
        for b in auto[i + 1:]:
            if lr._intersection(a["box_pt_before_margin"], b["box_pt_before_margin"]) > 0:
                pairs.append((a["region_id"], b["region_id"]))
    return pairs
