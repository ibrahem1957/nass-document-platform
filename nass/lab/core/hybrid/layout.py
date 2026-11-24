"""Phase 1 layout-aware page assembly for Hybrid PDF Extraction (local only).

One clean Markdown document per page: text blocks, tables and figure placeholders at
their reading position. No OCR/VLM call, no network. Everything is derived from the
run's saved data (decisions, tables.json, saved model text) plus the PDF itself,
whose SHA-256 must equal the run's document hash.

Text: block text uses the SAME rules as the accepted page-level text (raw blocks for
raw/not_needed pages, the processed reconstruction per block for reconstructed /
uncertain pages, plus the VLM-guided `library_rejoined` space removal for B pages).
Validation: the multiset of non-whitespace characters of all assembled text elements
(kept blocks, table-deduplicated blocks and chart labels) must equal that of the
accepted page-level text. Whole elements may be reordered; nothing may be lost,
duplicated or modified. On failure the accepted page text is placed as one region.

Layout version 2 (corrections over version 1): paragraph merging from several layout
signals (whitespace-only joins) and Markdown list items; conservative frame/inner raster
image deduplication; the small-image rule (2% area OR 100 pt longest side) with
borderline images flagged; table gridlines excluded from vector-chart detection by
coordinates (zero-area lines were missed before). Text characters are unchanged.

Layout version 3: side-by-side elements with a narrow gutter are ordered by direction
(right-to-left on Arabic pages) when they overlap vertically; bullet markers never make a
heading, formatting-only bullets are recorded and not rendered, trailing bullets of RTL
lines are list markers; a table continued from the previous page reuses that page's
header only when the continuation is reliable, otherwise it gets a blank header row (its
first data row is never promoted to a header); structural checks are recorded.

Phase 2 (not implemented here) plugs in through `assemble_page(..., transcriptions=)`:
{figure_id: {"text", "model"}} replaces a placeholder with the model's transcription.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter
from typing import Any

from lab.core.hybrid import textcheck
from lab.local_pdf_extraction import arabic_reorder as ar

LAYOUT_VERSION = "3"
DEFAULTS: dict[str, float] = {
    "min_image_area_share": 0.02,   # skip a raster image below 2% of the page area ...
    "min_image_long_side_pt": 100.0,  # ... OR whose longest side is below 100 pt (logos, icons)
    "image_borderline_factor": 0.8,  # skipped images within 80% of a threshold are flagged for review
    "frame_containment": 0.95,      # inner image lies (almost) fully inside the outer one
    "frame_inner_coverage": 0.5,    # ... and covers at least half of it
    "frame_density_ratio": 0.5,     # ... and the outer one has at most half its pixel density (frame/shadow)
    "image_overlap_review": 0.5,    # other overlaps above 50% of the smaller image -> review, both kept
    "table_tolerance_pt": 2.0,      # drawing paths within a table bbox (+2 pt) are table gridlines
    "chart_table_overlap": 0.3,     # a remaining drawing region overlapping a table this much -> review
    "paragraph_line_pitch": 1.5,    # next line starts within 1.5 x line height -> same paragraph
    "paragraph_font_tolerance": 0.6,
    "paragraph_align_overlap": 0.6,
    "paragraph_short_line": 0.85,   # a last line ending this far before the column edge ends the paragraph
    "paragraph_indent_factor": 1.5,  # first-line indent (x font size) starts a new paragraph
    "side_by_side_min_gap": 1.0,     # any real gutter (pt) between vertically overlapping elements
    "side_by_side_overlap": 0.3,     # vertical overlap (share of the shorter one) for side-by-side
    "continuation_top_share": 0.2,   # a continued table starts in the top 20% of the page ...
    "continuation_bottom_share": 0.8,  # ... and the previous one ends in the bottom 20%
    "continuation_column_tolerance": 0.03,  # column centres within 3% of the page width
    "table_overlap": 0.6,           # a text block mostly inside a table is the table's content
    "chart_overlap": 0.6,           # a text block mostly inside a chart region is a chart label
    "heading_ratio": 1.25,          # font size >= 1.25 x body -> heading
    "h2_ratio": 1.6,                # >= 1.6 x body -> level-2 heading
    "y_gap": 4.0,                   # minimum horizontal band gap (pt) for a row cut
    "x_gap": 12.0,                  # minimum vertical gutter (pt) for a column cut
    "chart_min_items": 10,          # vector drawing items needed for a chart region
    "chart_min_area_share": 0.02,
    "drawing_merge_gap": 12.0,
    "caption_gap": 28.0,
    "crop_dpi": 150,
}
LIST_MARKER = re.compile(r"^\s*(?P<marker>[•●▪◦‣∙·\-–*]|[0-9\u0660-\u0669]{1,3}[).]|[a-zA-Z][)])\s+(?P<rest>\S.*)$")
BULLET_GLYPHS = frozenset("•●▪◦‣∙·–")
TRAILING_BULLET = re.compile(r"^(?P<rest>\S.*?)\s*(?P<marker>[•●▪◦‣∙·])\s*$")


def bullet_only(text: str) -> bool:
    visible = [ch for ch in text if not ch.isspace()]
    return bool(visible) and all(ch in BULLET_GLYPHS for ch in visible)


def list_match(line: str, rtl: bool):
    """(marker, rest) when the line is a list item; RTL lines may carry the bullet at the end."""
    match = LIST_MARKER.match(line)
    if match:
        return match.group("marker"), match.group("rest")
    if rtl:
        match = TRAILING_BULLET.match(line.strip())
        if match and not bullet_only(match.group("rest")):
            return match.group("marker"), match.group("rest")
    return None
CAPTION_START = re.compile(r"^\s*(figure|fig\.|table|chart|source|شكل|الشكل|جدول|الجدول|رسم|مخطط|المصدر)",
                           re.IGNORECASE)
# Phase 2 design: rules every image/chart transcription request must state (not used yet).
FIGURE_TRANSCRIPTION_RULES = (
    "Transcribe literally the visible titles, legends, axis labels, units and numeric values. "
    "Never estimate values from bar heights, pie angles or visual proportions, and never use surrounding text. "
    "If a value is not readable, write that it is unreadable. Keep the original number formatting and units."
)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _inside_share(inner, outer) -> float:
    area = _area(inner)
    return _inter(inner, outer) / area if area else 0.0


def _union(boxes) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _r(box) -> list[float]:
    return [round(float(v), 1) for v in box]


def _nonspace(text: str, separator_chars: str) -> Counter:
    return Counter(ch for ch in text if not ch.isspace() and ch not in separator_chars)


# ------------------------------------------------------------- extraction


def _child_regions(pix) -> list[list[float]]:
    """Possible sub-figures of one raster crop, split by clear near-white gutters.

    Returned as fractions of the crop. Empty when no gutter is clearly present (no
    invented subdivision)."""
    width, height, n = pix.width, pix.height, pix.n
    if width < 60 or height < 60 or n < 1:
        return []
    data = pix.samples

    def white(x: int, y: int) -> bool:
        offset = (y * width + x) * n
        return all(data[offset + c] >= 245 for c in range(min(n, 3)))

    def bands(length: int, other: int, column: bool) -> list[tuple[int, int]]:
        blank = []
        step = max(1, other // 120)
        for i in range(length):
            pixels = [white(i, j) if column else white(j, i) for j in range(0, other, step)]
            blank.append(sum(pixels) / len(pixels) >= 0.995)
        parts, start = [], None
        for i, is_blank in enumerate(blank + [True]):
            if not is_blank and start is None:
                start = i
            elif is_blank and start is not None:
                parts.append((start, i))
                start = None
        minimum = 0.2 * length
        return [p for p in parts if p[1] - p[0] >= minimum]

    for column in (True, False):
        length, other = (width, height) if column else (height, width)
        parts = bands(length, other, column)
        if len(parts) >= 2:
            return [[p[0] / length, 0.0, p[1] / length, 1.0] if column else [0.0, p[0] / length, 1.0, p[1] / length]
                    for p in parts]
    return []


def _within(box, outer, tolerance: float) -> bool:
    """Coordinate containment (works for zero-area lines, unlike an area share)."""
    return (box[0] >= outer[0] - tolerance and box[1] >= outer[1] - tolerance
            and box[2] <= outer[2] + tolerance and box[3] <= outer[3] + tolerance)


def _vector_regions(page, table_boxes: list, settings: dict, diagnostics: dict | None = None) -> list[dict]:
    """Chart candidates from drawing paths that remain after table gridlines, borders and
    rules are removed. Tables are identified first (find_tables bboxes from tables.json)."""
    diagnostics = diagnostics if diagnostics is not None else {}
    excluded = Counter()
    width, height = page.rect.width, page.rect.height
    page_area = width * height or 1.0
    tolerance = settings["table_tolerance_pt"]
    items = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None:
            continue
        box = (rect.x0, rect.y0, rect.x1, rect.y1)
        w, h = rect.width, rect.height
        if w * h >= 0.9 * page_area:
            excluded["page frame or background"] += 1
            continue
        if any(_within(box, table, tolerance) for table in table_boxes):
            excluded["inside an extracted table (gridlines, borders, cell shading)"] += 1
            continue
        if min(w, h) < 1.5 and max(w, h) > 0.5 * width:
            excluded["long horizontal/vertical rule"] += 1
            continue
        kinds = {item[0] for item in drawing.get("items") or []}
        items.append({"bbox": box, "count": len(drawing.get("items") or []), "fill": drawing.get("fill"),
                      "kinds": kinds})
    gap = settings["drawing_merge_gap"]
    groups: list[list[dict]] = []
    for item in items:
        grown = (item["bbox"][0] - gap, item["bbox"][1] - gap, item["bbox"][2] + gap, item["bbox"][3] + gap)
        touching = [group for group in groups if any(_inter(grown, other["bbox"]) > 0 for other in group)]
        merged = [item] + [member for group in touching for member in group]
        groups = [group for group in groups if group not in touching] + [merged]
    regions, ignored = [], []
    for group in groups:
        box = _union([item["bbox"] for item in group])
        count = sum(item["count"] for item in group)
        share = _area(box) / page_area
        if share < settings["chart_min_area_share"] or count < settings["chart_min_items"]:
            ignored.append({"bbox": _r(box), "items": count, "reason": "too small or too few drawing items"})
            continue
        fills = {str(item["fill"]) for item in group if item["fill"] is not None}
        curves = any("c" in item["kinds"] for item in group)
        only_lines = all(item["kinds"] <= {"l"} for item in group)
        if only_lines and not fills and not curves:
            ignored.append({"bbox": _r(box), "items": count, "reason": "only unfilled straight lines (decorative "
                                                                       "lines, borders or separators)"})
            continue
        overlap = max((_inter(box, table) / (_area(box) or 1.0) for table in table_boxes), default=0.0)
        confidence = "likely" if (len(fills) >= 2 or curves) else "suspected"
        region = {"bbox": _r(box), "items": count, "distinct_fills": len(fills), "curves": curves,
                  "area_share": round(share, 4), "confidence": confidence}
        if overlap >= settings["chart_table_overlap"]:
            region.update(confidence="suspected", table_overlap=round(overlap, 3),
                          review="drawing region partially overlaps an extracted table; kept for review")
        regions.append(region)
    diagnostics["vector_exclusions"] = dict(excluded)
    diagnostics["ignored_vector_regions"] = ignored
    return regions


def extract_page_elements(page, page_number: int, *, reorder_config: dict, tables_doc: dict | None,
                          settings: dict | None = None) -> tuple[dict, dict[str, bytes]]:
    """Geometry of one page (blocks, images, vector regions) and the figure crops (PNG)."""
    import pymupdf

    settings = DEFAULTS | (settings or {})
    cfg = ar.ReorderConfig.from_dict(reorder_config)
    width, height = page.rect.width, page.rect.height
    sizes: Counter = Counter()
    blocks = []
    for index, block in enumerate(page.get_text("rawdict")["blocks"]):
        if block.get("type") != 0:
            continue
        spans, lines, max_size, line_boxes = [], [], 0.0, []
        block_sizes: Counter = Counter()
        bold_chars = 0
        for line in block.get("lines", []):
            text = ""
            for span in line["spans"]:
                spans.append(span["chars"])
                piece = "".join(char["c"] for char in span["chars"])
                text += piece
                size = float(span.get("size") or 0)
                max_size = max(max_size, size)
                sizes[round(size, 1)] += len(piece.strip())
                block_sizes[round(size, 1)] += len(piece.strip())
                if int(span.get("flags") or 0) & 16 or "bold" in str(span.get("font") or "").lower():
                    bold_chars += len(piece.strip())
            if text.strip():
                line_boxes.append(_r(line["bbox"]))
            lines.append(text)
        raw_text = "\n".join(lines)
        if not raw_text.strip():
            continue
        processed = ar.reconstruct_clusters(ar.clusters_from_spans(spans, cfg, ar.Reconstruction(text="")), cfg).text
        visible = sum(block_sizes.values()) or 1
        blocks.append({"id": f"b{index}", "bbox": _r(block["bbox"]), "raw_text": raw_text,
                       "processed_text": processed, "max_font_size": round(max_size, 2), "line_count": len(lines),
                       "font_size": max(block_sizes.items(), key=lambda item: item[1])[0] if block_sizes else 0.0,
                       "bold": bold_chars / visible >= 0.5, "line_boxes": line_boxes})
    body = max(sizes.items(), key=lambda item: item[1])[0] if sizes else 0.0

    page_area = width * height or 1.0
    figures, skipped, crops = [], [], {}
    images = []
    for index, info in enumerate(page.get_image_info()):
        box = info.get("bbox") or (0, 0, 0, 0)
        w, h = box[2] - box[0], box[3] - box[1]
        images.append({"id": f"img{index}", "bbox": _r(box), "area_share": round(_area(box) / page_area, 4),
                       "width_pt": round(w, 1), "height_pt": round(h, 1),
                       "width_px": info.get("width"), "height_px": info.get("height"),
                       "px_per_pt": round(float(info.get("width") or 0) / w, 3) if w > 0 else 0.0})
    merges, image_review = deduplicate_images(images, settings)
    absorbed = {merge["absorbed"] for merge in merges}
    thresholds = {"min_area_share": settings["min_image_area_share"],
                  "min_long_side_pt": settings["min_image_long_side_pt"]}
    for image in images:
        if image["id"] in absorbed:
            continue
        reason = skip_reason(image, settings)
        if reason:
            factor = settings["image_borderline_factor"]
            borderline = (image["area_share"] >= factor * settings["min_image_area_share"]
                          and max(image["width_pt"], image["height_pt"]) >= factor * settings["min_image_long_side_pt"])
            skipped.append(image | {"reason": reason, "thresholds": thresholds, "borderline": borderline})
            if borderline:
                image_review.append({"images": [image["id"]], "reason": f"borderline small image skipped ({reason})"})
            continue
        figures.append(image | {"origin": "raster", "thresholds": thresholds})
    tables_boxes = [table["bbox"] for table in (tables_doc or {}).get("tables") or []]
    vector_diagnostics: dict = {}
    for index, region in enumerate(_vector_regions(page, tables_boxes, settings, vector_diagnostics)):
        if any(_inside_share(region["bbox"], fig["bbox"]) >= 0.8 for fig in figures):
            continue  # drawing on top of a raster image (frame/annotation): the image is the figure
        figures.append({"id": f"vec{index}", "origin": "vector", **region})
    for fig in figures:
        clip = pymupdf.Rect(*fig["bbox"]) & page.rect
        if clip.is_empty:
            fig["crop_sha256"] = None
            continue
        pix = page.get_pixmap(clip=clip, dpi=int(settings["crop_dpi"]))
        png = pix.tobytes("png")
        name = f"page_{page_number:03d}_{fig['id']}.png"
        crops[name] = png
        fig.update(crop_file=name, crop_sha256=hashlib.sha256(png).hexdigest(),
                   crop_px=[pix.width, pix.height])
        if fig["origin"] == "raster":
            small = page.get_pixmap(clip=clip, dpi=48, colorspace=pymupdf.csRGB)
            children = _child_regions(small)
            x0, y0, x1, y1 = fig["bbox"]
            fig["possible_child_regions"] = [
                _r((x0 + c[0] * (x1 - x0), y0 + c[1] * (y1 - y0), x0 + c[2] * (x1 - x0), y0 + c[3] * (y1 - y0)))
                for c in children]
            fig["child_regions_note"] = ("possible sub-figures separated by blank gutters; not subdivided "
                                         "automatically, review the boundaries") if children else None
        fig["phase2"] = {"units": 1, "requires_explicit_selection": fig["origin"] == "vector",
                         "transcription": None}
    return {"page": page_number, "width": round(width, 1), "height": round(height, 1),
            "body_font_size": body, "blocks": blocks, "figures": figures, "skipped_images": skipped,
            "images_found": images, "image_merges": merges, "image_review": image_review,
            "vector_exclusions": vector_diagnostics.get("vector_exclusions", {}),
            "ignored_vector_regions": vector_diagnostics.get("ignored_vector_regions", []),
            "settings": settings}, crops


def skip_reason(image: dict, settings: dict) -> str | None:
    """Skip only if the area is below the minimum share OR the longest side is below the minimum."""
    long_side = max(image["width_pt"], image["height_pt"])
    if image["area_share"] < settings["min_image_area_share"]:
        return (f"area {image['area_share']:.1%} of the page < {settings['min_image_area_share']:.0%} "
                f"({image['width_pt']:.0f}x{image['height_pt']:.0f} pt)")
    if long_side < settings["min_image_long_side_pt"]:
        return f"longest side {long_side:.0f} pt < {settings['min_image_long_side_pt']:.0f} pt"
    return None


def deduplicate_images(images: list[dict], settings: dict) -> tuple[list[dict], list[dict]]:
    """Merge a low-resolution frame/shadow with the higher-resolution image inside it.

    Only with strong evidence: (almost) full containment, the inner image covers a large
    part of the outer one, and the outer one has a much lower pixel density. Anything
    else that overlaps is kept as two regions and flagged for review. Side-by-side images
    never overlap, so they are never merged. Original records are not changed except that
    the kept inner image takes the combined (outer) region geometry."""
    merges, review = [], []
    absorbed: set[str] = set()
    for outer in sorted(images, key=lambda item: _area(item["bbox"]), reverse=True):
        if outer["id"] in absorbed:
            continue
        for inner in images:
            if inner is outer or inner["id"] in absorbed or _area(inner["bbox"]) > _area(outer["bbox"]):
                continue
            contained = _inside_share(inner["bbox"], outer["bbox"])
            smaller = min(_area(inner["bbox"]), _area(outer["bbox"])) or 1.0
            overlap = _inter(inner["bbox"], outer["bbox"]) / smaller
            if contained >= settings["frame_containment"]:
                coverage = _area(inner["bbox"]) / (_area(outer["bbox"]) or 1.0)
                density = (outer["px_per_pt"] / inner["px_per_pt"]) if inner["px_per_pt"] else 1.0
                if coverage >= settings["frame_inner_coverage"] and density <= settings["frame_density_ratio"]:
                    absorbed.add(outer["id"])
                    inner["image_bbox"] = list(inner["bbox"])
                    inner["bbox"] = _r(_union([inner["bbox"], outer["bbox"]]))
                    inner["merged_from"] = [outer["id"]]
                    merges.append({"kept": inner["id"], "absorbed": outer["id"],
                                   "reason": (f"{outer['id']} is a low-resolution frame/shadow around {inner['id']} "
                                              f"(contains it, coverage {coverage:.0%}, pixel density "
                                              f"{outer['px_per_pt']:.2f} vs {inner['px_per_pt']:.2f} px/pt)"),
                                   "combined_bbox": inner["bbox"], "content_from": inner["id"]})
                    break
                review.append({"images": [outer["id"], inner["id"]],
                               "reason": "image inside another image without frame evidence; both kept"})
            elif overlap >= settings["image_overlap_review"]:
                review.append({"images": [outer["id"], inner["id"]],
                               "reason": f"images overlap {overlap:.0%}; deduplication uncertain, both kept"})
    return merges, review


# ---------------------------------------------------------------- ordering


def _gaps(items: list[dict], axis: int) -> list[tuple[float, float]]:
    """(gap size, cut position) between merged projections on x (0) or y (1)."""
    spans = sorted((item["bbox"][axis], item["bbox"][axis + 2]) for item in items)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(merged[i + 1][0] - merged[i][1], (merged[i + 1][0] + merged[i][1]) / 2) for i in range(len(merged) - 1)]


def _split(items: list[dict], axis: int, cuts: list[float]) -> list[list[dict]]:
    groups: list[list[dict]] = [[] for _ in range(len(cuts) + 1)]
    for item in items:
        centre = (item["bbox"][axis] + item["bbox"][axis + 2]) / 2
        groups[sum(1 for cut in cuts if centre > cut)].append(item)
    return [group for group in groups if group]


def _y_overlap(a, b) -> float:
    shorter = max(1.0, min(a[3] - a[1], b[3] - b[1]))
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / shorter


def xy_cut(items: list[dict], *, rtl: bool, settings: dict, ambiguous: list[dict],
           side_by_side: list[dict] | None = None) -> list[dict]:
    """Recursive XY-cut: regions first (rows / columns), then order inside each region.
    Columns and side-by-side regions are read right-to-left on Arabic pages."""
    if len(items) <= 1:
        return list(items)
    y_gaps = [gap for gap in _gaps(items, 1) if gap[0] >= settings["y_gap"]]
    x_gaps = [gap for gap in _gaps(items, 0) if gap[0] >= settings["x_gap"]]
    best_y = max((gap[0] for gap in y_gaps), default=0.0)
    best_x = max((gap[0] for gap in x_gaps), default=0.0)
    if x_gaps and best_x >= best_y:
        groups = _split(items, 0, [gap[1] for gap in x_gaps])
        if rtl:
            groups.reverse()
    elif y_gaps:
        groups = _split(items, 1, [gap[1] for gap in y_gaps])
    else:
        # Side-by-side elements with a narrow gutter (below x_gap): order them by direction
        # when the groups genuinely overlap vertically; otherwise do not force an order.
        narrow = [gap for gap in _gaps(items, 0) if gap[0] >= settings["side_by_side_min_gap"]]
        if narrow:
            groups = _split(items, 0, [gap[1] for gap in narrow])
            spans = [_union([item["bbox"] for item in group]) for group in groups]
            overlapping_rows = all(_y_overlap(spans[i], spans[i + 1]) >= settings["side_by_side_overlap"]
                                   for i in range(len(spans) - 1))
            if len(groups) > 1 and overlapping_rows:
                if rtl:
                    groups.reverse()
                if side_by_side is not None:
                    side_by_side.append({"groups": [[item["id"] for item in group] for group in groups],
                                         "direction": "rtl" if rtl else "ltr",
                                         "gutter_pt": round(max(gap[0] for gap in narrow), 1)})
                ordered = []
                for group in groups:
                    ordered += xy_cut(group, rtl=rtl, settings=settings, ambiguous=ambiguous, side_by_side=side_by_side)
                return ordered
            if len(groups) > 1:
                ambiguous.append({"bbox": _r(_union([item["bbox"] for item in items])),
                                  "elements": [item["id"] for item in items],
                                  "reason": "horizontally separated elements without clear vertical overlap"})
        overlapping = any(_inter(a["bbox"], b["bbox"]) > 0.2 * min(_area(a["bbox"]), _area(b["bbox"]))
                          for i, a in enumerate(items) for b in items[i + 1:])
        if overlapping:
            ambiguous.append({"bbox": _r(_union([item["bbox"] for item in items])),
                              "elements": [item["id"] for item in items],
                              "reason": "overlapping elements: reading order not determined by the layout"})
        return sorted(items, key=lambda item: (round(item["bbox"][1]), -item["bbox"][2] if rtl else item["bbox"][0]))
    if len(groups) == 1:
        return sorted(items, key=lambda item: (round(item["bbox"][1]), -item["bbox"][2] if rtl else item["bbox"][0]))
    if x_gaps and best_x >= best_y and side_by_side is not None and len(groups) > 1:
        side_by_side.append({"groups": [[item["id"] for item in group] for group in groups],
                             "direction": "rtl" if rtl else "ltr", "gutter_pt": round(best_x, 1), "column_cut": True})
    ordered = []
    for group in groups:
        ordered += xy_cut(group, rtl=rtl, settings=settings, ambiguous=ambiguous, side_by_side=side_by_side)
    return ordered


# ---------------------------------------------------------------- assembly


def _row_texts(table: dict, row: int) -> list[str]:
    width = max((cell["col"] for cell in table.get("cells") or []), default=-1) + 1
    found = {cell["col"]: cell for cell in table.get("cells") or [] if cell["row"] == row}
    return [str((found.get(c) or {}).get("processed_cell_text") or "")
            if (found.get(c) or {}).get("reorder_status") not in {"merged", "not_detected"} else "" for c in range(width)]


def _column_bounds(table: dict) -> list[tuple[float, float]] | None:
    columns: dict[int, tuple[float, float]] = {}
    for cell in table.get("cells") or []:
        box = cell.get("bbox")
        if not box:
            continue
        low, high = columns.get(cell["col"], (box[0], box[2]))
        columns[cell["col"]] = (min(low, box[0]), max(high, box[2]))
    width = max((cell["col"] for cell in table.get("cells") or []), default=-1) + 1
    if not width or len(columns) != width:
        return None
    return [columns[c] for c in range(width)]


def table_continuation(table: dict, previous_tables_doc: dict | None, *, page_width: float, page_height: float,
                       settings: dict, first_on_page: bool) -> dict | None:
    """Whether this table continues one from the previous page, and if its header can be reused."""
    flag = table.get("possible_continuation_of")
    previous_tables = list((previous_tables_doc or {}).get("tables") or [])
    width = max((cell["col"] for cell in table.get("cells") or []), default=-1) + 1
    if not first_on_page:
        return None
    if not previous_tables:
        return ({"of_page": flag.get("page"), "of_table": flag.get("table_index"), "detected_by": "tables.json flag",
                 "reliable": False, "uncertainty": ["the previous page's tables are not available"], "header": None}
                if flag else None)
    previous = next((t for t in previous_tables if flag and t["table_index"] == flag.get("table_index")),
                    None) or previous_tables[-1]
    previous_width = max((cell["col"] for cell in previous.get("cells") or []), default=-1) + 1
    top = table["bbox"][1] <= settings["continuation_top_share"] * page_height
    bottom = previous["bbox"][3] >= settings["continuation_bottom_share"] * page_height
    if not flag and not (top and bottom and width == previous_width):
        return None
    reasons = []
    if width != previous_width:
        reasons.append("different column count")
    mine, theirs = _column_bounds(table), _column_bounds(previous)
    tolerance = settings["continuation_column_tolerance"] * page_width
    if mine is None or theirs is None:
        reasons.append("column geometry unavailable")
    elif len(mine) == len(theirs) and any(abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2) > tolerance
                                          for a, b in zip(mine, theirs)):
        reasons.append("column positions differ from the previous page")
    if previous.get("possible_continuation_of"):
        reasons.append("the previous table is itself a continuation (its header is not verified)")
    header = _row_texts(previous, min((cell["row"] for cell in previous.get("cells") or []), default=0))
    if not any(text.strip() for text in header):
        reasons.append("the previous table has no header text")
    if _row_texts(table, min((cell["row"] for cell in table.get("cells") or []), default=0)) == header:
        return None  # it repeats the header row: it has its own header
    return {"of_page": (flag or {}).get("page"), "of_table": previous["table_index"],
            "detected_by": "tables.json continuation flag" if flag else "layout geometry (top/bottom, same columns)",
            "reliable": not reasons, "uncertainty": reasons, "header": header if not reasons else None}


def _table_markdown(table: dict, continuation: dict | None = None) -> str:
    cells = table.get("cells") or []
    width = max((cell["col"] for cell in cells), default=-1) + 1
    if not width:
        return ""
    rows: dict[int, dict[int, dict]] = {}
    for cell in cells:
        rows.setdefault(cell["row"], {})[cell["col"]] = cell

    def text(cell: dict | None) -> str:
        if not cell or cell.get("reorder_status") in {"merged", "not_detected"}:
            return ""
        return str(cell.get("processed_cell_text") or "").replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")

    ordered = [rows[key] for key in sorted(rows)]
    separator = "|" + " --- |" * width
    if continuation is None:
        lines = ["| " + " | ".join(text(ordered[0].get(c)) for c in range(width)) + " |", separator]
        lines += ["| " + " | ".join(text(row.get(c)) for c in range(width)) + " |" for row in ordered[1:]]
        return "\n".join(lines)
    # A continued table: every row of this page is a data row (none is promoted to a header).
    if continuation["reliable"]:
        header = [value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
                  for value in continuation["header"]]
        lines = [f"_(table continued from page {continuation['of_page'] or ''}; header repeated from that page)_", "",
                 "| " + " | ".join(header) + " |", separator]
    else:
        lines = ["| " + " | ".join("" for _ in range(width)) + " |", separator]
    lines += ["| " + " | ".join(text(row.get(c)) for c in range(width)) + " |" for row in ordered]
    return "\n".join(lines)


def _table_text(table: dict) -> str:
    return " ".join(str(cell.get("processed_cell_text") or "") for cell in table.get("cells") or []
                    if cell.get("reorder_status") not in {"merged", "not_detected"})


def _block_text(block: dict, mode: str, vlm_text: str) -> str:
    text = block["processed_text"] if mode in {"processed", "rejoined"} else block["raw_text"]
    if mode in {"rejoined", "rejoined_raw_blocks"}:
        text = textcheck.rejoin(text, vlm_text)["text"]  # same VLM-guided space removal, per block
    return text


def text_mode(decision: dict, library_page: dict) -> str:
    source = decision.get("final_source") or ""
    status = (library_page.get("reorder") or {}).get("reorder_status")
    processed_like = status in {"reconstructed", "uncertain"}
    if source.startswith("vlm_raw"):
        return "single_region"
    if source.startswith("library_rejoined"):
        # not_needed pages: the processed page text equals the raw text, so raw blocks are the base
        return "rejoined" if processed_like else "rejoined_raw_blocks"
    if source.startswith("pymupdf_processed") and processed_like:
        return "processed"
    return "raw"


def _line_box(item: dict, which: int) -> list[float]:
    boxes = item.get("line_boxes") or []
    return boxes[which] if boxes else item["bbox"]


def _mergeable(a: dict, b: dict, *, rtl: bool, settings: dict, captions: set) -> tuple[bool, dict]:
    """Whether paragraph b continues paragraph a (adjacent in reading order, nothing between)."""
    if a["type"] != "paragraph" or b["type"] != "paragraph" or a["id"] in captions or b["id"] in captions:
        return False, {}
    first_b = b["text"].split("\n", 1)[0]
    if list_match(first_b, rtl) or bullet_only(first_b):
        return False, {"reason": "next line starts a list item"}
    size_a, size_b = float(a.get("font_size") or 0), float(b.get("font_size") or 0)
    if abs(size_a - size_b) > settings["paragraph_font_tolerance"] or bool(a.get("bold")) != bool(b.get("bold")):
        return False, {"reason": "different font size or weight"}
    last = _line_box(a, -1)
    line_height = max(1.0, last[3] - last[1])
    gap = b["bbox"][1] - a["bbox"][3]
    if gap < -0.5 * line_height or gap > (settings["paragraph_line_pitch"] - 1.0) * line_height:
        return False, {"reason": "vertical distance too large"}
    narrower = max(1.0, min(a["bbox"][2] - a["bbox"][0], b["bbox"][2] - b["bbox"][0]))
    overlap = max(0.0, min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0])) / narrower
    if overlap < settings["paragraph_align_overlap"]:
        return False, {"reason": "not horizontally aligned"}
    left, right = min(a["bbox"][0], b["bbox"][0]), max(a["bbox"][2], b["bbox"][2])
    width = max(1.0, right - left)
    if (right - last[2] if not rtl else last[0] - left) > (1 - settings["paragraph_short_line"]) * width:
        return False, {"reason": "previous line ends before the column edge (paragraph end)"}
    first = _line_box(b, 0)
    indent = (first[0] - left) if not rtl else (right - first[2])
    if indent > settings["paragraph_indent_factor"] * max(size_b, 1.0):
        return False, {"reason": "first-line indent"}
    return True, {"gap_pt": round(gap, 1), "line_height_pt": round(line_height, 1),
                  "font_size": size_a, "alignment_overlap": round(overlap, 2)}


def merge_paragraphs(ordered: list[dict], *, rtl: bool, settings: dict, captions: set) -> tuple[list[dict], list[dict]]:
    """Merge consecutive paragraph elements. Only the whitespace between them changes."""
    out: list[dict] = []
    merges: list[dict] = []
    for item in ordered:
        previous = out[-1] if out else None
        ok, signals = _mergeable(previous, item, rtl=rtl, settings=settings, captions=captions) if previous else (False, {})
        if ok:
            merges.append({"into": previous["id"], "merged": item["id"], "signals": signals})
            previous["text"] = previous["text"].rstrip("\n") + "\n" + item["text"].lstrip("\n")
            previous["bbox"] = _r(_union([previous["bbox"], item["bbox"]]))
            previous["line_count"] += item["line_count"]
            previous["line_boxes"] = (previous.get("line_boxes") or []) + (item.get("line_boxes") or [])
            previous["members"] = previous.get("members", [previous["id"]]) + [item["id"]]
        else:
            out.append(dict(item))
    return out, merges


def paragraph_markdown(text: str, rtl: bool = False, discarded: list | None = None) -> tuple[str, list[dict]]:
    """Lines joined with spaces; list-marker lines become Markdown list items. Lines made only
    of bullet glyphs are formatting, not content: they are not rendered (recorded in `discarded`)."""
    segments: list[dict] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if bullet_only(line):
            if discarded is not None:
                discarded.append({"text": line.strip(), "reason": "formatting-only bullet marker"})
            continue
        match = list_match(line, rtl)
        if match:
            segments.append({"list": True, "marker": match[0], "lines": [match[1]]})
        elif segments:
            segments[-1]["lines"].append(line)
        else:
            segments.append({"list": False, "marker": None, "lines": [line]})
    rendered, items = [], []
    for segment in segments:
        body = " ".join(line.strip() for line in segment["lines"])
        if not segment["list"]:
            rendered.append(body)
            continue
        marker = segment["marker"]
        if marker in BULLET_GLYPHS or marker in {"-", "*"}:
            prefix = "- "  # the bullet glyph is shown as Markdown list syntax (recorded)
        elif re.fullmatch(r"[0-9]{1,3}[).]", marker):
            prefix = f"{marker} "  # CommonMark ordered list marker
        else:
            prefix = f"- {marker} "  # e.g. Arabic-Indic numbering kept as written
        items.append({"marker": marker, "rendered_prefix": prefix.strip()})
        rendered.append(prefix + body)
    separator = "\n" if any(segment["list"] for segment in segments) else "\n\n"
    return separator.join(rendered), items


def _figure_markdown(figure: dict, labels: list[str], transcription: dict | None) -> str:
    x0, y0, x1, y1 = figure["bbox"]
    icon, kind = ("🖼️", "Image") if figure["origin"] == "raster" else ("📊", "Figure")
    note = f"{x1 - x0:.0f}×{y1 - y0:.0f} pt, {figure.get('area_share', 0):.0%} of the page, at x={x0:.0f}, y={y0:.0f}"
    if figure.get("confidence") == "suspected":
        note += " · suspected chart region, needs review"
    children = figure.get("possible_child_regions") or []
    if children:
        note += f" · {len(children)} possible sub-figures (boundaries need review)"
    if transcription:  # Phase 2
        body = "\n".join(f"> {line}" for line in str(transcription["text"]).splitlines())
        out = f"> {icon} **{kind}** · {note}\n>\n{body}\n>\n> <sub>(read by {transcription['model']}; numbers need review)</sub>"
    else:
        out = f"> {icon} **{kind} (not read)** · {note}"
    if labels:
        out += "\n>\n" + "\n".join(f"> {line}" for label in labels for line in label.splitlines() if line.strip())
    return out


def assemble_page(elements: dict, *, decision: dict, library_page: dict, tables_doc: dict | None, vlm_text: str,
                  reorder_config: dict, transcriptions: dict | None = None,
                  previous_tables_doc: dict | None = None) -> dict:
    """Markdown for one page + the layout record (order, validation, review flags)."""
    settings = DEFAULTS | (elements.get("settings") or {})
    separator_chars = "".join(ch for ch in ar.ReorderConfig.from_dict(reorder_config).segment_separator
                              if not ch.isspace())
    rtl = float(library_page.get("arabic_letter_share") or 0.0) >= float(
        (reorder_config or {}).get("min_arabic_letter_share", ar.ReorderConfig().min_arabic_letter_share))
    final_text = decision.get("final_text") or ""
    mode = text_mode(decision, library_page)
    tables = list((tables_doc or {}).get("tables") or [])
    figures = list(elements.get("figures") or [])
    record: dict[str, Any] = {"page": elements["page"], "layout_version": LAYOUT_VERSION, "direction": "rtl" if rtl else "ltr",
                              "text_mode": mode, "final_source": decision.get("final_source"),
                              "text_status": decision.get("status"),
                              "figures": figures, "skipped_images": elements.get("skipped_images") or [],
                              "image_merges": elements.get("image_merges") or [],
                              "image_review": elements.get("image_review") or [],
                              "vector_exclusions": elements.get("vector_exclusions") or {},
                              "ignored_vector_regions": elements.get("ignored_vector_regions") or [],
                              "table_dedup": [], "chart_labels": {}, "ambiguous_regions": [], "captions": [],
                              "headings": [], "paragraph_merges": [], "list_items": [], "discarded_markers": [],
                              "side_by_side": [], "table_continuations": [], "structure_checks": {},
                              "validation": None,
                              "fallback_reason": None, "needs_review": []}
    record["needs_review"] += [f"images {', '.join(item['images'])}: {item['reason']}" for item in record["image_review"]]
    record["needs_review"] += [f"vector region at {fig['bbox']}: {fig['review']}" for fig in figures if fig.get("review")]

    def fallback(reason: str) -> dict:
        record["fallback_reason"] = reason
        record["needs_review"].append(f"layout fallback: {reason}")
        parts = [final_text.strip()] if final_text.strip() else []
        parts += [_figure_markdown(fig, [], (transcriptions or {}).get(f"p{elements['page']}_{fig['id']}"))
                  for fig in figures]
        record["order"] = ["page_text"] + [fig["id"] for fig in figures]
        record["tables_rendered"] = False
        record["tables_note"] = "tables are inside the page text; not rendered separately to avoid duplication"
        _layout_status(record)
        return {"markdown": "\n\n".join(parts) + "\n", "plain_text_layer": final_text, "record": record}

    if mode == "single_region":
        return fallback("the final text has no block geometry (it comes from the fallback model)")

    blocks = [dict(block, text=_block_text(block, mode, vlm_text)) for block in elements.get("blocks") or []]
    kept, labels_by_fig = [], {fig["id"]: [] for fig in figures if fig["origin"] == "vector"}
    for block in blocks:
        table = next((t for t in tables if _inside_share(block["bbox"], t["bbox"]) >= settings["table_overlap"]), None)
        if table is not None:
            record["table_dedup"].append({"block": block["id"], "table_index": table["table_index"]})
            continue
        chart = next((f for f in figures if f["origin"] == "vector"
                      and _inside_share(block["bbox"], f["bbox"]) >= settings["chart_overlap"]), None)
        if chart is not None:
            labels_by_fig[chart["id"]].append(block)
            continue
        kept.append(block)

    deduped = [b for b in blocks if any(d["block"] == b["id"] for d in record["table_dedup"])]
    labels = [b for group in labels_by_fig.values() for b in group]
    assembled = _nonspace("".join(b["text"] for b in kept + deduped + labels), separator_chars)
    reference = _nonspace(final_text, separator_chars)
    table_cells = _nonspace("".join(_table_text(t) for t in tables), separator_chars)
    table_blocks = _nonspace("".join(b["text"] for b in deduped), separator_chars)
    record["validation"] = {
        "method": "multiset of non-whitespace characters (whole elements may be reordered)",
        "ignored_separator_characters": separator_chars,
        "text_elements_match_page_text": assembled == reference,
        "missing_from_layout": sum((reference - assembled).values()),
        "extra_in_layout": sum((assembled - reference).values()),
        "table_cells_vs_deduplicated_blocks": {"identical": table_cells == table_blocks,
                                               "missing": sum((table_blocks - table_cells).values()),
                                               "extra": sum((table_cells - table_blocks).values())},
    }
    if assembled != reference:
        return fallback("block-level text does not reproduce the accepted page text "
                        f"({record['validation']['missing_from_layout']} missing, "
                        f"{record['validation']['extra_in_layout']} extra characters)")
    if not record["validation"]["table_cells_vs_deduplicated_blocks"]["identical"]:
        record["needs_review"].append("table cells and the replaced text blocks differ in characters (see validation)")

    body = float(elements.get("body_font_size") or 0)
    items: list[dict] = []
    for block in kept:
        if bullet_only(block["text"]):
            record["discarded_markers"].append({"element": block["id"], "text": block["text"].strip(),
                                                "reason": "isolated formatting-only bullet marker (not rendered)"})
            continue
        first_line = block["text"].strip().split("\n", 1)[0]
        is_list = bool(list_match(first_line, rtl))
        heading = body and not is_list and block["max_font_size"] >= settings["heading_ratio"] * body \
            and block["line_count"] <= 2 and len(block["text"]) <= 150
        level = 2 if heading and block["max_font_size"] >= settings["h2_ratio"] * body else 3
        items.append({"id": block["id"], "type": "heading" if heading else "paragraph", "bbox": block["bbox"],
                      "text": block["text"], "level": level, "line_count": block["line_count"],
                      "font_size": block.get("font_size") or block["max_font_size"], "bold": block.get("bold", False),
                      "line_boxes": block.get("line_boxes") or []})
    for table in tables:
        items.append({"id": f"t{table['table_index']}", "type": "table", "bbox": table["bbox"], "table": table})
    for fig in figures:
        items.append({"id": fig["id"], "type": "figure", "bbox": fig["bbox"], "figure": fig,
                      "labels": [b["text"] for b in labels_by_fig.get(fig["id"], [])]})
        record["chart_labels"][fig["id"]] = [b["id"] for b in labels_by_fig.get(fig["id"], [])]

    ambiguous: list[dict] = []
    ordered = xy_cut(items, rtl=rtl, settings=settings, ambiguous=ambiguous, side_by_side=record["side_by_side"])
    record["ambiguous_regions"] = ambiguous
    record["needs_review"] += [f"ambiguous reading order: {a['reason']} ({', '.join(a['elements'])})" for a in ambiguous]

    # Captions stay with their figure/table (moved right after, or before when above it).
    targets = [item for item in ordered if item["type"] in {"figure", "table"}]
    for item in list(ordered):
        if item["type"] != "paragraph" or item["line_count"] > 3 or not CAPTION_START.match(item["text"]):
            continue
        best = None
        for target in targets:
            tb, cb = target["bbox"], item["bbox"]
            overlap = max(0.0, min(tb[2], cb[2]) - max(tb[0], cb[0])) / max(1.0, min(tb[2] - tb[0], cb[2] - cb[0]))
            below, above = cb[1] - tb[3], tb[1] - cb[3]
            distance = below if 0 <= below <= settings["caption_gap"] else above if 0 <= above <= settings["caption_gap"] else None
            if overlap >= 0.5 and distance is not None and (best is None or distance < best[0]):
                best = (distance, target, "below" if distance == below else "above")
        if best:
            ordered.remove(item)
            position = ordered.index(best[1]) + (1 if best[2] == "below" else 0)
            ordered.insert(position, item)
            record["captions"].append({"caption": item["id"], "attached_to": best[1]["id"], "position": best[2]})
    ordered, record["paragraph_merges"] = merge_paragraphs(
        ordered, rtl=rtl, settings=settings, captions={c["caption"] for c in record["captions"]})
    for index, item in enumerate(ordered[:-1]):
        if item["type"] == "heading":
            record["headings"].append({"heading": item["id"], "followed_by": ordered[index + 1]["id"]})

    parts, plain = [], []
    for item in ordered:
        if item["type"] == "heading":
            parts.append("#" * item["level"] + " " + " ".join(item["text"].split()))
            plain.append(item["text"])
        elif item["type"] == "paragraph":
            markdown, list_items = paragraph_markdown(item["text"], rtl, record["discarded_markers"])
            parts.append(markdown)
            record["list_items"] += [entry | {"element": item["id"]} for entry in list_items]
            plain.append(item["text"])
        elif item["type"] == "table":
            continuation = table_continuation(
                item["table"], previous_tables_doc, page_width=float(elements.get("width") or 595),
                page_height=float(elements.get("height") or 842), settings=settings,
                first_on_page=item["table"]["table_index"] == min(t["table_index"] for t in tables))
            if continuation:
                record["table_continuations"].append({"table": item["id"]} | continuation)
                if not continuation["reliable"]:
                    record["needs_review"].append(
                        f"table {item['id']} may continue from the previous page ("
                        + "; ".join(continuation["uncertainty"]) + "): blank header row, no row promoted")
            table_md = _table_markdown(item["table"], continuation)
            if table_md:
                parts.append(table_md)
            plain.append(_table_text(item["table"]))
        else:
            key = f"p{elements['page']}_{item['id']}"
            parts.append(_figure_markdown(item["figure"], item["labels"], (transcriptions or {}).get(key)))
            plain += item["labels"]
        if item["type"] == "figure" and item["figure"].get("confidence") == "suspected":
            record["needs_review"].append(f"suspected vector chart region {item['id']} (review its boundary)")
        if item["type"] == "figure" and item["figure"].get("possible_child_regions"):
            record["needs_review"].append(f"raster image {item['id']} may contain several sub-figures")
    record["order"] = [item["id"] for item in ordered]
    record["order_boxes"] = [{"id": item["id"], "type": item["type"], "bbox": item["bbox"]} for item in ordered]
    record["tables_rendered"] = True
    markdown = "\n\n".join(part for part in parts if part) + "\n"
    record["structure_checks"] = structure_checks(markdown, ordered, rtl=rtl, settings=settings)
    for name, check in record["structure_checks"].items():
        if not check["ok"]:
            record["needs_review"].append(f"structure check failed: {name} ({check['detail']})")
    _layout_status(record)
    return {"markdown": markdown, "plain_text_layer": "\n".join(plain), "record": record}


def structure_checks(markdown: str, ordered: list[dict], *, rtl: bool, settings: dict) -> dict:
    """Checks on the final Markdown and order (layout quality only; text is checked separately)."""
    lines = markdown.splitlines()
    bullets = "".join(sorted(BULLET_GLYPHS))
    heading_bullets = [line for line in lines if re.match(rf"^#{{1,6}}\s*[{re.escape(bullets)}]", line)]
    empty_items = [line for line in lines if re.fullmatch(rf"\s*(?:[-*]|[0-9]{{1,3}}[.)])\s*[{re.escape(bullets)}]?\s*", line)]
    isolated = [line for line in lines if bullet_only(line)]
    wrong_order = []
    for a, b in zip(ordered, ordered[1:]):
        if _y_overlap(a["bbox"], b["bbox"]) < 0.5 or _inter(a["bbox"], b["bbox"]) > 0:
            continue
        a_centre, b_centre = (a["bbox"][0] + a["bbox"][2]) / 2, (b["bbox"][0] + b["bbox"][2]) / 2
        if (rtl and a_centre < b_centre) or (not rtl and a_centre > b_centre):
            wrong_order.append(f"{a['id']} before {b['id']}")
    table_headers, bad_headers = [], []
    for index, line in enumerate(lines[:-1]):
        if line.startswith("|") and re.fullmatch(r"\|( --- \|)+", lines[index + 1]):
            table_headers.append(line)
            cells = len(re.split(r"(?<!\\)\|", line.strip())) - 2
            if cells != lines[index + 1].count("---"):
                bad_headers.append(line)
    return {
        "side_by_side_order": {"ok": not wrong_order, "detail": ", ".join(wrong_order) or "consistent with direction"},
        "no_bullet_headings": {"ok": not heading_bullets, "detail": "; ".join(heading_bullets) or "none"},
        "no_empty_list_items": {"ok": not empty_items and not isolated,
                                "detail": "; ".join(empty_items + isolated) or "none"},
        "valid_table_headers": {"ok": not bad_headers,
                                "detail": (f"{len(table_headers)} header row(s); column count mismatch: "
                                           + "; ".join(bad_headers)) if bad_headers
                                else f"{len(table_headers)} header row(s), column counts match"},
    }


def _layout_status(record: dict) -> None:
    """Layout quality is separate from text extraction status and never 'reviewed' automatically."""
    record["layout_status"] = "Needs Review" if record["needs_review"] else "No issues detected (automatic)"
    record["layout_reviewed"] = False


def order_overlay_png(page, record: dict, dpi: int = 96) -> bytes:
    """Page image with numbered reading-order boxes (for visual inspection before approval)."""
    from PIL import Image, ImageDraw

    pix = page.get_pixmap(dpi=dpi)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples) if pix.n == 3 else \
        Image.frombytes("RGBA", (pix.width, pix.height), pix.samples).convert("RGB")
    draw = ImageDraw.Draw(image)
    scale = dpi / 72
    colors = {"paragraph": (31, 111, 235), "heading": (130, 80, 223), "table": (26, 127, 55), "figure": (207, 34, 46),
              "page_text": (154, 103, 0)}
    boxes = record.get("order_boxes") or [{"id": "page_text", "type": "page_text",
                                           "bbox": [0, 0, page.rect.width, page.rect.height]}]
    for number, box in enumerate(boxes, start=1):
        x0, y0, x1, y1 = (v * scale for v in box["bbox"])
        color = colors.get(box["type"], (0, 0, 0))
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.rectangle([x0, y0, x0 + 22, y0 + 14], fill=color)
        draw.text((x0 + 3, y0 + 1), str(number), fill=(255, 255, 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
