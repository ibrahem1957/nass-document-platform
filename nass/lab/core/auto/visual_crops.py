"""Crops of DIGITAL / MIXED page visuals for Qari: native image bytes only when provably safe, otherwise
the complete visible region rendered from the PDF page at 250 DPI.

Why: Word (and similar) exports often draw one visible figure as several PDF objects: a low-resolution
shadow/frame image behind the chart, the chart image itself, vector frames, and axis labels or legends
as separate text. Layout v3 merges a frame with the image inside it and gives the kept figure the
COMBINED box, which is exactly the shadow's box, so picking "the embedded image at this box" can return
the shadow (a dark, nearly uniform rectangle) instead of the chart.

Native extraction (crop method `native_embedded_image`) is used only when ALL checks pass:
  1. exactly one embedded image intersects the visual region (any second layer, e.g. a shadow, fails);
  2. no vector drawing and no native text word overlaps the region;
  3. the image's displayed box covers the region within `native_bbox_tolerance_pt` (default 2 pt);
  4. effective resolution >= `native_min_dpi` (default 200) in BOTH directions;
  5. plain placement (no rotation/skew/flip, no soft mask);
  6. the figure was not frame-merged by layout v3 and contains no other figure;
  7. the decoded pixels pass the quality guard (not nearly uniform / flat background).
Otherwise the region is rendered (`rendered_page_region`) at `digital_render_dpi` (250, proven setting):
the complete visible composition, all layers in their PDF draw order, as lossless PNG, aspect ratio
kept, no resizing. The region is the layout v3 box, extended only by layers that mostly lie inside it
(shadows, frames) and never into another figure. Draw order is recorded but not needed for the
decision: any second layer forces rendering. A rejected native candidate is saved for audit.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

CROP_VERSION = "1"
NATIVE, RENDERED = "native_embedded_image", "rendered_page_region"
LAYER_INSIDE_SHARE = 0.5      # a layer belongs to a visual when >= 50% of it lies inside the visual's box
WORD_INSIDE_SHARE = 0.5       # a native text word overlaps the region when >= 50% of it lies inside
PAGE_BACKGROUND_SHARE = 0.9   # drawings covering >= 90% of the page are page backgrounds, ignored
UNIFORM_DOMINANT_SHARE = 0.999  # >= 99.9% of pixels at one tone: blank
TOLERANCE = 1e-6


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inter(a, b) -> float:
    return _area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def _inside_share(inner, outer) -> float:
    return _inter(inner, outer) / (_area(inner) or 1.0)


def _union(boxes) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _r(box) -> list[float]:
    return [round(float(v), 2) for v in box]


# ------------------------------------------------------------------ page layers


def page_layers(page) -> dict[str, Any]:
    """Embedded images (with xref, size, transform, draw number), vector drawing boxes and text words."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    rasters = []
    for index, info in enumerate(page.get_image_info(xrefs=True)):
        box = [float(v) for v in info.get("bbox") or (0, 0, 0, 0)]
        rasters.append({"index": index, "number": info.get("number"), "xref": info.get("xref"), "bbox": _r(box),
                        "width_px": info.get("width"), "height_px": info.get("height"),
                        "transform": list(info.get("transform") or ()), "has_mask": bool(info.get("has-mask"))})
    drawings = []
    try:
        for index, path in enumerate(page.get_drawings()):
            rect = path.get("rect")
            if rect is None:
                continue
            box = [rect.x0, rect.y0, rect.x1, rect.y1]
            if _area(box) >= PAGE_BACKGROUND_SHARE * area:
                continue
            drawings.append({"index": index, "seqno": path.get("seqno"), "bbox": _r(box),
                             "fill": path.get("fill") is not None})
    except Exception:  # noqa: BLE001 - drawings are evidence only; without them native stays possible
        drawings = []
    words = [[w[0], w[1], w[2], w[3]] for w in page.get_text("words")]
    return {"rasters": rasters, "drawings": drawings, "words": words, "page_rect": _r(list(page.rect))}


def composition_region(figure: dict, layers: dict, other_figures: list[dict]) -> tuple[list[float], dict]:
    """The figure box extended by layers (images, drawings) that mostly lie inside it, never into another
    figure. Returns (region, record)."""
    box = [float(v) for v in figure["bbox"]]
    extra = []
    for kind in ("rasters", "drawings"):
        for layer in layers[kind]:
            if _inside_share(layer["bbox"], box) >= LAYER_INSIDE_SHARE and _inside_share(layer["bbox"], box) < 1.0:
                extra.append({"kind": kind[:-1], "index": layer["index"], "bbox": layer["bbox"]})
    region = _union([box] + [e["bbox"] for e in extra]) if extra else box
    page_rect = layers["page_rect"]
    region = [max(region[0], page_rect[0]), max(region[1], page_rect[1]),
              min(region[2], page_rect[2]), min(region[3], page_rect[3])]
    record = {"figure_bbox": _r(box), "extended_by": extra, "ambiguous": False}
    clash = [f["id"] for f in other_figures if _inter(f["bbox"], region) > _inter(f["bbox"], box) + TOLERANCE]
    if clash:  # extending would reach into another figure: keep the layout box and flag for review
        record.update(ambiguous=True, reason=f"extending over its layers would overlap figure(s) {clash}; the "
                                             "layout v3 box is kept (review the boundary)", extended_by=[])
        region = box
    return _r(region), record


# ------------------------------------------------------------------ quality guard


def quality(image, *, min_std: float) -> dict:
    """Pixel statistics; `suspicious` only for (nearly) blank or flat crops. Few colours alone is fine."""
    from PIL import ImageStat  # noqa: PLC0415

    gray = image.convert("L")
    sample = gray if gray.width * gray.height <= 4_000_000 else gray.resize((gray.width // 4 or 1, gray.height // 4 or 1))
    std = float(ImageStat.Stat(sample).stddev[0])
    histogram = sample.histogram()
    total = sum(histogram) or 1
    dominant = max(sum(histogram[max(0, i - 4): i + 5]) for i in range(0, 256, 2)) / total
    reasons = []
    if std < min_std:
        reasons.append(f"nearly uniform pixels (luminance std {std:.2f} < {min_std})")
    if dominant >= UNIFORM_DOMINANT_SHARE:
        reasons.append(f"{dominant:.2%} of the pixels share one tone (flat background only)")
    return {"luminance_std": round(std, 3), "dominant_tone_share": round(dominant, 5),
            "mean_luminance": round(float(ImageStat.Stat(sample).mean[0]), 2),
            "suspicious": bool(reasons), "reasons": reasons}


# ------------------------------------------------------------------ eligibility


def _plain(transform: list[float]) -> bool:
    a, b, c, d = (list(transform) + [0, 0, 0, 0])[:4]
    return abs(b) < TOLERANCE and abs(c) < TOLERANCE and a > TOLERANCE and d > TOLERANCE


def assess_native(region: list[float], layers: dict, *, merged: bool, contains_figures: list[str],
                  tolerance_pt: float, min_dpi: float) -> dict:
    """Geometric eligibility of native image bytes for this region (pixel quality is checked after)."""
    touching = [r for r in layers["rasters"] if _inter(r["bbox"], region) > TOLERANCE]
    drawings = [d for d in layers["drawings"] if _inter(d["bbox"], region) > TOLERANCE]
    words = [w for w in layers["words"] if _inside_share(w, region) >= WORD_INSIDE_SHARE]
    failures = []
    if merged:
        failures.append("frame-merged by layout v3 (a frame/shadow layer plus the image inside it): always rendered")
    if contains_figures:
        failures.append(f"contains other figure(s) {contains_figures}: one visual made of several objects")
    if len(touching) != 1:
        failures.append(f"{len(touching)} embedded images intersect the region (layers: "
                        + ", ".join(f"#{r['index']} {r['width_px']}x{r['height_px']} px at {r['bbox']}" for r in touching)
                        + ")")
    if drawings:
        failures.append(f"{len(drawings)} vector drawing(s) overlap the region (frame, shadow or chart parts)")
    if words:
        failures.append(f"{len(words)} native text word(s) lie on the region (labels/legend stored as PDF text)")
    # With several layers, the reported candidate is the one a box-matching lookup would pick (often the
    # shadow behind the chart), so the audit shows exactly what native extraction would have sent.
    at_box = [r for r in touching if all(abs(a - b) <= tolerance_pt for a, b in zip(r["bbox"], region))]
    candidate = touching[0] if len(touching) == 1 else (at_box[0] if at_box else (
        max(touching, key=lambda r: (r["width_px"] or 0) * (r["height_px"] or 0)) if touching else None))
    native: dict[str, Any] = {}
    if candidate is not None:
        width_pt = (candidate["bbox"][2] - candidate["bbox"][0]) or 1e-6
        height_pt = (candidate["bbox"][3] - candidate["bbox"][1]) or 1e-6
        dpi_x = (candidate["width_px"] or 0) / (width_pt / 72.0)
        dpi_y = (candidate["height_px"] or 0) / (height_pt / 72.0)
        native = {"xref": candidate["xref"], "layer_index": candidate["index"], "draw_number": candidate["number"],
                  "width_px": candidate["width_px"], "height_px": candidate["height_px"],
                  "displayed_bbox": candidate["bbox"], "displayed_pt": [round(width_pt, 2), round(height_pt, 2)],
                  "effective_dpi": [round(dpi_x, 1), round(dpi_y, 1)]}
        if len(touching) == 1:
            if any(abs(a - b) > tolerance_pt for a, b in zip(candidate["bbox"], region)):
                failures.append(f"the image box {candidate['bbox']} does not cover the visual region {region} "
                                f"within {tolerance_pt} pt")
            if min(dpi_x, dpi_y) < min_dpi:
                failures.append(f"effective resolution {dpi_x:.0f}x{dpi_y:.0f} DPI is below {min_dpi:.0f} DPI")
            if not _plain(candidate["transform"]):
                failures.append("the image is rotated, skewed or flipped")
            if candidate["has_mask"]:
                failures.append("the image has a soft mask (transparency over other content)")
            if not candidate["xref"]:
                failures.append("the image is inline (no xref)")
    else:
        failures.append("no embedded image in the region")
    return {"eligible": not failures, "failures": failures, "native": native,
            "layers": {"images": [{k: r[k] for k in ("index", "number", "bbox", "width_px", "height_px")} for r in touching],
                       "drawings": len(drawings), "text_words": len(words)}}


# ------------------------------------------------------------------ crops


def render_region(page, region: list[float], dpi: int):
    """The visible page content inside `region` at `dpi` (all layers, lossless), as a PIL image."""
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    clip = pymupdf.Rect(*region) & page.rect
    pix = page.get_pixmap(clip=clip, dpi=int(dpi), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    width_pt, height_pt = clip.width or 1e-6, clip.height or 1e-6
    return image, {"clip_bbox": _r([clip.x0, clip.y0, clip.x1, clip.y1]), "render_dpi": int(dpi),
                   "effective_dpi": [round(pix.width / (width_pt / 72.0), 1), round(pix.height / (height_pt / 72.0), 1)],
                   "pixel_size": [pix.width, pix.height]}


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def figure_crop(document, page_number: int, figure: dict, *, layers: dict, other_figures: list[dict],
                merged: bool, contains_figures: list[str], render_dpi: int, min_dpi: float, tolerance_pt: float,
                min_std: float) -> tuple[bytes, dict, bytes | None]:
    """(PNG bytes, provenance, rejected native candidate PNG or None) for one digital/mixed visual."""
    from lab.core.auto import images  # noqa: PLC0415

    page = document[page_number - 1]
    region, composition = composition_region(figure, layers, other_figures)
    meta: dict[str, Any] = {"crop_version": CROP_VERSION, "page": page_number, "figure_id": figure["id"],
                            "figure_origin": figure.get("origin"), "original_bbox": _r(figure["bbox"]),
                            "composition": composition, "configured_render_dpi": int(render_dpi),
                            "native_min_dpi": min_dpi, "bbox_tolerance_pt": tolerance_pt}
    candidate_png = None
    if figure.get("origin") == "raster":
        assessment = assess_native(region, layers, merged=merged, contains_figures=contains_figures,
                                   tolerance_pt=tolerance_pt, min_dpi=min_dpi)
        meta["native_assessment"] = assessment
        if assessment["eligible"]:
            native_image = images._decode_xref(document, assessment["native"]["xref"])
            check = quality(native_image, min_std=min_std)
            meta["native_quality"] = check
            if not check["suspicious"]:
                data = _png(native_image)
                meta.update(method=NATIVE, method_reason="all native-eligibility checks passed",
                            crop_bbox=assessment["native"]["displayed_bbox"], pixel_size=[native_image.width,
                                                                                          native_image.height],
                            effective_dpi=assessment["native"]["effective_dpi"], render_dpi=None,
                            quality=check, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), format="png",
                            native_pixels=True, resized=None)
                return data, meta, None
            assessment["failures"].append("quality guard: " + "; ".join(check["reasons"]))
            candidate_png = _png(native_image)  # kept for audit
        elif assessment["native"].get("xref"):
            try:  # the rejected native layer, for audit (e.g. the shadow that was picked before)
                candidate_png = _png(images._decode_xref(document, assessment["native"]["xref"]))
            except Exception:  # noqa: BLE001
                candidate_png = None
        meta["method_reason"] = "native rejected: " + "; ".join(assessment["failures"])
    else:
        meta["method_reason"] = "vector figure: always rendered"
    image, rendered = render_region(page, region, render_dpi)
    check = quality(image, min_std=min_std)
    data = _png(image)
    meta.update(method=RENDERED, crop_bbox=rendered["clip_bbox"], pixel_size=rendered["pixel_size"],
                effective_dpi=rendered["effective_dpi"], render_dpi=rendered["render_dpi"], quality=check,
                review=(["rendered crop looks blank or flat: " + "; ".join(check["reasons"])] if check["suspicious"]
                        else []) + ([composition["reason"]] if composition.get("ambiguous") else []),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), format="png", native_pixels=False,
                resized=None)
    return data, meta, candidate_png


def contained_figures(figures: list[dict], *, share: float = 0.9) -> dict[str, str]:
    """{inner figure id: outer figure id} for figures lying >= 90% inside a larger figure (layers of one
    visual, e.g. a chart inside its frame); the outer one is processed as one composite."""
    out = {}
    for inner in figures:
        holders = [o for o in figures if o is not inner and _area(o["bbox"]) > _area(inner["bbox"])
                   and _inside_share(inner["bbox"], o["bbox"]) >= share]
        if holders:
            out[inner["id"]] = min(holders, key=lambda o: _area(o["bbox"]))["id"]
    return out
