"""Page and region images at native pixels (no resizing unless explicitly configured).

Order of preference, recorded as `method` on every image:
1. native_jpeg      - the page's single embedded JPEG, bytes unchanged (crops.native_page_raster);
2. native_decoded   - a single embedded raster covering the page (any filter), decoded to its own
                      pixel grid with PyMuPDF (no re-render, no scaling);
3. rendered         - the page rendered at the configured DPI (pages without a usable raster; e.g.
                      garbled text layers or vector-only pages). Recorded with the DPI.
Crops are cut from that pixel grid with the proven margin (60 px, clamped) and encoded as lossless PNG.
Downscaling happens only when `max_image_pixels` is configured; the original and new sizes are recorded.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

from lab.core.hybrid import crops as hybrid_crops

PAGE_IMAGE_SHARE = 0.85
TOLERANCE = 1e-6


@dataclass
class PageImage:
    image: Any                      # PIL.Image (RGB)
    bbox_pt: list[float]            # the page area this pixel grid covers
    method: str
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> list[int]:
        return [self.image.width, self.image.height]


def _plain_scale(info: dict) -> bool:
    a, b, c, d = (tuple(info.get("transform") or (0, 0, 0, 0, 0, 0)) + (0, 0, 0, 0))[:4]
    return abs(b) < TOLERANCE and abs(c) < TOLERANCE and a > TOLERANCE and d > TOLERANCE


def _decode_xref(document, xref: int):
    import pymupdf  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    pix = pymupdf.Pixmap(document, xref)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.n - pix.alpha not in (1, 3):  # CMYK or other: convert colour space only (same pixel grid)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    mode = "L" if pix.n == 1 else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")


def raster_infos(page) -> list[dict]:
    """Embedded rasters drawn on the page with their xref, bbox and page-area share."""
    area = abs(page.rect.width * page.rect.height) or 1.0
    found = []
    for info in page.get_image_info(xrefs=True):
        x0, y0, x1, y1 = info.get("bbox") or (0, 0, 0, 0)
        found.append(info | {"area_share": max(0.0, (x1 - x0) * (y1 - y0)) / area})
    return found


def page_image(document, page_number: int, *, render_dpi: int) -> PageImage:
    """The scanned page's pixel grid (see the module docstring for the order of preference)."""
    from PIL import Image  # noqa: PLC0415

    page = document[page_number - 1]
    native, reason = hybrid_crops.native_page_raster(document, page)
    if native is not None:
        image = Image.open(io.BytesIO(native["bytes"])).convert("RGB")
        return PageImage(image, list(native["bbox"]), "native_jpeg",
                         {"xref": native["xref"], "effective_dpi": native["effective_dpi"]})
    covering = [info for info in raster_infos(page) if info["area_share"] >= PAGE_IMAGE_SHARE and info.get("xref")]
    if len(covering) == 1 and not page.rotation and _plain_scale(covering[0]):
        info = covering[0]
        try:
            image = _decode_xref(document, info["xref"])
            width_pt = (info["bbox"][2] - info["bbox"][0]) or 1.0
            return PageImage(image, [round(v, 2) for v in info["bbox"]], "native_decoded",
                             {"xref": info["xref"], "native_reason_not_jpeg": reason,
                              "effective_dpi": round(image.width / (width_pt / 72.0), 1)})
        except Exception as exc:  # noqa: BLE001 - recorded; the page is rendered instead
            reason = f"{reason}; decoding the embedded raster failed: {type(exc).__name__}: {exc}"
    pix = page.get_pixmap(dpi=int(render_dpi), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    rect = page.rect
    return PageImage(image, [rect.x0, rect.y0, rect.x1, rect.y1], "rendered",
                     {"dpi": int(render_dpi), "native_reason": reason})


def embedded_image(document, page_number: int, bbox_pt: list[float]) -> PageImage | None:
    """Native pixels of the embedded raster drawn at `bbox_pt` on a MIXED page (None if not decodable)."""
    page = document[page_number - 1]
    matches = [info for info in raster_infos(page) if info.get("xref")
               and all(abs(float(a) - float(b)) <= 2.0 for a, b in zip(info["bbox"], bbox_pt))]
    if len(matches) != 1:  # several layers at this box (e.g. a shadow behind the image): never guess one
        return None
    for info in matches:
        if page.rotation or not _plain_scale(info):
            return None
        try:
            image = _decode_xref(document, info["xref"])
        except Exception:  # noqa: BLE001
            return None
        return PageImage(image, [round(v, 2) for v in info["bbox"]], "native_decoded", {"xref": info["xref"]})
    return None


def crop_png(page_image: PageImage, pixel_box: list[int], *, max_image_pixels: int | None = None) -> tuple[bytes, dict]:
    """Lossless PNG of an exact pixel box; downscaled only when `max_image_pixels` is set and exceeded."""
    x0, y0, x1, y1 = (int(v) for v in pixel_box)
    width, height = page_image.size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"pixel box {pixel_box} is outside the {width}x{height} px page image")
    crop = page_image.image.crop((x0, y0, x1, y1))
    return encode_png(crop, max_image_pixels=max_image_pixels,
                      meta={"pixel_box": [x0, y0, x1, y1], "page_image_method": page_image.method,
                            "page_image_px": [width, height], "page_image_bbox_pt": page_image.bbox_pt})


def encode_png(image, *, max_image_pixels: int | None = None, meta: dict | None = None) -> tuple[bytes, dict]:
    meta = dict(meta or {})
    original = [image.width, image.height]
    resized = None
    if max_image_pixels and image.width * image.height > max_image_pixels:
        scale = (max_image_pixels / float(image.width * image.height)) ** 0.5
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size)
        resized = {"from_px": original, "to_px": list(size), "max_image_pixels": max_image_pixels,
                   "reason": "explicit max_image_pixels setting"}
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()
    meta.update(format="png", pixel_size=[image.width, image.height], native_pixels=resized is None,
                resized=resized, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return data, meta
