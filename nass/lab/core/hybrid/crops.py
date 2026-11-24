"""Crop options for Phase 2 region sets (Benchmark Mode). Local only: no network.

resolution "native": the page's single embedded raster is used exactly as stored (no re-render,
no re-encode) when it fully represents the visible page; a sub-region of such a page is cut from
the embedded image's own pixels (pixel box recorded). Otherwise the region is rendered at the
requested DPI (150 / 200 / 250) as PNG or JPEG (quality 90), and the fallback reason is recorded.
"auto" = native when eligible, else 250 DPI PNG.

A crop made with other options has other bytes (another SHA-256), so it is a new crop in a new
region set; existing sets are never changed.
"""

from __future__ import annotations

import io

from lab.core.e2.api.providers import ProviderSpec

RESOLUTIONS = ("auto", "native", 150, 200, 250)
FORMATS = ("auto", "png", "jpeg")
DEFAULT_DPI = 250
JPEG_QUALITY = 90
FULL_PAGE_TOLERANCE_PT = 2.0
LARGE_FILE_BYTES = 2 * 1024 * 1024
SMALL_REGION_SHARE = 0.25
TIMEOUT_SMALL_S = 60
TIMEOUT_LARGE_S = 180
SIZE_BASED_TIMEOUT_PROVIDERS = frozenset({"baidu-aistudio-paddleocr-vl-1.6"})  # others keep registry timeouts
LARGE_REGION_SHARE = 0.25  # rendered fallback above this share uses JPEG when the format is "auto"
TOLERANCE = 1e-6


def _zero(value: float) -> bool:
    """True for 0.0 and -0.0 alike (compared by magnitude, never by sign)."""
    return abs(float(value)) < TOLERANCE
JPEG_MAGIC = b"\xff\xd8\xff"


def covers(box, rect, tolerance: float = FULL_PAGE_TOLERANCE_PT) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(box, (rect.x0, rect.y0, rect.x1, rect.y1)))


def native_page_raster(document, page) -> tuple[dict | None, str | None]:
    """The page's single embedded raster when it fully represents the visible page, else (None, reason)."""
    images = page.get_images(full=True)
    if len(images) != 1:
        return None, f"{len(images)} embedded images on the page (native needs exactly one)"
    xref, smask = images[0][0], images[0][1]
    infos = page.get_image_info(xrefs=True)
    if len(infos) != 1:
        return None, "the embedded image is drawn more than once or not at all"
    info = infos[0]
    if not covers(info.get("bbox") or (0, 0, 0, 0), page.rect):
        return None, "the embedded image does not cover the whole page (within 2 pt)"
    if page.rotation:
        return None, f"the page is rotated ({page.rotation} degrees)"
    a, b, c, d = (tuple(info.get("transform") or (0, 0, 0, 0, 0, 0)) + (0, 0, 0, 0))[:4]
    # Page scaling only: off-diagonal terms are zero (-0.0 counts as 0) and both scales positive.
    if not (_zero(b) and _zero(c)) or a <= TOLERANCE or d <= TOLERANCE:
        return None, "the image is rotated, skewed or flipped (transform beyond page scaling)"
    if page.get_text("text").strip():
        return None, "the page has a text layer"
    if page.get_drawings():
        return None, "the page has vector drawings"
    if page.first_annot is not None or page.first_widget is not None:
        return None, "the page has annotations or form fields"
    if smask:
        return None, "the embedded image has a soft mask (transparency)"
    filter_type, filter_value = document.xref_get_key(xref, "Filter")
    if filter_type != "name" or filter_value != "/DCTDecode":
        if filter_value == "/JPXDecode":
            return None, "the embedded image is JPEG 2000 (JPX), not accepted by the providers"
        return None, (f"the embedded image is stored with filter {filter_value} (not plain JPEG); it cannot be "
                      "exported unchanged without re-encoding")
    if document.xref_get_key(xref, "Decode")[0] != "null":
        return None, "the embedded image has a Decode array (colour inversion)"
    extracted = document.extract_image(xref)
    if int(extracted.get("colorspace") or 0) not in (1, 3):
        return None, f"the embedded image has {extracted.get('colorspace')} colour components (CMYK or unknown)"
    raw = document.xref_stream_raw(xref)
    if not raw.startswith(JPEG_MAGIC):
        return None, "the embedded stream is not a JPEG file"
    width_pt = page.rect.width or 1.0
    return {"xref": xref, "bytes": raw, "width_px": int(extracted["width"]), "height_px": int(extracted["height"]),
            "bbox": [round(v, 2) for v in info["bbox"]], "colorspace_components": int(extracted["colorspace"]),
            "effective_dpi": round(int(extracted["width"]) / (width_pt / 72.0), 1)}, None


def _encode_pil(image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    if fmt == "jpeg":
        (image if image.mode in {"L", "RGB"} else image.convert("RGB")).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_crop(document, page, bbox: list[float], *, resolution="auto", fmt: str = "auto") -> tuple[bytes, dict]:
    """(bytes, provenance) for one region with the requested options."""
    import pymupdf

    if resolution not in RESOLUTIONS or fmt not in FORMATS:
        raise ValueError(f"Unsupported crop options: resolution={resolution!r}, format={fmt!r}")
    clip = pymupdf.Rect(*bbox) & page.rect
    if clip.is_empty:
        raise ValueError("the region is outside its page")
    full_page = covers(bbox, page.rect)
    native, reason = (None, None)
    if resolution in ("auto", "native"):
        native, reason = native_page_raster(document, page)
    meta = {"requested_resolution": resolution, "requested_format": fmt, "full_page_region": full_page,
            "fallback_reason": None, "pixel_box": None, "native_xref": None, "jpeg_quality": None,
            "re_encoded": True}
    if native is not None and full_page:
        data = native["bytes"]
        meta.update(resolution_mode="native", effective_dpi=native["effective_dpi"], format="jpeg",
                    format_note="original embedded JPEG, not re-encoded", pixel_size=[native["width_px"],
                                                                                     native["height_px"]],
                    native_xref=native["xref"], re_encoded=False)
    elif native is not None:
        from PIL import Image

        image = Image.open(io.BytesIO(native["bytes"]))
        x0, y0, x1, y1 = native["bbox"]
        sx, sy = native["width_px"] / max(x1 - x0, 1e-6), native["height_px"] / max(y1 - y0, 1e-6)
        box = [max(0, int(round((clip.x0 - x0) * sx))), max(0, int(round((clip.y0 - y0) * sy))),
               min(native["width_px"], int(round((clip.x1 - x0) * sx))),
               min(native["height_px"], int(round((clip.y1 - y0) * sy)))]
        out_fmt = "jpeg" if fmt == "jpeg" else "png"
        data = _encode_pil(image.crop(tuple(box)), out_fmt)
        meta.update(resolution_mode="native_subregion", effective_dpi=native["effective_dpi"], format=out_fmt,
                    format_note="cut from the embedded image's own pixels (not re-rendered)",
                    pixel_size=[box[2] - box[0], box[3] - box[1]], pixel_box=box, native_xref=native["xref"],
                    jpeg_quality=JPEG_QUALITY if out_fmt == "jpeg" else None)
    else:
        dpi = DEFAULT_DPI if resolution in ("auto", "native") else int(resolution)
        if resolution == "native" or (resolution == "auto" and reason):
            meta["fallback_reason"] = f"native not used: {reason}"
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        out_fmt = "jpeg" if fmt == "jpeg" else "png"
        page_area = (page.rect.width * page.rect.height) or 1.0
        share = (clip.width * clip.height) / page_area
        if fmt == "auto" and share > LARGE_REGION_SHARE:
            out_fmt = "jpeg"
            meta["format_reason"] = (f"format auto: rendered region covers {share:.0%} of the page (> "
                                     f"{LARGE_REGION_SHARE:.0%}) and no native image is usable, so JPEG quality "
                                     f"{JPEG_QUALITY} is used instead of PNG to keep the upload small")
        data = pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY) if out_fmt == "jpeg" else pix.tobytes("png")
        meta.update(resolution_mode="rendered", effective_dpi=dpi, format=out_fmt, format_note="rendered locally",
                    pixel_size=[pix.width, pix.height], jpeg_quality=JPEG_QUALITY if out_fmt == "jpeg" else None)
    meta["file_bytes"] = len(data)
    meta["large_file_warning"] = len(data) > LARGE_FILE_BYTES
    return data, meta


def crop_native_pixels(document, page, pixel_box: list[int]) -> tuple[bytes, dict]:
    """Cut an exact pixel box from the page's embedded raster (no re-render) as a lossless PNG.

    Raises ValueError when the page is not a single-raster scanned page (the reason is given)."""
    from PIL import Image

    native, reason = native_page_raster(document, page)
    if native is None:
        raise ValueError(f"no native page image: {reason}")
    x0, y0, x1, y1 = (int(v) for v in pixel_box)
    if not (0 <= x0 < x1 <= native["width_px"] and 0 <= y0 < y1 <= native["height_px"]):
        raise ValueError(f"pixel box {pixel_box} is outside the embedded image "
                         f"({native['width_px']}x{native['height_px']} px)")
    data = _encode_pil(Image.open(io.BytesIO(native["bytes"])).crop((x0, y0, x1, y1)), "png")
    return data, {
        "requested_resolution": "native", "requested_format": "png", "full_page_region": False,
        "fallback_reason": None, "resolution_mode": "native_subregion", "effective_dpi": native["effective_dpi"],
        "format": "png", "format_note": "lossless copy of the embedded image's own pixels (not re-rendered)",
        "pixel_size": [x1 - x0, y1 - y0], "pixel_box": [x0, y0, x1, y1], "native_xref": native["xref"],
        "native_image_px": [native["width_px"], native["height_px"]], "jpeg_quality": None, "re_encoded": True,
        "file_bytes": len(data), "large_file_warning": len(data) > LARGE_FILE_BYTES,
    }


def extension(meta: dict) -> str:
    return "jpg" if meta.get("format") == "jpeg" else "png"


def legacy_options(region: dict) -> dict:
    """Options of a crop made before crop options existed (rendered PNG at the set DPI)."""
    return {"resolution_mode": "rendered", "effective_dpi": region.get("crop_dpi"), "format": "png",
            "jpeg_quality": None, "file_bytes": region.get("crop_bytes"), "pixel_size": region.get("crop_px"),
            "re_encoded": True, "legacy": True, "fallback_reason": None,
            "large_file_warning": bool((region.get("crop_bytes") or 0) > LARGE_FILE_BYTES)}


def region_options(region: dict) -> dict:
    return dict(region.get("crop_options") or legacy_options(region))


def size_based_timeout(region: dict) -> int:
    """60 s below 25% of the page area, 180 s otherwise (unknown size counts as large)."""
    share = region.get("area_share")
    return TIMEOUT_SMALL_S if isinstance(share, (int, float)) and share < SMALL_REGION_SHARE else TIMEOUT_LARGE_S


def default_timeout(region: dict, spec: ProviderSpec | None = None, fallback_s: float | None = None) -> float:
    """Default request timeout: size-based for Baidu AI Studio only; every other provider keeps its
    registry timeout (`spec.limits.request_timeout_s`, or `fallback_s` for the Qari endpoint)."""
    if spec is not None and spec.id in SIZE_BASED_TIMEOUT_PROVIDERS:
        return float(size_based_timeout(region))
    if spec is not None:
        return float(spec.limits.request_timeout_s)
    if fallback_s is not None:
        return float(fallback_s)
    raise ValueError("No default timeout: give the provider spec or the endpoint's timeout")


def checked_timeout(spec: ProviderSpec | None, value: float) -> float:
    """A per-request timeout override, bounded by the provider registry rules."""
    value = float(value)
    if not 10 <= value <= 600:
        raise ValueError("The request timeout must be between 10 and 600 seconds")
    if spec is not None and spec.pricing.basis == "per_compute_second":
        bound = float(spec.pricing.max_billable_seconds_per_request or 0)
        if value > bound:
            raise ValueError(f"{spec.provider}: the timeout cannot exceed the billing bound of {bound:.0f} s "
                             "used for the cost check")
    return value


def with_timeout(spec: ProviderSpec, timeout_s: float) -> ProviderSpec:
    """A copy of the spec with another request timeout (the registry entry is unchanged)."""
    if float(spec.limits.request_timeout_s) == float(timeout_s):
        return spec
    return spec.model_copy(update={"limits": spec.limits.model_copy(update={"request_timeout_s": float(timeout_s)})})
