"""Per-page router: DIGITAL / SCANNED / MIXED, reusing the Phase 1 quality checks.

The Phase 1 library pass (`hybrid.analysis.analyze_pdf`: PyMuPDF raw + processed text, Needs OCR
diagnostics, decision categories A/B/C, find_tables) is run unchanged; the router only adds image
geometry and text-layer visibility, then applies the first matching rule. Every rule and its evidence
are recorded. No model is loaded and nothing is sent.

Rules (first match wins):
  S1 page-sized image (>= PAGE_IMAGE_SHARE of the page) and no text layer           -> SCANNED
  S2 page-sized image and the text layer is unreliable (category B or C)             -> SCANNED
  S3 page-sized image and the text layer is invisible (OCR layer, render mode 3)     -> SCANNED
     (an OCR layer of unknown origin is never treated as native text)
  S4 no page-sized image, category C (text layer missing or garbled)                 -> SCANNED (rendered)
  M1 page-sized image under a reliable visible text layer                            -> MIXED
  M2 reliable text layer and raster images covering >= LARGE_IMAGE_SHARE             -> MIXED
  D1 reliable text layer (category A or B), no large raster image                    -> DIGITAL
Category B (Arabic words split by spaces) stays DIGITAL/MIXED: native text is kept and the proven
`library_rejoined` step uses a full-page Qari transcription only as spacing guidance.
"""

from __future__ import annotations

from typing import Any

from lab.core.hybrid import analysis as hybrid_analysis

DIGITAL, SCANNED, MIXED = "DIGITAL", "SCANNED", "MIXED"
ROUTER_VERSION = "1"
PAGE_IMAGE_SHARE = 0.85      # an image covering >= 85% of the page is a page image
LARGE_IMAGE_SHARE = hybrid_analysis.LARGE_IMAGE_AREA_SHARE  # 5%, the Phase 1 "large raster image" rule
EMBEDDED_SCAN_SHARE = 0.5    # on MIXED pages, a raster >= 50% of the page gets layout detection inside it
INVISIBLE_TEXT_SHARE = 0.9   # >= 90% of the text characters invisible -> OCR layer


def text_visibility(page) -> dict[str, Any]:
    """Characters drawn visibly vs invisibly (render mode 3 = typical OCR layer)."""
    visible = invisible = 0
    try:
        for trace in page.get_texttrace():
            count = len(trace.get("chars") or ())
            if int(trace.get("type", 0)) == 3 or float(trace.get("opacity", 1.0)) == 0.0:
                invisible += count
            else:
                visible += count
    except Exception as exc:  # noqa: BLE001 - informational; treated as unknown
        return {"visible_chars": None, "invisible_chars": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"visible_chars": visible, "invisible_chars": invisible, "error": None}


def classify_page(analysis_page: dict, visibility: dict[str, Any]) -> dict[str, Any]:
    """Route one page from its Phase 1 analysis record and text visibility. Pure function."""
    category = analysis_page["category"]
    images = (analysis_page.get("images") or {}).get("images") or []
    page_images = [image for image in images if image["area_share"] >= PAGE_IMAGE_SHARE]
    large = [image for image in images if image["area_share"] >= LARGE_IMAGE_SHARE]
    raw_text = (analysis_page.get("raw_text") or "").strip()
    visible, invisible = visibility.get("visible_chars"), visibility.get("invisible_chars")
    total = (visible or 0) + (invisible or 0)
    invisible_layer = bool(total) and (invisible or 0) / total >= INVISIBLE_TEXT_SHARE
    evidence = {
        "phase1_category": category, "phase1_reason": analysis_page.get("category_reason"),
        "library_error": analysis_page.get("library_error"), "text_chars": len(raw_text),
        "visible_chars": visible, "invisible_chars": invisible, "invisible_text_layer": invisible_layer,
        "page_image_share": max((image["area_share"] for image in images), default=0.0),
        "page_images": len(page_images), "large_images": len(large), "images": len(images),
        "table_count": analysis_page.get("table_count", 0),
        "arabic_letter_share": analysis_page.get("arabic_letter_share"),
        "thresholds": {"page_image_share": PAGE_IMAGE_SHARE, "large_image_share": LARGE_IMAGE_SHARE,
                       "invisible_text_share": INVISIBLE_TEXT_SHARE},
    }
    unreliable = category in {hybrid_analysis.CATEGORY_B, hybrid_analysis.CATEGORY_C}
    if page_images and not raw_text:
        route, rule, why = SCANNED, "S1", "page-sized image and no text layer"
    elif page_images and unreliable:
        route, rule, why = SCANNED, "S2", f"page-sized image; text layer unreliable (Phase 1 category {category})"
    elif page_images and invisible_layer:
        route, rule, why = SCANNED, "S3", "page-sized image under an invisible (OCR) text layer of unknown origin"
    elif category == hybrid_analysis.CATEGORY_C:
        route, rule, why = SCANNED, "S4", "no usable text layer (Phase 1 category C); the page is rendered"
    elif page_images:
        route, rule, why = MIXED, "M1", "reliable visible text layer over a page-sized image"
    elif large:
        route, rule, why = MIXED, "M2", f"reliable text layer with {len(large)} large raster image(s)"
    else:
        route, rule, why = DIGITAL, "D1", f"reliable text layer (Phase 1 category {category}), no large raster image"
    embedded_scans = [image for image in large if image["area_share"] >= EMBEDDED_SCAN_SHARE] if route == MIXED else []
    return {"page": analysis_page["page"], "route": route, "rule": rule, "reason": why, "evidence": evidence,
            "category_b_rejoin": route != SCANNED and category == hybrid_analysis.CATEGORY_B,
            "embedded_scan_regions": [image["bbox"] for image in embedded_scans],
            "router_version": ROUTER_VERSION}


def route_document(document, analysis: dict) -> list[dict[str, Any]]:
    """Route every page. `document` is an open PyMuPDF document of the same PDF as `analysis`."""
    if len(document) != analysis["page_count"]:
        raise ValueError("the analysis does not match this PDF")
    return [classify_page(page, text_visibility(document[page["page"] - 1])) for page in analysis["pages"]]


def counts(routes: list[dict]) -> dict[str, int]:
    return {name: sum(1 for item in routes if item["route"] == name) for name in (DIGITAL, SCANNED, MIXED)}
