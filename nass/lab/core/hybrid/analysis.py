"""Library pass of Hybrid PDF Extraction (local only: no network, no model).

Reuses the Local PDF Extraction Benchmark unchanged: PyMuPDF raw text, the processed
method (`pymupdf_ar_reordered` with its gating and diagnostics) and `find_tables()`.
The decision is made per page:

A  library text is usable: no "Needs OCR?" flag and reorder status reconstructed /
   not_needed / not_applicable.
B  "Needs OCR?" only because Arabic words are split by spaces, with a real text layer:
   send to the fallback model, then try the `library_rejoined` candidate.
C  no useful text layer, garbled glyphs, or an uncertain reconstruction: the model
   output becomes the primary candidate.

Text quantity is never taken as proof of correctness; every reason is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from lab.local_pdf_extraction import arabic_reorder, diagnostics, extractors, tables

ANALYSIS_VERSION = "1"
CATEGORY_A, CATEGORY_B, CATEGORY_C = "A", "B", "C"
CATEGORY_LABELS = {
    CATEGORY_A: "A - library text usable",
    CATEGORY_B: "B - Arabic words split by spaces",
    CATEGORY_C: "C - text layer missing, garbled or unreliable",
}
IMAGE_WARNING = "Contains raster images; text inside images may not have been extracted"
LARGE_IMAGE_AREA_SHARE = 0.05  # a raster image covering >= 5% of the page


def default_settings() -> dict:
    return {
        "reorder": arabic_reorder.ReorderConfig().to_dict(),
        "tables": {"enabled": True, "version": tables.TABLES_VERSION},
        "diagnostics": {"version": diagnostics.DIAGNOSTICS_VERSION,
                        "single_letter_threshold": diagnostics.DEFAULT_SINGLE_LETTER_THRESHOLD,
                        "single_letter_min_arabic_tokens": diagnostics.DEFAULT_SINGLE_LETTER_MIN_TOKENS,
                        "applies_to": "processed method"},
        "large_image_area_share": LARGE_IMAGE_AREA_SHARE,
    }


def _images(page) -> dict:
    area = abs(page.rect.width * page.rect.height) or 1.0
    found = []
    for info in page.get_image_info():
        x0, y0, x1, y1 = info.get("bbox") or (0, 0, 0, 0)
        share = max(0.0, (x1 - x0) * (y1 - y0)) / area
        found.append({"bbox": [round(v, 1) for v in (x0, y0, x1, y1)], "area_share": round(share, 4),
                      "width_px": info.get("width"), "height_px": info.get("height")})
    large = [item for item in found if item["area_share"] >= LARGE_IMAGE_AREA_SHARE]
    try:
        drawings = len(page.get_drawings())
    except Exception:  # noqa: BLE001 - informational only
        drawings = None
    return {"count": len(found), "large_count": len(large), "images": found, "vector_drawings": drawings,
            "warning": IMAGE_WARNING if large else None}


def decide(reorder: dict, indicators: dict, *, extraction_error: str | None = None) -> dict:
    status = (reorder or {}).get("reorder_status")
    reason = indicators.get("needs_ocr_reason") or ""
    reasons = [item for item in reason.split("; ") if item]
    if extraction_error:
        return {"category": CATEGORY_C, "reason": f"library extraction failed: {extraction_error}"}
    no_text = "no text layer content" in reasons
    garbage = any("garbage" in item for item in reasons)
    if diagnostics.SPLIT_WORDS_REASON in reasons and not no_text and not garbage:
        return {"category": CATEGORY_B, "reason": f"Needs OCR: {diagnostics.SPLIT_WORDS_REASON}"}
    if indicators.get("needs_ocr"):
        return {"category": CATEGORY_C, "reason": f"Needs OCR: {reason}"}
    if status == "uncertain":
        return {"category": CATEGORY_C,
                "reason": f"processed reconstruction is uncertain ({(reorder or {}).get('decision')})"}
    if status in {"reconstructed", "not_needed", "not_applicable"}:
        return {"category": CATEGORY_A, "reason": f"no Needs OCR flag; reorder status {status}"}
    return {"category": CATEGORY_C, "reason": f"unexpected reorder status {status!r}"}


def library_choice(page: dict) -> tuple[str, str, str]:
    """(text, source label, reason) of the library base text for this page.

    Arabic pages (Arabic letter share >= the reconstruction threshold) use the processed
    text even when its reorder_status is "uncertain": on visual-order PDFs the raw text
    reverses numbers and phrase order. Raw PyMuPDF is the base only for not_applicable /
    mainly_ltr pages, or when the processed extraction failed or is empty.
    """
    reorder = page.get("reorder") or {}
    status, decision = reorder.get("reorder_status"), reorder.get("decision")
    processed = page.get("processed_text") or ""
    threshold = float((reorder.get("config") or {}).get("min_arabic_letter_share",
                                                        arabic_reorder.ReorderConfig().min_arabic_letter_share))
    share = float(page.get("arabic_letter_share") or 0.0)
    if page.get("library_error") or not processed.strip():
        return page.get("raw_text") or "", "pymupdf_raw", "processed extraction failed or is empty"
    if status == "not_applicable":
        return page["raw_text"], "pymupdf_raw", "no right-to-left text (processed status not_applicable)"
    if decision == "mainly_ltr":
        return page["raw_text"], "pymupdf_raw", f"page is mainly LTR (Arabic share {share:.0%} < {threshold:.0%})"
    if share >= threshold:
        return processed, f"pymupdf_processed ({status})", (
            f"Arabic share {share:.0%} >= {threshold:.0%}: processed text (logical order and numbers repaired)"
            + ("; its reorder_status is uncertain, review it" if status == "uncertain" else ""))
    return page["raw_text"], "pymupdf_raw", f"Arabic share {share:.0%} below {threshold:.0%}"


def analyze_pdf(pdf_path: Path, settings: dict | None = None,
                on_page: Callable[[int, int], None] | None = None) -> dict:
    import pymupdf

    settings = settings or default_settings()
    rule = diagnostics.needs_ocr_settings(settings)
    raw_session = extractors.open_session("pymupdf", Path(pdf_path), settings)
    processed_session = extractors.open_session(extractors.PROCESSED_METHOD_ID, Path(pdf_path), settings)
    document = pymupdf.open(Path(pdf_path))
    pages = []
    try:
        total = len(document)
        for number in range(1, total + 1):
            error = None
            raw_text = processed_text = ""
            reorder: dict = {}
            tables_doc = None
            extra: dict = {}
            try:
                raw_text = raw_session.extract(number).text
                output = processed_session.extract(number)
                processed_text = output.text
                extra = dict(output.extra)
                reorder = extra.get("reorder") or {}
                name = f"page_{number:03d}_tables.json"
                if name in output.files:
                    tables_doc = json.loads(output.files[name].decode("utf-8", errors="surrogatepass"))
            except Exception as exc:  # noqa: BLE001 - recorded; the page goes to category C
                error = f"{type(exc).__name__}: {exc}"
            page_rule = None if reorder.get("decision") == "mainly_ltr" else rule
            indicators = diagnostics.page_indicators(processed_text, page_rule)
            raw_indicators = diagnostics.page_indicators(raw_text, None)
            images = _images(document[number - 1])
            decision = decide(reorder, indicators, extraction_error=error)
            pages.append({
                "page": number,
                "raw_text": raw_text,
                "processed_text": processed_text,
                "reorder": reorder,
                "indicators_processed": indicators,
                "indicators_raw": raw_indicators,
                "arabic_letter_share": diagnostics.arabic_letter_share(raw_text),
                "tables": tables_doc,
                "table_count": len((tables_doc or {}).get("tables") or []),
                "images": images,
                "category": decision["category"],
                "category_reason": decision["reason"],
                "library_error": error,
            })
            if on_page is not None:
                on_page(number, total)
    finally:
        raw_session.close()
        processed_session.close()
        document.close()
    return {"version": ANALYSIS_VERSION, "settings": settings, "page_count": len(pages), "pages": pages,
            "methods": {"raw": "PyMuPDF Page.get_text('text')", "processed": arabic_reorder.METHOD_LABEL,
                        "tables": "PyMuPDF Page.find_tables() with per-cell reconstruction"}}


def default_vlm_pages(analysis: dict, *, include_image_pages: bool = False) -> dict[int, str]:
    """Pages proposed for the fallback model, with the reason (healthy pages are not proposed)."""
    chosen: dict[int, str] = {}
    for page in analysis["pages"]:
        if page["category"] in {CATEGORY_B, CATEGORY_C}:
            chosen[page["page"]] = page["category_reason"]
        elif include_image_pages and page["images"]["warning"]:
            chosen[page["page"]] = "user chose to process pages with raster images"
    return chosen
