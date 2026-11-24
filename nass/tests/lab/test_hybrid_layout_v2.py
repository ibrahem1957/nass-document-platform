"""Layout version 2 regressions with small synthetic PDFs built in the tests (no real files).

Patterns: frame/shadow image around a high-resolution image, a wide but short KPI image,
table gridlines that must not become a chart, a real vector chart outside tables,
paragraph merging, separate paragraphs and list items.
"""

from __future__ import annotations

import fitz

from lab.core.hybrid import layout


def _pixmap(width: int, height: int, value: int = 180) -> fitz.Pixmap:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height), False)
    pix.clear_with(value)
    return pix


def _page():
    document = fitz.open()
    return document, document.new_page(width=595, height=842)


def _extract(page, tables_doc=None):
    return layout.extract_page_elements(page, 1, reorder_config={}, tables_doc=tables_doc)[0]


# ------------------------------------------------------------------ images


def test_low_resolution_frame_around_high_resolution_image_becomes_one_figure() -> None:
    document, page = _page()
    page.insert_image(fitz.Rect(136, 395, 529, 600), pixmap=_pixmap(137, 72, 230), keep_proportion=False)
    page.insert_image(fitz.Rect(151, 410, 499, 570), pixmap=_pixmap(883, 405, 90), keep_proportion=False)
    elements = _extract(page)
    rasters = [fig for fig in elements["figures"] if fig["origin"] == "raster"]
    assert len(rasters) == 1 and rasters[0]["merged_from"] == ["img0"]
    assert elements["image_merges"][0]["absorbed"] == "img0" and elements["image_merges"][0]["kept"] == "img1"
    assert rasters[0]["bbox"][0] <= 137 and rasters[0]["bbox"][3] >= 599  # combined region geometry
    document.close()


def test_distinct_overlapping_images_are_kept_and_flagged() -> None:
    document, page = _page()
    page.insert_image(fitz.Rect(100, 100, 400, 300), pixmap=_pixmap(600, 400), keep_proportion=False)
    page.insert_image(fitz.Rect(150, 120, 420, 310), pixmap=_pixmap(540, 380), keep_proportion=False)
    elements = _extract(page)
    assert len([fig for fig in elements["figures"] if fig["origin"] == "raster"]) == 2
    assert not elements["image_merges"] and elements["image_review"]
    document.close()


def test_side_by_side_images_are_never_merged() -> None:
    document, page = _page()
    page.insert_image(fitz.Rect(50, 100, 280, 300), pixmap=_pixmap(460, 400), keep_proportion=False)
    page.insert_image(fitz.Rect(310, 100, 540, 300), pixmap=_pixmap(460, 400), keep_proportion=False)
    elements = _extract(page)
    assert len(elements["figures"]) == 2 and not elements["image_merges"] and not elements["image_review"]
    document.close()


def test_wide_short_kpi_image_is_kept_and_tiny_logo_skipped() -> None:
    document, page = _page()
    page.insert_image(fitz.Rect(100, 100, 500, 191), pixmap=_pixmap(976, 208), keep_proportion=False)  # 7.3%, 91 pt
    page.insert_image(fitz.Rect(500, 20, 530, 50), pixmap=_pixmap(60, 60), keep_proportion=False)  # logo
    elements = _extract(page)
    assert [fig["id"] for fig in elements["figures"]] == ["img0"]
    skipped = elements["skipped_images"]
    assert skipped[0]["id"] == "img1" and "area" in skipped[0]["reason"]
    assert skipped[0]["thresholds"] == {"min_area_share": 0.02, "min_long_side_pt": 100.0}
    document.close()


def test_skip_rule_area_or_longest_side() -> None:
    settings = layout.DEFAULTS
    wide = {"area_share": 0.073, "width_pt": 400, "height_pt": 91}
    narrow_long = {"area_share": 0.03, "width_pt": 95, "height_pt": 99}
    assert layout.skip_reason(wide, settings) is None
    assert "longest side" in layout.skip_reason(narrow_long, settings)


# ------------------------------------------------------------------ vectors


def _grid(page, box, rows=6, cols=4) -> None:
    x0, y0, x1, y1 = box
    for i in range(rows + 1):
        y = y0 + (y1 - y0) * i / rows
        page.draw_line((x0, y), (x1, y))
    for j in range(cols + 1):
        x = x0 + (x1 - x0) * j / cols
        page.draw_line((x, y0), (x, y1))


def test_table_gridlines_are_not_a_vector_chart() -> None:
    document, page = _page()
    _grid(page, (50, 400, 550, 700))
    tables_doc = {"tables": [{"table_index": 0, "bbox": [50, 400, 550, 700], "cells": []}]}
    elements = _extract(page, tables_doc)
    assert not [fig for fig in elements["figures"] if fig["origin"] == "vector"]
    assert sum(elements["vector_exclusions"].values()) > 0
    document.close()


def test_unfilled_lines_without_a_table_are_decorative_not_a_chart() -> None:
    document, page = _page()
    _grid(page, (50, 400, 400, 600))
    elements = _extract(page)
    assert not [fig for fig in elements["figures"] if fig["origin"] == "vector"]
    assert elements["ignored_vector_regions"]
    document.close()


def test_vector_chart_outside_tables_is_kept() -> None:
    document, page = _page()
    colors = [(0.8, 0.1, 0.1), (0.1, 0.6, 0.2), (0.1, 0.3, 0.8)]
    for index in range(12):
        x = 60 + index * 18
        page.draw_rect(fitz.Rect(x, 300 - 10 * (index % 6 + 1), x + 12, 300), color=None,
                       fill=colors[index % 3])
    _grid(page, (50, 450, 550, 750))
    tables_doc = {"tables": [{"table_index": 0, "bbox": [50, 450, 550, 750], "cells": []}]}
    vectors = [fig for fig in _extract(page, tables_doc)["figures"] if fig["origin"] == "vector"]
    assert len(vectors) == 1 and vectors[0]["confidence"] == "likely" and vectors[0]["bbox"][3] <= 301
    document.close()


# ---------------------------------------------------------------- paragraphs


def _line(block_id, bbox, text, size=10.0):
    return {"id": block_id, "bbox": bbox, "raw_text": text, "processed_text": text, "max_font_size": size,
            "font_size": size, "bold": False, "line_count": 1, "line_boxes": [bbox]}


def _assemble(blocks):
    elements = {"page": 1, "width": 595, "height": 842, "body_font_size": 10.0, "blocks": blocks, "figures": [],
                "skipped_images": [], "settings": {}}
    text = "\n".join(block["raw_text"] for block in blocks)
    return layout.assemble_page(elements, decision={"final_source": "pymupdf_raw", "final_text": text, "status": "Accepted"},
                                library_page={"reorder": {}, "arabic_letter_share": 0.0}, tables_doc=None,
                                vlm_text="", reorder_config={})


def test_lines_of_one_paragraph_are_merged_and_paragraphs_stay_separate() -> None:
    out = _assemble([
        _line("b0", [50, 100, 550, 112], "line one of the paragraph"),
        _line("b1", [50, 114, 550, 126], "line two of the paragraph"),
        _line("b2", [50, 128, 400, 140], "last line."),
        _line("b3", [50, 160, 550, 172], "A new paragraph."),
    ])
    assert out["markdown"] == "line one of the paragraph line two of the paragraph last line.\n\nA new paragraph.\n"
    assert [m["merged"] for m in out["record"]["paragraph_merges"]] == ["b1", "b2"]
    assert out["record"]["validation"]["text_elements_match_page_text"]
    assert out["record"]["text_status"] == "Accepted" and out["record"]["layout_reviewed"] is False


def test_first_line_indent_starts_a_new_paragraph() -> None:
    out = _assemble([_line("b0", [50, 100, 550, 112], "end of a full line"),
                     _line("b1", [80, 114, 550, 126], "indented new paragraph")])
    assert not out["record"]["paragraph_merges"]
    assert out["markdown"] == "end of a full line\n\nindented new paragraph\n"


def test_list_items_are_not_merged_and_render_as_markdown_lists() -> None:
    out = _assemble([_line("b0", [50, 100, 550, 112], "• first item"),
                     _line("b1", [50, 114, 550, 126], "• second item"),
                     _line("b2", [50, 128, 550, 140], "٢) بند عربي")])
    assert not out["record"]["paragraph_merges"]
    md = out["markdown"]
    assert "- first item" in md and "- second item" in md and "- ٢) بند عربي" in md
    assert len(out["record"]["list_items"]) == 3


def test_heading_is_never_merged_into_the_paragraph() -> None:
    out = _assemble([_line("b0", [50, 100, 550, 120], "Title", size=18.0),
                     _line("b1", [50, 122, 550, 134], "Body line.")])
    assert out["markdown"].startswith("## Title\n\nBody line.")
