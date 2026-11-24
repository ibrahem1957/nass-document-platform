"""Layout version 3 regressions on synthetic elements (no real files): RTL side-by-side order,
bullet/list formatting and tables continued from the previous page."""

from __future__ import annotations

from lab.core.hybrid import layout


def _line(block_id, bbox, text, size=10.0):
    return {"id": block_id, "bbox": bbox, "raw_text": text, "processed_text": text, "max_font_size": size,
            "font_size": size, "bold": False, "line_count": 1, "line_boxes": [bbox]}


def _assemble(blocks, *, rtl=True, figures=(), tables=None, previous=None, page=2):
    elements = {"page": page, "width": 595, "height": 842, "body_font_size": 10.0, "blocks": blocks,
                "figures": list(figures), "skipped_images": [], "settings": {}}
    text = "\n".join(block["raw_text"] for block in blocks)
    return layout.assemble_page(
        elements, decision={"final_source": "pymupdf_raw", "final_text": text, "status": "Accepted"},
        library_page={"reorder": {}, "arabic_letter_share": 1.0 if rtl else 0.0},
        tables_doc={"tables": tables} if tables else None, vlm_text="", reorder_config={},
        previous_tables_doc={"tables": previous} if previous else None)


def _figure(bbox):
    return {"id": "img0", "origin": "raster", "bbox": bbox, "area_share": 0.1}


PARAGRAPH = [_line("b0", [320, 100, 550, 112], "رغم أن إجمالي الحوادث"),
             _line("b1", [320, 114, 550, 126], "انخفض في العام الماضي"),
             _line("b2", [400, 128, 550, 140], "بشكل واضح.")]


# ------------------------------------------------------------ side by side


def test_rtl_paragraph_on_the_right_comes_before_the_image_on_the_left() -> None:
    out = _assemble(PARAGRAPH, figures=[_figure([50, 98, 312, 160])])  # narrow 8 pt gutter
    order = out["record"]["order"]
    assert order.index("b0") < order.index("img0")
    assert out["markdown"].index("رغم أن") < out["markdown"].index("🖼️")
    assert out["record"]["side_by_side"] and out["record"]["structure_checks"]["side_by_side_order"]["ok"]


def test_ltr_page_reads_the_left_image_first() -> None:
    blocks = [_line("b0", [320, 100, 550, 112], "Text on the right"), _line("b1", [320, 114, 550, 126], "continues here.")]
    out = _assemble(blocks, rtl=False, figures=[_figure([50, 98, 312, 160])])
    assert out["record"]["order"].index("img0") < out["record"]["order"].index("b0")


def test_vertically_stacked_elements_are_not_reordered() -> None:
    out = _assemble(PARAGRAPH[:1] and [_line("b0", [320, 200, 550, 212], "نص تحت الصورة")],
                    figures=[_figure([50, 40, 312, 150])])
    assert out["record"]["order"] == ["img0", "b0"]


# ----------------------------------------------------------------- lists


def test_bullet_item_is_never_a_heading_and_isolated_bullets_are_dropped() -> None:
    out = _assemble([_line("b0", [300, 100, 550, 120], "• البيانات", size=18.0),
                     _line("b1", [560, 100, 570, 112], "•")])
    md = out["markdown"]
    assert "### •" not in md and "## •" not in md and "- البيانات" in md
    assert not any(line.strip() == "•" for line in md.splitlines())
    assert out["record"]["discarded_markers"][0]["element"] == "b1"
    assert out["record"]["validation"]["text_elements_match_page_text"]  # characters unchanged
    checks = out["record"]["structure_checks"]
    assert checks["no_bullet_headings"]["ok"] and checks["no_empty_list_items"]["ok"]


def test_trailing_bullet_on_rtl_line_is_a_list_marker() -> None:
    markdown, items = layout.paragraph_markdown("السبب الأول •\nالسبب الثاني •", rtl=True)
    assert markdown == "- السبب الأول\n- السبب الثاني" and len(items) == 2


def test_list_markers_without_text_never_create_empty_items() -> None:
    discarded = []
    markdown, items = layout.paragraph_markdown("•\n- بند\n•", rtl=True, discarded=discarded)
    assert markdown == "- بند" and len(discarded) == 2 and len(items) == 1


# ---------------------------------------------------------------- tables


def _cells(rows, columns=((50, 300), (300, 550)), top=40.0, height=20.0):
    cells = []
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            x0, x1 = columns[c]
            cells.append({"row": r, "col": c, "bbox": [x0, top + r * height, x1, top + (r + 1) * height],
                          "processed_cell_text": value, "reorder_status": "not_needed"})
    return cells


def test_continued_table_reuses_the_verified_header_and_keeps_all_rows_as_data() -> None:
    previous = [{"table_index": 0, "bbox": [50, 600, 550, 800],
                 "cells": _cells([["السنة", "العدد"], ["2023", "10"]], top=600)}]
    table = {"table_index": 0, "bbox": [50, 40, 550, 100], "possible_continuation_of": {"page": 1, "table_index": 0},
             "cells": _cells([["2024", "12"], ["2025", "15"]])}
    out = _assemble([], tables=[table], previous=previous)
    md = out["markdown"]
    assert "| السنة | العدد |\n| --- | --- |\n| 2024 | 12 |\n| 2025 | 15 |" in md
    continuation = out["record"]["table_continuations"][0]
    assert continuation["reliable"] and continuation["of_page"] == 1
    assert out["record"]["structure_checks"]["valid_table_headers"]["ok"]


def test_uncertain_continuation_gets_a_blank_header_and_no_promoted_row() -> None:
    previous = [{"table_index": 0, "bbox": [50, 600, 550, 800],
                 "cells": _cells([["السنة", "العدد"]], columns=((50, 200), (200, 550)), top=600)}]
    table = {"table_index": 0, "bbox": [50, 40, 550, 100], "possible_continuation_of": {"page": 1, "table_index": 0},
             "cells": _cells([["2024", "12"], ["2025", "15"]])}
    out = _assemble([], tables=[table], previous=previous)
    assert "|  |  |\n| --- | --- |\n| 2024 | 12 |" in out["markdown"]
    assert not out["record"]["table_continuations"][0]["reliable"]
    assert any("may continue from the previous page" in note for note in out["record"]["needs_review"])


def test_ordinary_table_keeps_its_own_first_row_as_header() -> None:
    table = {"table_index": 0, "bbox": [50, 300, 550, 360], "cells": _cells([["Year", "Value"], ["2025", "15"]], top=300)}
    out = _assemble([], rtl=False, tables=[table])
    assert "| Year | Value |\n| --- | --- |\n| 2025 | 15 |" in out["markdown"]
    assert not out["record"]["table_continuations"]
