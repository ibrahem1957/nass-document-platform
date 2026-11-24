"""Structured table extraction of the processed method (synthetic PDFs and hand-built geometry only).

No real PDF and no existing result folder is used.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pymupdf
import pytest

from lab.local_pdf_extraction import arabic_reorder as ar
from lab.local_pdf_extraction import exports, extractors, runner, storage, tables
from tests.lab import lpe_pdf_factory as factory

needs_arial = pytest.mark.skipif(not factory.ARIAL.exists(), reason="needs C:/Windows/Fonts/arial.ttf")
CSS = f"@font-face {{font-family: F; src: url({factory.ARIAL.name});}} * {{font-family: F; font-size: 10px;}}"


def _draw_table(page, x0: float, y0: float, col_width: float, row_height: float, rows: list[list[str]]) -> None:
    """Ruled grid (vector lines, so find_tables() can detect it) with text in each cell.
    `rows` are given VISUALLY left-to-right; "\\n" inside a cell makes two paragraphs."""
    archive = pymupdf.Archive(str(factory.FONT_DIR))
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            rect = pymupdf.Rect(x0 + c * col_width, y0 + r * row_height,
                                x0 + (c + 1) * col_width, y0 + (r + 1) * row_height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            if text:
                body = "".join(f"<p dir='auto'>{line}</p>" for line in text.split("\n"))
                page.insert_htmlbox(rect + (3, 2, -3, -2), body, css=CSS, archive=archive)


def _pdf(path: Path, pages: list[list[tuple]]) -> Path:
    doc = pymupdf.open()
    for page_tables in pages:
        page = doc.new_page(width=595, height=842)
        for spec in page_tables:
            _draw_table(page, *spec)
    doc.save(path)
    doc.close()
    return path


def _cell(table: dict, row: int, col: int) -> dict:
    return next(cell for cell in table["cells"] if cell["row"] == row and cell["col"] == col)


# --------------------------------------------------------------- RTL order
@needs_arial
def test_arabic_table_columns_are_ordered_right_to_left(tmp_path) -> None:
    path = _pdf(tmp_path / "rtl.pdf", [[(60, 100, 140, 40, [["النسبة", "العدد", "المنطقة"],
                                                            ["45.5%", "120", "الرياض"]])]])
    with pymupdf.open(path) as document:
        doc = tables.page_tables(document, 1)
    assert doc["status"] in ("ok", "uncertain") and len(doc["tables"]) == 1
    table = doc["tables"][0]
    assert table["has_arabic"] and table["direction"] == "rtl"
    assert (table["rows"], table["cols"]) == (2, 3)
    assert _cell(table, 0, 0)["processed_cell_text"] == "المنطقة"   # first logical column = rightmost cell
    assert _cell(table, 0, 0)["visual_col"] == 2
    assert _cell(table, 0, 2)["processed_cell_text"] == "النسبة"
    assert _cell(table, 1, 0)["processed_cell_text"] == "الرياض"
    assert _cell(table, 1, 2)["processed_cell_text"] == "45.5%"
    for cell in table["cells"]:
        assert cell["raw_cell_text"]  # raw text kept for every non-empty cell


@needs_arial
def test_multiline_cell_lines_are_joined_with_one_space(tmp_path) -> None:
    path = _pdf(tmp_path / "multi.pdf", [[(60, 100, 160, 60, [["السطر الثاني", "السطر الأول\nالسطر الثاني"]])]])
    with pymupdf.open(path) as document:
        table = tables.page_tables(document, 1)["tables"][0]
    cell = _cell(table, 0, 0)  # rightmost cell
    assert "\n" in cell["raw_cell_text"]
    assert cell["processed_cell_text"] == "السطر الأول السطر الثاني"


# ------------------------------------------------------- numeric preservation
@needs_arial
def test_numbers_dates_percentages_are_preserved(tmp_path) -> None:
    path = _pdf(tmp_path / "numbers.pdf", [[(60, 100, 140, 40, [["2025/01/02", "45.5%", "Data Quality"],
                                                                ["120", "7.25", "Overfitting"]])]])
    with pymupdf.open(path) as document:
        table = tables.page_tables(document, 1)["tables"][0]
    assert table["direction"] == "ltr"
    values = {_cell(table, r, c)["processed_cell_text"] for r in (0, 1) for c in (0, 1, 2)}
    assert {"2025/01/02", "45.5%", "Data Quality", "120", "7.25", "Overfitting"} <= values
    for cell in table["cells"]:
        assert cell["value_check"]["ok"], cell
        assert cell["reorder_status"] == "not_applicable"


@pytest.mark.parametrize("raw, processed, ok", [
    ("45.5\n%", "45.5%", True),
    ("45.5%", "%45.5", False),              # percent sign moved in front of the number
    ("2025/01/02", "02/01/2025", False),    # date changed
    ("و7\n1.3", "و1.37", True),             # documented join of a number split across lines
    ("Overfitting", "Overfti ting", False),
    ("٢٠٢٥", "٥٢٠٢", False),                # Arabic-Indic digits compared as values
    ("المنطقة 120", "120 المنطقة", True),   # order of words may change, values may not
    ("120", "", False),                     # value lost
    ("", "5", False),                       # value added
])
def test_value_check(raw, processed, ok) -> None:
    assert tables.value_check(raw, processed)["ok"] is ok


def test_changed_value_makes_cell_uncertain_and_keeps_both_texts(monkeypatch) -> None:
    monkeypatch.setattr(tables, "processed_cell_text", lambda lines, cfg: "%45.5 النسبة")
    line = [[{"c": ch, "bbox": (10 + i, 0, 11 + i, 10)} for i, ch in enumerate("النسبة 45.5%")]]
    result = tables.cell_result([line], ar.ReorderConfig())
    assert result["reorder_status"] == "uncertain"
    assert result["raw_cell_text"] == "النسبة 45.5%"
    assert result["processed_cell_text"] == "%45.5 النسبة"
    assert not result["value_check"]["ok"]


# ------------------------------------------------------------------- empty
@needs_arial
def test_empty_cells_stay_empty_strings(tmp_path) -> None:
    path = _pdf(tmp_path / "empty.pdf", [[(60, 100, 140, 40, [["أ", "", "ب"], ["", "ج", ""]])]])
    with pymupdf.open(path) as document:
        table = tables.page_tables(document, 1)["tables"][0]
    empties = [cell for cell in table["cells"] if cell["reorder_status"] == "empty"]
    assert len(empties) == 3
    for cell in empties:
        assert cell["raw_cell_text"] == "" and cell["processed_cell_text"] == ""
    assert (table["rows"], table["cols"]) == (2, 3)  # nothing added or dropped


# ------------------------------------------------------- tables over two pages
@needs_arial
def test_table_spanning_two_pages_is_not_merged_but_flagged(tmp_path) -> None:
    first = [["العمود ج", "العمود ب", "العمود أ"], ["3", "2", "1"], ["6", "5", "4"]]
    path = _pdf(tmp_path / "span.pdf", [
        [(60, 640, 140, 50, first)],                              # ends near the bottom of page 1
        [(60, 60, 140, 50, [["9", "8", "7"], ["12", "11", "10"]])],  # top of page 2, no header row
    ])
    with pymupdf.open(path) as document:
        page2 = tables.page_tables(document, 2)
        page1 = tables.page_tables(document, 1)
    table = page2["tables"][0]
    assert table["possible_continuation_of"]["page"] == 1
    assert table["possible_continuation_of"]["table_index"] == 0
    assert table["rows"] == 2  # only its own rows: never merged
    assert page1["tables"][0]["rows"] == 3
    assert page1["tables"][0]["possible_continuation_of"] is None


@needs_arial
def test_repeated_header_on_next_page_is_not_recorded_as_headerless_continuation(tmp_path) -> None:
    header = ["العمود ج", "العمود ب", "العمود أ"]
    path = _pdf(tmp_path / "repeat.pdf", [
        [(60, 640, 140, 50, [header, ["3", "2", "1"], ["6", "5", "4"]])],
        [(60, 60, 140, 50, [header, ["9", "8", "7"]])],
    ])
    with pymupdf.open(path) as document:
        assert tables.page_tables(document, 2)["tables"][0]["possible_continuation_of"] is None


# --------------------------------------------------------- geometry checks
def _geom(rows, bbox=(0, 0, 200, 100), row_count=None, col_count=None) -> tables.TableGeom:
    return tables.TableGeom(bbox, rows, len(rows) if row_count is None else row_count,
                            max(len(r) for r in rows) if col_count is None else col_count)


def test_valid_grid_has_no_problems() -> None:
    problems, merged = tables.validate_geometry(_geom([[(0, 0, 100, 50), (100, 0, 200, 50)],
                                                       [(0, 50, 100, 100), (100, 50, 200, 100)]]))
    assert problems == [] and merged == {}


@pytest.mark.parametrize("rows, kwargs, fragment", [
    ([[(0, 0, 120, 50), (100, 0, 200, 50)]], {}, "overlap"),
    ([[(0, 0, 100, 50), (100, 0, 260, 50)]], {}, "outside the table bbox"),
    ([[(0, 0, 100, 50), (100, 0, 200, 50)], [(0, 50, 100, 100)]], {}, "row 1 has 1 cells"),
    ([[(0, 0, 100, 50)]], {"row_count": 2}, "row count mismatch"),
    ([[None, None]], {}, "no cell bounding boxes"),
])
def test_inconsistent_geometry_marks_table_uncertain(rows, kwargs, fragment) -> None:
    geom = _geom(rows, **kwargs)
    problems, _ = tables.validate_geometry(geom)
    assert any(fragment in problem for problem in problems), problems
    table = tables.analyse_table(geom, [], ar.ReorderConfig(), 0)
    assert table["status"] == "uncertain"
    assert table["geometry_problems"] == problems
    assert len(table["cells"]) == sum(len(row) for row in rows)  # saved as detected, nothing added


def test_missing_and_merged_cells_are_recorded_without_guessing() -> None:
    geom = _geom([[(0, 0, 200, 50), (0, 0, 200, 50)], [None, (100, 50, 200, 100)]])
    problems, merged = tables.validate_geometry(geom)
    assert merged == {(0, 1): (0, 0)}
    table = tables.analyse_table(geom, [], ar.ReorderConfig(), 0)
    by_visual = {(c["row"], c["visual_col"]): c for c in table["cells"]}
    assert by_visual[(0, 1)]["reorder_status"] == "merged" and by_visual[(0, 1)]["processed_cell_text"] == ""
    assert by_visual[(1, 0)]["reorder_status"] == "not_detected" and by_visual[(1, 0)]["bbox"] is None
    assert by_visual[(1, 0)]["raw_cell_text"] == ""


def test_characters_are_assigned_by_centre_in_content_stream_order() -> None:
    line = [[{"c": "B", "bbox": (60, 0, 70, 10)}, {"c": "A", "bbox": (10, 0, 20, 10)},
             {"c": "C", "bbox": (150, 0, 160, 10)}]]
    assigned = tables.assign_chars([line], [((0, 0), (0, 0, 100, 20)), ((0, 1), (100, 0, 200, 20))])
    assert tables.raw_cell_text(assigned[(0, 0)]) == "BA"  # stream order, no reordering
    assert tables.raw_cell_text(assigned[(0, 1)]) == "C"


# ----------------------------------------------------------- markdown / html
def _doc_with(cells) -> dict:
    return {"page": 3, "status": "ok", "error": None, "notes": [], "tables": [{
        "table_index": 0, "rows": 1, "cols": len(cells), "status": "ok", "direction": "rtl", "warnings": [],
        "possible_continuation_of": None, "column_order": "logical right-to-left", "cells": cells}]}


def test_markdown_marks_uncertain_cells_and_escapes_pipes() -> None:
    cells = [
        {"row": 0, "col": 0, "reorder_status": "reconstructed", "processed_cell_text": "أ | ب", "raw_cell_text": "x"},
        {"row": 0, "col": 1, "reorder_status": "uncertain", "processed_cell_text": "%45.5", "raw_cell_text": "45.5%"},
        {"row": 0, "col": 2, "reorder_status": "not_detected", "processed_cell_text": "", "raw_cell_text": ""},
    ]
    markdown = tables.to_markdown(_doc_with(cells))
    assert "PROCESSED output, not raw library output" in markdown
    assert "أ \\| ب" in markdown and "%45.5 ⚠" in markdown and "⟨no cell⟩" in markdown
    rendered = exports.table_html(_doc_with(cells)["tables"][0])
    assert "dir='rtl'" in rendered and "⚠" in rendered and "\n" not in rendered
    assert "raw: 45.5%" in rendered  # raw text available as tooltip


# ------------------------------------------------- storage, page text, exports
@needs_arial
def test_tables_are_saved_separately_and_page_text_is_unchanged(tmp_path) -> None:
    data = _pdf(tmp_path / "t.pdf", [[(60, 100, 140, 40, [["النسبة", "العدد", "المنطقة"], ["45.5%", "120", "الرياض"]])]]
                ).read_bytes()
    root = tmp_path / "root"
    run_path = storage.create_run(root, [("t.pdf", data)],
                                  runner.method_entries(["pymupdf", "pymupdf_ar_reordered"], {}),
                                  {"reorder": {}, "tables": {"enabled": True}})
    assert runner.run(run_path)["status"] == "COMPLETED"
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    folder = storage.method_dir(run_path, key, "pymupdf_ar_reordered")
    record = json.loads((folder / "page_001.json").read_text(encoding="utf-8"))
    assert set(record["side_files"]) == {"page_001_tables.json", "page_001_tables.md"}
    assert record["table_regions"] and record["table_regions"][0]["bbox"]
    assert "never both" in record["table_regions_note"]
    # Page text is the processed page text, exactly as without tables.
    with pymupdf.open(storage.original_pdf(run_path, key)) as document:
        page = document[0]
        expected = ar.process_page(page, page.get_text("text")).text
    assert (folder / "page_001.txt").read_bytes() == expected.encode("utf-8")
    assert not list(storage.method_dir(run_path, key, "pymupdf").glob("*_tables.*"))  # raw untouched
    saved = json.loads((folder / "page_001_tables.json").read_text(encoding="utf-8"))
    assert saved["output_type"] == "processed" and saved["tables"][0]["cells"]
    assert "⚠" in (folder / "page_001_tables.md").read_text(encoding="utf-8") or saved["tables"][0]["uncertain_cells"] == 0

    collected = exports.collect(run_path)
    assert collected["documents"][0]["pages"][0]["outputs"]["pymupdf_ar_reordered"]["tables"]["tables"]
    report = exports.to_html(collected, lambda k, p: None, embed_images=False).decode("utf-8")
    assert "Tables — PROCESSED output" in report
    names = zipfile.ZipFile(io.BytesIO(exports.to_zip(run_path, {}))).namelist()
    assert any(name.endswith("pymupdf_ar_reordered/page_001_tables.json") for name in names)
    assert json.loads(exports.to_json(collected))["processed_tables"]

    # Tampering with the tables file invalidates the page; approved re-extraction quarantines it.
    (folder / "page_001_tables.json").write_text("{}", encoding="utf-8")
    assert storage.verify_page(run_path, key, "pymupdf_ar_reordered", 1)[0] == "invalid"
    request = runner.create_resume_request(run_path, None, [{"doc_key": key, "method_id": "pymupdf_ar_reordered",
                                                              "page": 1}])
    runner.run(run_path, resume_request=request)
    assert storage.verify_page(run_path, key, "pymupdf_ar_reordered", 1)[0] == "valid"
    assert len(list((folder / "_quarantine").glob("page_001_tables.*"))) == 2


def test_runs_without_table_setting_keep_previous_behaviour(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)],
                                  runner.method_entries(["pymupdf_ar_reordered"], {}), {"reorder": {}})
    runner.run(run_path)
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    folder = storage.method_dir(run_path, key, "pymupdf_ar_reordered")
    record = json.loads((folder / "page_001.json").read_text(encoding="utf-8"))
    assert "side_files" not in record and "table_regions" not in record
    assert not list(folder.glob("*_tables.*"))
    assert exports.collect(run_path)["documents"][0]["pages"][0]["outputs"]["pymupdf_ar_reordered"]["tables"] is None


def test_orphan_side_file_is_an_incomplete_write(tmp_path) -> None:
    data = factory.small_pdf(tmp_path / "s.pdf", pages=1).read_bytes()
    run_path = storage.create_run(tmp_path / "root", [("s.pdf", data)],
                                  runner.method_entries(["pymupdf_ar_reordered"], {}), {"reorder": {}})
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    folder = storage.method_dir(run_path, key, "pymupdf_ar_reordered")
    folder.mkdir(parents=True)
    (folder / "page_001_tables.json").write_text("{}", encoding="utf-8")
    assert storage.verify_page(run_path, key, "pymupdf_ar_reordered", 1)[0] == "invalid"
    with pytest.raises(storage.StorageError):
        storage.write_page_result(run_path, key, "pymupdf_ar_reordered", 1, {"status": "COMPLETED"}, "x",
                                  side_files={"../evil.json": b"{}"}, replace_approved=True)


def test_page_without_tables_but_with_images_does_not_guess(tmp_path) -> None:
    path = tmp_path / "img.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 20, 20), False)
    pixmap.clear_with(200)
    page.insert_image(pymupdf.Rect(50, 50, 250, 250), pixmap=pixmap)
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as document:
        result = tables.page_tables(document, 1)
    assert result["tables"] == [] and result["page_has_images"]
    assert any("cannot read tables inside images" in note for note in result["notes"])
    assert extractors.PageOutput("x").files == {}
