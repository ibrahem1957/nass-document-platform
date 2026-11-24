"""PROCESSED table extraction: PyMuPDF `find_tables()` + per-cell Arabic logical-order
reconstruction. NOT raw library output, and not verified cell by cell against ground truth.

Per page this produces a structure saved next to the processed page text as
`page_NNN_tables.json` / `page_NNN_tables.md`. The page text (`page_NNN.txt`) is never
changed: the table regions are listed in the page JSON (`table_regions`) so that a
consumer uses EITHER the page text OR the structured table for a region, never both.

Rules (see lab/README.md):
* table geometry is validated (counts, overlaps, cells inside the table); a failure marks
  the whole table "uncertain" and the detected structure is still saved as detected;
* missing/merged cells are recorded as such; no row, column or cell content is invented;
* raw_cell_text = characters whose centre lies in the cell, content-stream order, no reordering;
* processed_cell_text = the production `arabic_reorder` rules on those characters only,
  lines inside a cell joined with a single space;
* per-cell value check (numbers, dates, percentages, Latin tokens) against the raw cell;
  only the documented joins of split numbers are allowed; otherwise the cell is "uncertain";
* Arabic tables: columns in logical right-to-left order (col 0 = rightmost cell);
* tables are never merged across pages; only `possible_continuation_of` is recorded;
* tables inside images cannot be detected and are never guessed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace

from lab.local_pdf_extraction import arabic_reorder as ar
from lab.local_pdf_extraction import diagnostics

TABLES_VERSION = "1"
METHOD_ID = "pymupdf_ar_reordered"
LABEL = ("Processed — PyMuPDF find_tables() + per-cell Arabic logical-order reconstruction "
         "(not raw library output)")
RAG_NOTE = ("page_NNN.txt still contains the text of these table regions. For each region listed in "
            "table_regions, use EITHER the page text OR page_NNN_tables.json, never both.")
BBOX_TOLERANCE = 1.0      # pt: a cell may exceed the table bbox by this much
OVERLAP_TOLERANCE = 0.5   # pt: shared borders are not overlaps
TOP_FRACTION = 0.2        # a table "at the top of a page" starts in the top 20 %
BOTTOM_FRACTION = 0.8     # a table "at the end of a page" ends in the bottom 20 %
MAX_PROBLEMS = 25
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DATE = re.compile(r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}")

Bbox = tuple[float, float, float, float]


@dataclass
class TableGeom:
    bbox: Bbox
    rows: list[list[Bbox | None]]  # as detected: row -> cells left-to-right, None = no cell
    row_count: int
    col_count: int
    header_external: bool | None = None


def _round(box) -> list[float] | None:
    return None if box is None else [round(float(value), 2) for value in box]


# ------------------------------------------------------------------ detection
def detect_tables(page) -> tuple[list[TableGeom], str | None]:
    """PyMuPDF `page.find_tables()`; geometry only. Returns (tables top-to-bottom, error)."""
    try:
        found = page.find_tables()
    except Exception as error:  # noqa: BLE001 - recorded, never raised into the page result
        return [], f"find_tables() failed: {type(error).__name__}: {error}"
    geoms: list[TableGeom] = []
    for table in getattr(found, "tables", None) or []:
        rows = [[None if cell is None else tuple(float(v) for v in cell) for cell in row.cells]
                for row in table.rows]
        try:
            external = bool(table.header.external)
        except Exception:  # noqa: BLE001 - header detection is optional metadata
            external = None
        geoms.append(TableGeom(tuple(float(v) for v in table.bbox), rows, int(table.row_count),
                               int(table.col_count), external))
    geoms.sort(key=lambda geom: (geom.bbox[1], geom.bbox[0]))
    return geoms, None


# ------------------------------------------------------------------ geometry
def _inside(cell: Bbox, box: Bbox, tolerance: float = BBOX_TOLERANCE) -> bool:
    return (cell[0] >= box[0] - tolerance and cell[1] >= box[1] - tolerance
            and cell[2] <= box[2] + tolerance and cell[3] <= box[3] + tolerance)


def _overlap(a: Bbox, b: Bbox) -> bool:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return width > OVERLAP_TOLERANCE and height > OVERLAP_TOLERANCE


def _key(box: Bbox) -> tuple:
    return tuple(round(value, 1) for value in box)


def validate_geometry(geom: TableGeom) -> tuple[list[str], dict[tuple[int, int], tuple[int, int]]]:
    """Return (problems, merged). `merged` maps a grid position whose bbox repeats an earlier
    cell's bbox (a merged cell reported twice) to that earlier position; this is recorded,
    not treated as an overlap. Any problem makes the table "uncertain"."""
    problems: list[str] = []
    merged: dict[tuple[int, int], tuple[int, int]] = {}
    if geom.row_count < 1 or geom.col_count < 1:
        problems.append("no rows or columns detected")
    if len(geom.rows) != geom.row_count:
        problems.append(f"row count mismatch: {len(geom.rows)} rows listed, row_count={geom.row_count}")
    for r, row in enumerate(geom.rows):
        if len(row) != geom.col_count:
            problems.append(f"row {r} has {len(row)} cells, col_count={geom.col_count}")
    present = [(r, c, cell) for r, row in enumerate(geom.rows) for c, cell in enumerate(row) if cell is not None]
    if not present:
        problems.append("no cell bounding boxes detected")
    for r, c, cell in present:
        if cell[2] <= cell[0] or cell[3] <= cell[1]:
            problems.append(f"cell ({r},{c}) has an empty bbox")
        if not _inside(cell, geom.bbox):
            problems.append(f"cell ({r},{c}) lies outside the table bbox")
    first_seen: dict[tuple, tuple[int, int]] = {}
    for index, (r, c, cell) in enumerate(present):
        if _key(cell) in first_seen:
            merged[(r, c)] = first_seen[_key(cell)]
            continue
        first_seen[_key(cell)] = (r, c)
        for r2, c2, other in present[index + 1:]:
            if _key(other) != _key(cell) and _overlap(cell, other):
                problems.append(f"cells ({r},{c}) and ({r2},{c2}) overlap")
    if len(problems) > MAX_PROBLEMS:
        problems = problems[:MAX_PROBLEMS] + [f"... and {len(problems) - MAX_PROBLEMS} more"]
    return problems, merged


# -------------------------------------------------------------- characters
def page_char_lines(page) -> list[list[list[dict]]]:
    """rawdict characters: line -> span -> chars, in PyMuPDF content-stream order."""
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            lines.append([span["chars"] for span in line["spans"]])
    return lines


def _centre_inside(char: dict, box: Bbox) -> bool:
    x0, y0, x1, y1 = char["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def assign_chars(lines: list[list[list[dict]]], cells: list[tuple[tuple[int, int], Bbox]]) -> dict:
    """Each character goes to the first cell whose bbox contains its centre. The line ->
    span -> char structure and the content-stream order are preserved inside each cell."""
    out: dict = {position: [] for position, _ in cells}
    if not cells:
        return out
    outer = (min(b[0] for _, b in cells), min(b[1] for _, b in cells),
             max(b[2] for _, b in cells), max(b[3] for _, b in cells))
    for line in lines:
        per_line: dict = {}
        for span in line:
            per_span: dict = {}
            for char in span:
                if not _centre_inside(char, outer):
                    continue
                for position, box in cells:
                    if _centre_inside(char, box):
                        per_span.setdefault(position, []).append(char)
                        break
            for position, chars in per_span.items():
                per_line.setdefault(position, []).append(chars)
        for position, spans in per_line.items():
            out[position].append(spans)
    return out


def raw_cell_text(cell_lines: list[list[list[dict]]]) -> str:
    """Content-stream order, no reordering; PyMuPDF lines separated by a newline."""
    return "\n".join("".join(char["c"] for span in line for char in span) for line in cell_lines)


def processed_cell_text(cell_lines: list[list[list[dict]]], cfg: ar.ReorderConfig) -> str:
    """Production arabic_reorder rules on this cell's characters only; lines joined by one space."""
    cell_cfg = replace(cfg, segment_separator=" ")
    spans = [span for line in cell_lines for span in line]
    rebuilt = ar.reconstruct_clusters(ar.clusters_from_spans(spans, cell_cfg), cell_cfg)
    return " ".join(part.strip() for part in rebuilt.text.split("\n") if part.strip())


# ------------------------------------------------------------- value check
def value_check(raw: str, processed: str) -> dict:
    """Numbers, dates, percentages and Latin tokens must be preserved (Arabic-Indic digits are
    compared as their values). Only the documented joins of split numbers are allowed."""
    a = unicodedata.normalize("NFKC", raw).translate(_DIGITS)
    b = unicodedata.normalize("NFKC", processed).translate(_DIGITS)
    runs = ar.latin_digit_runs_check(a, b)
    dates_raw, dates_processed = Counter(_DATE.findall(a)), Counter(_DATE.findall(b))
    percent = {"raw": a.count("%") + a.count("٪"), "processed": b.count("%") + b.count("٪"),
               "before_digit_raw": diagnostics.percent_before_digit(a),
               "before_digit_processed": diagnostics.percent_before_digit(b)}
    problems = []
    if not runs["ok"]:
        problems.append("numbers or Latin tokens differ")
    if dates_raw != dates_processed:
        problems.append("dates differ")
    if percent["raw"] != percent["processed"]:
        problems.append("number of percent signs differs")
    if percent["before_digit_processed"] > percent["before_digit_raw"]:
        problems.append("a percent sign moved in front of a number")
    return {"ok": not problems, "problems": problems, "numbers_and_latin": runs,
            "dates": {"raw": sorted(dates_raw.elements()), "processed": sorted(dates_processed.elements())},
            "percent_signs": percent}


def cell_result(cell_lines: list[list[list[dict]]], cfg: ar.ReorderConfig) -> dict:
    raw = raw_cell_text(cell_lines)
    if not raw.strip():
        return {"raw_cell_text": raw, "processed_cell_text": "", "reorder_status": "empty", "value_check": None}
    processed = processed_cell_text(cell_lines, cfg)
    check = value_check(raw, processed)
    garbage = sorted({f"U+{ord(ch):04X}" for ch in raw if diagnostics.is_garbage_char(ch)})
    if garbage:
        status, reason = "uncertain", f"unmapped/garbage glyphs: {', '.join(garbage[:8])}"
    elif not check["ok"]:
        status, reason = "uncertain", "; ".join(check["problems"])
    elif not ar.contains_rtl(raw):
        status, reason = "not_applicable", "no Arabic/Hebrew text in this cell"
    else:
        status, reason = "reconstructed", "reconstructed, not verified against ground truth"
    return {"raw_cell_text": raw, "processed_cell_text": processed, "reorder_status": status,
            "reason": reason, "value_check": check}


# ------------------------------------------------------------------- tables
def analyse_table(geom: TableGeom, lines: list[list[list[dict]]], cfg: ar.ReorderConfig, index: int) -> dict:
    problems, merged = validate_geometry(geom)
    positions = [((r, c), box) for r, row in enumerate(geom.rows) for c, box in enumerate(row)
                 if box is not None and (r, c) not in merged]
    assigned = assign_chars(lines, positions)
    has_arabic = any(ar.contains_rtl(raw_cell_text(assigned[position])) for position, _ in positions)
    cells = []
    for r, row in enumerate(geom.rows):
        for c, box in enumerate(row):
            entry = {"row": r, "col": (len(row) - 1 - c) if has_arabic else c, "visual_col": c,
                     "bbox": _round(box), "detected": box is not None}
            if box is None:
                entry.update(raw_cell_text="", processed_cell_text="", reorder_status="not_detected",
                             value_check=None,
                             note="No cell detected at this grid position (merged or missing); nothing guessed.")
            elif (r, c) in merged:
                first = merged[(r, c)]
                entry.update(raw_cell_text="", processed_cell_text="", reorder_status="merged", value_check=None,
                             merged_with={"row": first[0], "visual_col": first[1]},
                             note="Same bbox as an earlier cell (merged cell); its text is stored once there.")
            else:
                entry.update(cell_result(assigned[(r, c)], cfg))
            cells.append(entry)
    cells.sort(key=lambda cell: (cell["row"], cell["col"]))
    uncertain = sum(1 for cell in cells if cell["reorder_status"] == "uncertain")
    warnings = list(problems)
    if uncertain:
        warnings.append(f"{uncertain} cell(s) uncertain (value check failed or unreliable glyphs)")
    return {
        "table_index": index, "bbox": _round(geom.bbox), "rows": geom.row_count, "cols": geom.col_count,
        "rows_listed": len(geom.rows), "status": "uncertain" if problems else "ok",
        "geometry_problems": problems, "warnings": warnings, "has_arabic": has_arabic,
        "direction": "rtl" if has_arabic else "ltr",
        "column_order": ("logical right-to-left: col 0 is the rightmost cell" if has_arabic
                         else "left-to-right as on the page"),
        "header": {"pymupdf_header_external": geom.header_external,
                   "note": "No header row is inferred; row 0 is simply the first detected row."},
        "uncertain_cells": uncertain, "possible_continuation_of": None, "cells": cells,
    }


def _first_row_texts(geom: TableGeom, lines: list[list[list[dict]]]) -> list[str]:
    if not geom.rows:
        return []
    cells = [((0, c), box) for c, box in enumerate(geom.rows[0]) if box is not None]
    assigned = assign_chars(lines, cells)
    return [" ".join(raw_cell_text(assigned[position]).split()) for position, _ in cells]


def previous_page_summary(document, page_number: int) -> dict | None:
    """The last table on the previous page (geometry and first-row raw text only)."""
    if page_number < 2:
        return None
    page = document[page_number - 2]
    geoms, error = detect_tables(page)
    if error or not geoms:
        return None
    index, last = max(enumerate(geoms), key=lambda item: item[1].bbox[3])
    return {"page": page_number - 1, "table_index": index, "cols": last.col_count,
            "ends_near_bottom": last.bbox[3] >= BOTTOM_FRACTION * float(page.rect.height),
            "first_row": _first_row_texts(last, page_char_lines(page))}


def continuation_of(geom: TableGeom, first_row: list[str], page_height: float, previous: dict | None) -> dict | None:
    """Record (never merge) a possible continuation: this table starts at the top of the page,
    its first row does not repeat the previous table's first row (so no header row is seen),
    and the previous page ended with a table that has the same column count."""
    if previous is None or not previous["ends_near_bottom"]:
        return None
    if geom.bbox[1] > TOP_FRACTION * page_height or geom.col_count != previous["cols"]:
        return None
    if first_row and first_row == previous["first_row"]:
        return None  # it repeats the header row -> it has its own header row
    return {"page": previous["page"], "table_index": previous["table_index"],
            "criteria": "top of page, no repeated header row, previous page ends with a table with the same "
                        "column count; NOT merged"}


def page_tables(document, page_number: int, cfg: ar.ReorderConfig | None = None) -> dict:
    """All tables of one page (1-based). Never raises: failures are recorded in the result."""
    cfg = cfg or ar.ReorderConfig()
    out = {"version": TABLES_VERSION, "page": page_number, "method_id": METHOD_ID, "output_type": "processed",
           "label": LABEL, "status": "no_tables", "error": None, "page_has_images": False, "notes": [],
           "rag_note": RAG_NOTE, "verified_against_ground_truth": False, "tables": []}
    try:
        page = document[page_number - 1]
        try:
            out["page_has_images"] = bool(page.get_images(full=False))
        except Exception:  # noqa: BLE001
            pass
        geoms, error = detect_tables(page)
        if error:
            out.update(status="detection_failed", error=error)
            out["notes"].append("Table detection failed; no structured table is saved for this page.")
            return out
        if not geoms:
            if out["page_has_images"]:
                # Renderers already say "No table detected on this page." once; this note only adds why.
                out["notes"].append("The page contains images: find_tables() cannot read tables inside images "
                                    "(see the 'Needs OCR?' indicator). Nothing is guessed.")
            return out
        lines = page_char_lines(page)
        height = float(page.rect.height)
        for index, geom in enumerate(geoms):
            table = analyse_table(geom, lines, cfg, index)
            if index == 0 and geom.bbox[1] <= TOP_FRACTION * height:
                table["possible_continuation_of"] = continuation_of(
                    geom, _first_row_texts(geom, lines), height, previous_page_summary(document, page_number))
            out["tables"].append(table)
        out["status"] = "uncertain" if any(t["status"] == "uncertain" or t["uncertain_cells"] for t in out["tables"]) else "ok"
    except Exception as error:  # noqa: BLE001 - tables must never break the page text result
        out.update(status="detection_failed", error=f"{type(error).__name__}: {error}", tables=[])
        out["notes"].append("Table processing failed; no structured table is saved for this page.")
    return out


def clean_note(note: str) -> str:
    """Notes saved by version 1 started with "No table detected." (shown twice); display without it."""
    return note.removeprefix("No table detected. ").strip()


def table_regions(doc: dict) -> list[dict]:
    return [{"table_index": table["table_index"], "bbox": table["bbox"], "status": table["status"]}
            for table in doc.get("tables", [])]


# ------------------------------------------------------------------ markdown
def _md_cell(cell: dict) -> str:
    status = cell["reorder_status"]
    if status == "not_detected":
        return "⟨no cell⟩"
    if status == "merged":
        return "⟨merged⟩"
    text = cell["processed_cell_text"].replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
    return f"{text} ⚠" if status == "uncertain" else text


def grid(table: dict) -> tuple[int, list[dict[int, dict]]]:
    """(width, rows) where rows[i] maps logical column -> cell."""
    width = max((cell["col"] for cell in table["cells"]), default=-1) + 1
    rows: dict[int, dict[int, dict]] = {}
    for cell in table["cells"]:
        rows.setdefault(cell["row"], {})[cell["col"]] = cell
    return width, [rows[r] for r in sorted(rows)]


def to_markdown(doc: dict) -> str:
    lines = [f"<!-- {LABEL}. {RAG_NOTE} -->", "",
             f"# Page {doc['page']} — tables (PROCESSED output, not raw library output)", ""]
    if doc["status"] == "detection_failed":
        lines.append(f"Table detection failed: {doc.get('error')}")
    if not doc["tables"] and doc["status"] != "detection_failed":
        lines.append("No table detected on this page.")
    lines += [f"- {note}" for note in doc.get("notes", [])]
    for table in doc["tables"]:
        lines += ["", f"## Table {table['table_index']} — {table['rows']}×{table['cols']}, "
                      f"status: {table['status']}, direction: {table['direction']}", ""]
        lines += [f"- ⚠ {warning}" for warning in table["warnings"]]
        continuation = table.get("possible_continuation_of")
        if continuation:
            lines.append(f"- Possible continuation of page {continuation['page']} table "
                         f"{continuation['table_index']} (not merged).")
        if table["direction"] == "rtl":
            lines.append("- c1 is the first logical column (the rightmost cell on the page).")
        width, rows = grid(table)
        if not width:
            continue
        lines += ["", "| " + " | ".join(f"c{i + 1}" for i in range(width)) + " |", "|" + "---|" * width]
        for row in rows:
            lines.append("| " + " | ".join(_md_cell(row[c]) if c in row else "" for c in range(width)) + " |")
    lines += ["", "⚠ = uncertain cell (value check failed or unreliable glyphs); raw_cell_text is in "
                  f"page_{doc['page']:03d}_tables.json."]
    return "\n".join(lines) + "\n"
