"""Synthetic Nass-like runs for tests. No real document content is used."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from docprep.config import Config, load_config


def make_config(**overrides: Any) -> Config:
    """Code defaults only (the project TOML is not read), with optional overrides."""
    return load_config(None, overrides or None, use_default_file=False)


def el(eid: str, page: int, order: int, content: str | None, *, type: str = "text", bbox: list[float] | None = None,
       source: str = "pymupdf_processed", state: str = "ok", **extra: Any) -> dict[str, Any]:
    data = {"id": eid, "page": page, "reading_order": order, "type": type, "bbox_pt": bbox, "content": content,
            "source": source, "model": None, "validation_state": state, "provenance": {}}
    data.update(extra)
    return data


def write_run(root: Path, name: str, elements: list[dict[str, Any]], *, page_count: int | None = None,
              routes: dict[int, str] | None = None, directions: dict[int, str] | None = None,
              layout: dict[str, Any] | None = None, validation: dict[str, Any] | None = None,
              manifest: bool = True, pdf_bytes: bytes | None = None) -> Path:
    run = root / name
    run.mkdir(parents=True)
    pages = {e["page"] for e in elements}
    count = page_count or (max(pages) if pages else 1)
    document = {
        "schema_version": 1, "kind": "automatic_extraction_document", "assembly_version": "2",
        "document": {"display_name": "synthetic.pdf", "sha256": "0" * 64, "bytes": 0, "page_count": count},
        "run_id": name,
        "pages": [{"page": p, "route": (routes or {}).get(p, "DIGITAL"), "rule": "T",
                   "direction": (directions or {}).get(p, "ltr"), "reading_order_method": "synthetic",
                   "element_ids": [e["id"] for e in elements if e["page"] == p], "dedup_removed": [],
                   "phase1": None, "layout": None} for p in range(1, count + 1)],
        "elements": elements,
    }
    (run / "document.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    (run / "validation_report.json").write_text(json.dumps(validation or {"status": "ok", "flags": []}),
                                                encoding="utf-8")
    if manifest:
        (run / "execution_manifest.json").write_text(json.dumps({"synthetic": True}), encoding="utf-8")
    if layout is not None:
        (run / "plan").mkdir()
        (run / "plan" / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    if pdf_bytes is not None:
        (run / "source.pdf").write_bytes(pdf_bytes)
    return run


def build_pdf(lines: list[dict[str, Any]], page_count: int) -> tuple[bytes, dict[str, list[float]]]:
    """lines: {id, page, y, text, size=11, bold=False, fake_bold=False, color=(0,0,0), x=72}. Returns PDF bytes and the
    element box of each line (approximately the glyph box, as Nass would record it)."""
    import pymupdf

    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page(width=612, height=792)
    boxes = {}
    for line in lines:
        page = document[line["page"] - 1]
        font = "hebo" if line.get("bold") else "helv"
        size = float(line.get("size", 11))
        x, y = float(line.get("x", 72)), float(line["y"])
        color = line.get("color", (0, 0, 0))
        if line.get("fake_bold"):
            # Word-style "bold": the same glyphs filled and stroked (text render mode 2), regular font.
            page.insert_text((x, y), line["text"], fontsize=size, fontname=font, color=color, fill=color,
                             render_mode=2)
        else:
            page.insert_text((x, y), line["text"], fontsize=size, fontname=font, color=color)
        width = pymupdf.get_text_length(line["text"], fontname=font, fontsize=size)
        boxes[line["id"]] = [round(x, 2), round(y - 0.9 * size, 2), round(x + width, 2), round(y + 0.25 * size, 2)]
    data = document.tobytes()
    document.close()
    return data, boxes


def pdf_run(root: Path, name: str, lines: list[dict[str, Any]], *, page_count: int = 1,
            routes: dict[int, str] | None = None) -> Path:
    data, boxes = build_pdf(lines, page_count)
    elements = []
    order_on_page: dict[int, int] = {}
    for line in lines:
        page = line["page"]
        order = order_on_page.get(page, 0)
        order_on_page[page] = order + 1
        elements.append(el(line["id"], page, order, line["text"], type=line.get("type", "text"),
                           bbox=boxes[line["id"]], state=line.get("state", "ok")))
    return write_run(root, name, elements, page_count=page_count, routes=routes, pdf_bytes=data)


BODY = [
    "This report describes the operations of the fictional network in detail.",
    "It covers ridership, punctuality and maintenance for the whole year.",
    "Every value in this synthetic document was generated for testing only.",
    "The figures are not related to any real organisation or person.",
]


def body_lines(prefix: str, page: int, y0: float, count: int = 3) -> list[dict[str, Any]]:
    return [{"id": f"{prefix}{i}", "page": page, "y": y0 + 18 * i, "text": BODY[i % len(BODY)], "size": 11}
            for i in range(count)]
