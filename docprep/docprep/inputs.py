"""Read-only loading of a saved Nass Automatic Extraction run.

Only reads files. Never writes into the run folder.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model import SourceElement

REQUIRED = "document.json"
OPTIONAL = {
    "validation_report": "validation_report.json",
    "execution_manifest": "execution_manifest.json",
    "layout": "plan/layout.json",
}
SOURCE_PDF = "source.pdf"
HASHED_INPUTS = [REQUIRED, *OPTIONAL.values(), SOURCE_PDF]


class InputError(ValueError):
    """The run folder is missing, incomplete or contains invalid JSON."""


@dataclass
class RunInputs:
    run_dir: Path
    run_id: str
    document: dict[str, Any]
    validation_report: dict[str, Any] | None
    execution_manifest: dict[str, Any] | None
    layout: dict[str, Any] | None
    pdf_path: Path | None
    input_hashes: dict[str, str | None]
    warnings: list[str] = field(default_factory=list)
    elements: list[SourceElement] = field(default_factory=list)
    pages: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        doc = self.document.get("document") or {}
        count = doc.get("page_count")
        if isinstance(count, int) and count > 0:
            return count
        return max(self.pages) if self.pages else 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inputs(run_dir: Path) -> dict[str, str | None]:
    return {name: (sha256_file(run_dir / name) if (run_dir / name).is_file() else None) for name in HASHED_INPUTS}


def _read_json(path: Path, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{label} cannot be read: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(f"{label} is not valid JSON: {exc}") from exc


def _validate_document(document: Any) -> None:
    if not isinstance(document, dict):
        raise InputError("document.json must contain a JSON object")
    for key in ("pages", "elements"):
        if not isinstance(document.get(key), list):
            raise InputError(f"document.json is incomplete: '{key}' must be a list")
    for index, element in enumerate(document["elements"]):
        if not isinstance(element, dict):
            raise InputError(f"document.json element #{index} is not an object")
        for key in ("id", "page", "type"):
            if key not in element:
                raise InputError(f"document.json element #{index} has no '{key}'")
        if not isinstance(element["page"], int) or element["page"] < 1:
            raise InputError(f"document.json element {element['id']} has an invalid page number")
    ids = [element["id"] for element in document["elements"]]
    if len(ids) != len(set(ids)):
        raise InputError("document.json contains duplicate element ids")
    for index, page in enumerate(document["pages"]):
        if not isinstance(page, dict) or not isinstance(page.get("page"), int):
            raise InputError(f"document.json page #{index} has no integer 'page'")


def _ordered_elements(document: dict[str, Any]) -> list[SourceElement]:
    """Document-wide Nass order: pages ascending; within a page the page's element_ids order, then any
    remaining elements by reading_order."""
    by_page: dict[int, list[dict]] = {}
    for element in document["elements"]:
        by_page.setdefault(element["page"], []).append(element)
    page_order = {p["page"]: p.get("element_ids") or [] for p in document["pages"]}
    ordered: list[dict] = []
    for page in sorted(by_page):
        items = by_page[page]
        listed = [i for i in page_order.get(page, []) if any(e["id"] == i for e in items)]
        index = {e["id"]: e for e in items}
        rest = sorted((e for e in items if e["id"] not in listed),
                      key=lambda e: (e.get("reading_order") is None, e.get("reading_order") or 0, e["id"]))
        ordered += [index[i] for i in listed] + rest
    out = []
    for position, e in enumerate(ordered):
        bbox = e.get("bbox_pt")
        out.append(SourceElement(
            element_id=e["id"], page=e["page"], element_type=str(e["type"]),
            reading_order=e.get("reading_order") if isinstance(e.get("reading_order"), int) else None,
            bbox=[float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else None,
            text=e.get("content") if isinstance(e.get("content"), str) else None,
            validation_state=e.get("validation_state"), source=e.get("source"), order=position, raw=e))
    return out


def load_run(run_dir: Path | str) -> RunInputs:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise InputError(f"run folder not found: {run_dir}")
    doc_path = run_dir / REQUIRED
    if not doc_path.is_file():
        raise InputError(f"incomplete run: {REQUIRED} is missing in {run_dir}")
    hashes = hash_inputs(run_dir)
    document = _read_json(doc_path, REQUIRED)
    _validate_document(document)
    warnings: list[str] = []
    loaded: dict[str, Any] = {}
    for key, name in OPTIONAL.items():
        path = run_dir / name
        if path.is_file():
            data = _read_json(path, name)
            if not isinstance(data, dict):
                raise InputError(f"{name} must contain a JSON object")
            loaded[key] = data
        else:
            loaded[key] = None
            warnings.append(f"{name} is missing; the related signals are unavailable")
    pdf = run_dir / SOURCE_PDF
    if not pdf.is_file():
        warnings.append("source.pdf is missing; typography signals are unavailable")
    run_id = str(document.get("run_id") or run_dir.name)
    inputs = RunInputs(run_dir=run_dir, run_id=run_id, document=document,
                       validation_report=loaded["validation_report"], execution_manifest=loaded["execution_manifest"],
                       layout=loaded["layout"], pdf_path=pdf if pdf.is_file() else None, input_hashes=hashes,
                       warnings=warnings)
    inputs.pages = {p["page"]: p for p in document["pages"]}
    inputs.elements = _ordered_elements(document)
    return inputs
