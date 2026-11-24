"""Repeated page headers and footers, and page numbers ("page furniture")."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

from .config import FurnitureConfig
from .model import Unit
from .text_patterns import is_page_number, match_text


@dataclass
class FurnitureInfo:
    kind: str            # header | footer | page_number
    reason: str
    pages: list[int]

    def to_dict(self) -> dict:
        return asdict(self)


def furniture_key(text: str) -> str:
    return re.sub(r"\d+", "#", match_text(text).lower())


def detect_furniture(units: list[Unit], page_sizes: dict[int, tuple[float, float]], page_count: int,
                     cfg: FurnitureConfig) -> dict[str, FurnitureInfo]:
    found: dict[str, FurnitureInfo] = {}
    need = max(cfg.min_repeat_pages, math.ceil(cfg.min_repeat_share * max(page_count, 1)))
    banded: dict[tuple[str, str], list[Unit]] = defaultdict(list)
    for unit in units:
        bbox, size = unit.bbox, page_sizes.get(unit.page)
        if not unit.clean or bbox is None or size is None:
            continue
        height = size[1]
        if bbox[3] - bbox[1] > 0.6 * height:
            continue  # page-wide element: not a running header/footer
        band = "top" if bbox[3] <= cfg.top_band * height else "bottom" if bbox[1] >= cfg.bottom_band * height else None
        if band is None:
            continue
        if is_page_number(unit.clean):
            found[unit.unit_id] = FurnitureInfo("page_number", f"page number in the {band} band", [unit.page])
            continue
        banded[(band, furniture_key(unit.clean))].append(unit)
    if page_count >= 2:
        for (band, key), group in banded.items():
            pages = sorted({u.page for u in group})
            if len(pages) >= need and key:
                kind = "header" if band == "top" else "footer"
                for unit in group:
                    found[unit.unit_id] = FurnitureInfo(kind, f"same text in the {band} band on {len(pages)} pages",
                                                        pages)
        # Page-OCR text has no coordinates: a leading span repeated on many pages is a running header.
        leading: dict[str, list[Unit]] = defaultdict(list)
        for unit in units:
            if unit.span is not None and unit.span.start == 0 and unit.clean:
                leading[furniture_key(unit.clean)].append(unit)
        for key, group in leading.items():
            pages = sorted({u.page for u in group})
            if key and len(pages) >= max(need, 3):
                for unit in group:
                    found.setdefault(unit.unit_id, FurnitureInfo(
                        "header", f"same leading text of the page OCR on {len(pages)} pages", pages))
    return found
