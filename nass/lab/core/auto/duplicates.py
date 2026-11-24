"""Order-insensitive duplicate detection between full-page OCR and visual-crop OCR (REPORT ONLY).

A printed value is paired with the label words on the same line. A candidate duplicate is the same
(value, unit) with the same label in both sources, whatever the word order, e.g.
    page  "المسار الأحمر 18,450"   and   crop  "18,450 المسار الأحمر".
- digits: Arabic-Indic and Persian digits are compared as Western digits;
- separators keep their meaning: an unambiguous thousands/decimal value is compared by its value
  (hybrid.numbers), an ambiguous one ("6,4", "1.234") only by its exact written form, so "6,4" never
  matches "64" or "6.4";
- units stay distinct (% vs ‰, K/M, currency); a number without a label word is never compared, so a
  value repeated legitimately in different contexts is not a candidate;
- labels match when the normalized word sets are equal, or (both with >= 2 words) one contains the other.
Nothing is removed here: candidates are listed in validation_report.json so the effect of visual
masking can be measured.
"""

from __future__ import annotations

import re

from lab.core.auto import dedup
from lab.core.hybrid import numbers

CHECK_VERSION = "1"
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫٬−", "01234567890123456789.,-")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def _value_key(value: dict) -> str:
    if value.get("value") is not None:
        return value["value"]
    return "as-written:" + re.sub(r"\s+", "", value["original"].translate(_DIGITS))


def pairs(text: str) -> list[dict]:
    """(value, unit, label words) per printed number, line by line."""
    found = []
    for number, line in enumerate((text or "").splitlines(), start=1):
        values = numbers.extract_values(line)
        if not values:
            continue
        label = sorted({token for _s, _e, token in dedup.tokens(line)
                        if _HAS_LETTER.search(token) and not any(ch.isdigit() for ch in token)})
        if not label:
            continue
        for value in values:
            found.append({"line": number, "text": line.strip(), "value": _value_key(value), "unit": value.get("unit"),
                          "kind": value.get("kind"), "original": value["original"], "label": label})
    return found


def labels_match(a: list[str], b: list[str]) -> bool:
    sa, sb = set(a), set(b)
    if sa == sb:
        return True
    return min(len(sa), len(sb)) >= 2 and (sa <= sb or sb <= sa)


def candidates(page: int, page_text: str, crops: dict[str, str], *, source: str) -> list[dict]:
    """Value-and-label pairs present in both the page OCR and a crop OCR of the same page."""
    page_pairs = pairs(page_text)
    out, seen = [], set()
    for region_id, crop_text in crops.items():
        for crop_pair in pairs(crop_text):
            for page_pair in page_pairs:
                if (page_pair["value"], page_pair["unit"]) != (crop_pair["value"], crop_pair["unit"]):
                    continue
                if not labels_match(page_pair["label"], crop_pair["label"]):
                    continue
                key = (region_id, page_pair["line"], crop_pair["line"], page_pair["value"])
                if key in seen:
                    continue
                seen.add(key)
                out.append({"page": page, "region_id": region_id, "value": page_pair["value"],
                            "unit": page_pair["unit"], "label": page_pair["label"], "page_source": source,
                            "page_line": page_pair["line"], "page_text": page_pair["text"],
                            "crop_line": crop_pair["line"], "crop_text": crop_pair["text"],
                            "same_order": page_pair["text"] == crop_pair["text"]})
    return out


def summary(found_raw: list[dict], found_final: list[dict], masked_ids: set[str]) -> dict:
    def split(items: list[dict]) -> dict:
        return {"total": len(items),
                "masked_regions": sum(1 for c in items if c["region_id"] in masked_ids),
                "unmasked_regions": sum(1 for c in items if c["region_id"] not in masked_ids)}

    return {"version": CHECK_VERSION, "raw_page_ocr": split(found_raw), "final_document": split(found_final),
            "note": "report only: value-and-label pairs found in both the page OCR and a crop OCR (any word "
                    "order); nothing was removed by this check"}
