"""Arabic-Indic / Persian digit checks for OCR outputs (REPORT ONLY: nothing is ever corrected).

Static checks (every OCR text):
  arabic_digits_ungrouped   an Arabic-Indic/Persian integer of 4+ digits without a grouping separator
                            (e.g. a KPI value "٣٬٤٥٠" read as "٥٠٣٤٣": verify the digits); a 4-digit
                            value from 1300 to 2100 is reported as a possible year instead
  mixed_digit_scripts       Western and Arabic-Indic/Persian digits inside one numeric token
  irregular_grouping        grouping that is inconsistent (mixed separator scripts, groups that are not
                            3 digits after an Arabic thousands separator or between repeated separators)
Decimal-comma ambiguity stays with hybrid.numbers (unchanged). Dates, times, version strings and
identifiers (digits joined to letters or codes, e.g. "ISO-9001", "A12", "v2.1.3") are recognized and
not treated as ordinary values.

Second-read comparison (primary OCR vs the verification read of the same crop): numeric tokens are
aligned by their label words on the line, or by position when both reads have the same number of
values; digit shapes are normalized for comparison only, separators/percent/units are kept:
  second_read_agreement     same digits and same separator/unit shape: consistent evidence, NOT proof
  separator_difference      same digits, different separators/units: kept visible (never "agreement")
  possible_digit_reversal   the digit sequence of one read is the reverse of the other
  digit_disagreement        different digits
  verification_inconclusive the value cannot be aligned reliably
"""

from __future__ import annotations

import re

from lab.core.auto import dedup
from lab.core.hybrid import numbers

DIGITS_VERSION = "1"
WESTERN, ARABIC_INDIC, PERSIAN = "western", "arabic_indic", "persian"
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
_TO_ASCII = str.maketrans(_ARABIC + _PERSIAN, "0123456789" * 2)
_D = "0-9٠-٩۰-۹"
_SEPS = ".,٫٬"
TOKEN = re.compile(rf"[+\-−]?[{_D}]+(?:[{_SEPS}][{_D}]+)*\s?[%٪‰؉]?")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_VERSION_WORDS = ("v", "version", "ver", "الإصدار", "الاصدار", "إصدار", "اصدار", "نسخة")
EASTERN = re.compile(r"[٠-٩۰-۹]")
NOTE = ("Report only. Agreement between two readings is consistent evidence, not proof of correctness; "
        "no value was corrected.")


def script_of(ch: str) -> str | None:
    if "0" <= ch <= "9":
        return WESTERN
    if ch in _ARABIC:
        return ARABIC_INDIC
    if ch in _PERSIAN:
        return PERSIAN
    return None


def has_eastern_digits(text: str) -> bool:
    return bool(EASTERN.search(text or ""))


def _excluded_spans(line: str) -> list[tuple[int, int, str]]:
    spans = [(m.start(), m.end(), "date") for m in numbers.DATE.finditer(line)]
    spans += [(m.start(), m.end(), "time") for m in numbers.TIME.finditer(line)]
    return spans


def tokens(text: str) -> list[dict]:
    """Numeric tokens per line with their kind (value / date / time / version / identifier) and label."""
    out = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        excluded = _excluded_spans(line)
        label = sorted({t for _s, _e, t in dedup.tokens(line) if _LETTER.search(t) and not any(c.isdigit() for c in t)})
        for match in TOKEN.finditer(line):
            start, end = match.start(), match.end()
            raw = match.group().strip()
            kind = "value"
            hit = next((k for s, e, k in excluded if s <= start < e), None)
            before = line[start - 1] if start > 0 else ""
            after = line[end] if end < len(line) else ""
            words_before = line[max(0, start - 16):start].lower().replace("-", " ").split()
            last_word = words_before[-1] if words_before else ""  # the word just before the number
            glued_v = before.lower() == "v" and (start < 2 or not _LETTER.search(line[start - 2]))
            if hit:
                kind = hit
            elif raw.count(".") >= 2 or raw.count("٫") >= 2 or glued_v or last_word in _VERSION_WORDS:
                kind = "version"
            elif _LETTER.search(before) or _LETTER.search(after) or before in "-/_#" or (after in "-/_" and
                                                                                         end + 1 < len(line)):
                kind = "identifier"
            body = raw.rstrip("%٪‰؉ ").lstrip("+-−")
            digit_chars = [c for c in body if script_of(c)]
            unit = raw[len(raw.rstrip("%٪‰؉")):].strip() or None
            out.append({"line": line_no, "position": start, "original": raw, "kind": kind, "label": label,
                        "digits": "".join(digit_chars).translate(_TO_ASCII),
                        "scripts": sorted({script_of(c) for c in digit_chars}),
                        "separators": [c for c in body if c in _SEPS],
                        "shape": _shape(raw), "unit": {"٪": "%", "؉": "‰"}.get(unit, unit)})
    return out


def _shape(raw: str) -> str:
    """Digits -> 9; separators kept by meaning (Arabic ٬ thousands = 'T', ٫ decimal = 'D'); units kept."""
    out = []
    for ch in raw:
        if script_of(ch):
            out.append("9")
        elif ch == "٬":
            out.append("T")
        elif ch == "٫":
            out.append("D")
        elif ch in "%٪":
            out.append("%")
        elif ch in "‰؉":
            out.append("‰")
        elif ch in ".,-−+":
            out.append("-" if ch == "−" else ch)
    return "".join(out)


def _irregular_grouping(token: dict, body: str) -> str | None:
    seps = token["separators"]
    if not seps:
        return None
    kinds = set(seps)
    if len(kinds) > 1 and kinds not in ({",", "."}, {"٬", "٫"}):
        return f"mixed separator scripts {''.join(sorted(kinds))}"
    integer = re.split("[٫]", body)[0] if "٫" in body else body
    if "٬" in integer:
        groups = integer.split("٬")
        if any(len(g) != 3 for g in groups[1:]) or not 1 <= len(groups[0]) <= 3:
            return "Arabic thousands separator with groups that are not 3 digits"
    for sep in (",", "."):
        if seps.count(sep) >= 2:
            groups = body.split(sep)
            if any(len(g) != 3 for g in groups[1:]):
                return f"repeated '{sep}' with groups that are not 3 digits"
    return None


def static_flags(text: str) -> list[dict]:
    flags = []
    for token in tokens(text):
        if token["kind"] != "value":
            continue
        where = {"token": token["original"], "line": token["line"], "position": token["position"]}
        if len(token["scripts"]) > 1:
            flags.append(where | {"code": "mixed_digit_scripts", "severity": "warning",
                                  "message": f"{token['original']}: Western and Arabic-Indic/Persian digits mixed in "
                                             "one value"})
        eastern = bool(set(token["scripts"]) & {ARABIC_INDIC, PERSIAN})
        if eastern and not token["separators"] and len(token["digits"]) >= 4:
            value = int(token["digits"])
            if len(token["digits"]) == 4 and 1300 <= value <= 2100:
                flags.append(where | {"code": "possible_year", "severity": "info",
                                      "message": f"{token['original']}: 4-digit value read as a possible year"})
            else:
                flags.append(where | {"code": "arabic_digits_ungrouped", "severity": "warning",
                                      "message": f"{token['original']}: {len(token['digits'])}-digit Arabic-Indic/Persian "
                                                 "value without a grouping separator; verify the digits"})
        body = token["original"].rstrip("%٪‰؉ ").lstrip("+-−")
        irregular = _irregular_grouping(token, body)
        if irregular:
            flags.append(where | {"code": "irregular_grouping", "severity": "warning",
                                  "message": f"{token['original']}: {irregular}"})
    return flags


def _labels_match(a: list[str], b: list[str]) -> bool:
    sa, sb = set(a), set(b)
    return bool(sa) and (sa == sb or (min(len(sa), len(sb)) >= 2 and (sa <= sb or sb <= sa)))


def classify(primary: dict, verification: dict) -> str:
    if primary["digits"] == verification["digits"]:
        same = primary["shape"] == verification["shape"] and primary["unit"] == verification["unit"]
        return "second_read_agreement" if same else "separator_difference"
    if len(primary["digits"]) >= 2 and primary["digits"] != primary["digits"][::-1] \
            and primary["digits"] == verification["digits"][::-1]:
        return "possible_digit_reversal"
    return "digit_disagreement"


OUTCOMES = ("second_read_agreement", "separator_difference", "possible_digit_reversal", "digit_disagreement",
            "verification_inconclusive")


def compare(primary_text: str, verification_text: str) -> dict:
    """Align the numeric values of two readings of the same crop and classify each primary value."""
    primary = [t for t in tokens(primary_text) if t["kind"] == "value"]
    second = [t for t in tokens(verification_text) if t["kind"] == "value"]
    used: set[int] = set()
    results = []
    by_order = len(primary) == len(second)
    for index, token in enumerate(primary):
        candidates = [i for i, v in enumerate(second) if i not in used and _labels_match(token["label"], v["label"])]
        same_digits = [i for i in candidates if second[i]["digits"] == token["digits"]]
        chosen, method = None, None
        if len(candidates) == 1:
            chosen, method = candidates[0], "label"
        elif len(same_digits) == 1:
            chosen, method = same_digits[0], "label + digits"
        elif by_order and index not in used and not candidates:
            chosen, method = index, "position (same number of values in both reads)"
        if chosen is None:
            results.append({"outcome": "verification_inconclusive", "primary": token["original"], "verification": None,
                            "label": token["label"], "line": token["line"],
                            "reason": "no unique value with the same label (or same position) in the second read"})
            continue
        used.add(chosen)
        other = second[chosen]
        results.append({"outcome": classify(token, other), "primary": token["original"], "verification": other["original"],
                        "primary_digits": token["digits"], "verification_digits": other["digits"],
                        "label": token["label"], "line": token["line"], "alignment": method})
    counts = {name: sum(1 for r in results if r["outcome"] == name) for name in OUTCOMES}
    return {"version": DIGITS_VERSION, "values": results, "counts": counts,
            "primary_values": len(primary), "verification_values": len(second), "note": NOTE}
