"""Numeric value and unit validation for OCR/VLM outputs (Phase 2). Validation only: nothing is
corrected, converted or replaced, and raw outputs are never changed.

Each value is extracted with its unit/format: % and ‰ (never interchangeable), K/M/B suffixes,
currency, times (10:45 AM), durations (h/min/s), dates, thousands/decimal separators (also
Arabic-Indic digits, "٫" decimal and "٬" thousands) and the minus sign (U+2212 is read as "-" for
comparison only). Flags for review: ambiguous separators (e.g. "٦,٤"), the same value with a
different unit in two providers' outputs for one region, and long arithmetic sequences that look
like axis ticks (stored as axis candidates, not data values).
"""

from __future__ import annotations

import re
from collections import Counter

VALIDATION_VERSION = "1"
_DIGITS = "0-9٠-٩۰-۹"
_TO_ASCII = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARABIC_DIGITS = re.compile(r"[٠-٩۰-۹]")
CURRENCY_PREFIX = r"(?:US\$|\$|€|£|¥|SAR|USD|EUR|GBP|AED|﷼)"
CURRENCY_SUFFIX = r"(?:SAR|USD|EUR|GBP|AED|ريال|ر\.س|﷼|\$|€|£)"
SCALE = r"(?:bn|mn|[KkMB]|ألف|مليون|مليار)"
DURATION = r"(?:hours?|hrs?|h|minutes?|mins?|min|seconds?|secs?|s|ساعة|ساعات|دقيقة|دقائق|ثانية|ثوان)"

TIME = re.compile(rf"(?<![{_DIGITS}])([{_DIGITS}]{{1,2}}):([{_DIGITS}]{{2}})(?::([{_DIGITS}]{{2}}))?\s?(AM|PM|am|pm|ص|م)?(?![{_DIGITS}])")
DATE = re.compile(rf"(?<![{_DIGITS}])(?:[{_DIGITS}]{{4}}[-/][{_DIGITS}]{{1,2}}[-/][{_DIGITS}]{{1,2}}"
                  rf"|[{_DIGITS}]{{1,2}}[-/][{_DIGITS}]{{1,2}}[-/][{_DIGITS}]{{2,4}})(?![{_DIGITS}])")
NUMBER = re.compile(
    rf"(?P<cur_pre>{CURRENCY_PREFIX}\s?)?(?P<sign>[-−+])?(?P<body>[{_DIGITS}]+(?:[.,٫٬][{_DIGITS}]+)*)"
    rf"(?P<suffix>\s?(?:%|٪|‰|؉|{SCALE}(?![A-Za-z])|{DURATION}(?![A-Za-z])|{CURRENCY_SUFFIX}))?",
)


def _separators(body: str) -> tuple[str | None, list[str]]:
    """(normalized ASCII number or None, flags) without guessing an ambiguous separator."""
    digits = body.translate(_TO_ASCII)
    flags: list[str] = []
    arabic = bool(_ARABIC_DIGITS.search(body))
    if "٫" in body or "٬" in body:  # Arabic decimal / thousands separators are unambiguous
        return digits.replace("٬", "").replace("٫", "."), flags
    separators = re.findall(r"[.,]", digits)
    if not separators:
        return digits, flags
    if arabic and "," in separators:
        flags.append("ambiguous separator: ',' with Arabic-Indic digits may be a decimal comma or a thousands "
                     "separator")
        return None, flags
    if len(set(separators)) == 2:  # both kinds: the last one is the decimal separator
        decimal = digits.rfind(".") > digits.rfind(",") and "." or ","
        thousands = "," if decimal == "." else "."
        return digits.replace(thousands, "").replace(decimal, "."), flags
    sep = separators[0]
    groups = digits.split(sep)
    if len(separators) > 1:  # repeated separator: thousands grouping when every group has 3 digits
        if all(len(group) == 3 for group in groups[1:]):
            return digits.replace(sep, ""), flags
        flags.append(f"ambiguous separator: repeated '{sep}' with irregular grouping")
        return None, flags
    tail = groups[1]
    if sep == "," and len(tail) == 3:
        return digits.replace(",", ""), flags  # 1,284 -> thousands (common English usage)
    if sep == "." and len(tail) == 3:
        flags.append("ambiguous separator: '.' followed by three digits may be a decimal point or a thousands "
                      "separator")
        return None, flags
    if sep == ",":
        flags.append("ambiguous separator: ',' followed by 1-2 digits is probably a decimal comma")
        return None, flags
    return digits, flags


def _unit(suffix: str | None, currency_prefix: str | None) -> tuple[str | None, str]:
    text = (suffix or "").strip()
    if text in {"%", "٪"}:
        return "%", "percent"
    if text in {"‰", "؉"}:
        return "‰", "per mille"
    if currency_prefix:
        return currency_prefix.strip(), "currency"
    if re.fullmatch(CURRENCY_SUFFIX, text):
        return text, "currency"
    if re.fullmatch(SCALE, text):
        return {"k": "K", "mn": "M", "bn": "B", "ألف": "K", "مليون": "M", "مليار": "B"}.get(text, text), "scale"
    if re.fullmatch(DURATION, text):
        lowered = text.lower()
        unit = ("h" if lowered.startswith("h") or text in {"ساعة", "ساعات"} else
                "min" if lowered.startswith("min") or text in {"دقيقة", "دقائق"} else "s")
        return unit, "duration"
    return None, "number"


def extract_values(text: str) -> list[dict]:
    """Numbers with unit/format, in reading order. Times and dates are kept as their own kinds."""
    taken: list[tuple[int, int]] = []
    values: list[dict] = []
    for kind, pattern in (("date", DATE), ("time", TIME)):
        for match in pattern.finditer(text):
            taken.append(match.span())
            values.append({"original": match.group(0), "kind": kind, "unit": None, "value": match.group(0).translate(_TO_ASCII),
                           "position": match.start(), "flags": []})
    for match in NUMBER.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        normalized, flags = _separators(match.group("body"))
        unit, kind = _unit(match.group("suffix"), match.group("cur_pre"))
        sign = "-" if match.group("sign") in {"-", "−"} else ""
        values.append({"original": match.group(0), "kind": kind, "unit": unit,
                       "value": (sign + normalized) if normalized is not None else None,
                       "minus_sign": match.group("sign") if match.group("sign") in {"-", "−"} else None,
                       "position": match.start(), "flags": flags})
    values.sort(key=lambda item: item["position"])
    return values


def axis_candidates(values: list[dict], *, min_length: int = 4) -> list[list[int]]:
    """Runs of >= min_length plain numbers (same unit) forming an arithmetic sequence with a
    non-zero step, e.g. 0 50 100 150 200: probably axis ticks, not data values (index lists)."""
    runs: list[list[int]] = []
    index = 0
    numeric = [(i, v) for i, v in enumerate(values) if v["kind"] in {"number", "percent", "scale"} and v["value"]]
    while index < len(numeric):
        run = [numeric[index]]
        step = None
        for nxt in numeric[index + 1:]:
            prev = run[-1]
            try:
                delta = round(float(nxt[1]["value"]) - float(prev[1]["value"]), 6)
            except ValueError:
                break
            if nxt[1]["unit"] != prev[1]["unit"] or delta == 0 or (step is not None and delta != step):
                break
            step = delta
            run.append(nxt)
        if len(run) >= min_length:
            runs.append([i for i, _v in run])
            index += len(run)
        else:
            index += 1
    return runs


def validate(text: str) -> dict:
    values = extract_values(text)
    axes = axis_candidates(values)
    axis_positions = {i for run in axes for i in run}
    for i, value in enumerate(values):
        value["axis_candidate"] = i in axis_positions
    flags = [f"{v['original']}: {flag}" for v in values for flag in v["flags"]]
    flags += [f"axis-like sequence: {' '.join(values[i]['original'] for i in run)} (kept as axis candidates, not "
              "data values)" for run in axes]
    data_values = [v for v in values if not v["axis_candidate"]]
    return {"version": VALIDATION_VERSION, "values": values, "data_values": len(data_values),
            "axis_candidates": [[values[i]["original"] for i in run] for run in axes],
            "flags": flags, "needs_review": bool(flags),
            "note": "Validation only: no value is corrected, converted or replaced."}


def _value_units(validation: dict) -> dict[str, Counter]:
    found: dict[str, Counter] = {}
    for value in validation["values"]:
        if value["value"] is None or value["kind"] in {"date", "time"} or value.get("axis_candidate"):
            continue
        found.setdefault(value["value"].lstrip("+"), Counter())[value["unit"] or "-"] += 1
    return found


def unit_conflicts(validations: dict[str, dict]) -> list[dict]:
    """Same value with different units in different outputs of one region (e.g. 3.01% vs 3.01‰)."""
    per_output = {key: _value_units(validation) for key, validation in validations.items()}
    conflicts = []
    values = set().union(*[set(units) for units in per_output.values()]) if per_output else set()
    for value in sorted(values):
        units = {key: sorted(found[value]) for key, found in per_output.items() if value in found}
        distinct = {unit for unit_list in units.values() for unit in unit_list}
        if len(units) >= 2 and len(distinct) >= 2:
            conflicts.append({"value": value, "units_by_output": units})
    return conflicts


def unit_mismatches(reference_text: str, output_text: str) -> dict:
    """Benchmark scoring: values present in both with a different unit (value equal, unit different)."""
    reference = _value_units(validate(reference_text))
    output = _value_units(validate(output_text))
    mismatches = []
    for value in sorted(set(reference) & set(output)):
        ref_units, out_units = reference[value], output[value]
        extra = out_units - ref_units
        missing = ref_units - out_units
        count = min(sum(extra.values()), sum(missing.values()))
        if count:
            mismatches.append({"value": value, "reference_units": dict(ref_units), "output_units": dict(out_units),
                               "count": count})
    return {"count": sum(item["count"] for item in mismatches), "items": mismatches}
