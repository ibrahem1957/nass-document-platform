"""Language-aware text, number, caption and list helpers.

Normalization is used only for comparison.  Output evidence always retains the
verbatim source substrings and their offsets.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
ARABIC_LETTERS = r"\u0621-\u063a\u0641-\u064a\u066e-\u066f\u0671-\u06d3\u06fa-\u06fc"
WORD_RE = re.compile(rf"[A-Za-z{ARABIC_LETTERS}]{{2,}}")
CAPTION_RE = re.compile(
    r"(?i)^(?P<kind>figure|fig\.?|table|الشكل|شكل|الجدول|جدول)\s*"
    r"(?:[:：]\s*)?(?P<number>[0-9٠-٩۰-۹]+)(?:\s*[:：.\-–—])?"
)
EXPLICIT_REF_RE = re.compile(
    r"(?i)(?P<kind>figure|fig\.?|table|الشكل|شكل|الجدول|جدول)\s*"
    r"(?P<number>[0-9٠-٩۰-۹]+|التالي(?:ة)?)?"
)
NUMBER_RE = re.compile(
    r"(?:(?<![\w])|(?<=و))(?P<number>[+\-]?[0-9٠-٩۰-۹]+(?:[,.٬،][0-9٠-٩۰-۹]{3})*"
    r"(?:[.٫,][0-9٠-٩۰-۹]+)?)(?:(?P<scale>K|M|k|m)|\s*(?P<symbol>%|٪|‰))?"
    rf"(?:\s*(?P<unit>[A-Za-z{ARABIC_LETTERS}]{{1,12}}))?"
)
LIST_MARKER_RE = re.compile(
    r"(?m)(?P<prefix>^|[\n\r])(?P<indent>[ \t]*)(?P<marker>[•●◦▪‣*\-–—]|[0-9٠-٩۰-۹]+[.)]|[A-Za-z][.)])\s+"
)
RTL_TRAILING_MARKER_RE = re.compile(r"(?m)(?P<body>[^\n\r]{3,}?)[ \t]+(?P<marker>[•●◦▪‣\-–—])(?=\s*(?:\r?\n|$))")

STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "على", "في", "من", "إلى", "الى",
    "عن", "مع", "هذا", "هذه", "التي", "الذي", "تم", "كما", "كل",
}


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).translate(ARABIC_DIGITS)
    value = ARABIC_DIACRITICS.sub("", value)
    value = value.replace("ـ", "").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    value = re.sub(r"[^0-9A-Za-z\u0600-\u06ff%‰]+", " ", value.casefold())
    return " ".join(value.split())


def content_terms(text: str) -> set[str]:
    return {term for term in WORD_RE.findall(normalize_text(text)) if term not in STOPWORDS}


def lexical_match(left: str, right: str) -> tuple[float, list[str]]:
    a, b = content_terms(left), content_terms(right)
    shared = sorted(a & b)
    return (len(shared) / max(1, len(a | b)), shared)


def caption_signature(text: str) -> tuple[str, str | None] | None:
    match = CAPTION_RE.match(text.strip())
    if not match:
        return None
    kind = "table" if "table" in match.group("kind").casefold() or "جدول" in match.group("kind") else "figure"
    number = match.group("number")
    return kind, number.translate(ARABIC_DIGITS) if number else None


@dataclass(frozen=True)
class NumericToken:
    raw: str
    start: int
    end: int
    value: str
    dimension: str
    unit: str | None
    scale: int = 1
    normalization: str = "literal"


# Dimensions whose tokens carry no comparable numeric value (never matched, never in a numeric conflict).
NON_VALUE_DIMENSIONS = {"identifier", "ambiguous_number"}
_VALID_COMMA_GROUPING = re.compile(r"[+\-]?\d{1,3}(?:,\d{3})+(?:[.٫]\d+)?")


def separator_status(raw_number: str) -> str:
    """'ok', or 'ambiguous' when a comma (',' or Arabic '،') is not a valid thousands separator.

    A comma followed by anything other than groups of exactly three digits (e.g. '٦,٤', '2,15', '12,5') may be
    an OCR rendering of the Arabic decimal separator '٫' or a European decimal comma: the intended value cannot
    be established from the token alone, so no value is claimed. '٫' is the Arabic decimal separator and '٬' the
    Arabic thousands separator; both are unambiguous."""
    value = raw_number.translate(ARABIC_DIGITS).replace("،", ",")
    if "," not in value:
        return "ok"
    return "ok" if _VALID_COMMA_GROUPING.fullmatch(value) else "ambiguous"


def _numeric_value(raw_number: str, suffix: str | None) -> str | None:
    value = raw_number.translate(ARABIC_DIGITS).replace("٬", "").replace("،", "").replace(",", "")
    value = value.replace("٫", ".")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if suffix and suffix.casefold() == "k":
        number *= 1_000
    elif suffix and suffix.casefold() == "m":
        number *= 1_000_000
    return format(number.normalize(), "f")


DIMENSION_ALIASES = {
    "time": {"minute", "minutes", "min", "hour", "hours", "دقيقه", "دقائق", "ساعه", "ساعات"},
    "count": {"case", "cases", "item", "items", "incident", "incidents", "حاله", "حالات", "حادث", "حوادث"},
    "currency": {"sar", "usd", "riyal", "riyals", "ريال", "ريالات"},
    "distance": {"km", "kilometer", "kilometers", "meter", "meters", "كم", "كيلومتر", "كيلومترات", "متر", "امتار"},
    "mass": {"kg", "kilogram", "kilograms", "كجم", "كيلوجرام"},
    "year": {"year", "years", "سنه", "سنوات", "عام", "اعوام"},
    "people": {"passenger", "passengers", "rider", "riders", "راكب", "ركاب"},
    "trip": {"trip", "trips", "ride", "rides", "رحله", "رحلات"},
}
KNOWN_UNITS = set().union(*DIMENSION_ALIASES.values())
IDENTIFIER_PREFIX_RE = re.compile(
    r"(?i)(?:(?:agent|id|identifier|record|رقم|معرف|المعرف)"
    # A figure/table label number ("Figure 11", "الشكل 11") names a visual; it is not a measured value.
    r"|(?<![\w؀-ۿ])(?:figure|fig\.?|table|chart|الشكل|شكل|الجدول|جدول|المخطط|مخطط))\s*$"
)
SCALE_ALIASES = {
    "k": 1_000, "thousand": 1_000, "thousands": 1_000, "الف": 1_000, "الاف": 1_000,
    "m": 1_000_000, "million": 1_000_000, "millions": 1_000_000,
    "مليون": 1_000_000, "ملايين": 1_000_000,
}
SMALL_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "صفر": 0, "واحد": 1, "واحدة": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2,
    "ثلاثة": 3, "ثلاث": 3, "اربعة": 4, "أربعة": 4, "اربع": 4, "أربع": 4,
    "خمسة": 5, "خمس": 5,
    "ستة": 6, "ست": 6, "سبعة": 7, "سبع": 7, "ثمانية": 8, "ثمان": 8,
    "تسعة": 9, "تسع": 9, "عشرة": 10, "عشر": 10,
}
CLOCK_RE = re.compile(r"(?i)(?<!\w)([0-9٠-٩۰-۹]{1,2})(?::([0-9٠-٩۰-۹]{2}))?\s*(am|pm)(?!\w)")
ARABIC_CLOCK_RE = re.compile(
    r"الساعة\s+(الواحدة|الثانية|الثالثة|الرابعة|الخامسة|السادسة|السابعة|الثامنة|التاسعة|العاشرة|الحادية\s+عشرة|الثانية\s+عشرة)\s+(صباحا|صباحًا|مساء|مساءً)"
)
ARABIC_HOURS = {
    "الواحدة": 1, "الثانية": 2, "الثالثة": 3, "الرابعة": 4, "الخامسة": 5,
    "السادسة": 6, "السابعة": 7, "الثامنة": 8, "التاسعة": 9, "العاشرة": 10,
    "الحادية عشرة": 11, "الثانية عشرة": 12,
}


def numeric_tokens(text: str) -> list[NumericToken]:
    tokens: list[NumericToken] = []
    occupied: list[tuple[int, int]] = []
    for match in CLOCK_RE.finditer(text):
        hour = int(match.group(1).translate(ARABIC_DIGITS)) % 12
        minute = int((match.group(2) or "0").translate(ARABIC_DIGITS))
        if match.group(3).casefold() == "pm":
            hour += 12
        tokens.append(NumericToken(match.group(0), match.start(), match.end(), f"{hour:02d}:{minute:02d}",
                                   "time_of_day", None, 1, "explicit_clock"))
        occupied.append((match.start(), match.end()))
    for raw_match in ARABIC_CLOCK_RE.finditer(text):
        hour_name = " ".join(normalize_text(raw_match.group(1)).split())
        hour = next((value for name, value in ARABIC_HOURS.items() if normalize_text(name) == hour_name), None)
        if hour is None:
            continue
        if normalize_text(raw_match.group(2)).startswith("مساء"):
            hour = hour % 12 + 12
        tokens.append(NumericToken(raw_match.group(0), raw_match.start(), raw_match.end(), f"{hour:02d}:00",
                                   "time_of_day", None, 1, "arabic_time_expression"))
        occupied.append((raw_match.start(), raw_match.end()))
    for match in NUMBER_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        suffix = match.group("scale") or match.group("symbol")
        unit = match.group("unit")
        if separator_status(match.group("number")) == "ambiguous":
            # Preserve the raw token, claim no value (it may be 6.4 or 64): never matched or compared.
            end = match.end("symbol") if match.group("symbol") else match.end("number")
            tokens.append(NumericToken(text[match.start():end], match.start(), end,
                                       "ambiguous:" + match.group("number").translate(ARABIC_DIGITS),
                                       "ambiguous_number", None, 1, "ambiguous_decimal_or_thousands_separator"))
            continue
        value = _numeric_value(match.group("number"), suffix)
        if value is None:
            continue
        unit_normalized = normalize_text(unit or "")
        scale = SCALE_ALIASES.get(unit_normalized, SCALE_ALIASES.get((suffix or "").casefold(), 1))
        if scale != 1 and not suffix:
            value = format((Decimal(value) * scale).normalize(), "f")
        if unit and unit_normalized not in KNOWN_UNITS and unit_normalized not in SCALE_ALIASES:
            unit = None
            unit_normalized = ""
        if suffix in ("%", "٪"):
            dimension = "percent"
            unit = None
        elif suffix == "‰":
            dimension = "per_mille"
            unit = None
        else:
            dimension = unit_normalized or "scalar"
        prefix = text[max(0, match.start() - 24):match.start()]
        if suffix is None and IDENTIFIER_PREFIX_RE.search(prefix):
            dimension = "identifier"
            unit = None
        numeric_end = match.end("symbol") if match.group("symbol") else (
            match.end("scale") if match.group("scale") else match.end("number")
        )
        if unit_normalized in SCALE_ALIASES:
            unit = None
            dimension = "scalar"
        end = match.end("unit") if (unit or unit_normalized in SCALE_ALIASES) else numeric_end
        tokens.append(NumericToken(text[match.start():end], match.start(), end, value, dimension, unit,
                                   scale, "scaled_expression" if scale != 1 else "literal"))
    word_pattern = "|".join(sorted((re.escape(value) for value in SMALL_NUMBER_WORDS), key=len, reverse=True))
    scale_words = {
        "thousand": 1_000, "thousands": 1_000, "ألف": 1_000, "الف": 1_000,
        "آلاف": 1_000, "الاف": 1_000, "million": 1_000_000, "millions": 1_000_000,
        "مليون": 1_000_000, "ملايين": 1_000_000,
    }
    scale_pattern = "|".join(sorted((re.escape(value) for value in scale_words), key=len, reverse=True))
    for match in re.finditer(rf"(?i)(?<!\w)({word_pattern})\s+({scale_pattern})(?!\w)", text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        value = SMALL_NUMBER_WORDS[match.group(1).casefold()] * scale_words[match.group(2).casefold()]
        tokens.append(NumericToken(match.group(0), match.start(), match.end(), str(value), "scalar", None,
                                   scale_words[match.group(2).casefold()], "number_word_scale"))
        occupied.append((match.start(), match.end()))
    for word, value in SMALL_NUMBER_WORDS.items():
        for raw_match in re.finditer(rf"(?i)(?<!\w){re.escape(word)}(?!\w)", text):
            if not any(raw_match.start() < end and raw_match.end() > start for start, end in occupied):
                tokens.append(NumericToken(raw_match.group(0), raw_match.start(), raw_match.end(), str(value),
                                           "scalar", None, 1, "number_word"))
                occupied.append((raw_match.start(), raw_match.end()))
    return sorted(tokens, key=lambda item: (item.start, item.end, item.dimension))


CONTEXT_TERMS = {
    "time": {"duration", "time", "waiting", "processing", "مده", "زمن", "وقت", "انتظار", "معالجه", "فحص"},
    "count": {"count", "number", "total", "average", "عدد", "اجمالي", "متوسط", "حالات", "حاله", "حوادث"},
    "currency": {"cost", "price", "amount", "تكلفه", "سعر", "قيمه", "مبلغ", "ريال"},
    "distance": {"distance", "length", "مسافه", "طول", "كيلومتر", "كم"},
    "mass": {"weight", "mass", "وزن", "كتله"},
    "year": {"age", "year", "years", "عمر", "سنه", "سنوات", "عام"},
    "people": {"passenger", "passengers", "rider", "riders", "ركاب", "راكب"},
    "trip": {"trip", "trips", "ride", "rides", "رحله", "رحلات"},
}


def _dimension_family(dimension: str) -> str | None:
    value = normalize_text(dimension).removeprefix("ال")
    for family, aliases in DIMENSION_ALIASES.items():
        if value in aliases:
            return family
    return None


def _context_supports(family: str, text: str) -> bool:
    normalized = normalize_text(text)
    terms = content_terms(text)
    vocabulary = CONTEXT_TERMS[family] | DIMENSION_ALIASES[family]
    return any(keyword in terms or keyword in normalized for keyword in vocabulary)


def _compatible_dimensions(source: NumericToken, target: NumericToken, left: str, right: str) -> tuple[bool, str, str]:
    if source.dimension == target.dimension:
        return True, "exact", source.dimension
    protected = {"percent", "per_mille"}
    if source.dimension in protected or target.dimension in protected:
        return False, "incompatible", ""
    source_family = _dimension_family(source.dimension)
    target_family = _dimension_family(target.dimension)
    if source_family and source_family == target_family:
        return True, "unit_family", source_family
    if source.dimension == "scalar" and target_family and _context_supports(target_family, left):
        return True, "context_labels_unlabeled_source_value", target_family
    if target.dimension == "scalar" and source_family and _context_supports(source_family, right):
        return True, "context_labels_unlabeled_target_value", source_family
    return False, "incompatible", ""


QUALIFIER_BEFORE = {
    "greater_than": r"more than|over|above|exceed(?:s|ed|ing)?|greater than|in excess of|أكثر من|اكثر من|"
                    r"تجاوزت?|يتجاوز|تتجاوز|يزيد (?:على|عن)|تزيد (?:على|عن)|فوق",
    "at_least": r"at least|no less than|not less than|لا (?:يقل|تقل) عن",
    "less_than": r"less than|fewer than|under|below|أقل من|اقل من|دون",
    "at_most": r"at most|no more than|not more than|up to|لا (?:يزيد|تزيد) (?:على|عن)",
    "approximately": r"about|approximately|approx\.?|around|nearly|almost|roughly|~|نحو|حوالي|حوالى|قرابة|"
                     r"تقريبا|تقريبًا|ما يقارب|زهاء",
}
QUALIFIER_AFTER = {
    "at_least": r"or more|or above|and above|and over|\+|فأكثر|أو أكثر|او اكثر",
    "approximately": r"approximately|roughly|تقريبا|تقريبًا",
}
_BEFORE_RES = {key: re.compile(rf"(?i)(?:^|[\s(])(?P<q>{value})\s*$") for key, value in QUALIFIER_BEFORE.items()}
_AFTER_RES = {key: re.compile(rf"(?i)^\s*(?P<q>{value})(?=\s|$|[.,;:)])") for key, value in QUALIFIER_AFTER.items()}
BOUND_RELATIONS = {"greater_than", "at_least", "less_than", "at_most"}


def numeric_qualifier(text: str, token: NumericToken) -> dict[str, object] | None:
    """A comparison qualifier attached to a number ("more than 3%", "4,000 or more", "نحو 20", "+20").

    Recorded as evidence only: a qualified number is never treated as an exact value."""
    before = text[max(0, token.start - 32):token.start]
    for relation, pattern in _BEFORE_RES.items():
        match = pattern.search(before)
        if match:
            start = token.start - (len(before) - match.start("q"))
            return {"relation": relation, "raw": match.group("q"), "offsets": [start, start + len(match.group("q"))],
                    "position": "before"}
    after = text[token.end:token.end + 24]
    for relation, pattern in _AFTER_RES.items():
        match = pattern.search(after)
        if match:
            start = token.end + match.start("q")
            return {"relation": relation, "raw": match.group("q"), "offsets": [start, start + len(match.group("q"))],
                    "position": "after"}
    if token.raw.startswith("+"):
        return {"relation": "sign_or_open_bound_ambiguous", "raw": "+", "offsets": [token.start, token.start + 1],
                "position": "before"}
    return None


def _display_precision(token: NumericToken) -> Decimal:
    """Half-width of the interval a displayed value stands for (e.g. 4K -> 500, 3.01 -> 0.005)."""
    digits = re.sub(r"[^0-9.]", "", token.raw.translate(ARABIC_DIGITS).replace("٫", ".").split()[0]
                    if token.raw.split() else token.raw)
    places = len(digits.split(".", 1)[1]) if "." in digits else 0
    return Decimal(token.scale) * Decimal(1).scaleb(-places) / 2


def _bound_consistent(relation: str, bound: Decimal, value: Decimal, half_width: Decimal) -> bool:
    """Some value inside the displayed interval satisfies the bound (not contradicted)."""
    low, high = value - half_width, value + half_width
    return {"greater_than": high > bound, "at_least": high >= bound,
            "less_than": low < bound, "at_most": low <= bound}[relation]


def _bound_proved(relation: str, bound: Decimal, value: Decimal, half_width: Decimal) -> bool:
    """Every value inside the displayed interval satisfies the bound: the displayed precision proves it."""
    low, high = value - half_width, value + half_width
    return {"greater_than": low > bound, "at_least": low >= bound,
            "less_than": high < bound, "at_most": high <= bound}[relation]


# How a matched pair of numbers agrees, and whether that agreement can count as evidence.
NUMERIC_AGREEMENT = {
    "exact_equality": ("exact_agreement", "supports"),
    "display_precision_equality": ("rounded_agreement", "supports"),
    "rounded_equality": ("rounded_agreement", "supports"),
    "approximate_consistent": ("approximate_agreement", "supports"),
    "qualifier_ambiguous": ("qualifier_ambiguous", "inconclusive"),
}


def _match_semantics(source: NumericToken, target: NumericToken, left: str, right: str,
                     comparison_method: str) -> tuple[bool, dict[str, object]]:
    """(consistent, fields). Inequalities, approximations and display rounding never become exact equality."""
    source_q, target_q = numeric_qualifier(left, source), numeric_qualifier(right, target)
    fields: dict[str, object] = {"source_qualifier": source_q, "target_qualifier": target_q}
    bound_checks = []
    for qualifier, bound_token, other in ((source_q, source, target), (target_q, target, source)):
        if qualifier and qualifier["relation"] in BOUND_RELATIONS:
            args = (str(qualifier["relation"]), Decimal(bound_token.value), Decimal(other.value),
                    _display_precision(other))
            consistent = _bound_consistent(*args)
            other_q = target_q if bound_token is source else source_q
            other_is_bound = bool(other_q) and other_q["relation"] != "approximately"
            proved = consistent and not other_is_bound and _bound_proved(*args)
            if not consistent:
                verdict = "contradicted"
                reason = "no value within the displayed precision of the compared number satisfies the bound"
            elif other_is_bound:
                verdict = "consistent_not_proved"
                reason = ("the compared number is itself qualified (" + str(other_q["raw"]) + "): it states a bound "
                          "or an ambiguous sign, not a value, so it cannot prove the bound")
            elif proved:
                verdict = "proved"
                reason = "every value within the displayed precision of the compared number satisfies the bound"
            else:
                verdict = "consistent_not_proved"
                reason = ("insufficient precision: the compared number's displayed interval "
                          f"[{args[2] - args[3]}, {args[2] + args[3]}] contains values on both sides of the bound")
            bound_checks.append({"bound_side": "source" if bound_token is source else "target",
                                 "relation": qualifier["relation"], "bound": bound_token.value,
                                 "compared_value": other.value,
                                 "compared_display_half_width": format(_display_precision(other).normalize(), "f"),
                                 "consistent": consistent, "proved": proved, "verdict": verdict, "reason": reason})
    if bound_checks:
        fields["bound_checks"] = bound_checks
        if not all(item["consistent"] for item in bound_checks):
            return False, fields
        semantics = "bound_consistent"
        if all(item["proved"] for item in bound_checks):
            agreement, status = "inequality_proved", "supports"
        elif any("qualified" in item["reason"] for item in bound_checks if not item["proved"]):
            agreement, status = "inequality_unverifiable_qualified_value", "inconclusive"
        else:
            agreement, status = "inequality_consistent_insufficient_precision", "inconclusive"
    elif any(q and q["relation"] == "approximately" for q in (source_q, target_q)):
        semantics = "approximate_consistent"
    elif any(q for q in (source_q, target_q)):
        semantics = "qualifier_ambiguous"
    elif comparison_method != "exact":
        semantics = "rounded_equality"
    elif source.scale != target.scale or (source.normalization != target.normalization
                                          and "scaled_expression" in (source.normalization, target.normalization)):
        semantics = "display_precision_equality"
    else:
        semantics = "exact_equality"
    if semantics != "bound_consistent":
        agreement, status = NUMERIC_AGREEMENT[semantics]
    fields["match_semantics"] = semantics
    fields["equality_asserted"] = semantics == "exact_equality"
    fields["numeric_agreement"] = agreement
    # An inconclusive match stays recorded as evidence but is never counted as proof of a relationship.
    fields["evidential_status"] = status
    return True, fields


def match_numeric_tokens(left: str, right: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    used: set[int] = set()
    for source in numeric_tokens(left):
        for index, target in enumerate(numeric_tokens(right)):
            if source.dimension in NON_VALUE_DIMENSIONS or target.dimension in NON_VALUE_DIMENSIONS:
                continue
            compatible, compatibility, dimension = _compatible_dimensions(source, target, left, right)
            if index in used or not compatible:
                continue
            comparison_method = "exact"
            tolerance = "0"
            rounding = None
            if source.value != target.value:
                if source.dimension != "percent" or target.dimension != "percent":
                    continue
                source_value, target_value = Decimal(source.value), Decimal(target.value)
                source_places = max(0, -source_value.as_tuple().exponent)
                target_places = max(0, -target_value.as_tuple().exponent)
                places = min(source_places, target_places)
                quantum = Decimal(1).scaleb(-places)
                if source_value.quantize(quantum) != target_value.quantize(quantum):
                    continue
                comparison_method = "rounded_percent"
                tolerance = format(quantum / 2, "f")
                rounding = {"decimal_places": places, "method": "ROUND_HALF_EVEN"}
            consistent, semantics = _match_semantics(source, target, left, right, comparison_method)
            if not consistent:
                continue  # a stated bound contradicts the other value: not supporting evidence
            used.add(index)
            matches.append({
                "source_raw": source.raw,
                "source_offsets": [source.start, source.end],
                "target_raw": target.raw,
                "target_offsets": [target.start, target.end],
                "normalized_value": source.value,
                "dimension": dimension,
                "unit_compatibility": compatibility,
                "source_scale": source.scale,
                "target_scale": target.scale,
                "source_normalization": source.normalization,
                "target_normalization": target.normalization,
                "comparison_method": comparison_method,
                "rounding": rounding,
                "tolerance": tolerance,
                **semantics,
            })
            break
    return matches


def list_markers(text: str) -> list[dict[str, object]]:
    found = [
        {"start": match.start("marker"), "end": match.end("marker"), "marker": match.group("marker"),
         "indent": len(match.group("indent")), "position": "leading"}
        for match in LIST_MARKER_RE.finditer(text)
    ]
    found.extend(
        {"start": match.start("marker"), "end": match.end("marker"), "marker": match.group("marker"),
         "indent": 0, "position": "trailing"}
        for match in RTL_TRAILING_MARKER_RE.finditer(text)
    )
    return sorted(found, key=lambda item: int(item["start"]))


def marker_family(marker: str) -> str:
    if re.fullmatch(r"[0-9٠-٩۰-۹]+[.)]", marker):
        return "number"
    if re.fullmatch(r"[A-Za-z][.)]", marker):
        return "letter"
    return "bullet"

