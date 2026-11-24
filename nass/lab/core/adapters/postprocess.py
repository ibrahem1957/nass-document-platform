from __future__ import annotations

import re
import secrets
import unicodedata
from dataclasses import dataclass

from bidi.algorithm import get_display

from lab.core.models import PageResult, RunResult


# v2: bidi_mirror_fix requires page-level visual-mirroring evidence;
# digit_run_fix always carries an explicit producer-dependence warning.
VARIANT_VERSION = "2"
PRESENTATION_RANGES = ((0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9٠-٩۰-۹_])(?:[$€£¥₹﷼]|ر\.س\.?|د\.إ\.?)?\s*"
    r"[+\-−]?\s*[0-9٠-٩۰-۹]+(?:[.,٫٬:/\-][0-9٠-٩۰-۹]+)*"
    r"(?:\s*(?:%|٪|‰|bps|USD|SAR|AED|EUR|GBP|ر\.س\.?|د\.إ\.?))?"
    r"(?![A-Za-z0-9٠-٩۰-۹_])",
    re.IGNORECASE,
)
REVERSIBLE_NUMERIC_RE = re.compile(
    r"(?<![0-9٠-٩۰-۹])"
    r"(?P<prefix>(?:(?:[$€£¥₹﷼]|ر\.س\.?|د\.إ\.?)\s*)?"
    r"(?:[+\-−]\s*)?(?:[%٪‰]\s*)?)"
    r"(?P<core>[٠-٩۰-۹]+(?:[.,٫٬:/\-][٠-٩۰-۹]+)+|[٠-٩۰-۹]{2,})"
    r"(?P<suffix>(?:\s*[%٪‰])?(?:\s*[+\-−])?"
    r"(?:\s*(?:[$€£¥₹﷼]|ر\.س\.?|د\.إ\.?))?)"
    r"(?![0-9٠-٩۰-۹])"
)
IDENTIFIER_TOKEN_RE = re.compile(
    r"(?<![\w])[A-Za-z\u0600-\u06ff0-9٠-٩۰-۹]+"
    r"(?:[-_/][A-Za-z\u0600-\u06ff0-9٠-٩۰-۹]+)+(?![\w])"
)
MIRROR_TABLE = str.maketrans({"<": ">", ">": "<", "≤": "≥", "≥": "≤"})
MIRRORED_OPERATOR_SIGNATURE_RE = re.compile(
    r"(?P<operator>[<>≤≥])(?P<space>\s*)(?P<marker>[٪%‰])"
    r"(?P<number>[٠-٩۰-۹])"
)
MIRRORED_NUMERIC_BRACKETS_RE = re.compile(
    r"\)(?P<body>\s*[٪%‰]?[٠-٩۰-۹]+(?:[.,٫٬:/\-][٠-٩۰-۹]+)*\s*)\("
)
MIRRORABLE_RE = re.compile(r"[<>≤≥()\[\]{}]")
REVERSED_BRACKET_PAIR_RE = re.compile(r"\)[^()\n]{1,60}\(")
NORMAL_BRACKET_PAIR_RE = re.compile(r"\([^()\n]{1,60}\)")
DIGIT_RUN_FIX_WARNING = (
    "Experimental digit_run_fix assumes this PDF's text layer stores Arabic-Indic "
    "numbers in visual (reversed) order, as observed for one Chrome/Skia-produced "
    "benchmark. On PDFs from other producers whose numbers are already in logical "
    "order it corrupts correct values (e.g. ٣٣٫٨٤٪ -> ٤٨٫٣٣٪, ١٢٣٬٤٥٦ -> ٦٥٤٬٣٢١). "
    "Verify every changed value against the page image; RAW remains the reference."
)


def page_shows_visual_mirroring(text: str) -> bool:
    """Page-level evidence that the extractor emitted visually mirrored glyphs.

    A reversed pair `)…(` never occurs in correct logical text, while logical text
    normally contains `(…)`. Mirroring is a property of the PDF producer, so it is
    judged per page and only from extractor output (never from ground truth).
    """
    reversed_pairs = len(REVERSED_BRACKET_PAIR_RE.findall(text))
    normal_pairs = len(NORMAL_BRACKET_PAIR_RE.findall(text))
    return reversed_pairs >= 1 and reversed_pairs > normal_pairs


@dataclass(frozen=True)
class CorrectionChange:
    kind: str
    before: str
    after: str
    start: int
    end: int
    confidence: str
    reason: str


@dataclass(frozen=True)
class CorrectionOutput:
    text: str
    changes: tuple[CorrectionChange, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DetectedToken:
    kind: str
    value: str
    start: int
    end: int


def detect_sensitive_tokens(text: str) -> tuple[DetectedToken, ...]:
    tokens = [
        DetectedToken("numeric", match.group(0).strip(), match.start(), match.end())
        for match in NUMERIC_TOKEN_RE.finditer(text)
    ]
    tokens += [
        DetectedToken("identifier", match.group(0), match.start(), match.end())
        for match in IDENTIFIER_TOKEN_RE.finditer(text)
    ]
    return tuple(sorted(tokens, key=lambda token: (token.start, token.end, token.kind)))


def available_variants(
    adapter_id: str, *, include_experimental: bool = False
) -> list[str]:
    common = ["raw", "arabic_fold"]
    if adapter_id in {"text.pypdfium2", "text.pdfplumber", "text.pypdf"}:
        common += ["bidi_mirror_fix", "arabic_fold+bidi_mirror_fix"]
    if include_experimental:
        if adapter_id in {"text.pymupdf", "text.pypdf"}:
            common += ["digit_run_fix", "arabic_fold+digit_run_fix"]
        common += ["bidi_reorder"]
    common += ["unmapped_marker"]
    return list(dict.fromkeys(common))


def _arabic_dominant(value: str) -> bool:
    arabic = len(ARABIC_RE.findall(value))
    latin = sum(char.isascii() and char.isalpha() for char in value)
    return arabic > latin and arabic > 0


def _fold_presentation_forms(text: str) -> tuple[str, list[CorrectionChange]]:
    output: list[str] = []
    changes: list[CorrectionChange] = []
    for index, char in enumerate(text):
        codepoint = ord(char)
        if codepoint != 0xFDFC and any(
            start <= codepoint <= end for start, end in PRESENTATION_RANGES
        ):
            folded = unicodedata.normalize("NFKC", char)
            output.append(folded)
            if folded != char:
                changes.append(
                    CorrectionChange(
                        "arabic_fold",
                        char,
                        folded,
                        index,
                        index + 1,
                        "high",
                        "Arabic presentation-form compatibility normalization",
                    )
                )
        else:
            output.append(char)
    return "".join(output), changes


def _valid_date_parts(parts: list[str]) -> bool:
    try:
        values = [int(part) for part in parts]
    except ValueError:
        return False
    if len(values) != 3:
        return False
    day, month, year = values
    return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100


def _repair_date_core(core: str) -> str | None:
    separator = next((value for value in "/-." if value in core), None)
    if separator is None:
        return None
    parts = core.split(separator)
    if len(parts) != 3:
        return None
    reversed_groups = [part[::-1] for part in parts]
    if separator == "/":
        candidates = [reversed_groups, list(reversed(reversed_groups))]
        candidates.sort(key=lambda value: len(value[0]) == 4)
    else:
        candidates = [reversed_groups, list(reversed(reversed_groups))]
        candidates.sort(key=lambda value: len(value[0]) != 4)
    for candidate in candidates:
        logical = candidate
        if len(candidate[0]) == 4:
            valid = _valid_date_parts([candidate[2], candidate[1], candidate[0]])
        else:
            valid = _valid_date_parts(candidate)
        if valid:
            return separator.join(logical)
    return None


def _valid_grouped_number(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[٠-٩۰-۹]{1,3}(?:٬[٠-٩۰-۹]{3})+(?:٫[٠-٩۰-۹]{1,3})?",
            value,
        )
        or re.fullmatch(
            r"[٠-٩۰-۹]{1,3}(?:,[٠-٩۰-۹]{3})+(?:\.[٠-٩۰-۹]{1,3})?",
            value,
        )
    )


def _logical_numeric_token(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    suffix = match.group("suffix")
    raw_core = match.group("core")
    currency_pattern = r"[$€£¥₹﷼]|ر\.س\.?|د\.إ\.?"
    currencies_before = re.findall(currency_pattern, prefix)
    currencies_after = re.findall(currency_pattern, suffix)
    signs = re.findall(r"[+\-−]", prefix + suffix)
    markers = re.findall(r"[%٪‰]", prefix + suffix)
    date_core = _repair_date_core(raw_core)
    reversed_core = raw_core[::-1]
    core: str | None = None
    if date_core is not None:
        core = date_core
    elif markers:
        core = reversed_core
    elif currencies_before or currencies_after or "٬" in raw_core or "," in raw_core:
        if _valid_grouped_number(reversed_core):
            core = reversed_core
    elif len(raw_core) == 4 and raw_core.isdigit():
        raw_year = int(raw_core)
        reversed_year = int(reversed_core)
        if not 1900 <= raw_year <= 2100 and 1900 <= reversed_year <= 2100:
            core = reversed_core
    if core is None:
        return match.group(0)

    local_currencies = {"﷼", "ر.س", "ر.س.", "د.إ", "د.إ."}
    logical_suffix_currencies = [
        value
        for value in currencies_before + currencies_after
        if value in local_currencies
    ]
    logical_prefix_currencies = [
        value
        for value in currencies_before + currencies_after
        if value not in local_currencies
    ]
    currency_before = " ".join(logical_prefix_currencies)
    before = f"{currency_before} " if currency_before else ""
    before += "".join(signs)
    after = "".join(markers)
    if logical_suffix_currencies:
        after += f" {' '.join(logical_suffix_currencies)}"
    return before + core + after


def _fix_digit_runs(text: str, adapter_id: str) -> tuple[str, list[CorrectionChange], list[str]]:
    if adapter_id not in {"text.pymupdf", "text.pypdf"}:
        return text, [], ["Digit reversal not applied: this extractor profile preserves digit order."]
    output: list[str] = []
    changes: list[CorrectionChange] = []
    skipped_ambiguous = 0
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        suffix = line[len(body) :]
        cursor = 0
        if _arabic_dominant(body):
            for match in REVERSIBLE_NUMERIC_RE.finditer(body):
                token = match.group(0)
                # Captured E1-A RAW shows the complete numeric core, including
                # decimal/group/date separators, in visual reverse order.
                fixed = _logical_numeric_token(match)
                if fixed == token:
                    skipped_ambiguous += 1
                    continue
                output.append(body[cursor : match.start()])
                output.append(fixed)
                changes.append(
                    CorrectionChange(
                        "digit_run_fix",
                        token,
                        fixed,
                        offset + match.start(),
                        offset + match.end(),
                        "medium",
                        f"Measured reversed-digit profile for {adapter_id}",
                    )
                )
                cursor = match.end()
        output.append(body[cursor:])
        output.append(suffix)
        offset += len(line)
    warnings = [DIGIT_RUN_FIX_WARNING]
    if skipped_ambiguous:
        warnings.append(
            f"Left {skipped_ambiguous} ambiguous Arabic-script numeric tokens "
            "unchanged; structural evidence was insufficient."
        )
    return "".join(output), changes, warnings


def _fix_mirrors(
    text: str, adapter_id: str
) -> tuple[str, list[CorrectionChange], list[str]]:
    if adapter_id not in {"text.pypdfium2", "text.pdfplumber", "text.pypdf"}:
        return text, [], ["Mirrored operators not changed for this extractor profile."]
    output: list[str] = []
    changes: list[CorrectionChange] = []
    warnings: list[str] = []
    # Guard (E1-B review N2): the local signature `≥ ‰٠٫٥` is indistinguishable
    # from logical text that prefixes the percent sign (`≥ ٪٥٠`). Only act when
    # the page itself shows visual mirroring.
    page_mirrored = page_shows_visual_mirroring(text)
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        suffix = line[len(body) :]
        fixed = body
        if _arabic_dominant(body) and not page_mirrored:
            if MIRRORED_OPERATOR_SIGNATURE_RE.search(body) or MIRRORED_NUMERIC_BRACKETS_RE.search(body):
                warnings.append(
                    "Possible mirrored operator/bracket left unchanged: the page shows "
                    "no visual-mirroring evidence (it may be logical text such as ≥ ٪٥٠)."
                )
            elif MIRRORABLE_RE.search(body):
                warnings.append(
                    "Ambiguous bidirectional operator/bracket left unchanged; "
                    "local evidence was insufficient."
                )
        elif _arabic_dominant(body):
            fixed = MIRRORED_OPERATOR_SIGNATURE_RE.sub(
                lambda match: (
                    match.group("operator").translate(MIRROR_TABLE)
                    + match.group("space")
                    + match.group("marker")
                    + match.group("number")
                ),
                fixed,
            )
            fixed = MIRRORED_NUMERIC_BRACKETS_RE.sub(
                lambda match: f"({match.group('body')})", fixed
            )
            if fixed != body:
                changes.append(
                    CorrectionChange(
                        "bidi_mirror_fix",
                        body,
                        fixed,
                        offset,
                        offset + len(body),
                        "medium",
                        f"Local mirrored-token signature with page-level mirroring evidence for {adapter_id}",
                    )
                )
            elif MIRRORABLE_RE.search(body):
                warnings.append(
                    "Ambiguous bidirectional operator/bracket left unchanged; "
                    "local evidence was insufficient."
                )
        output.append(fixed)
        output.append(suffix)
        offset += len(line)
    return "".join(output), changes, list(dict.fromkeys(warnings))


def _bidi_reorder(text: str) -> tuple[str, list[CorrectionChange], list[str]]:
    output: list[str] = []
    changes: list[CorrectionChange] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        suffix = line[len(body) :]
        fixed = get_display(body, base_dir="R") if _arabic_dominant(body) else body
        if fixed != body:
            changes.append(
                CorrectionChange(
                    "bidi_reorder",
                    body,
                    fixed,
                    offset,
                    offset + len(body),
                    "low",
                    "Experimental display-order conversion; requires review",
                )
            )
        output.extend((fixed, suffix))
        offset += len(line)
    return "".join(output), changes, ["bidi_reorder is experimental and must be reviewed visually."]


def correct_text(text: str, adapter_id: str, variant: str) -> CorrectionOutput:
    current = text
    changes: list[CorrectionChange] = []
    warnings: list[str] = []
    if variant == "raw":
        return CorrectionOutput(text, (), ())
    for step in variant.split("+"):
        if step == "arabic_fold":
            current, new_changes = _fold_presentation_forms(current)
            changes.extend(new_changes)
        elif step == "digit_run_fix":
            current, new_changes, new_warnings = _fix_digit_runs(current, adapter_id)
            changes.extend(new_changes)
            warnings.extend(new_warnings)
        elif step == "bidi_mirror_fix":
            current, new_changes, new_warnings = _fix_mirrors(current, adapter_id)
            changes.extend(new_changes)
            warnings.extend(new_warnings)
        elif step == "bidi_reorder":
            current, new_changes, new_warnings = _bidi_reorder(current)
            changes.extend(new_changes)
            warnings.extend(new_warnings)
        elif step == "unmapped_marker":
            positions = [index for index, char in enumerate(current) if char in {"\x00", "�"}]
            for index in positions:
                changes.append(
                    CorrectionChange(
                        "unmapped_marker",
                        current[index],
                        "⟦?⟧",
                        index,
                        index + 1,
                        "high",
                        "Unrecoverable/unmapped glyph made explicit",
                    )
                )
            current = current.replace("\x00", "⟦?⟧").replace("�", "⟦?⟧")
            if positions:
                warnings.append("Unmapped glyphs cannot be reconstructed; route the page to OCR/VLM evaluation.")
        else:
            raise ValueError(f"Unknown post-processing variant: {step}")
    return CorrectionOutput(current, tuple(changes), tuple(dict.fromkeys(warnings)))


def derive_variant(raw_run: RunResult, variant: str) -> RunResult:
    if raw_run.variant != "raw" or raw_run.raw_run_id is not None:
        raise ValueError("Variants must be derived directly from an immutable RAW run")
    if variant == "raw":
        return raw_run.model_copy(deep=True)
    pages: list[PageResult] = []
    change_counts: dict[int, int] = {}
    warnings: dict[int, list[str]] = {}
    for page in raw_run.pages:
        corrected = correct_text(page.text, raw_run.adapter.adapter_id, variant)
        pages.append(page.model_copy(update={"text": corrected.text, "blocks": None}))
        change_counts[page.page] = len(corrected.changes)
        warnings[page.page] = list(corrected.warnings)
    environment = dict(raw_run.environment)
    environment["postprocess"] = {
        "version": VARIANT_VERSION,
        "change_counts": change_counts,
        "warnings": warnings,
    }
    return raw_run.model_copy(
        deep=True,
        update={
            "run_id": f"run_{secrets.token_hex(9)}",
            "variant": variant,
            "pages": pages,
            "raw_run_id": raw_run.run_id,
            "variant_version": VARIANT_VERSION,
            "cache_hit": False,
            "environment": environment,
        },
    )
