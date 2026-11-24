"""Per-page diagnostic INDICATORS for extracted text.

These are heuristics, not accuracy measurements. More characters does not mean a
more accurate extraction. CER/WER are computed only against attached ground truth.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

ARABIC_LETTER = re.compile(r"[ؠ-يٮ-ۓۺ-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
PRESENTATION_FORM = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
LAMALEF_PREFIXES = ("اال", "األ", "اإل", "اآل")
# Tokens seen in Word exports where the lam-alef ligature came out as alef + lam.
KNOWN_LAMALEF_ARTIFACTS = frozenset({"السالمة", "سالمة", "إال", "اال", "ال"})
DIGITS = "0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹"
_DIGIT_VALUE = {ch: str(index % 10) for index, ch in enumerate(DIGITS)}
YEAR = re.compile(r"^(19|20)\d\d$")
NEEDS_OCR_GARBAGE_RATIO = 0.05
DIAGNOSTICS_VERSION = "2"  # 2: single-letter Arabic token ratio


def has_rtl(text: str) -> bool:
    return any(unicodedata.bidirectional(ch) in ("AL", "R") for ch in text)


def is_garbage_char(ch: str) -> bool:
    """Private Use, replacement char, control chars (except line breaks/tabs), Latin Extended-B."""
    code = ord(ch)
    if ch == "�" or 0xE000 <= code <= 0xF8FF or 0xF0000 <= code:
        return True
    if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t\f\v":
        return True
    return 0x0180 <= code <= 0x024F


def _tokens(text: str) -> list[str]:
    return text.split()


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_fragment_line(line: str) -> bool:
    """A line made only of digits, % or punctuation (e.g. a number split from its line)."""
    stripped = line.replace(" ", "")
    return bool(stripped) and all(
        ch in DIGITS or ch == "%" or unicodedata.category(ch)[0] in ("P", "S") for ch in stripped
    )


def _is_punct_only_line(line: str) -> bool:
    stripped = line.replace(" ", "")
    return bool(stripped) and all(ch == "%" or unicodedata.category(ch)[0] in ("P", "S") for ch in stripped)


def lamalef_artifacts(text: str) -> int:
    count = 0
    for token in _tokens(unicodedata.normalize("NFKC", text)):
        core = token.strip("().,،:؛!?؟\"'«»[]")
        if core.startswith(LAMALEF_PREFIXES) or core in KNOWN_LAMALEF_ARTIFACTS:
            count += 1
    return count


def reversed_article(text: str) -> dict[str, int]:
    tokens = [token.strip("().,،:؛!?؟\"'«»[]") for token in _tokens(unicodedata.normalize("NFKC", text))]
    ends = sum(1 for token in tokens if len(token) >= 4 and token.endswith("لا"))
    starts = sum(1 for token in tokens if len(token) >= 3 and token.startswith("ال"))
    return {"ending_la": ends, "starting_al": starts}


def likely_reversed(ending_la: int, starting_al: int) -> bool:
    """Indicator only: many tokens end in "لا" (reversed "ال") and clearly fewer start with "ال"."""
    return ending_la >= 5 and ending_la > 2 * starting_al


def detached_diacritics(text: str) -> int:
    count = 0
    previous = " "
    for ch in text:
        if unicodedata.category(ch) == "Mn" and (previous.isspace() or previous in "|"):
            count += 1
        previous = ch
    return count


def reversed_years(text: str) -> int:
    count = 0
    for token in re.findall(f"[{DIGITS}]+", text):
        value = "".join(_DIGIT_VALUE[ch] for ch in token)
        if len(value) == 4 and not YEAR.match(value) and YEAR.match(value[::-1]):
            count += 1
    return count


# A numbered/lettered heading such as "1) ..." or "أ) ...": its ")" has no "(" by design.
_ENUMERATOR = re.compile(r"^\s*(?:[0-9٠-٩۰-۹]{1,3}|[A-Za-zء-ي])\s*\)")
_MIRRORED_ENUMERATOR = re.compile(r"^\s*[0-9٠-٩۰-۹]{1,3}\s*\(")


def flipped_bracket_lines(text: str) -> int:
    """Lines where a ")" appears BEFORE its matching "(" (e.g. ")+20 سنة("), RTL lines only.

    An unmatched ")" counts only if an unmatched "(" follows it on the same line; a lone ")"
    after a leading number/letter (numbered heading) is never counted.
    """
    count = 0
    for line in _lines(text):
        if not has_rtl(line):
            continue
        mirrored = _MIRRORED_ENUMERATOR.match(line)
        if mirrored and ")" not in line[mirrored.end():]:
            count += 1  # "1( heading": a numbered heading whose ")" was mirrored
            continue
        enumerator = _ENUMERATOR.match(line)
        if enumerator:
            line = line[enumerator.end():]
        opened: list[int] = []
        unmatched_close: list[int] = []
        for index, ch in enumerate(line):
            if ch == "(":
                opened.append(index)
            elif ch == ")":
                if opened:
                    opened.pop()  # a normal "(...)" pair
                else:
                    unmatched_close.append(index)
        # `opened` now holds only unmatched "(" positions.
        if unmatched_close and opened and opened[-1] > unmatched_close[0]:
            count += 1
    return count


_ARABIC_BASE = "ؠ-يٮ-ۓۺ-ۿ"
_GLUED = re.compile(
    f"[{_ARABIC_BASE}][A-Za-z0-9]|[A-Za-z0-9][{_ARABIC_BASE}]|%[0-9{DIGITS}]"
)


def glued_mixed_script(text: str) -> int:
    """Arabic letters touching Latin letters/digits, or % directly before a digit."""
    return len(_GLUED.findall(unicodedata.normalize("NFKC", text)))


_PERCENT_BEFORE_DIGIT = re.compile(f"%[0-9{DIGITS}]")


def percent_before_digit(text: str) -> int:
    """`%45.5`: in logical-order text the percent sign normally follows the number."""
    return len(_PERCENT_BEFORE_DIGIT.findall(text))


def repair_signals(text: str) -> dict[str, int]:
    """Signals that a page's raw text may need logical-order repair (heuristics)."""
    lines = _lines(text)
    article = reversed_article(text)
    return {
        "glued_mixed_script": glued_mixed_script(text),
        "percent_before_digit": percent_before_digit(text),
        "lamalef_artifacts": lamalef_artifacts(text),
        "punct_only_lines": sum(1 for line in lines if _is_punct_only_line(line)),
        "digit_only_lines": sum(1 for line in lines if _is_fragment_line(line) and not _is_punct_only_line(line)),
        "reversed_years": reversed_years(text),
        "detached_diacritics": detached_diacritics(text),
        "presentation_forms": len(PRESENTATION_FORM.findall(text)),
        "flipped_brackets": flipped_bracket_lines(text),
        "reversed_article": article["ending_la"] if article["ending_la"] >= 2 and article["ending_la"] > article["starting_al"] else 0,
    }


# Signals strong enough on their own to attempt a repair. `digit_only_lines` is
# recorded but is not used for gating: correct table extractions often contain
# number-only lines.
STRONG_SIGNALS = (
    "glued_mixed_script",
    "lamalef_artifacts",
    "punct_only_lines",
    "reversed_years",
    "detached_diacritics",
    "presentation_forms",
    "flipped_brackets",
    "reversed_article",
)
# A reconstruction must never increase these compared with the raw text.
REGRESSION_SIGNALS = ("percent_before_digit", "lamalef_artifacts", "punct_only_lines", "detached_diacritics", "flipped_brackets", "reversed_article", "reversed_years")


def garbage_in_arabic_tokens(text: str) -> tuple[int, int]:
    arabic_tokens = [token for token in _tokens(text) if ARABIC_LETTER.search(token)]
    bad = sum(1 for token in arabic_tokens if any(is_garbage_char(ch) for ch in token))
    return bad, len(arabic_tokens)


_PUNCT_STRIP = "().,،:؛!?؟\"'«»[]{}-–—…|/\\*•"


def _is_single_arabic_letter(token: str) -> bool:
    core = "".join(ch for ch in unicodedata.normalize("NFKC", token).strip(_PUNCT_STRIP)
                   if unicodedata.category(ch) != "Mn" and ch != "ـ")  # ignore harakat and tatweel
    return len(core) == 1 and bool(ARABIC_LETTER.match(core))


def _is_arabic_letter(ch: str) -> bool:
    code = ord(ch)
    return ch.isalpha() and (0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF
                             or 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFF)


def arabic_letter_share(text: str) -> float:
    """Share of Arabic letters among all letters (0.0 when the text has no letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    return sum(1 for ch in letters if _is_arabic_letter(ch)) / len(letters) if letters else 0.0


def single_letter_stats(text: str) -> dict:
    """Share of Arabic tokens (split on whitespace) that are a single Arabic letter.

    High on PDFs whose text layer has spaces inside words (e.g. Chrome/Skia output
    "ا خ ت بار"), ~0 on normal Arabic text. Indicator only, not an accuracy measure."""
    arabic = [token for token in _tokens(text) if ARABIC_LETTER.search(token)]
    single = sum(1 for token in arabic if _is_single_arabic_letter(token))
    return {"single_letter_arabic_tokens": single, "arabic_tokens": len(arabic),
            "single_letter_arabic_ratio": round(single / len(arabic), 4) if arabic else 0.0}


DEFAULT_SINGLE_LETTER_THRESHOLD = 0.10
DEFAULT_SINGLE_LETTER_MIN_TOKENS = 10
SPLIT_WORDS_REASON = "words split by spaces in the PDF text layer"


def needs_ocr_settings(settings: dict | None) -> dict | None:
    """The run's 'Needs OCR?' single-letter rule, or None for runs created before it existed."""
    rule = (settings or {}).get("diagnostics")
    if not rule or rule.get("single_letter_threshold") is None:
        return None
    return {"single_letter_threshold": float(rule["single_letter_threshold"]),
            "single_letter_min_arabic_tokens": int(rule.get("single_letter_min_arabic_tokens",
                                                            DEFAULT_SINGLE_LETTER_MIN_TOKENS))}


def page_indicators(text: str, needs_ocr_rule: dict | None = None) -> dict:
    """`needs_ocr_rule` (processed method only): flag 'Needs OCR?' when the single-letter
    Arabic token ratio exceeds the threshold on a page with enough Arabic tokens."""
    chars = len(text)
    tokens = _tokens(text)
    lines = _lines(text)
    garbage_chars = sum(1 for ch in text if is_garbage_char(ch))
    garbage_tokens, arabic_tokens = garbage_in_arabic_tokens(text)
    presentation = len(PRESENTATION_FORM.findall(text))
    fragment_lines = sum(1 for line in lines if _is_fragment_line(line))
    article = reversed_article(text)
    single = single_letter_stats(text)
    non_space = sum(1 for ch in text if not ch.isspace())
    garbage_ratio = garbage_chars / non_space if non_space else 0.0
    garbage_token_ratio = garbage_tokens / arabic_tokens if arabic_tokens else 0.0
    reasons = []
    if non_space == 0:
        reasons.append("no text layer content")
    if garbage_ratio > NEEDS_OCR_GARBAGE_RATIO or garbage_token_ratio > NEEDS_OCR_GARBAGE_RATIO:
        reasons.append("garbage glyph ratio above threshold")
    if (needs_ocr_rule and single["arabic_tokens"] >= needs_ocr_rule["single_letter_min_arabic_tokens"]
            and single["single_letter_arabic_ratio"] > needs_ocr_rule["single_letter_threshold"]):
        reasons.append(SPLIT_WORDS_REASON)
    needs_ocr = bool(reasons)
    return {
        **single,
        "version": DIAGNOSTICS_VERSION,
        "has_arabic": bool(ARABIC_LETTER.search(text)),
        "lines": len(lines),
        "broken_lamalef_signs": lamalef_artifacts(text),
        "reversed_word_signs": article,
        "detached_diacritics": detached_diacritics(text),
        "garbage_chars": garbage_chars,
        "garbage_arabic_tokens": garbage_tokens,
        "arabic_tokens": arabic_tokens,
        "presentation_form_ratio": round(presentation / chars, 4) if chars else 0.0,
        "fragmented_line_ratio": round(fragment_lines / len(lines), 4) if lines else 0.0,
        "needs_ocr": needs_ocr,
        "needs_ocr_reason": "; ".join(reasons) or None,
        "needs_ocr_rule": needs_ocr_rule,
        "words": len(tokens),
    }


# ---------------------------------------------------------------- ground truth
ARABIC_NORMALIZATION = (
    "NFKC; strip Arabic diacritics (U+064B-U+065F, U+0670) and tatweel (U+0640); "
    "unify أ إ آ -> ا, ة -> ه, ى -> ي; collapse whitespace"
)
_DIACRITICS = re.compile(r"[ً-ٰٟـ]")
_PAGE_SEPARATOR = re.compile(r"^\s*===\s*page\s+(\d+)\s*===\s*$", re.IGNORECASE | re.MULTILINE)


def arabic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"}))
    return " ".join(text.split())


def combine_ground_truth_files(files: list[tuple[str, bytes]]) -> str:
    """One .txt = the document (may use `=== page N ===` separators); several .txt = one per page,
    numbered by the first number in each file name (e.g. page_3.txt). UTF-8 (BOM allowed)."""
    if not files:
        raise ValueError("no ground-truth file")
    try:
        if len(files) == 1:
            return files[0][1].decode("utf-8-sig")
        numbered = []
        for name, data in files:
            match = re.search(r"(\d+)", name.rsplit(".", 1)[0])
            if not match:
                raise ValueError(f"{name}: no page number in the file name")
            numbered.append((int(match.group(1)), data.decode("utf-8-sig")))
    except UnicodeDecodeError as error:
        raise ValueError(f"ground truth must be UTF-8 text: {error}") from error
    numbers = [number for number, _ in numbered]
    if len(set(numbers)) != len(numbers):
        raise ValueError("two ground-truth files have the same page number")
    return "\n".join(f"=== page {number} ===\n{text}" for number, text in sorted(numbered))


def describe_ground_truth(text: str, page_count: int) -> dict:
    """What a ground-truth text will be used for (shown before Run Extraction)."""
    parsed = parse_ground_truth(text)
    if isinstance(parsed, dict):
        pages = sorted(parsed)
        outside = [page for page in pages if page < 1 or page > page_count]
        return {"level": "page", "pages": pages, "outside_document": outside,
                "message": f"per-page ground truth for {len(pages)} page(s)"
                           + (f"; pages {outside} are outside this PDF and will be ignored" if outside else "")}
    if not text.strip():
        return {"level": None, "pages": [], "outside_document": [], "message": "empty file"}
    return {"level": "document", "pages": [], "outside_document": [],
            "message": "document-level ground truth (compared with the full document text)"}


def parse_ground_truth(text: str) -> dict[int, str] | str:
    """`=== page N ===` separators -> {page: text}; otherwise the whole document text."""
    matches = list(_PAGE_SEPARATOR.finditer(text))
    if not matches:
        return text
    pages: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages[int(match.group(1))] = text[match.end():end].strip("\r\n")
    return pages


def _levenshtein(a, b) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:  # pragma: no cover - rapidfuzz is a lab dependency
        previous = list(range(len(b) + 1))
        for i, item in enumerate(a, 1):
            current = [i]
            for j, other in enumerate(b, 1):
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item != other)))
            previous = current
        return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict:
    """CER/WER with no normalization and with the Arabic normalization."""
    def rates(ref: str, hyp: str) -> dict:
        ref_words, hyp_words = ref.split(), hyp.split()
        return {
            "cer": _levenshtein(ref, hyp) / len(ref) if ref else None,
            "wer": _levenshtein(ref_words, hyp_words) / len(ref_words) if ref_words else None,
            "reference_chars": len(ref),
            "reference_words": len(ref_words),
        }

    return {
        "none": rates(reference, hypothesis),
        "arabic_normalized": rates(arabic_normalize(reference), arabic_normalize(hypothesis)),
        "normalization_none": "no normalization (exact characters, including whitespace)",
        "normalization_arabic": ARABIC_NORMALIZATION,
    }


def aggregate(records: list[dict]) -> dict:
    """Comparison metrics for one method from its page records (completed/failed only)."""
    completed = [record for record in records if record.get("status") == "COMPLETED"]
    failed = [record for record in records if record.get("status") == "FAILED"]
    total_ms = sum(float(record.get("extraction_ms") or 0.0) for record in completed + failed)
    statuses = Counter(record.get("reorder", {}).get("reorder_status") for record in completed if record.get("reorder"))
    single = sum(int((record.get("diagnostics") or {}).get("single_letter_arabic_tokens") or 0) for record in completed)
    arabic = sum(int((record.get("diagnostics") or {}).get("arabic_tokens") or 0) for record in completed)
    return {
        "single_letter_arabic_tokens": single,
        "arabic_tokens": arabic,
        "single_letter_arabic_ratio": round(single / arabic, 4) if arabic else None,
        "completed_pages": len(completed),
        "failed_pages": len(failed),
        "total_time_ms": round(total_ms, 3),
        "avg_time_per_page_ms": round(total_ms / len(completed + failed), 3) if completed or failed else None,
        "chars": sum(int(record.get("char_count") or 0) for record in completed),
        "words": sum(int(record.get("word_count") or 0) for record in completed),
        "empty_pages": sum(1 for record in completed if record.get("is_empty")),
        "needs_ocr_pages": sum(1 for record in completed if (record.get("diagnostics") or {}).get("needs_ocr")),
        "broken_lamalef_signs": sum((record.get("diagnostics") or {}).get("broken_lamalef_signs", 0) for record in completed),
        "detached_diacritics": sum((record.get("diagnostics") or {}).get("detached_diacritics", 0) for record in completed),
        "garbage_arabic_tokens": sum((record.get("diagnostics") or {}).get("garbage_arabic_tokens", 0) for record in completed),
        "reversed_ending_la": sum(((record.get("diagnostics") or {}).get("reversed_word_signs") or {}).get("ending_la", 0)
                                  for record in completed),
        "reversed_starting_al": sum(((record.get("diagnostics") or {}).get("reversed_word_signs") or {}).get("starting_al", 0)
                                    for record in completed),
        "reorder_status_counts": dict(statuses),
    }
