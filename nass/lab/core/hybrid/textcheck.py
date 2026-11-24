"""Text checks used by the hybrid combiner. None of them invents or rewrites text.

- `rejoin`: the `library_rejoined` candidate. It may ONLY delete horizontal whitespace
  between fragments of an Arabic word in the library text, and only when the joined
  word appears in the VLM text of the same page (compared without diacritics/tatweel).
  The non-whitespace character sequence is verified to be identical.
- `number_audit`: numbers of the library text vs the VLM text, occurrence-counted, with
  original and normalized forms. Nothing is replaced.
- `vlm_quality`: truncation and repetition indicators (word, phrase and character
  level) with table-friendly thresholds.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from lab.core.e2.metrics_e2 import repetition_stats

REJOIN_VERSION = "1"
_ARABIC_RANGES = "؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿"
_ARABIC_WORD = re.compile(rf"^[{_ARABIC_RANGES}]+$")
_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭـ]")
COMPARISON_NORMALIZATION = "NFKC; Arabic diacritics and tatweel removed (comparison only; output keeps them)"


def _is_edge_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PS"


def _strip_edges(token: str) -> str:
    start, end = 0, len(token)
    while start < end and _is_edge_punct(token[start]):
        start += 1
    while end > start and _is_edge_punct(token[end - 1]):
        end -= 1
    return token[start:end]


def comparison_key(word: str) -> str:
    return _DIACRITICS.sub("", unicodedata.normalize("NFKC", word))


def _arabic(text: str) -> bool:
    """Arabic letters/marks only (Arabic punctuation such as '،' or '؟' is excluded)."""
    return bool(text) and bool(_ARABIC_WORD.match(text)) and not any(_is_edge_punct(ch) for ch in text)


def _joinable(group: list[str]) -> bool:
    """First token may carry leading punctuation, last may carry trailing punctuation;
    every joined boundary must be between Arabic characters."""
    if len(group) < 2 or any(not token for token in group):
        return False
    first, last = group[0], group[-1]
    start = 0
    while start < len(first) and _is_edge_punct(first[start]):
        start += 1
    end = len(last)
    while end > 0 and _is_edge_punct(last[end - 1]):
        end -= 1
    return _arabic(first[start:]) and _arabic(last[:end]) and all(_arabic(token) for token in group[1:-1])


def rejoin(library_text: str, vlm_text: str, *, max_parts: int = 8) -> dict:
    """Build the `library_rejoined` candidate. Returns text, joins (with evidence) and checks."""
    vlm_words = {comparison_key(_strip_edges(word)) for word in vlm_text.split()}
    vlm_words.discard("")
    joins: list[dict] = []
    out_lines = []
    for line_number, line in enumerate(library_text.split("\n"), start=1):
        parts = re.split(r"([ \t ‏‎]+)", line)
        tokens, spaces = parts[0::2], parts[1::2]
        rebuilt: list[str] = []
        index = 0
        while index < len(tokens):
            end = index
            for candidate in range(min(len(tokens) - 1, index + max_parts - 1), index, -1):
                group = tokens[index:candidate + 1]
                if not _joinable(group):
                    continue
                key = comparison_key(_strip_edges("".join(group)))
                if key in vlm_words:
                    end = candidate
                    joins.append({"line": line_number, "fragments": group, "joined": "".join(group),
                                  "evidence": f"word {key!r} appears in the VLM text of this page"})
                    break
            rebuilt.append("".join(tokens[index:end + 1]))
            if end < len(spaces):
                rebuilt.append(spaces[end])
            index = end + 1
        out_lines.append("".join(rebuilt))
    candidate_text = "\n".join(out_lines)
    same_chars = re.sub(r"\s+", "", library_text) == re.sub(r"\s+", "", candidate_text)
    same_lines = library_text.count("\n") == candidate_text.count("\n")
    return {
        "version": REJOIN_VERSION,
        "text": candidate_text,
        "joins": joins,
        "join_count": len(joins),
        "non_space_sequence_identical": same_chars,
        "line_count_identical": same_lines,
        "valid": same_chars and same_lines,
        "comparison_normalization": COMPARISON_NORMALIZATION,
        "note": "Character conservation does not prove correct reading order or correct words.",
    }


# ------------------------------------------------------------------ numbers

_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_D = "0-9٠-٩۰-۹"
NUMBER = re.compile(rf"(?<![{_D}])([+\-−]?)([{_D}]+(?:[.,٫٬][{_D}]+)*)(\s?[%٪])?")
NUMBER_NORMALIZATION = (
    "Arabic-Indic and Persian digits -> 0-9; '٬' and ',' in groups of three -> thousands "
    "separators (removed); '٫' and '.' -> decimal point; other ',' -> decimal point; "
    "'−' -> '-'; '%'/'٪' kept as '%'. Counted per occurrence."
)


def normalize_number(sign: str, body: str, percent: str | None) -> str:
    ascii_body = body.translate(_DIGIT_MAP).replace("٬", ",").replace("٫", ".")
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", ascii_body):
        ascii_body = ascii_body.replace(",", "")
    elif "," in ascii_body and "." not in ascii_body and ascii_body.count(",") == 1:
        ascii_body = ascii_body.replace(",", ".")
    sign = "-" if sign in {"-", "−"} else ("+" if sign == "+" else "")
    return f"{sign}{ascii_body}{'%' if percent else ''}"


def extract_numbers(text: str) -> list[dict]:
    found = []
    for match in NUMBER.finditer(text):
        sign, body, percent = match.group(1), match.group(2), match.group(3)
        found.append({"original": match.group(0), "normalized": normalize_number(sign, body, percent),
                      "position": match.start()})
    return found


def number_audit(library_text: str, vlm_text: str, *, library_reliable: bool = True) -> dict:
    library = extract_numbers(library_text)
    vlm = extract_numbers(vlm_text)
    lib_counts = Counter(item["normalized"] for item in library)
    vlm_counts = Counter(item["normalized"] for item in vlm)
    vlm_only = vlm_counts - lib_counts
    library_only = lib_counts - vlm_counts
    originals: dict[str, dict[str, list[str]]] = {}
    for side, items in (("library", library), ("vlm", vlm)):
        for item in items:
            originals.setdefault(item["normalized"], {"library": [], "vlm": []})[side].append(item["original"])
    conflicts = [
        {"normalized": value, "library_count": lib_counts.get(value, 0), "vlm_count": vlm_counts.get(value, 0),
         "library_originals": originals.get(value, {}).get("library", []),
         "vlm_originals": originals.get(value, {}).get("vlm", [])}
        for value in sorted(set(vlm_only) | set(library_only))
    ]
    return {
        "library_numbers": len(library), "vlm_numbers": len(vlm),
        "matched_occurrences": sum((lib_counts & vlm_counts).values()),
        "vlm_only": dict(vlm_only), "library_only": dict(library_only),
        "conflicts": conflicts,
        "needs_review": bool(vlm_only),
        "library_reference_reliable": library_reliable,
        "normalization": NUMBER_NORMALIZATION,
        "note": "No number is replaced. The library value is not assumed correct"
                + ("" if library_reliable else " (the library text layer is unreliable on this page)") + ".",
    }


# ------------------------------------------------------------------ quality

_IGNORED_RUN_CHARS = set("-=_|.·*#~ ")


def _phrase_repeats(tokens: list[str], *, min_repeats: int = 4) -> dict | None:
    for size in range(2, 9):
        index = 0
        while index + size * min_repeats <= len(tokens):
            phrase = tokens[index:index + size]
            count = 1
            while tokens[index + count * size:index + (count + 1) * size] == phrase:
                count += 1
            # Rows of numbers/separators legitimately repeat in tables: ignore them.
            if count >= min_repeats and not all(re.fullmatch(rf"[{_D}.,%|:\-]+", token) for token in phrase):
                return {"phrase": " ".join(phrase), "repeats": count, "position_token": index}
            index += 1
    return None


def _char_runs(text: str) -> list[dict]:
    runs = []
    for match in re.finditer(r"(\S)\1{24,}", text):
        if match.group(1) not in _IGNORED_RUN_CHARS:
            runs.append({"char": f"U+{ord(match.group(1)):04X}", "length": len(match.group(0))})
    for match in re.finditer(rf"([{_D}])\1{{11,}}", text):  # e.g. ٠٠٠٠٠٠٠٠٠٠٠٠
        runs.append({"char": f"U+{ord(match.group(1)):04X}", "length": len(match.group(0)), "digits": True})
    for match in re.finditer(r"ـ{10,}", text):  # long tatweel
        runs.append({"char": "U+0640", "length": len(match.group(0))})
    return runs


def _table_like(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") or bool(re.fullmatch(rf"[{_D}\s.,%|:\-+()]*", stripped))


def vlm_quality(text: str, *, finish_reason: str | None = None, truncated: bool | None = None) -> dict:
    tokens = text.split()
    # Identical table rows and number-only lines are legitimate; they are excluded from the
    # word/line repetition check (character runs are still checked on the whole text).
    words = repetition_stats("\n".join(line for line in text.splitlines() if not _table_like(line)))
    phrase = _phrase_repeats(tokens)
    runs = _char_runs(text)
    reasons = []
    if not text.strip():
        reasons.append("empty output")
    if truncated is True or finish_reason == "length":
        reasons.append("generation reached max_tokens (output may be incomplete)")
    if words["flagged"]:
        reasons.append("word-level repetition")
    if phrase:
        reasons.append(f"phrase repeated {phrase['repeats']} times")
    if runs:
        reasons.append("long run of one character")
    return {"word_repetition": words, "phrase_repetition": phrase, "char_runs": runs,
            "finish_reason": finish_reason, "truncated_at_token_limit": truncated,
            "flagged": bool(reasons), "reasons": reasons}
