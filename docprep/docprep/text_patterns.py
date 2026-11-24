"""Language-aware text patterns (Arabic and English). Used for matching only; the text itself is never changed."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
TASHKEEL_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
ARABIC_LETTER_RE = re.compile(r"[ء-يٮ-ۓۺ-ۿﭐ-﷿ﹰ-ﻼ]")
LATIN_LETTER_RE = re.compile(r"[A-Za-zÀ-ɏ]")
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[^\W_]+")


def ascii_digits(text: str) -> str:
    """Arabic-Indic and Persian digits -> ASCII digits. Same length as the input (offsets stay valid)."""
    return text.translate(_DIGIT_MAP)


def strip_tashkeel(text: str) -> str:
    return TASHKEEL_RE.sub("", text)


def collapse(text: str) -> str:
    return " ".join(text.split())


def match_text(text: str) -> str:
    """Normalized copy for pattern matching only."""
    return collapse(strip_tashkeel(ascii_digits(text or "")))


def words(text: str) -> list[str]:
    return WORD_RE.findall(match_text(text))


def letter_count(text: str) -> int:
    return len(ARABIC_LETTER_RE.findall(text)) + len(LATIN_LETTER_RE.findall(text))


def digit_count(text: str) -> int:
    return len(DIGIT_RE.findall(ascii_digits(text)))


def script_of(text: str) -> str:
    arabic, latin = len(ARABIC_LETTER_RE.findall(text)), len(LATIN_LETTER_RE.findall(text))
    if arabic == latin == 0:
        return "none"
    if arabic >= 0.8 * (arabic + latin):
        return "arabic"
    if latin >= 0.8 * (arabic + latin):
        return "latin"
    return "mixed"


# ------------------------------------------------------------------ numbering

@dataclass(frozen=True)
class Numbering:
    scheme: str        # decimal | decimal_multi | paren_number | latin_letter | arabic_letter | roman | ordinal | keyword
    label: str
    depth: int
    marker_length: int

    def to_dict(self) -> dict:
        return asdict(self)


ORDINALS_AR = ("اولا", "أولا", "ثانيا", "ثالثا", "رابعا", "خامسا", "سادسا", "سابعا", "ثامنا", "تاسعا", "عاشرا")
ORDINAL_ALT = "|".join(ORDINALS_AR)
_ORDINAL_RE = re.compile(rf"^(?:{ORDINAL_ALT})(?=\s*[:\-–.]|\s|$)")
_KEYWORD_RE = re.compile(
    r"^(?:chapter|section|part|appendix|الفصل|الباب|المبحث|المطلب|القسم|الجزء|الملحق)\s+(\S+)",
    re.IGNORECASE,
)
_MULTI_RE = re.compile(r"^(\d{1,3}(?:\.\d{1,3})+)\.?(?=\s|$|[):\-–])")
_DECIMAL_RE = re.compile(r"^(\d{1,3})\s?[.)\-–]\s*(?=[^\d\s])")
_PAREN_RE = re.compile(r"^\((\d{1,3})\)\s*(?=\S)")
_LATIN_LETTER_RE = re.compile(r"^\(?([a-zA-Z])\)\s*(?=\S)")
_ARABIC_LETTER_RE = re.compile(r"^\(?([اأبجدهوزحطيكلمنسعفصقرشتثخذضظغ])\s?\)\s*(?=\S)")
_ROMAN_RE = re.compile(
    r"^(?=[IVXLCDM])(M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\s?[.)\-–]\s+(?=\S)")


def numbering(text: str) -> Numbering | None:
    t = match_text(text)
    if not t:
        return None
    m = _MULTI_RE.match(t)
    if m:
        return Numbering("decimal_multi", m.group(1), m.group(1).count(".") + 1, m.end())
    m = _PAREN_RE.match(t)
    if m:
        return Numbering("paren_number", m.group(1), 1, m.end())
    m = _DECIMAL_RE.match(t)
    if m:
        return Numbering("decimal", m.group(1), 1, m.end())
    m = _ROMAN_RE.match(t)
    if m and m.group(1):
        return Numbering("roman", m.group(1), 1, m.end())
    m = _LATIN_LETTER_RE.match(t)
    if m:
        return Numbering("latin_letter", m.group(1), 1, m.end())
    m = _ARABIC_LETTER_RE.match(t)
    if m:
        return Numbering("arabic_letter", m.group(1), 1, m.end())
    m = _ORDINAL_RE.match(t)
    if m:
        return Numbering("ordinal", m.group(0), 1, m.end())
    m = _KEYWORD_RE.match(t)
    if m:
        return Numbering("keyword", m.group(0), 1, m.end())
    return None


# ------------------------------------------------------------------ bullets, captions, punctuation

_BULLET_CHARS = "•●▪■◦‣∙·○◆◇►▶➢➤✓✔※❖➔→◄◘▫"
_BULLET_RE = re.compile(rf"^([{_BULLET_CHARS}])\s*(?=\S)")
_DASH_BULLET_RE = re.compile(r"^([\-–—*])\s+(?=[^\d\s])")
# After logical reordering of an RTL line that holds left-to-right text, the visual list marker can end up at
# the end of the logical string ("Power BI -").
_TRAILING_MARKER_RE = re.compile(rf"(?<=\S)\s+([\-–—*{_BULLET_CHARS}])\s*$")


def bullet_marker(text: str) -> str | None:
    t = collapse(text or "")
    m = _BULLET_RE.match(t) or _DASH_BULLET_RE.match(t)
    return m.group(1) if m else None


def list_marker(text: str, direction: str | None = None, max_words: int = 12) -> tuple[str | None, str | None]:
    """(marker, position) with position 'start' or 'end'. End markers are accepted only on RTL pages and on
    short lines, where they come from the logical reordering of a visual list marker."""
    marker = bullet_marker(text)
    if marker:
        return marker, "start"
    if direction == "rtl":
        t = collapse(text or "")
        m = _TRAILING_MARKER_RE.search(t)
        if m and len(words(t)) <= max_words and letter_count(t[:m.start()]) > 0:
            return m.group(1), "end"
    return None, None


# ------------------------------------------------------------------ list introductions (language cues only)

def _unify(text: str) -> str:
    return match_text(text).lower().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")


INTRO_OPENERS = {
    # Arabic connectives, discourse markers and verbs that open a sentence rather than a title
    "لذلك", "لذا", "كما", "ثم", "حيث", "وقد", "قد", "ويمكن", "يمكن", "ومن", "وتتضمن", "تتضمن", "يتضمن", "ويتضمن",
    "وتشمل", "تشمل", "يشمل", "ويشمل", "وشملت", "شملت", "وشمل", "شمل", "تم", "يتم", "وتم", "وهي", "وهو", "وهذه",
    "هذه", "فيما", "اذ", "وذلك", "بالاضافة", "وفيما", "ومنها", "منها", "سيجيب", "يجيب", "يوضح", "توضح", "يعرض",
    "تعرض", "نذكر", "نورد", "يوصي", "يوصى", "ويوصى", "نوصي", "وتوصي", "كذلك", "اضافة",
    # English
    "this", "these", "there", "we", "it", "they", "for", "such", "including", "includes", "include", "in",
    "additionally", "therefore", "thus", "also", "as", "here", "below", "following",
}
INTRO_ENDINGS = [("مثل",), ("يلي",), ("كالتالي",), ("التالي",), ("الاتي",), ("التالية",), ("الاتية",), ("من", "خلال"),
                 ("ب",), ("منها",), ("ابرزها",), ("following",), ("as", "follows"), ("such", "as"), ("include",),
                 ("includes",), ("including",), ("are",), ("by",), ("through",), ("like",)]


def list_intro_cues(text: str) -> list[str]:
    """Language cues that a colon-ended line is a sentence introducing what follows (not a title)."""
    tokens = [t for t in re.split(r"[\s:：،,]+", _unify(text).replace("ـ", "")) if t]
    cues: list[str] = []
    if tokens and tokens[0] in INTRO_OPENERS:
        cues.append(f"opens with the connective or verb '{tokens[0]}'")
    for ending in INTRO_ENDINGS:
        if len(tokens) >= len(ending) and tuple(tokens[-len(ending):]) == ending:
            cues.append(f"ends with the introducing phrase '{' '.join(ending)}'")
            break
    return cues


CAPTION_WORDS = (r"(?:الشكل|شكل|الجدول|جدول|المخطط|مخطط|الرسم\s+البياني|رسم\s+بياني|الصورة|صورة|"
                 r"figure|fig\.?|table|chart|exhibit|graph|image|plate)")
_CAPTION_START_RE = re.compile(
    rf"^\(?\s*{CAPTION_WORDS}\s*(?:رقم\s*)?[(\[]?\s*\d+(?:[.\-]\d+)*\s*[)\]]?\s*(?:[:.\-–—]|$)",
    re.IGNORECASE,
)
_CAPTION_REF_RE = re.compile(
    rf"^\(?\s*{CAPTION_WORDS}\s*(?:رقم\s*)?\d+(?:[.\-]\d+)*\s+\S",
    re.IGNORECASE,
)
# Used by segmentation on ascii_digits(text) only (same length as the original text, so offsets stay valid).
CAPTION_INLINE_RE = re.compile(
    rf"(?:(?<=\s)|^){CAPTION_WORDS}\s*(?:رقم\s*)?\d+(?:[.\-]\d+)*\s*[:\-–—]",
    re.IGNORECASE,
)

_TERMINAL_RE = re.compile(r"[.!?؟…。][\"'»”’)\]]*\s*$")
_COLON_END_RE = re.compile(r"[:：]\s*$")
_PAGE_NUMBER_RE = re.compile(
    r"^[\-–—\s]*(?:(?:page|p\.|صفحة|الصفحة)\s*)?\d{1,4}(?:\s*(?:/|of|من)\s*\d{1,4})?[\-–—\s]*$",
    re.IGNORECASE,
)
_NUMBER_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*%?")


def caption_match(text: str, short_words: int = 10) -> tuple[bool, str]:
    t = match_text(text)
    if _CAPTION_START_RE.match(t):
        return True, "starts with a figure/table caption label and number"
    if _CAPTION_REF_RE.match(t) and len(words(t)) <= short_words and not ends_with_terminal(t):
        return True, "short label starting with a figure/table word and number"
    return False, ""


def ends_with_terminal(text: str) -> bool:
    return bool(_TERMINAL_RE.search(collapse(text or "")))


def ends_with_colon(text: str) -> bool:
    return bool(_COLON_END_RE.search(collapse(text or "")))


def is_page_number(text: str) -> bool:
    return bool(_PAGE_NUMBER_RE.match(match_text(text)))


def is_numeric_label(text: str) -> bool:
    t = match_text(text)
    return bool(t) and letter_count(t) <= 1 and digit_count(t) >= 1 and len(t.split()) <= 4


def is_axis_like(text: str) -> bool:
    t = match_text(text)
    return len(_NUMBER_TOKEN_RE.findall(t)) >= 3 and letter_count(t) <= 3


def is_fragment(text: str, max_chars: int) -> bool:
    t = collapse(text or "")
    return len(t) <= max_chars or (letter_count(t) <= 1 and digit_count(t) == 0)


def has_content(text: str) -> bool:
    return letter_count(text) > 0 or digit_count(text) > 0
