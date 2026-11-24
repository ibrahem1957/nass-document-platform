"""Derived text segmentation inside one original element (for example a scanned page's single page_text).

Guarantees:
- the original text is never modified; every span is exactly original[start:end];
- spans are contiguous and non-overlapping and cover the whole text (no character is dropped);
- no coordinates are invented: geometry is attached only when the saved layout gives a reliable container,
  and it is labelled with its precision;
- spans whose role rests on weak evidence are marked uncertain.
"""

from __future__ import annotations

import re
from itertools import pairwise

from .config import SegmentationConfig
from .layout import LayoutBox, role_of
from .model import DerivedSpan
from .text_patterns import (
    CAPTION_INLINE_RE,
    ORDINAL_ALT,
    ascii_digits,
    bullet_marker,
    caption_match,
    collapse,
    ends_with_colon,
    ends_with_terminal,
    has_content,
    match_text,
    numbering,
    strip_tashkeel,
    words,
)

_SENTENCE_END_RE = re.compile(r"[.!?؟…。]+[\"'»”’)\]]*\s+")
_COLON_RE = re.compile(r"[:：]\s+")
_BULLET_INLINE_RE = re.compile(r"(?<=\s)[•●▪■◦‣∙○◆►▶➢➤✓✔](?=\s)")
# Applied to strip_tashkeel_keep_length(probe), where diacritics became U+200B (offsets unchanged).
_ORDINAL_INLINE_RE = re.compile(rf"(?<=\s)(?:{ORDINAL_ALT})\u200B*\s*[:\-–]")
_TOKEN_RE = re.compile(r"\S+")
_DECIMAL_TAIL_RE = re.compile(r"\d[.,]$")


def _norm_token(token: str) -> str:
    return re.sub(r"[^\w]", "", strip_tashkeel(ascii_digits(token))).lower()


def _title_echo(text: str, end: int, cfg: SegmentationConfig) -> int | None:
    """Position where the first k words of text[0:end] are repeated (a title echoed at the start of the
    following sentence). Returns None when not found or when the prefix already ends a sentence."""
    tokens = [(m.start(), _norm_token(m.group())) for m in _TOKEN_RE.finditer(text, 0, end)]
    tokens = [(p, t) for p, t in tokens if t]
    k = cfg.title_echo_min_words
    if len(tokens) < 2 * k:
        return None
    head = [t for _, t in tokens[:k]]
    for j in range(k, min(len(tokens) - k, cfg.title_echo_max_words) + 1):
        if [t for _, t in tokens[j:j + k]] == head:
            if ends_with_terminal(text[:tokens[j][0]]):
                return None
            return tokens[j][0]
    return None


def _cut_points(text: str, cfg: SegmentationConfig) -> tuple[dict[int, str], bool]:
    probe = ascii_digits(text)  # same length as text: offsets remain valid
    cuts: dict[int, str] = {}

    def add(position: int, reason: str) -> None:
        if 0 < position < len(text) and position not in cuts:
            cuts[position] = reason

    for m in CAPTION_INLINE_RE.finditer(probe):
        add(m.start(), "caption_start")
    for m in _SENTENCE_END_RE.finditer(probe):
        if _DECIMAL_TAIL_RE.search(probe[max(0, m.start() - 1):m.start() + 1]) and m.end() < len(probe) \
                and probe[m.end()].isdigit():
            continue  # "3. 5" style numbers are not sentence ends
        add(m.end(), "sentence_end")
    for m in _BULLET_INLINE_RE.finditer(probe):
        add(m.start(), "bullet")
    for m in _ORDINAL_INLINE_RE.finditer(strip_tashkeel_keep_length(probe)):
        add(m.start(), "ordinal")
    for m in _COLON_RE.finditer(probe):
        start = max([0] + [c for c in cuts if c <= m.start()])
        label = text[start:m.start() + 1]
        if caption_match(label)[0]:
            continue
        if 0 < len(words(label)) <= cfg.colon_label_max_words:
            add(m.end(), "after_colon_label")
    first_end = min(list(cuts) + [len(text)])
    echo = _title_echo(text, first_end, cfg)
    if echo is not None:
        add(echo, "title_echo")
    return cuts, echo is not None


def strip_tashkeel_keep_length(text: str) -> str:
    """Replace diacritics by a neutral character so regex offsets still match the original text."""
    return re.sub(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]", "\u200B", text)


def _role(span_text: str, is_title_echo_prefix: bool, cfg: SegmentationConfig) -> str:
    t = collapse(span_text)
    if not t:
        return "unknown"
    if caption_match(t)[0]:
        return "caption"
    if bullet_marker(t):
        return "list_item"
    n = len(words(t))
    if is_title_echo_prefix:
        return "heading"
    if ends_with_colon(t) and n <= cfg.colon_label_max_words:
        return "heading"
    if numbering(t) and n <= cfg.colon_label_max_words and not ends_with_terminal(t):
        return "heading"
    return "paragraph"


_LAYOUT_MATCH = {"heading": {"heading", "furniture"}, "caption": {"caption"}, "paragraph": {"body"},
                 "list_item": {"body"}}


def segment_text(element_id: str, text: str, boxes: list[LayoutBox], cfg: SegmentationConfig) -> list[DerivedSpan]:
    if not text or not text.strip():
        return []
    cuts, echoed = _cut_points(text, cfg)
    points = [0] + sorted(cuts) + [len(text)]
    raw = [(points[i], points[i + 1], "element_start" if i == 0 else cuts[points[i]]) for i in range(len(points) - 1)]
    merged: list[list] = []
    for start, end, reason in raw:
        if merged and not has_content(text[start:end]):
            merged[-1][1] = end  # punctuation-only remainder joins the previous span (nothing is dropped)
            continue
        merged.append([start, end, reason])
    if len(merged) > 1 and not has_content(text[merged[0][0]:merged[0][1]]):
        merged[1][0] = merged[0][0]
        merged[1][2] = merged[0][2]
        merged.pop(0)

    labels = [box.label for box in boxes]
    roles_present = {role_of(label) for label in labels}
    single_block = boxes[0] if len(boxes) == 1 else None
    spans: list[DerivedSpan] = []
    for index, (start, end, reason) in enumerate(merged, start=1):
        piece = text[start:end]
        is_echo_prefix = echoed and index == 1 and len(merged) > 1 and merged[1][2] == "title_echo"
        role = _role(piece, is_echo_prefix, cfg)
        basis = []
        if reason not in ("element_start",):
            basis.append(f"text:{reason}")
        if index < len(merged):
            basis.append(f"text:ends_before_{merged[index][2]}")
        if is_echo_prefix:
            basis.append("text:title_echoed_by_following_sentence")
        layout_hits = sorted(label for label in labels if role_of(label) in _LAYOUT_MATCH.get(role, set()))
        for label in layout_hits:
            basis.append(f"layout:{label} present in this text segment (offsets not recorded by the layout)")
        text_supported = len(merged) > 1
        if text_supported and layout_hits:
            support = "both"
        elif text_supported:
            support = "text"
        elif layout_hits:
            support = "layout"
        else:
            support = "none"
        geometry = None
        if single_block is not None:
            geometry = {"container_bbox_pt": [round(v, 2) for v in single_block.bbox],
                        "layout_label": single_block.label, "block_index": single_block.block_index,
                        "precision": "segment container only (one layout block); the span position inside it "
                                     "is unknown"}
        uncertain = (role == "heading" and support != "both") or reason in ("title_echo", "after_colon_label") \
            or is_echo_prefix or support == "none"
        notes = []
        if len(merged) == 1:
            notes.append("no reliable boundary found; the whole element is one span")
        if roles_present and not layout_hits and role in _LAYOUT_MATCH:
            notes.append(f"layout labels in this segment ({', '.join(labels)}) do not corroborate role '{role}'")
        spans.append(DerivedSpan(
            span_id=f"{element_id}_span_{index:03d}", element_id=element_id, start=start, end=end, text=piece,
            cut_reason=reason, role_hint=role, support=support, basis=basis, geometry=geometry,
            uncertain=bool(uncertain), notes=notes))
    _sequence_geometry(spans, boxes)
    return spans


def _sequence_geometry(spans: list[DerivedSpan], boxes: list[LayoutBox]) -> None:
    """When the spans and the segment's layout blocks correspond one to one in order and role, attach each
    block as an approximate location. Otherwise leave geometry as it is (container or none)."""
    if len(boxes) < 2 or len(spans) != len(boxes):
        return
    for span, box in zip(spans, boxes):
        if role_of(box.label) not in _LAYOUT_MATCH.get(span.role_hint, set()):
            return
    for span, box in zip(spans, boxes):
        span.geometry = {"bbox_pt": [round(v, 2) for v in box.bbox], "layout_label": box.label,
                         "block_index": box.block_index,
                         "precision": "approximate: matched to the layout block by order and role"}
        if "layout:sequence_match" not in span.basis:
            span.basis.append("layout:sequence_match")


def check_partition(text: str, spans: list[DerivedSpan]) -> list[str]:
    """Problems with a segmentation (empty when the spans exactly partition the text)."""
    problems = []
    if not spans:
        return ["no spans"] if text else []
    if spans[0].start != 0:
        problems.append("first span does not start at offset 0")
    if spans[-1].end != len(text):
        problems.append("last span does not end at the end of the text")
    for a, b in pairwise(spans):
        if a.end != b.start:
            problems.append(f"gap or overlap between {a.span_id} and {b.span_id}")
    for span in spans:
        if text[span.start:span.end] != span.text:
            problems.append(f"{span.span_id} text differs from the original slice")
    if "".join(s.text for s in spans) != text:
        problems.append("concatenated spans differ from the original text")
    return problems


__all__ = ["check_partition", "match_text", "segment_text"]
