"""Typography from the original source.pdf (digital and mixed pages only; never inferred from images).

PyMuPDF is imported lazily: without it, or without source.pdf, typography is reported as unavailable.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import TypographyConfig

_BOLD_NAME_RE = re.compile(
    r"bold|black|heavy|semibold|demibold|extrabold|ultrabold|\bbd\b|-bd\b|,b\b",
    re.IGNORECASE,
)
_ITALIC_NAME_RE = re.compile(r"italic|oblique", re.IGNORECASE)


@dataclass
class PdfSpan:
    page: int
    bbox: tuple[float, float, float, float]
    text: str
    size: float
    font: str
    flags: int
    color: int
    bold: bool
    italic: bool


@dataclass
class TextTrace:
    """One PyMuPDF text-trace record reduced to what fake-bold detection needs."""
    page: int
    kind: int                                   # 0 fill, 1 stroke (other kinds are ignored)
    bbox: tuple[float, float, float, float]
    text: str


@dataclass
class PdfTypography:
    spans: dict[int, list[PdfSpan]]
    page_sizes: dict[int, tuple[float, float]]
    pages_read: list[int]
    warnings: list[str]
    traces: dict[int, list[TextTrace]] = field(default_factory=dict)
    trace_pages: set[int] = field(default_factory=set)   # pages whose text trace could be read


@dataclass
class ElementTypography:
    available: bool
    reason: str
    size: float | None = None
    size_share: float | None = None
    font: str | None = None
    bold_share: float | None = None
    color: str | None = None
    chars: int = 0
    char_match: float | None = None
    coverage: float | None = None
    span_count: int = 0
    size_histogram: dict[float, int] = field(default_factory=dict)
    fake_bold_status: str = "unknown"       # fill_and_stroke | none | unknown
    fake_bold_share: float | None = None    # share of element characters drawn with fill and stroke
    fake_bold_note: str = "text trace not evaluated"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["size_histogram"] = {str(k): v for k, v in self.size_histogram.items()}
        return data


@dataclass
class BodyProfile:
    size: float | None
    color: str | None
    bold_share: float | None
    chars: int
    histogram: dict[float, int]
    note: str
    fake_bold_share: float | None = None    # among characters whose fake-bold status is known

    def to_dict(self) -> dict:
        return {"size": self.size, "color": self.color, "bold_share": self.bold_share, "chars": self.chars,
                "histogram": {str(k): v for k, v in sorted(self.histogram.items())}, "note": self.note,
                "fake_bold_share": self.fake_bold_share}


def is_bold(font: str, flags: int) -> bool:
    return bool(flags & 16) or bool(_BOLD_NAME_RE.search(font or ""))


def color_hex(color: int | None) -> str:
    return f"#{(color or 0) & 0xFFFFFF:06x}"


def color_distance(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    ra, ga, ba = (int(a[i:i + 2], 16) for i in (1, 3, 5))
    rb, gb, bb = (int(b[i:i + 2], 16) for i in (1, 3, 5))
    return ((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2) ** 0.5


def read_pdf(pdf_path: Path, pages_for_spans: set[int]) -> PdfTypography:
    """Page sizes for every page; text spans only for the requested (digital/mixed) pages."""
    try:
        import pymupdf
    except ImportError:
        return PdfTypography({}, {}, [], ["PyMuPDF is not installed; typography is unavailable"])
    spans: dict[int, list[PdfSpan]] = {}
    sizes: dict[int, tuple[float, float]] = {}
    traces: dict[int, list[TextTrace]] = {}
    trace_pages: set[int] = set()
    trace_warnings: list[str] = []
    try:
        with pymupdf.open(str(pdf_path)) as document:
            for number, page in enumerate(document, start=1):
                sizes[number] = (float(page.rect.width), float(page.rect.height))
                if number not in pages_for_spans:
                    continue
                collected = []
                for block in page.get_text("dict").get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            if not text.strip():
                                continue
                            font, flags = str(span.get("font", "")), int(span.get("flags", 0))
                            collected.append(PdfSpan(
                                page=number, bbox=tuple(float(v) for v in span["bbox"]), text=text,
                                size=float(span.get("size", 0.0)), font=font, flags=flags,
                                color=int(span.get("color", 0)), bold=is_bold(font, flags),
                                italic=bool(flags & 2) or bool(_ITALIC_NAME_RE.search(font))))
                spans[number] = collected
                try:
                    trace_list = []
                    for item in page.get_texttrace():
                        kind = int(item.get("type", -1))
                        if kind not in (0, 1):
                            continue
                        chars = "".join(chr(c[0]) for c in item.get("chars", ()) if c and c[0] > 0)
                        if chars.strip():
                            trace_list.append(TextTrace(number, kind, tuple(float(v) for v in item["bbox"]), chars))
                    traces[number] = trace_list
                    trace_pages.add(number)
                except Exception as exc:  # noqa: BLE001 - fake bold then stays "unknown"
                    trace_warnings.append(f"page {number}: text trace unavailable ({type(exc).__name__}); "
                                          "fake-bold status unknown")
    except Exception as exc:  # noqa: BLE001 - any unreadable PDF only disables typography
        return PdfTypography({}, sizes, [], [(f"source.pdf could not be read ({type(exc).__name__}: {exc}); "
                                              "typography is unavailable")])
    return PdfTypography(spans, sizes, sorted(spans), trace_warnings, traces, trace_pages)


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a, b) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _chars(text: str) -> collections.Counter:
    return collections.Counter(ch for ch in unicodedata.normalize("NFKC", text) if not ch.isspace())


def associate_page(elements: list[tuple[str, list[float], str]], spans: list[PdfSpan],
                   cfg: TypographyConfig) -> dict[str, ElementTypography]:
    """Attach each span to at most one element (largest overlap), then accept an element's spans only when
    their characters agree with the element text. Avoids attaching a neighbour's font to an element."""
    tol = cfg.bbox_tolerance
    owners: dict[str, list[PdfSpan]] = collections.defaultdict(list)
    for span in spans:
        span_area = _area(span.bbox)
        if span_area <= 0:
            continue
        best: tuple[float, float, str] | None = None
        for element_id, bbox, _ in elements:
            grown = (bbox[0] - tol, bbox[1] - tol, bbox[2] + tol, bbox[3] + tol)
            overlap = _intersection(span.bbox, grown) / span_area
            key = (overlap, -_area(bbox), element_id)
            if overlap > 0 and (best is None or key[:2] > best[:2]):
                best = key
        if best is not None and best[0] >= cfg.min_span_overlap:
            owners[best[2]].append(span)
    result: dict[str, ElementTypography] = {}
    for element_id, _bbox, text in elements:
        own = owners.get(element_id, [])
        if not own:
            result[element_id] = ElementTypography(False, "no PDF text span inside the element box")
            continue
        span_chars: collections.Counter = collections.Counter()
        for span in own:
            span_chars += _chars(span.text)
        element_chars = _chars(text or "")
        matched = sum((span_chars & element_chars).values())
        char_match = matched / max(1, sum(span_chars.values()))
        coverage = matched / max(1, sum(element_chars.values()))
        if char_match < cfg.min_char_match or coverage < cfg.min_element_coverage:
            result[element_id] = ElementTypography(
                False, f"spans inside the box do not match the element text (match {char_match:.2f}, "
                       f"coverage {coverage:.2f}); typography not attached",
                char_match=round(char_match, 3), coverage=round(coverage, 3), span_count=len(own))
            continue
        histogram: collections.Counter = collections.Counter()
        fonts: collections.Counter = collections.Counter()
        colors: collections.Counter = collections.Counter()
        bold_chars = total = 0
        for span in own:
            n = sum(_chars(span.text).values())
            total += n
            histogram[round(span.size * 2) / 2] += n
            fonts[span.font] += n
            colors[color_hex(span.color)] += n
            bold_chars += n if span.bold else 0
        size = max(histogram.items(), key=lambda kv: (kv[1], kv[0]))[0]
        share = sum(n for s, n in histogram.items() if abs(s - size) <= cfg.size_tolerance) / max(1, total)
        result[element_id] = ElementTypography(
            True, "matched PDF spans by position and characters", size=size, size_share=round(share, 3),
            font=fonts.most_common(1)[0][0], bold_share=round(bold_chars / max(1, total), 3),
            color=colors.most_common(1)[0][0], chars=total, char_match=round(char_match, 3),
            coverage=round(coverage, 3), span_count=len(own), size_histogram=dict(histogram))
    return result


def body_profile(typographies: Iterable[ElementTypography]) -> BodyProfile:
    """Predominant body style: the character-weighted most frequent size (ties -> the smaller size)."""
    histogram: collections.Counter = collections.Counter()
    colors: collections.Counter = collections.Counter()
    bold = 0.0
    total = 0
    for typography in typographies:
        if not typography.available:
            continue
        for size, n in typography.size_histogram.items():
            histogram[size] += n
        colors[typography.color] += typography.chars
        bold += (typography.bold_share or 0.0) * typography.chars
        total += typography.chars
    if not total:
        return BodyProfile(None, None, None, 0, {}, "no typography available (scanned pages, no source.pdf, "
                                                    "or no matching text spans)")
    size = max(histogram.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return BodyProfile(size, colors.most_common(1)[0][0], round(bold / total, 3), total, dict(histogram),
                       "character-weighted over text elements with matched PDF spans")


def body_fake_bold_share(typographies: Iterable[ElementTypography]) -> float | None:
    """Character-weighted share of fake-bold characters among elements whose status is known."""
    total = fake = 0.0
    for t in typographies:
        if t.available and t.fake_bold_status in ("fill_and_stroke", "none") and t.fake_bold_share is not None:
            total += t.chars
            fake += t.chars * t.fake_bold_share
    return round(fake / total, 3) if total else None


def attach_fake_bold(elements: list[tuple[str, list[float], str]], traces: list[TextTrace],
                     typographies: dict[str, ElementTypography], cfg: TypographyConfig) -> None:
    """Word renders 'bold' for some fonts by filling and stroking the same glyphs (PDF text render mode 2);
    the font name and span flags then show no bold. A stroked trace counts only when an identical fill trace
    exists at the same position, it lies inside the element box, and its characters match the element text.
    Anything that cannot be mapped reliably stays 'unknown'."""
    tol = cfg.bbox_tolerance
    fills = [t for t in traces if t.kind == 0]
    strokes = [t for t in traces if t.kind == 1]

    def paired(stroke: TextTrace) -> bool:
        return any(f.text == stroke.text and all(abs(a - b) <= 1.0 for a, b in zip(f.bbox, stroke.bbox))
                   for f in fills)

    def inside(trace: TextTrace, bbox: list[float]) -> bool:
        cx, cy = (trace.bbox[0] + trace.bbox[2]) / 2, (trace.bbox[1] + trace.bbox[3]) / 2
        return bbox[0] - tol <= cx <= bbox[2] + tol and bbox[1] - tol <= cy <= bbox[3] + tol

    for element_id, bbox, text in elements:
        typography = typographies.get(element_id)
        if typography is None or not typography.available:
            continue
        element_chars = _chars(text or "")
        total = sum(element_chars.values())
        own_fills = [f for f in fills if inside(f, bbox)]
        if not own_fills or not total:
            typography.fake_bold_status, typography.fake_bold_note = "unknown", "no fill trace inside the element box"
            continue
        fill_chars: collections.Counter = collections.Counter()
        for f in own_fills:
            fill_chars += _chars(f.text)
        if sum((fill_chars & element_chars).values()) / max(1, sum(fill_chars.values())) < cfg.min_char_match:
            typography.fake_bold_status, typography.fake_bold_note = (
                "unknown", "trace characters do not match the element text"
            )
            continue
        stroked: collections.Counter = collections.Counter()
        for s in strokes:
            if inside(s, bbox) and paired(s):
                stroked += _chars(s.text)
        matched = sum((stroked & element_chars).values())
        if stroked and matched / max(1, sum(stroked.values())) < cfg.min_char_match:
            typography.fake_bold_status, typography.fake_bold_note = (
                "unknown", "stroked characters do not match the element text"
            )
            continue
        share = matched / total
        typography.fake_bold_share = round(share, 3)
        typography.fake_bold_status = "fill_and_stroke" if share >= cfg.bold_share else "none"
        typography.fake_bold_note = f"{share:.0%} of characters drawn with fill and stroke"
