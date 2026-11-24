"""PROCESSED method: PyMuPDF + Arabic logical-order reconstruction (not raw library output).

Ported from the frozen prototype `reference/ar_pdf_extract.py`. With the default
`ReorderConfig` the reconstruction (`reconstruct_page`) reproduces the prototype's
`page_text()` exactly (see tests/lab/test_lpe_arabic_reorder.py).

Every rule here is a document-dependent HEURISTIC, spot-checked on one
Word-exported PDF only. It is therefore applied as a conditional, per-page
repair (`process_page`), never as a blanket transformation:

  not_applicable  no Arabic/Hebrew text: the PyMuPDF text is returned unchanged
  not_needed      reconstruction equals the raw text after whitespace-only
                  normalization, OR the raw text shows no repair signal at all
                  (it may already be in correct logical order): raw text kept
  uncertain       garbage/unmapped glyphs, a failed character-conservation check,
                  or the reconstruction introduced artifacts the raw text did not
                  have: the reconstruction is saved with a warning
  reconstructed   the repair changed the text AND passed every check; still NOT
                  verified against ground truth
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field

from bidi.algorithm import get_display

from lab.local_pdf_extraction import diagnostics

REORDER_VERSION = "2"
METHOD_LABEL = "Processed — PyMuPDF + Arabic logical-order reconstruction (not raw library output)"

ALEFS = frozenset("اأإآ")
# The PDF already stores logical bracket code points; stop bidi from mirroring them again.
PROTECT = str.maketrans({"(": "¦", ")": "§", "[": "¶", "]": "†"})
RESTORE = str.maketrans({"¦": "(", "§": ")", "¶": "[", "†": "]"})
PLACEHOLDERS = frozenset("¦§¶†")
BIDI_CONTROLS = frozenset("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")
# Paired punctuation that bidi may mirror; compared as one class in the conservation check.
MIRROR_CLASS = str.maketrans({")": "(", "]": "[", "}": "{", ">": "<", "»": "«", "›": "‹"})


@dataclass(frozen=True)
class ReorderConfig:
    """Thresholds (prototype values as defaults). Recorded in every run manifest.

    Line tolerance and space gap are relative to the glyph height. The prototype's
    table-cell gap is a fixed 18 pt; set `cell_gap_ratio` to make it relative to
    the glyph height instead (then `cell_gap_pt` is ignored).
    """

    line_tolerance_ratio: float = 0.45
    space_gap_ratio: float = 0.12
    cell_gap_pt: float = 18.0
    cell_gap_ratio: float | None = None
    zero_width_alef_pt: float = 0.2
    min_glyph_height: float = 1.0
    segment_separator: str = " | "
    # v2 fixes (all off in `prototype()`, which reproduces reference/ar_pdf_extract.py exactly):
    merge_zero_width_latin: bool = True       # "fi" ligature: zero-width "i" stays after "f"
    attach_orphan_marks: bool = True          # tanween in its own span joins the base it covers
    boundary_spaces_from_stream: bool = True  # Latin/Arabic boundary with no geometric gap
    keep_punctuation_with_latin: bool = True  # "Overfitting." keeps its period
    # Pages whose Arabic letters are below this share of all letters (raw PyMuPDF text)
    # are mainly LTR: the raw text is kept unchanged (not_needed). 0 disables the gate.
    min_arabic_letter_share: float = 0.30

    @classmethod
    def prototype(cls) -> "ReorderConfig":
        return cls(merge_zero_width_latin=False, attach_orphan_marks=False, boundary_spaces_from_stream=False,
                   keep_punctuation_with_latin=False, min_arabic_letter_share=0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict | None) -> "ReorderConfig":
        known = {key: value for key, value in (values or {}).items() if key in cls.__dataclass_fields__}
        if values and "min_arabic_letter_share" not in values:
            known["min_arabic_letter_share"] = 0.0  # runs created before the gate keep their behavior
        return cls(**known)


@dataclass
class Cluster:
    x0: float
    x1: float
    yc: float
    height: float
    text: str
    lam_alef: bool = False
    space_before: bool = False  # content stream had whitespace right before this cluster
    space_after: bool = False   # ... right after it
    follows_latin: bool = False  # content stream: directly after a Latin letter/digit (no space)


@dataclass
class Reconstruction:
    text: str
    separators_inserted: int = 0
    lam_alef_fixes: int = 0
    bracket_line_swaps: int = 0
    zero_width_latin_merged: int = 0
    orphan_marks_attached: int = 0
    boundary_spaces_inserted: int = 0
    latin_punctuation_kept: int = 0
    rtl_lines: int = 0
    ltr_lines: int = 0


@dataclass
class ReorderResult:
    text: str
    reorder_status: str
    decision: str
    warnings: list[str] = field(default_factory=list)
    signals_raw: dict = field(default_factory=dict)
    signals_reconstruction: dict = field(default_factory=dict)
    conservation: dict | None = None
    reconstruction_changed_text: bool = False
    candidate_text: str | None = None
    stats: dict = field(default_factory=dict)

    def to_record(self, config: ReorderConfig) -> dict:
        record = asdict(self)
        record.pop("text")
        record["config"] = config.to_dict()
        record["version"] = REORDER_VERSION
        record["verified_against_ground_truth"] = False
        return record


def is_rtl(ch: str) -> bool:
    return unicodedata.bidirectional(ch) in ("AL", "R")


def contains_rtl(text: str) -> bool:
    return any(is_rtl(ch) for ch in text)


# ------------------------------------------------------------ 1. characters
def page_spans(page) -> list[list[dict]]:
    """Characters with bounding boxes, one list per span (`page.get_text("rawdict")`)."""
    spans: list[list[dict]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                spans.append(span["chars"])
    return spans


# ------------------------------------------------------- 2-3. clusters
def _is_latin_or_digit_cluster(text: str) -> bool:
    return bool(text) and not contains_rtl(text) and not text.isspace()


def clusters_from_spans(spans: list[list[dict]], cfg: ReorderConfig,
                        stats: Reconstruction | None = None) -> list[Cluster]:
    out: list[Cluster] = []
    for chars in spans:
        i = 0
        span_start = len(out)
        while i < len(chars):
            start = i
            char = chars[i]
            t = char["c"]
            x0, y0, x1, y1 = char["bbox"]
            lam_alef = False
            # Lam-alef ligature (Word exports): zero-width alef followed by lam -> logical "لا".
            if t in ALEFS and abs(x1 - x0) < cfg.zero_width_alef_pt and i + 1 < len(chars) and chars[i + 1]["c"] == "ل":
                lx0, _, lx1, _ = chars[i + 1]["bbox"]
                logical = "ل" + t
                x0, x1 = min(x0, lx0), max(x1, lx1)
                lam_alef = True
                i += 2
            elif t.isspace():
                logical = " "  # kept for LTR lines; RTL lines rebuild spaces from geometry
                i += 1
            else:
                logical = t
                i += 1
            # Attach following combining marks (harakat / tanween) to the base character.
            while i < len(chars) and unicodedata.category(chars[i]["c"]) == "Mn":
                logical += chars[i]["c"]
                i += 1
            before = start > 0 and chars[start - 1]["c"].isspace()
            after = i < len(chars) and chars[i]["c"].isspace()
            previous = out[-1] if len(out) > span_start else None
            if (cfg.merge_zero_width_latin and previous is not None and not lam_alef
                    and _is_latin_or_digit_cluster(logical) and abs(x1 - x0) < cfg.zero_width_alef_pt
                    and _is_latin_or_digit_cluster(previous.text) and not previous.space_after):
                # Latin ligature glyph ("fi", "fl", "ffi"): the continuation character has zero
                # width; keep it right after its predecessor instead of sorting it by x.
                previous.text += logical
                previous.space_after = after
                if stats is not None:
                    stats.zero_width_latin_merged += 1
                continue
            # Only after a Latin LETTER: "49.58" must stay one number for bidi.
            follows = start > 0 and chars[start - 1]["c"].isascii() and chars[start - 1]["c"].isalpha()
            out.append(Cluster(x0, x1, (y0 + y1) / 2, y1 - y0, logical, lam_alef, before, after, follows))
    return out


def _only_marks(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch) == "Mn" for ch in text)


def attach_orphan_marks(line: list[Cluster], cfg: ReorderConfig) -> tuple[list[Cluster], int]:
    """A cluster made only of combining marks (e.g. tanween stored in its own span, after a
    lam-alef ligature) is attached to the RTL base cluster on the same line that covers it."""
    if not cfg.attach_orphan_marks:
        return line, 0
    bases = [c for c in line if contains_rtl(c.text) and not _only_marks(c.text)]
    kept, attached = [], 0
    for cluster in line:
        if _only_marks(cluster.text):
            x = (cluster.x0 + cluster.x1) / 2
            covering = [b for b in bases if b.x0 - 0.5 <= x <= b.x1 + 0.5]
            if covering:
                base = min(covering, key=lambda b: abs((b.x0 + b.x1) / 2 - x))
                base.text += cluster.text
                attached += 1
                continue
        kept.append(cluster)
    return kept, attached


# ------------------------------------------------------------- 4. lines
def group_lines(clusters: list[Cluster], cfg: ReorderConfig) -> list[list[Cluster]]:
    ordered = sorted(clusters, key=lambda c: c.yc)
    lines: list[list[Cluster]] = []
    for cluster in ordered:
        if lines and abs(lines[-1][-1].yc - cluster.yc) < cfg.line_tolerance_ratio * max(cluster.height, cfg.min_glyph_height):
            lines[-1].append(cluster)
        else:
            lines.append([cluster])
    return lines


# ---------------------------------------------------------- 5. segments
def split_segments(line: list[Cluster], cfg: ReorderConfig) -> list[list[Cluster]]:
    segments = [[line[0]]]
    for cluster in line[1:]:
        gap = (
            cfg.cell_gap_ratio * max(cluster.height, cfg.min_glyph_height)
            if cfg.cell_gap_ratio is not None
            else cfg.cell_gap_pt
        )
        if cluster.x0 - segments[-1][-1].x1 > gap:
            segments.append([cluster])  # table cells / big gaps
        else:
            segments[-1].append(cluster)
    return segments


# --------------------------------------------------- 7-8. visual -> logical
def visual_cluster_text(text: str) -> str:
    """Arabic clusters must appear reversed in a visual string so bidi flips them back."""
    if not contains_rtl(text):
        return text
    parts = [text[0]] + ([text[1:]] if len(text) > 1 and unicodedata.category(text[1]) == "Mn" else list(text[1:]))
    return "".join(reversed(parts))


def _ltr_run(segment: list[Cluster], index: int, step: int) -> list[Cluster]:
    run = []
    while 0 <= index < len(segment) and not contains_rtl(segment[index].text):
        run.append(segment[index])
        index += step
    return run if any(ch.isalnum() for c in run for ch in c.text) else []


def _boundary_space(segment: list[Cluster], index: int) -> bool:
    """Latin/Arabic boundary without a geometric gap. Word may draw the logical space on the
    other side of the Latin run, so use the content stream's own space instead."""
    previous, current = segment[index - 1], segment[index]
    if contains_rtl(previous.text) and not contains_rtl(current.text):
        run = _ltr_run(segment, index, 1)  # LTR run starts here: logically the space follows its end
        return bool(run) and run[-1].space_after
    if not contains_rtl(previous.text) and contains_rtl(current.text):
        run = _ltr_run(segment, index - 1, -1)  # LTR run ends here: logically the space precedes its start
        return bool(run) and run[-1].space_before
    return False


def visual_string(segment: list[Cluster], cfg: ReorderConfig, stats: Reconstruction | None = None) -> str:
    pieces: list[str] = []
    previous: Cluster | None = None
    for index, cluster in enumerate(segment):
        if previous is not None:
            if cluster.x0 - previous.x1 > cfg.space_gap_ratio * max(cluster.height, cfg.min_glyph_height):
                pieces.append(" ")
            elif cfg.boundary_spaces_from_stream and _boundary_space(segment, index):
                pieces.append(" ")
                if stats is not None:
                    stats.boundary_spaces_inserted += 1
        pieces.append(visual_cluster_text(cluster.text))
        if (cfg.keep_punctuation_with_latin and previous is not None and cluster.follows_latin
                and _is_neutral_punctuation(cluster.text) and _is_latin_or_digit_cluster(previous.text)
                and previous.text[-1].isascii() and previous.text[-1].isalpha()
                and cluster.x0 - previous.x1 <= cfg.space_gap_ratio * max(cluster.height, cfg.min_glyph_height)):
            # Drawn right after the Latin word and stored right after it: keep it inside the
            # LTR run (an invisible LRM in the visual string only; removed after bidi).
            pieces.append(LRM)
            if stats is not None:
                stats.latin_punctuation_kept += 1
        previous = cluster
    return "".join(pieces)


def _is_neutral_punctuation(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch)[0] == "P" and ch not in "()[]" for ch in text)


LRM = "\u200e"


def visual_to_logical(visual: str) -> str:
    logical = get_display(visual.translate(PROTECT), base_dir="R").translate(RESTORE)
    return logical.replace(LRM, "") if LRM in visual else logical


def fix_premirrored_brackets(line: str) -> tuple[str, bool]:
    """Safety net for producers that store visual (pre-mirrored) brackets: ")..(" -> "(..)"."""
    opening, closing = line.find("("), line.find(")")
    if closing != -1 and (opening == -1 or closing < opening) and line.count("(") == line.count(")"):
        return line.translate(str.maketrans("()", ")(")), True
    return line, False


# ------------------------------------------------------------ 6-9. lines
def reconstruct_line(line: list[Cluster], cfg: ReorderConfig,
                     stats: Reconstruction | None = None) -> tuple[str, int, bool, bool]:
    """Return (text, separators_inserted, is_rtl_line, brackets_swapped)."""
    line = sorted(line, key=lambda c: c.x0)  # visual order, left -> right
    if not any(contains_rtl(c.text) for c in line):
        # 6. LTR / neutral line: keep original spaces and natural order, no bidi.
        return re.sub(r" +", " ", "".join(c.text for c in line)).strip(), 0, False, False
    line = [c for c in line if c.text != " "]  # 7. stored spaces are unreliable on RTL lines
    if not line:
        return "", 0, True, False
    parts = [visual_to_logical(visual_string(segment, cfg, stats)) for segment in split_segments(line, cfg)]
    # 9. segments of an RTL line are joined right-to-left.
    text = cfg.segment_separator.join(reversed(parts)).strip()
    text = unicodedata.normalize("NFKC", text)  # presentation forms -> standard letters
    text, swapped = fix_premirrored_brackets(text)
    return text, len(parts) - 1, True, swapped


def reconstruct_clusters(clusters: list[Cluster], cfg: ReorderConfig) -> Reconstruction:
    result = Reconstruction(text="", lam_alef_fixes=sum(1 for c in clusters if c.lam_alef))
    lines: list[str] = []
    for line in group_lines(clusters, cfg):
        line, attached = attach_orphan_marks(line, cfg)
        result.orphan_marks_attached += attached
        text, separators, rtl, swapped = reconstruct_line(line, cfg, result)
        if rtl:
            result.rtl_lines += 1
        else:
            result.ltr_lines += 1
        result.bracket_line_swaps += int(swapped)
        if text:
            result.separators_inserted += separators
            lines.append(text)
    result.text = "\n".join(lines)
    return result


def reconstruct_page(page, cfg: ReorderConfig | None = None) -> Reconstruction:
    cfg = cfg or ReorderConfig()
    stats = Reconstruction(text="")
    result = reconstruct_clusters(clusters_from_spans(page_spans(page), cfg, stats), cfg)
    result.zero_width_latin_merged = stats.zero_width_latin_merged
    return result


# ------------------------------------------------------- gating / checks
def whitespace_normalized(text: str) -> str:
    """Whitespace-only normalization: every whitespace run (incl. line breaks) -> one space."""
    return " ".join(text.split())


def _canonical_chars(text: str) -> Counter:
    text = unicodedata.normalize("NFKC", text).translate(MIRROR_CLASS)
    return Counter(ch for ch in text if not ch.isspace() and ch not in BIDI_CONTROLS)


_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_DIGIT_RUN = re.compile(r"[.,]?[0-9]+(?:[.,][0-9]+)*")  # ".08" + "1" -> "1.08"


def latin_digit_runs_check(raw_text: str, rebuilt_text: str) -> dict:
    """Every Latin word / ASCII number run of the reconstruction must also be a run of the raw
    text (order changes inside a run, e.g. "Overfitting" -> "Overfti ting", are not caught by the
    character-conservation check). A number may be the exact concatenation of consecutive raw
    runs, because PyMuPDF splits numbers across lines ("7" + "1.3" -> "71.3")."""
    raw = unicodedata.normalize("NFKC", raw_text)
    rebuilt = unicodedata.normalize("NFKC", rebuilt_text)
    raw_words, rebuilt_words = Counter(_LATIN_WORD.findall(raw)), Counter(_LATIN_WORD.findall(rebuilt))
    missing = sorted((raw_words - rebuilt_words).elements())
    unexpected = sorted((rebuilt_words - raw_words).elements())
    raw_numbers = _DIGIT_RUN.findall(raw)
    available = [True] * len(raw_numbers)
    leftover = []
    for number in _DIGIT_RUN.findall(rebuilt):  # pass 1: exact runs
        index = next((i for i, value in enumerate(raw_numbers) if available[i] and value == number), None)
        if index is None:
            leftover.append(number)
        else:
            available[index] = False
    for number in leftover:  # pass 2: 2-3 consecutive raw fragments joined (in either order)
        match = None
        for start in range(len(raw_numbers)):
            for size in (2, 3):
                window = range(start, start + size)
                if window.stop > len(raw_numbers) or not all(available[i] for i in window):
                    continue
                parts = [raw_numbers[i] for i in window]
                if number in ("".join(parts), "".join(reversed(parts))):
                    match = window
                    break
            if match:
                break
        if match:
            for i in match:
                available[i] = False
        else:
            unexpected.append(number)
    missing += [value for value, free in zip(raw_numbers, available) if free]
    return {"ok": not missing and not unexpected, "missing": missing, "unexpected": unexpected}


def conservation_check(raw_text: str, reconstruction: Reconstruction, cfg: ReorderConfig) -> dict:
    """The reconstruction must be a re-ordering of the raw characters.

    Allowed changes (applied to both sides before comparing character multisets):
    whitespace; lam-alef pair swaps (order-only); bracket placeholder round-trip and
    bidi mirroring of paired punctuation; NFKC of presentation forms; invisible
    bidi control characters; and the segment separators this module inserts.
    """
    raw = _canonical_chars(raw_text)
    rebuilt = _canonical_chars(reconstruction.text)
    separator_chars = _canonical_chars(cfg.segment_separator)
    for ch, count in separator_chars.items():
        rebuilt[ch] -= count * reconstruction.separators_inserted
    added = {ch: n for ch, n in (rebuilt - raw).items() if n > 0}
    lost = {ch: n for ch, n in (raw - rebuilt).items() if n > 0}
    negative = {ch: -n for ch, n in rebuilt.items() if n < 0}
    placeholders = sorted({ch for ch in raw_text if ch in PLACEHOLDERS})

    def show(counter: dict) -> dict:
        return {f"U+{ord(ch):04X} {ch}": n for ch, n in sorted(counter.items())[:20]}

    ok = not added and not lost and not negative and not placeholders
    return {
        "ok": ok,
        "added": show(added),
        "lost": show({**lost, **negative}),
        "placeholder_chars_in_raw": placeholders,
        "raw_chars_compared": sum(raw.values()),
    }


def process_page(page, raw_text: str, cfg: ReorderConfig | None = None) -> ReorderResult:
    """Conditional per-page repair. `raw_text` is PyMuPDF's plain `page.get_text("text")`."""
    cfg = cfg or ReorderConfig()
    if not contains_rtl(raw_text):
        return ReorderResult(raw_text, "not_applicable", "no_rtl_text")
    share = diagnostics.arabic_letter_share(raw_text)
    if share < cfg.min_arabic_letter_share:
        return ReorderResult(
            raw_text, "not_needed", "mainly_ltr",
            [f"page is mainly LTR; Arabic share {share:.1%}"],
            stats={"arabic_letter_share": share, "min_arabic_letter_share": cfg.min_arabic_letter_share},
        )
    reconstruction = reconstruct_page(page, cfg)
    signals_raw = diagnostics.repair_signals(raw_text)
    signals_rec = diagnostics.repair_signals(reconstruction.text)
    stats = {
        "lam_alef_fixes": reconstruction.lam_alef_fixes,
        "separators_inserted": reconstruction.separators_inserted,
        "bracket_line_swaps": reconstruction.bracket_line_swaps,
        "zero_width_latin_merged": reconstruction.zero_width_latin_merged,
        "orphan_marks_attached": reconstruction.orphan_marks_attached,
        "boundary_spaces_inserted": reconstruction.boundary_spaces_inserted,
        "latin_punctuation_kept": reconstruction.latin_punctuation_kept,
        "rtl_lines": reconstruction.rtl_lines,
        "ltr_lines": reconstruction.ltr_lines,
        "arabic_letter_share": share,
    }
    changed = whitespace_normalized(reconstruction.text) != whitespace_normalized(raw_text)
    base = {"signals_raw": signals_raw, "signals_reconstruction": signals_rec, "stats": stats,
            "reconstruction_changed_text": changed}

    garbage = sorted({f"U+{ord(ch):04X}" for ch in raw_text if diagnostics.is_garbage_char(ch)})
    if garbage:
        return ReorderResult(
            reconstruction.text, "uncertain", "garbage_or_unmapped_glyphs",
            [f"Unmapped/garbage glyphs in the PDF text layer ({', '.join(garbage[:12])}). "
             "This text cannot be recovered without OCR; the reconstruction is unreliable."],
            conservation=conservation_check(raw_text, reconstruction, cfg),
            candidate_text=reconstruction.text, **base,
        )
    if not changed:
        return ReorderResult(raw_text, "not_needed", "equal_after_whitespace_normalization", **base)
    if not any(signals_raw[name] for name in diagnostics.STRONG_SIGNALS):
        return ReorderResult(
            raw_text, "not_needed", "no_repair_signals",
            ["The reconstruction differs, but the raw text shows no repair signal; it may already be "
             "in correct logical order, so the raw PyMuPDF text was kept (candidate saved in JSON)."],
            candidate_text=reconstruction.text, **base,
        )
    conservation = conservation_check(raw_text, reconstruction, cfg)
    introduced = [name for name in diagnostics.REGRESSION_SIGNALS if signals_rec[name] > signals_raw[name]]
    if introduced:
        return ReorderResult(
            reconstruction.text, "uncertain", "reconstruction_introduced_artifacts",
            [f"The reconstruction has more {', '.join(introduced)} than the raw text; it may have damaged "
             "text that was already correct. Compare with the raw PyMuPDF output."],
            conservation=conservation, candidate_text=reconstruction.text, **base,
        )
    if not conservation["ok"]:
        return ReorderResult(
            reconstruction.text, "uncertain", "conservation_check_failed",
            ["Character-conservation check failed: the reconstruction added, lost or substituted characters."],
            conservation=conservation, candidate_text=reconstruction.text, **base,
        )
    runs = latin_digit_runs_check(raw_text, reconstruction.text)
    base["stats"]["latin_digit_runs"] = runs
    if not runs["ok"]:
        return ReorderResult(
            reconstruction.text, "uncertain", "latin_or_digit_run_changed",
            ["A Latin word or number in the reconstruction differs from the raw text (order inside the run "
             f"changed): {', '.join((runs['missing'] + runs['unexpected'])[:8])}. Compare with raw PyMuPDF."],
            conservation=conservation, candidate_text=reconstruction.text, **base,
        )
    return ReorderResult(
        reconstruction.text, "reconstructed", "repaired_and_conserved",
        ["Reconstructed, not verified against ground truth."],
        conservation=conservation, **base,
    )
