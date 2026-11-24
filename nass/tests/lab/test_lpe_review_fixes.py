"""Regression tests for the review of run_20261118_173555_a28c8d (one per reported issue).

Character fixtures reuse glyph coordinates measured on that Word-exported PDF (pages 3
and 18); only a few words and their bounding boxes are included, not the document.
"""

from __future__ import annotations

import pymupdf
import pytest

from lab.local_pdf_extraction import arabic_reorder as ar
from lab.local_pdf_extraction import diagnostics
from tests.lab import lpe_pdf_factory as factory


def _span(chars: list[tuple[str, float, float]], y0: float = 0.0, y1: float = 16.5) -> list[dict]:
    return [{"c": c, "bbox": (x0, y0, x1, y1)} for c, x0, x1 in chars]


def _rebuild(spans: list[list[dict]], cfg: ar.ReorderConfig | None = None) -> str:
    cfg = cfg or ar.ReorderConfig()
    return ar.reconstruct_clusters(ar.clusters_from_spans(spans, cfg), cfg).text


# ------------------------------------------- 1. numbered headings are not flipped brackets
@pytest.mark.parametrize("line, expected", [
    ("1) مؤشرات التحليل الزمني وخطورة الحوادث:", 0),
    ("٣) توزيع الحوادث الخطيرة حسب الفترة:", 0),
    ("أ) بند فرعي", 0),
    ("الفئة العمرية (+20 سنة) تقريبًا", 0),
    ("1) العوامل (مثل السرعة) المؤثرة", 0),
    ("الفئة العمرية )+20 سنة( تقريبًا", 1),
    ("1) الفئة )+20 سنة(", 1),
    ("Heading 1) English only )x(", 0),  # LTR lines are not checked
])
def test_flipped_brackets_only_when_close_precedes_its_open(line, expected) -> None:
    assert diagnostics.flipped_bracket_lines(line) == expected


def test_word_style_numbered_heading_is_not_marked_uncertain(monkeypatch) -> None:
    """Pages 5-8, 10: logical "1) ..." headings (Word stores logical brackets) were flagged."""
    raw = "1) ﻣﺆﺷﺮﺍﺕ ﺍﻟﺘﺤﻠﻴﻞ ﺍﻟﺴﺎﻟﻤﺔ\n3) ﺗﻮﺯﻳﻊ ﺍﻟﺤﻮﺍﺩﺙ\n"
    rebuilt = "1) مؤشرات التحليل السلامة\n3) توزيع الحوادث"
    monkeypatch.setattr(ar, "reconstruct_page", lambda page, cfg=None: ar.Reconstruction(text=rebuilt))
    result = ar.process_page(object(), raw)
    assert result.signals_reconstruction["flipped_brackets"] == 0
    assert result.reorder_status == "reconstructed", (result.decision, result.warnings)


@pytest.mark.parametrize("line, expected", [("1( مؤشرات التحليل:", 1), ("( مؤشرات التحليل:", 0), ("1(أ) بند", 0)])
def test_mirrored_numbered_heading_counts_as_flipped(line, expected) -> None:
    assert diagnostics.flipped_bracket_lines(line) == expected


@pytest.mark.skipif(not factory.ARIAL.exists(), reason="needs C:/Windows/Fonts/arial.ttf")
def test_premirrored_producer_heading_is_uncertain_not_silently_wrong(tmp_path) -> None:
    """insert_htmlbox stores pre-mirrored brackets; its reconstruction "1( ..." must not pass."""
    path = tmp_path / "heading.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    css = f"@font-face {{font-family: F; src: url({factory.ARIAL.name});}} * {{font-family: F; font-size: 14px;}}"
    page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), "<p dir='rtl'>1) مؤشرات التحليل الزمني:</p>",
                        css=css, archive=pymupdf.Archive(str(factory.FONT_DIR)))
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as document:
        page = document[0]
        result = ar.process_page(page, page.get_text("text"))
    assert result.text.startswith("1(")
    assert result.reorder_status == "uncertain"
    assert result.decision == "reconstruction_introduced_artifacts"


# ------------------------------------- 2. tanween on a lam-alef ligature stays attached
TANWEEN_SPANS = [
    _span([(" ", 237.38, 240.43), ("ت", 234.02, 237.39), ("م", 227.54, 234.0), ("ث", 223.46, 227.49),
           ("ي", 219.38, 223.41), ("ا", 219.42, 219.42), ("ل", 213.26, 219.42)], 72.03, 88.53),
    _span([("\u064b", 215.78, 215.78)], 70.83, 87.33),  # tanween stored in its own span
    _span([(" ", 213.21, 215.78), (".", 211.22, 213.21), (" ", 208.22, 211.27), ("و", 203.18, 208.24),
           ("ب", 200.06, 203.19), ("ن", 196.22, 200.0)], 70.83, 88.53),
]


def test_tanween_after_lamalef_stays_attached() -> None:
    text = _rebuild(TANWEEN_SPANS)
    assert text.startswith("تمثيلاً")
    assert "ً" not in text.replace("تمثيلاً", "")  # no detached mark anywhere else
    assert "تمثي ًلا" in _rebuild(TANWEEN_SPANS, ar.ReorderConfig.prototype())  # the reported defect


# ------------------------------------- 3. Latin ligature "fi" inside an RTL line
OVERFITTING_SPAN = _span([
    ("ا", 537.82, 540.22), ("ل", 534.7, 537.87), ("ـ", 533.74, 534.7), (" ", 473.26, 476.31),
    ("O", 476.26, 484.55), ("v", 484.55, 489.82), ("e", 489.7, 495.15), ("r", 495.1, 499.12),
    ("f", 499.17, 505.58), ("i", 505.58, 505.58), ("t", 505.53, 509.75), ("t", 509.75, 513.98),
    ("i", 513.93, 516.45), ("n", 516.45, 522.58), ("g", 522.58, 528.34), (".", 528.68, 530.67),
    (" ", 530.7, 533.75), ("ك", 468.24, 473.19), ("م", 461.78, 468.24), ("ا", 458.91, 461.73),
    (" ", 455.94, 458.99), ("ت", 452.57, 455.94), ("م", 445.95, 452.57),
])


def test_latin_ligature_keeps_word_order() -> None:
    assert _rebuild([OVERFITTING_SPAN]) == "الـ Overfitting. كما تم"
    assert "Overfti ting" in _rebuild([OVERFITTING_SPAN], ar.ReorderConfig.prototype())  # the reported defect


@pytest.mark.parametrize("raw, rebuilt, ok", [
    ("الـ Overfitting. كما", "الـ Overfitting. كما", True),
    ("الـ Overfitting. كما", "الـ .Overfti ting كما", False),
    ("بنسبة49.58\n%\n و7\n1.3 منها", "بنسبة 49.58% و 1.37% منها", True),   # split fragments joined
    ("و.08\n1 للحوادث", "و1.08% للحوادث", True),
    ("بنسبة 49.58%", "بنسبة 58.49%", False),                               # reversed number
    ("Data Quality", "Quality Data", True),                                # word order is not checked
])
def test_latin_digit_runs_check(raw, rebuilt, ok) -> None:
    assert ar.latin_digit_runs_check(raw, rebuilt)["ok"] is ok


def test_changed_latin_run_makes_page_uncertain(monkeypatch) -> None:
    raw = "ﺍﻟـ Overfitting. ﻛﻤﺎ ﺍﻟﺴﻼﻣﺔ 45.5\n%\n"
    monkeypatch.setattr(ar, "reconstruct_page",
                        lambda page, cfg=None: ar.Reconstruction(text="الـ .Overfti ting كما السلامة 45.5%"))
    result = ar.process_page(object(), raw)
    assert result.reorder_status == "uncertain"
    assert result.decision == "latin_or_digit_run_changed"
    assert result.conservation["ok"]  # the character-conservation check alone would have passed


# ------------------------------------- 4. space at a Latin/Arabic boundary
INJURY_SPANS = [
    _span([("ب", 260.33, 263.46), ("ـ", 259.37, 260.33), (" ", 256.49, 259.54)]),
    _span([("N", 208.73, 217.11), ("o", 217.11, 223.07), (" ", 223.24, 226.29), ("I", 226.24, 229.23),
           ("n", 229.23, 235.36), ("j", 235.36, 237.88), ("u", 237.88, 243.95), ("r", 243.99, 248.01),
           ("y", 248.05, 253.31), (" ", 253.49, 256.54)]),
    _span([("ل", 205.48, 208.64), ("ت", 201.41, 205.44), ("م", 195.05, 201.5), ("ي", 190.98, 195.01),
           ("ي", 186.91, 190.94), ("ز", 182.71, 186.98)]),
]


def test_space_kept_at_latin_arabic_boundary() -> None:
    assert _rebuild(INJURY_SPANS) == "بـ No Injury لتمييز"
    assert "Injuryلتمييز" in _rebuild(INJURY_SPANS, ar.ReorderConfig.prototype())  # the reported defect


def test_decimal_number_stays_intact_in_rtl_line() -> None:
    """Regression guard for the punctuation rule: "49.58" must not be split by bidi."""
    spans = [_span([("ب", 120, 125), ("ن", 115, 120), ("س", 109, 115), ("ب", 104, 109), ("ة", 99, 104),
                    ("4", 70, 75), ("9", 75, 80), (".", 80, 82), ("5", 82, 87), ("8", 87, 92)]),
             _span([("%", 60, 69)])]
    assert "49.58" in _rebuild(spans)
    assert "58.49" not in _rebuild(spans)


# ------------------------------------- 5. reversed-word indicator is aggregated and shown
def test_reversed_word_signs_are_aggregated_and_flagged() -> None:
    reversed_text = " ".join(word[::-1] for word in ["البيانات", "الحوادث", "المرورية", "السرعة", "الطريق", "الفترة"])
    diag = diagnostics.page_indicators(reversed_text)
    assert diag["reversed_word_signs"] == {"ending_la": 6, "starting_al": 0}
    records = [{"status": "COMPLETED", "diagnostics": diag}]
    totals = diagnostics.aggregate(records)
    assert (totals["reversed_ending_la"], totals["reversed_starting_al"]) == (6, 0)
    assert diagnostics.likely_reversed(569, 0)
    assert not diagnostics.likely_reversed(0, 28)
    assert not diagnostics.likely_reversed(3, 0)
