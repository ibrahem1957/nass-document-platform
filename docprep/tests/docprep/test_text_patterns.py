"""Text pattern signals (Arabic and English). Written, not executed in the implementation task."""

from __future__ import annotations

import pytest

from docprep import text_patterns as tp


@pytest.mark.parametrize("text,scheme,label,depth", [
    ("1. Introduction", "decimal", "1", 1),
    ("1) مؤشرات التحليل الزمني وخطورة الحوادث:", "decimal", "1", 1),
    ("١) مؤشرات التحليل:", "decimal", "1", 1),
    ("2.3 Data sources", "decimal_multi", "2.3", 2),
    ("1.2.4 Details", "decimal_multi", "1.2.4", 3),
    ("(4) Fourth item", "paren_number", "4", 1),
    ("(a) first option", "latin_letter", "a", 1),
    ("أ) الخيار الأول", "arabic_letter", "أ", 1),
    ("IV. Results", "roman", "IV", 1),
    ("أولاً: مقدمة", "ordinal", "أولا", 1),
    ("ثانياً - النتائج", "ordinal", "ثانيا", 1),
    ("Chapter 3 Methods", "keyword", "Chapter 3", 1),
    ("الفصل الثاني", "keyword", "الفصل الثاني", 1),
])
def test_numbering_schemes(text, scheme, label, depth):
    n = tp.numbering(text)
    assert n is not None and n.scheme == scheme and n.label == label and n.depth == depth


@pytest.mark.parametrize("text", ["2024 was a busy year.", "12,730", "I am here", "Introduction", "5 - 10 minutes"])
def test_no_false_numbering(text):
    assert tp.numbering(text) is None


def test_bullets():
    assert tp.bullet_marker("• وصف البيانات:") == "•"
    assert tp.bullet_marker("- مدة الفحص.") == "-"
    assert tp.bullet_marker("-5% change") is None
    assert tp.bullet_marker("Plain text") is None


@pytest.mark.parametrize("text,expected", [
    ("Figure 3: Monthly ridership", True), ("الشكل 1: الرحلات اليومية", True), ("جدول ٢ - الإيرادات", True),
    ("Table 4", True), ("Table 3 shows the results for every line.", False), ("Data description:", False),
])
def test_captions(text, expected):
    assert tp.caption_match(text)[0] is expected


@pytest.mark.parametrize("text,expected", [("12", True), ("Page 3", True), ("صفحة ٤", True), ("- 7 -", True),
                                           ("3 / 18", True), ("Section 3", False)])
def test_page_numbers(text, expected):
    assert tp.is_page_number(text) is expected


def test_numeric_and_axis_labels():
    assert tp.is_numeric_label("18,450") and tp.is_numeric_label("20K")
    assert not tp.is_numeric_label("Total riders")
    assert tp.is_axis_like("0 20 40 60 80 100") and tp.is_axis_like("16 12 8 4 0")
    assert not tp.is_axis_like("The 3 lines carried 40 thousand riders in 12 months")


def test_punctuation_and_colon():
    assert tp.ends_with_colon("معالجة وتنظيف البيانات:") and tp.ends_with_colon("Scope：")
    assert tp.ends_with_terminal("A sentence.") and tp.ends_with_terminal("هل هذا سؤال؟")
    assert not tp.ends_with_terminal("A heading without punctuation")


def test_offsets_preserved_by_digit_normalization():
    text = "الشكل ١: عنوان"
    assert len(tp.ascii_digits(text)) == len(text)


def test_script_detection():
    assert tp.script_of("مرحبا بالعالم") == "arabic"
    assert tp.script_of("Hello world") == "latin"
    assert tp.script_of("12 34") == "none"
