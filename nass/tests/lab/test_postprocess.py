from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lab.core.adapters.postprocess import (
    VARIANT_VERSION,
    available_variants,
    correct_text,
    derive_variant,
    detect_sensitive_tokens,
)
from lab.core.models import AdapterInfo, PageResult, RunResult


@pytest.mark.parametrize(
    ("adapter_id", "text"),
    [
        ("text.pypdfium2", "القيمة ١٢٣٬٤٥٦٫٧٨ ر.س والنسبة ٢٥٪"),
        ("text.pdfplumber", "ID-00125 dated 2025-11-06 costs $52,214.30"),
        ("text.pypdfium2", "نسخة v2.10 ومعرّف ABC-00125"),
    ],
)
def test_digit_fix_does_not_change_correct_values_for_preserving_extractors(
    adapter_id: str, text: str
) -> None:
    assert correct_text(text, adapter_id, "digit_run_fix").text == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("الهامش ٤٨٫٣٣٪", "الهامش ٣٣٫٨٤٪"),
        ("المبلغ ٥٤٫٠٣٧٬٤١٢٬٢٥", "المبلغ ٥٢٬٢١٤٬٧٣٠٫٤٥"),
        ("التاريخ ٦٢٠٢/٣٠/٥١", "التاريخ ١٥/٠٣/٢٠٢٦"),
        ("التاريخ ٥١/٣٠/٦٢٠٢", "التاريخ ١٥/٠٣/٢٠٢٦"),
        ("الحد ٥٫٠‰", "الحد ٠٫٥‰"),
        ("النسبة ٪٤٨٫٣٣", "النسبة ٣٣٫٨٤٪"),
        ("النمو ٪٧٤٫٢١+", "النمو +١٢٫٤٧٪"),
        ("المبلغ ﷼ ٠٠٫٠٥٧٬٤", "المبلغ ٤٬٧٥٠٫٠٠ ﷼"),
    ],
)
def test_measured_pymupdf_raw_repairs_complete_numeric_tokens(
    raw: str, expected: str
) -> None:
    assert correct_text(raw, "text.pymupdf", "digit_run_fix").text == expected


def test_ascii_digits_and_mixed_english_are_not_reversed() -> None:
    raw = "النص العربي with English ID ABC-00125 and 2025-11-06"
    assert correct_text(raw, "text.pymupdf", "digit_run_fix").text == raw


def test_ambiguous_identifier_digits_are_not_guessed() -> None:
    raw = "المعرف INV-٥٢١٠٠"
    assert correct_text(raw, "text.pymupdf", "digit_run_fix").text == raw


@pytest.mark.parametrize("digits", ["00125", "٠٠١٢٥", "۰۰۱۲۵"])
def test_digit_script_and_leading_zeros_are_preserved(digits: str) -> None:
    raw = f"English ID X-{digits}"
    assert correct_text(raw, "text.pypdfium2", "digit_run_fix").text == raw


def test_arabic_fold_is_scoped_and_idempotent() -> None:
    raw = "ﻣﺮﺣﺒﺎ ١٢٣ English"
    once = correct_text(raw, "text.pdfplumber", "arabic_fold")
    twice = correct_text(once.text, "text.pdfplumber", "arabic_fold")
    assert once.text == "مرحبا ١٢٣ English"
    assert twice.text == once.text
    assert len(twice.changes) == 0


def test_arabic_fold_preserves_saudi_riyal_symbol_exactly() -> None:
    raw = "﷼ ١٢٣٫٤٥"
    corrected = correct_text(raw, "text.pdfplumber", "arabic_fold")
    assert corrected.text == raw
    # NFKC would produce the word with Persian yeh (U+06CC), observed in review.
    assert "ریال" not in corrected.text
    assert "ريال" not in corrected.text


# Page context captured from pypdfium2 RAW of the visually mirrored benchmark
# (synthetic content): reversed bracket pairs outnumber normal ones.
MIRRORED_PAGE_CONTEXT = "سراب لأنظمة البيانات )جهة خيالية(\nمعدل الخطأ )٪١٫٣٧(\n"


def test_mirror_fix_requires_captured_local_signature() -> None:
    raw = MIRRORED_PAGE_CONTEXT + "حد السجلات ≥ ‰٠٫٥ ت\nالحد المطلوب CER ≤ 0.05\nx < y"
    corrected = correct_text(raw, "text.pypdfium2", "bidi_mirror_fix")
    assert "حد السجلات ≤ ‰٠٫٥ ت" in corrected.text
    assert corrected.text.endswith("\nالحد المطلوب CER ≤ 0.05\nx < y")


def test_signature_without_page_mirroring_evidence_is_left_unchanged() -> None:
    # Logical text may prefix the percent sign; locally identical to the
    # visual-order signature, so it must not be flipped (review note N2).
    raw = "يجب أن تكون النسبة ≥ ٪٥٠ دائمًا"
    corrected = correct_text(raw, "text.pypdfium2", "bidi_mirror_fix")
    assert corrected.text == raw
    assert any("no visual-mirroring evidence" in warning for warning in corrected.warnings)


def test_logical_lettered_items_are_not_mirroring_evidence() -> None:
    # `) و (` between normal pairs must not count as a reversed pair.
    raw = "البنود (أ) و (ب) و (ج) مطلوبة\nيجب أن تكون النسبة ≥ ٪٥٠ دائمًا"
    corrected = correct_text(raw, "text.pypdf", "bidi_mirror_fix")
    assert corrected.text == raw


def test_ambiguous_arabic_operator_is_preserved_and_flagged() -> None:
    raw = "المعدل المطلوب ≤ ٠٫٥"
    corrected = correct_text(raw, "text.pypdfium2", "bidi_mirror_fix")
    assert corrected.text == raw
    assert any("Ambiguous" in warning for warning in corrected.warnings)


def test_locally_mirrored_numeric_brackets_are_repaired() -> None:
    raw = "سراب )جهة خيالية(\nمعدل الخطأ )٪١٫٣٧("
    corrected = correct_text(raw, "text.pypdfium2", "bidi_mirror_fix")
    assert corrected.text == "سراب )جهة خيالية(\nمعدل الخطأ (٪١٫٣٧)"


def test_isolated_reversed_numeric_bracket_without_page_evidence_is_flagged() -> None:
    raw = "معدل الخطأ )٪١٫٣٧( مقابل (٢٫٠٪) و (٣٫٥٪)"
    corrected = correct_text(raw, "text.pypdfium2", "bidi_mirror_fix")
    assert corrected.text == raw
    assert corrected.warnings


def test_digit_run_fix_always_warns_about_producer_dependence() -> None:
    corrected = correct_text("الهامش ٤٨٫٣٣٪", "text.pymupdf", "digit_run_fix")
    assert corrected.text == "الهامش ٣٣٫٨٤٪"
    assert any("producers" in warning for warning in corrected.warnings)
    # The documented hazard: logical input is corrupted, so it stays experimental.
    logical = correct_text("الهامش ٣٣٫٨٤٪", "text.pymupdf", "digit_run_fix")
    assert logical.text == "الهامش ٤٨٫٣٣٪"
    assert any("corrupts correct values" in warning for warning in logical.warnings)


def test_unmapped_marker_is_explicit_and_warns() -> None:
    corrected = correct_text("نص\x00 text�", "text.pdfplumber", "unmapped_marker")
    assert corrected.text == "نص⟦?⟧ text⟦?⟧"
    assert corrected.warnings


def test_detects_numbers_dates_currency_percentages_and_identifiers() -> None:
    tokens = detect_sensitive_tokens(
        "INV-00125 2025-11-06 $52,214.30 النسبة ٢٥٪ والمعرّف س-۰۰۱۲۵"
    )
    values = {token.value for token in tokens}
    assert {"INV-00125", "2025-11-06", "$52,214.30", "٢٥٪", "س-۰۰۱۲۵"} <= values


def _raw_run() -> RunResult:
    adapter = AdapterInfo(
        adapter_id="text.pymupdf",
        task="text",
        label="PyMuPDF",
        kind="local",
        library="pymupdf",
        library_version="1",
        license="AGPL-3.0",
        model_id=None,
        params={},
        available=True,
        unavailable_reason=None,
        supports_blocks=True,
    )
    return RunResult(
        run_id="run_raw",
        doc_id="doc_x",
        doc_sha256="a" * 64,
        adapter=adapter,
        variant="raw",
        pages=[
            PageResult(
                page=1,
                text="القيمة ٥٢٪",
                blocks=[],
                status="ok",
                error=None,
                elapsed_ms=1,
            )
        ],
        page_mapping="exact",
        started_at=datetime.now(timezone.utc),
        wall_ms=1,
        cpu_ms=1,
        usage=None,
        status="ok",
        error=None,
        cache_hit=False,
        environment={},
    )


def test_derived_run_preserves_raw_byte_for_byte_and_tracks_lineage() -> None:
    raw = _raw_run()
    before = raw.model_dump_json()
    derived = derive_variant(raw, "digit_run_fix")
    assert raw.model_dump_json() == before
    assert raw.pages[0].text == "القيمة ٥٢٪"
    assert derived.pages[0].text == "القيمة ٢٥٪"
    assert derived.raw_run_id == raw.run_id
    assert derived.variant_version == VARIANT_VERSION == "2"
    assert derived.run_id != raw.run_id


def test_cannot_chain_corrections_from_corrected_run() -> None:
    derived = derive_variant(_raw_run(), "digit_run_fix")
    with pytest.raises(ValueError, match="RAW"):
        derive_variant(derived, "arabic_fold")


def test_unsafe_numeric_and_bidi_variants_are_hidden_by_default() -> None:
    default = available_variants("text.pymupdf")
    experimental = available_variants(
        "text.pymupdf", include_experimental=True
    )
    assert "digit_run_fix" not in default
    assert "bidi_reorder" not in default
    assert "digit_run_fix" in experimental
    assert "bidi_reorder" in experimental
