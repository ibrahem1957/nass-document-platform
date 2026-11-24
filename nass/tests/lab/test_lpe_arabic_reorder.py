"""Regression tests for the PROCESSED Arabic logical-order reconstruction.

Synthetic PDFs only (tests/lab/lpe_pdf_factory.py). LibreOffice is not installed on
this machine, so the soffice producer is skipped (see test_libreoffice_producer).
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pymupdf
import pytest

from lab.local_pdf_extraction import arabic_reorder as ar
from lab.local_pdf_extraction import diagnostics
from tests.lab import lpe_pdf_factory as factory

PROJECT_DIR = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not factory.ARIAL.exists(), reason="needs C:/Windows/Fonts/arial.ttf")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _reference():
    spec = importlib.util.spec_from_file_location("ar_pdf_extract_reference", PROJECT_DIR / "reference" / "ar_pdf_extract.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def producers(tmp_path_factory) -> dict[str, Path]:
    return factory.all_producers(tmp_path_factory.mktemp("lpe_producers"))


def _results(path: Path) -> dict[str, tuple[str, ar.ReorderResult]]:
    out = {}
    with pymupdf.open(path) as document:
        for name, page in zip(factory.PAGE_ORDER, document):
            raw = page.get_text("text")
            out[name] = (raw, ar.process_page(page, raw))
    return out


def test_reference_prototype_is_the_given_frozen_file() -> None:
    source = (PROJECT_DIR / "reference" / "ar_pdf_extract.py").read_text(encoding="utf-8")
    assert source.startswith('"""Arabic-aware text extraction for digital PDFs (PyMuPDF + bidi).')
    assert "def page_text(page, ytol=0.45, gap=18):" in source


def test_reconstruction_matches_reference_prototype_on_every_producer(producers) -> None:
    reference = _reference()
    compared = 0
    for path in producers.values():
        with pymupdf.open(path) as document:
            for page in document:
                assert ar.reconstruct_page(page, ar.ReorderConfig.prototype()).text == reference.page_text(page)
                compared += 1
    assert compared == len(producers) * len(factory.PAGE_ORDER)


def test_v2_default_differs_from_prototype_only_where_a_v2_rule_fired(producers) -> None:
    for path in producers.values():
        with pymupdf.open(path) as document:
            for page in document:
                default = ar.reconstruct_page(page)
                fired = (default.zero_width_latin_merged + default.orphan_marks_attached
                         + default.boundary_spaces_inserted + default.latin_punctuation_kept)
                if not fired:
                    assert default.text == ar.reconstruct_page(page, ar.ReorderConfig.prototype()).text
                assert "‎" not in default.text  # the LRM helper never leaks into output


def test_already_correct_arabic_is_left_unchanged(producers) -> None:
    results = _results(producers["tounicode_logical"])
    for name in ("correct", "lamalef"):
        raw, result = results[name]
        assert result.reorder_status == "not_needed"
        assert result.text == raw  # byte-identical to PyMuPDF raw
    assert "السلامة الأنماط لا إلا الآن" in results["lamalef"][0]


def test_prototype_would_damage_logical_lamalef_and_gating_prevents_it(producers) -> None:
    reference = _reference()
    with pymupdf.open(producers["tounicode_logical"]) as document:
        page = document[factory.PAGE_ORDER.index("lamalef")]
        assert "السالمة" in reference.page_text(page)  # documented heuristic failure
        assert "السالمة" not in ar.process_page(page, page.get_text("text")).text
    raw, result = _results(producers["tounicode_logical"])["lamalef_plus_fragment"]
    assert result.reorder_status == "uncertain"
    assert result.decision == "reconstruction_introduced_artifacts"
    assert result.warnings


def test_wordlike_lamalef_is_reconstructed(producers) -> None:
    raw, result = _results(producers["tounicode_wordlike"])["lamalef"]
    assert "السالمة" in raw and "األنماط" in raw
    assert result.reorder_status == "reconstructed"
    assert result.text == "السلامة الأنماط لا إلا الآن"
    assert result.conservation["ok"]


@pytest.mark.parametrize("producer", ["htmlbox_arial", "inserttext_visual"])
def test_mixed_numbers_brackets(producers, producer) -> None:
    results = _results(producers[producer])
    assert results["mixed"][1].text == "جودة البيانات Data Quality مهمة"
    assert results["numbers"][1].text.translate(DIGITS) == "بلغت النسبة 45.5% في عام 2025 و 2024"
    assert "(+20 سنة)" in results["brackets"][1].text
    for name in ("mixed", "numbers", "brackets"):
        assert results[name][1].reorder_status == "reconstructed"
        assert results[name][1].conservation["ok"]


def test_english_only_pages_are_not_applicable(producers) -> None:
    for path in producers.values():
        raw, result = _results(path)["english"]
        assert result.reorder_status == "not_applicable"
        assert result.text == raw


def test_unmapped_glyphs_are_uncertain_never_silently_fixed(producers) -> None:
    results = _results(producers["htmlbox_default"])
    for name in factory.CASES:
        raw, result = results[name]
        assert result.reorder_status == "uncertain"
        assert result.decision == "garbage_or_unmapped_glyphs"
        assert any("OCR" in warning for warning in result.warnings)
    indicators = diagnostics.page_indicators(results["correct"][0])
    assert indicators["garbage_arabic_tokens"] > 0


def test_table_percent_regression_is_flagged(producers) -> None:
    raw, result = _results(producers["htmlbox_arial"])["table"]
    assert "45.5%" in raw
    assert result.reorder_status == "uncertain"  # reconstruction turned 45.5% into %45.5
    assert result.decision == "reconstruction_introduced_artifacts"


def test_placeholder_characters_make_page_uncertain(tmp_path) -> None:
    path = tmp_path / "placeholder.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    css = f"@font-face {{font-family: F; src: url({factory.ARIAL.name});}} * {{font-family: F; font-size: 14px;}}"
    page.insert_htmlbox(pymupdf.Rect(50, 50, 545, 800), "<p dir='rtl'>السلامة ¦ النسبة 45.5% تقريبًا</p>",
                        css=css, archive=pymupdf.Archive(str(factory.FONT_DIR)))
    doc.save(path)
    doc.close()
    with pymupdf.open(path) as document:
        page = document[0]
        raw = page.get_text("text")
        assert "¦" in raw
        result = ar.process_page(page, raw)
    assert result.reorder_status == "uncertain"
    assert result.conservation["placeholder_chars_in_raw"] == ["¦"]


def test_small_functions() -> None:
    assert ar.visual_cluster_text("لا") == "ال"
    assert ar.visual_cluster_text("بَ") == "َب"
    assert ar.visual_cluster_text("A") == "A"
    assert ar.fix_premirrored_brackets(")سنة(") == ("(سنة)", True)
    assert ar.fix_premirrored_brackets("(سنة)") == ("(سنة)", False)
    cfg = ar.ReorderConfig()
    added = ar.conservation_check("ابت", ar.Reconstruction(text="ابتث"), cfg)
    assert not added["ok"] and added["added"]
    swapped = ar.conservation_check("السالمة", ar.Reconstruction(text="السلامة"), cfg)
    assert swapped["ok"]
    separated = ar.conservation_check("اب تث", ar.Reconstruction(text="تث | اب", separators_inserted=1), cfg)
    assert separated["ok"]


def test_config_is_relative_where_possible_and_round_trips() -> None:
    cfg = ar.ReorderConfig(cell_gap_ratio=1.5)
    assert ar.ReorderConfig.from_dict(cfg.to_dict()) == cfg
    assert ar.ReorderConfig.from_dict({"unknown": 1}) == ar.ReorderConfig()
    line = [ar.Cluster(0, 5, 10, 10, "ا"), ar.Cluster(21, 26, 10, 10, "ب")]
    assert len(ar.split_segments(line, ar.ReorderConfig())) == 1  # 16 pt gap < fixed 18 pt
    assert len(ar.split_segments(line, cfg)) == 2  # 16 pt gap > 1.5 x 10 pt glyph height


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice (soffice) is not installed")
def test_libreoffice_producer() -> None:  # pragma: no cover - not available on this machine
    pytest.skip("LibreOffice producer test not implemented without a .docx generator")
