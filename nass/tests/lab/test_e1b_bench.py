from __future__ import annotations

from lab.core.e1b_bench import _report
from lab.core.metrics import TOKEN_CATEGORIES


def _record(variant: str, cer: float, *, regressed: int = 0) -> dict:
    regressions = {
        category: {
            "reference": 2,
            "correct_raw": 2,
            "correct_cleaned": 2 - (regressed if category == "numbers" else 0),
            "regressed": regressed if category == "numbers" else 0,
            "fixed": 0,
            "introduced_wrong": regressed if category == "numbers" else 0,
        }
        for category in TOKEN_CATEGORIES
    }
    record = {
        "language": "English",
        "extractor": "Test",
        "variant": variant,
        "cer": cer,
        "wer": 0.1,
        "char_edits": 10,
        "ref_chars": 100,
        "word_edits": 2,
        "ref_words": 20,
        "reading_order_tau": 1.0,
        "reading_order_matched_lines": 5,
        "reading_order_ref_lines": 5,
        "reading_order_pages_with_tau": 1,
        "verified_pages": 1,
        "regressions": regressions,
    }
    for prefix in ("numeric", "date", "identifier", "operator"):
        record[f"{prefix}_correct"] = 2
        record[f"{prefix}_reference"] = 2
        record[f"{prefix}_hypothesis"] = 2
        record[f"{prefix}_page_weighted_accuracy"] = 1.0
        record[f"{prefix}_pages"] = 1
    return record


def test_report_discloses_denominators_and_sensitive_regression_gate() -> None:
    raw = _record("raw", 0.10)
    cleaned = _record("cleaned", 0.09, regressed=1)
    report = _report([raw, cleaned])
    assert "10/100=0.1000" in report
    assert "item 2/2=1.0000 (hyp 2)" in report
    assert "lines 5/5; pages 1/1" in report
    assert "FAIL (CER -0.0100; regressed 1; new wrong 1)" in report


def _with_curated(record: dict, *, table_correct: int, regressed: int, new: int) -> dict:
    record.update(
        {
            "curated_numeric_text_correct": 3,
            "curated_numeric_text_reference": 3,
            "curated_numeric_table_correct": table_correct,
            "curated_numeric_table_reference": 4,
            "curated_numeric_unmatched_hypothesis": new,
        }
    )
    record["regressions"]["curated_numbers"] = {
        "reference": 7,
        "correct_raw": 7,
        "correct_cleaned": 3 + table_correct,
        "regressed": regressed,
        "fixed": 0,
        "introduced_wrong": new,
    }
    return record


def test_report_shows_curated_table_coverage_and_gates_on_curated_regression() -> None:
    raw = _with_curated(_record("raw", 0.10), table_correct=4, regressed=0, new=0)
    # The regex categories look clean; only the curated table value regressed.
    cleaned = _with_curated(_record("cleaned", 0.10), table_correct=3, regressed=1, new=1)
    report = _report([raw, cleaned])
    assert "text 3/3; table 4/4" in report
    assert "text 3/3; table 3/4" in report
    assert "0/1/1" in report
    assert "FAIL (CER +0.0000; regressed 1; new wrong 1)" in report


def test_report_renders_legacy_records_without_page_mean_or_curated_keys() -> None:
    raw = _record("raw", 0.10)
    cleaned = _record("cleaned", 0.10)
    assert all(not key.endswith("page_mean_accuracy") for key in raw)
    report = _report([raw, cleaned])
    assert "PASS (CER +0.0000; regressed 0; new wrong 0)" in report
