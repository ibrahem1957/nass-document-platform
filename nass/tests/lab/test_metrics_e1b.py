from __future__ import annotations

from lab.core.metrics import (
    UnverifiedGroundTruthError,
    aggregate_metrics,
    compare_text_regressions,
    evaluate_page,
)
from lab.core.models import GroundTruthPage, PageResult


def _page(text: str) -> PageResult:
    return PageResult(
        page=1,
        text=text,
        blocks=None,
        status="ok",
        error=None,
        elapsed_ms=1,
    )


def _gt(text: str, status: str = "verified") -> GroundTruthPage:
    return GroundTruthPage(
        page=1,
        text=text,
        status=status,
        seeded_from=None,
        verified_by="reviewer" if status == "verified" else None,
        verified_at=None,
        notes=None,
    )


def test_hand_computed_cer_wer_and_whitespace_collapse() -> None:
    metric = evaluate_page("run", "raw", _page("abc  def"), _gt("abc xef"), 1)
    assert metric.cer == 1 / 7
    assert metric.wer == 1 / 2


def test_exact_numeric_date_identifier_operator_metrics() -> None:
    reference = "فاتورة INV-00125 بتاريخ 2025-11-06: $52,214.30 والنسبة ٢٥٪ والحد ≤ 0.5"
    metric = evaluate_page("run", "raw", _page(reference), _gt(reference), 1)
    assert metric.num_recall == 1
    assert metric.num_precision == 1
    assert metric.date_accuracy == 1
    assert metric.identifier_accuracy == 1
    assert metric.comparison_operator_accuracy == 1
    assert (metric.num_correct, metric.num_ref_count, metric.num_hyp_count) == (
        3,
        3,
        3,
    )
    assert (metric.date_correct, metric.date_ref_count) == (1, 1)
    assert (metric.identifier_correct, metric.identifier_ref_count) == (1, 1)
    assert (metric.operator_correct, metric.operator_ref_count) == (1, 1)


def test_operator_metric_handles_values_without_spaces() -> None:
    metric = evaluate_page("run", "raw", _page("x≤0.5 y>=10"), _gt("x≤0.5 y>=10"), 1)
    assert metric.comparison_operator_accuracy == 1


def test_digit_form_conversion_is_an_error_not_equivalence() -> None:
    metric = evaluate_page("run", "raw", _page("القيمة 123"), _gt("القيمة ١٢٣"), 1)
    assert metric.num_recall == 0
    assert metric.num_precision == 0


def test_unverified_ground_truth_is_gated() -> None:
    try:
        evaluate_page("run", "raw", _page("x"), _gt("x", "draft"), 1)
    except UnverifiedGroundTruthError:
        pass
    else:
        raise AssertionError("draft ground truth produced metrics")


def test_reading_order_tau_detects_reversed_lines() -> None:
    metric = evaluate_page(
        "run",
        "raw",
        _page("third block\nsecond block\nfirst block"),
        _gt("first block\nsecond block\nthird block"),
        1,
    )
    assert metric.reading_order_tau == -1
    assert metric.bow_f1 == 1
    assert metric.reading_order_matched_lines == 3
    assert metric.reading_order_ref_lines == 3


def test_unreadable_markers_counted_and_aggregate_reports_verified_pages() -> None:
    first = evaluate_page("run", "raw", _page("a\x00b�"), _gt("ab"), 1)
    second = evaluate_page("run", "raw", _page("ok"), _gt("ok"), 1)
    aggregate = aggregate_metrics([first, second], "run", "raw", 1)
    assert aggregate.unreadable_chars == 2
    assert aggregate.verified_pages == 2


def test_item_level_and_page_weighted_accuracy_are_distinct() -> None:
    first = evaluate_page("run", "raw", _page("1"), _gt("1"), 1)
    second = evaluate_page(
        "run",
        "raw",
        _page("2 90 91 92 93 94 95 96 97"),
        _gt("2 3 4 5 6 7 8 9 10"),
        1,
    )
    aggregate = aggregate_metrics([first, second], "run", "raw", 1)
    assert aggregate.num_correct == 2
    assert aggregate.num_ref_count == 10
    assert aggregate.num_recall == 0.2
    assert aggregate.num_page_weighted_accuracy == (1 + 1 / 9) / 2


def test_regression_detection_counts_correct_to_wrong_and_new_wrong_values() -> None:
    report = compare_text_regressions(
        "2025-11-06 INV-00125 25% x <= y",
        "2025-11-06 INV-00125 25% x <= y",
        "2026-11-06 INV-00521 52% x >= y",
    )
    assert report["dates"]["regressed"] == 1
    assert report["identifiers"]["regressed"] == 1
    assert report["numbers"]["regressed"] == 1
    assert report["operators"]["regressed"] == 1
    assert all(report[category]["introduced_wrong"] == 1 for category in report)


def test_regression_detection_counts_fixed_values_without_false_regression() -> None:
    report = compare_text_regressions("25%", "52%", "25%")
    assert report["numbers"]["fixed"] == 1
    assert report["numbers"]["regressed"] == 0
    assert report["numbers"]["introduced_wrong"] == 0
