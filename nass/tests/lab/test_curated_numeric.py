from __future__ import annotations

from lab.core.metrics import (
    aggregate_metrics,
    compare_curated_numeric_regressions,
    evaluate_page,
    scan_curated_numeric_values,
)
from lab.core.models import GroundTruthNumericValue, GroundTruthPage, PageResult


def _value(text: str, source: str = "text") -> GroundTruthNumericValue:
    return GroundTruthNumericValue(text=text, source=source)


def _page(text: str) -> PageResult:
    return PageResult(page=1, text=text, blocks=None, status="ok", error=None, elapsed_ms=1)


def _gt(values: list[GroundTruthNumericValue], text: str = "مرجع") -> GroundTruthPage:
    return GroundTruthPage(
        page=1,
        text=text,
        status="verified",
        seeded_from=None,
        verified_by="source",
        verified_at=None,
        notes=None,
        numeric_values=values,
    )


def test_duplicate_curated_values_need_separate_occurrences() -> None:
    scan = scan_curated_numeric_values([_value("٢٠٢٥"), _value("٢٠٢٥")], "التقرير ٢٠٢٥ فقط")
    assert scan.counts("text") == (1, 2)
    scan = scan_curated_numeric_values([_value("٢٠٢٥"), _value("٢٠٢٥")], "٢٠٢٥ و ٢٠٢٥")
    assert scan.counts("text") == (2, 2)


def test_value_inside_a_longer_number_does_not_count() -> None:
    # The year is part of the date; it must not also count as a standalone value.
    scan = scan_curated_numeric_values([_value("١٥/٠٣/٢٠٢٥"), _value("٢٠٢٥")], "التاريخ ١٥/٠٣/٢٠٢٥")
    assert scan.found == (True, False)


def test_overlapping_values_are_not_double_counted() -> None:
    values = [_value("52,214,730.45"), _value("($52,214,730.45)")]
    scan = scan_curated_numeric_values(values, "Revenue ($52,214,730.45)")
    assert sum(scan.found) == 1
    assert scan.found == (False, True)  # longest claims the span first


def test_digit_script_must_match_exactly() -> None:
    scan = scan_curated_numeric_values([_value("٢٠٢٥")], "Report 2025")
    assert scan.counts("text") == (0, 1)
    assert scan.unmatched_hypothesis == {"2025": 1}


def test_table_and_running_text_are_reported_separately() -> None:
    values = [_value("٣٣٫٨٤٪"), _value("١٤٢٫٦", "table"), _value("٩٨٫١", "table")]
    metric = evaluate_page("run", "raw", _page("الهامش ٣٣٫٨٤٪ الجدول ١٤٢٫٦"), _gt(values), 1)
    assert (metric.curated_num_text_correct, metric.curated_num_text_ref) == (1, 1)
    assert (metric.curated_num_table_correct, metric.curated_num_table_ref) == (1, 2)


def test_curated_regressions_fixed_regressed_and_new_wrong() -> None:
    values = [_value("٣٣٫٨٤٪"), _value("١٢٣", "table"), _value("٤٥٦", "table")]
    raw = "الهامش ٤٨٫٣٣٪ الجدول ١٢٣ و ٤٥٦"
    cleaned = "الهامش ٣٣٫٨٤٪ الجدول ٣٢١ و ٤٥٦"
    report = compare_curated_numeric_regressions(values, raw, cleaned)
    assert report["fixed"] == 1
    assert report["regressed"] == 1  # the correct table value ١٢٣ became ٣٢١
    assert report["introduced_wrong"] == 1  # ٣٢١ is a new unexplained number
    assert (report["table_correct_raw"], report["table_correct_cleaned"]) == (2, 1)


def test_aggregate_sums_curated_counts_and_keeps_page_mean_alias() -> None:
    first = evaluate_page("run", "raw", _page("٢٠٢٥ و ١٢٣"), _gt([_value("٢٠٢٥"), _value("١٢٣", "table")], "٢٠٢٥ و ١٢٣"), 1)
    second = evaluate_page("run", "raw", _page("لا شيء"), _gt([_value("٧", "table")], "٧"), 1)
    aggregate = aggregate_metrics([first, second], "run", "raw", 1)
    assert (aggregate.curated_num_text_correct, aggregate.curated_num_text_ref) == (1, 1)
    assert (aggregate.curated_num_table_correct, aggregate.curated_num_table_ref) == (1, 2)
    # Deprecated alias keeps its value for compatibility; the new name is correct.
    assert aggregate.num_page_mean_accuracy == aggregate.num_page_weighted_accuracy
    assert aggregate.num_page_mean_accuracy == (1.0 + 0.0) / 2


def test_pages_without_curated_values_report_zero_reference() -> None:
    metric = evaluate_page("run", "raw", _page("١٢٣"), _gt([], "١٢٣"), 1)
    assert metric.curated_num_text_ref == 0
    assert metric.curated_num_table_ref == 0
