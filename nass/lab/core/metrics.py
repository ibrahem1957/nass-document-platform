from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from rapidfuzz.distance import Levenshtein

from lab.core.models import (
    GroundTruth,
    GroundTruthNumericValue,
    GroundTruthPage,
    MetricResult,
    PageResult,
    RunResult,
)


DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9٠-٩۰-۹_])"
    r"(?:(?:[$€£¥₹﷼]|ر\.س\.?|د\.إ\.?)\s*)?"
    r"(?:[+\-−]\s*)?"
    r"[0-9٠-٩۰-۹]+(?:[.,٫٬:/\-][0-9٠-٩۰-۹]+)*"
    r"(?:\s*(?:%|٪|‰|bps|USD|SAR|AED|EUR|GBP|ر\.س\.?|د\.إ\.?))?"
    r"(?![A-Za-z0-9٠-٩۰-۹_])",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"(?<![\w])(?:[0-9٠-٩۰-۹]{1,4}[-/.][0-9٠-٩۰-۹]{1,2}[-/.][0-9٠-٩۰-۹]{1,4})(?![\w])"
)
IDENTIFIER_RE = re.compile(
    r"(?<![\w])[A-Za-z\u0600-\u06ff0-9٠-٩۰-۹]+"
    r"(?:[-_/][A-Za-z\u0600-\u06ff0-9٠-٩۰-۹]+)+(?![\w])"
)
OPERATOR_RE = re.compile(
    r"(?<![<>=≤≥!])(?:<=|>=|!=|==|[<>≤≥=±])(?![<>=≤≥!])"
)
UNREADABLE_RE = re.compile(r"\x00|�|\(cid:\d+\)|⟦\?⟧")


class UnverifiedGroundTruthError(ValueError):
    pass


@dataclass(frozen=True)
class TokenCounts:
    correct: int
    reference: int
    hypothesis: int

    @property
    def recall(self) -> float | None:
        return self.correct / self.reference if self.reference else None

    @property
    def precision(self) -> float | None:
        return self.correct / self.hypothesis if self.hypothesis else None


TOKEN_CATEGORIES = ("numbers", "dates", "identifiers", "operators")


def normalize_metric_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _ratio_distance(reference, hypothesis) -> float:
    denominator = len(reference)
    distance = Levenshtein.distance(reference, hypothesis)
    if denominator:
        return distance / denominator
    return 0.0 if not hypothesis else 1.0


def _bow_f1(reference: list[str], hypothesis: list[str]) -> float:
    ref_counts = Counter(reference)
    hyp_counts = Counter(hypothesis)
    overlap = sum((ref_counts & hyp_counts).values())
    if not reference and not hypothesis:
        return 1.0
    precision = overlap / len(hypothesis) if hypothesis else 0.0
    recall = overlap / len(reference) if reference else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _counter(pattern: re.Pattern[str], text: str) -> Counter[str]:
    return Counter(match.group(0).strip() for match in pattern.finditer(text))


def _number_counter(text: str) -> Counter[str]:
    excluded_spans = [
        match.span()
        for pattern in (DATE_RE, IDENTIFIER_RE)
        for match in pattern.finditer(text)
    ]
    values: Counter[str] = Counter()
    for match in NUMBER_RE.finditer(text):
        if any(
            match.start() >= start and match.end() <= end
            for start, end in excluded_spans
        ):
            continue
        values[match.group(0).strip()] += 1
    return values


def _category_counter(category: str, text: str) -> Counter[str]:
    if category == "numbers":
        return _number_counter(text)
    if category == "identifiers":
        values = _counter(IDENTIFIER_RE, text)
        for value in list(values):
            if DATE_RE.fullmatch(value):
                del values[value]
        return values
    patterns = {
        "dates": DATE_RE,
        "operators": OPERATOR_RE,
    }
    return _counter(patterns[category], text)


def _token_counts(category: str, reference: str, hypothesis: str) -> TokenCounts:
    ref = _category_counter(category, reference)
    hyp = _category_counter(category, hypothesis)
    return TokenCounts(
        correct=sum((ref & hyp).values()),
        reference=sum(ref.values()),
        hypothesis=sum(hyp.values()),
    )


_DIGITS = frozenset("0123456789٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹")
_NUMBER_JOINERS = frozenset(".,٫٬:/-")


def _standalone(text: str, start: int, end: int) -> bool:
    """True when text[start:end] is not part of a longer number.

    A neighbour digit, or a separator that continues into a digit, means the
    match sits inside a larger value (e.g. a year inside a date), so it must
    not count as the curated value.
    """
    if start > 0:
        before = text[start - 1]
        if before in _DIGITS or (
            before in _NUMBER_JOINERS and start >= 2 and text[start - 2] in _DIGITS
        ):
            return False
    if end < len(text):
        after = text[end]
        if after in _DIGITS or (
            after in _NUMBER_JOINERS and end + 1 < len(text) and text[end + 1] in _DIGITS
        ):
            return False
    return True


@dataclass(frozen=True)
class CuratedNumericScan:
    found: tuple[bool, ...]  # one flag per curated occurrence, same order as input
    sources: tuple[str, ...]
    unmatched_hypothesis: Counter  # hypothesis numeric tokens not explained by a curated value

    def counts(self, source: str) -> tuple[int, int]:
        pairs = [flag for flag, item in zip(self.found, self.sources, strict=True) if item == source]
        return sum(pairs), len(pairs)


def scan_curated_numeric_values(
    values: list[GroundTruthNumericValue], hypothesis: str
) -> CuratedNumericScan:
    """Match each curated occurrence to its own exact, non-overlapping span.

    Values are claimed longest-first so a short value (e.g. `2025`) cannot reuse
    characters already claimed by a longer one (e.g. `15/03/2025`). Duplicate
    curated entries must each find a separate occurrence. Ground truth is used
    only to score; it never alters the hypothesis.
    """
    text = normalize_metric_text(hypothesis)
    targets = [normalize_metric_text(value.text) for value in values]
    claimed: list[tuple[int, int]] = []
    found = [False] * len(values)
    for index in sorted(range(len(values)), key=lambda item: -len(targets[item])):
        target = targets[index]
        if not target:
            continue
        cursor = 0
        while (position := text.find(target, cursor)) >= 0:
            end = position + len(target)
            overlaps = any(position < right and left < end for left, right in claimed)
            if not overlaps and _standalone(text, position, end):
                claimed.append((position, end))
                found[index] = True
                break
            cursor = position + 1
    unmatched: Counter[str] = Counter()
    for match in NUMBER_RE.finditer(text):
        start, end = match.span()
        if not any(start < right and left < end for left, right in claimed):
            unmatched[match.group(0).strip()] += 1
    return CuratedNumericScan(
        found=tuple(found),
        sources=tuple(value.source for value in values),
        unmatched_hypothesis=unmatched,
    )


def compare_curated_numeric_regressions(
    values: list[GroundTruthNumericValue], raw_text: str, cleaned_text: str
) -> dict[str, int]:
    """RAW → CLEANED changes on curated values, occurrence by occurrence."""
    raw = scan_curated_numeric_values(values, raw_text)
    cleaned = scan_curated_numeric_values(values, cleaned_text)
    report = {"reference": len(values)}
    for source in ("text", "table"):
        raw_correct, reference = raw.counts(source)
        cleaned_correct, _ = cleaned.counts(source)
        report[f"{source}_reference"] = reference
        report[f"{source}_correct_raw"] = raw_correct
        report[f"{source}_correct_cleaned"] = cleaned_correct
    report["correct_raw"] = sum(raw.found)
    report["correct_cleaned"] = sum(cleaned.found)
    report["regressed"] = sum(r and not c for r, c in zip(raw.found, cleaned.found, strict=True))
    report["fixed"] = sum(c and not r for r, c in zip(raw.found, cleaned.found, strict=True))
    report["introduced_wrong"] = sum(
        (cleaned.unmatched_hypothesis - raw.unmatched_hypothesis).values()
    )
    return report


def _line_order_tau(
    reference: str, hypothesis: str
) -> tuple[float | None, int, int, int]:
    ref_lines = [normalize_metric_text(line) for line in reference.splitlines() if line.strip()]
    hyp_lines = [normalize_metric_text(line) for line in hypothesis.splitlines() if line.strip()]
    if len(ref_lines) < 2 or len(hyp_lines) < 2:
        return None, 0, len(ref_lines), len(hyp_lines)
    used: set[int] = set()
    order: list[int] = []
    for ref_line in ref_lines:
        candidates = [
            (SequenceMatcher(None, ref_line, hyp_line).ratio(), index)
            for index, hyp_line in enumerate(hyp_lines)
            if index not in used
        ]
        if not candidates:
            continue
        score, index = max(candidates)
        if score >= 0.6:
            order.append(index)
            used.add(index)
    if len(order) < 2:
        return None, len(order), len(ref_lines), len(hyp_lines)
    concordant = discordant = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            if order[left] < order[right]:
                concordant += 1
            elif order[left] > order[right]:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else None
    return tau, len(order), len(ref_lines), len(hyp_lines)


def evaluate_page(
    run_id: str,
    variant: str,
    page: PageResult,
    ground_truth: GroundTruthPage,
    gt_version: int,
) -> MetricResult:
    if ground_truth.status != "verified":
        raise UnverifiedGroundTruthError("Verify ground truth before computing accuracy")
    reference = normalize_metric_text(ground_truth.text)
    hypothesis = normalize_metric_text(page.text)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    char_edits = Levenshtein.distance(reference, hypothesis)
    word_edits = Levenshtein.distance(ref_words, hyp_words)
    ref_no_diacritics = DIACRITICS_RE.sub("", reference)
    hyp_no_diacritics = DIACRITICS_RE.sub("", hypothesis)
    char_edits_no_diacritics = Levenshtein.distance(
        ref_no_diacritics, hyp_no_diacritics
    )
    numbers = _token_counts("numbers", reference, hypothesis)
    dates = _token_counts("dates", reference, hypothesis)
    identifiers = _token_counts("identifiers", reference, hypothesis)
    operators = _token_counts("operators", reference, hypothesis)
    order_tau, matched_lines, ref_lines, hyp_lines = _line_order_tau(
        ground_truth.text, page.text
    )
    curated = scan_curated_numeric_values(ground_truth.numeric_values, page.text)
    curated_text_correct, curated_text_ref = curated.counts("text")
    curated_table_correct, curated_table_ref = curated.counts("table")
    return MetricResult(
        run_id=run_id,
        variant=variant,
        page=page.page,
        gt_version=gt_version,
        cer=_ratio_distance(reference, hypothesis),
        wer=_ratio_distance(ref_words, hyp_words),
        cer_no_diacritics=_ratio_distance(ref_no_diacritics, hyp_no_diacritics),
        bow_f1=_bow_f1(ref_words, hyp_words),
        num_recall=numbers.recall,
        num_precision=numbers.precision,
        date_accuracy=dates.recall,
        identifier_accuracy=identifiers.recall,
        comparison_operator_accuracy=operators.recall,
        reading_order_tau=order_tau,
        unreadable_chars=len(UNREADABLE_RE.findall(page.text)),
        ref_chars=len(reference),
        hyp_chars=len(hypothesis),
        char_edits=char_edits,
        word_edits=word_edits,
        ref_words=len(ref_words),
        hyp_words=len(hyp_words),
        char_edits_no_diacritics=char_edits_no_diacritics,
        ref_chars_no_diacritics=len(ref_no_diacritics),
        num_correct=numbers.correct,
        num_ref_count=numbers.reference,
        num_hyp_count=numbers.hypothesis,
        date_correct=dates.correct,
        date_ref_count=dates.reference,
        date_hyp_count=dates.hypothesis,
        identifier_correct=identifiers.correct,
        identifier_ref_count=identifiers.reference,
        identifier_hyp_count=identifiers.hypothesis,
        operator_correct=operators.correct,
        operator_ref_count=operators.reference,
        operator_hyp_count=operators.hypothesis,
        num_page_weighted_accuracy=numbers.recall,
        date_page_weighted_accuracy=dates.recall,
        identifier_page_weighted_accuracy=identifiers.recall,
        operator_page_weighted_accuracy=operators.recall,
        num_page_mean_accuracy=numbers.recall,
        date_page_mean_accuracy=dates.recall,
        identifier_page_mean_accuracy=identifiers.recall,
        operator_page_mean_accuracy=operators.recall,
        curated_num_text_correct=curated_text_correct,
        curated_num_text_ref=curated_text_ref,
        curated_num_table_correct=curated_table_correct,
        curated_num_table_ref=curated_table_ref,
        curated_num_unmatched_hyp=sum(curated.unmatched_hypothesis.values()),
        num_pages_with_items=int(numbers.reference > 0),
        date_pages_with_items=int(dates.reference > 0),
        identifier_pages_with_items=int(identifiers.reference > 0),
        operator_pages_with_items=int(operators.reference > 0),
        reading_order_matched_lines=matched_lines,
        reading_order_ref_lines=ref_lines,
        reading_order_hyp_lines=hyp_lines,
        reading_order_pages_with_tau=int(order_tau is not None),
    )


def evaluate_run(run: RunResult, ground_truth: GroundTruth) -> list[MetricResult]:
    if run.doc_id != ground_truth.doc_id or run.doc_sha256 != ground_truth.doc_sha256:
        raise ValueError("Run and ground truth document identity differ")
    gt_by_page = {page.page: page for page in ground_truth.pages}
    results: list[MetricResult] = []
    for page in run.pages:
        gt_page = gt_by_page.get(page.page)
        if gt_page is None or gt_page.status != "verified":
            continue
        results.append(evaluate_page(run.run_id, run.variant, page, gt_page, ground_truth.version))
    return results


def aggregate_metrics(results: list[MetricResult], run_id: str, variant: str, gt_version: int) -> MetricResult:
    if not results:
        raise UnverifiedGroundTruthError("No verified ground-truth pages overlap this run")
    ref_chars = sum(result.ref_chars for result in results)
    hyp_chars = sum(result.hyp_chars for result in results)
    ref_words = sum(result.ref_words for result in results)
    hyp_words = sum(result.hyp_words for result in results)
    ref_chars_no_diacritics = sum(
        result.ref_chars_no_diacritics for result in results
    )
    def total(field: str) -> int:
        return sum(int(getattr(result, field)) for result in results)

    def item_accuracy(correct_field: str, reference_field: str) -> float | None:
        reference_count = total(reference_field)
        return total(correct_field) / reference_count if reference_count else None

    def item_precision(correct_field: str, hypothesis_field: str) -> float | None:
        hypothesis_count = total(hypothesis_field)
        return total(correct_field) / hypothesis_count if hypothesis_count else None

    def page_mean(field: str) -> float | None:
        values = [
            float(value)
            for result in results
            if (value := getattr(result, field)) is not None
        ]
        return sum(values) / len(values) if values else None

    return MetricResult(
        run_id=run_id,
        variant=variant,
        page=None,
        gt_version=gt_version,
        cer=total("char_edits") / ref_chars if ref_chars else 0.0,
        wer=total("word_edits") / ref_words if ref_words else 0.0,
        cer_no_diacritics=(
            total("char_edits_no_diacritics") / ref_chars_no_diacritics
            if ref_chars_no_diacritics
            else 0.0
        ),
        bow_f1=sum(result.bow_f1 for result in results) / len(results),
        num_recall=item_accuracy("num_correct", "num_ref_count"),
        num_precision=item_precision("num_correct", "num_hyp_count"),
        date_accuracy=item_accuracy("date_correct", "date_ref_count"),
        identifier_accuracy=item_accuracy(
            "identifier_correct", "identifier_ref_count"
        ),
        comparison_operator_accuracy=item_accuracy(
            "operator_correct", "operator_ref_count"
        ),
        reading_order_tau=page_mean("reading_order_tau"),
        unreadable_chars=sum(result.unreadable_chars for result in results),
        ref_chars=ref_chars,
        hyp_chars=hyp_chars,
        verified_pages=len(results),
        char_edits=total("char_edits"),
        word_edits=total("word_edits"),
        ref_words=ref_words,
        hyp_words=hyp_words,
        char_edits_no_diacritics=total("char_edits_no_diacritics"),
        ref_chars_no_diacritics=ref_chars_no_diacritics,
        num_correct=total("num_correct"),
        num_ref_count=total("num_ref_count"),
        num_hyp_count=total("num_hyp_count"),
        date_correct=total("date_correct"),
        date_ref_count=total("date_ref_count"),
        date_hyp_count=total("date_hyp_count"),
        identifier_correct=total("identifier_correct"),
        identifier_ref_count=total("identifier_ref_count"),
        identifier_hyp_count=total("identifier_hyp_count"),
        operator_correct=total("operator_correct"),
        operator_ref_count=total("operator_ref_count"),
        operator_hyp_count=total("operator_hyp_count"),
        num_page_weighted_accuracy=page_mean("num_recall"),
        date_page_weighted_accuracy=page_mean("date_accuracy"),
        identifier_page_weighted_accuracy=page_mean("identifier_accuracy"),
        operator_page_weighted_accuracy=page_mean(
            "comparison_operator_accuracy"
        ),
        num_page_mean_accuracy=page_mean("num_recall"),
        date_page_mean_accuracy=page_mean("date_accuracy"),
        identifier_page_mean_accuracy=page_mean("identifier_accuracy"),
        operator_page_mean_accuracy=page_mean("comparison_operator_accuracy"),
        curated_num_text_correct=total("curated_num_text_correct"),
        curated_num_text_ref=total("curated_num_text_ref"),
        curated_num_table_correct=total("curated_num_table_correct"),
        curated_num_table_ref=total("curated_num_table_ref"),
        curated_num_unmatched_hyp=total("curated_num_unmatched_hyp"),
        num_pages_with_items=total("num_pages_with_items"),
        date_pages_with_items=total("date_pages_with_items"),
        identifier_pages_with_items=total("identifier_pages_with_items"),
        operator_pages_with_items=total("operator_pages_with_items"),
        reading_order_matched_lines=total("reading_order_matched_lines"),
        reading_order_ref_lines=total("reading_order_ref_lines"),
        reading_order_hyp_lines=total("reading_order_hyp_lines"),
        reading_order_pages_with_tau=sum(
            result.reading_order_tau is not None for result in results
        ),
    )


def compare_text_regressions(
    reference: str, raw_text: str, cleaned_text: str
) -> dict[str, dict[str, int]]:
    reference = normalize_metric_text(reference)
    raw_text = normalize_metric_text(raw_text)
    cleaned_text = normalize_metric_text(cleaned_text)
    report: dict[str, dict[str, int]] = {}
    for category in TOKEN_CATEGORIES:
        ref = _category_counter(category, reference)
        raw = _category_counter(category, raw_text)
        cleaned = _category_counter(category, cleaned_text)
        raw_correct = ref & raw
        cleaned_correct = ref & cleaned
        raw_wrong = raw - ref
        cleaned_wrong = cleaned - ref
        report[category] = {
            "reference": sum(ref.values()),
            "correct_raw": sum(raw_correct.values()),
            "correct_cleaned": sum(cleaned_correct.values()),
            "regressed": sum((raw_correct - cleaned_correct).values()),
            "fixed": sum((cleaned_correct - raw_correct).values()),
            "introduced_wrong": sum((cleaned_wrong - raw_wrong).values()),
        }
    return report


def compare_run_regressions(
    raw_run: RunResult, cleaned_run: RunResult, ground_truth: GroundTruth
) -> dict[str, dict[str, int]]:
    if cleaned_run.raw_run_id != raw_run.run_id:
        raise ValueError("Cleaned run is not derived from the supplied RAW run")
    totals = {
        category: {
            "reference": 0,
            "correct_raw": 0,
            "correct_cleaned": 0,
            "regressed": 0,
            "fixed": 0,
            "introduced_wrong": 0,
        }
        for category in TOKEN_CATEGORIES
    }
    curated_totals: dict[str, int] = {}
    gt_by_page = {page.page: page for page in ground_truth.pages}
    cleaned_by_page = {page.page: page for page in cleaned_run.pages}
    for raw_page in raw_run.pages:
        gt_page = gt_by_page.get(raw_page.page)
        cleaned_page = cleaned_by_page.get(raw_page.page)
        if (
            gt_page is None
            or gt_page.status != "verified"
            or cleaned_page is None
        ):
            continue
        page_report = compare_text_regressions(
            gt_page.text, raw_page.text, cleaned_page.text
        )
        for category in TOKEN_CATEGORIES:
            for field, value in page_report[category].items():
                totals[category][field] += value
        if gt_page.numeric_values:
            curated = compare_curated_numeric_regressions(
                gt_page.numeric_values, raw_page.text, cleaned_page.text
            )
            for field, value in curated.items():
                curated_totals[field] = curated_totals.get(field, 0) + value
    if curated_totals:
        # Separate key: running-text regex categories stay comparable with
        # earlier reports, while curated values add table-cell coverage.
        totals["curated_numbers"] = curated_totals
    return totals


def finite_metric(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.4f}"
