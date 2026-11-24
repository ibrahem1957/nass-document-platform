"""E2-A metrics that extend E1-B definitions for OCR/VLM output.

- Table cells: *exact transcription* (the cell text appears as a standalone,
  occurrence-claimed span anywhere) is separate from *row/column association*
  (the value sits in a parsed table under the right row header and column).
- Mixed RTL/LTR: reference lines containing both Arabic and Latin letters.
- Missing / extra words against reference text plus table cells.
- `markup_strip` is a separately identified post-processing view; RAW model
  output is never altered.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser

from rapidfuzz.distance import Levenshtein

from lab.core.metrics import aggregate_metrics, evaluate_run, normalize_metric_text
from lab.core.models import GroundTruth, GroundTruthPage, GroundTruthTable, PageResult, RunResult

MARKUP_VARIANT = "markup_strip"
MARKUP_VARIANT_VERSION = "1"
ARABIC_LETTER_RE = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
SEPARATOR_ROW_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _norm(text: str) -> str:
    return normalize_metric_text(text)


# ------------------------------------------------------------- table parsing


class _HTMLTables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._depth += 1
            self.tables.append([])
        elif tag == "tr" and self._depth:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(_norm("".join(self._cell)))
            self._cell = None
        elif tag == "tr" and self._row is not None and self.tables:
            self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_hypothesis_tables(text: str) -> list[list[list[str]]]:
    """Markdown pipe tables and HTML tables found in a model response."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") >= 2 and not SEPARATOR_ROW_RE.match(stripped):
            cells = stripped.strip("|").split("|")
            current.append([_norm(cell) for cell in cells])
        elif SEPARATOR_ROW_RE.match(stripped) and current:
            continue
        else:
            if current:
                tables.append(current)
            current = []
    if current:
        tables.append(current)
    if "<table" in text.lower():
        parser = _HTMLTables()
        parser.feed(text)
        tables.extend(table for table in parser.tables if table)
    return tables


# ------------------------------------------------------------- table metrics


def _boundary_ok(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before and (before.isalnum() or before in ".,٫٬")) and not (
        after and (after.isalnum() or (after in ".,٫٬" and end + 1 < len(text) and text[end + 1].isdigit()))
    )


def claim_exact_values(values: list[str], hypothesis: str) -> list[bool]:
    """Occurrence-claimed exact matching; longest first; no overlapping spans."""
    text = _norm(hypothesis)
    found = [False] * len(values)
    claimed: list[tuple[int, int]] = []
    order = sorted(range(len(values)), key=lambda index: -len(values[index]))
    for index in order:
        needle = _norm(values[index])
        if not needle:
            continue
        start = 0
        while True:
            position = text.find(needle, start)
            if position < 0:
                break
            end = position + len(needle)
            overlaps = any(position < c_end and c_start < end for c_start, c_end in claimed)
            if not overlaps and _boundary_ok(text, position, end):
                claimed.append((position, end))
                found[index] = True
                break
            start = position + 1
    return found


def _column_keys(column: str) -> set[str]:
    parts = [_norm(part) for part in column.split(">")]
    return {_norm(column), parts[-1], _norm(" ".join(parts))}


def _associate(table: GroundTruthTable, hyp_tables: list[list[list[str]]]) -> tuple[int, int]:
    """(correct, total) for data cells (columns 1..n) by row header + column."""
    width = len(table.columns)
    total = sum(1 for row in table.rows for _ in row[1:])
    if not hyp_tables or width < 2:
        return 0, total
    best = 0
    for hyp in hyp_tables:
        rows = [list(row) for row in hyp]
        # Accept visual RTL order (row header last) by mirroring such rows.
        headers = {_norm(row[0]) for row in table.rows}
        for row in rows:
            if row and row[0] not in headers and row[-1] in headers:
                row.reverse()
        header_index: dict[int, int] = {}
        for candidate in rows[:3]:
            for gt_index, column in enumerate(table.columns[1:], start=1):
                keys = _column_keys(column)
                for hyp_index, cell in enumerate(candidate):
                    if cell in keys and gt_index not in header_index:
                        header_index[gt_index] = hyp_index
        by_header = {row[0]: row for row in rows if row}
        correct = 0
        for gt_row in table.rows:
            hyp_row = by_header.get(_norm(gt_row[0]))
            if hyp_row is None:
                continue
            for gt_index, value in enumerate(gt_row[1:], start=1):
                hyp_index = header_index.get(gt_index)
                if hyp_index is None and len(hyp_row) == width:
                    hyp_index = gt_index  # positional fallback for a full-width row
                if hyp_index is not None and hyp_index < len(hyp_row):
                    correct += hyp_row[hyp_index] == _norm(value)
        best = max(best, correct)
    return best, total


@dataclass
class TableScores:
    tables_reference: int = 0
    tables_detected: int = 0
    cells_reference: int = 0
    cells_exact: int = 0
    data_cells_reference: int = 0
    data_cells_associated: int = 0

    def add(self, other: "TableScores") -> None:
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)


def score_tables(tables: list[GroundTruthTable], hypothesis: str) -> TableScores:
    scores = TableScores(tables_reference=len(tables))
    if not tables:
        return scores
    parsed = parse_hypothesis_tables(hypothesis)
    scores.tables_detected = len(parsed)
    cells = [cell for table in tables for row in table.rows for cell in row]
    scores.cells_reference = len(cells)
    scores.cells_exact = sum(claim_exact_values(cells, hypothesis))
    for table in tables:
        correct, total = _associate(table, parsed)
        scores.data_cells_associated += correct
        scores.data_cells_reference += total
    return scores


# ------------------------------------------------- mixed direction / coverage


def _is_mixed(line: str) -> bool:
    return bool(ARABIC_LETTER_RE.search(line) and LATIN_LETTER_RE.search(line))


@dataclass
class PageExtras:
    page: int
    tables: TableScores = field(default_factory=TableScores)
    mixed_lines_reference: int = 0
    mixed_lines_exact: int = 0
    mixed_char_edits: int = 0
    mixed_ref_chars: int = 0
    reference_words: int = 0
    hypothesis_words: int = 0
    missing_words: int = 0
    extra_words: int = 0


def page_extras(gt_page: GroundTruthPage, hypothesis: str) -> PageExtras:
    extras = PageExtras(page=gt_page.page, tables=score_tables(gt_page.tables, hypothesis))
    hyp_norm = _norm(hypothesis)
    hyp_lines = [_norm(line) for line in hypothesis.splitlines() if line.strip()]
    for line in gt_page.text.splitlines():
        ref = _norm(line)
        if not ref or not _is_mixed(ref):
            continue
        extras.mixed_lines_reference += 1
        extras.mixed_ref_chars += len(ref)
        if ref in hyp_norm:
            extras.mixed_lines_exact += 1
            continue
        extras.mixed_char_edits += min(
            (Levenshtein.distance(ref, candidate) for candidate in hyp_lines), default=len(ref)
        )
    reference_parts = [gt_page.text]
    for table in gt_page.tables:
        reference_parts.extend(cell for row in table.rows for cell in row)
        reference_parts.extend(part for column in table.columns for part in column.split(">"))
    ref_words = Counter(_norm(" ".join(reference_parts)).split())
    hyp_words = Counter(_norm(strip_markup(hypothesis)).split())
    extras.reference_words = sum(ref_words.values())
    extras.hypothesis_words = sum(hyp_words.values())
    extras.missing_words = sum((ref_words - hyp_words).values())
    extras.extra_words = sum((hyp_words - ref_words).values())
    return extras


# ----------------------------------------------------- markup post-processing


_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def strip_markup(text: str) -> str:
    """Remove Markdown/HTML presentation markup; keep every character of content."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or SEPARATOR_ROW_RE.match(stripped):
            continue
        if stripped.count("|") >= 2:
            line = " ".join(cell.strip() for cell in stripped.strip("|").split("|"))
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"(\*\*|__)(.+?)\1", r"\2", line)
        line = _HTML_TAG_RE.sub(" ", line)
        lines.append(unicodedata.normalize("NFC", line))
    return "\n".join(lines)


def derive_markup_strip(run: RunResult) -> RunResult:
    """A separately identified post-processing run. The RAW run is not modified."""
    if run.raw_run_id is not None or run.variant != "raw":
        raise ValueError("markup_strip derives from a RAW run only")
    pages = [
        PageResult(
            page=page.page,
            text=strip_markup(page.text),
            blocks=None,
            status=page.status,
            error=page.error,
            elapsed_ms=page.elapsed_ms,
        )
        for page in run.pages
    ]
    return run.model_copy(
        deep=True,
        update={
            "run_id": f"{run.run_id}_{MARKUP_VARIANT}",
            "variant": MARKUP_VARIANT,
            "pages": pages,
            "raw_run_id": run.run_id,
            "variant_version": MARKUP_VARIANT_VERSION,
        },
    )


# ----------------------------------------------------------- run summaries


def repetition_stats(text: str) -> dict[str, int | float | bool]:
    """Conservative loop signal; it records evidence but never changes RAW text."""
    tokens = re.findall(r"\S+", text)
    maximum = 0
    current = 0
    previous: str | None = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        maximum = max(maximum, current)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repeated_line_excess = sum(count - 1 for count in Counter(lines).values() if count > 1)
    max_consecutive_identical_lines = 0
    current_line_run = 0
    previous_line: str | None = None
    for line in lines:
        if line == previous_line:
            current_line_run += 1
        else:
            previous_line = line
            current_line_run = 1
        max_consecutive_identical_lines = max(
            max_consecutive_identical_lines, current_line_run
        )
    unique_ratio = len(set(tokens)) / len(tokens) if tokens else 1.0
    flagged = maximum >= 8 or max_consecutive_identical_lines >= 4 or (
        len(tokens) >= 80 and unique_ratio < 0.12
    )
    return {
        "flagged": flagged,
        "tokens": len(tokens),
        "unique_token_ratio": unique_ratio,
        "max_consecutive_identical_tokens": maximum,
        "repeated_line_excess": repeated_line_excess,
        "max_consecutive_identical_lines": max_consecutive_identical_lines,
    }


def summarize_run(run: RunResult, ground_truth: GroundTruth, *, table_source: RunResult | None = None) -> dict:
    """E1-B aggregate metrics plus E2 extras, with exact counts.

    `table_source` (normally the RAW run) supplies table structure when `run` is
    a markup-stripped view, so association is always scored on RAW structure.
    """
    page_metrics = evaluate_run(run, ground_truth)
    aggregate = aggregate_metrics(page_metrics, run.run_id, run.variant, ground_truth.version)
    verified = {page.page: page for page in ground_truth.pages if page.status == "verified"}
    structure = {page.page: page.text for page in (table_source or run).pages}
    tables = TableScores()
    mixed = {"reference": 0, "exact": 0, "char_edits": 0, "ref_chars": 0}
    words = {"reference": 0, "hypothesis": 0, "missing": 0, "extra": 0}
    by_page = {item.page: item for item in page_metrics}
    e2 = run.environment.get("e2") or {}
    raw_pages = e2.get("page_raw") or {}
    quality = {
        "pages_checked": 0,
        "repetition_flagged_pages": [],
        "truncation_known_pages": 0,
        "truncated_pages": [],
    }
    per_page: list[dict] = []
    for page in run.pages:
        gt_page = verified.get(page.page)
        if gt_page is None:
            continue
        extras = page_extras(gt_page, page.text)
        table_scores = score_tables(gt_page.tables, structure.get(page.page, page.text))
        tables.add(table_scores)
        mixed["reference"] += extras.mixed_lines_reference
        mixed["exact"] += extras.mixed_lines_exact
        mixed["char_edits"] += extras.mixed_char_edits
        mixed["ref_chars"] += extras.mixed_ref_chars
        words["reference"] += extras.reference_words
        words["hypothesis"] += extras.hypothesis_words
        words["missing"] += extras.missing_words
        words["extra"] += extras.extra_words
        metric = by_page.get(page.page)
        if metric is not None:
            repetition = repetition_stats(page.text)
            quality["pages_checked"] += 1
            if repetition["flagged"]:
                quality["repetition_flagged_pages"].append(page.page)
            raw = (raw_pages.get(str(page.page)) or {}).get("response")
            truncated = None
            if (
                isinstance(raw, dict)
                and isinstance(raw.get("truncated_at_max_new_tokens"), bool)
            ):
                quality["truncation_known_pages"] += 1
                truncated = raw["truncated_at_max_new_tokens"]
                if truncated:
                    quality["truncated_pages"].append(page.page)
            per_page.append(
                {
                    "page": page.page,
                    "status": page.status,
                    "error": page.error,
                    "elapsed_ms": page.elapsed_ms,
                    "cpu_ms": page.extraction_cpu_ms,
                    "char_edits": metric.char_edits,
                    "ref_chars": metric.ref_chars,
                    "hyp_chars": metric.hyp_chars,
                    "cer": metric.cer,
                    "word_edits": metric.word_edits,
                    "ref_words": metric.ref_words,
                    "wer": metric.wer,
                    "numbers": [metric.num_correct, metric.num_ref_count, metric.num_hyp_count],
                    "dates": [metric.date_correct, metric.date_ref_count, metric.date_hyp_count],
                    "identifiers": [
                        metric.identifier_correct,
                        metric.identifier_ref_count,
                        metric.identifier_hyp_count,
                    ],
                    "operators": [metric.operator_correct, metric.operator_ref_count, metric.operator_hyp_count],
                    "curated_text": [metric.curated_num_text_correct, metric.curated_num_text_ref],
                    "curated_table": [metric.curated_num_table_correct, metric.curated_num_table_ref],
                    "curated_unmatched_hyp": metric.curated_num_unmatched_hyp,
                    "reading_order": {
                        "tau": metric.reading_order_tau,
                        "matched_lines": metric.reading_order_matched_lines,
                        "ref_lines": metric.reading_order_ref_lines,
                    },
                    "tables": asdict(table_scores),
                    "mixed_direction": [extras.mixed_lines_exact, extras.mixed_lines_reference],
                    "words_missing_extra": [extras.missing_words, extras.reference_words, extras.extra_words],
                    "repetition": repetition,
                    "truncated_at_max_new_tokens": truncated,
                }
            )
    timed = [page.elapsed_ms for page in run.pages if page.elapsed_ms is not None]
    return {
        "run_id": run.run_id,
        "adapter_id": run.adapter.adapter_id,
        "label": run.adapter.label,
        "model": run.adapter.model_id,
        "variant": run.variant,
        "is_mock": bool(e2.get("is_mock")),
        "status": run.status,
        "pages_ok": sum(page.status == "ok" for page in run.pages),
        "pages_total": len(run.pages),
        "metrics": aggregate.model_dump(mode="json"),
        "tables": asdict(tables),
        "mixed_direction": mixed,
        "words": words,
        "quality": quality,
        "per_page": per_page,
        "timing": {
            "load_ms": e2.get("load_ms"),
            "page_ms_total": sum(timed) if timed else None,
            "page_ms_mean": (sum(timed) / len(timed)) if timed else None,
            "pages_timed": len(timed),
        },
        "memory": {
            "peak_rss_mb": e2.get("peak_rss_mb"),
            "peak_vram_mb": e2.get("peak_vram_mb"),
            "note": e2.get("vram_note"),
        },
    }
