from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from lab.core.adapters.postprocess import available_variants, derive_variant
from lab.core.adapters.registry import AdapterRegistry
from lab.core.groundtruth import GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.metrics import (
    TOKEN_CATEGORIES,
    aggregate_metrics,
    compare_run_regressions,
    evaluate_run,
    finite_metric,
)
from lab.core.runner import LabRunner
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager


ENGLISH_REGRESSION_TOLERANCE = 0.001


def _metric_record(
    language: str, label: str, result, regressions: dict[str, dict[str, int]]
) -> dict:
    return {
        "language": language,
        "extractor": label,
        "run_id": result.run_id,
        "variant": result.variant,
        "verified_pages": result.verified_pages,
        "cer": result.cer,
        "wer": result.wer,
        "cer_no_diacritics": result.cer_no_diacritics,
        "bow_f1": result.bow_f1,
        "numeric_recall": result.num_recall,
        "numeric_precision": result.num_precision,
        "date_accuracy": result.date_accuracy,
        "identifier_accuracy": result.identifier_accuracy,
        "operator_accuracy": result.comparison_operator_accuracy,
        "reading_order_tau": result.reading_order_tau,
        "unreadable_chars": result.unreadable_chars,
        "char_edits": result.char_edits,
        "ref_chars": result.ref_chars,
        "word_edits": result.word_edits,
        "ref_words": result.ref_words,
        "numeric_correct": result.num_correct,
        "numeric_reference": result.num_ref_count,
        "numeric_hypothesis": result.num_hyp_count,
        "numeric_page_weighted_accuracy": result.num_page_weighted_accuracy,
        "numeric_pages": result.num_pages_with_items,
        "date_correct": result.date_correct,
        "date_reference": result.date_ref_count,
        "date_hypothesis": result.date_hyp_count,
        "date_page_weighted_accuracy": result.date_page_weighted_accuracy,
        "date_pages": result.date_pages_with_items,
        "identifier_correct": result.identifier_correct,
        "identifier_reference": result.identifier_ref_count,
        "identifier_hypothesis": result.identifier_hyp_count,
        "identifier_page_weighted_accuracy": (
            result.identifier_page_weighted_accuracy
        ),
        "identifier_pages": result.identifier_pages_with_items,
        "operator_correct": result.operator_correct,
        "operator_reference": result.operator_ref_count,
        "operator_hypothesis": result.operator_hyp_count,
        "operator_page_weighted_accuracy": result.operator_page_weighted_accuracy,
        "operator_pages": result.operator_pages_with_items,
        "reading_order_matched_lines": result.reading_order_matched_lines,
        "reading_order_ref_lines": result.reading_order_ref_lines,
        "reading_order_hyp_lines": result.reading_order_hyp_lines,
        "reading_order_pages_with_tau": result.reading_order_pages_with_tau,
        # Correct names; the *_page_weighted_accuracy keys above are deprecated
        # aliases holding the same unweighted page mean.
        "numeric_page_mean_accuracy": result.num_page_mean_accuracy,
        "date_page_mean_accuracy": result.date_page_mean_accuracy,
        "identifier_page_mean_accuracy": result.identifier_page_mean_accuracy,
        "operator_page_mean_accuracy": result.operator_page_mean_accuracy,
        "curated_numeric_text_correct": result.curated_num_text_correct,
        "curated_numeric_text_reference": result.curated_num_text_ref,
        "curated_numeric_table_correct": result.curated_num_table_correct,
        "curated_numeric_table_reference": result.curated_num_table_ref,
        "curated_numeric_unmatched_hypothesis": result.curated_num_unmatched_hyp,
        "regressions": regressions,
    }


def _raw_regression_baseline(result) -> dict[str, dict[str, int]]:
    counts = {
        "numbers": (result.num_ref_count, result.num_correct),
        "dates": (result.date_ref_count, result.date_correct),
        "identifiers": (result.identifier_ref_count, result.identifier_correct),
        "operators": (result.operator_ref_count, result.operator_correct),
    }
    baseline = {
        category: {
            "reference": reference,
            "correct_raw": correct,
            "correct_cleaned": correct,
            "regressed": 0,
            "fixed": 0,
            "introduced_wrong": 0,
        }
        for category, (reference, correct) in counts.items()
    }
    curated_reference = result.curated_num_text_ref + result.curated_num_table_ref
    if curated_reference:
        curated_correct = result.curated_num_text_correct + result.curated_num_table_correct
        baseline["curated_numbers"] = {
            "reference": curated_reference,
            "text_reference": result.curated_num_text_ref,
            "text_correct_raw": result.curated_num_text_correct,
            "text_correct_cleaned": result.curated_num_text_correct,
            "table_reference": result.curated_num_table_ref,
            "table_correct_raw": result.curated_num_table_correct,
            "table_correct_cleaned": result.curated_num_table_correct,
            "correct_raw": curated_correct,
            "correct_cleaned": curated_correct,
            "regressed": 0,
            "fixed": 0,
            "introduced_wrong": 0,
        }
    return baseline


def _curated(item: dict, field: str) -> int:
    return int(item["regressions"].get("curated_numbers", {}).get(field, 0))


def evaluate_document(
    manager: WorkspaceManager,
    pdf_path: Path,
    gt_path: Path,
    language: str,
) -> list[dict]:
    workspace = manager.create()
    try:
        settings = manager.settings
        ingest = IngestService(manager, settings)
        registry = AdapterRegistry()
        runner = LabRunner(manager, settings, registry)
        gt_store = GroundTruthStore(manager)
        with pdf_path.open("rb") as handle:
            document = ingest.ingest(workspace.workspace_id, handle, pdf_path.name)
        ground_truth = gt_store.import_json(
            workspace.workspace_id,
            document,
            gt_path.read_bytes(),
            verified=True,
            reviewer="benchmark-source-validated",
        )
        pages = list(range(1, document.page_count + 1))
        source = ingest.original_path(workspace.workspace_id, document.doc_id)
        records: list[dict] = []
        for entry in registry.list("text"):
            if entry.info.kind != "local" or not entry.info.available:
                continue
            raw = runner.run_local(
                workspace.workspace_id,
                document,
                source,
                entry.info.adapter_id,
                pages,
                bypass_cache=True,
            )
            raw_snapshot = raw.model_dump_json()
            variants = ["raw"] + [
                item
                for item in available_variants(
                    entry.info.adapter_id, include_experimental=True
                )
                if item not in {"raw", "unmapped_marker"}
            ]
            for variant in variants:
                run = raw if variant == "raw" else derive_variant(raw, variant)
                page_metrics = evaluate_run(run, ground_truth)
                aggregate = aggregate_metrics(
                    page_metrics, run.run_id, run.variant, ground_truth.version
                )
                regressions = (
                    _raw_regression_baseline(aggregate)
                    if variant == "raw"
                    else compare_run_regressions(raw, run, ground_truth)
                )
                records.append(
                    _metric_record(
                        language, entry.info.label, aggregate, regressions
                    )
                )
                if raw.model_dump_json() != raw_snapshot:
                    raise AssertionError("A correction mutated the RAW run")
        return records
    finally:
        manager.end(workspace.workspace_id)


def _report(records: list[dict]) -> str:
    raw_by_key = {
        (item["language"], item["extractor"]): item
        for item in records
        if item["variant"] == "raw"
    }
    lines = [
        "# E1-B Local Evaluation Report",
        "",
        "All extraction and scoring ran locally. The report contains metrics only; PDFs,",
        "Ground Truth text, extracted text, and temporary workspace data are excluded.",
        "",
        "CER/WER use NFC and whitespace collapse. Their edit counts and denominators",
        "are shown. Sensitive-item accuracy is exact character/script match.",
        "`item` pools all reference items; `page` is the unweighted mean across pages",
        "that contain at least one reference item. These are intentionally separate.",
        "",
        "`Num` uses a regex over the running-text reference, which excludes table",
        "cells. `Curated` scores every exact numeric value listed in the verified",
        "ground truth, split into running text and table cells; each occurrence is",
        "matched to its own span (no double counting, duplicates kept). `Curated Δ`",
        "is RAW → CLEANED: fixed / regressed / new unexplained numeric tokens.",
        "",
        "| Language | Extractor | Variant | CER edits/ref | WER edits/ref | Num item; page | Curated text; table | Curated Δ fixed/regressed/new | Date item; page | ID item; page | Op item; page | Order tau coverage | Regressed N/D/I/O | New wrong N/D/I/O | English safety |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in sorted(
        records,
        key=lambda value: (
            value["language"],
            value["extractor"],
            value["cer"],
        ),
    ):
        regression = "—"
        if item["language"] == "English" and item["variant"] != "raw":
            raw = raw_by_key[(item["language"], item["extractor"])]
            delta = item["cer"] - raw["cer"]
            regressed = sum(
                item["regressions"][category]["regressed"]
                for category in TOKEN_CATEGORIES
            ) + _curated(item, "regressed")
            introduced = sum(
                item["regressions"][category]["introduced_wrong"]
                for category in TOKEN_CATEGORIES
            ) + _curated(item, "introduced_wrong")
            regression = (
                f"PASS (CER {delta:+.4f}; regressed 0; new wrong 0)"
                if delta <= ENGLISH_REGRESSION_TOLERANCE
                and regressed == 0
                and introduced == 0
                else (
                    f"FAIL (CER {delta:+.4f}; regressed {regressed}; "
                    f"new wrong {introduced})"
                )
            )

        def item_page(prefix: str) -> str:
            correct = item[f"{prefix}_correct"]
            reference = item[f"{prefix}_reference"]
            hypothesis = item[f"{prefix}_hypothesis"]
            item_value = correct / reference if reference else None
            page_value = item.get(
                f"{prefix}_page_mean_accuracy",
                item.get(f"{prefix}_page_weighted_accuracy"),
            )
            pages = item[f"{prefix}_pages"]
            return (
                f"item {correct}/{reference}={finite_metric(item_value)} "
                f"(hyp {hypothesis}); page {finite_metric(page_value)} "
                f"over {pages} pages"
            )

        def vector(field: str) -> str:
            return "/".join(
                str(item["regressions"][category][field])
                for category in TOKEN_CATEGORIES
            )

        def curated_cell() -> str:
            text_ref = item.get("curated_numeric_text_reference", 0)
            table_ref = item.get("curated_numeric_table_reference", 0)
            if not text_ref and not table_ref:
                return "—"
            return (
                f"text {item['curated_numeric_text_correct']}/{text_ref}; "
                f"table {item['curated_numeric_table_correct']}/{table_ref}"
            )

        curated_delta = (
            "—"
            if "curated_numbers" not in item["regressions"]
            else (
                f"{_curated(item, 'fixed')}/{_curated(item, 'regressed')}/"
                f"{_curated(item, 'introduced_wrong')}"
            )
        )
        lines.append(
            f"| {item['language']} | {item['extractor']} | {item['variant']} | "
            f"{item['char_edits']}/{item['ref_chars']}={item['cer']:.4f} | "
            f"{item['word_edits']}/{item['ref_words']}={item['wer']:.4f} | "
            f"{item_page('numeric')} | {curated_cell()} | {curated_delta} | "
            f"{item_page('date')} | "
            f"{item_page('identifier')} | {item_page('operator')} | "
            f"tau {finite_metric(item['reading_order_tau'])}; "
            f"lines {item['reading_order_matched_lines']}/"
            f"{item['reading_order_ref_lines']}; pages "
            f"{item['reading_order_pages_with_tau']}/{item['verified_pages']} | "
            f"{vector('regressed')} | {vector('introduced_wrong')} | "
            f"{regression} |"
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Regression vectors are numbers/dates/identifiers/operators. A variant fails",
        "  the English gate when CER worsens by more than 0.001, any correct sensitive",
        "  item becomes wrong, or any new wrong sensitive item is introduced.",
        "- Small denominators are not generalized: the exact item and page counts above",
        "  are part of every result. A score from one or two items is weak evidence.",
        "- Regex `New wrong N` counts numbers absent from the running-text reference. Because",
        "  that reference excludes table cells, correct table values can appear there; the",
        "  curated columns are authoritative for numeric correctness, including tables.",
        "- Curated `new` counts hypothesis numbers that match no remaining curated value.",
        "  It is a review queue, not proof of a wrong value: a correct number whose curated",
        "  form includes brackets that the extractor split or mirrored, an uncurated",
        "  header value, or an extra occurrence (e.g. a repeated footer year) is counted.",
        "- `digit_run_fix` is producer-specific: on PDFs whose numbers are already in",
        "  logical order it corrupts correct values. It must never be applied automatically.",
        "- `digit_run_fix` and `bidi_reorder` are benchmarked for audit evidence but",
        "  hidden from the normal UI. They appear only after explicitly enabling",
        "  experimental unsafe variants.",
        "- `arabic_fold` is reversible only through the preserved RAW artifact; corrected",
        "  output never overwrites RAW.",
        "- Unmapped/NUL glyphs cannot be reconstructed deterministically and should be",
        "  routed to a later OCR/VLM evaluation.",
        "- Phase E1 evaluates native PDF text layers only. It does not evaluate OCR or",
        "  OpenRouter vision models (Phase E2) and is not a basis for selecting an Arabic",
        "  OCR solution. Results apply only to the documents and pages listed.",
    ]
    return "\n".join(lines) + "\n"


def run(arguments: argparse.Namespace) -> int:
    settings = replace(load_settings(), adapter_timeout_s=max(120, load_settings().adapter_timeout_s))
    manager = WorkspaceManager(settings)
    records = []
    records += evaluate_document(
        manager, arguments.arabic_pdf.resolve(), arguments.arabic_gt.resolve(), "Arabic"
    )
    records += evaluate_document(
        manager,
        arguments.english_pdf.resolve(),
        arguments.english_gt.resolve(),
        "English",
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(_report(records), encoding="utf-8")
    json_path = arguments.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "english_regression_tolerance": ENGLISH_REGRESSION_TOLERANCE,
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote metrics-only report: {arguments.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arabic-pdf", type=Path, required=True)
    parser.add_argument("--arabic-gt", type=Path, required=True)
    parser.add_argument("--english-pdf", type=Path, required=True)
    parser.add_argument("--english-gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
