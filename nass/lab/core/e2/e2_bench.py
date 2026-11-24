"""E2-A local comparison: E1 text-layer baselines vs. OCR/VLM models.

Executes only models named with --models (explicit authorization) whose readiness
is `ready`; every other model is listed with its status and never scored. Results
persist write-once in the evaluation library (see lab.core.e2.store). Mock models run only with
--include-mock and are labelled as architecture tests, never as results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab.core.adapters.postprocess import available_variants, derive_variant
from lab.core.adapters.registry import AdapterRegistry
from lab.core.e2.backends import Readiness
from lab.core.e2.hardware import probe_hardware
from lab.core.e2.library import build_comparison
from lab.core.e2.metrics_e2 import derive_markup_strip, summarize_run
from lab.core.e2.runner import E2Runner, E2RunRefused, new_run_id, save_run
from lab.core.e2.store import ResultStore, results_root
from lab.core.e2.specs import ModelSpec, load_model_specs, mock_spec
from lab.core.groundtruth import GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.metrics import finite_metric
from lab.core.runner import LabRunner
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager

BASELINE_ADAPTERS = ("text.pymupdf", "text.pypdfium2")


def _kind_for(variant: str, safe: set[str]) -> str:
    if variant == "raw":
        return "E1 RAW"
    return "E1 safe variant" if variant in safe else "E1 EXPERIMENTAL variant"


def _store_run(
    store: ResultStore | None,
    handle,
    raw,
    *,
    derived: list,
    summaries: list[dict],
    language: str,
    notes: list[str] | None = None,
) -> str | None:
    if store is None:
        return None
    if handle is None:
        handle = store.begin(
            adapter_id=raw.adapter.adapter_id,
            revision=raw.adapter.library_version or "unknown",
            run_id=raw.run_id,
            started={
                "language": language,
                "adapter_id": raw.adapter.adapter_id,
                "label": raw.adapter.label,
                "kind": "E1 text-layer baseline",
                "pages": [page.page for page in raw.pages],
                "environment": raw.environment,
            },
        )
        for page in raw.pages:
            handle.write_page(page.page, page.model_dump(mode="json"))
    status = {
        "ok": "completed",
        "partial": "partial",
        "error": "failed",
        "timeout": "failed",
    }.get(raw.status, raw.status)
    return str(
        handle.finalize(
            raw, derived=derived, summaries=summaries, language=language, status=status, notes=notes
        )
    )


def evaluate_language(
    manager: WorkspaceManager,
    pdf_path: Path,
    gt_path: Path,
    language: str,
    specs: list[ModelSpec],
    runner: E2Runner,
    *,
    run_ids: set[str] | None = None,
    pages: list[int] | None = None,
    allow_mock: bool = False,
    store: ResultStore | None = None,
    hardware: dict | None = None,
) -> dict:
    """Score baselines and models on identical pages.

    Only specs whose id is in run_ids are executed (explicit authorization);
    every other spec is listed with its readiness and never scored.
    """
    run_ids = run_ids or set()
    workspace = manager.create()
    try:
        settings = manager.settings
        ingest = IngestService(manager, settings)
        registry = AdapterRegistry()
        lab_runner = LabRunner(manager, settings, registry)
        with pdf_path.open("rb") as handle:
            document = ingest.ingest(workspace.workspace_id, handle, pdf_path.name)
        ground_truth = GroundTruthStore(manager).import_json(
            workspace.workspace_id,
            document,
            gt_path.read_bytes(),
            verified=True,
            reviewer="benchmark-source-validated",
        )
        selected = pages or list(range(1, document.page_count + 1))
        source = ingest.original_path(workspace.workspace_id, document.doc_id)
        rows: list[dict] = []
        stored: list[str] = []
        for adapter_id in BASELINE_ADAPTERS:
            entry = registry.get(adapter_id)
            if not entry.info.available:
                continue
            raw = lab_runner.run_local(
                workspace.workspace_id, document, source, adapter_id, selected, bypass_cache=True
            )
            snapshot = raw.model_dump_json()
            safe = set(available_variants(adapter_id))
            variants = ["raw"] + [
                item
                for item in available_variants(adapter_id, include_experimental=True)
                if item not in {"raw", "unmapped_marker"}
            ]
            derived_runs = []
            baseline_rows = []
            for variant in variants:
                run = raw if variant == "raw" else derive_variant(raw, variant)
                if variant != "raw":
                    derived_runs.append(run)
                summary = summarize_run(run, ground_truth, table_source=raw)
                summary["kind"] = _kind_for(variant, safe)
                baseline_rows.append(summary)
                if raw.model_dump_json() != snapshot:
                    raise AssertionError("A correction mutated the RAW run")
            rows.extend(baseline_rows)
            path = _store_run(
                store, None, raw, derived=derived_runs, summaries=baseline_rows, language=language
            )
            if path:
                stored.append(path)
        not_run: list[dict] = []
        for spec in specs:
            readiness = runner.readiness(spec)
            if spec.id not in run_ids:
                not_run.append(
                    readiness.to_dict()
                    | {"label": spec.label, "authorization": "not authorized for this run"}
                )
                continue
            if not readiness.runnable:
                not_run.append(readiness.to_dict() | {"label": spec.label})
                continue
            try:
                inference_preflight = runner.verified_preflight(spec)
            except E2RunRefused as exc:
                not_run.append(
                    (exc.readiness or Readiness(spec.id, "disabled")).to_dict()
                    | {"label": spec.label, "refused": str(exc)}
                )
                continue
            run_id = new_run_id()
            handle = None
            if store is not None and not spec.is_mock:
                handle = store.begin(
                    adapter_id=f"e2.{spec.id}",
                    revision=spec.storage_revision,
                    run_id=run_id,
                    started={
                        "language": language,
                        "adapter_id": f"e2.{spec.id}",
                        "label": spec.label,
                        "kind": "E2 model",
                        "pages": selected,
                        "spec": spec.model_dump(mode="json"),
                        "readiness": readiness.to_dict(),
                        "preflight": inference_preflight,
                        "hardware": hardware,
                        "document_sha256": document.sha256,
                        "ground_truth_version": ground_truth.version,
                    },
                )

            def save_page(page: int, record: dict, handle=handle) -> None:
                if handle is not None:
                    handle.write_page(page, record)

            try:
                raw = runner.run(
                    workspace.workspace_id,
                    document,
                    source,
                    spec,
                    selected,
                    allow_mock=allow_mock,
                    run_id=run_id,
                    on_page=save_page,
                    preflight_report=inference_preflight,
                )
            except E2RunRefused as exc:
                not_run.append(
                    (exc.readiness or Readiness(spec.id, "disabled")).to_dict()
                    | {"label": spec.label, "refused": str(exc)}
                )
                continue
            save_run(manager, workspace.workspace_id, raw)
            snapshot = raw.model_dump_json()
            kind = "MOCK architecture test" if spec.is_mock else "E2 model RAW"
            stripped = derive_markup_strip(raw)
            model_rows = [
                summarize_run(raw, ground_truth) | {"kind": kind},
                summarize_run(stripped, ground_truth, table_source=raw)
                | {"kind": "MOCK post-processed" if spec.is_mock else "E2 post-processed"},
            ]
            if raw.model_dump_json() != snapshot:
                raise AssertionError("Post-processing mutated the RAW model run")
            rows.extend(model_rows)
            path = _store_run(
                store,
                handle,
                raw,
                derived=[stripped],
                summaries=model_rows,
                language=language,
                notes=[spec.notes] if spec.notes else None,
            )
            if path:
                stored.append(path)
        return {
            "language": language,
            "pages": selected,
            "verified_pages": sum(
                page.status == "verified" and page.page in selected for page in ground_truth.pages
            ),
            "rows": rows,
            "not_run": not_run,
            "stored_runs": stored,
        }
    finally:
        manager.end(workspace.workspace_id)


def _ratio(correct: int, total: int) -> str:
    return f"{correct}/{total}={finite_metric(correct / total if total else None)}"


def _row(item: dict) -> str:
    metric = item["metrics"]
    tables = item["tables"]
    mixed = item["mixed_direction"]
    words = item["words"]
    timing = item["timing"]
    memory = item["memory"]
    quality = item.get("quality") or {}
    mean = timing.get("page_ms_mean")
    peak = memory.get("peak_rss_mb")
    return (
        f"| {item['label']} | {item['variant']} | {item['kind']} | "
        f"{metric['char_edits']}/{metric['ref_chars']}={metric['cer']:.4f} | "
        f"{metric['word_edits']}/{metric['ref_words']}={metric['wer']:.4f} | "
        f"{_ratio(metric['num_correct'], metric['num_ref_count'])} | "
        f"text {metric['curated_num_text_correct']}/{metric['curated_num_text_ref']}; "
        f"table {metric['curated_num_table_correct']}/{metric['curated_num_table_ref']} | "
        f"{_ratio(metric['date_correct'], metric['date_ref_count'])} | "
        f"{_ratio(metric['identifier_correct'], metric['identifier_ref_count'])} | "
        f"{_ratio(metric['operator_correct'], metric['operator_ref_count'])} | "
        f"{tables['cells_exact']}/{tables['cells_reference']} | "
        f"{tables['data_cells_associated']}/{tables['data_cells_reference']} "
        f"({tables['tables_detected']} parsed) | "
        f"{mixed['exact']}/{mixed['reference']}; edits {mixed['char_edits']}/{mixed['ref_chars']} | "
        f"{words['missing']}/{words['reference']} missing; {words['extra']} extra | "
        f"tau {finite_metric(metric['reading_order_tau'])} "
        f"({metric['reading_order_matched_lines']}/{metric['reading_order_ref_lines']}) | "
        f"repeat {len(quality.get('repetition_flagged_pages', []))}/"
        f"{quality.get('pages_checked', item.get('pages_total', 0))}; trunc "
        f"{len(quality.get('truncated_pages', []))}/"
        f"{quality.get('truncation_known_pages', 0) or 'n/a'} | "
        f"{'—' if mean is None else f'{mean:.0f} ms/page'} | "
        f"{'—' if peak is None else f'{peak:.0f} MB'} |"
    )


def render_report(results: list[dict], hardware: dict) -> str:
    lines = [
        "# E2-A Local OCR/VLM Comparison Report",
        "",
        "All processing ran locally. The report contains metrics only; PDFs, page",
        "images, ground truth and extracted text are excluded.",
        "",
        "Models receive only a rendered page image and a static prompt template.",
        "Ground truth and previously extracted text are never inference inputs.",
        "",
        f"Hardware: {hardware.get('cpu')}, {hardware.get('logical_cpus')} logical CPUs, "
        f"RAM {finite_metric(hardware.get('ram_total_gb'))} GB, "
        f"CUDA {'yes' if hardware.get('cuda_available') else 'no'}, "
        f"GPUs {[gpu.get('name') for gpu in hardware.get('gpus', [])] or 'none detected'}.",
        "",
        "Definitions follow E1-B (exact digit script, pooled counts with denominators).",
        "`Table cells exact` = every reference table cell found as an exact standalone",
        "span; `Table association` = data cells found in a parsed output table under the",
        "correct row header and column. Text-layer baselines carry no table structure, so",
        "their association is 0 by construction. `E2 post-processed` = `markup_strip`",
        "(Markdown/HTML markup removed), a separate run; RAW model output is unchanged.",
        "",
    ]
    for result in results:
        lines += [
            f"## {result['language']} — pages {', '.join(map(str, result['pages']))} "
            f"({result['verified_pages']} verified)",
            "",
            "| Source | Variant | Kind | CER edits/ref | WER edits/ref | Num item | Curated num | "
            "Date | ID | Operator | Table cells exact | Table association | Mixed RTL/LTR lines exact; "
            "edits | Words missing; extra | Reading order | Repetition / truncation | Time | Peak RAM |",
            "|" + "---|" * 18,
        ]
        lines += [_row(item) for item in result["rows"]]
        lines.append("")
        if result["not_run"]:
            lines += ["Models not run (no metrics are claimed for them):", ""]
            for item in result["not_run"]:
                lines.append(
                    f"- **{item['label']}** — `{item['status']}`: " + " ".join(item["reasons"])
                )
                for step in item.get("next_steps", []):
                    lines.append(f"  - Next step: {step}")
            lines.append("")
    if any(row["is_mock"] for result in results for row in result["rows"]):
        lines += [
            "> **MOCK rows are architecture tests only. They are not model inference and",
            "> must not be read as accuracy results.**",
            "",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arabic-pdf", type=Path, required=True)
    parser.add_argument("--arabic-gt", type=Path, required=True)
    parser.add_argument("--english-pdf", type=Path, required=True)
    parser.add_argument("--english-gt", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="Also write a metrics-only report here")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model ids AUTHORIZED to execute in this run (default: none; all are listed)",
    )
    parser.add_argument("--languages", nargs="*", default=["Arabic", "English"])
    parser.add_argument("--pages", nargs="*", type=int)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--no-store", action="store_true", help="Do not persist results")
    parser.add_argument(
        "--include-mock",
        action="store_true",
        help="Add a MOCK echo model to validate the architecture (not inference).",
    )
    args = parser.parse_args(argv)
    specs = load_model_specs()
    known = {spec.id for spec in specs}
    unknown = set(args.models) - known
    if unknown:
        parser.error(f"Unknown model ids: {sorted(unknown)}")
    run_ids = set(args.models)
    if args.include_mock:
        mock = mock_spec()
        specs.append(mock)
        run_ids.add(mock.id)
    store = None if args.no_store else ResultStore(args.results_root or results_root())
    settings = load_settings()
    manager = WorkspaceManager(settings)
    hardware = probe_hardware()
    runner = E2Runner(manager, settings, hardware=hardware)
    inputs = {
        "Arabic": (args.arabic_pdf, args.arabic_gt),
        "English": (args.english_pdf, args.english_gt),
    }
    results = [
        evaluate_language(
            manager, inputs[language][0], inputs[language][1], language, specs, runner,
            run_ids=run_ids, pages=args.pages, allow_mock=args.include_mock,
            store=store, hardware=hardware.to_dict(),
        )
        for language in args.languages
    ]
    report = render_report(results, hardware.to_dict())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps(
                {"schema_version": 2, "hardware": hardware.to_dict(), "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote metrics-only report: {args.out}")
    if store is not None:
        print(f"Stored run report: {store.write_report('e2_bench', report)}")
        library = build_comparison(store, load_model_specs(), runner)
        print(f"Updated comparison library: {store.write_report('comparison', library)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
