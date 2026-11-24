from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from lab.app.api_panel import render_api_panel
from lab.app.components import diff_html, text_html
from lab.app.full_pdf_panel import render_full_pdf_view
from lab.core.e2.airgap import (
    ManifestError,
    deployment_record,
    load_manifest,
    preflight_deployment,
    write_preflight_report,
)
from lab.core.e2.backends import check_readiness
from lab.core.e2.hardware import probe_hardware
from lab.core.e2.metrics_e2 import page_extras, repetition_stats, score_tables, summarize_run
from lab.core.e2.render import render_pdf_page
from lab.core.e2.runner import (
    E2Runner,
    E2RunRefused,
    is_e2_run,
    is_mock_run,
    new_run_id,
    save_run,
)
from lab.core.e2.specs import ModelSpec, load_model_specs
from lab.core.e2.store import ResultStore
from lab.core.groundtruth import GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.metrics import (
    UnverifiedGroundTruthError,
    evaluate_page,
    finite_metric,
    scan_curated_numeric_values,
)
from lab.core.models import RunResult
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager

STATUS_TEXT = {
    "ready": "Ready for local inference",
    "not_installed": "Required local packages are not installed",
    "not_downloaded": "Required weights are absent (approval required)",
    "insufficient_hardware": "Unsupported on current hardware",
    "server_unavailable": "Local server unavailable",
    "remote_disabled": "Remote inference disabled",
    "disabled": "Disabled",
    "mock": "MOCK architecture test only",
}


def _mode_label(spec: ModelSpec) -> str:
    if spec.pipeline_mode == "official_full_page":
        if spec.configuration == "experimental":
            return "Official full-page pipeline - EXPERIMENTAL configuration (" + ", ".join(
                f"{key}={value}" for key, value in spec.generation.items()
            ) + ")"
        return "Official full-page pipeline - vendor-default configuration (baseline)"
    return "Element-level prompt on whole page (historical mode)"


def _run_mode(run: RunResult) -> str:
    e2 = run.environment.get("e2") or {}
    mode = e2.get("pipeline_mode") or (e2.get("spec") or {}).get("pipeline_mode")
    if mode == "hosted_api":
        return "hosted API"
    return "official full-page" if mode == "official_full_page" else "element-level"


def _run_label(run: RunResult) -> str:
    if run.adapter.kind == "api":
        prefix = "API"
    else:
        prefix = "MOCK" if is_mock_run(run) else ("E2" if is_e2_run(run) else "E1")
    return f"{prefix} | {_run_mode(run)} | {run.adapter.label} | {run.run_id[-8:]}"


def _load_library_runs(store: ResultStore, doc_sha256: str) -> tuple[list[RunResult], list[dict]]:
    runs: list[RunResult] = []
    incomplete: list[dict] = []
    for record in store.runs():
        directory = Path(record["directory"])
        path = directory / "run.json"
        if not path.is_file():
            if (record.get("started") or {}).get("document_sha256") == doc_sha256:
                incomplete.append(record)
            continue
        try:
            run = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if run.doc_sha256 == doc_sha256:
            runs.append(run)
    return runs, incomplete


def _page_raw(run: RunResult, page: int) -> dict:
    return ((((run.environment.get("e2") or {}).get("page_raw") or {}).get(str(page))) or {})


st.set_page_config(page_title="OCR/VLM Evaluation - Extraction Lab", layout="wide")
settings = load_settings()
manager = WorkspaceManager(settings)
if "workspace_id" not in st.session_state:
    st.warning("Create a workspace on Home first.")
    if st.button("Create a workspace now"):
        st.session_state.workspace_id = manager.create().workspace_id
        st.rerun()
    st.stop()
workspace_id = st.session_state.workspace_id
ingest = IngestService(manager, settings)
gt_store = GroundTruthStore(manager)
store = ResultStore()

st.title("E2 OCR/VLM evaluation (local + explicitly gated hosted API)")
st.caption(
    "Inference receives rendered page pixels only. Ground Truth and text-layer output are "
    "used after inference for scoring and never enter a model. Every manual run is saved "
    "write-once under D:\\agenticdoc_e2\\results."
)
VIEW_SIMPLE = "Full PDF run (hosted API)"
VIEW_ADVANCED = "Advanced: single page, local models, manual review"
# The simple view is the default. Nothing runs when the page opens or a setting changes;
# the advanced view below is the previous page, unchanged.
view = st.radio("View", [VIEW_SIMPLE, VIEW_ADVANCED], horizontal=True, key="e2-view")
if view == VIEW_SIMPLE:
    render_full_pdf_view(
        manager=manager, settings=settings, store=store, ingest=ingest, workspace_id=workspace_id
    )
    st.stop()
if "e2_hardware" not in st.session_state:
    st.session_state.e2_hardware = probe_hardware()
hardware = st.session_state.e2_hardware
with st.expander("Local hardware and package readiness", expanded=False):
    st.text(
        f"CPU: {hardware.cpu} ({hardware.logical_cpus} logical)\n"
        f"RAM: {finite_metric(hardware.ram_total_gb)} GB total; "
        f"{finite_metric(hardware.ram_available_gb)} GB available\n"
        f"CUDA: {hardware.cuda_available}; GPUs: "
        f"{[gpu.get('name') for gpu in hardware.gpus] or 'none detected'}\n"
        f"HF cache: {hardware.hf_cache_dir}\n"
        f"Packages: {json.dumps(hardware.packages)}"
    )

documents = [item for item in ingest.list_documents(workspace_id) if item.kind == "pdf"]
if not documents:
    st.info("Upload a PDF on Documents first.")
    st.stop()
document = st.selectbox(
    "PDF", documents, format_func=lambda item: f"{item.display_name} ({item.page_count} pages)"
)
page_number = int(
    st.number_input("Page", min_value=1, max_value=document.page_count, value=1)
)
source = ingest.original_path(workspace_id, document.doc_id)

# Ground Truth is used only AFTER inference, for scoring; it is never sent to a model or API.
ground_truth = gt_store.load(workspace_id, document.doc_id)
gt_page = (
    next((item for item in ground_truth.pages if item.page == page_number), None)
    if ground_truth
    else None
)
if gt_page is None:
    st.warning("Missing Ground Truth for this page; inference can run, but metrics are unavailable.")
elif gt_page.status != "verified":
    st.warning("Ground Truth is draft; verify it before treating metrics as evaluation evidence.")
location = st.radio("Inference location", ["Local", "Hosted API (experimental)"], horizontal=True)
if location == "Hosted API (experimental)":
    render_api_panel(
        manager=manager, settings=settings, store=store, workspace_id=workspace_id,
        document=document, source=source, page_number=page_number, ground_truth=ground_truth,
    )
else:
    specs = load_model_specs()
    paddle_specs = [spec for spec in specs if spec.id.startswith("paddleocr-vl-1.6")]
    other_specs = [spec for spec in specs if spec not in paddle_specs]
    families = ["PaddleOCR-VL 1.6"] + [spec.label for spec in other_specs]
    family = st.selectbox("Model", families)
    if family == "PaddleOCR-VL 1.6":
        mode_names = [_mode_label(spec) for spec in paddle_specs]
        mode = st.selectbox("Inference mode", mode_names)
        spec = next(item for item in paddle_specs if _mode_label(item) == mode)
    else:
        spec = next(item for item in other_specs if item.label == family)
        st.selectbox("Inference mode", ["Element-level / model-native mode"], disabled=True)

    readiness = check_readiness(spec, hardware)
    label = STATUS_TEXT.get(readiness.status, readiness.status)
    (st.success if readiness.runnable else st.warning)(f"{spec.label}: {label}")
    st.text(
        f"Mode: {_mode_label(spec)}\n"
        f"VL model: {spec.pinned_ref}\n"
        + (
            f"Layout model: {spec.layout_repo_id}@{spec.layout_revision}\n"
            if spec.layout_repo_id
            else ""
        )
        + f"Backend: {spec.backend}; license: {spec.license}\n"
        + "\n".join(f"- {reason}" for reason in readiness.reasons)
    )
    for step in readiness.next_steps:
        st.caption(f"Blocked next step: {step}")

    try:
        airgap_manifest = load_manifest()
        airgap_record = deployment_record(airgap_manifest, spec.id)
    except ManifestError as exc:
        st.error(f"Air-gap manifest is invalid: {exc}")
        airgap_manifest = None
        airgap_record = None

    if airgap_record is not None:
        component_index = {
            item["component_id"]: item for item in airgap_manifest["components"]
        }
        supporting = [
            {
                "component": component_index[item]["name"],
                "revision": component_index[item]["revision"],
                "status": component_index[item]["declared_status"],
            }
            for item in airgap_record["components"]
        ]
        historical = sum(
            1
            for item in store.runs()
            if (item.get("started") or {}).get("adapter_id") == f"e2.{spec.id}"
        )
        with st.expander("Exact model version and offline deployment readiness", expanded=True):
            st.text(
                f"Deployment: {airgap_record['deployment_id']}\n"
                f"Primary revision: {spec.repo_id}@{spec.revision}\n"
                f"Variant / quantization: {airgap_record['variant']} / "
                f"{airgap_record['quantization']}\n"
                f"Pipeline: {airgap_record['pipeline_mode']}\n"
                f"Declared air-gap status: {airgap_record['deployment_status']}\n"
                f"Saved finalized run records: {historical}"
            )
            st.caption("Supporting components")
            st.json(supporting, expanded=False)
            if airgap_record["missing_or_unverified"]:
                st.warning(
                    "Missing or unverified:\n- "
                    + "\n- ".join(airgap_record["missing_or_unverified"])
                )
            if st.button("Run offline preflight (no downloads)", key=f"preflight-{spec.id}"):
                with st.spinner("Hashing declared local artifacts and checking runtime..."):
                    report = preflight_deployment(
                        airgap_manifest, spec.id, environment="local", hardware=hardware
                    )
                    saved_report = write_preflight_report(report, store.root / "reports")
                (st.success if report["ready_for_airgap"] else st.warning)(
                    f"Preflight status: {report['status']}; saved write-once report at "
                    f"{saved_report}"
                )
                st.json(report, expanded=False)

    scope = st.radio("Inference scope", ["Selected page", "All pages"], horizontal=True)
    runner = E2Runner(manager, settings, hardware=hardware)
    run_button = st.button("Run local inference", disabled=not readiness.runnable)
    if run_button:
        try:
            inference_preflight = runner.verified_preflight(spec)
        except E2RunRefused as exc:
            st.error(str(exc))
            st.stop()
        pages = [page_number] if scope == "Selected page" else list(range(1, document.page_count + 1))
        run_id = new_run_id()
        handle = store.begin(
            adapter_id=f"e2.{spec.id}",
            revision=spec.storage_revision,
            run_id=run_id,
            started={
                "language": "Manual",
                "adapter_id": f"e2.{spec.id}",
                "label": spec.label,
                "pipeline_mode": spec.pipeline_mode,
                "pages": pages,
                "spec": spec.model_dump(mode="json"),
                "readiness": readiness.to_dict(),
                "preflight": inference_preflight,
                "hardware": hardware.to_dict(),
                "document_sha256": document.sha256,
                "ground_truth_version": ground_truth.version if ground_truth else None,
            },
        )
        progress_bar = st.progress(0.0, text="Loading local models...")

        def update_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Processed {done}/{total} page(s)")

        try:
            run = runner.run(
                workspace_id,
                document,
                source,
                spec,
                pages,
                progress=update_progress,
                run_id=run_id,
                on_page=handle.write_page,
                preflight_report=inference_preflight,
            )
        except E2RunRefused as exc:
            st.error(str(exc))
        except Exception as exc:  # incomplete write-once directory remains visible for diagnosis
            st.error(f"Inference failed; the incomplete run was preserved: {type(exc).__name__}: {exc}")
        else:
            save_run(manager, workspace_id, run)
            summaries: list[dict] = []
            if ground_truth is not None:
                try:
                    summaries.append(summarize_run(run, ground_truth))
                except UnverifiedGroundTruthError:
                    pass
            terminal = {"ok": "completed", "partial": "partial"}.get(run.status, "failed")
            saved = handle.finalize(
                run,
                summaries=summaries,
                language="Manual",
                status=terminal,
                notes=["Manual Streamlit run; final user acceptance remains pending."],
            )
            st.session_state.e2_selected_run = run.run_id
            st.success(f"Saved write-once run {run.run_id} at {saved}")

rendered = render_pdf_page(
    source,
    page_number,
    manager.path(workspace_id) / "e2_images" / document.doc_id,
    dpi=settings.e2_render_dpi,
)

workspace_runs = [
    RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    for path in sorted((manager.path(workspace_id) / "runs").glob("run_*.json"))
]
library_runs, incomplete_runs = _load_library_runs(store, document.sha256)
by_id = {run.run_id: run for run in workspace_runs + library_runs}
doc_runs = [
    run
    for run in by_id.values()
    if run.doc_id == document.doc_id or run.doc_sha256 == document.sha256
    if run.raw_run_id is None and any(page.page == page_number for page in run.pages)
]
doc_runs.sort(key=lambda run: run.started_at)
if incomplete_runs:
    st.warning(f"{len(incomplete_runs)} incomplete saved run(s) exist for this PDF; no data was deleted.")

if not doc_runs:
    st.subheader("Original page")
    st.image(rendered.path, width="stretch")
    st.info("No saved run contains this page yet.")
    st.stop()

default_id = st.session_state.get("e2_selected_run")
primary_index = next(
    (index for index, run in enumerate(doc_runs) if run.run_id == default_id), len(doc_runs) - 1
)
primary = st.selectbox(
    "Primary run for visual inspection", doc_runs, index=primary_index, format_func=_run_label
)
primary_page = next(page for page in primary.pages if page.page == page_number)
primary_raw = _page_raw(primary, page_number)

left, right = st.columns(2)
with left:
    st.subheader("Original rendered PDF page")
    st.image(rendered.path, width="stretch")
    st.caption(
        f"{rendered.width}x{rendered.height}px at {rendered.dpi} DPI | "
        f"sha256 {rendered.sha256[:16]}..."
    )
with right:
    st.subheader("Exact RAW output")
    st.caption(
        f"{_run_label(primary)} | status {primary_page.status} | "
        f"{finite_metric(primary_page.elapsed_ms)} ms"
    )
    api_record = primary.environment.get("api")
    if api_record:
        identity = api_record.get("identity") or {}
        cost = api_record.get("cost") or {}
        st.warning(
            f"Hosted API result from {identity.get('provider')} ({identity.get('api_model_id')}). "
            f"Classification: {identity.get('classification')}; exact revision verified: "
            f"{identity.get('exact_revision_verified')}. Cost: {cost.get('usd')} USD "
            f"({cost.get('source')}: 'response' = reported by the provider, 'estimated' = "
            "from the registry's published price)."
        )
        with st.expander("Hosted API identity, transmitted content and exact exchanges (secrets removed)"):
            st.json(
                {
                    "identity": identity,
                    "transmission": api_record.get("transmission"),
                    "timings_ms": api_record.get("timings_ms"),
                    "exchanges": api_record.get("exchanges"),
                },
                expanded=False,
            )
    if primary_page.error:
        st.error(primary_page.error)
    run_errors = (primary.environment.get("e2") or {}).get("errors") or []
    if run_errors:
        blocked = [
            block
            for item in run_errors
            for block in item.get("application_control_blocks") or []
        ]
        if blocked:
            st.error(
                "Root cause evidence: Windows application control (Smart App Control / App "
                "Control policy) blocked native module(s) of this environment during the run: "
                + ", ".join(item["file"] for item in blocked)
                + ". This is not a memory or model error. Do not disable security controls; "
                "see docs/local/IT_APPLICATION_CONTROL_REQUEST.md."
            )
        with st.expander(f"Failure details ({len(run_errors)} recorded error(s))", expanded=True):
            for item in run_errors:
                st.markdown(
                    f"**{item.get('code')}**"
                    + (f" · {item.get('type')}" if item.get("type") else "")
                    + (f" · page {item.get('page')}" if item.get("page") else "")
                )
                st.text(str(item.get("message") or ""))
                if item.get("traceback"):
                    st.code(item["traceback"], language="text")
                if item.get("application_control_blocks"):
                    st.json(item["application_control_blocks"], expanded=False)
    st.markdown(text_html(primary_page.text, show_invisibles=True), unsafe_allow_html=True)
    response = primary_raw.get("response")
    if isinstance(response, dict) and response.get("truncated_at_max_new_tokens"):
        st.error("Truncation warning: generation reached max_new_tokens.")
    elif isinstance(response, dict) and response.get("truncation_status", "").startswith("unknown"):
        st.warning("Truncation status is unknown because the backend did not report token termination.")
    repetition = repetition_stats(primary_page.text)
    if repetition["flagged"]:
        st.error(
            "Repetition warning: output-loop heuristic flagged this page "
            f"(max identical-token run {repetition['max_consecutive_identical_tokens']}; "
            f"unique-token ratio {repetition['unique_token_ratio']:.3f})."
        )
    if response is not None:
        with st.expander("Exact structured/raw intermediate output", expanded=False):
            st.json(response, expanded=False)

st.subheader("Manual user review (separate from automated metrics)")
st.caption(
    "Each submission is append-only and linked to this exact run, page, model revision, "
    "and inference configuration. Submitting a new review never replaces an earlier one."
)
review_categories = [
    "numeric_error",
    "table_error",
    "reading_order_error",
    "hallucination",
    "missing_text",
    "arabic_error",
    "english_error",
    "truncation_or_repetition",
    "other",
]
with st.form(f"manual-review-{primary.run_id}-{page_number}", clear_on_submit=True):
    review_decision = st.selectbox("Decision", ["NEEDS_REVIEW", "PASS", "FAIL"])
    review_errors = st.multiselect("Observed error categories", review_categories)
    review_notes = st.text_area("Review notes", placeholder="Describe what you verified on the page.")
    submit_review = st.form_submit_button("Save manual review locally")
if submit_review:
    e2_environment = primary.environment.get("e2") or {}
    image_record = next(
        (
            item
            for item in e2_environment.get("images", [])
            if item.get("page") == page_number
        ),
        {},
    )
    review_path = store.write_manual_review(
        {
            "run_id": primary.run_id,
            "page": page_number,
            "decision": review_decision,
            "notes": review_notes,
            "error_categories": review_errors,
            "adapter_id": primary.adapter.adapter_id,
            "model_revision": primary.adapter.model_id or "UNVERIFIED",
            # Full identity: a full-page run also depends on the layout model, and an
            # adapter on its base model; both are recorded, never inferred later.
            "model_identity": {
                "primary": primary.adapter.model_id,
                "layout_model": primary.adapter.params.get("layout_model"),
                "base_model": primary.adapter.params.get("base_model"),
                "library": primary.adapter.library,
                "library_version": primary.adapter.library_version,
            },
            "pipeline_mode": e2_environment.get("pipeline_mode"),
            "inference_config": e2_environment.get("inference_config")
            or primary.adapter.params.get("generation", {}),
            "rendered_page_sha256": image_record.get("sha256"),
            "raw_page_status": primary_page.status,
        }
    )
    st.success(f"Saved append-only manual review at {review_path}")

historical_reviews = store.manual_reviews(run_id=primary.run_id, page=page_number)
with st.expander(f"Historical manual reviews ({len(historical_reviews)})", expanded=False):
    if historical_reviews:
        st.json(historical_reviews, expanded=False)
    else:
        st.caption("No manual review has been submitted for this exact run and page.")

selected = st.multiselect(
    "Runs to compare without overwriting history",
    doc_runs,
    default=doc_runs[-min(3, len(doc_runs)) :],
    format_func=_run_label,
)
if any(is_mock_run(run) for run in selected):
    st.error("MOCK runs are architecture tests, not evaluation results.")

for run in selected:
    page = next(item for item in run.pages if item.page == page_number)
    st.markdown(f"### {_run_label(run)}")
    if gt_page is None or gt_page.status != "verified":
        st.markdown(text_html(page.text, show_invisibles=True), unsafe_allow_html=True)
        continue
    metric = evaluate_page(run.run_id, run.variant, page, gt_page, ground_truth.version)
    extras = page_extras(gt_page, page.text)
    tables = score_tables(gt_page.tables, page.text)
    scan = scan_curated_numeric_values(gt_page.numeric_values, page.text)
    missing = [value.text for value, found in zip(gt_page.numeric_values, scan.found) if not found]
    st.text(
        f"CER {metric.char_edits}/{metric.ref_chars}={metric.cer:.4f}; "
        f"WER {metric.word_edits}/{metric.ref_words}={metric.wer:.4f}\n"
        f"Numbers {metric.num_correct}/{metric.num_ref_count}; dates "
        f"{metric.date_correct}/{metric.date_ref_count}; identifiers "
        f"{metric.identifier_correct}/{metric.identifier_ref_count}; operators "
        f"{metric.operator_correct}/{metric.operator_ref_count}\n"
        f"Curated numeric text {metric.curated_num_text_correct}/{metric.curated_num_text_ref}; "
        f"tables {metric.curated_num_table_correct}/{metric.curated_num_table_ref}\n"
        f"Table cells {tables.cells_exact}/{tables.cells_reference}; row/column association "
        f"{tables.data_cells_associated}/{tables.data_cells_reference}; "
        f"parsed tables {tables.tables_detected}\n"
        f"Mixed RTL/LTR exact {extras.mixed_lines_exact}/{extras.mixed_lines_reference}; "
        f"missing words {extras.missing_words}/{extras.reference_words}; extra {extras.extra_words}\n"
        f"Reading-order tau {finite_metric(metric.reading_order_tau)}"
    )
    st.text(
        f"Missing curated values ({len(missing)}): {missing}\n"
        f"Unmatched output numbers: {dict(scan.unmatched_hypothesis)}"
    )
    with st.expander("Ground Truth -> RAW diff"):
        st.markdown(diff_html(gt_page.text, page.text), unsafe_allow_html=True)

st.caption(
    "Manual acceptance is required: confirm the original page, exact RAW output, layout order, "
    "Arabic/English text, tables, numbers, warnings, and the new write-once D: run before approval."
)
