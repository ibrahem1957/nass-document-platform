"""Hybrid PDF Extraction: libraries first, OCR/VLM fallback per page, one folder per run.

Upload PDF -> Select Fallback Model -> API Key / Local Connection -> Analyze PDF ->
Review Plan -> Approve & Run -> Download ZIP.

Nothing runs when the page opens or a setting changes. "Analyze PDF" is local only
(no model, no network). Pages go to a model only after the explicit approval.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from lab.app import full_pdf_panel as panel
from lab.app.components import text_html
from lab.app.hybrid_phase2_panel import render_phase2
from lab.core.e2.api import full_pdf
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused
from lab.core.e2.render import render_pdf_page
from lab.core.e2.store import ResultStore
from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import enrich as hybrid_enrich
from lab.core.hybrid import layout as hybrid_layout
from lab.core.hybrid import qari
from lab.core.hybrid import run as hybrid
from lab.core.ingest import IngestService
from lab.core.settings import load_settings
from lab.core.workspace import WorkspaceManager
from lab.local_pdf_extraction import diagnostics

OXEN_WARNING = ("Earlier AgenticDoc tests of Qwen3-VL via Oxen on our documents showed incomplete pages, "
                "repetition and wrong numbers. It stays available; review its output carefully.")
_STATUS_COLOR = {hybrid.ACCEPTED: "#1a7f37", hybrid.NEEDS_REVIEW: "#9a6700", hybrid.UNCERTAIN: "#bc4c00",
                 hybrid.FAILED: "#cf222e"}

st.set_page_config(page_title="Hybrid PDF Extraction - Extraction Lab", layout="wide")
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
store = ResultStore()
specs = {spec.id: spec for spec in load_providers() if spec.id in {hybrid.OXEN, hybrid.BAIDU} and spec.runnable}

st.title("Hybrid PDF Extraction")
st.markdown("**Upload PDF → Select Fallback Model → API Key / Local Connection → Analyze PDF → Review Plan → "
            "Approve & Run → Download ZIP**")
st.caption("Text comes from the PDF text layer first (PyMuPDF raw + processed, find_tables()). Only pages that need "
           "it, and that you approve, go to the fallback model. Raw outputs are saved unchanged; the final text per "
           f"page is a derived result with its source and reason. Runs are saved write-once under "
           f"{hybrid.hybrid_root(store.root)}.")
flash = st.session_state.pop("hy-flash", None)
if flash:
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[flash[0]](flash[1])


def _status_badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "#57606a")
    return f"<span style='background:{color};color:#fff;border-radius:8px;padding:0 8px'>{panel._esc(status)}</span>"


# ---------------------------------------------------------------- 1-3 inputs

MODE = st.radio("Mode", ["Automatic Extraction", "Benchmark Mode"], horizontal=True, key="hy-mode",
                help="Automatic Extraction: one button from PDF to a validated document (Qari only). "
                     "Benchmark Mode: the existing step-by-step workflow, unchanged.")
document = panel._choose_document(ingest, workspace_id)
if document is None:
    st.stop()
source = ingest.original_path(workspace_id, document.doc_id)
if MODE == "Automatic Extraction":
    from lab.app.auto_panel import render_automatic  # noqa: E402

    render_automatic(document=document, source=source)
    st.stop()

st.subheader("2. Fallback model")
options = [hybrid.QARI, hybrid.BAIDU, hybrid.OXEN, hybrid.LIBRARY_ONLY]
model = st.radio("Fallback model (your choice; it is never changed automatically)", options,
                 format_func=lambda key: hybrid.MODEL_LABELS[key], key="hy-model")
spec = specs.get(model)
if model in {hybrid.OXEN, hybrid.BAIDU} and spec is None:
    st.error(f"{hybrid.MODEL_LABELS[model]} is not implemented in providers_e2.toml.")
if model == hybrid.OXEN:
    st.warning(OXEN_WARNING)

st.subheader("3. API key / local connection")
credential = ""
qari_ready = False
qari_url = ""
qari_token = ""
if model == hybrid.QARI:
    st.caption(f"Model {qari.QARI_MODEL} | DPI {qari.QARI_DPI} | max_new_tokens {qari.QARI_MAX_NEW_TOKENS} | "
               "do_sample=False | prompt 'Free OCR.' | OpenAI-compatible server (e.g. vLLM) on a local GPU or RunPod.")
    qari_url = st.text_input("Qari server URL (http://localhost:8000 or https://<pod>-8000.proxy.runpod.net)",
                             key="hy-qari-url")
    token_store = st.session_state.setdefault("hy_qari_tokens", {})
    typed = st.text_input("Server token (optional; kept only in this session's memory)", type="password",
                          key="hy-qari-token-input")
    if typed:
        token_store[qari_url] = typed
    qari_token = token_store.get(qari_url, "")
    served_model = st.text_input("Model name served by the server", value=qari.QARI_MODEL, key="hy-qari-model")
    statuses = st.session_state.setdefault("hy_qari_status", {})
    if st.button("Check connection (lists models only; no inference)", key="hy-qari-check", disabled=not qari_url):
        statuses[(qari_url, served_model)] = qari.check_ready(qari_url, qari_token or None, model=served_model)
    status = statuses.get((qari_url, served_model))
    if status is None:
        st.info("Qari status: not checked. Qari is not assumed to be installed or running.")
    elif status["ready"]:
        qari_ready = True
        st.success(f"Qari status: Ready at {status['endpoint']} (model listed; no page was sent).")
    else:
        st.error(f"Qari status: Unavailable - {status['error']}")
elif model in {hybrid.OXEN, hybrid.BAIDU} and spec is not None:
    credential = panel._api_key(spec)
else:
    st.caption("Library Only: no model, no key, no network.")

st.subheader("Ground truth (optional, evaluation only)")
st.caption("UTF-8 .txt; `=== page N ===` separators for per-page scores, or one file per page named with its "
           "number. Used only after processing to compute CER/WER; never sent and never used for decisions.")
gt_files = st.file_uploader("Ground truth file(s)", type=["txt"], accept_multiple_files=True, key="hy-gt")
gt_scope = st.radio("What does this ground truth cover?", list(hybrid.GT_SCOPES),
                    format_func=lambda key: hybrid.GT_SCOPES[key], key="hy-gt-scope",
                    help="Scores are reported only for the scope the reference covers. A text-layer score is never "
                         "an overall extraction accuracy.")
ground_truth = None
ground_truth_line = "Ground truth: not attached. CER/WER will NOT be computed for the next run."
if gt_files:
    names = ", ".join(f.name for f in gt_files)
    try:
        gt_text = diagnostics.combine_ground_truth_files([(f.name, f.getvalue()) for f in gt_files])
        described = diagnostics.describe_ground_truth(gt_text, document.page_count)
        if described["level"] is None:
            raise ValueError("the file is empty")
        ground_truth = (gt_text, [f.name for f in gt_files])
        inside = [page for page in described["pages"] if page not in described["outside_document"]]
        coverage = (f"{len(inside)} of {document.page_count} page(s) (pages {inside})" if described["level"] == "page"
                    else f"document level (all {document.page_count} pages compared as one text)")
        ground_truth_line = f"Ground truth: attached - {names} - covers {coverage}. CER/WER will be computed."
        st.success(ground_truth_line + (f" Pages {described['outside_document']} are outside this PDF and ignored."
                                        if described["outside_document"] else ""))
    except ValueError as exc:
        ground_truth_line = f"Ground truth: {names} is NOT usable ({exc}); it will not be attached."
        st.error(ground_truth_line)
else:
    st.info(ground_truth_line)

# ---------------------------------------------------------------- 4 analyze

st.subheader("4. Analyze PDF (local libraries only)")
analyses = st.session_state.setdefault("hy_analyses", {})
if st.button("Analyze PDF", key="hy-analyze"):
    bar = st.progress(0.0, text="Analyzing page 1...")
    try:
        analyses[document.sha256] = hybrid_analysis.analyze_pdf(
            source, on_page=lambda page, total: bar.progress(page / total, text=f"Analyzed page {page}/{total}"))
    except Exception as exc:  # noqa: BLE001 - shown, nothing else happens
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
analysis = analyses.get(document.sha256)
if analysis is None:
    st.info("Click Analyze PDF. It reads the text layer locally; no model is called and nothing is sent.")
else:
    rows = []
    for page in analysis["pages"]:
        reorder = page.get("reorder") or {}
        rows.append([
            panel._esc(page["page"]), panel._esc(hybrid_analysis.CATEGORY_LABELS[page["category"]]),
            panel._esc(page["category_reason"]), panel._esc(reorder.get("reorder_status")),
            panel._esc(f"{page['arabic_letter_share']:.0%}"),
            panel._esc(page["indicators_processed"].get("single_letter_arabic_ratio")),
            panel._esc(page["table_count"]), panel._esc(page["images"]["warning"] or "-"),
        ])
    st.markdown(panel._table(["Page", "Decision", "Reason", "Reorder status", "Arabic share", "Single-letter ratio",
                              "Tables", "Images"], rows), unsafe_allow_html=True)

# ---------------------------------------------------------------- 5 plan

vlm_pages: dict[int, str] = {}
ready_to_run = False
provider_plan = None
qari_pages = None
runner = APIEvaluationRunner(manager, settings, store)
if analysis is not None:
    st.subheader("5. Review plan")
    if model == hybrid.LIBRARY_ONLY:
        st.info("Library Only: every page keeps its library text and diagnosis; no page is sent anywhere.")
        ready_to_run = True
    else:
        include_images = st.checkbox("Also send pages that contain raster images (text inside images)",
                                     value=False, key="hy-images")
        proposed = hybrid_analysis.default_vlm_pages(analysis, include_image_pages=include_images)
        chosen = st.multiselect("Pages to send to the fallback model", [p["page"] for p in analysis["pages"]],
                                default=sorted(proposed), key=f"hy-pages-{document.sha256}-{include_images}")
        vlm_pages = {page: proposed.get(page, "selected by the user") for page in chosen}
        if not vlm_pages:
            st.info("No page needs the model. Use Library Only, or select pages explicitly.")
        st.caption("Healthy pages are not proposed. Every selected page and its reason is recorded in plan.json.")
        if vlm_pages and model in {hybrid.OXEN, hybrid.BAIDU} and spec is not None:
            document_ok = panel._document_gate(document, source, store)
            provider_ok = panel._provider_gate(spec, store, len(vlm_pages))
            plans = st.session_state.setdefault("hy_provider_plans", {})
            key = (document.sha256, spec.id, tuple(sorted(vlm_pages)))
            if document_ok and st.button("Prepare the page images (rendered locally; nothing is sent)",
                                         key="hy-prepare-provider"):
                try:
                    plans[key] = full_pdf.build_plan(runner, spec, workspace_id, document, source,
                                                     pages=sorted(vlm_pages))[0]
                except (APIRunRefused, full_pdf.FullPDFError) as exc:
                    st.error(f"Cannot prepare the pages: {exc}")
            provider_plan = plans.get(key)
            if provider_plan:
                st.text(f"Destination: POST {provider_plan['endpoint']} ({spec.provider})\n"
                        f"Allowed hosts: {', '.join(provider_plan['outbound_hosts_allowed'])}\n"
                        f"Estimated maximum cost: {provider_plan['cost']['estimated_max_usd_total']} USD\n"
                        "Not sent: " + "; ".join(provider_plan["not_sent"]))
                st.markdown(panel._table(["Page", "Reason", "Pixels", "Bytes", "SHA-256"], [
                    [panel._esc(i["page"]), panel._esc(vlm_pages[i["page"]]), panel._esc(f"{i['width']}x{i['height']}"),
                     panel._esc(f"{i['bytes']:,}"), f"<code>{panel._esc(i['image_sha256'])}</code>"]
                    for i in provider_plan["pages"]]), unsafe_allow_html=True)
            ready_to_run = bool(document_ok and provider_ok and provider_plan and credential)
        elif vlm_pages and model == hybrid.QARI:
            endpoint = None
            try:
                endpoint = qari.parse_endpoint(qari_url) if qari_url else None
            except qari.QariError as exc:
                st.error(str(exc))
            document_ok = True
            if endpoint and not endpoint["loopback"]:
                st.warning(f"Remote server {endpoint['host']}: page images leave this computer. The document must be "
                           "authorized by its exact SHA-256.")
                document_ok = panel._document_gate(document, source, store)
            images = st.session_state.setdefault("hy_qari_images", {})
            key = (document.sha256, tuple(sorted(vlm_pages)))
            if st.button(f"Prepare the page images at {qari.QARI_DPI} DPI (local; nothing is sent)",
                         key="hy-prepare-qari"):
                rendered = []
                for page in sorted(vlm_pages):
                    image = render_pdf_page(source, page, manager.path(workspace_id) / "e2_images" / document.doc_id,
                                            dpi=qari.QARI_DPI)
                    rendered.append({"page": page, "png": Path(image.path).read_bytes(), "width": image.width,
                                     "height": image.height, "effective_dpi": image.dpi})
                images[key] = rendered
            qari_pages = images.get(key)
            if qari_pages and endpoint:
                st.text(f"Destination: POST {endpoint['chat_url']} ({'this computer' if endpoint['loopback'] else 'remote server'})\n"
                        f"Model: {served_model} | prompt 'Free OCR.' | max_tokens {qari.QARI_MAX_NEW_TOKENS} | temperature 0\n"
                        "Not sent: Ground Truth, library text / PDF text layer, other pages, the PDF file")
                st.markdown(panel._table(["Page", "Reason", "Pixels", "Bytes", "SHA-256"], [
                    [panel._esc(i["page"]), panel._esc(vlm_pages[i["page"]]), panel._esc(f"{i['width']}x{i['height']}"),
                     panel._esc(f"{len(i['png']):,}"), f"<code>{hashlib.sha256(i['png']).hexdigest()}</code>"]
                    for i in qari_pages]), unsafe_allow_html=True)
            ready_to_run = bool(qari_ready and endpoint and qari_pages and document_ok)
            if not qari_ready:
                st.caption("Run is disabled until the Qari connection check reports Ready.")

# ---------------------------------------------------------------- 6 approve & run


def _execute(action, run_dir: Path) -> None:
    st.button("Stop after the current page", key="hy-stop")
    bar = st.progress(0.0, text="Starting...")

    def progress(event, index, total, record):
        label = (record or {}).get("status") if event == "done" else event
        bar.progress(min(1.0, (index - (0 if event == "done" else 1)) / max(total, 1)),
                     text=f"Page {index}/{total}: {label}")

    message = None
    try:
        result = action(progress) or {}
        message = result.get("stopped")
    except (full_pdf.FullPDFError, APIRunRefused, hybrid.HybridError, qari.QariError) as exc:
        message = str(exc)
    st.session_state["hy-run"] = str(run_dir)
    st.session_state["hy-flash"] = ("warning", f"Stopped: {message}") if message else (
        "success", "Model pages processed. Review the results below.")
    st.rerun()


def _model_action(run_dir: Path, metadata: dict):
    if metadata["model"] == hybrid.QARI:
        token = st.session_state.get("hy_qari_tokens", {}).get(
            qari.load_plan(run_dir / "vlm")["endpoint"]["display"]) or qari_token or None
        return lambda progress: hybrid.run_qari(run_dir, token=token, root=store.root, on_progress=progress)
    run_spec = specs[metadata["model"]]
    key = panel._credential(run_spec)
    return lambda progress: full_pdf.run_pending(run_dir / "vlm", runner, run_spec, workspace_id=workspace_id,
                                                 document=document, pdf_path=source, credential=key, root=store.root,
                                                 on_progress=progress)


if analysis is not None:
    st.subheader("6. Approve & Run")
    approved_by = ""
    agree = False
    if model != hybrid.LIBRARY_ONLY and vlm_pages:
        approved_by = st.text_input("Your name (recorded with the approval)", key="hy-approve-by")
        agree = st.checkbox(f"I approve sending {len(vlm_pages)} page image(s) of {document.display_name} to "
                            f"{hybrid.MODEL_LABELS[model]}. {hybrid.APPROVAL_STATEMENT}",
                            key=f"hy-approve-{model}-{document.sha256}-{tuple(sorted(vlm_pages))}")
    blocked = not ready_to_run or (model != hybrid.LIBRARY_ONLY and (not vlm_pages or not agree or not approved_by.strip()))
    st.caption(ground_truth_line)
    label = "Run (Library Only)" if model == hybrid.LIBRARY_ONLY else "Approve & Run"
    if st.button(label, type="primary", disabled=blocked, key="hy-run-button"):
        try:
            run_dir = hybrid.create_run(
                root=store.root, document=document, pdf_path=source, analysis=analysis, model=model,
                vlm_pages=vlm_pages, approved_by=approved_by,
                statement=hybrid.APPROVAL_STATEMENT if agree else None, ground_truth=ground_truth, ground_truth_scope=gt_scope, spec=spec,
                provider_plan=provider_plan, qari_endpoint=qari_url or None, qari_pages=qari_pages,
                qari_model=st.session_state.get("hy-qari-model", qari.QARI_MODEL))
        except Exception as exc:  # noqa: BLE001 - refused before anything was sent
            st.error(f"Run not started (nothing was sent): {exc}")
        else:
            st.session_state["hy-run"] = str(run_dir)
            if model == hybrid.LIBRARY_ONLY:
                hybrid.finalize(run_dir, None, store.root, pdf_path=source)
                st.session_state["hy-flash"] = ("success", f"Library-only run {run_dir.name} saved.")
                st.rerun()
            metadata, _plan, _library = hybrid.load(run_dir)
            _execute(_model_action(run_dir, metadata), run_dir)

# ---------------------------------------------------------------- 7 results

st.subheader("7. Results")
runs = [item for item in hybrid.list_runs(store.root) if item["metadata"]["document"]["sha256"] == document.sha256]
if not runs:
    st.info("No hybrid run exists for this PDF yet.")
    st.stop()
paths = [str(item["run_dir"]) for item in runs]
wanted = st.session_state.get("hy-run")
chosen_run = st.selectbox("Run", runs, index=paths.index(wanted) if wanted in paths else 0, key="hy-run-select",
                          format_func=lambda item: f"{item['run_id']} | {item['metadata']['model_label']} | "
                                                   f"{'finalized' if item['finalized'] else 'open'}")
run_dir = Path(chosen_run["run_dir"])
run_spec = specs.get(chosen_run["metadata"]["model"])
try:
    state = hybrid.summary(run_dir, run_spec, store.root)
except (hybrid.HybridError, full_pdf.FullPDFError, qari.QariError) as exc:
    st.error(f"Cannot read this run: {exc}")
    st.stop()
counts = state["counts"]
st.text(f"Folder: {run_dir}\nAccepted {counts['Accepted']} | Needs Review {counts['Needs Review']} | Uncertain "
        f"{counts['Uncertain']} | Failed {counts['Failed']}\n"
        + ("Finalized." if state["finalized"] else "Open: final files are written when you finalize.")
        + (f"\nModel pages never sent: {state['pending']['unsent']}" if state["pending"]["unsent"] else "")
        + (f"\nModel pages with unfinished jobs: {state['pending']['open']}" if state["pending"]["open"] else ""))
st.markdown(panel._table(["Page", "Category", "Status", "Final source", "Number conflicts", "Warnings"], [
    [panel._esc(p["page"]), panel._esc(p["category"]), _status_badge(p["status"]), panel._esc(p["final_source"]),
     panel._esc(len((p.get("number_audit") or {}).get("conflicts") or [])), panel._esc("; ".join(p["warnings"]))]
    for p in state["pages"]]), unsafe_allow_html=True)
zips = st.session_state.setdefault("hy_zips", {})
if st.button("Prepare ZIP (whole run folder)", key=f"hy-zip-prepare-{run_dir.name}"):
    zips[run_dir.name] = hybrid.zip_bytes(run_dir, run_spec, store.root, pdf_path=source)
if run_dir.name in zips:
    st.download_button("Download ZIP", data=zips[run_dir.name], file_name=f"{run_dir.name}.zip",
                       mime="application/zip", key=f"hy-zip-{run_dir.name}")
if not state["finalized"]:
    st.caption("Before finalizing, the ZIP adds provisional derived files under provisional/ (nothing is written).")

page_number = st.selectbox("Show page", [p["page"] for p in state["pages"]], key=f"hy-page-{run_dir.name}")
item = next(p for p in state["pages"] if p["page"] == page_number)
library_page = next(p for p in state["library"]["pages"] if p["page"] == page_number)
st.markdown(_status_badge(item["status"]) + " " + panel._esc(" | ".join(item["reasons"])), unsafe_allow_html=True)
for warning in item["warnings"]:
    st.warning(warning)
tabs = st.tabs(["Final", "PyMuPDF raw", "PyMuPDF processed", "OCR/VLM raw", "library_rejoined", "Numbers"])
with tabs[0]:
    st.caption(f"Source: {item['final_source']}")
    st.markdown(text_html(item["final_text"]), unsafe_allow_html=True)
with tabs[1]:
    st.markdown(text_html(library_page["raw_text"]), unsafe_allow_html=True)
with tabs[2]:
    st.caption(f"reorder_status: {(library_page.get('reorder') or {}).get('reorder_status')}")
    st.markdown(text_html(library_page["processed_text"]), unsafe_allow_html=True)
with tabs[3]:
    record = state["vlm"].get(page_number)
    if record is None:
        st.caption("This page was not sent to a model.")
    else:
        st.caption(f"{record.get('status')} | {record.get('status_reason')} | finish reason {record.get('finish_reason')}")
        quality = item.get("vlm_quality") or {}
        if quality.get("flagged"):
            st.warning("Quality flags: " + "; ".join(quality["reasons"]))
        st.markdown(text_html(record.get("text") or ""), unsafe_allow_html=True)
with tabs[4]:
    rejoined = item.get("rejoined")
    if not rejoined:
        st.caption("No library_rejoined candidate for this page.")
    else:
        st.caption(f"Accepted: {rejoined['accepted']} | joins: {rejoined['join_count']} | characters identical: "
                   f"{rejoined['non_space_sequence_identical']}"
                   + (f" | rejected: {'; '.join(rejoined['rejection_reasons'])}" if rejoined["rejection_reasons"] else ""))
        st.markdown(text_html(rejoined["text"]), unsafe_allow_html=True)
        with st.expander(f"Joins and evidence ({rejoined['join_count']})"):
            st.json(rejoined["joins"], expanded=False)
with tabs[5]:
    audit = item.get("number_audit")
    if not audit:
        st.caption("No number audit (no model output for this page).")
    else:
        st.caption(f"Library numbers {audit['library_numbers']} | VLM numbers {audit['vlm_numbers']} | matched "
                   f"occurrences {audit['matched_occurrences']}. {audit['note']}")
        if audit["conflicts"]:
            st.markdown(panel._table(["Normalized", "Library (count: originals)", "VLM (count: originals)"], [
                [panel._esc(c["normalized"]), panel._esc(f"{c['library_count']}: {' '.join(c['library_originals'])}"),
                 panel._esc(f"{c['vlm_count']}: {' '.join(c['vlm_originals'])}")] for c in audit["conflicts"]]),
                unsafe_allow_html=True)

st.markdown("**Page documents (Markdown, layout-aware)**")
latest = hybrid.latest_layout(run_dir)
enriched_docs = hybrid_enrich.latest_enriched(run_dir)
if enriched_docs:  # the approved Phase 2 revision is the current page document
    latest = enriched_docs | {"layout_version": latest.get("layout_version") if latest else None}
if latest:
    origin = ("saved automatically at Finalize (initial layout, not reviewed)"
              if latest.get("kind") == "initial layout at finalize" else f"approved by {latest.get('approved_by')}")
    st.caption(f"Layout revision {latest['revision']} ({Path(latest['folder']).name}), layout version "
               f"{latest.get('layout_version')}, {origin}. Text extraction and layout quality are reported separately.")
    doc_page = next((p for p in latest["pages"] if p["page"] == page_number), None)
    if doc_page:
        md_left, md_right = st.columns(2)
        md_left.download_button(f"Download page_{page_number:03d}.md", data=doc_page["markdown"].encode("utf-8"),
                                file_name=f"page_{page_number:03d}.md", mime="text/markdown",
                                key=f"hy-md-{run_dir.name}-{page_number}")
        md_right.download_button("Download full_document.md", data=latest["full_document"].encode("utf-8"),
                                 file_name="full_document.md", mime="text/markdown", key=f"hy-full-{run_dir.name}")
        for note in (doc_page.get("record") or {}).get("needs_review") or []:
            st.warning(note)
        st.markdown(doc_page["markdown"])
else:
    st.caption("No layout revision yet: preview and save one below (Finalize also saves an initial one).")

with st.expander("Recompute layout (offline: saved data and the run's PDF copy only)",
                 expanded=latest is None):
    st.caption("Assembles one Markdown document per page from the accepted decisions, table geometry, image "
               "regions and saved model text. No model, network or provider is called. Finalized runs get a new "
               "revision under layout/ only; nothing existing is changed.")
    layout_previews = st.session_state.setdefault("hy_layout_preview", {})
    if st.button("Preview page documents", key=f"hy-layout-preview-{run_dir.name}"):
        try:
            layout_previews[run_dir.name] = hybrid.preview_layout(run_dir, run_spec, store.root, pdf_path=source)
        except (hybrid.HybridError, full_pdf.FullPDFError, qari.QariError) as exc:
            st.error(str(exc))
    preview = layout_previews.get(run_dir.name)
    if preview:
        st.info(f"AUTOMATICALLY GENERATED layout (version {preview['layout_version']}), not saved and not reviewed. "
                "Inspect it, then save it as a revision below.")
        st.markdown(panel._table(["Page", "Text extraction", "Layout", "Direction", "Elements", "Text integrity",
                                  "Paragraph merges", "Image merges", "Fallback", "Needs review"], [
            [panel._esc(p["page"]), panel._esc(p["record"].get("text_status")), panel._esc(p["record"].get("layout_status")),
             panel._esc(p["record"]["direction"]), panel._esc(len(p["record"].get("order") or [])),
             panel._esc("ok" if (p["record"]["validation"] or {}).get("text_elements_match_page_text") else
                        ("-" if p["record"]["validation"] is None else "failed")),
             panel._esc(len(p["record"].get("paragraph_merges") or [])), panel._esc(len(p["record"].get("image_merges") or [])),
             panel._esc(p["record"]["fallback_reason"] or "-"), panel._esc("; ".join(p["record"]["needs_review"]) or "-")]
            for p in preview["pages"]]), unsafe_allow_html=True)
        if preview.get("previous_revision"):
            st.caption(f"Previous approved revision: {preview['previous_revision']}")
        preview_page = st.selectbox("Preview page", [p["page"] for p in preview["pages"]],
                                    key=f"hy-layout-page-{run_dir.name}")
        chosen_preview = next(p for p in preview["pages"] if p["page"] == preview_page)
        left_col, right_col = st.columns(2)
        with left_col:
            st.caption("Reading order (numbered boxes: blue paragraph, purple heading, green table, red figure)")
            try:
                import pymupdf  # noqa: PLC0415

                pdf_file, _origin = hybrid.layout_pdf(run_dir, state["metadata"], source)
                with pymupdf.open(pdf_file) as overlay_doc:
                    st.image(hybrid_layout.order_overlay_png(overlay_doc[preview_page - 1], chosen_preview["record"]),
                             width="stretch")
            except Exception as exc:  # noqa: BLE001 - preview only
                st.caption(f"Overlay unavailable: {exc}")
            st.caption("Order: " + " → ".join(chosen_preview["record"].get("order") or []))
            rec = chosen_preview["record"]
            with st.expander("Layout details (merges, exclusions, warnings, integrity)"):
                st.json({
                    "text_extraction_status": rec.get("text_status"), "layout_status": rec.get("layout_status"),
                    "text_integrity": rec.get("validation"), "fallback_reason": rec.get("fallback_reason"),
                    "needs_review": rec.get("needs_review"), "image_merges": rec.get("image_merges"),
                    "image_review": rec.get("image_review"),
                    "skipped_images": [{k: v for k, v in img.items() if k in {"id", "bbox", "area_share", "width_pt",
                                                                               "height_pt", "reason", "borderline"}}
                                       for img in rec.get("skipped_images") or []],
                    "figures": [{k: v for k, v in fig.items() if k in {"id", "origin", "bbox", "confidence", "review",
                                                                        "merged_from", "table_overlap",
                                                                        "possible_child_regions", "crop_sha256"}}
                                for fig in rec.get("figures") or []],
                    "vector_exclusions": rec.get("vector_exclusions"),
                    "ignored_vector_regions": rec.get("ignored_vector_regions"),
                    "table_dedup": rec.get("table_dedup"), "paragraph_merges": rec.get("paragraph_merges"),
                    "list_items": rec.get("list_items"), "discarded_markers": rec.get("discarded_markers"),
                    "side_by_side": rec.get("side_by_side"), "table_continuations": rec.get("table_continuations"),
                    "structure_checks": rec.get("structure_checks"), "captions": rec.get("captions"),
                    "ambiguous_regions": rec.get("ambiguous_regions"),
                }, expanded=False)
        with right_col:
            st.caption("Assembled Markdown (rendered)")
            st.markdown(chosen_preview["markdown"])
            with st.expander("Markdown source"):
                st.code(chosen_preview["markdown"], language="markdown")
        with st.form(f"hy-layout-apply-{run_dir.name}"):
            layout_by = st.text_input("Your name")
            layout_agree = st.checkbox(hybrid.LAYOUT_STATEMENT)
            save_layout = st.form_submit_button("Save as a new layout revision")
        if save_layout:
            try:
                if not layout_agree:
                    raise hybrid.HybridError("Tick the confirmation to save the layout revision")
                folder = hybrid.apply_layout(run_dir, run_spec, store.root, by=layout_by,
                                             statement=hybrid.LAYOUT_STATEMENT, pdf_path=source)
                layout_previews.pop(run_dir.name, None)
                zips.pop(run_dir.name, None)
                st.session_state["hy-flash"] = ("success", f"Layout revision saved: {folder.name}")
                st.rerun()
            except (hybrid.HybridError, full_pdf.FullPDFError, qari.QariError) as exc:
                st.error(str(exc))

gt_eval_path = run_dir / "evaluation.json"
evaluation = None
if gt_eval_path.exists():
    evaluation = json.loads(gt_eval_path.read_text(encoding="utf-8"))
elif not state["finalized"] and (run_dir / "ground_truth.txt").exists():
    evaluations = st.session_state.setdefault("hy_evaluations", {})
    if st.button("Compute provisional evaluation", key=f"hy-eval-{run_dir.name}"):
        evaluations[run_dir.name] = json.loads(
            hybrid.outputs(run_dir, run_spec, store.root, pdf_path=source)["evaluation.json"])
    evaluation = evaluations.get(run_dir.name)
if evaluation:
    st.markdown(f"**Evaluation against Ground Truth** (after processing only; scope: "
                f"{evaluation.get('scope_label') or 'text layer (not recorded)'})")
    if evaluation.get("statement") or evaluation.get("not_computed"):
        st.caption(evaluation.get("statement") or evaluation.get("not_computed"))
    st.markdown(panel._table(["Source", "CER", "WER", "CER (normalized)", "WER (normalized)", "Pages"], [
        [panel._esc(source_name)] + [panel._esc("-" if v is None else f"{v:.2%}") for v in (
            ((data.get("total") or {}).get("none") or {}).get("cer"), ((data.get("total") or {}).get("none") or {}).get("wer"),
            ((data.get("total") or {}).get("arabic_normalized") or {}).get("cer"),
            ((data.get("total") or {}).get("arabic_normalized") or {}).get("wer"))]
        + [panel._esc(data.get("pages_compared", "document"))]
        for source_name, data in evaluation["sources"].items()]), unsafe_allow_html=True)
    st.caption(f"Normalized = {diagnostics.ARABIC_NORMALIZATION}. Unnormalized compares exact characters.")

metadata = state["metadata"]
if not state["finalized"]:
    model_run = metadata["model"] != hybrid.LIBRARY_ONLY
    if model_run and state["pending"]["unsent"]:
        if st.button(f"Continue: send {len(state['pending']['unsent'])} page(s) never sent", key="hy-continue"):
            _execute(_model_action(run_dir, metadata), run_dir)
    if metadata["model"] == hybrid.BAIDU and run_spec is not None and state["pending"]["open"]:
        if st.button("Resume paused Baidu pages from their Job IDs (no resubmission)", key="hy-resume"):
            _execute(lambda progress: full_pdf.resume_paused(run_dir / "vlm", runner, run_spec,
                                                             credential=panel._credential(run_spec),
                                                             root=store.root, on_progress=progress), run_dir)
    if st.button("Finalize (write page files, full_text.txt, report.html, results.json once)", key="hy-finalize"):
        try:
            hybrid.finalize(run_dir, run_spec, store.root, pdf_path=source)
            st.rerun()
        except hybrid.HybridError as exc:
            st.error(str(exc))

st.caption(f"Decisions shown, downloaded and finalized: logic version {state['decision_logic_version']}"
           + (f" (approved revision {state['decision_revision']})" if state["decision_revision"]
              else " (as created; no recomputation approved)"))
if not state["finalized"]:
    with st.expander("Recompute decisions (offline: saved outputs only)"):
        st.caption("Recomputes only the derived decisions (candidate selection, final text, status, warnings, "
                   "evaluation) from the saved library text, saved model outputs, diagnostics and tables. No "
                   "extraction, model, network or provider is called. Ground truth is not used for decisions.")
        previews = st.session_state.setdefault("hy_recompute_preview", {})
        if st.button(f"Preview decisions with logic version {hybrid.DECISION_LOGIC_VERSION}",
                     key=f"hy-recompute-preview-{run_dir.name}"):
            try:
                previews[run_dir.name] = hybrid.preview_recompute(run_dir, run_spec, store.root)
            except (hybrid.HybridError, full_pdf.FullPDFError, qari.QariError) as exc:
                st.error(str(exc))
        preview = previews.get(run_dir.name)
        if preview:
            st.markdown(panel._table(
                ["Page", "Previous source", "Previous status", "Proposed source", "Proposed status", "Proposed reason"],
                [[panel._esc(row["page"]) + (" *" if row["changed"] else ""),
                  panel._esc(row["previous"]["final_source"]), panel._esc(row["previous"]["status"]),
                  panel._esc(row["proposed"]["final_source"]), panel._esc(row["proposed"]["status"]),
                  panel._esc(row["proposed"]["reason"])] for row in preview["pages"]]), unsafe_allow_html=True)
            st.caption(f"Version {preview['previous_logic_version']} -> {preview['proposed_logic_version']}; "
                       f"changed pages (*): {preview['changed_pages'] or 'none'}")
            with st.form(f"hy-recompute-apply-{run_dir.name}"):
                by = st.text_input("Your name")
                agree_recompute = st.checkbox(hybrid.RECOMPUTE_STATEMENT)
                apply_now = st.form_submit_button("Apply this decision revision")
            if apply_now:
                try:
                    if not agree_recompute:
                        raise hybrid.HybridError("Tick the confirmation to apply")
                    path = hybrid.apply_recompute(run_dir, run_spec, store.root, by=by,
                                                  statement=hybrid.RECOMPUTE_STATEMENT)
                    previews.pop(run_dir.name, None)
                    st.session_state["hy-flash"] = ("success", f"Decision revision saved: {Path(path).name}")
                    st.rerun()
                except (hybrid.HybridError, full_pdf.FullPDFError, qari.QariError) as exc:
                    st.error(str(exc))

with st.expander("Advanced Details"):
    st.caption("Run metadata (no API key), plan and per-page model records")
    st.json({"metadata": metadata, "plan": state["plan"]}, expanded=False)
    st.json({str(k): {key: value for key, value in v.items() if key != "text"} for k, v in state["vlm"].items()},
            expanded=False)
    if not (run_dir / "ground_truth.txt").exists():
        late = st.file_uploader("Attach ground truth to this run (write-once; evaluation only)", type=["txt"],
                                accept_multiple_files=True, key=f"hy-late-gt-{run_dir.name}")
        if late and st.button("Attach ground truth", key=f"hy-late-gt-save-{run_dir.name}"):
            try:
                text = diagnostics.combine_ground_truth_files([(f.name, f.getvalue()) for f in late])
                hybrid.attach_ground_truth(run_dir, text, [f.name for f in late], run_spec, store.root,
                                           scope=st.session_state.get("hy-gt-scope", "text_layer"))
                st.rerun()
            except (ValueError, hybrid.HybridError) as exc:
                st.error(str(exc))
    if not state["finalized"] and metadata["model"] != hybrid.LIBRARY_ONLY:
        retryable = [p for p, r in state["vlm"].items()
                     if r.get("job_state") in {"FAILED", "ABANDONED", "SUBMISSION_UNCONFIRMED"}]
        if retryable:
            with st.form(f"hy-retry-{run_dir.name}"):
                st.markdown("**Retry a failed or unconfirmed page (explicit; never automatic)**")
                retry_page = st.selectbox("Page", sorted(retryable))
                by = st.text_input("Your name")
                agree_retry = st.checkbox(full_pdf.RETRY_STATEMENT)
                submit = st.form_submit_button("Send this page again")
            if submit:
                try:
                    if not agree_retry:
                        raise hybrid.HybridError("Tick the retry statement")
                    if metadata["model"] == hybrid.QARI:
                        hybrid.retry_qari_page(run_dir, int(retry_page), token=qari_token or None, root=store.root,
                                               by=by, statement=full_pdf.RETRY_STATEMENT)
                    else:
                        full_pdf.retry_page(run_dir / "vlm", runner, run_spec, int(retry_page),
                                            workspace_id=workspace_id, document=document, pdf_path=source,
                                            credential=panel._credential(run_spec), root=store.root, by=by,
                                            statement=full_pdf.RETRY_STATEMENT)
                    st.rerun()
                except (hybrid.HybridError, full_pdf.FullPDFError, APIRunRefused, qari.QariError) as exc:
                    st.error(str(exc))
if run_spec is not None:  # its own expander (expanders cannot be nested)
    panel._unconfirmed_submissions(runner, run_spec)

render_phase2(run_dir=run_dir, run_spec=run_spec, document=document, source=source, store=store, runner=runner,
              workspace_id=workspace_id)
