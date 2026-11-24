"""Simple full-PDF view of the OCR/VLM Evaluation page (E2, hosted API).

Upload PDF -> Select Model -> API Key -> Authorize -> Run Full PDF -> Download ZIP.

Nothing is sent when the page opens or a setting changes: rendering the page images
(local only) and every transmission start from an explicit button. The API key lives
only in this browser session's memory. All gates are the existing ones (exact-hash
document authorization, provider enablement and limits, confirmed image hashes),
re-checked by the runner before every page.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st

from lab.app.components import text_html
from lab.core.e2.api import documents, full_pdf, host_policy
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import (
    REQUIRED_STATEMENT,
    AuthorizationError,
    check_authorization,
    latest_enablement,
    record_enablement,
    usage_policy,
)
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused
from lab.core.ingest import IngestError

MODEL_LABELS = {
    "oxen-qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct (Oxen.ai)",
    "baidu-aistudio-paddleocr-vl-1.6": "PaddleOCR-VL 1.6 (Baidu AI Studio)",
}
_STATUS_COLOR = {full_pdf.SUCCESS: "#1a7f37", full_pdf.UNCERTAIN: "#9a6700",
                 full_pdf.FAILED: "#cf222e", full_pdf.NOT_SENT_STATUS: "#57606a"}


def _esc(value) -> str:
    return html.escape("" if value is None else str(value)).replace("$", "&#36;").replace("\n", "&#10;")


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th style='text-align:left;padding:2px 8px'>{_esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td style='padding:2px 8px;vertical-align:top'>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table style='border-collapse:collapse;font-size:0.9em'><tr>{head}</tr>{body}</table>"


def _badge(status: str) -> str:
    color = _STATUS_COLOR.get(status, "#57606a")
    return (f"<span style='background:{color};color:#fff;border-radius:8px;padding:0 8px'>"
            f"{_esc(status)}</span>")


def _status_rows(records: list[dict]) -> list[list]:
    rows = []
    for record in records:
        tokens = record.get("tokens") or {}
        rows.append([
            _esc(record["page"]), _badge(record["status"]),
            _esc(record.get("processing_seconds") if record.get("processing_seconds") is not None else "-"),
            _esc(f"{tokens.get('prompt', '-')} / {tokens.get('completion', '-')}" if tokens.get("reported") else "not reported"),
            _esc(record.get("chars", "-")), _esc(record["status_reason"]),
        ])
    return rows


def _status_table(records: list[dict]) -> str:
    return _table(["Page", "Status", "Time (s)", "Tokens in / out", "Chars", "Details"], _status_rows(records))


def _credential(spec) -> str:
    """The key typed for this provider in this session (memory only), else the env var."""
    store = st.session_state.setdefault("fp_credentials", {})
    return store.get(spec.id) or os.environ.get(spec.credential_env, "")


# ---------------------------------------------------------------- sections


def _choose_document(ingest, workspace_id):
    st.subheader("1. PDF")
    source = st.radio("PDF source", ["Choose an uploaded PDF", "Upload a new PDF"], horizontal=True,
                      key="fp-source")
    if source == "Upload a new PDF":
        uploaded = st.file_uploader("PDF file", type=["pdf"], key="fp-upload")
        if uploaded is not None and st.button("Add this PDF", key="fp-add"):
            try:
                document = ingest.ingest(workspace_id, uploaded, uploaded.name)
            except IngestError as exc:
                st.error(f"Upload refused: {exc}")
            else:
                st.session_state["fp-doc"] = document.doc_id
                st.session_state["fp-source"] = "Choose an uploaded PDF"
                st.rerun()
    pdfs = [item for item in ingest.list_documents(workspace_id) if item.kind == "pdf"]
    if not pdfs:
        st.info("No PDF in this workspace yet: upload one above.")
        return None
    ids = [item.doc_id for item in pdfs]
    wanted = st.session_state.get("fp-doc")
    document = st.selectbox(
        "PDF", pdfs, index=ids.index(wanted) if wanted in ids else len(pdfs) - 1,
        format_func=lambda item: f"{item.display_name} ({item.page_count} pages)", key="fp-doc-select",
    )
    st.session_state["fp-doc"] = document.doc_id
    st.caption(f"{document.page_count} pages | {document.size_bytes:,} bytes | SHA-256 {document.sha256}")
    return document


def _choose_model():
    st.subheader("2. Model")
    specs = {spec.id: spec for spec in load_providers() if spec.id in full_pdf.MODEL_FOLDERS and spec.runnable}
    if not specs:
        st.error("Neither supported provider is implemented in providers_e2.toml.")
        return None
    choice = st.radio("Model", list(specs), format_func=lambda key: MODEL_LABELS.get(key, specs[key].label),
                      horizontal=True, key="fp-model")
    spec = specs[choice]
    st.caption(
        f"{spec.provider} | model {spec.api_model_id} | endpoint {spec.endpoint} | "
        f"prompt {spec.prompt_id or 'none (document-parsing endpoint)'} | "
        f"generation {json.dumps(spec.generation) if spec.generation else 'vendor defaults'} | "
        f"exact revision verified: {spec.exact_revision_verified}. Settings come from providers_e2.toml unchanged."
    )
    return spec


def _api_key(spec) -> str:
    st.subheader("3. API key")
    store = st.session_state.setdefault("fp_credentials", {})
    typed = st.text_input(
        f"{spec.credential_label} (kept only in this browser session's memory; never shown, saved or written "
        "to any file or report)", type="password", key=f"fp-key-input-{spec.id}",
    )
    if typed:
        store[spec.id] = typed
    credential = _credential(spec)
    st.caption("Key source: " + ("this session" if store.get(spec.id) else
                                 f"environment variable {spec.credential_env}" if credential else "none yet"))
    return credential


def _document_gate(document, source, store) -> bool:
    sha = document.sha256
    entry = documents.load_registry(root=store.root).get(sha)
    if entry is None:
        st.warning("Step 4a: this PDF is not registered for hosted API evaluation (its exact SHA-256 is unknown).")
        with st.form(f"fp-register-{sha}"):
            by = st.text_input("Your name", key=f"fp-register-by-{sha}")
            agree = st.checkbox(documents.REGISTRATION_STATEMENT, key=f"fp-register-agree-{sha}")
            submit = st.form_submit_button("Register this PDF (user-owned test document)")
        if submit:
            if not agree:
                st.error("Tick the registration statement.")
            else:
                try:
                    documents.register_document(source, display_name=document.display_name, registered_by=by,
                                                statement=documents.REGISTRATION_STATEMENT, root=store.root)
                    st.rerun()
                except documents.DocumentAuthorizationError as exc:
                    st.error(str(exc))
        return False
    state = documents.authorization_state(sha, root=store.root)
    if state["allowed"]:
        auth = state["authorization"]
        st.success(f"4a. Document authorized by {auth.get('reviewer')} until {auth.get('expires_at')} "
                   f"({documents.CATEGORY_LABELS[entry.authorization_category]}).")
        return True
    st.warning("Step 4a: document not authorized: " + "; ".join(state["reasons"]))
    if entry.authorization_category == documents.CATEGORY_USER_OWNED:
        st.caption("Review every page first. Authorize only if the declaration is true for every page.")
    with st.form(f"fp-authorize-{sha}"):
        reviewer = st.text_input("Reviewer name", key=f"fp-auth-reviewer-{sha}")
        typed = st.text_input("Type the first 12 characters of the SHA-256 above", key=f"fp-auth-sha-{sha}")
        days = st.number_input("Valid for (days)", 1, documents.MAX_AUTHORIZATION_DAYS, 7, key=f"fp-auth-days-{sha}")
        agree = st.checkbox(entry.declaration, key=f"fp-auth-agree-{sha}")
        approve = st.form_submit_button("Authorize this exact document")
    if approve:
        if not agree or typed.strip().lower() != sha[:12]:
            st.error("Tick the declaration and type the first 12 characters of the SHA-256 exactly.")
        else:
            try:
                documents.authorize_document(entry, file_path=source, reviewer=reviewer,
                                             declaration=entry.declaration, days_valid=int(days), root=store.root)
                st.rerun()
            except documents.DocumentAuthorizationError as exc:
                st.error(str(exc))
    return False


def _provider_gate(spec, store, pages: int) -> bool:
    state = check_authorization(spec, pages=1, root=store.root)
    latest = latest_enablement(spec, root=store.root) or {}
    remaining = state.remaining_pages
    enough = state.allowed and remaining >= pages
    if enough:
        st.success(f"4b. {spec.provider} enabled until {latest.get('expires_at')}: {state.describe()}; "
                   f"this PDF needs {pages}.")
    else:
        reasons = list(state.reasons)
        if state.allowed and remaining < pages:
            reasons.append(f"only {remaining} page(s) left within the limits; this PDF needs {pages}")
        st.warning(f"Step 4b: {spec.provider} cannot receive this PDF: " + "; ".join(reasons))
    with st.expander("Provider enablement and usage limits", expanded=not enough):
        policy = usage_policy(spec)
        ceiling, day_ceiling = policy["max_pages_per_enablement"], policy["max_pages_per_day_ceiling"]
        st.caption(f"Limit source: {policy['source']}. {policy['counting']}. {policy['provider_quota']}.")
        st.caption(f"History (all enablements): {state.pages_used_total} page(s) submitted; current enablement: "
                   f"{state.pages_used_enablement} since {state.enablement_started_at or '-'}"
                   + (f"; cumulative cap {policy['max_pages_cumulative']}" if policy["max_pages_cumulative"] else ""))
        need_total = min(ceiling, max(16, pages))
        with st.form(f"fp-enable-{spec.id}"):
            by = st.text_input("Your name", key=f"fp-enable-by-{spec.id}")
            max_total = st.number_input(f"Maximum pages for this enablement (internal ceiling {ceiling})", 1, ceiling,
                                        need_total, key=f"fp-enable-total-{spec.id}")
            max_day = st.number_input(f"Maximum pages per day (internal ceiling {day_ceiling})", 1, day_ceiling,
                                      min(need_total, day_ceiling, max(pages, 16)), key=f"fp-enable-day-{spec.id}")
            days = st.number_input("Valid for (days)", 1, 30, policy["default_enablement_days"],
                                   key=f"fp-enable-days-{spec.id}")
            agree = st.checkbox(REQUIRED_STATEMENT, key=f"fp-enable-agree-{spec.id}")
            enable = st.form_submit_button("Enable this provider with these limits")
        if enable:
            if not agree:
                st.error("Tick the confirmation statement.")
            else:
                try:
                    record_enablement(spec, enabled=True, enabled_by=by, statement=REQUIRED_STATEMENT,
                                      max_pages_total=int(max_total), max_pages_per_day=int(max_day),
                                      days_valid=int(days), root=store.root)
                    st.rerun()
                except AuthorizationError as exc:
                    st.error(str(exc))
    return enough


def _plan_section(runner, spec, workspace_id, document, source):
    key = f"{document.sha256}|{spec.id}"
    plans = st.session_state.setdefault("fp_plans", {})
    if st.button("Prepare the page images (rendered locally; nothing is sent)", key=f"fp-plan-{key}"):
        bar = st.progress(0.0, text="Rendering page 1...")
        try:
            plans[key] = full_pdf.build_plan(
                runner, spec, workspace_id, document, source,
                on_page=lambda page, total: bar.progress(page / total, text=f"Rendered page {page}/{total}"),
            )
        except (APIRunRefused, full_pdf.FullPDFError) as exc:
            st.error(f"Cannot prepare the pages: {exc}")
            return None
    if key not in plans:
        st.info("Step 4c: prepare the page images to see exactly what will be sent.")
        return None
    plan, images = plans[key]
    st.markdown("**4c. Exactly what will be sent, and where**")
    template = plan["request_template"]
    lines = [
        f"Destination: {template.get('method')} {plan['endpoint']} ({spec.provider})",
        f"Allowed outbound hosts: {', '.join(plan['outbound_hosts_allowed'])}",
        f"What: {plan['page_count']} page image(s) of {document.display_name}, one PNG per request, one page "
        f"at a time ({plan['pages'][0]['dpi']} DPI)",
        f"Request fields: {json.dumps(template.get('json_body') or template.get('form_fields'), ensure_ascii=False)}",
        "Not sent: " + "; ".join(plan["not_sent"]),
        f"Estimated maximum cost: {plan['cost']['estimated_max_usd_total']} USD "
        f"({plan['cost']['basis']}; price verified: {plan['cost']['price_verified']})",
    ]
    if plan["auto_result_hosts"]:
        lines.append("Results are downloaded from Baidu result storage: " + ", ".join(plan["auto_result_hosts"]))
    st.text("\n".join(lines))
    with st.expander(f"Page images and SHA-256 ({plan['page_count']})", expanded=False):
        st.markdown(_table(["Page", "File", "Pixels", "Bytes", "SHA-256"], [
            [_esc(item["page"]), _esc(item["filename"]), _esc(f"{item['width']}x{item['height']}"),
             _esc(f"{item['bytes']:,}"), f"<code>{_esc(item['image_sha256'])}</code>"] for item in plan["pages"]
        ]), unsafe_allow_html=True)
        columns = st.columns(4)
        for index, item in enumerate(plan["pages"]):
            path = images.get(item["page"])
            if path and Path(path).is_file():
                columns[index % 4].image(path, caption=f"page {item['page']}", width="stretch")
    return plan


def _run_section(runner, spec, store, workspace_id, document, source, plan, credential, gates_ok):
    st.subheader("5. Run Full PDF")
    checked = full_pdf.preflight(spec, plan, root=store.root, pages_to_send=plan["page_count"])
    by = st.text_input("Your name (recorded with the approval)", key="fp-approve-by")
    trust = False
    if host_policy.supported(spec):
        trust = st.checkbox(
            "Download results automatically from the listed Baidu result-storage hosts for this run "
            "(otherwise each page waits for a manual host approval)", key=f"fp-trust-{plan['plan_sha256']}",
        )
        if trust:
            st.caption(host_policy.POLICY_STATEMENT)
    agree = st.checkbox(
        f"I approve sending all {plan['page_count']} page image(s) of {document.display_name} (plan "
        f"{plan['plan_sha256'][:12]}) to {spec.provider} at {plan['endpoint']}. {full_pdf.APPROVAL_STATEMENT}",
        key=f"fp-approve-{plan['plan_sha256']}",
    )
    blockers = list(checked["reasons"])
    if not credential:
        blockers.append(f"enter the {spec.credential_label}")
    if not by.strip():
        blockers.append("enter your name")
    if not agree:
        blockers.append("tick the approval")
    if blockers:
        st.caption("Run Full PDF is disabled until: " + "; ".join(blockers))
    if not st.button("Run Full PDF", type="primary", disabled=bool(blockers) or not gates_ok, key="fp-run-button"):
        return
    try:
        run_dir = full_pdf.create_run(store.root, spec, document, plan, approved_by=by,
                                      statement=full_pdf.APPROVAL_STATEMENT, trust_result_hosts=trust,
                                      render_dpi=plan["pages"][0]["dpi"])
    except Exception as exc:  # noqa: BLE001 - refused before anything was sent
        st.error(f"Run not started (nothing was sent): {exc}")
        return
    st.session_state["fp-run"] = str(run_dir)
    _execute(lambda progress: full_pdf.run_pending(
        run_dir, runner, spec, workspace_id=workspace_id, document=document, pdf_path=source,
        credential=credential, root=store.root, on_progress=progress), run_dir)


def _execute(action, run_dir) -> None:
    """Run an action with live per-page progress; any click stops after the current page."""
    st.button("Stop after the current page", key="fp-stop")
    st.caption("Pages are processed one at a time. Stopping never loses saved pages; continue later from Results.")
    bar = st.progress(0.0, text="Starting...")
    slot = st.empty()
    seen: dict[int, dict] = {}

    def progress(event: str, page: int, total: int, record: dict | None) -> None:
        if event in {"sending", "resuming"}:
            bar.progress((page - 1) / total, text=f"Page {page}/{total}: {event}...")
        else:
            seen[page] = record or {}
            bar.progress(page / total, text=f"Page {page}/{total}: {(record or {}).get('status')}")
        slot.markdown(_status_table([seen[key] for key in sorted(seen)]), unsafe_allow_html=True)

    try:
        summary = action(progress)
    except (full_pdf.FullPDFError, APIRunRefused) as exc:
        st.session_state["fp-flash"] = ("error", f"Stopped: {exc}")
    else:
        counts = summary["counts"]
        text = (f"Run {summary['run_id']}: Success {counts['Success']}, Uncertain {counts['Uncertain']}, "
                f"Failed {counts['Failed']}, Not sent {counts['Not sent']}.")
        notes = [summary.get("stopped")] + list(summary.get("messages") or [])
        notes = [item for item in notes if item]
        level = "success" if summary["all_completed"] else "warning"
        st.session_state["fp-flash"] = (level, text + (" " + " ".join(notes) if notes else "")
                                        + (" Finalized." if summary["finalized"] else ""))
    st.session_state["fp-run"] = str(run_dir)
    st.rerun()


# ------------------------------------------------------------------ results


def _results_section(runner, store, workspace_id, document, source) -> None:
    st.subheader("6. Results")
    runs = [item for item in full_pdf.list_runs(store.root)
            if (item["document"] or {}).get("sha256") == document.sha256]
    if not runs:
        st.info("No full-PDF run exists for this PDF yet.")
        return
    paths = [str(item["run_dir"]) for item in runs]
    wanted = st.session_state.get("fp-run")
    chosen = st.selectbox(
        "Run", runs, index=paths.index(wanted) if wanted in paths else 0, key="fp-run-select",
        format_func=lambda item: (f"{item['run_id']} | {MODEL_LABELS.get(item['provider_id'], item['provider_id'])} "
                                  f"| {'finalized' if item['finalized'] else 'open'}"),
    )
    run_dir = Path(chosen["run_dir"])
    spec = next((item for item in load_providers() if item.id == chosen["provider_id"]), None)
    if spec is None:
        st.error(f"Provider {chosen['provider_id']} is no longer in providers_e2.toml; the saved files are intact.")
        return
    summary = full_pdf.run_summary(run_dir, spec, root=store.root)
    counts = summary["counts"]
    st.text(f"Folder: {run_dir}\nSuccess {counts['Success']} | Uncertain {counts['Uncertain']} | "
            f"Failed {counts['Failed']} | Not sent {counts['Not sent']} | of {summary['total']} page(s)\n"
            + ("Finalized: results.json, full_text.txt and report.html are saved." if summary["finalized"]
               else "Open: final files are written when every page completed or when you finalize."))
    st.markdown(_status_table(summary["records"]), unsafe_allow_html=True)
    st.download_button(
        "Download ZIP", data=full_pdf.zip_bytes(run_dir, spec, root=store.root),
        file_name=f"{run_dir.parent.name}_{run_dir.name}.zip", mime="application/zip",
        key=f"fp-zip-{run_dir.name}-{run_dir.parent.name}",
    )
    if not summary["finalized"]:
        st.caption("Before finalizing, the ZIP adds a provisional results.json, full_text.txt and report.html "
                   "generated in memory (nothing is written).")
    page = st.selectbox("Show the text of page", [item["page"] for item in summary["records"]],
                        key=f"fp-page-{run_dir}")
    record = next(item for item in summary["records"] if item["page"] == page)
    st.markdown(_badge(record["status"]) + " " + _esc(record["status_reason"]), unsafe_allow_html=True)
    text = full_pdf.page_text(run_dir, record)
    if text:
        st.markdown(text_html(text), unsafe_allow_html=True)
    else:
        st.caption("No text saved for this page.")

    busy = not summary["finalized"] and full_pdf.running(run_dir)
    if busy:
        st.info("This run is executing in another session; actions are disabled until it stops.")
    if not summary["finalized"] and not busy:
        _open_run_actions(runner, spec, store, workspace_id, document, source, run_dir, summary)
    _advanced(runner, spec, store, workspace_id, document, source, run_dir, summary, busy)


def _open_run_actions(runner, spec, store, workspace_id, document, source, run_dir, summary) -> None:
    credential = _credential(spec)
    approval = summary["approval"]
    sendable = [item for item in summary["records"]
                if item["job_state"] in {None, "REFUSED_BEFORE_SENDING", "NOT_TRANSMITTED"}]
    paused = [item for item in summary["records"]
              if item["job_state"] in {job_store.SUBMITTED, job_store.AWAITING_RESULT_HOST_APPROVAL,
                                       job_store.RESUMABLE_ERROR, job_store.RESUMING}
              and spec.protocol != "openai_chat_completions_v1"]
    if not credential:
        st.caption(f"Enter the {spec.credential_label} above (select this run's model) to continue it.")
    if sendable:
        label = f"Continue the run ({len(sendable)} page(s) never sent)"
        if not approval["valid"]:
            st.caption("Cannot continue: " + "; ".join(approval["reasons"]))
        if st.button(label, disabled=not (credential and approval["valid"]), key=f"fp-continue-{run_dir.name}"):
            _execute(lambda progress: full_pdf.run_pending(
                run_dir, runner, spec, workspace_id=workspace_id, document=document, pdf_path=source,
                credential=credential, root=store.root, on_progress=progress), run_dir)
    if paused:
        if st.button(f"Resume {len(paused)} paused page(s) from their Job IDs (no resubmission)",
                     disabled=not credential, key=f"fp-resume-{run_dir.name}"):
            _execute(lambda progress: full_pdf.resume_paused(
                run_dir, runner, spec, credential=credential, root=store.root, on_progress=progress), run_dir)


def _advanced(runner, spec, store, workspace_id, document, source, run_dir, summary, busy) -> None:
    with st.expander("Advanced Details", expanded=False):
        metadata, plan, approval = full_pdf.load_run(run_dir)
        st.caption("Run metadata (model, provider, actual settings, prompt, document; no API key)")
        st.json(metadata, expanded=False)
        st.caption("Plan (exact page images) and the single approval")
        st.json({"plan": plan, "approval": approval, "approval_state": summary["approval"]}, expanded=False)
        st.caption("Per-page records (job journal ids, store directories, tokens, indicators, errors)")
        st.json([{key: value for key, value in item.items() if key != "text"} for item in summary["records"]],
                expanded=False)
        if summary["finalized"] or busy:
            return
        credential = _credential(spec)
        for record in summary["records"]:
            if record["job_state"] == job_store.AWAITING_RESULT_HOST_APPROVAL:
                host = record.get("awaiting_host")
                with st.form(f"fp-host-{run_dir.name}-{record['page']}"):
                    st.markdown(f"**Page {record['page']}: result host `{_esc(host)}` awaits approval**")
                    by = st.text_input("Your name", key=f"fp-host-by-{run_dir.name}-{record['page']}")
                    typed = st.text_input("Type the hostname exactly", key=f"fp-host-typed-{run_dir.name}-{record['page']}")
                    agree = st.checkbox(job_store.HOST_APPROVAL_STATEMENT, key=f"fp-host-agree-{run_dir.name}-{record['page']}")
                    submit = st.form_submit_button("Approve this result host for this page")
                if submit:
                    if not agree or typed.strip().lower() != host:
                        st.error("Tick the statement and type the exact hostname.")
                    else:
                        try:
                            full_pdf.approve_page_host(run_dir, spec, record["page"], host=typed.strip().lower(),
                                                       approved_by=by, statement=job_store.HOST_APPROVAL_STATEMENT,
                                                       root=store.root)
                            st.rerun()
                        except full_pdf.FullPDFError as exc:
                            st.error(str(exc))
        retryable = [item for item in summary["records"]
                     if item["status"] in {full_pdf.FAILED, full_pdf.UNCERTAIN}
                     and item["job_state"] in {job_store.FAILED, job_store.ABANDONED,
                                               job_store.SUBMISSION_UNCONFIRMED}
                     | ({job_store.RESUMABLE_ERROR} if spec.protocol == "openai_chat_completions_v1" else set())
                     and item.get("attempts", 0) < full_pdf.MAX_ATTEMPTS_PER_PAGE]
        if retryable:
            st.markdown("**Retry a failed or unconfirmed page (explicit; never automatic)**")
            with st.form(f"fp-retry-{run_dir.name}"):
                page = st.selectbox("Page", [item["page"] for item in retryable], key=f"fp-retry-page-{run_dir.name}")
                by = st.text_input("Your name", key=f"fp-retry-by-{run_dir.name}")
                agree = st.checkbox(full_pdf.RETRY_STATEMENT, key=f"fp-retry-agree-{run_dir.name}")
                submit = st.form_submit_button("Send this page again")
            if submit:
                if not agree or not credential:
                    st.error("Tick the retry statement and enter the API key.")
                else:
                    try:
                        record = full_pdf.retry_page(run_dir, runner, spec, int(page), workspace_id=workspace_id,
                                                     document=document, pdf_path=source, credential=credential,
                                                     root=store.root, by=by, statement=full_pdf.RETRY_STATEMENT)
                        st.session_state["fp-flash"] = ("info", f"Page {page}: {record['status']} "
                                                                f"({record['status_reason']})")
                        st.rerun()
                    except (full_pdf.FullPDFError, APIRunRefused) as exc:
                        st.error(str(exc))
        st.markdown("**Finalize this run** (writes page records, full_text.txt, report.html and results.json once; "
                    "no page can be sent afterwards)")
        with st.form(f"fp-finalize-{run_dir.name}"):
            if summary["open_jobs"]:
                st.warning(f"{len(summary['open_jobs'])} page(s) have unfinished jobs (pages "
                           f"{', '.join(str(item['page']) for item in summary['open_jobs'])}). Finalizing abandons "
                           "them; nothing is resent.")
            by = st.text_input("Your name", key=f"fp-final-by-{run_dir.name}")
            agree = bool(summary["open_jobs"]) and st.checkbox(
                full_pdf.ABANDON_STATEMENT, key=f"fp-final-agree-{run_dir.name}")
            submit = st.form_submit_button("Finalize now")
        if submit:
            try:
                full_pdf.finalize(run_dir, spec, root=store.root, runner=runner, abandon_by=by,
                                  statement=full_pdf.ABANDON_STATEMENT if agree else None)
                st.rerun()
            except (full_pdf.FullPDFError, APIRunRefused) as exc:
                st.error(str(exc))


def _unconfirmed_submissions(runner, spec) -> None:
    """Manually settle SUBMISSION_UNCONFIRMED journals of this provider (any run).

    Abandoning appends a separate ABANDONED event; nothing is deleted or resent. It only
    lifts the block on submitting the same page image again (a later, deliberate run)."""
    pending = [item for item in runner.journals(spec) if item.state == job_store.SUBMISSION_UNCONFIRMED]
    if not pending:
        return
    with st.expander(f"Advanced Details: unconfirmed submissions ({len(pending)})", expanded=True):
        for journal in pending:
            summary = journal.summary()
            st.text(f"Journal {summary['journal_id']} | page {summary['page']} | Job ID {summary['job_id']} | "
                    f"image {str(summary['image_sha256'])[:16]}...")
            confirm_key = f"fp-abandon-open-{journal.journal_id}"
            if not st.session_state.get(confirm_key):
                if st.button("Abandon Unconfirmed Submission", key=f"fp-abandon-{journal.journal_id}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
                continue
            st.warning(f"{spec.provider} may already have received this request and charged for it. Abandoning "
                       "only records your decision and lifts the block; nothing is resent automatically.")
            with st.form(f"fp-abandon-form-{journal.journal_id}"):
                by = st.text_input("Your name", key=f"fp-abandon-by-{journal.journal_id}")
                reason = st.text_input("Reason", value="Submission unconfirmed (timeout); settled manually",
                                       key=f"fp-abandon-reason-{journal.journal_id}")
                agree = st.checkbox("I understand the request may have been received and billed, and I abandon it.",
                                    key=f"fp-abandon-agree-{journal.journal_id}")
                submit = st.form_submit_button("Confirm abandon")
            if submit:
                if not agree:
                    st.error("Tick the confirmation to abandon.")
                else:
                    try:
                        runner.abandon(spec, journal.journal_id, by=by, reason=reason)
                        st.session_state.pop(confirm_key, None)
                        st.session_state["fp-flash"] = ("info", f"Journal {journal.journal_id} abandoned; "
                                                                "the block is lifted. Nothing was resent.")
                        st.rerun()
                    except APIRunRefused as exc:
                        st.error(str(exc))


def _flash() -> None:
    flash = st.session_state.pop("fp-flash", None)
    if flash:
        {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[flash[0]](flash[1])


def render_full_pdf_view(*, manager, settings, store, ingest, workspace_id) -> None:
    st.markdown("**Upload PDF → Select Model → API Key → Authorize → Run Full PDF → Download ZIP**")
    st.caption(
        "Every page is rendered to an image and sent to the selected hosted model, one page at a time. "
        "Ground Truth and the PDF's text layer are never sent, and no accuracy is computed automatically: "
        "the raw text is saved for manual comparison. Each run gets its own write-once folder under "
        f"{store.root}."
    )
    _flash()
    document = _choose_document(ingest, workspace_id)
    if document is None:
        return
    spec = _choose_model()
    if spec is None:
        return
    credential = _api_key(spec)
    source = ingest.original_path(workspace_id, document.doc_id)
    runner = APIEvaluationRunner(manager, settings, store)
    _unconfirmed_submissions(runner, spec)
    st.subheader("4. Authorize")
    document_ok = _document_gate(document, source, store)
    provider_ok = _provider_gate(spec, store, document.page_count)
    plan = _plan_section(runner, spec, workspace_id, document, source) if document_ok else None
    if plan is not None:
        _run_section(runner, spec, store, workspace_id, document, source, plan, credential,
                     gates_ok=document_ok and provider_ok)
    _results_section(runner, store, workspace_id, document, source)
