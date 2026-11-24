"""Automatic Extraction Mode (page 10): Upload PDF -> Automatic Extraction -> Review Results -> Download ZIP.

There is no approval step, no user name and no per-document authorization. Images go only to the Qari
service the administrator configured and trusts (QARI_SERVICE_URL + QARI_DEPLOYMENT_MODE; see
qari_client.endpoint_info). Nothing runs when the page opens; "Check Qari service" only calls
GET /health (no document content). Tokens come from the environment / nass/.env, or are
entered for this browser session only, and are never shown or saved.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from lab.core.auto import assemble, pipeline, qari_client, regions, router
from lab.core.auto import config as auto_config

SESSION_TOKEN = "auto-qari-token"
FLASH = "auto-flash"


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ""))


@st.cache_resource(show_spinner=False)
def _detector(model_name: str, model_dir: str | None, device: str, threshold: float | None) -> regions.LayoutDetector:
    """One detector per process and setting (the model loads lazily on first use, then stays loaded)."""
    return regions.LayoutDetector(model_name=model_name, model_dir=model_dir, device=device, threshold=threshold)


def _config() -> auto_config.AutoConfig | None:
    try:
        return auto_config.load_config()
    except auto_config.ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        return None


def _token(config: auto_config.AutoConfig) -> str | None:
    return config.qari_token() or st.session_state.get(SESSION_TOKEN) or None


def _client(config: auto_config.AutoConfig) -> qari_client.QariClient:
    return qari_client.QariClient(config.qari.service_url, _token(config), timeout_s=config.qari.request_timeout_s,
                                  connect_retries=config.qari.max_connect_retries,
                                  deployment_mode=config.qari.deployment_mode,
                                  trusted_development_hosts=config.qari.trusted_development_hosts)


def _flash(kind: str, message: str) -> None:
    st.session_state.setdefault(FLASH, []).append((kind, message))


def _show_flash() -> None:
    for kind, message in st.session_state.pop(FLASH, []):
        {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](message)


def _configuration_section(config: auto_config.AutoConfig) -> None:
    with st.expander("Operational configuration (set by the administrator)", expanded=False):
        st.caption(f"Sources (later wins): defaults → {auto_config.DEFAULT_CONFIG_FILE} (or ${auto_config.CONFIG_ENV}) → "
                   f"{auto_config.DEFAULT_DOTENV} → environment variables. Proven extraction settings are separate "
                   "and change only with AUTO_ALLOW_PROVEN_OVERRIDES=true (recorded in the manifest).")
        snapshot = config.snapshot()
        st.json({key: snapshot[key] for key in ("app", "layout", "qari", "processing", "validation")}, expanded=False)
        st.markdown("**Proven settings (benchmark)**")
        st.json(snapshot["proven"], expanded=False)
        differences = auto_config.proven_differences(config)
        if differences:
            st.warning("Proven settings overridden: " + json.dumps(differences, ensure_ascii=False))
        if not config.qari_token():
            st.text_input(f"Qari token for this session ({config.qari.token_env} is not set; never saved)",
                          type="password", key=SESSION_TOKEN)
        st.caption("Changing LAYOUT_MODEL_DIR / LAYOUT_DEVICE / LAYOUT_SCORE_THRESHOLD requires a Streamlit "
                   "restart (the layout model is loaded once per process).")


def _status_section(config: auto_config.AutoConfig) -> None:
    layout = auto_config.layout_model_status(config)
    if layout["ok"]:
        st.caption(f"PP-DocLayout: {config.proven.layout.model_name} · {layout['model_dir']} · device "
                   f"{config.layout.device}")
    else:
        st.error("PP-DocLayout is not ready (needed for scanned pages): " + "; ".join(layout["problems"]))
    try:
        endpoint = pipeline.check_service(config)
    except qari_client.QariClientError as exc:
        st.warning(f"Qari: {exc}. Runs are still planned; OCR jobs wait (resumable) or can be exported for an "
                   "offline Qari run. No other model is used.")
        return
    st.caption(f"Qari: {endpoint['display']} · mode {endpoint['deployment_mode']} · {endpoint['trust_basis']} · "
               f"token {'set' if _token(config) else 'not set'}")
    if st.button("Check Qari service (GET /health only)", key="auto-health"):
        try:
            with _client(config) as client:
                st.session_state["auto-health-result"] = client.health()
        except qari_client.QariClientError as exc:
            st.session_state["auto-health-result"] = {"status": "unavailable", "error": str(exc)}
    health = st.session_state.get("auto-health-result")
    if health:
        (st.success if health.get("status") == "ready" else st.error)(
            f"Status: {health.get('status')} · engine {json.dumps(health.get('engine'))} · running "
            f"{health.get('running')} · waiting {health.get('waiting')} · {health.get('error') or ''}")


def _progress():
    bar = st.progress(0.0)
    label = st.empty()

    def update(stage: str, done: int, total: int) -> None:
        bar.progress(min(1.0, done / total) if total else 0.0)
        label.caption(f"{stage}: {done}/{total}")

    return update


def _run_ocr(run_dir: Path, config: auto_config.AutoConfig) -> dict | None:
    """Send the pending jobs. An untrusted, disabled or unreachable service pauses the run (nothing sent)."""
    try:
        with _client(config) as client:
            summary = pipeline.execute(run_dir, config, client, on_progress=_progress(),
                                       health=st.session_state.get("auto-health-result"))
            if not summary.get("stopped"):
                # Report-only second read of crops with Arabic-Indic/Persian digits; never fails the run.
                try:
                    summary["digit_verification"] = pipeline.verify_digits(run_dir, config, client,
                                                                           on_progress=_progress())
                except Exception as exc:  # noqa: BLE001 - verification must never block the extraction
                    summary["digit_verification"] = {"stopped": f"{type(exc).__name__}: {exc}"}
    except (qari_client.QariClientError, pipeline.PipelineError) as exc:
        _flash("warning", f"Paused, nothing sent: {exc}. The run is saved and can be resumed.")
        return None
    if summary.get("stopped"):
        _flash("warning", f"Paused: {summary['stopped']}. The run is saved and can be resumed.")
    verification = summary.get("digit_verification") or {}
    if verification.get("stopped"):
        _flash("info", f"Digit second read: {verification['stopped']} (report only; the extraction is unaffected)")
    return summary


def _finalize_if_complete(run_dir: Path, config: auto_config.AutoConfig) -> None:
    counts = pipeline.status(run_dir)["counts"]
    if counts["pending"] + counts["unconfirmed"] + counts["failed"]:
        return
    try:
        assemble.finalize(run_dir, config)
        _flash("success", "Finalized: document.md, document.json, validation_report.json, rag_chunks.jsonl, "
                          "execution_manifest.json")
    except pipeline.PipelineError as exc:
        _flash("warning", f"Not finalized: {exc}")


def _start(document, source: Path, config: auto_config.AutoConfig) -> None:
    if not st.button("Automatic Extraction", type="primary", key="auto-start"):
        return
    detector = _detector(config.proven.layout.model_name, config.layout.model_dir, config.layout.device,
                         config.layout.score_threshold)
    try:
        with st.spinner("Planning locally: Phase 1 analysis, routing, layout detection, crops"):
            run_dir = pipeline.create_run(source, display_name=document.display_name, config=config,
                                          detector=detector, on_progress=_progress())
    except Exception as exc:  # noqa: BLE001 - shown; nothing was sent
        st.error(f"Planning failed: {type(exc).__name__}: {exc}")
        return
    st.session_state["auto-run"] = str(run_dir)
    if pipeline.pending_jobs(run_dir):
        _run_ocr(run_dir, config)
    _finalize_if_complete(run_dir, config)
    st.rerun()


def _run_view(run_dir: Path, config: auto_config.AutoConfig) -> None:
    plan = pipeline.load_plan(run_dir)
    status = pipeline.status(run_dir)
    finalized = status["finalized"]
    st.markdown(f"**Run** `{run_dir.name}` · {'finalized (immutable)' if finalized else 'open'} · "
                f"jobs {status['counts']}")

    st.markdown("**Pages**")
    rows = "".join(
        f"<tr><td>{r['page']}</td><td><b>{_esc(r['route'])}</b></td><td>{_esc(r['rule'])}</td>"
        f"<td>{_esc(r['reason'])}</td><td>{_esc(r['evidence']['phase1_category'])}</td>"
        f"<td>{r['evidence']['page_image_share']:.0%}</td></tr>" for r in plan["routing"]["pages"])
    st.markdown("<table><tr><th>Page</th><th>Route</th><th>Rule</th><th>Reason</th><th>Phase 1</th>"
                f"<th>Largest image</th></tr>{rows}</table>", unsafe_allow_html=True)
    st.caption(f"Counts: {plan['routing']['counts']} (router v{router.ROUTER_VERSION})")

    if not finalized:
        _actions(run_dir, config, status)
    _review_queue(run_dir, config, finalized)

    try:
        document = assemble.assemble(run_dir)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Assembly preview failed: {type(exc).__name__}: {exc}")
        return
    report = document["validation"]
    st.markdown(f"**Validation**: {report['status']} · {report['summary']}")
    for flag in report["flags"][:200]:
        if flag["severity"] == "info":
            continue
        (st.error if flag["severity"] == "error" else st.warning)(
            f"p{flag.get('page')} {flag.get('region_id') or flag.get('job_id') or flag.get('element_id') or ''} "
            f"[{flag['code']}] {flag['message']}")
    with st.expander("Document preview (Markdown)", expanded=True):
        st.markdown(document["markdown"])
    st.caption("Reading order per page: " + ", ".join(f"p{p['page']}: {p['reading_order_method']}"
                                                      for p in document["pages"]))
    if not finalized:
        incomplete = st.checkbox("Finalize with the open OCR jobs flagged as incomplete",
                                 key=f"auto-fin-inc-{run_dir.name}")
        if st.button("Finalize", key=f"auto-fin-{run_dir.name}"):
            try:
                assemble.finalize(run_dir, config, allow_incomplete=incomplete)
                st.rerun()
            except pipeline.PipelineError as exc:
                st.error(str(exc))
    if st.button("Prepare ZIP", key=f"auto-zip-{run_dir.name}"):
        st.session_state[f"auto-zip-{run_dir.name}-data"] = assemble.zip_bytes(run_dir)
    data = st.session_state.get(f"auto-zip-{run_dir.name}-data")
    if data:
        st.download_button("Download ZIP", data, file_name=f"{run_dir.name}.zip", mime="application/zip",
                           key=f"auto-zip-dl-{run_dir.name}")


def _actions(run_dir: Path, config: auto_config.AutoConfig, status: dict) -> None:
    counts = status["counts"]
    if counts["pending"] and st.button(f"Resume ({counts['pending']} pending)", key=f"auto-resume-{run_dir.name}"):
        _run_ocr(run_dir, config)
        _finalize_if_complete(run_dir, config)
        st.rerun()
    open_rows = [row for row in status["jobs"] if row["bucket"] in {"unconfirmed", "failed"}]
    if open_rows:
        st.warning(f"{len(open_rows)} job(s) without a confirmed result. Unconfirmed requests are never resent "
                   "automatically: 'Recover' asks the service for its saved result (no image is sent); "
                   "'Retry' sends the image again as a new attempt.")
        for row in open_rows[:50]:
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.caption(f"{row['job_id']} · {row['bucket']} · attempts {row['attempts']}")
            if row["bucket"] == "unconfirmed" and c2.button("Recover", key=f"auto-rec-{row['job_id']}"):
                try:
                    with _client(config) as client:
                        found = pipeline.recover_job(run_dir, row["job_id"], client)
                    _flash("info", "recovered" if found else "the service has no saved result for this request")
                    st.rerun()
                except (qari_client.QariClientError, pipeline.PipelineError) as exc:
                    st.error(str(exc))
            if c3.button("Retry", key=f"auto-retry-{row['job_id']}"):
                try:
                    with _client(config) as client:
                        pipeline.retry_job(run_dir, row["job_id"], config, client)
                    st.rerun()
                except (qari_client.QariClientError, pipeline.PipelineError) as exc:
                    st.error(str(exc))
    with st.expander("Offline Qari (when the service is unavailable)"):
        if st.button("Prepare offline job ZIP (pending images only)", key=f"auto-off-{run_dir.name}"):
            st.session_state[f"auto-off-{run_dir.name}-data"] = pipeline.export_offline(run_dir)
        data = st.session_state.get(f"auto-off-{run_dir.name}-data")
        if data:
            st.download_button("Download offline jobs", data, file_name=f"{run_dir.name}_qari_jobs.zip",
                               key=f"auto-off-dl-{run_dir.name}")
        uploaded = st.file_uploader("results.jsonl from qari_offline_batch.py", type=["jsonl"],
                                    key=f"auto-imp-{run_dir.name}")
        if uploaded is not None and st.button("Import results", key=f"auto-imp-btn-{run_dir.name}"):
            try:
                record = pipeline.import_offline(run_dir, uploaded.getvalue())
                st.success(f"Imported {len(record['imported'])}; skipped {len(record['skipped'])}")
                if record["skipped"]:
                    st.json(record["skipped"], expanded=False)
            except (pipeline.PipelineError, UnicodeDecodeError) as exc:
                st.error(str(exc))


def _review_queue(run_dir: Path, config: auto_config.AutoConfig, finalized: bool) -> None:
    queue = pipeline.review_queue(run_dir)
    st.markdown(f"**Review queue** ({len(queue)} region(s))")
    for item in queue:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            if item["image_file"] and (run_dir / item["image_file"]).is_file():
                c1.image(str(run_dir / item["image_file"]), caption=item["region_id"])
            c2.markdown(f"**{_esc(item['region_id'])}** · {_esc(item['label'])} · {_esc(item['decision'])} · "
                        f"status {_esc(item['status'])}")
            c2.caption(f"box (pt, before margin) {item['bbox_pt']} · crop px {item['pixel_box']}")
            for reason in item["reasons"]:
                c2.caption(f"• {reason}")
            if item.get("proposed_expansion_pt"):
                c2.caption(f"proposed expansion (not applied): {item['proposed_expansion_pt']}")
            if item["decision"] == regions.REVIEW and not item["requested"] and not finalized:
                if c2.button("Process now", key=f"auto-pn-{item['region_id']}"):
                    try:
                        pipeline.process_now(run_dir, item["region_id"])
                        _run_ocr(run_dir, config)
                        st.rerun()
                    except pipeline.PipelineError as exc:
                        st.error(str(exc))
            elif item["requested"]:
                c2.caption("Process now requested")


def render_automatic(*, document, source: Path) -> None:
    config = _config()
    if config is None:
        return
    _show_flash()
    _status_section(config)
    _start(document, source, config)
    _configuration_section(config)
    runs = [r for r in pipeline.list_runs(config) if r.get("sha256") == document.sha256]
    if not runs:
        return
    names = [r["run_dir"] for r in runs]
    wanted = st.session_state.get("auto-run")
    chosen = st.selectbox("Run", names, index=names.index(wanted) if wanted in names else 0,
                          format_func=lambda path: Path(path).name + (" (finalized)" if pipeline.is_finalized(Path(path))
                                                                      else ""), key="auto-run-select")
    st.session_state["auto-run"] = chosen
    _run_view(Path(chosen), config)
