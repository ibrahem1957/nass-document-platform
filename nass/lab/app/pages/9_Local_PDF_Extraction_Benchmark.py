"""Local PDF Extraction Benchmark — thin UI over `lab.local_pdf_extraction`.

Standalone: independent of the Two-Stage API Evaluation and of the Baidu / Qwen /
Qari experiments. Text-layer extraction only (no OCR, no LLM, no network).
Results live under <results root>/local_pdf_extraction/ and are reloaded from disk.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from lab.local_pdf_extraction import (arabic_reorder, config, diagnostics, exports, extractors, ocr_links, runner,
                                      storage, tables)

st.set_page_config(page_title="Local PDF Extraction Benchmark", layout="wide")
ROOT = config.benchmark_root()
RUN_PARAM = "lpe_run"
TEXT_BOX = ("white-space:pre-wrap; unicode-bidi:plaintext; max-height:460px; overflow:auto; "
            "border:1px solid rgba(128,128,128,.4); border-radius:6px; padding:8px; font-size:.9rem; line-height:1.55;")
PROCESSED_BOX = "border:2px dashed #c98a00; border-radius:8px; padding:6px 10px; margin-top:4px;"
KNOWN_LIMITS = """
**Processed method — known limits (heuristics, not proven rules)**

- Applied per page and only when needed: pages without Arabic/Hebrew are `not_applicable`; pages whose raw text
  already matches, or that show no repair signal, keep the raw PyMuPDF text (`not_needed`).
- `reconstructed` means the repair changed the text and passed the character-conservation check — it is **not
  verified against ground truth**.
- Bracket placeholders (`¦ § ¶ †`) are converted back to brackets; a page that genuinely contains them is marked
  `uncertain`.
- Spot-checked on one Word-exported PDF only; other producers (Google Docs, LibreOffice, scanners) may differ.
  On synthetic PDFs whose ToUnicode already maps lam-alef logically, the reconstruction would *damage* correct
  text — those pages are kept raw or marked `uncertain`.
- Glyphs without a Unicode mapping cannot be recovered without OCR (e.g. lam-alef extracted as `ǁ ƾ Ǆ`).
- In the page text, tables come out as rows of `|`-separated cells; cell order is not guaranteed.
- Optional structured tables (PyMuPDF `find_tables()`) are a separate processed output, not verified cell by
  cell: cells whose numbers/dates/percentages/Latin tokens changed are `uncertain` (⚠); tables inside images
  are not detected; tables are never merged across pages. For any table region, use either the page text or the
  structured table, never both.
"""


# ------------------------------------------------------------------ helpers
def _esc(text: str) -> str:
    """Escape for an HTML block inside st.markdown without changing what is displayed."""
    return (html.escape(text).replace("$", "&#36;").replace("\r", "&#13;").replace("\n", "&#10;"))


def _text_box(text: str) -> None:
    st.markdown(f'<div dir="auto" style="{TEXT_BOX}">{_esc(text)}</div>', unsafe_allow_html=True)


def _html_table(header: list[str], rows: list[list[str]], separator_before: dict[int, str] | None = None) -> None:
    cells = "".join(f"<th style='text-align:left;padding:4px 8px'>{html.escape(h)}</th>" for h in header)
    body = []
    for index, row in enumerate(rows):
        if separator_before and index in separator_before:
            body.append(f"<tr><td colspan='{len(header)}' style='padding:6px 8px;border-top:3px double #c98a00;"
                        f"font-weight:600'>{html.escape(separator_before[index])}</td></tr>")
        body.append("<tr>" + "".join(f"<td style='padding:3px 8px;border-top:1px solid rgba(128,128,128,.3)'>{cell}</td>"
                                     for cell in row) + "</tr>")
    st.markdown(f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:.88rem'>"
                f"<tr>{cells}</tr>{''.join(body)}</table></div>", unsafe_allow_html=True)


def _size(num: int) -> str:
    return f"{num / 1024 / 1024:.2f} MB" if num >= 1024 * 1024 else f"{num / 1024:.1f} KB"


def _current_run() -> str | None:
    run_id = st.session_state.get("lpe-current") or st.query_params.get(RUN_PARAM)
    if run_id and storage.RUN_ID_PATTERN.match(run_id) and (ROOT / run_id / "manifest.json").exists():
        return run_id
    return None


def _set_run(run_id: str) -> None:
    st.session_state["lpe-current"] = run_id
    st.query_params[RUN_PARAM] = run_id


def _quick_status(run_path: Path) -> dict:
    """Cheap status for lists/progress (state.json + lock). Full verification lives in Results."""
    state = storage.load_state(run_path) or {}
    manifest = storage.load_manifest(run_path)
    lock = storage.lock_status(run_path)
    methods = state.get("methods", {})
    total = sum(m.get("total", 0) for m in methods.values())
    done = sum(m.get("done", 0) + m.get("failed", 0) for m in methods.values())
    worker_done = bool((state.get("worker") or {}).get("finished_at")) and bool(manifest.get("finished_at"))
    if lock["alive"]:
        status = "RUNNING"
    elif worker_done and manifest.get("status") not in (None, "PENDING", "RUNNING"):
        # The worker finished normally and wrote the verified final status to the manifest (older
        # runs could still say RUNNING in state.json because of the fixed lock-timing bug).
        status = manifest["status"]
    elif state.get("status") in ("RUNNING", "STARTING") or (state.get("status") == "PENDING" and state.get("attempt")):
        status = "INTERRUPTED"
    else:
        status = manifest.get("status") or state.get("status") or "PENDING"
    label = status if status in ("COMPLETED", "PENDING") else f"{status} — {done}/{total} pages"
    return {"status": status, "label": label, "state": state, "manifest": manifest, "lock": lock,
            "done": done, "total": total}


def _run_choices() -> list[str]:
    return [path.name for path in storage.list_runs(ROOT)]


def _run_picker(label: str) -> str | None:
    """Key-less on purpose: the index follows the current run (set from any tab or the URL)."""
    choices = _run_choices()
    if not choices:
        st.info("No saved runs yet.")
        return None
    current = _current_run()
    index = choices.index(current) if current in choices else 0
    chosen = st.selectbox(label, choices, index=index)
    if chosen != current:
        _set_run(chosen)
    return chosen


# ================================================================== page
st.title("Local PDF Extraction Benchmark")
st.caption("Direct text-layer extraction with local Python libraries — no OCR, no LLMs, no cloud services. "
           f"Results are saved permanently under `{ROOT}`.")
tab_new, tab_progress, tab_results, tab_runs = st.tabs(["New run", "Progress", "Results", "Saved runs"])

# ----------------------------------------------------------------- new run
with tab_new:
    st.subheader("Libraries")
    availability = {method.method_id: extractors.availability(method) for method in extractors.METHODS}
    rows = []
    for method in extractors.METHODS:
        info = availability[method.method_id]
        state = "available" if info["available"] else f"<span style='color:#c00'>unavailable</span> {html.escape(info['error'] or '')}"
        rows.append([html.escape(method.label), html.escape(method.output_type.upper()),
                     html.escape(info["version"] or "—"), state, html.escape(method.api),
                     html.escape(method.note)])
    _html_table(["Library / method", "Output", "Version", "Status", "API", "Note"], rows,
                {len(extractors.RAW_METHOD_IDS): "PROCESSED — not raw library output"})

    st.subheader("PDF files")
    uploads = st.file_uploader("Upload one or several PDF files (Arabic or English)", type=["pdf"],
                               accept_multiple_files=True, key="lpe-upload")
    valid_uploads: list[tuple[str, bytes]] = []
    valid_infos: list[dict] = []
    if uploads:
        file_rows = []
        for upload in uploads:
            data = upload.getvalue()
            try:
                info = storage.inspect_upload(upload.name, data, config.MAX_FILE_MB, config.MAX_PAGES_PER_FILE)
                valid_uploads.append((upload.name, data))
                valid_infos.append(info)
                file_rows.append([f"<span dir='auto'>{html.escape(upload.name)}</span>", str(info["page_count"]),
                                  _size(info["size_bytes"]), html.escape(info["sha256"][:16]) + "…", "ok"])
            except Exception as error:  # noqa: BLE001 - show, do not crash the page
                file_rows.append([f"<span dir='auto'>{html.escape(upload.name)}</span>", "—", _size(len(data)), "—",
                                  f"<span style='color:#c00'>{html.escape(str(error))}</span>"])
        _html_table(["File", "Pages", "Size", "SHA-256", "Check"], file_rows)

    # Ground truth for the new run: saved into the run BEFORE extraction starts, so the
    # worker's summary.json already contains CER/WER.
    new_ground_truth: list[tuple[str, list[str]] | None] = []
    gt_rows: list[list[str]] = []
    gt_errors = False
    if valid_uploads:
        st.subheader("Ground truth (optional)")
        st.caption("One .txt per PDF (use `=== page N ===` separators for per-page scoring), or one file per page "
                   "named with its page number (e.g. page_3.txt). UTF-8. CER/WER are computed only where ground "
                   "truth is attached.")
        for index, ((name, _), info) in enumerate(zip(valid_uploads, valid_infos)):
            files = st.file_uploader(f"Ground truth for {name}", type=["txt"], accept_multiple_files=True,
                                     key=f"lpe-new-gt-{index}-{info['sha256'][:8]}")
            if not files:
                new_ground_truth.append(None)
                gt_rows.append([f"<span dir='auto'>{html.escape(name)}</span>", "not attached",
                                "CER/WER will NOT be computed"])
                continue
            try:
                text = diagnostics.combine_ground_truth_files([(f.name, f.getvalue()) for f in files])
                described = diagnostics.describe_ground_truth(text, info["page_count"])
                if described["level"] is None:
                    raise ValueError("the ground-truth file is empty")
                new_ground_truth.append((text, [f.name for f in files]))
                gt_rows.append([f"<span dir='auto'>{html.escape(name)}</span>",
                                f"<b>attached</b> ({html.escape(', '.join(f.name for f in files))})",
                                html.escape(described["message"]) + " → CER/WER will be computed"])
            except ValueError as error:
                gt_errors = True
                new_ground_truth.append(None)
                gt_rows.append([f"<span dir='auto'>{html.escape(name)}</span>",
                                f"<span style='color:#c00'>invalid: {html.escape(str(error))}</span>",
                                "fix or remove the file before running"])
        _html_table(["PDF", "Ground truth", "Effect"], gt_rows)

    st.subheader("Select libraries")

    def _select_all(value: bool) -> None:
        for method_id in extractors.RAW_METHOD_IDS:
            st.session_state[f"lpe-m-{method_id}"] = value

    buttons = st.columns([1, 1, 4])
    buttons[0].button("Select All", on_click=_select_all, args=(True,), key="lpe-select-all")
    buttons[1].button("Clear", on_click=_select_all, args=(False,), key="lpe-select-none")
    columns = st.columns(len(extractors.RAW_METHOD_IDS))
    chosen: list[str] = []
    for column, method_id in zip(columns, extractors.RAW_METHOD_IDS):
        method = extractors.METHOD_BY_ID[method_id]
        key = f"lpe-m-{method_id}"
        if key not in st.session_state:
            st.session_state[key] = method.default_selected and availability[method_id]["available"]
        if column.checkbox(f"{method.label} (raw)", key=key, disabled=not availability[method_id]["available"]):
            chosen.append(method_id)
    processed_key = f"lpe-m-{extractors.PROCESSED_METHOD_ID}"
    if processed_key not in st.session_state:
        st.session_state[processed_key] = availability[extractors.PROCESSED_METHOD_ID]["available"]
    use_processed = st.checkbox(arabic_reorder.METHOD_LABEL, key=processed_key,
                                disabled=not availability[extractors.PROCESSED_METHOD_ID]["available"],
                                help="Adds a separate, clearly labeled PROCESSED output. Raw outputs are never modified.")
    use_tables = st.checkbox("Extract tables with the processed method (PyMuPDF find_tables(), saved separately)",
                             value=True, key="lpe-tables", disabled=not use_processed,
                             help="Saves page_NNN_tables.json/.md next to the processed text. The page text is "
                                  "not changed; table regions are recorded so a consumer uses one or the other.")
    with st.expander("Processed method: thresholds and known limits"):
        st.markdown(KNOWN_LIMITS)
        defaults = arabic_reorder.ReorderConfig()
        c1, c2, c3 = st.columns(3)
        line_tol = c1.number_input("Line tolerance (× glyph height)", 0.05, 2.0, defaults.line_tolerance_ratio, 0.05)
        space_gap = c2.number_input("Space gap (× glyph height)", 0.01, 1.0, defaults.space_gap_ratio, 0.01)
        relative_cells = c3.checkbox("Cell gap relative to glyph height", value=False,
                                     help="The prototype uses a fixed 18 pt; enable to scale with font size.")
        if relative_cells:
            cell_ratio = c3.number_input("Cell gap (× glyph height)", 0.2, 10.0, 1.5, 0.1)
            cell_pt = defaults.cell_gap_pt
        else:
            cell_ratio = None
            cell_pt = c3.number_input("Cell gap (pt)", 1.0, 200.0, defaults.cell_gap_pt, 1.0)
        min_share = st.number_input(
            "Minimum Arabic letter share to reconstruct a page", 0.0, 1.0, defaults.min_arabic_letter_share, 0.05,
            key="lpe-min-arabic-share",
            help="Share of Arabic letters among all letters in the raw PyMuPDF text. Below it the page is mainly "
                 "LTR: the raw text is kept unchanged (not_needed) and the single-letter 'Needs OCR?' rule is not "
                 "applied (the ratio is still recorded). Tables are unaffected. 0 disables the gate.")
    reorder_config = arabic_reorder.ReorderConfig(line_tolerance_ratio=float(line_tol), space_gap_ratio=float(space_gap),
                                                  cell_gap_pt=float(cell_pt),
                                                  cell_gap_ratio=None if cell_ratio is None else float(cell_ratio),
                                                  min_arabic_letter_share=float(min_share))
    with st.expander("'Needs OCR?' rule (recommendation only; OCR is never run)"):
        st.caption("A processed page is flagged when the share of Arabic tokens that are a single Arabic letter "
                   f"exceeds the threshold: '{diagnostics.SPLIT_WORDS_REASON}' (e.g. Chrome/Skia PDFs: \"ا خ ت بار\"). "
                   "Pages with fewer Arabic tokens than the minimum are not judged by this rule.")
        d1, d2 = st.columns(2)
        single_threshold = d1.number_input("Single-letter Arabic token ratio threshold", 0.01, 1.0,
                                           diagnostics.DEFAULT_SINGLE_LETTER_THRESHOLD, 0.01, key="lpe-ocr-threshold")
        single_min_tokens = d2.number_input("Minimum Arabic tokens on the page", 1, 1000,
                                            diagnostics.DEFAULT_SINGLE_LETTER_MIN_TOKENS, 1, key="lpe-ocr-min-tokens")
    selected = chosen + ([extractors.PROCESSED_METHOD_ID] if use_processed else [])
    if valid_uploads:
        attached = sum(1 for item in new_ground_truth if item is not None)
        (st.info if attached else st.warning)(
            f"Ground truth attached for {attached} of {len(valid_uploads)} PDF(s). "
            + ("CER/WER will be computed for those after the run." if attached
               else "No CER/WER will be computed for this run."))
    if st.button("Run Extraction", type="primary", disabled=not valid_uploads or not selected or gt_errors,
                 key="lpe-run"):
        try:
            run_path = storage.create_run(ROOT, valid_uploads, runner.method_entries(selected, reorder_config.to_dict()),
                                          {"reorder": reorder_config.to_dict(), "viewer_dpi": config.VIEWER_DPI,
                                           "tables": {"enabled": bool(use_processed and use_tables),
                                                      "version": tables.TABLES_VERSION},
                                           "diagnostics": {"version": diagnostics.DIAGNOSTICS_VERSION,
                                                           "single_letter_threshold": float(single_threshold),
                                                           "single_letter_min_arabic_tokens": int(single_min_tokens),
                                                           "applies_to": "processed method"}},
                                          max_mb=config.MAX_FILE_MB, max_pages=config.MAX_PAGES_PER_FILE,
                                          ground_truth=new_ground_truth)
            pid = runner.launch_worker(run_path)
            _set_run(run_path.name)
            st.success(f"Started {run_path.name} (local worker process {pid}). Follow it in the Progress tab.")
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not start: {error}")


# ---------------------------------------------------------------- progress
def _progress_body(run_id: str) -> None:
    run_path = ROOT / run_id
    quick = _quick_status(run_path)
    state = quick["state"]
    st.markdown(f"**{run_id}** — status: **{quick['label']}**")
    if state.get("message"):
        st.caption(state["message"])
    labels = {m["method_id"]: m["label"] for m in quick["manifest"]["methods"]}
    for method_id, info in state.get("methods", {}).items():
        status = info.get("status", "PENDING")
        if status == "RUNNING" and not quick["lock"]["alive"]:
            status = "INTERRUPTED"
        done, failed, total = info.get("done", 0), info.get("failed", 0), max(info.get("total", 0), 1)
        kind = "PROCESSED · " if method_id == extractors.PROCESSED_METHOD_ID else ""
        text = (f"{kind}{labels.get(method_id, method_id)} — {done + failed}/{total} pages · {status}"
                f"{f' · {failed} failed' if failed else ''} · {info.get('elapsed_ms', 0) / 1000:.2f} s")
        if info.get("current") and status == "RUNNING":
            text += f" · now: {info['current']}"
        st.progress(min(1.0, (done + failed) / total), text=text)
    if quick["lock"]["alive"]:
        cols = st.columns(2)
        if cols[0].button("Stop after the current page", key=f"lpe-cancel-{run_id}"):
            storage.request_cancel(run_path)
            st.info("Cancel requested; the worker stops after the current page.")
        if cols[1].button("Force-stop worker", key=f"lpe-kill-{run_id}",
                          help="Only if a library hangs and the worker ignores the cancel request."):
            runner.terminate_worker(run_path)
            st.warning("Worker terminated. The run is INTERRUPTED; use Resume to continue.")
    elif st.session_state.get(f"lpe-was-running-{run_id}"):
        st.session_state[f"lpe-was-running-{run_id}"] = False
        st.rerun()
    if quick["lock"]["alive"]:
        st.session_state[f"lpe-was-running-{run_id}"] = True


_progress_live = st.fragment(run_every=2)(_progress_body)
_progress_static = st.fragment(_progress_body)

with tab_progress:
    progress_run = _run_picker("Run to monitor")
    if progress_run:
        run_path = ROOT / progress_run
        lock = storage.lock_status(run_path)
        (_progress_live if lock["alive"] else _progress_static)(progress_run)
        if lock["locked"] and not lock["alive"]:
            st.warning(f"Stale lock: worker process {lock.get('pid')} is not running. It is ignored only after "
                       "you clear it explicitly.")
            if st.button("Clear stale lock", key="lpe-clear-lock"):
                storage.clear_stale_lock(run_path)
                st.rerun()
        if not lock["locked"]:
            st.subheader("Resume")
            st.caption("Resume re-verifies every saved page first, then processes only MISSING pages. Corrupt, "
                       "incomplete or failed pages are re-extracted only if you approve them below. Valid "
                       "completed pages are never overwritten.")
            if st.button("Verify saved pages", key="lpe-verify"):
                st.session_state[f"lpe-scan-{progress_run}"] = storage.scan_run(run_path)
            scan = st.session_state.get(f"lpe-scan-{progress_run}")
            if scan:
                manifest = storage.load_manifest(run_path)
                rows = [[html.escape(m["label"]), str(scan["methods"][m["method_id"]]["valid"]),
                         str(scan["methods"][m["method_id"]]["missing"]), str(scan["methods"][m["method_id"]]["invalid"]),
                         str(scan["methods"][m["method_id"]]["failed"])] for m in manifest["methods"]]
                _html_table(["Method", "Valid", "Missing", "Corrupt/incomplete", "Failed"], rows)
                issue_labels = {f"{i['state'].upper()} · {i['document']} · {i['method_id']} · page {i['page']} — {i['reason']}": i
                                for i in scan["issues"]}
                approved = st.multiselect("Approve re-extraction of these pages (their old files are moved to "
                                          "_quarantine/ first)", list(issue_labels), key=f"lpe-approve-{progress_run}")
                method_ids = [m["method_id"] for m in manifest["methods"]]
                resume_methods = st.multiselect("Libraries to include in this resume", method_ids, default=method_ids,
                                                key=f"lpe-resume-methods-{progress_run}")
                pending = scan["counts"]["missing"] + len(approved)
                if st.button(f"Resume ({pending} page results to produce)", disabled=pending == 0 or not resume_methods,
                             key="lpe-resume"):
                    request = runner.create_resume_request(run_path, resume_methods, [issue_labels[a] for a in approved])
                    pid = runner.launch_worker(run_path, request)
                    st.session_state.pop(f"lpe-scan-{progress_run}", None)
                    st.success(f"Resume started (worker {pid}).")
                    st.rerun()


# ----------------------------------------------------------------- results
@st.cache_data(max_entries=4, show_spinner="Loading saved results...")
def _collect(run_path: str, stamp: tuple) -> dict:
    return exports.collect(Path(run_path))


def _stamp(run_path: Path) -> tuple:
    def mtime(path: Path) -> int:
        return path.stat().st_mtime_ns if path.exists() else 0
    gt = run_path / "ground_truth"
    return (mtime(run_path / "state.json"), mtime(run_path / "manifest.json"),
            max([mtime(p) for p in gt.rglob("ground_truth.txt")] or [0]), st.session_state.get("lpe-reload", 0))


def _pct(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def _fmt(value, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _render_output(method: dict, output: dict) -> None:
    record = output["record"] or {}
    st.markdown(f"**{html.escape(method['label'])}**" + ("" if method["output_type"] == "raw" else " · PROCESSED"))
    if output["text"] is None:
        reason = record.get("error") if output["verification"] == "failed" else output["verification"]
        st.error(f"NOT AVAILABLE — {reason}")
        return
    diag_line = record.get("diagnostics") or {}
    single = ""
    if diag_line.get("arabic_tokens"):
        single = (f" · single-letter Arabic tokens {_pct(diag_line.get('single_letter_arabic_ratio'))} "
                  f"({diag_line.get('single_letter_arabic_tokens')}/{diag_line.get('arabic_tokens')})")
    st.caption(f"{record.get('extraction_ms', 0):.1f} ms · {record.get('char_count', 0)} chars · "
               f"{record.get('word_count', 0)} words · {method['library']} {method.get('library_version') or ''}"
               + single)
    reorder = record.get("reorder")
    if reorder:
        status = reorder.get("reorder_status")
        note = {"reconstructed": "reconstructed, not verified against ground truth",
                "uncertain": "uncertain — potentially unreliable, compare with raw PyMuPDF"}.get(status, status)
        (st.warning if status == "uncertain" else st.caption)(f"reorder_status: {note} ({reorder.get('decision')})")
        for warning in reorder.get("warnings") or []:
            st.caption(warning)
    diag = record.get("diagnostics") or {}
    if diag.get("needs_ocr"):
        st.caption(f"Needs OCR? recommended — {diag.get('needs_ocr_reason')} (recommendation only; OCR is not run)")
    _text_box(output["text"])


@st.cache_data(ttl=120, show_spinner="Looking for earlier OCR/VLM results (read-only)...")
def _earlier_ocr(results_root: str, sha256: str) -> dict:
    return ocr_links.find_previous_ocr(Path(results_root), sha256)


def _render_tables(document: dict, page: dict, method: dict) -> None:
    """Detected tables of the processed method (separate output; the page text is unchanged)."""
    output = page["outputs"][method["method_id"]]
    doc = output.get("tables")
    st.markdown("**Tables** — PROCESSED output (PyMuPDF find_tables() + per-cell reconstruction), not raw "
                "library output; not verified cell by cell")
    if doc is None:
        st.caption("No table data saved for this page (run created without table extraction, or the page result "
                   "is not available).")
        return
    if doc.get("error"):
        st.warning(f"Table detection failed: {doc['error']}")
    notes = [tables.clean_note(note) for note in doc.get("notes", [])]
    if not doc.get("tables") and not doc.get("error"):
        st.caption("No table detected on this page." + (" " + " ".join(notes) if notes else ""))
    else:
        for note in notes:
            st.caption(note)
    for table in doc.get("tables", []):
        header = (f"Table {table['table_index']} — {table['rows']}×{table['cols']} · status: {table['status']} · "
                  f"{table['column_order']}")
        (st.warning if table["status"] == "uncertain" else st.caption)(header)
        for warning in table["warnings"]:
            st.caption(f"⚠ {warning}")
        continuation = table.get("possible_continuation_of")
        if continuation:
            st.caption(f"Possible continuation of page {continuation['page']} table {continuation['table_index']} "
                       "(recorded only; not merged).")
        st.markdown("<div style='overflow-x:auto'>" + exports.table_html(table).replace("$", "&#36;") + "</div>",
                    unsafe_allow_html=True)
        with st.expander(f"Table {table['table_index']}: raw_cell_text vs processed_cell_text"):
            cells = table["cells"]
            labels = [f"row {c['row']} · col {c['col']} · {c['reorder_status']}" for c in cells]
            choice = st.selectbox("Cell", range(len(cells)), format_func=lambda i: labels[i],
                                  key=f"lpe-cell-{document['doc_key']}-{page['page']}-{table['table_index']}")
            cell = cells[choice]
            left, right = st.columns(2)
            with left:
                st.caption("raw_cell_text (content-stream order, no reordering)")
                _text_box(cell["raw_cell_text"])
            with right:
                st.caption("processed_cell_text")
                _text_box(cell["processed_cell_text"])
            if cell.get("reason"):
                st.caption(cell["reason"])
            if cell.get("value_check"):
                st.code(json.dumps(cell["value_check"], ensure_ascii=False, indent=1), language="json")


def _render_page(run_path: Path, document: dict, page: dict, raw_methods: list[dict], processed: list[dict],
                 earlier: dict | None = None, show_tables: bool = False) -> None:
    with st.container(border=True):
        st.markdown(f"**Page {page['page']}** · <span dir='auto'>{html.escape(document['original_name'])}</span>",
                    unsafe_allow_html=True)
        columns = st.columns([1.1] + [1] * max(1, len(raw_methods)))
        with columns[0]:
            try:
                st.image(str(storage.page_image(run_path, document["doc_key"], page["page"], config.VIEWER_DPI)),
                         width="stretch")
            except Exception as error:  # noqa: BLE001
                st.warning(f"Page image unavailable: {error}")
        for column, method in zip(columns[1:], raw_methods):
            with column:
                _render_output(method, page["outputs"][method["method_id"]])
        if processed:
            st.markdown(f"<div style='{PROCESSED_BOX}'><b>PROCESSED output — not raw library output</b></div>",
                        unsafe_allow_html=True)
            for method in processed:
                _render_output(method, page["outputs"][method["method_id"]])
                if show_tables:
                    _render_tables(document, page, method)
        if earlier:
            st.markdown("<div style='border:2px dotted #3a7bd5;border-radius:8px;padding:6px 10px;margin-top:4px'>"
                        "<b>OCR/VLM outputs from earlier runs</b> — read-only, matched by PDF SHA-256; not part of "
                        "this benchmark</div>", unsafe_allow_html=True)
            columns = st.columns(len(earlier))
            for column, (provider, pages) in zip(columns, sorted(earlier.items())):
                with column:
                    entry = pages.get(page["page"])
                    st.markdown(f"**{html.escape(provider)}**")
                    if entry is None:
                        st.caption("No verified completed result for this page.")
                        continue
                    st.caption(f"{entry['model_id'] or ''} · run {entry['run_id']} · {len(entry['text'])} chars")
                    _text_box(entry["text"])


def _ground_truth_section(run_path: Path, collected: dict) -> None:
    documents = {d["doc_key"]: d for d in collected["documents"]}
    scores = exports.ground_truth_scores(collected)
    status = exports.ground_truth_status(collected, scores)
    st.markdown("**Ground truth / CER-WER status**")
    for doc_key, info in status.items():
        (st.success if info["computed"] else st.warning if info["attached"] else st.info)(
            f"{documents[doc_key]['original_name']}: {info['message']}")
    with st.expander("Attach or replace ground truth — CER/WER only when attached"):
        st.caption("Attach a .txt per document (use `=== page N ===` separators for per-page scoring) or one file "
                   "per page named with its page number (e.g. page_3.txt). It is saved into the run folder and "
                   "summary.json is updated.")
        key = st.selectbox("Document", list(documents), format_func=lambda k: documents[k]["original_name"],
                           key="lpe-gt-doc")
        files = st.file_uploader("Ground-truth text file(s)", type=["txt"], accept_multiple_files=True, key="lpe-gt-files")
        replace = st.checkbox("Replace existing ground truth (old file is kept as ground_truth.replaced_*.txt)",
                              key="lpe-gt-replace")
        if files and st.button("Attach ground truth", key="lpe-gt-save"):
            try:
                text = diagnostics.combine_ground_truth_files([(f.name, f.getvalue()) for f in files])
                storage.save_ground_truth(run_path, key, text, [f.name for f in files], replace=replace)
                if not storage.lock_status(run_path)["alive"]:
                    exports.write_summary(run_path)  # derived file: keep its CER/WER in step with the run
                    st.success("Ground truth saved; summary.json updated.")
                else:
                    st.success("Ground truth saved; summary.json will be updated when the worker finishes.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(str(error))
        labels = {m["method_id"]: m["label"] for m in collected["methods"]}
        for doc_key, per_method in scores.items():
            st.markdown(f"**{html.escape(documents[doc_key]['original_name'])}** — normalization: none, and Arabic "
                        f"normalization ({html.escape(diagnostics.ARABIC_NORMALIZATION)})")
            rows = []
            for method_id, result in per_method.items():
                if result["level"] == "document":
                    entries = [("document", result.get("scores"))]
                else:
                    entries = [(f"page {n}", s) for n, s in result["pages"].items()]
                for scope, value in entries:
                    if not value:
                        rows.append([html.escape(labels[method_id]), scope, "not available", "", "", ""])
                        continue
                    rows.append([html.escape(labels[method_id]), scope, _fmt(value["none"]["cer"], 4), _fmt(value["none"]["wer"], 4),
                                 _fmt(value["arabic_normalized"]["cer"], 4), _fmt(value["arabic_normalized"]["wer"], 4)])
            _html_table(["Method", "Scope", "CER (none)", "WER (none)", "CER (Arabic norm.)", "WER (Arabic norm.)"], rows)


def _downloads(run_path: Path, collected: dict, raw_methods: list[dict], processed: list[dict]) -> None:
    st.subheader("Downloads")
    st.caption("Built from the saved files only — downloading never re-runs extraction. Copies are also written to "
               "the run's exports/ folder.")
    run_id = collected["run_id"]
    documents = {d["doc_key"]: d for d in collected["documents"]}
    methods = {m["method_id"]: m for m in collected["methods"]}
    c1, c2 = st.columns(2)
    txt_doc = c1.selectbox("TXT: document", list(documents), format_func=lambda k: documents[k]["original_name"],
                           key="lpe-dl-doc")
    txt_method = c2.selectbox("TXT: library / method", list(methods), format_func=lambda k: methods[k]["label"],
                              key="lpe-dl-method")
    cols = st.columns(5)
    cols[0].download_button("Download TXT", exports.document_txt(run_path, txt_doc, txt_method),
                            file_name=f"{txt_doc}_{txt_method}_full_document.txt", mime="text/plain",
                            key="lpe-dl-txt")
    json_bytes = exports.to_json(collected)
    csv_bytes = exports.to_csv(collected)
    cols[1].download_button("Download JSON", json_bytes, file_name=f"{run_id}_all_results.json",
                            mime="application/json", key="lpe-dl-json")
    cols[2].download_button("Download CSV", csv_bytes, file_name=f"{run_id}_summary.csv", mime="text/csv",
                            key="lpe-dl-csv")
    html_key, zip_key = f"lpe-html-{run_id}", f"lpe-zip-{run_id}"
    with cols[3]:
        embed = st.checkbox("Embed page images", value=True, key="lpe-dl-embed")
        if st.button("Prepare HTML", key="lpe-dl-html-build"):
            with st.spinner("Building the offline report..."):
                data = exports.to_html(collected, lambda k, p: storage.page_image(run_path, k, p, config.VIEWER_DPI),
                                       embed_images=embed,
                                       methods=[m["method_id"] for m in raw_methods + processed])
                exports.save_export(run_path, "comparison.html", data)
                st.session_state[html_key] = data
        if st.session_state.get(html_key):
            st.download_button("Download HTML Comparison", st.session_state[html_key],
                               file_name=f"{run_id}_comparison.html", mime="text/html", key="lpe-dl-html")
    with cols[4]:
        if st.button("Prepare ZIP", key="lpe-dl-zip-build"):
            with st.spinner("Packing all saved results..."):
                exports.save_export(run_path, "summary.csv", csv_bytes)
                exports.save_export(run_path, "all_results.json", json_bytes)
                html_bytes = st.session_state.get(html_key) or exports.to_html(
                    collected, lambda k, p: storage.page_image(run_path, k, p, config.VIEWER_DPI), embed_images=True)
                data = exports.to_zip(run_path, {"summary.csv": csv_bytes, "all_results.json": json_bytes,
                                                 "comparison.html": html_bytes})
                exports.save_export(run_path, "all_results.zip", data)
                st.session_state[zip_key] = data
        if st.session_state.get(zip_key):
            st.download_button("Download All Results ZIP", st.session_state[zip_key],
                               file_name=f"{run_id}_all_results.zip", mime="application/zip", key="lpe-dl-zip")


with tab_results:
    results_run = _run_picker("Run to review")
    if results_run:
        run_path = ROOT / results_run
        if st.button("Reload from disk", key="lpe-reload-btn"):
            st.session_state["lpe-reload"] = st.session_state.get("lpe-reload", 0) + 1
        collected = _collect(str(run_path), _stamp(run_path))
        status = collected["status"]
        st.markdown(f"Status (verified from saved files): **{status['label']}** · valid {status['counts']['valid']}, "
                    f"failed {status['counts']['failed']}, missing {status['counts']['missing']}, "
                    f"corrupt/incomplete {status['counts']['invalid']}")
        comp = exports.comparison(collected)
        raw_all = [m for m in collected["methods"] if m["output_type"] == "raw"]
        processed_all = [m for m in collected["methods"] if m["output_type"] == "processed"]
        rows, rows_diag = [], []
        for method in raw_all + processed_all:
            r = comp["methods"][method["method_id"]]
            rows.append([html.escape(method["label"]), str(r["completed_pages"]), str(r["failed_pages"]),
                         f"{r['total_time_ms'] / 1000:.3f} s", _fmt(r["avg_time_per_page_ms"], 2) + " ms",
                         str(r["chars"]), str(r["words"]), str(r["empty_pages"])])
            reversed_cell = f"{r['reversed_ending_la']} / {r['reversed_starting_al']}"
            if diagnostics.likely_reversed(r["reversed_ending_la"], r["reversed_starting_al"]):
                reversed_cell += " <b style='color:#c00'>⚠ likely character-reversed</b>"
            single_cell = (f"{_pct(r['single_letter_arabic_ratio'])} "
                           f"({r['single_letter_arabic_tokens']}/{r['arabic_tokens']})")
            rows_diag.append([html.escape(method["label"]), str(r["broken_lamalef_signs"]), reversed_cell,
                              single_cell, str(r["detached_diacritics"]), str(r["garbage_arabic_tokens"]),
                              str(r["needs_ocr_pages"])])
        separator = {len(raw_all): "PROCESSED — PyMuPDF + Arabic logical-order reconstruction (not raw library output)"} \
            if processed_all else None
        st.subheader("Comparison")
        _html_table(["Method", "Completed Pages", "Failed Pages", "Total Time", "Average Time per Page",
                     "Extracted Characters", "Extracted Words", "Empty Pages"], rows, separator)
        st.caption("More characters does not mean more accurate extraction. No accuracy score is given without ground truth.")
        st.markdown("**Arabic diagnostic indicators** — *indicators, not accuracy measurements*")
        _html_table(["Method", "Broken lam-alef signs", "Reversed-word signs (tokens ending لا / starting ال)",
                     "Single-letter Arabic tokens (ratio, single/all Arabic)", "Detached diacritics",
                     "Garbage glyphs in Arabic tokens", "Pages flagged 'Needs OCR?'"],
                    rows_diag, separator)
        rule = comp.get("needs_ocr_rule")
        st.caption(
            "'Needs OCR?' is a recommendation only (OCR is never run). "
            + (f"This run flags processed pages whose single-letter Arabic token ratio is above "
               f"{rule['single_letter_threshold']:.2f} (pages with at least {rule['single_letter_min_arabic_tokens']} "
               f"Arabic tokens): {diagnostics.SPLIT_WORDS_REASON}." if rule else
               "This run was created before the single-letter rule; only empty/garbage pages are flagged. "
               "Single-letter ratios of older runs are computed from the saved text on display."))
        per_document = comp["per_document"]
        doc_names = {d["doc_key"]: d["original_name"] for d in collected["documents"]}
        methods_order = raw_all + processed_all
        _html_table(["Document — single-letter Arabic token ratio", *[html.escape(m["label"]) for m in methods_order]],
                    [[f"<span dir='auto'>{html.escape(doc_names[k])}</span>",
                      *[_pct(per_document[k][m["method_id"]]["single_letter_arabic_ratio"]) for m in methods_order]]
                     for k in per_document])
        if comp["reorder_status_by_document"]:
            documents = {d["doc_key"]: d for d in collected["documents"]}
            _html_table(["Document (processed method)", *exports.REORDER_STATUSES],
                        [[f"<span dir='auto'>{html.escape(documents[k]['original_name'])}</span>",
                          *[str(v[s]) for s in exports.REORDER_STATUSES]]
                         for k, v in comp["reorder_status_by_document"].items()])
        _ground_truth_section(run_path, collected)

        st.subheader("Browse")
        documents = {d["doc_key"]: d for d in collected["documents"]}
        c1, c2, c3 = st.columns([2, 3, 2])
        doc_key = c1.selectbox("Document", list(documents), format_func=lambda k: documents[k]["original_name"],
                               key="lpe-view-doc")
        shown_raw_ids = c2.multiselect("Raw library columns", [m["method_id"] for m in raw_all],
                                       default=[m["method_id"] for m in raw_all],
                                       format_func=lambda k: next(m["label"] for m in raw_all if m["method_id"] == k),
                                       key="lpe-view-cols")
        show_processed = c3.checkbox("Show processed method next to raw", value=bool(processed_all),
                                     disabled=not processed_all, key="lpe-view-proc")
        show_tables = c3.checkbox("Tables (processed)", value=False, disabled=not processed_all,
                                  key="lpe-view-tables",
                                  help="Structured tables saved by the processed method, shown under it.")
        show_earlier = c3.checkbox("Show earlier OCR/VLM results (read-only)", value=False, key="lpe-view-ocr",
                                   help="Baidu / Qwen page results for the SAME PDF (full SHA-256 match, integrity "
                                        "verified). Nothing is shown when matching is uncertain.")
        view = st.radio("View", ["Single Page", "All Pages"], horizontal=True, key="lpe-view-mode")
        document = documents[doc_key]
        raw_methods = [m for m in raw_all if m["method_id"] in shown_raw_ids]
        processed = processed_all if show_processed else []
        earlier = _earlier_ocr(str(ROOT.parent), document["sha256"]) if show_earlier else {}
        if show_earlier and not earlier:
            st.info("No verified earlier OCR/VLM page results match this PDF's SHA-256.")
        if view == "Single Page":
            number = int(st.number_input(f"Page (1-{document['page_count']})", 1, document["page_count"], 1,
                                         key="lpe-view-page"))
            _render_page(run_path, document, document["pages"][number - 1], raw_methods, processed, earlier,
                         show_tables and bool(processed))
        else:
            per_screen = st.select_slider("Pages per screen", options=[3, 5, 10, 20], value=5, key="lpe-view-per")
            screens = max(1, -(-document["page_count"] // per_screen))
            screen = min(int(st.number_input(f"Screen (1-{screens})", 1, screens, 1, key="lpe-view-screen")), screens)
            start = (screen - 1) * per_screen
            st.caption(f"Pages {start + 1}-{min(start + per_screen, document['page_count'])} of {document['page_count']}. "
                       "Images are rendered lazily (only this screen) and cached in the run folder.")
            for page in document["pages"][start:start + per_screen]:
                _render_page(run_path, document, page, raw_methods, processed, earlier,
                             show_tables and bool(processed))
        _downloads(run_path, collected, raw_methods, processed)


# -------------------------------------------------------------- saved runs
def _duration(manifest: dict, state: dict) -> str:
    total_ms = sum(float(m.get("elapsed_ms") or 0) for m in state.get("methods", {}).values())
    try:
        start = datetime.fromisoformat(manifest["created_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat((manifest.get("finished_at") or "").replace("Z", "+00:00"))
        wall = f"{(end - start).total_seconds():.1f} s wall"
    except ValueError:
        wall = "—"
    return f"{total_ms / 1000:.2f} s extraction · {wall}"


with tab_runs:
    run_ids = _run_choices()
    if not run_ids:
        st.info("No saved runs yet.")
    else:
        rows = []
        for run_id in run_ids:
            try:
                quick = _quick_status(ROOT / run_id)
            except Exception as error:  # noqa: BLE001
                rows.append([html.escape(run_id), "—", "—", "—", "—", f"unreadable: {html.escape(str(error))}", "—", "—"])
                continue
            manifest = quick["manifest"]
            with_gt = sum(1 for d in manifest["documents"]
                          if (ROOT / run_id / "ground_truth" / d["doc_key"] / "ground_truth.txt").exists())
            rows.append([html.escape(run_id), html.escape(manifest["created_at"]),
                         "<br>".join(f"<span dir='auto'>{html.escape(d['original_name'])}</span>" for d in manifest["documents"]),
                         html.escape(", ".join(m["method_id"] for m in manifest["methods"])),
                         str(sum(d["page_count"] for d in manifest["documents"])), html.escape(quick["label"]),
                         html.escape(_duration(manifest, quick["state"])),
                         f"{with_gt}/{len(manifest['documents'])} attached" if with_gt else "none"])
        _html_table(["Run", "Created (UTC)", "Files", "Libraries / methods", "Pages", "Status", "Execution time",
                     "Ground truth"], rows)
        st.caption("Status here comes from state.json and the worker lock; the Results tab re-verifies every page file.")
        chosen_run = st.selectbox("Open a saved run", run_ids, key="lpe-open-run")
        if st.button("Open in Progress / Results", key="lpe-open"):
            _set_run(chosen_run)
            st.rerun()
        with st.expander("Delete a run (permanent)"):
            target = st.selectbox("Run to delete", run_ids, key="lpe-del-run")
            typed = st.text_input("Type the run id to confirm", key="lpe-del-typed")
            sure = st.checkbox("I understand this permanently deletes this one run folder", key="lpe-del-sure")
            if st.button("Delete run", disabled=not sure or typed != target, key="lpe-del"):
                try:
                    storage.delete_run(ROOT, target, typed)
                    if st.session_state.get("lpe-current") == target:
                        st.session_state.pop("lpe-current", None)
                        st.query_params.pop(RUN_PARAM, None)
                    st.success(f"Deleted {target}.")
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
