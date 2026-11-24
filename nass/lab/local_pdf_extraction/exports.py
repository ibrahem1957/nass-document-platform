"""Summaries and downloads, built ONLY from the saved run files (never re-extracts).

Exports are derived artifacts: they are (re)written atomically to `<run>/exports/`
and offered through the app's download buttons.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

from lab.local_pdf_extraction import diagnostics, storage, tables
from lab.local_pdf_extraction.arabic_reorder import METHOD_LABEL as PROCESSED_LABEL

REORDER_STATUSES = ("not_applicable", "not_needed", "reconstructed", "uncertain")
CSV_FIELDS = (
    "run_id", "document", "document_key", "document_sha256", "page", "method_id", "method_label",
    "output_type", "library", "library_version", "status", "verification", "error", "extraction_ms",
    "char_count", "word_count", "is_empty", "reorder_status", "reorder_decision", "needs_ocr",
    "broken_lamalef_signs", "reversed_ending_la", "reversed_starting_al", "detached_diacritics",
    "garbage_arabic_tokens", "presentation_form_ratio",
    "fragmented_line_ratio", "single_letter_arabic_ratio", "single_letter_arabic_tokens", "arabic_tokens",
    "needs_ocr_reason", "tables_detected", "text",
)


def collect(run_path: Path, *, worker_finished: bool = False) -> dict:
    """Everything a view/export needs, read from disk and re-verified page by page."""
    manifest = storage.load_manifest(run_path)
    status = storage.derive_status(run_path, manifest, worker_finished=worker_finished)
    methods = manifest["methods"]
    documents = []
    for document in manifest["documents"]:
        key = document["doc_key"]
        pages = []
        for page in range(1, document["page_count"] + 1):
            outputs = {}
            for method in methods:
                method_id = method["method_id"]
                verification = status["scan"]["pages"].get((key, method_id, page), "missing")
                record, text = storage.read_page(run_path, key, method_id, page)
                record = _with_single_letter_stats(record, text if verification == "valid" else None)
                outputs[method_id] = {"verification": verification, "record": record,
                                      "text": text if verification == "valid" else None,
                                      "tables": _load_tables(run_path, key, method_id, page, record, verification)}
            pages.append({"page": page, "outputs": outputs})
        documents.append({**document, "pages": pages, "ground_truth": storage.load_ground_truth(run_path, key)})
    return {"run_id": manifest["run_id"], "manifest": manifest, "status": status, "methods": methods,
            "documents": documents, "run_path": str(run_path)}


def _with_single_letter_stats(record: dict | None, text: str | None) -> dict | None:
    """Pages saved before the single-letter indicator existed: compute it from the saved text,
    in memory only (the record on disk is never rewritten)."""
    if not record or text is None:
        return record
    diag = record.get("diagnostics") or {}
    if "single_letter_arabic_ratio" in diag:
        return record
    return {**record, "diagnostics": {**diag, **diagnostics.single_letter_stats(text),
                                      "single_letter_computed_on_load": True}}


def _load_tables(run_path: Path, key: str, method_id: str, page: int, record: dict | None,
                 verification: str) -> dict | None:
    """Structured tables (processed method only), only from a verified page (hash-checked side file)."""
    if verification != "valid" or not record or not record.get("tables_file"):
        return None
    try:
        data = storage.read_side_file(run_path, key, method_id, page, record["tables_file"])
        return json.loads(data.decode("utf-8", errors="surrogatepass")) if data else None
    except (storage.StorageError, ValueError):
        return None


def table_html(table: dict) -> str:
    """One detected table as an HTML table (single line, escaped). RTL tables use dir='rtl', so
    the first logical column is shown on the right. Uncertain cells are highlighted with ⚠;
    the raw cell text is in each cell's tooltip."""
    width, rows = _table_grid(table)
    out = [f"<table dir='{table['direction']}' style='border-collapse:collapse;font-size:.85rem;margin:4px 0'>"]
    for row in rows:
        out.append("<tr>")
        for col in range(width):
            cell = row.get(col)
            if cell is None:
                out.append("<td style='border:1px solid rgba(128,128,128,.5)'></td>")
                continue
            status = cell["reorder_status"]
            text = {"not_detected": "⟨no cell⟩", "merged": "⟨merged⟩"}.get(status, cell["processed_cell_text"])
            style = ("border:1px solid rgba(128,128,128,.5);padding:3px 6px;vertical-align:top;"
                     "unicode-bidi:plaintext;")
            if status == "uncertain":
                style += "background:rgba(255,170,0,.30);"
                text += " ⚠"
            title = html.escape(f"raw: {cell['raw_cell_text']}", quote=True).replace("\n", "&#10;")
            out.append(f"<td dir='auto' style='{style}' title='{title}'>{html.escape(text)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _table_grid(table: dict) -> tuple[int, list[dict[int, dict]]]:
    width = max((cell["col"] for cell in table["cells"]), default=-1) + 1
    rows: dict[int, dict[int, dict]] = {}
    for cell in table["cells"]:
        rows.setdefault(cell["row"], {})[cell["col"]] = cell
    return width, [rows[r] for r in sorted(rows)]


def tables_section_html(doc: dict) -> str:
    """All tables of one page for the offline report: status, warnings, table, raw vs processed."""
    esc = html.escape
    parts = ["<div class='tables'><h3>Tables — PROCESSED output (PyMuPDF find_tables() + per-cell "
             "reconstruction), not raw library output</h3>"]
    if doc.get("error"):
        parts.append(f"<div class='na'>Table detection failed: {esc(str(doc['error']))}</div>")
    if not doc.get("tables") and not doc.get("error"):
        parts.append("<small>No table detected on this page.</small>")
    for note in doc.get("notes", []):
        parts.append(f"<br><small>{esc(tables.clean_note(note))}</small>")
    for table in doc.get("tables", []):
        parts.append(f"<p><b>Table {table['table_index']}</b> — {table['rows']}×{table['cols']}, status: "
                     f"<b>{esc(table['status'])}</b>, {esc(table['column_order'])}</p>")
        for warning in table["warnings"]:
            parts.append(f"<div class='na'>⚠ {esc(warning)}</div>")
        continuation = table.get("possible_continuation_of")
        if continuation:
            parts.append(f"<small>Possible continuation of page {continuation['page']} table "
                         f"{continuation['table_index']} (not merged).</small>")
        parts.append(table_html(table))
        parts.append("<details><summary>Raw vs processed cell text</summary><table><tr><th>row</th><th>col</th>"
                     "<th>status</th><th>raw_cell_text</th><th>processed_cell_text</th></tr>")
        for cell in table["cells"]:
            parts.append(f"<tr><td>{cell['row']}</td><td>{cell['col']}</td><td>{esc(cell['reorder_status'])}</td>"
                         f"<td dir='auto' class='text'>{esc(cell['raw_cell_text'])}</td>"
                         f"<td dir='auto' class='text'>{esc(cell['processed_cell_text'])}</td></tr>")
        parts.append("</table></details>")
    parts.append("</div>")
    return "".join(parts)


def _records_for(collected: dict, method_id: str, key: str | None = None) -> list[dict]:
    records = []
    for document in collected["documents"]:
        if key and document["doc_key"] != key:
            continue
        for page in document["pages"]:
            output = page["outputs"][method_id]
            if output["verification"] in ("valid", "failed") and output["record"]:
                records.append(output["record"])
    return records


def comparison(collected: dict) -> dict:
    """Per-method comparison metrics + per-document reorder status counts + optional CER/WER."""
    table = {}
    for method in collected["methods"]:
        table[method["method_id"]] = {**diagnostics.aggregate(_records_for(collected, method["method_id"])),
                                      "output_type": method["output_type"], "label": method["label"]}
    reorder = {}
    for document in collected["documents"]:
        counts = Counter()
        for page in document["pages"]:
            output = page["outputs"].get("pymupdf_ar_reordered")
            if output and output["record"] and output["verification"] == "valid":
                counts[(output["record"].get("reorder") or {}).get("reorder_status", "unknown")] += 1
        if counts:
            reorder[document["doc_key"]] = {name: counts.get(name, 0) for name in REORDER_STATUSES}
    per_document = {
        document["doc_key"]: {method["method_id"]: diagnostics.aggregate(
            _records_for(collected, method["method_id"], document["doc_key"])) for method in collected["methods"]}
        for document in collected["documents"]
    }
    scores = ground_truth_scores(collected)
    return {"methods": table, "per_document": per_document, "reorder_status_by_document": reorder,
            "ground_truth": scores, "ground_truth_status": ground_truth_status(collected, scores),
            "needs_ocr_rule": diagnostics.needs_ocr_settings(collected["manifest"].get("settings")),
            "note": "Indicators are heuristics, not accuracy. CER/WER only where ground truth is attached."}


def ground_truth_status(collected: dict, scores: dict | None = None) -> dict:
    """Per document: is ground truth attached, at which level, and were CER/WER computed (and for what)."""
    scores = ground_truth_scores(collected) if scores is None else scores
    status = {}
    for document in collected["documents"]:
        key = document["doc_key"]
        if not document.get("ground_truth"):
            status[key] = {"attached": False, "level": None, "computed": False, "scored_methods": [],
                           "message": "No ground truth attached: CER/WER not computed."}
            continue
        parsed = diagnostics.parse_ground_truth(document["ground_truth"])
        level = "page" if isinstance(parsed, dict) else "document"
        scored = []
        for method_id, result in scores.get(key, {}).items():
            if result["level"] == "page":
                if any(value for value in result["pages"].values()):
                    scored.append(method_id)
            elif result.get("scores"):
                scored.append(method_id)
        pages = sorted(parsed) if isinstance(parsed, dict) else None
        status[key] = {
            "attached": True, "level": level, "pages_with_ground_truth": pages,
            "computed": bool(scored), "scored_methods": scored,
            "message": (f"Ground truth attached ({level} level"
                        + (f", pages {', '.join(map(str, pages))}" if pages else "") + "): "
                        + (f"CER/WER computed for {len(scored)} method(s)." if scored
                           else "CER/WER NOT computed (no matching page text available).")),
        }
    return status


def ground_truth_scores(collected: dict) -> dict:
    scores: dict = {}
    for document in collected["documents"]:
        if not document.get("ground_truth"):
            continue
        parsed = diagnostics.parse_ground_truth(document["ground_truth"])
        per_method = {}
        for method in collected["methods"]:
            method_id = method["method_id"]
            texts = {page["page"]: page["outputs"][method_id]["text"] for page in document["pages"]}
            if isinstance(parsed, dict):
                pages = {}
                for number, reference in sorted(parsed.items()):
                    hypothesis = texts.get(number)
                    pages[str(number)] = None if hypothesis is None else diagnostics.error_rates(reference, hypothesis)
                per_method[method_id] = {"level": "page", "pages": pages}
            else:
                if any(text is None for text in texts.values()):
                    per_method[method_id] = {"level": "document", "scores": None,
                                             "note": "some pages are not available"}
                else:
                    full = "\n".join(texts[number] for number in sorted(texts))
                    per_method[method_id] = {"level": "document", "scores": diagnostics.error_rates(parsed, full)}
        scores[document["doc_key"]] = per_method
    return scores


def write_summary(run_path: Path, *, worker_finished: bool = False) -> dict:
    """summary.json is derived and rewritten (atomically) after a run and after ground truth is attached."""
    collected = collect(run_path, worker_finished=worker_finished)
    summary = {"run_id": collected["run_id"], "generated_at": storage.utc_now(),
               "status": collected["status"]["status"], "counts": collected["status"]["counts"],
               **comparison(collected)}
    storage.atomic_write_json(Path(run_path) / "summary.json", summary)
    return summary


# ----------------------------------------------------------------- downloads
def document_txt(run_path: Path, key: str, method_id: str) -> bytes:
    path = storage.method_dir(run_path, key, method_id) / "full_document.txt"
    if not path.exists():
        manifest = storage.load_manifest(run_path)
        count = next(d["page_count"] for d in manifest["documents"] if d["doc_key"] == key)
        path = storage.write_full_document(run_path, key, method_id, count)
    return path.read_bytes()


def _flat_rows(collected: dict):
    for document in collected["documents"]:
        for page in document["pages"]:
            for method in collected["methods"]:
                output = page["outputs"][method["method_id"]]
                record = output["record"] or {}
                diag = record.get("diagnostics") or {}
                reorder = record.get("reorder") or {}
                yield {
                    "run_id": collected["run_id"], "document": document["original_name"],
                    "document_key": document["doc_key"], "document_sha256": document["sha256"],
                    "page": page["page"], "method_id": method["method_id"],
                    "method_label": method["label"], "output_type": method["output_type"],
                    "library": method["library"], "library_version": method.get("library_version"),
                    "status": record.get("status") if output["verification"] in ("valid", "failed") else "NOT AVAILABLE",
                    "verification": output["verification"], "error": record.get("error"),
                    "extraction_ms": record.get("extraction_ms"), "char_count": record.get("char_count"),
                    "word_count": record.get("word_count"), "is_empty": record.get("is_empty"),
                    "reorder_status": reorder.get("reorder_status"), "reorder_decision": reorder.get("decision"),
                    "needs_ocr": diag.get("needs_ocr"), "broken_lamalef_signs": diag.get("broken_lamalef_signs"),
                    "reversed_ending_la": (diag.get("reversed_word_signs") or {}).get("ending_la"),
                    "reversed_starting_al": (diag.get("reversed_word_signs") or {}).get("starting_al"),
                    "detached_diacritics": diag.get("detached_diacritics"),
                    "garbage_arabic_tokens": diag.get("garbage_arabic_tokens"),
                    "presentation_form_ratio": diag.get("presentation_form_ratio"),
                    "fragmented_line_ratio": diag.get("fragmented_line_ratio"),
                    "single_letter_arabic_ratio": diag.get("single_letter_arabic_ratio"),
                    "single_letter_arabic_tokens": diag.get("single_letter_arabic_tokens"),
                    "arabic_tokens": diag.get("arabic_tokens"),
                    "needs_ocr_reason": diag.get("needs_ocr_reason"),
                    "tables_detected": len(output["tables"].get("tables", [])) if output.get("tables") else None,
                    "text": output["text"],
                }


def to_csv(collected: dict) -> bytes:
    """UTF-8 with BOM so Excel shows Arabic correctly."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in _flat_rows(collected):
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    return "﻿".encode("utf-8") + storage.encode_text(buffer.getvalue())


def to_json(collected: dict) -> bytes:
    payload = {
        "run_id": collected["run_id"], "manifest": collected["manifest"],
        "status": {key: value for key, value in collected["status"].items() if key not in ("scan",)},
        "comparison": comparison(collected),
        "pages": list(_flat_rows(collected)),
        "processed_method_label": PROCESSED_LABEL,
        "processed_tables": [
            {"document_key": document["doc_key"], "page": page["page"], "method_id": method_id, **output["tables"]}
            for document in collected["documents"] for page in document["pages"]
            for method_id, output in page["outputs"].items() if output.get("tables")
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8", errors="surrogatepass")


_HTML_CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--muted:#666;--line:#d6d6d6;--card:#fafafa;--proc:#7a4b00;--procbg:#fff6e5;--bad:#a40000}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#e8e8e8;--muted:#9a9a9a;--line:#3a3a3a;--card:#1c1c1c;--proc:#ffc36b;--procbg:#2a2110;--bad:#ff8a8a}}
*{box-sizing:border-box}body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Tahoma,Arial,sans-serif}
h1{font-size:1.4rem;margin:0 0 4px}h2{font-size:1.15rem;margin:28px 0 8px}small,.muted{color:var(--muted)}
.page{border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0;background:var(--card)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.col{min-width:0}.col h3{font-size:.95rem;margin:0 0 4px}
.text{white-space:pre-wrap;unicode-bidi:plaintext;max-height:520px;overflow:auto;border:1px solid var(--line);border-radius:6px;padding:8px;background:var(--bg)}
.proc{border:2px dashed var(--proc);border-radius:8px;padding:8px;background:var(--procbg)}.proc h3{color:var(--proc)}
.badge{display:inline-block;padding:0 6px;border-radius:4px;border:1px solid var(--line);font-size:.8rem}
.na{color:var(--bad);font-weight:600}img{max-width:100%;height:auto;border:1px solid var(--line)}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid var(--line);padding:4px 8px;text-align:left}
"""


def _ratio(value) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def to_html(collected: dict, image_for, *, embed_images: bool = True, methods: list[str] | None = None) -> bytes:
    """Standalone, offline HTML: page images as base64, no CDN, no external fonts."""
    esc = html.escape
    chosen = [m for m in collected["methods"] if methods is None or m["method_id"] in methods]
    raw = [m for m in chosen if m["output_type"] == "raw"]
    processed = [m for m in chosen if m["output_type"] == "processed"]
    comp = comparison(collected)
    out = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Local PDF extraction comparison</title><style>{_HTML_CSS}</style></head><body>",
           f"<h1>Local PDF Extraction Benchmark — {esc(collected['run_id'])}</h1>",
           f"<p class='muted'>Status: {esc(collected['status']['label'])}. Generated from saved results only; "
           "text-layer extraction, no OCR. Raw outputs are exactly what each library returned. "
           "Diagnostic indicators are heuristics, not accuracy.</p>",
           "<table><tr><th>Method</th><th>Type</th><th>Completed</th><th>Failed</th><th>Total ms</th><th>Chars</th><th>Words</th><th>Empty</th><th>Reversed-word signs (ending لا / starting ال)</th>"
           "<th>Single-letter Arabic tokens</th><th>Needs OCR? pages</th></tr>"]
    for method in chosen:
        row = comp["methods"][method["method_id"]]
        out.append(f"<tr><td>{esc(method['label'])}</td><td>{esc(method['output_type'].upper())}</td><td>{row['completed_pages']}</td>"
                   f"<td>{row['failed_pages']}</td><td>{row['total_time_ms']:.1f}</td><td>{row['chars']}</td>"
                   f"<td>{row['words']}</td><td>{row['empty_pages']}</td>"
                   f"<td>{row['reversed_ending_la']} / {row['reversed_starting_al']}"
                   f"{' ⚠ likely character-reversed' if diagnostics.likely_reversed(row['reversed_ending_la'], row['reversed_starting_al']) else ''}"
                   f"</td><td>{_ratio(row['single_letter_arabic_ratio'])} ({row['single_letter_arabic_tokens']}/"
                   f"{row['arabic_tokens']})</td><td>{row['needs_ocr_pages']}</td></tr>")
    out.append("</table>")
    rule = comp.get("needs_ocr_rule")
    out.append("<p class='muted'>'Needs OCR?' (recommendation only, OCR is not run): "
               + (f"processed pages with a single-letter Arabic token ratio above {rule['single_letter_threshold']:.2f} "
                  f"(at least {rule['single_letter_min_arabic_tokens']} Arabic tokens) are flagged: "
                  f"{esc(diagnostics.SPLIT_WORDS_REASON)}." if rule else
                  "this run was created before the single-letter rule; only empty/garbage pages are flagged.")
               + "</p>")
    out.append("<h2>Single-letter Arabic token ratio per document</h2><table><tr><th>Document</th>"
               + "".join(f"<th>{esc(m['label'])}</th>" for m in chosen) + "</tr>")
    for document in collected["documents"]:
        per = comp["per_document"][document["doc_key"]]
        out.append(f"<tr><td dir='auto'>{esc(document['original_name'])}</td>"
                   + "".join(f"<td>{_ratio(per[m['method_id']]['single_letter_arabic_ratio'])}</td>" for m in chosen)
                   + "</tr>")
    out.append("</table>")
    out.append("<h2>Ground truth</h2><ul>")
    for document in collected["documents"]:
        info = comp["ground_truth_status"][document["doc_key"]]
        out.append(f"<li dir='auto'>{esc(document['original_name'])}: {esc(info['message'])}</li>")
    out.append("</ul>")

    def column(method: dict, output: dict) -> str:
        record = output["record"] or {}
        if output["text"] is None:
            reason = record.get("error") if output["verification"] == "failed" else output["verification"]
            return (f"<div class='col'><h3>{esc(method['label'])}</h3><div class='na'>NOT AVAILABLE — "
                    f"{esc(str(reason))}</div></div>")
        meta = f"{record.get('extraction_ms', 0):.1f} ms · {record.get('char_count', 0)} chars"
        diag = record.get("diagnostics") or {}
        if diag.get("arabic_tokens"):
            meta += (f" · single-letter Arabic tokens: {_ratio(diag.get('single_letter_arabic_ratio'))} "
                     f"({diag.get('single_letter_arabic_tokens')}/{diag.get('arabic_tokens')})")
        if diag.get("needs_ocr"):
            meta += f"<br><span class='na'>Needs OCR? {esc(str(diag.get('needs_ocr_reason')))}</span>"
        reorder = record.get("reorder")
        if reorder:
            meta += f" · reorder_status: <b>{esc(reorder.get('reorder_status', ''))}</b>"
            for warning in reorder.get("warnings") or []:
                meta += f"<br><span class='na'>{esc(warning)}</span>"
        return (f"<div class='col'><h3>{esc(method['label'])}</h3><small>{meta}</small>"
                f"<div class='text' dir='auto'>{esc(output['text'])}</div></div>")

    for document in collected["documents"]:
        out.append(f"<h2 dir='auto'>{esc(document['original_name'])} <small>({document['page_count']} pages, "
                   f"SHA-256 {esc(document['sha256'][:16])}…)</small></h2>")
        counts = comp["reorder_status_by_document"].get(document["doc_key"])
        if counts and processed:
            out.append("<p class='muted'>Processed reorder_status counts: " +
                       ", ".join(f"{esc(k)}: {v}" for k, v in counts.items()) + "</p>")
        for page in document["pages"]:
            out.append(f"<div class='page'><b>Page {page['page']}</b><div class='grid'>")
            if embed_images:
                path = image_for(document["doc_key"], page["page"])
                if path:
                    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
                    out.append(f"<div class='col'><h3>Original page</h3><img loading='lazy' alt='page {page['page']}' "
                               f"src='data:image/png;base64,{data}'></div>")
            out.append("".join(column(method, page["outputs"][method["method_id"]]) for method in raw))
            out.append("</div>")
            if processed:
                out.append("<div class='proc'><h3>PROCESSED output — not raw library output</h3><div class='grid'>")
                out.append("".join(column(method, page["outputs"][method["method_id"]]) for method in processed))
                out.append("</div>")
                for method in processed:
                    doc = page["outputs"][method["method_id"]].get("tables")
                    if doc is not None:
                        out.append(tables_section_html(doc))
                out.append("</div>")
            out.append("</div>")
    out.append("</body></html>")
    return storage.encode_text("\n".join(out))


def to_zip(run_path: Path, extra_files: dict[str, bytes]) -> bytes:
    """All saved run files (manifest, state, summary, documents, ground truth) + generated exports."""
    run_path = Path(run_path)
    buffer = io.BytesIO()
    skip_dirs = {"page_images", "exports"}
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_path.rglob("*")):
            relative = path.relative_to(run_path)
            if not path.is_file() or relative.parts[0] in skip_dirs or path.name.endswith(".tmp") or path.name == storage.LOCK_NAME:
                continue
            archive.write(path, f"{run_path.name}/{relative.as_posix()}")
        for name, data in extra_files.items():
            archive.writestr(f"{run_path.name}/exports/{name}", data)
    return buffer.getvalue()


def save_export(run_path: Path, name: str, data: bytes) -> Path:
    if name not in {"comparison.html", "summary.csv", "all_results.zip", "all_results.json"}:
        raise storage.StorageError(f"Unexpected export name {name!r}")
    path = Path(run_path) / "exports" / name
    storage.atomic_write_bytes(path, data)
    return path
