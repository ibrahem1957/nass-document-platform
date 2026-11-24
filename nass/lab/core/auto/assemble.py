"""Document assembly, de-duplication, validation and finalization of an Automatic Extraction run.

Everything here is derived from saved files only (no model, no network). Raw OCR outputs are never
changed; de-duplicated text is a derived value with every removed span recorded.

Per page:
- DIGITAL / MIXED: Phase 1 decision logic v2 (`hybrid.run.combine_page`, with the full-page Qari text
  only as `library_rejoined` guidance on category-B pages), layout v3 (`extract_page_elements` +
  `assemble_page`: reading order, headings, lists, find_tables with cross-page continuation). Qari
  text enters only as figure transcriptions; native text is preferred over OCR copies of it.
  Embedded scans: region transcriptions plus the scan's own body text, de-duplicated.
- SCANNED: the page OCR comes from a MASKED copy of the page (automatically processed visuals painted
  white, masking.py), so visual content comes only from the unmasked crops. Elements are ordered by
  the PP-DocLayout boxes and the body text is split over the text segments between visuals
  (order.py; method and limitation recorded per page). Ordered de-duplication remains a safety net,
  and an order-insensitive value-and-label check (duplicates.py) reports what is left.
Tables: Markdown plus JSON cells (find_tables cells on digital pages; parsed from the OCR Markdown/HTML
table on scanned pages when it is a table, otherwise cells are null with the reason). Figures stay in
their positions. Chart values are never estimated: only the OCR'd text is used.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from lab.core.auto import config as auto_config
from lab.core.auto import dedup, digits, duplicates, order, pipeline, redundancy, regions, router
from lab.core.auto import validate as validation
from lab.core.hybrid import layout as hybrid_layout
from lab.core.hybrid import run as hybrid_run
from lab.local_pdf_extraction import arabic_reorder, diagnostics

ASSEMBLY_VERSION = "2"
MODEL_LABEL = "Qari-OCR 0.4.0 (Qari service)"
RAG_CHUNK_CHARS = 2000


# ------------------------------------------------------------------ helpers


def _ocr(run_dir: Path, job: dict | None) -> dict | None:
    """Final text and provenance of a completed job, else None."""
    if not job or not pipeline.runnable(job):
        return None
    outcome = pipeline.latest(run_dir, job["job_id"])
    if not outcome or outcome.get("state") != pipeline.COMPLETED:
        return None
    response = outcome.get("response") or {}
    return {"text": response.get("final_text") or "", "job_id": job["job_id"], "attempt": outcome.get("attempt"),
            "request_id": outcome.get("request_id"), "source": outcome.get("source"),
            "image_file": job["image_file"], "image_sha256": job["image_sha256"],
            "model": {"label": MODEL_LABEL, **(response.get("model") or {}),
                      "service_version": response.get("service_version"), "recipe": response.get("recipe")},
            "settings": response.get("settings"), "final_attempt": response.get("final_attempt"),
            "retried": response.get("retried"), "final_flags": response.get("final_flags"),
            "proven_overrides": response.get("proven_overrides") or {}}


def _rtl(text: str) -> bool:
    return diagnostics.arabic_letter_share(text or "") >= arabic_reorder.ReorderConfig().min_arabic_letter_share


def parse_table(text: str) -> tuple[list[dict] | None, str]:
    """JSON cells from an OCR'd Markdown or HTML table (no cell is guessed)."""
    rows = [line.strip() for line in (text or "").splitlines() if line.strip().startswith("|")]
    if len(rows) >= 2:
        cells = []
        data_rows = [r for r in rows if not re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", r)]
        for r_index, row in enumerate(data_rows):
            for c_index, value in enumerate(row.strip("|").split("|")):
                cells.append({"row": r_index, "col": c_index, "text": value.strip()})
        return cells, "parsed from the OCR Markdown table"
    if "<table" in (text or "").lower():
        cells = []
        for r_index, row in enumerate(re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.S | re.I)):
            for c_index, cell in enumerate(re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.S | re.I)):
                cells.append({"row": r_index, "col": c_index, "text": re.sub(r"<[^>]+>", "", cell).strip()})
        return (cells or None), ("parsed from the OCR HTML table" if cells else "HTML table without cells")
    return None, "the OCR output is not a Markdown/HTML table; cells not derived (text kept as-is)"


def _quote(lines: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in (lines or "").splitlines())


def _visual_markdown(element: dict) -> str:
    label = element["label"]
    head = f"> **{label.capitalize()}** · {element['region_id']}"
    if element["validation_state"] == "pending_review":
        return f"{head} · not processed (pending review: {'; '.join(element['reasons'][:3])})"
    if element["type"] == "table" and element.get("content", "").lstrip().startswith("|"):
        return f"{head}\n\n{element['content']}"
    body = _quote(element.get("content") or "")
    return f"{head}\n>\n{body}\n>\n> <sub>(read by {MODEL_LABEL}; numbers need review)</sub>"


def _suppression(verdict: dict | None) -> dict:
    """Element fields of a provably redundant review placeholder (kept in document.json, not in Markdown/RAG)."""
    if not verdict:
        return {}
    return {"suppressed_redundant": True, "covered_by": verdict["covered_by"], "suppression_reason": verdict["reason"],
            "suppression_coverage": verdict["coverage"]}


def _element(page: int, index: int, **fields) -> dict:
    return {"id": f"p{page}_e{index}", "page": page, "reading_order": index, **fields}


# ------------------------------------------------------------------ scanned


def _masking_flags(number: int, masking_record: dict, processed: dict[str, dict], body_present: bool,
                   region_prefix: str = "") -> list[dict]:
    """Warnings tied to masking: visuals transcribed from a crop but NOT masked in the page OCR (Process
    now, or runs created before masking), and text that a mask may hide from the page OCR."""
    flags = []
    masked = set(masking_record.get("masked_region_ids") or [])
    if body_present:
        for region_id in processed:
            if region_id in masked:
                continue
            why = ("processed with Process now after the page OCR" if masking_record else
                   "the page OCR of this run was not masked (run created before visual masking)")
            flags.append({"severity": validation.WARNING, "code": "unmasked_region_possible_duplicate",
                          "page": number, "region_id": region_id,
                          "message": f"{region_prefix}{region_id}: {why}; its content may also be in the page "
                                     "text. The page OCR was not rerun; ordered duplicates are removed by the "
                                     "safety net, the rest is kept and listed in duplicate_check"})
    for item in masking_record.get("text_under_masks") or []:
        flags.append({"severity": validation.INFO if item["attached_title"] else validation.WARNING,
                      "code": "text_possibly_masked", "page": number, "region_id": item["region_id"],
                      "message": f"{item['label']} block {item['block_index']} is {item['share_masked']:.0%} under the "
                                 f"mask of {item['region_id']}: {item['note']}"})
    return flags


def _suppression_flags(number: int, suppressed: dict[str, dict]) -> list[dict]:
    return [{"severity": validation.INFO, "code": "redundant_region_suppressed", "page": number, "region_id": rid,
             "covered_by": verdict["covered_by"],
             "message": f"{rid} not shown in document.md / RAG chunks: {verdict['reason']} (still in the review "
                        "queue; Process now remains available)"} for rid, verdict in suppressed.items()]


def _scanned_page(run_dir: Path, page_layout: dict, jobs: dict[str, dict], route: dict) -> dict:
    number = page_layout["page"]
    selection = page_layout["selection"]
    page_job = jobs.get(f"p{number:03d}_page") or {}
    masking_record = page_job.get("masking") or {}
    body = _ocr(run_dir, page_job or None)
    region_jobs = {j.get("region_id"): j for j in jobs.values() if j["page"] == number and j.get("region_id")
                   and j.get("embedded_index") is None}
    visuals, crop_texts, dedup_log = [], {}, []
    for region in selection["regions"]:
        if region["decision"] == regions.REMOVED:
            continue
        job = region_jobs.get(region["region_id"])
        result = _ocr(run_dir, job)
        entry = {"region": region, "job": job, "ocr": result, "text": None}
        if result is not None:
            # Rule 1: the text of individual blocks is unknown on scanned pages -> no line is removed.
            entry["text"] = result["text"]
            entry["dedup"] = {"rule": "margin_line", "checked_blocks": [], "removed": [],
                              "note": "scanned page: text of the blocks outside the visual is not known "
                                      "(only the full page is transcribed); no margin line removed",
                              # diagnostic only: title regions inside the crop margin; the text is never changed
                              "suspected_margin_leakage": dedup.suspected_margin_leakage(
                                  region, selection, result["text"])}
            crop_texts[region["region_id"]] = result["text"]
        visuals.append(entry)
    body_text, body_dedup = None, None
    if body is not None:
        # Safety net only: masking keeps visual content out of the page OCR in the first place.
        body_dedup = dedup.remove_figure_spans(body["text"], crop_texts)
        body_text = body_dedup["text"]
        dedup_log += body_dedup["removed"]
    masked_ids = set(masking_record.get("masked_region_ids") or [])
    raw_candidates = duplicates.candidates(number, body["text"], crop_texts, source="page_ocr_raw") if body else []
    final_candidates = duplicates.candidates(number, body_text, crop_texts, source="document") if body else []
    rtl = _rtl(body["text"] if body else " ".join(crop_texts.values()))
    suppressed = {}
    for v in visuals:
        if v["ocr"] is None and not pipeline.runnable(v["job"] or {}):
            verdict = redundancy.assess(v["region"], selection, crop_texts)
            if verdict["redundant"]:
                suppressed[v["region"]["region_id"]] = verdict
    placed = order.place_body(body_text, [dict(v["region"], _entry=v) for v in visuals],
                              selection.get("text_blocks") or [], rtl=rtl)
    body_provenance = None
    if body is not None:
        body_provenance = {"route": route["route"], "rule": route["rule"], "job_id": body["job_id"],
                           "attempt": body["attempt"], "request_id": body["request_id"],
                           "result_source": body["source"], "image_file": body["image_file"],
                           "image_sha256": body["image_sha256"], "page_image": page_layout["page_image"],
                           "masking": {key: masking_record.get(key) for key in (
                               "policy", "version", "applied", "masked_region_ids", "masked_image_file",
                               "masked_image_sha256", "original_image_file", "original_image_sha256")}
                           if masking_record else {"applied": False, "note": "run created before visual masking"},
                           "reading_order": {"method": placed["method"], "limitation": placed["limitation"]}}
    elements, parts, first_body = [], [], None
    for slot in placed["sequence"]:
        index = len(elements)
        if slot["kind"] == "body":
            fields = {"segment": slot["segment"], "text_blocks": slot["text_blocks"]}
            if first_body is None:
                element = _element(number, index, type="page_text", bbox_pt=page_layout["page_image"]["bbox_pt"],
                                   content=slot["text"], raw_content=body["text"], source="qari_full_page",
                                   model=body["model"], validation_state="ok", body_segment=fields,
                                   provenance=body_provenance,
                                   dedup={"removed": body_dedup["removed"],
                                          "ambiguous_kept": body_dedup["ambiguous_kept"]})
                first_body = element["id"]
            else:
                element = _element(number, index, type="page_text", bbox_pt=page_layout["page_image"]["bbox_pt"],
                                   content=slot["text"], source="qari_full_page", model=body["model"],
                                   validation_state="ok", body_segment=fields | {"continues": first_body},
                                   provenance=body_provenance)
            parts.append(slot["text"])
        else:
            entry = slot["item"]["_entry"]
            region = entry["region"]
            processed = entry["ocr"] is not None
            label = region["label"]
            kind = "table" if label == "table" else "figure"
            content = entry["text"] if processed else ""
            cells, cells_note = parse_table(content) if kind == "table" and processed else (None, None)
            state = "ok" if processed else ("pending_review" if not pipeline.runnable(entry["job"] or {}) else "not_processed")
            element = _element(number, index, type=kind, label=label, region_id=region["region_id"],
                               bbox_pt=region["box_pt_before_margin"], crop_bbox_pt=region.get("crop_bbox_pt"),
                               content=content, source="qari_crop" if processed else None,
                               model=entry["ocr"]["model"] if processed else None, validation_state=state,
                               reasons=region["reasons"], decision=region["decision"],
                               cells=cells, cells_note=cells_note,
                               masked_in_page_ocr=region["region_id"] in masked_ids,
                               **_suppression(suppressed.get(region["region_id"])),
                               crop={"file": (entry["job"] or {}).get("image_file"),
                                     "sha256": (entry["job"] or {}).get("image_sha256"),
                                     "pixel_box": region["pixel_box"], "margin_px": region["margin_px"],
                                     "unmasked_original": True},
                               provenance={"route": route["route"], "rule": route["rule"],
                                           "job_id": (entry["job"] or {}).get("job_id"),
                                           "attempt": entry["ocr"]["attempt"] if processed else None,
                                           "request_id": entry["ocr"]["request_id"] if processed else None,
                                           "result_source": entry["ocr"]["source"] if processed else None,
                                           "layout_label": label, "layout_score": region.get("score"),
                                           "title_attachments": region.get("title_attachments") or [],
                                           # optional: unattached title blocks measured next to this visual
                                           # (geometry and policy outcome only, never a confirmed caption)
                                           "title_candidates": region.get("title_candidates") or [],
                                           "title_ambiguity": region.get("title_ambiguity") or [],
                                           "decisions": region.get("decisions") or []},
                               dedup=entry.get("dedup"))
            if not element.get("suppressed_redundant"):  # redundant placeholders stay out of the Markdown
                parts.append(_visual_markdown(element))
        elements.append(element)
    flags = []
    if body is None:
        flags.append({"severity": validation.ERROR, "code": "page_text_missing", "page": number,
                      "message": "the full-page OCR of this scanned page is not available"})
    if page_layout.get("detection_error"):
        flags.append({"severity": validation.WARNING, "code": "layout_detection_failed", "page": number,
                      "message": page_layout["detection_error"]})
    if placed["method"] == "proportional_to_text_block_area":
        flags.append({"severity": validation.INFO, "code": "reading_order_approximate", "page": number,
                      "message": placed["limitation"]})
    flags += _masking_flags(number, masking_record, crop_texts, body is not None)
    flags += _suppression_flags(number, suppressed)
    for candidate in final_candidates:
        flags.append({"severity": validation.WARNING, "code": "cross_source_duplicate_candidate", "page": number,
                      "region_id": candidate["region_id"],
                      "message": f"value {candidate['value']}{candidate['unit'] or ''} with label "
                                 f"{' '.join(candidate['label'])} appears in the page text (line "
                                 f"{candidate['page_line']}) and in the crop OCR (line {candidate['crop_line']}); "
                                 "both kept (report only)"})
    return {"page": number, "route": route, "elements": elements, "markdown": "\n\n".join(p for p in parts if p) + "\n",
            "reading_order_method": placed["method"],
            "reading_order": {"method": placed["method"], "limitation": placed["limitation"],
                              "segments": placed["segments"], "units": placed["units"],
                              "order_version": order.ORDER_VERSION, "direction": "rtl" if rtl else "ltr",
                              "source": "PP-DocLayout text-block and visual boxes"},
            "direction": "rtl" if rtl else "ltr", "dedup": dedup_log, "flags": flags,
            "regions": [v["region"] for v in visuals],
            "duplicates": {"raw": raw_candidates, "final": final_candidates, "masked_ids": sorted(masked_ids)},
            "masking": masking_record}


# ------------------------------------------------------------------ digital / mixed


def _native_blocks(elements: dict, mode: str, vlm_text: str) -> list[dict]:
    return [{"id": block["id"], "text": hybrid_layout._block_text(block, mode, vlm_text), "bbox_pt": block["bbox"],
             "source": "native text layer"} for block in elements.get("blocks") or []]


def _intersecting(blocks: list[dict], box: list[float]) -> list[dict]:
    return [b for b in blocks if dedup._intersection(b["bbox_pt"], box) > 0]


def _embedded_text(run_dir: Path, number: int, scan: dict, jobs: dict[str, dict], native: list[dict]) -> dict:
    """Composite transcription of an embedded scan: its body text + its regions, de-duplicated."""
    log, region_texts, pending = [], {}, []
    for region in scan["selection"]["regions"]:
        if region["decision"] == regions.REMOVED:
            continue
        job = next((j for j in jobs.values() if j.get("region_id") == region["region_id"]), None)
        result = _ocr(run_dir, job)
        if result is None:
            pending.append(region)  # assessed for redundancy once all processed texts are known
            continue
        outside = dedup.outside_blocks_for(region, native)  # native text: known
        cleaned = dedup.remove_margin_lines(result["text"], outside)
        log += cleaned["removed"]
        region_texts[region["region_id"]] = (region, cleaned["text"], result)
    body_job = jobs.get(f"p{number:03d}_E{scan['index']}_body")
    masking_record = (body_job or {}).get("masking") or {}
    body = _ocr(run_dir, body_job)
    body_text = ""
    texts = {rid: t for rid, (_r, t, _o) in region_texts.items()}
    raw_candidates, final_candidates = [], []
    if body is not None:
        inside = [b for b in native if dedup._intersection(b["bbox_pt"], scan["bbox_pt"]) > 0]
        native_clean = dedup.remove_margin_lines(body["text"], inside,
                                                 reason="native text preferred (same text in the PDF text layer)")
        spans = dedup.remove_figure_spans(native_clean["text"], {rid: t for rid, (_r, t, _o) in region_texts.items()})
        log += native_clean["removed"] + spans["removed"]
        body_text = spans["text"]
        raw_candidates = duplicates.candidates(number, body["text"], texts, source="embedded_body_raw")
        final_candidates = duplicates.candidates(number, body_text, texts, source="document")
    rtl = _rtl(body_text or " ".join(t for _r, t, _o in region_texts.values()))
    parts = [body_text] if body_text else []
    for region in regions.reading_order([r for r, _t, _o in region_texts.values()], rtl=rtl):
        parts.append(f"[{region['label']} {region['region_id']}]\n{region_texts[region['region_id']][1]}")
    processed_texts = {rid: t for rid, (_r, t, _o) in region_texts.items()}
    suppressed = {}
    for region in pending:
        verdict = redundancy.assess(region, scan["selection"], processed_texts)
        if verdict["redundant"]:
            suppressed[region["region_id"]] = verdict
            continue
        parts.append(f"[{region['label']} {region['region_id']}: not processed — pending review]")
    flags = _masking_flags(number, masking_record, texts, body is not None, region_prefix="embedded scan ")
    flags += _suppression_flags(number, suppressed)
    for candidate in final_candidates:
        flags.append({"severity": validation.WARNING, "code": "cross_source_duplicate_candidate", "page": number,
                      "region_id": candidate["region_id"],
                      "message": f"value {candidate['value']}{candidate['unit'] or ''} with label "
                                 f"{' '.join(candidate['label'])} appears in the embedded scan's body text and its "
                                 "crop OCR; both kept (report only)"})
    return {"text": "\n\n".join(parts), "dedup": log, "pending": pending, "body": body,
            "regions": {rid: o for rid, (_r, _t, o) in region_texts.items()},
            "duplicates": {"raw": raw_candidates, "final": final_candidates,
                           "masked_ids": sorted(masking_record.get("masked_region_ids") or [])},
            "flags": flags, "masking": masking_record, "suppressed": suppressed,
            "pending_visible": [r for r in pending if r["region_id"] not in suppressed]}


def _digital_page(run_dir: Path, document, analysis_page: dict, analysis: dict, page_layout: dict,
                  jobs: dict[str, dict], route: dict, previous_tables: dict | None) -> dict:
    number = analysis_page["page"]
    rejoin = _ocr(run_dir, jobs.get(f"p{number:03d}_rejoin"))
    record = None
    if rejoin is not None:
        flags = rejoin["final_flags"] or {}
        record = {"job_state": "COMPLETED", "text": rejoin["text"], "finish_reason": "length" if flags.get("hit_limit") else "stop",
                  "truncated_at_token_limit": bool(flags.get("hit_limit"))}
    rule = diagnostics.needs_ocr_settings(analysis["settings"])
    page_rule = None if (analysis_page.get("reorder") or {}).get("decision") == "mainly_ltr" else rule
    decision = hybrid_run.combine_page(analysis_page, record, page_rule, hybrid_run.DECISION_LOGIC_VERSION)
    reorder = analysis["settings"].get("reorder") or {}
    tables_doc = analysis_page.get("tables")
    elements, _crops = hybrid_layout.extract_page_elements(document[number - 1], number, reorder_config=reorder,
                                                           tables_doc=tables_doc)
    mode = hybrid_layout.text_mode(decision, analysis_page)
    vlm_text = (record or {}).get("text") or ""
    native = _native_blocks(elements, mode, vlm_text)
    transcriptions, figure_info, dedup_log, flags = {}, {}, [], []
    page_duplicates: dict[str, list] = {"raw": [], "final": [], "masked_ids": []}
    scans = [s for s in page_layout.get("embedded_scans") or [] if s.get("status") == "layout detected"]
    composites = page_layout.get("composites") or {}
    for figure in elements["figures"]:
        key = f"p{number}_{figure['id']}"
        scan = next((s for s in scans if all(abs(float(a) - float(b)) <= 2.0 for a, b in
                                             zip(figure["bbox"], s.get("source_bbox_pt") or s["bbox_pt"]))), None)
        if scan is not None:
            composite = _embedded_text(run_dir, number, scan, jobs, native)
            dedup_log += composite["dedup"]
            flags += composite["flags"]
            for part in ("raw", "final", "masked_ids"):  # not `key`: that is the transcription key above
                page_duplicates[part] += composite["duplicates"][part]
            if composite["text"].strip():
                transcriptions[key] = {"text": composite["text"], "model": MODEL_LABEL}
            figure_info[figure["id"]] = {"kind": "embedded_scan", "composite": composite}
            continue
        if figure["id"] in composites:  # a layer of a larger figure: its content is in that figure's crop
            figure_info[figure["id"]] = {"kind": "layer", "of": composites[figure["id"]]}
            continue
        result = _ocr(run_dir, jobs.get(f"p{number:03d}_{figure['id']}"))
        if result is None:
            figure_info[figure["id"]] = {"kind": "figure", "ocr": None}
            continue
        cleaned = dedup.remove_margin_lines(result["text"], _intersecting(native, figure["bbox"]),
                                            reason="native text preferred (same text in the PDF text layer)")
        dedup_log += cleaned["removed"]
        transcriptions[key] = {"text": cleaned["text"], "model": MODEL_LABEL}
        figure_info[figure["id"]] = {"kind": "figure", "ocr": result, "dedup": cleaned}
    assembled = hybrid_layout.assemble_page(
        elements, decision=decision, library_page=analysis_page, tables_doc=tables_doc, vlm_text=vlm_text,
        reorder_config=reorder, transcriptions=transcriptions, previous_tables_doc=previous_tables)
    layout_record = assembled["record"]
    blocks = {b["id"]: b for b in native}
    merged_into: dict[str, list[str]] = {}
    for merge in layout_record.get("paragraph_merges") or []:
        merged_into.setdefault(merge["into"], []).append(merge["merged"])
    tables = {f"t{t['table_index']}": t for t in (tables_doc or {}).get("tables") or []}
    figures = {f["id"]: f for f in elements["figures"]}
    out = []
    base_provenance = {"route": route["route"], "rule": route["rule"], "phase1_category": analysis_page["category"],
                       "final_source": decision.get("final_source"), "text_mode": mode,
                       "decision_logic_version": decision.get("decision_logic_version"),
                       "layout_version": hybrid_layout.LAYOUT_VERSION}
    order_boxes = layout_record.get("order_boxes")
    if not order_boxes:  # layout fallback: page text + figures
        out.append(_element(number, 0, type="page_text", bbox_pt=None, content=assembled["plain_text_layer"],
                            source="pymupdf", model=None, validation_state="needs_review",
                            provenance=base_provenance | {"layout_fallback": layout_record.get("fallback_reason")}))
        order_boxes = [{"id": fid, "type": "figure", "bbox": figures[fid]["bbox"]} for fid in figures]
    for box in order_boxes:
        index = len(out)
        if box["type"] in {"paragraph", "heading"}:
            ids = [box["id"]] + merged_into.get(box["id"], [])
            out.append(_element(number, index, type="heading" if box["type"] == "heading" else "text",
                                bbox_pt=box["bbox"], content="\n".join(blocks[i]["text"] for i in ids if i in blocks),
                                source="pymupdf_" + mode, model=None, validation_state="ok",
                                provenance=base_provenance | {"blocks": ids}))
        elif box["type"] == "table":
            table = tables.get(box["id"]) or {}
            continuation = next((c for c in layout_record.get("table_continuations") or [] if c.get("table") == box["id"]), None)
            out.append(_element(number, index, type="table", bbox_pt=box["bbox"],
                                content=hybrid_layout._table_markdown(table, continuation) if table else "",
                                cells=[{"row": c["row"], "col": c["col"], "text": c.get("processed_cell_text"),
                                        "status": c.get("reorder_status")} for c in table.get("cells") or []],
                                cells_note="PyMuPDF find_tables cells (col 0 is the rightmost cell on RTL tables)",
                                source="pymupdf_find_tables", model=None,
                                validation_state="needs_review" if table.get("status") == "uncertain" else "ok",
                                provenance=base_provenance | {"table_index": table.get("table_index"),
                                                              "continuation": continuation,
                                                              "warnings": table.get("warnings")}))
        else:
            figure = figures.get(box["id"]) or {"id": box["id"], "bbox": box["bbox"]}
            info = figure_info.get(box["id"]) or {}
            text = (transcriptions.get(f"p{number}_{box['id']}") or {}).get("text") or ""
            job = jobs.get(f"p{number:03d}_{box['id']}")
            ocr = info.get("ocr")
            state = "ok" if text else "not_processed"
            if info.get("kind") == "embedded_scan" and info["composite"]["pending_visible"]:
                state = "pending_review"
            if info.get("kind") == "layer":
                state = "included_in_composite"
            image_meta = (job or {}).get("image") or {}
            out.append(_element(number, index, type="figure", label=figure.get("origin"), bbox_pt=box["bbox"],
                                content=text, source="qari_crop" if text else None,
                                model=(ocr or {}).get("model") if ocr else ({"label": MODEL_LABEL} if text else None),
                                validation_state=state,
                                crop={"file": (job or {}).get("image_file"), "sha256": (job or {}).get("image_sha256"),
                                      "method": image_meta.get("method"), "method_reason": image_meta.get("method_reason"),
                                      "original_bbox": image_meta.get("original_bbox"),
                                      "crop_bbox": image_meta.get("crop_bbox"), "render_dpi": image_meta.get("render_dpi"),
                                      "effective_dpi": image_meta.get("effective_dpi"),
                                      "pixel_size": image_meta.get("pixel_size"),
                                      "native": (image_meta.get("native_assessment") or {}).get("native"),
                                      "quality": image_meta.get("quality"),
                                      "native_candidate_file": image_meta.get("native_candidate_file"),
                                      "includes_figures": image_meta.get("includes_figures") or []},
                                provenance=base_provenance | {"figure_origin": figure.get("origin"),
                                                              "embedded_scan": info.get("kind") == "embedded_scan",
                                                              "included_in": info.get("of"),
                                                              "suppressed_regions": (info.get("composite") or {}).get("suppressed") or {},
                                                              "job_id": (job or {}).get("job_id"),
                                                              "attempt": (ocr or {}).get("attempt"),
                                                              "request_id": (ocr or {}).get("request_id"),
                                                              "result_source": (ocr or {}).get("source")},
                                dedup=(info.get("dedup") or {}).get("removed") if info.get("kind") == "figure"
                                else (info.get("composite") or {}).get("dedup")))
    for item in layout_record.get("needs_review") or []:
        flags.append({"severity": validation.WARNING, "code": "layout_review", "page": number, "message": item})
    for fid, root in composites.items():
        flags.append({"severity": validation.INFO, "code": "figure_included_in_composite", "page": number,
                      "element_id": fid, "message": f"figure {fid} lies inside figure {root}; both are one visual and "
                                                    f"were read from the complete crop of {root}"})
    for job in (j for j in jobs.values() if j["page"] == number and j.get("purpose") == "digital_figure"):
        image_meta = job.get("image") or {}
        for note in image_meta.get("review") or []:
            flags.append({"severity": validation.WARNING, "code": "crop_needs_review", "page": number,
                          "job_id": job["job_id"], "message": f"{job['figure_id']}: {note}"})
    if decision.get("status") in {hybrid_run.NEEDS_REVIEW, hybrid_run.UNCERTAIN, hybrid_run.FAILED}:
        flags.append({"severity": validation.WARNING, "code": "page_text_review", "page": number,
                      "message": f"Phase 1 status {decision.get('status')}: " + "; ".join(decision.get("reasons") or [])})
    if route["category_b_rejoin"] and rejoin is None:
        flags.append({"severity": validation.WARNING, "code": "rejoin_guidance_missing", "page": number,
                      "message": "category B page: the full-page Qari text (spacing guidance) is not available"})
    all_regions = [r for s in scans for r in s["selection"]["regions"]]
    return {"page": number, "route": route, "elements": out, "markdown": assembled["markdown"],
            "reading_order_method": "layout_v3_xy_cut" if layout_record.get("order_boxes") else "layout_fallback",
            "reading_order": {"method": "layout_v3_xy_cut" if layout_record.get("order_boxes") else "layout_fallback",
                              "source": "PyMuPDF text blocks + layout v3",
                              "limitation": layout_record.get("fallback_reason")},
            "duplicates": page_duplicates,
            "direction": layout_record.get("direction"), "dedup": dedup_log, "flags": flags, "regions": all_regions,
            "phase1": {k: v for k, v in decision.items() if k not in {"final_text", "alternatives"}},
            "layout_record": {k: layout_record.get(k) for k in ("order", "needs_review", "fallback_reason",
                                                               "table_continuations", "validation")}}


# ------------------------------------------------------------------ assemble / finalize


def assemble(run_dir: Path) -> dict[str, Any]:
    """The whole document in memory (nothing written)."""
    import pymupdf  # noqa: PLC0415

    run_dir = Path(run_dir)
    plan = pipeline.load_plan(run_dir)
    jobs = {j["job_id"]: j for j in pipeline.all_jobs(run_dir, plan)}
    routes = {r["page"]: r for r in plan["routing"]["pages"]}
    layouts = {p["page"]: p for p in plan["layout"]["pages"]}
    analysis = plan["analysis"]
    pages = []
    with pymupdf.open(run_dir / "source.pdf") as document:
        previous_tables = None
        for analysis_page in analysis["pages"]:
            number = analysis_page["page"]
            route = routes[number]
            if route["route"] == router.SCANNED:
                pages.append(_scanned_page(run_dir, layouts[number], jobs, route))
            else:
                pages.append(_digital_page(run_dir, document, analysis_page, analysis, layouts[number], jobs, route,
                                           previous_tables))
            previous_tables = analysis_page.get("tables")
    elements = [e for page in pages for e in page["elements"]]
    results = {job_id: pipeline.latest(run_dir, job_id) for job_id, job in jobs.items() if pipeline.runnable(job)}
    runnable_jobs = [job for job in jobs.values() if pipeline.runnable(job)]
    all_regions = [dict(r, processed_on_request=bool((jobs.get(r.get("job_id") or "") or {}).get("requested")))
                   for page in pages for r in page["regions"]]
    run_flags = []
    services = pipeline.service_records(run_dir)
    used = sorted({(r["endpoint"], r["deployment_mode"]) for r in services})
    if len(used) > 1:  # never silent: a run whose jobs went to more than one endpoint / mode is flagged
        run_flags.append({"severity": validation.WARNING, "code": "qari_service_changed", "page": None,
                          "message": "jobs of this run were sent to several Qari services: "
                                     + "; ".join(f"{e} ({m})" for e, m in used)})
    verification_report, verification_flags = digit_verification(run_dir, plan, elements)
    report = validation.report(elements, runnable_jobs, results, all_regions,
                               run_flags + [f for page in pages for f in page["flags"]] + verification_flags)
    report["digit_verification"] = verification_report
    masked_ids = {rid for page in pages for rid in (page.get("duplicates") or {}).get("masked_ids") or []}
    raw = [c for page in pages for c in (page.get("duplicates") or {}).get("raw") or []]
    final = [c for page in pages for c in (page.get("duplicates") or {}).get("final") or []]
    report["duplicate_check"] = duplicates.summary(raw, final, masked_ids) | {
        "by_page": {str(page["page"]): {"raw": len((page.get("duplicates") or {}).get("raw") or []),
                                        "final": len((page.get("duplicates") or {}).get("final") or [])}
                    for page in pages},
        "candidates": final}
    name = plan["run"]["document"]["display_name"]
    markdown = f"# {name}\n\n" + "\n".join(
        f"<!-- page {p['page']} · {p['route']['route']} ({p['route']['rule']}) -->\n\n{p['markdown']}" for p in pages)
    return {"plan": plan, "pages": pages, "elements": elements, "validation": report, "markdown": markdown,
            "jobs": jobs, "results": results}


def _crop_provenance(job: dict, result: dict | None) -> dict:
    """Manifest entry of one digital/mixed visual crop (crop method, DPI, native evidence, OCR outcome)."""
    meta = job.get("image") or {}
    response = (result or {}).get("response") or {}
    return {"page": job["page"], "figure_id": job.get("figure_id"), "method": meta.get("method"),
            "method_reason": meta.get("method_reason"), "original_bbox": meta.get("original_bbox"),
            "crop_bbox": meta.get("crop_bbox"), "configured_render_dpi": meta.get("configured_render_dpi"),
            "render_dpi": meta.get("render_dpi"), "effective_dpi": meta.get("effective_dpi"),
            "pixel_size": meta.get("pixel_size"), "native": (meta.get("native_assessment") or {}).get("native"),
            "native_failures": (meta.get("native_assessment") or {}).get("failures"),
            "native_candidate_file": meta.get("native_candidate_file"), "quality": meta.get("quality"),
            "includes_figures": meta.get("includes_figures") or [], "image_file": job.get("image_file"),
            "image_sha256": job.get("image_sha256"), "request_id": (result or {}).get("request_id"),
            "ocr_state": (result or {}).get("state"),
            "ocr_empty": bool(result) and result.get("state") == "completed" and not (response.get("final_text") or "").strip()}


DIGIT_FLAG_SEVERITY = {"digit_disagreement": validation.WARNING, "possible_digit_reversal": validation.WARNING,
                       "separator_difference": validation.WARNING, "verification_inconclusive": validation.INFO}


def digit_verification(run_dir: Path, plan: dict, elements: list[dict]) -> tuple[dict, list[dict]]:
    """Second-read report from saved records (read only). Never changes any OCR text."""
    settings = (plan["run"]["config"] or {}).get("verification") or {}
    by_job = {(e.get("provenance") or {}).get("job_id"): e for e in elements if (e.get("provenance") or {}).get("job_id")}
    jobs, flags = {}, []
    totals = {name: 0 for name in digits.OUTCOMES}
    status_counts = {"compared": 0, "verification_skipped": 0, "verification_failed": 0}
    for job, outcome in pipeline.verification_candidates(run_dir):
        records = pipeline.verification_records(run_dir, job["job_id"])
        attempts = records["attempts"]
        completed = [a for a in attempts if (a["result"] or {}).get("state") == pipeline.COMPLETED]
        primary_text = (outcome.get("response") or {}).get("final_text") or ""
        entry: dict = {"page": job["page"], "region_id": job.get("region_id"), "figure_id": job.get("figure_id"),
                       "primary_request_id": outcome.get("request_id")}
        if completed:
            attempt = completed[-1]
            response = attempt["result"].get("response") or {}
            result = digits.compare(primary_text, response.get("final_text") or "")
            entry.update(status="compared", verification_request_id=attempt["result"].get("request_id"),
                         input=attempt["intent"].get("input"), verification_attempts=response.get("attempts"),
                         verification_final_flags=response.get("final_flags"), **result)
            for name, count in result["counts"].items():
                totals[name] += count
            for value in result["values"]:
                severity = DIGIT_FLAG_SEVERITY.get(value["outcome"])
                if severity:
                    flags.append({"severity": severity, "code": value["outcome"], "page": job["page"],
                                  "job_id": job["job_id"], "region_id": job.get("region_id"),
                                  "message": f"primary {value['primary']} vs second read {value['verification']} "
                                             f"(label {' '.join(value['label']) or '-'}); report only, nothing corrected"})
        elif any((a["result"] or {}).get("state", pipeline.UNCONFIRMED) != pipeline.NOT_TRANSMITTED for a in attempts):
            last = attempts[-1]["result"] or {"state": pipeline.UNCONFIRMED}
            entry.update(status="verification_failed", state=last.get("state"), error=last.get("error"))
        else:
            reasons = [r["reason"] for r in records["skipped"]]
            if attempts:
                reasons.append(f"not transmitted: {((attempts[-1]['result'] or {}).get('error') or {}).get('message')}")
            entry.update(status="verification_skipped",
                         reason=reasons[-1] if reasons else "second read not run (no verification was requested)")
            flags.append({"severity": validation.INFO, "code": "verification_skipped", "page": job["page"],
                          "job_id": job["job_id"], "region_id": job.get("region_id"), "message": entry["reason"]})
        status_counts[entry["status"]] += 1
        jobs[job["job_id"]] = entry
        element = by_job.get(job["job_id"])
        if element is not None:
            element["digit_verification"] = {"status": entry["status"], "counts": entry.get("counts"),
                                             "reason": entry.get("reason")}
    report = {"version": digits.DIGITS_VERSION, "settings": settings,
              "requests_sent": pipeline.verification_requests_sent(run_dir), "eligible_crops": len(jobs),
              **status_counts, "outcomes": totals, "jobs": jobs, "note": digits.NOTE}
    return report, flags


def rag_chunks(document: dict) -> list[dict]:
    sha = document["plan"]["run"]["document"]["sha256"]
    chunks = []
    for element in document["elements"]:
        text = (element.get("content") or "").strip()
        if not text or element.get("suppressed_redundant"):
            continue
        pieces, current = [], ""
        for paragraph in re.split(r"\n\s*\n", text):
            if current and len(current) + len(paragraph) > RAG_CHUNK_CHARS:
                pieces.append(current)
                current = ""
            current = (current + "\n\n" + paragraph).strip()
        if current:
            pieces.append(current)
        for index, piece in enumerate(pieces):
            chunks.append({"chunk_id": f"{element['id']}_c{index}", "document_sha256": sha, "page": element["page"],
                           "element_id": element["id"], "element_type": element["type"], "text": piece,
                           "source": element.get("source"), "validation_state": element.get("validation_state"),
                           "bbox_pt": element.get("bbox_pt")})
    return chunks


def finalize(run_dir: Path, config: auto_config.AutoConfig, *, allow_incomplete: bool = False) -> Path:
    """Write document.md/json, validation_report.json, rag_chunks.jsonl, then execution_manifest.json (last)."""
    import pymupdf  # noqa: PLC0415

    run_dir = Path(run_dir)
    if pipeline.is_finalized(run_dir):
        raise pipeline.PipelineError("the run is already finalized")
    status = pipeline.status(run_dir)
    open_jobs = status["counts"]["pending"] + status["counts"]["unconfirmed"] + status["counts"]["failed"]
    if open_jobs and not allow_incomplete:
        raise pipeline.PipelineError(f"{open_jobs} OCR job(s) are not completed; finish them or confirm finalizing "
                                     "an incomplete document")
    document = assemble(run_dir)
    pending_review = sum(1 for f in document["validation"]["flags"] if f["code"] == "pending_review")
    if config.validation.review_policy == "block_finalize" and (document["validation"]["summary"]["error"] or pending_review):
        raise pipeline.PipelineError("validation errors or regions pending review block finalization "
                                     "(VALIDATION_REVIEW_POLICY=block_finalize)")
    plan = document["plan"]
    elements_doc = {
        "schema_version": 1, "kind": "automatic_extraction_document", "assembly_version": ASSEMBLY_VERSION,
        "document": plan["run"]["document"], "run_id": run_dir.name,
        "pages": [{"page": p["page"], "route": p["route"]["route"], "rule": p["route"]["rule"],
                   "reading_order_method": p["reading_order_method"], "direction": p["direction"],
                   "element_ids": [e["id"] for e in p["elements"]], "dedup_removed": p["dedup"],
                   "phase1": p.get("phase1"), "layout": p.get("layout_record")} for p in document["pages"]],
        "elements": document["elements"],
    }
    chunks = rag_chunks(document)
    files = {
        "document.md": document["markdown"].encode("utf-8", errors="surrogatepass"),
        "document.json": json.dumps(elements_doc, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass"),
        "validation_report.json": json.dumps(document["validation"], ensure_ascii=False, indent=2).encode("utf-8"),
        "rag_chunks.jsonl": "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks).encode("utf-8",
                                                                                                   errors="surrogatepass"),
    }
    for name, data in files.items():
        pipeline.write_once(run_dir / name, data)
    jobs = list(document["jobs"].values())
    responses = [r.get("response") or {} for r in document["results"].values() if r and r.get("state") == pipeline.COMPLETED]
    overrides = [r.get("proven_overrides") for r in responses if r.get("proven_overrides")]
    try:
        import paddleocr  # noqa: PLC0415, F401
        paddle_version = getattr(paddleocr, "__version__", None)
    except Exception:  # noqa: BLE001
        paddle_version = None
    manifest = {
        "schema_version": 1, "kind": "automatic_extraction_manifest", "run_id": run_dir.name,
        "pipeline_version": pipeline.PIPELINE_VERSION, "assembly_version": ASSEMBLY_VERSION,
        "router_version": router.ROUTER_VERSION, "validation_version": validation.VALIDATION_VERSION,
        "created_at": plan["run"]["created_at"], "finalized_at": pipeline._now(),
        "incomplete": bool(open_jobs), "open_jobs": open_jobs,
        "document": plan["run"]["document"],
        "configuration": plan["run"]["config"],
        "proven_settings": plan["run"]["config"]["proven"],
        "proven_differences": plan["run"]["proven_differences"],
        "generation_overrides_reported_by_service": overrides,
        "image_resizing": plan["run"]["image_resizing"],
        "routing": {"counts": plan["routing"]["counts"],
                    "pages": [{"page": r["page"], "route": r["route"], "rule": r["rule"], "reason": r["reason"]}
                              for r in plan["routing"]["pages"]]},
        "layout_detector": plan["layout"]["detector"],
        "reading_order": {str(p["page"]): p["reading_order_method"] for p in document["pages"]},
        "reading_order_provenance": {str(p["page"]): p.get("reading_order") for p in document["pages"]},
        "masking": {job_id: {key: job["masking"].get(key) for key in (
            "policy", "version", "applied", "masked_region_ids", "masks", "not_masked_review_region_ids",
            "masked_image_file", "masked_image_sha256", "original_image_file", "original_image_sha256",
            "image_px", "text_under_masks")} | {"request_id": (document["results"].get(job_id) or {}).get("request_id")}
            for job_id, job in document["jobs"].items() if job.get("masking")},
        "visual_crops": {job_id: _crop_provenance(job, document["results"].get(job_id))
                         for job_id, job in document["jobs"].items() if job.get("purpose") == "digital_figure"},
        "digit_verification": {key: document["validation"]["digit_verification"][key] for key in (
            "settings", "requests_sent", "eligible_crops", "compared", "verification_skipped", "verification_failed",
            "outcomes", "note")},
        "duplicate_check": {key: document["validation"]["duplicate_check"][key]
                            for key in ("version", "raw_page_ocr", "final_document")},
        "qari_service": {
            "configured": {"deployment_mode": config.qari.deployment_mode,
                           "automatic_enabled": config.qari.automatic_enabled,
                           "trusted_development_hosts": list(config.qari.trusted_development_hosts)},
            "executions": pipeline.service_records(run_dir),
            "endpoints_used": sorted({r["endpoint"] for r in pipeline.service_records(run_dir)}),
            "deployment_modes_used": sorted({r["deployment_mode"] for r in pipeline.service_records(run_dir)}),
            "approval": "none: Automatic Extraction sends only to the administrator-configured, trusted service"},
        "jobs": {"total": len(jobs), "runnable": sum(1 for j in jobs if pipeline.runnable(j)),
                 "processed_on_request": sum(1 for j in jobs if j.get("requested")), **status["counts"],
                 "retried_by_service": sum(1 for r in responses if r.get("retried")),
                 "offline_imports": sum(1 for r in document["results"].values() if r and r.get("source") == "offline_import")},
        "dedup": {"removed_spans": sum(len(p["dedup"]) for p in document["pages"])},
        "validation": {"status": document["validation"]["status"], **document["validation"]["summary"]},
        "software": {"lab_version": plan["run"]["lab_version"], "pymupdf": pymupdf.__version__,
                     "paddleocr": paddle_version},
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "secrets": "no token or API key is stored in this run",
    }
    return pipeline.write_once(run_dir / "execution_manifest.json", manifest)


def zip_bytes(run_dir: Path, *, include_pdf: bool = False) -> bytes:
    """Export ZIP: outputs, plan, crops, raw attempts (finalized or partial). Nothing is changed."""
    run_dir = Path(run_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or (path.name == "source.pdf" and not include_pdf):
                continue
            archive.write(path, path.relative_to(run_dir).as_posix())
        if not pipeline.is_finalized(run_dir):
            archive.writestr("PARTIAL_RUN.json", json.dumps(pipeline.status(run_dir), ensure_ascii=False, indent=2))
    return buffer.getvalue()
