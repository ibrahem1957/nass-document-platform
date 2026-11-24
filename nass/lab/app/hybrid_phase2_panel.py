"""Phase 2 (Benchmark Mode) UI: image regions, provider comparison, manual Qari import,
image references and unified page documents. Nothing is sent without the explicit approval;
export, import, comparison, evaluation and document recomputation never call a provider."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from lab.app import full_pdf_panel as panel
from lab.app.components import text_html
from lab.core.e2.api import documents
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIRunRefused
from lab.core.hybrid import crops as crop_options
from lab.core.hybrid import enrich, qari
from lab.core.hybrid import layout_regions

PROVIDERS = {"baidu-aistudio-paddleocr-vl-1.6": "PaddleOCR-VL 1.6 (Baidu AI Studio)",
             "oxen-qwen3-vl-4b-instruct": "Qwen3-VL-4B-Instruct (Oxen.ai)",
             "qari": "Qari-OCR 0.4.0 (your OpenAI-compatible endpoint)"}


def _pct(value) -> str:
    return "-" if value is None else f"{value:.1%}"


def _crop_line(options: dict) -> str:
    mode = options.get("resolution_mode")
    dpi = ("native " if mode in {"native", "native_subregion"} else "") + f"{options.get('effective_dpi')} DPI"
    size = options.get("pixel_size") or ["?", "?"]
    quality = f", q{options['jpeg_quality']}" if options.get("jpeg_quality") else ""
    return (f"{size[0]}×{size[1]} px · {(options.get('file_bytes') or 0) / 1_048_576:.2f} MB · "
            f"{options.get('format')}{quality} · {mode} · {dpi}" + (" · legacy set" if options.get("legacy") else ""))


def _specs() -> dict:
    return {spec.id: spec for spec in load_providers() if spec.id in PROVIDERS and spec.runnable}


# Module functions this panel calls. A running Streamlit process keeps modules imported before an
# update, so a missing name means stale code in memory (restart Streamlit), not a data problem.
REQUIRED_API = {
    "crops": ("RESOLUTIONS", "FORMATS", "JPEG_QUALITY", "region_options", "size_based_timeout"),
    "enrich": ("candidate_regions", "create_region_set", "load_set", "export_crops_zip", "check_import",
               "import_results", "capacity", "create_api_run", "execute_api_run", "resume_api_run", "create_qari_run",
               "execute_qari_run", "collect_outputs", "outputs_zip", "save_selection", "latest_selection",
               "import_reference", "propose_matches", "save_matches", "confirmed_matches", "evaluate",
               "compute_enriched", "apply_enriched", "latest_enriched", "mode_spec"),
    "layout_regions": ("saved_provider_output", "pp_doclayout_local", "doclayout_setup", "propose",
                       "preview_images", "create_layout_region_set", "DEFAULT_LABELS", "DEFAULT_MARGIN_PX",
                       "LayoutSourceError", "LAYOUT_SOURCES", "PP_DOCLAYOUT_MODELS"),
}


# Keyword parameters this panel passes; an older function in memory may lack them (TypeError otherwise).
REQUIRED_PARAMETERS = {
    ("layout_regions", "propose"): ("labels", "margin_px", "titles", "captions_below", "expansions"),
    ("layout_regions", "create_layout_region_set"): ("blocks", "labels", "margin_px", "selected", "source", "pdf_path",
                                                     "by", "titles", "captions_below", "detection_report", "expansions",
                                                     "expected_pixel_boxes"),
    ("layout_regions", "pp_doclayout_local"): ("pdf_path", "pages", "model_dir", "model_name", "device", "threshold"),
    ("enrich", "create_region_set"): ("pdf_path", "by", "resolution", "fmt"),
    ("enrich", "create_api_run"): ("by", "statement", "root", "mode", "timeout_override"),
    ("enrich", "create_qari_run"): ("endpoint_url", "model", "by", "statement", "root", "timeout_override"),
}


def missing_api() -> list[str]:
    import inspect  # noqa: PLC0415

    modules = {"crops": crop_options, "enrich": enrich, "layout_regions": layout_regions}
    missing = [f"{name}.{attribute}" for name, attributes in REQUIRED_API.items() for attribute in attributes
               if not hasattr(modules[name], attribute)]
    for (name, function), parameters in REQUIRED_PARAMETERS.items():
        target = getattr(modules[name], function, None)
        if target is None:
            continue  # already reported above
        accepted = inspect.signature(target).parameters
        missing += [f"{name}.{function}({parameter}=)" for parameter in parameters if parameter not in accepted]
    return missing


def render_phase2(*, run_dir: Path, run_spec, document, source: Path, store, runner, workspace_id: str) -> None:
    st.subheader("Phase 2 · Image, chart and screenshot extraction (Benchmark Mode)")
    stale = missing_api()
    if stale:
        st.error("The running app has an older version of the Phase 2 code in memory (missing: "
                 + ", ".join(stale) + "). Stop Streamlit completely and start it again; nothing was changed.")
        return
    st.caption("Regions come from the approved layout revision. Each selected region is made into a crop once "
               "(local; resolution and format chosen per region set) and every provider receives exactly that crop. "
               "Automatic Extraction Mode is not built yet.")
    specs = _specs()
    spec_for = specs.get
    _regions_section(run_dir, source)
    sets = enrich.region_sets(run_dir)
    if not sets:
        return
    _layout_boxes_section(run_dir, source, specs, store, sets)
    sets = enrich.region_sets(run_dir)
    names = [path.name for path in sets]
    set_dir = sets[names.index(st.selectbox("Region set", names, index=len(names) - 1, key=f"p2-set-{run_dir.name}"))]
    try:
        data = enrich.load_set(set_dir)
    except enrich.EnrichError as exc:
        st.error(str(exc))
        return
    _crops(set_dir, data)
    _manual_qari(set_dir, data)
    _send_section(set_dir, data, specs, document=document, source=source, store=store, runner=runner,
                  workspace_id=workspace_id)
    outputs = enrich.collect_outputs(set_dir, spec_for, store.root)
    saved = sum(len(items) for items in outputs.values())
    output_zips = st.session_state.setdefault("p2_output_zips", {})
    if st.button(f"Prepare saved provider outputs ZIP ({saved} output(s); read-only)", key=f"p2-out-{set_dir.name}",
                 disabled=not saved):
        output_zips[set_dir.name] = enrich.outputs_zip(set_dir, spec_for, store.root)
    if set_dir.name in output_zips:
        st.download_button("Download provider outputs", data=output_zips[set_dir.name],
                           file_name=f"{data['set_id']}_provider_outputs.zip", mime="application/zip",
                           key=f"p2-out-dl-{set_dir.name}")
    st.caption("Includes, per region and output: raw response files as saved, journal events, parsed and visible text, "
               "status and provenance. Available before any selection or enrichment revision. Nothing is sent.")
    reference = _reference_section(set_dir, data, outputs)
    _compare_section(set_dir, data, outputs, reference)
    _documents_section(run_dir, set_dir, spec_for, run_spec, store, source)


def _regions_section(run_dir: Path, source: Path) -> None:
    with st.expander("1 · Select image regions (local only)", expanded=not enrich.region_sets(run_dir)):
        try:
            candidates = enrich.candidate_regions(run_dir)
        except enrich.EnrichError as exc:
            st.info(str(exc))
            return
        if not candidates:
            st.info("The approved layout revision has no image or chart region.")
            return
        st.markdown(panel._table(["Region", "Page", "Detected type", "BBox (pt)", "Area", "Uncertainty"], [
            [panel._esc(c["region_id"]), panel._esc(c["page"]), panel._esc(c["detected_type"]),
             panel._esc(c["bbox"]), panel._esc(_pct(c.get("area_share"))),
             panel._esc("; ".join(c["uncertainty"]) or "-")] for c in candidates]), unsafe_allow_html=True)
        columns = st.columns(4)
        for index, candidate in enumerate(c for c in candidates if c.get("layout_crop")):
            if Path(candidate["layout_crop"]).is_file():
                columns[index % 4].image(candidate["layout_crop"], width="stretch",
                                         caption=candidate["region_id"] + (" ⚠ uncertain" if candidate["uncertain"] else ""))
        left, right = st.columns(2)
        resolution = left.selectbox(
            "Crop resolution", list(crop_options.RESOLUTIONS), key=f"p2-res-{run_dir.name}",
            format_func=lambda v: {"auto": "auto (native for a scanned full page, else 250 DPI)",
                                   "native": "native (embedded image as stored; fallback recorded)"}.get(v, f"{v} DPI"))
        fmt = right.selectbox("Crop format", list(crop_options.FORMATS), key=f"p2-fmt-{run_dir.name}",
                              format_func=lambda v: {"auto": "auto (PNG; native keeps its original format)",
                                                     "jpeg": f"JPEG (quality {crop_options.JPEG_QUALITY})"}.get(v, "PNG"))
        st.caption("Other options produce other bytes, i.e. a new region set; existing sets are never changed.")
        chosen = st.multiselect("Regions to include (uncertain regions are never preselected)",
                                [c["region_id"] for c in candidates],
                                default=[c["region_id"] for c in candidates if c["default_selected"]],
                                key=f"p2-regions-{run_dir.name}")
        by = st.text_input("Your name", key=f"p2-regions-by-{run_dir.name}")
        if st.button("Save region selection (renders the crops locally)", key=f"p2-regions-save-{run_dir.name}",
                     disabled=not chosen):
            try:
                folder = enrich.create_region_set(run_dir, chosen, pdf_path=source, by=by, resolution=resolution,
                                                  fmt=fmt)
                st.session_state["hy-flash"] = ("success", f"Region set {folder.name} saved.")
                st.rerun()
            except enrich.EnrichError as exc:
                st.error(str(exc))


def _decision_text(candidate: dict) -> str:
    parts = []
    for decision in candidate.get("decisions") or []:
        if decision["type"] == "containment":
            parts.append(decision["decision"] + (f" (in block {decision['contained_in']})"
                                                 if decision.get("contained_in") is not None else
                                                 f" (inner {decision.get('inner')})"))
        elif decision["type"] == "partial_overlap":
            parts.append(f"partial overlap with block {decision['with_block']}")
        elif decision["type"] == "incomplete_region":
            parts.append("content may be cut; expansion " + ("applied" if decision["applied"] else "proposed"))
        elif decision["type"] == "caption_trim":
            parts.append(("top trimmed below" if decision.get("applied") else "not trimmed below")
                         + f" caption block {decision.get('caption_block')}")
        elif decision["type"] == "duplicate":
            parts.append(f"exact duplicate of block {decision['of_block']}")
    for title in candidate.get("title_attachments") or []:
        parts.append(f"title block {title['block_index']} attached ({title['position']}"
                     + (", overlaps boundary" if title.get("overlaps_boundary") else "") + ")")
    if candidate.get("title_decision"):
        parts.append(candidate["title_decision"])
    if candidate.get("title_ambiguous"):
        parts.append("title association ambiguous")
    return "; ".join(parts) or "-"


def _layout_boxes_section(run_dir: Path, source: Path, specs: dict, store, sets: list[Path]) -> None:
    with st.expander("1b · Create region set from layout boxes (scanned pages)"):
        st.caption("Pipeline: layout detection -> automatic cropping (embedded page pixels, PNG) -> Qari OCR -> "
                   "validation. Paddle is used for layout detection only.")
        kind = st.radio("Layout source", list(layout_regions.LAYOUT_SOURCES),
                        format_func=layout_regions.LAYOUT_SOURCES.get, key=f"p2-lb-kind-{run_dir.name}")
        detected = st.session_state.setdefault("p2_layout_blocks", {})
        report = None
        if kind == "saved_provider_output":
            spec = specs.get(enrich.PADDLE_PROVIDER_ID)
            with_runs = [s_ for s_ in sets if spec and (s_ / "runs").is_dir() and any((s_ / "runs").glob(f"{spec.id}_*"))]
            if not with_runs:
                st.info("No region set has a saved PaddleOCR-VL run to take layout boxes from.")
                return
            names = [s_.name for s_ in with_runs]
            source_set = with_runs[names.index(st.selectbox("Source region set", names, key=f"p2-lb-set-{run_dir.name}"))]
            runs = sorted((source_set / "runs").glob(f"{spec.id}_*"))
            run_names = [r.name for r in runs]
            source_run = runs[run_names.index(st.selectbox("Saved PaddleOCR-VL run", run_names,
                                                            key=f"p2-lb-run-{run_dir.name}"))]
            key = f"saved/{source_set.name}/{source_run.name}"
            source_desc = {"kind": "saved_provider_output", "provider_id": spec.id, "set_id": source_set.name,
                           "run": source_run.name, "note": "historical benchmark; test of the cropping mechanism"}
            if st.button("Detect layout boxes (local; from the saved output)", key=f"p2-lb-detect-{run_dir.name}"):
                try:
                    detected[key] = (layout_regions.saved_provider_output(source_set, source_run, spec, store.root), None)
                except (layout_regions.LayoutSourceError, enrich.EnrichError) as exc:
                    st.error(str(exc))
        else:
            st.caption("Runs PaddleOCR's LayoutDetection on THIS machine, on each scanned page's embedded image at "
                       "native resolution, with no unwarping or orientation correction. Weights must already be "
                       "installed locally (no automatic download); the outbound network guard is active.")
            model_name = st.selectbox("Model", list(layout_regions.PP_DOCLAYOUT_MODELS), key=f"p2-lb-model-{run_dir.name}")
            model_dir = st.text_input("Local model directory (unpacked *_infer package)", key=f"p2-lb-dir-{run_dir.name}")
            device = st.text_input("Device (e.g. cpu, gpu:0; empty = library default)", key=f"p2-lb-dev-{run_dir.name}")
            threshold = st.number_input("Score threshold (0 = model default)", 0.0, 1.0, 0.0, 0.05,
                                        key=f"p2-lb-thr-{run_dir.name}")
            setups = st.session_state.setdefault("p2_doclayout_setup", {})
            if st.button("Check local model setup (no model load)", key=f"p2-lb-check-{run_dir.name}"):
                setups[(model_name, model_dir)] = layout_regions.doclayout_setup(model_dir or None, model_name)
            setup = setups.get((model_name, model_dir))
            if setup:
                (st.success if setup["ok"] else st.error)(
                    "Setup " + ("ready" if setup["ok"] else "not ready: " + "; ".join(setup["problems"])))
                st.caption(f"Versions: {setup['versions']} · labels: {setup['labels'] or 'not readable'} · "
                           f"title attachment available: {setup['title_attachment_available']}")
            key = f"local/{model_name}/{model_dir}/{device}/{threshold}"
            source_desc = {"kind": "pp_doclayout_local", "model_name": model_name, "model_dir": model_dir,
                           "device": device or None, "threshold": threshold or None}
            if st.button("Run local layout detection (this machine; no network)", key=f"p2-lb-run-local-{run_dir.name}",
                         disabled=not (setup and setup["ok"])):
                bar = st.progress(0.0, text="Detecting layout...")
                try:
                    blocks_, report_ = layout_regions.pp_doclayout_local(
                        run_dir, pdf_path=source, pages=None, model_dir=model_dir, model_name=model_name,
                        device=device or None, threshold=threshold or None)
                    detected[key] = (blocks_, report_)
                except layout_regions.LayoutSourceError as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001 - the model failed; nothing is saved
                    st.error(f"Local layout detection failed: {type(exc).__name__}: {exc}")
                bar.progress(1.0, text="Done")
        if key not in detected:
            return
        blocks, report = detected[key]
        if report:
            st.markdown(panel._table(["Page", "Status", "Details"], [
                [panel._esc(item["page"]), panel._esc(item["status"]),
                 panel._esc(item.get("reason") or f"{item.get('blocks')} block(s), {item.get('image_px_size')} px")]
                for item in report["pages"]]), unsafe_allow_html=True)
        if not blocks:
            st.warning("No layout boxes were found.")
            return
        # Options are kept outside widget state: Streamlit drops a widget's value after any rerun in which
        # this section is not drawn, which silently reset the margin to its default before saving.
        options = st.session_state.setdefault("p2_lb_options", {}).setdefault(run_dir.name, {})
        found_labels = sorted({b["label"] for b in blocks} | set(layout_regions.DEFAULT_LABELS))
        default_labels = [label for label in options.get("labels", layout_regions.DEFAULT_LABELS) if label in found_labels]
        labels = st.multiselect("Labels to keep", found_labels, default=default_labels,
                                key=f"p2-lb-labels-{run_dir.name}")
        margin = int(st.number_input("Margin around each box (px of the embedded page image)", 0, 200,
                                     int(options.get("margin", layout_regions.DEFAULT_MARGIN_PX)), 1,
                                     key=f"p2-lb-margin-{run_dir.name}"))
        titles = st.checkbox("Attach figure titles directly above (gap <= 30 pt, overlap >= 50%)",
                             value=bool(options.get("titles", True)), key=f"p2-lb-titles-{run_dir.name}")
        captions_below = st.checkbox("Also attach captions below (optional)",
                                     value=bool(options.get("captions_below", False)), key=f"p2-lb-captions-{run_dir.name}")
        options.update(labels=list(labels), margin=margin, titles=titles, captions_below=captions_below)
        import pymupdf  # noqa: PLC0415

        from lab.core.hybrid import run as hybrid  # noqa: PLC0415

        metadata, _plan, _library = hybrid.load(run_dir)
        pdf, _origin = hybrid.layout_pdf(run_dir, metadata, source)
        chosen: list[str] = []
        expansion_prefix = f"p2-lb-exp-{run_dir.name}-{key}-{margin}-{titles}-"
        expansions = [name[len(expansion_prefix):] for name, value in st.session_state.items()
                      if isinstance(name, str) and name.startswith(expansion_prefix) and value]
        with pymupdf.open(pdf) as document:
            candidates = layout_regions.propose(document, blocks, labels=labels, margin_px=margin, titles=titles,
                                                captions_below=captions_below, expansions=expansions)
            for page in sorted({c["page"] for c in candidates}):
                rows = [c for c in candidates if c["page"] == page]
                st.markdown(f"**Page {page}**")
                st.markdown(panel._table(["Region", "Label", "Score", "Box (pt)", "Size (pt)", "Status", "Decisions",
                                          "Reasons"], [
                    [panel._esc(c.get("region_id") or "-"), panel._esc(c["label"]),
                     panel._esc(f"{c['score']:.2f}" if isinstance(c.get("score"), (int, float)) else "-"),
                     panel._esc(c["bbox_pt"]),
                     panel._esc(f"{c['bbox_pt'][2] - c['bbox_pt'][0]:.0f}×{c['bbox_pt'][3] - c['bbox_pt'][1]:.0f}"),
                     panel._esc(c["status"]), panel._esc(_decision_text(c)),
                     panel._esc("; ".join(c["reasons"]) or "-")] for c in rows]), unsafe_allow_html=True)
                for candidate in rows:
                    if not candidate.get("region_id") or not candidate.get("pixel_box"):
                        continue
                    images = layout_regions.preview_images(document, candidate)
                    left, right = st.columns(2)
                    if images:
                        left.image(images[0], caption=f"{candidate['region_id']} on page {page} (box incl. margin)",
                                   width="stretch")
                        right.image(images[1], caption=f"crop {candidate['pixel_box']} px", width="stretch")
                    if candidate.get("proposed_expansion_pt"):
                        right.checkbox(f"Use proposed expansion for {candidate['region_id']} (detected box "
                                       f"{candidate.get('bbox_pt_detected')} is kept in the record)",
                                       value=False, key=expansion_prefix + candidate["region_id"])
                    if right.checkbox(f"Include {candidate['region_id']} ({candidate['label']}, {candidate['status']})",
                                      value=candidate["default_selected"],
                                      key=f"p2-lb-pick-{run_dir.name}-{key}-{candidate['region_id']}-{margin}-{titles}"):
                        chosen.append(candidate["region_id"])
        previewed = {c["region_id"]: c["pixel_box"] for c in candidates if c.get("region_id") in set(chosen)}
        st.caption(f"This preview uses a margin of {margin} px; the saved crops must equal the previewed boxes "
                   "(checked when saving).")
        by = st.text_input("Your name", key=f"p2-lb-by-{run_dir.name}")
        if st.button(f"Save region set from {len(chosen)} layout box(es) (margin {margin} px)",
                     key=f"p2-lb-save-{run_dir.name}", disabled=not chosen or not by.strip()):
            try:
                folder = layout_regions.create_layout_region_set(
                    run_dir, blocks=blocks, labels=labels, margin_px=margin, selected=chosen, pdf_path=source, by=by,
                    source=source_desc | ({"provenance": report["provenance"]} if report else {}), titles=titles,
                    captions_below=captions_below, detection_report=report, expansions=expansions,
                    expected_pixel_boxes=previewed)
                st.session_state["hy-flash"] = ("success", f"Region set {folder.name} saved from layout boxes.")
                st.rerun()
            except (enrich.EnrichError, ValueError) as exc:
                st.error(str(exc))


def _crops(set_dir: Path, data: dict) -> None:
    with st.expander(f"Regions in {data['set_id']} ({len(data['regions'])})"):
        st.caption(f"Document {data['document']['display_name']} · SHA-256 {data['document']['sha256']}")
        columns = st.columns(3)
        for index, region in enumerate(data["regions"]):
            with columns[index % 3]:
                st.image(str(set_dir / region["crop_file"]), width="stretch")
                options = crop_options.region_options(region)
                st.caption(f"{region['region_id']} · page {region['page']} · {region['detected_type']} · "
                           f"{_crop_line(options)} · bbox {region['bbox']} · SHA-256 {region['crop_sha256'][:16]}…")
                if options.get("fallback_reason"):
                    st.caption(options["fallback_reason"])
                if options.get("format_reason"):
                    st.caption(options["format_reason"])
                if options.get("large_file_warning"):
                    st.warning(f"Large file ({options['file_bytes'] / 1_048_576:.2f} MB > 2 MB): uploads may time out.")
                if region["uncertain"]:
                    st.warning("Uncertain region: " + "; ".join(region["uncertainty"]))


def _manual_qari(set_dir: Path, data: dict) -> None:
    with st.expander("2 · Offline workflow: export crops, run your own script, import the texts"):
        st.caption("Export → run Qari (or another model) yourself on crops/*.png → put one UTF-8 file per region "
                   "named {region_id}.txt plus the unchanged manifest.json in a ZIP → import. No request is sent.")
        provider = st.text_input("Intended provider", value="Qari-OCR 0.4.0 (local script)", key=f"p2-exp-prov-{set_dir.name}")
        model = st.text_input("Intended model", value=qari.QARI_MODEL, key=f"p2-exp-model-{set_dir.name}")
        by = st.text_input("Your name", key=f"p2-exp-by-{set_dir.name}")
        exports = st.session_state.setdefault("p2_exports", {})
        if st.button("Prepare crop export (ZIP + manifest)", key=f"p2-exp-{set_dir.name}", disabled=not by.strip()):
            exports[set_dir.name] = enrich.export_crops_zip(set_dir, provider=provider, model=model, by=by)
        if set_dir.name in exports:
            st.download_button("Download crops ZIP", data=exports[set_dir.name], file_name=f"{data['set_id']}_crops.zip",
                               mime="application/zip", key=f"p2-exp-dl-{set_dir.name}")
        upload = st.file_uploader("Results ZIP ({region_id}.txt files + manifest.json)", type=["zip"],
                                  key=f"p2-imp-file-{set_dir.name}")
        if upload is not None:
            report = enrich.check_import(set_dir, upload.getvalue())
            (st.success if report["ok"] else st.error)(
                "Check: " + ("ok" if report["ok"] else "; ".join(report["errors"]))
                + f" · texts {len(report['texts'])} · missing {report['missing'] or 'none'}"
                + f" · duplicates {report['duplicates'] or 'none'} · unknown {report['unknown'] or 'none'}")
            with st.form(f"p2-imp-form-{set_dir.name}"):
                declared = st.text_input("Declared model (recorded, not verified)", value=qari.QARI_MODEL)
                reviewer = st.text_input("Reviewer")
                agree = st.checkbox(enrich.IMPORT_STATEMENT)
                submit = st.form_submit_button("Import these texts", disabled=not report["ok"])
            if submit:
                try:
                    if not agree:
                        raise enrich.EnrichError("Tick the import statement")
                    folder = enrich.import_results(set_dir, upload.getvalue(), declared_model=declared,
                                                   reviewer=reviewer, statement=enrich.IMPORT_STATEMENT)
                    st.session_state["hy-flash"] = ("success", f"Imported as {folder.name} (manual_import).")
                    st.rerun()
                except enrich.EnrichError as exc:
                    st.error(str(exc))


def _send_section(set_dir: Path, data: dict, specs: dict, *, document, source, store, runner, workspace_id) -> None:
    with st.expander("3 · Send crops to providers (explicit approval; external transmission)"):
        doc_state = documents.authorization_state(data["document"]["sha256"], root=store.root)
        (st.success if doc_state["allowed"] else st.warning)(
            f"Document {data['document']['display_name']} · SHA-256 {data['document']['sha256']}: "
            + ("authorized for external transmission" if doc_state["allowed"]
               else "NOT authorized (" + "; ".join(doc_state["reasons"]) + "). Authorize it on the Hybrid page "
                    "(section 5) or on OCR/VLM Evaluation first."))
        providers = st.multiselect("Providers (each receives the same crops)", list(PROVIDERS),
                                   format_func=PROVIDERS.get, key=f"p2-providers-{set_dir.name}")
        regions = st.multiselect("Regions", [r["region_id"] for r in data["regions"]],
                                 default=[r["region_id"] for r in data["regions"]], key=f"p2-send-regions-{set_dir.name}")
        selected_regions = [r for r in data["regions"] if r["region_id"] in regions]
        override = st.number_input("Request timeout per call in seconds (0 = provider default: Baidu 60 s below 25% "
                                   "of the page, 180 s otherwise; Oxen and Qari keep their registry timeout)",
                                   0, 600, 0, 10, key=f"p2-timeout-{set_dir.name}")
        oxen = specs.get("oxen-qwen3-vl-4b-instruct")
        st.markdown(panel._table(["Region", "Crop", "Baidu default", "Oxen default", "Qari default", "Override"], [
            [panel._esc(r["region_id"]), panel._esc(_crop_line(crop_options.region_options(r))),
             panel._esc(f"{crop_options.size_based_timeout(r)} s"),
             panel._esc(f"{oxen.limits.request_timeout_s:.0f} s" if oxen else "-"),
             panel._esc(f"{qari.QARI_TIMEOUT_S} s"), panel._esc(f"{int(override)} s" if override else "-")]
            for r in selected_regions]), unsafe_allow_html=True)
        large = [r["region_id"] for r in selected_regions if crop_options.region_options(r).get("large_file_warning")]
        if large:
            st.warning(f"Files larger than 2 MB: {large}. Consider a new region set with native/JPEG/lower DPI.")
        uncertain = [r["region_id"] for r in data["regions"] if r["region_id"] in regions and r["uncertain"]]
        if uncertain:
            st.warning(f"Selected uncertain regions: {uncertain}. They are sent only because you selected them.")
        credentials, blockers, rows = {}, [], []
        qari_url = qari_token = qari_model = ""
        paddle_mode = "standard"
        if enrich.PADDLE_PROVIDER_ID in providers and enrich.PADDLE_PROVIDER_ID in specs:
            paddle_mode = st.radio("PaddleOCR-VL extraction mode", list(enrich.PADDLE_MODES),
                                   format_func=lambda key: enrich.PADDLE_MODES[key]["label"],
                                   key=f"p2-paddle-mode-{set_dir.name}")
            st.caption("Request options (optionalPayload) sent: "
                       + json.dumps(enrich.mode_spec(specs[enrich.PADDLE_PROVIDER_ID], paddle_mode).optional_payload)
                       + " · documented in " + "; ".join(enrich.PADDLE_DOC_SOURCES))
            for note in enrich.PADDLE_UNSUPPORTED:
                st.warning(note)
            st.caption("Earlier completed requests are kept unchanged. Sending the same crops in another mode is a "
                       "new, explicitly approved request that counts against the provider limits.")
        for provider in providers:
            if provider == "qari":
                qari_url = st.text_input("Qari endpoint URL", key=f"p2-qari-url-{set_dir.name}")
                qari_token = st.text_input("Qari server token (optional, session only)", type="password",
                                           key=f"p2-qari-token-{set_dir.name}")
                qari_model = st.text_input("Served model name", value=qari.QARI_MODEL, key=f"p2-qari-model-{set_dir.name}")
                checks = st.session_state.setdefault("p2_qari_ready", {})
                if st.button("Check Qari connection (lists models only)", key=f"p2-qari-check-{set_dir.name}",
                             disabled=not qari_url):
                    checks[(qari_url, qari_model)] = qari.check_ready(qari_url, qari_token or None, model=qari_model)
                status = checks.get((qari_url, qari_model))
                if not status or not status["ready"]:
                    blockers.append("Qari connection not verified" + (f": {status['error']}" if status else ""))
                rows.append([PROVIDERS["qari"], panel._esc(qari_url or "-"), panel._esc(len(regions)), "your server",
                             panel._esc(enrich.BILLING_NOTES["qari"])])
                continue
            spec = specs.get(provider)
            if spec is None:
                blockers.append(f"{PROVIDERS[provider]} is not configured")
                continue
            store_keys = st.session_state.setdefault("fp_credentials", {})
            typed = st.text_input(f"{spec.credential_label} (session memory only)", type="password",
                                  key=f"p2-key-{spec.id}-{set_dir.name}")
            if typed:
                store_keys[spec.id] = typed
            credentials[provider] = panel._credential(spec)
            room = enrich.capacity(spec, store.root)
            if not credentials[provider]:
                blockers.append(f"enter the {spec.credential_label}")
            if not room["allowed"] or room["remaining_requests"] < len(regions):
                blockers.append(f"{spec.provider}: {len(regions)} request(s) needed, {room['remaining_requests']} left "
                                + ("; " + "; ".join(room["reasons"]) if room["reasons"] else ""))
            label = PROVIDERS[provider] + (f" · {enrich.PADDLE_MODES[paddle_mode]['label']}"
                                           if provider == enrich.PADDLE_PROVIDER_ID else "")
            rows.append([panel._esc(label), panel._esc(spec.endpoint), panel._esc(len(regions)),
                         panel._esc(f"{room['summary']}; history {room['used_total']} · {room['limit_source']}"),
                         panel._esc(f"{room['billing']}; preflight bound {room['estimated_usd_per_request']} USD/request")])
        if not doc_state["allowed"] and providers:
            blockers.append("document not authorized")
        if rows:
            st.markdown(panel._table(["Provider", "Destination", "Requests", "Capacity", "Billing / quota"], rows),
                        unsafe_allow_html=True)
            st.caption(f"{len(regions)} region(s) × {len(providers)} provider(s) = {len(regions) * len(providers)} "
                       "request(s). Only the listed crops are transmitted; never the PDF, other regions, library text "
                       "or Ground Truth. No automatic retries.")
        by = st.text_input("Your name (recorded with the approval)", key=f"p2-send-by-{set_dir.name}")
        destinations = ", ".join(PROVIDERS[p] + (f" ({enrich.PADDLE_MODES[paddle_mode]['label']})"
                                                 if p == enrich.PADDLE_PROVIDER_ID else "") for p in providers)
        agree = st.checkbox(f"I approve transmitting {len(regions)} crop(s) of {data['document']['display_name']} "
                            f"to: {destinations or '-'}. {enrich.APPROVAL_STATEMENT}",
                            key=f"p2-send-agree-{set_dir.name}-{len(regions)}-{'-'.join(providers)}-{paddle_mode}")
        disabled = bool(blockers) or not providers or not regions or not agree or not by.strip()
        if blockers:
            st.caption("Sending is disabled: " + "; ".join(blockers))
        if st.button("Approve & send selected crops", type="primary", disabled=disabled, key=f"p2-send-{set_dir.name}"):
            bar = st.progress(0.0, text="Starting...")
            messages = []

            def progress(event, index, total, record):
                bar.progress(min(1.0, index / max(total, 1)), text=f"Region {index}/{total}: {event}")

            for provider in providers:
                try:
                    if provider == "qari":
                        run = enrich.create_qari_run(set_dir, regions, endpoint_url=qari_url, model=qari_model, by=by,
                                                     statement=enrich.APPROVAL_STATEMENT, root=store.root,
                                                     timeout_override=override or None)
                        result = enrich.execute_qari_run(run, token=qari_token or None,
                                                         document_sha256=data["document"]["sha256"], root=store.root,
                                                         on_progress=progress)
                    else:
                        run = enrich.create_api_run(
                            set_dir, specs[provider], regions, by=by, statement=enrich.APPROVAL_STATEMENT,
                            root=store.root, mode=paddle_mode if provider == enrich.PADDLE_PROVIDER_ID else "standard",
                            timeout_override=override or None)
                        result = enrich.execute_api_run(run, runner, specs[provider], workspace_id=workspace_id,
                                                        document=document, pdf_path=source,
                                                        credential=credentials[provider], root=store.root,
                                                        on_progress=progress)
                    messages.append(f"{PROVIDERS[provider]}: {result.get('stopped') or 'done'}")
                except (enrich.EnrichError, qari.QariError, APIRunRefused) as exc:
                    messages.append(f"{PROVIDERS[provider]}: not sent ({exc})")
            st.session_state["hy-flash"] = ("info", " · ".join(messages))
            st.rerun()
        baidu_runs = sorted((set_dir / "runs").glob("baidu-*")) if (set_dir / "runs").is_dir() else []
        if baidu_runs and "baidu-aistudio-paddleocr-vl-1.6" in specs:
            if st.button("Resume paused Baidu regions from their Job IDs (no resubmission)", key=f"p2-resume-{set_dir.name}"):
                spec = specs["baidu-aistudio-paddleocr-vl-1.6"]
                notes = []
                for run in baidu_runs:
                    notes += enrich.resume_api_run(run, runner, spec, credential=panel._credential(spec), root=store.root)
                st.session_state["hy-flash"] = ("info", "; ".join(notes) or "Paused Baidu regions processed.")
                st.rerun()


def _reference_section(set_dir: Path, data: dict, outputs: dict) -> dict | None:
    with st.expander("4 · Image Ground Truth (evaluation only; never used to choose outputs)"):
        upload = st.file_uploader("AgenticDoc benchmark JSON (pages[].visual_elements, pages[].tables)", type=["json"],
                                  key=f"p2-gt-{set_dir.name}")
        by = st.text_input("Your name", key=f"p2-gt-by-{set_dir.name}")
        if upload is not None and st.button("Import reference (stored unchanged)", key=f"p2-gt-save-{set_dir.name}",
                                            disabled=not by.strip()):
            try:
                enrich.import_reference(set_dir, upload.getvalue(), filename=upload.name, by=by)
                st.rerun()
            except enrich.EnrichError as exc:
                st.error(str(exc))
        refs = enrich.references(set_dir)
        if not refs:
            return None
        ref_dir = refs[[r.name for r in refs].index(st.selectbox("Reference", [r.name for r in refs], index=len(refs) - 1,
                                                                  key=f"p2-gt-pick-{set_dir.name}"))]
        proposals = enrich.propose_matches(set_dir, ref_dir)
        current = enrich.confirmed_matches(set_dir, ref_dir)
        with st.form(f"p2-match-{set_dir.name}-{ref_dir.name}"):
            confirmations = {}
            for item in proposals:
                options = ["(no reference)"] + [c["ref_id"] for c in item["candidates"]]
                default = current.get(item["region_id"]) or item["proposed"]
                st.caption(f"{item['region_id']} · page {item['page']} · {item['detected_type']} · {item['basis']}")
                choice = st.selectbox(item["region_id"], options,
                                      index=options.index(default) if default in options else 0,
                                      key=f"p2-match-{set_dir.name}-{ref_dir.name}-{item['region_id']}")
                confirmations[item["region_id"]] = None if choice == "(no reference)" else choice
            match_by = st.text_input("Your name")
            agree = st.checkbox(enrich.MATCH_STATEMENT)
            submit = st.form_submit_button("Confirm matches")
        if submit:
            try:
                if not agree:
                    raise enrich.EnrichError("Tick the confirmation")
                enrich.save_matches(set_dir, ref_dir, confirmations, by=match_by, statement=enrich.MATCH_STATEMENT)
                st.rerun()
            except enrich.EnrichError as exc:
                st.error(str(exc))
        if current:
            result = enrich.evaluate(set_dir, outputs, ref_dir)
            st.caption(result["note"])
            rows = []
            for provider, data_ in result["providers"].items():
                for scope, agg in data_.items():
                    rows.append([panel._esc(provider), panel._esc(scope), panel._esc(agg["regions"]),
                                 panel._esc(_pct(agg["cer_normalized"])),
                                 panel._esc(f"{agg['numbers_matched']}/{agg['numbers_reference']}"),
                                 panel._esc(agg["numbers_not_in_reference"]),
                                 panel._esc(agg.get("unit_mismatches", 0)),
                                 panel._esc(f"{agg['labels_found']}/{agg['labels_reference']}"),
                                 panel._esc(f"{agg['table_cells_found']}/{agg['table_cells_reference']}")])
            st.markdown(panel._table(["Provider", "Scope", "Regions", "CER (normalized)", "Numbers found",
                                      "Numbers not in reference", "Unit mismatches", "Labels found", "Table cells found"], rows),
                        unsafe_allow_html=True)
            st.caption(f"Comparable regions (every provider has an output): {result['comparable_regions'] or 'none'}")
        return {"dir": ref_dir, "matches": current,
                "elements": {e["ref_id"]: e for e in enrich.load_reference(ref_dir)}}


def _compare_section(set_dir: Path, data: dict, outputs: dict, reference: dict | None) -> None:
    with st.expander("5 · Compare outputs and choose the preferred one per region", expanded=True):
        selection = enrich.latest_selection(set_dir)
        choices = (selection or {}).get("choices") or {}
        new_choices = {}
        for region in data["regions"]:
            rid = region["region_id"]
            st.markdown(f"**{rid}** · page {region['page']} · {region['detected_type']}")
            region_outputs = outputs.get(rid, [])
            columns = st.columns(1 + max(1, len(region_outputs)) + (1 if reference and rid in reference["matches"] else 0))
            columns[0].image(str(set_dir / region["crop_file"]), width="stretch")
            for column, output in zip(columns[1:], region_outputs):
                with column:
                    assessment = output["assessment"]
                    st.caption(f"{enrich.output_label(output)} · {output['status']} · {output['job_state']}"
                               + (f" · {output['latency_s']} s" if output.get("latency_s") is not None else "")
                               + (f" · tokens {output['tokens'].get('prompt')}/{output['tokens'].get('completion')}"
                                  if (output.get("tokens") or {}).get("reported") else ""))
                    if assessment["content"]["kind"] == "html_image_reference_only":
                        st.error("Image reference only: " + ", ".join(assessment["content"]["image_references"])
                                 + ". No text, labels or numbers were extracted.")
                    elif assessment["reasons"]:
                        st.warning("; ".join(assessment["reasons"]))
                    validation = output.get("unit_validation") or {}
                    if validation.get("flags"):
                        st.warning("Number formats: " + "; ".join(validation["flags"][:6]))
                    if output.get("unit_conflicts") and output["usable"]:
                        st.warning("Unit differs between providers: " + "; ".join(
                            f"{c['value']} {c['units_by_output']}" for c in output["unit_conflicts"][:4]))
                    st.caption(f"Numbers (visible text only): "
                               f"{' '.join(n['original'] for n in assessment['numbers'][:30]) or 'none'}")
                    if output.get("crop_options"):
                        st.caption("Crop: " + _crop_line(output["crop_options"])
                                   + (f" · timeout {output['timeout_s']:.0f} s" if output.get("timeout_s") else ""))
                    if output.get("optional_payload") is not None:
                        st.caption(f"Request options: {json.dumps(output['optional_payload'])}")
                    charts = [b for b in output.get("structured_blocks") or [] if b.get("label") in {"chart", "table"}]
                    for block in charts:
                        st.caption(f"Returned {block['label']} block: "
                                   + (block["visible_text"][:300] or "(empty: no values returned)"))
                    st.markdown(text_html(output["text"] or "(no text)"), unsafe_allow_html=True)
            if reference and rid in reference["matches"]:
                element = reference["elements"].get(reference["matches"][rid])
                with columns[-1]:
                    st.caption(f"Reference {reference['matches'][rid]} (evaluation only)")
                    st.markdown(text_html((element or {}).get("text") or ""), unsafe_allow_html=True)
            options = ["(keep placeholder)"] + [o["output_key"] for o in region_outputs if o["usable"]]
            current = (choices.get(rid) or {}).get("output_key")
            pick = st.selectbox(f"Preferred output for {rid}", options,
                                index=options.index(current) if current in options else 0,
                                key=f"p2-pick-{set_dir.name}-{rid}")
            reason = st.text_input(f"Reason for {rid}", value=(choices.get(rid) or {}).get("reason") or "",
                                   key=f"p2-reason-{set_dir.name}-{rid}")
            new_choices[rid] = {"output_key": None if pick == "(keep placeholder)" else pick, "reason": reason}
        by = st.text_input("Your name", key=f"p2-select-by-{set_dir.name}")
        agree = st.checkbox(enrich.SELECTION_STATEMENT, key=f"p2-select-agree-{set_dir.name}")
        if st.button("Save selection (append-only)", key=f"p2-select-save-{set_dir.name}", disabled=not agree):
            try:
                enrich.save_selection(set_dir, new_choices, by=by, statement=enrich.SELECTION_STATEMENT)
                st.session_state["hy-flash"] = ("success", "Selection saved.")
                st.rerun()
            except enrich.EnrichError as exc:
                st.error(str(exc))


def _documents_section(run_dir: Path, set_dir: Path, spec_for, run_spec, store, source) -> None:
    with st.expander("6 · Unified page documents (from saved outputs; no request)"):
        previews = st.session_state.setdefault("p2_doc_preview", {})
        if st.button("Preview unified documents", key=f"p2-doc-preview-{set_dir.name}"):
            try:
                previews[set_dir.name] = enrich.compute_enriched(run_dir, set_dir, spec_for, store.root,
                                                                 run_spec=run_spec, pdf_path=source)
            except Exception as exc:  # noqa: BLE001 - preview only
                st.error(f"Preview failed: {exc}")
        preview = previews.get(set_dir.name)
        if preview:
            st.info("Automatically assembled from the latest selection (not saved, not reviewed).")
            page = st.selectbox("Page", [p["page"] for p in preview["pages"]], key=f"p2-doc-page-{set_dir.name}")
            chosen = next(p for p in preview["pages"] if p["page"] == page)
            for note in chosen["record"]["needs_review"]:
                st.warning(note)
            st.markdown(chosen["markdown"])
            st.caption("Image extraction metadata (sources, numbers, overlap)")
            st.json(chosen["record"].get("image_extraction"), expanded=False)
            with st.form(f"p2-doc-save-{set_dir.name}"):
                by = st.text_input("Your name")
                agree = st.checkbox(enrich.DOCUMENT_STATEMENT)
                submit = st.form_submit_button("Save as a new enrichment revision")
            if submit:
                try:
                    if not agree:
                        raise enrich.EnrichError("Tick the confirmation")
                    folder = enrich.apply_enriched(run_dir, set_dir, spec_for, store.root, run_spec=run_spec, by=by,
                                                   statement=enrich.DOCUMENT_STATEMENT, pdf_path=source)
                    previews.pop(set_dir.name, None)
                    st.session_state["hy-flash"] = ("success", f"Enrichment revision saved: {folder.name}")
                    st.rerun()
                except enrich.EnrichError as exc:
                    st.error(str(exc))
        latest = enrich.latest_enriched(run_dir)
        if latest:
            st.caption(f"Current approved page documents: {latest['revision']} by {latest['approved_by']}")
            st.download_button("Download full_document.md (enriched)", data=latest["full_document"].encode("utf-8"),
                               file_name="full_document.md", mime="text/markdown", key=f"p2-doc-full-{run_dir.name}")
            st.caption(json.dumps({"folder": latest["folder"]}, ensure_ascii=False))
