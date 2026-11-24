"""Automatic Extraction runs: plan -> Qari jobs (resumable) -> assemble -> finalize. No approval step.

    results/automatic_extraction/run_YYYYMMDD_HHMMSS_xxxxxx/        (unique; every file write-once)
        source.pdf                     exact copy of the uploaded PDF (same SHA-256)
        plan/run.json                  document, config snapshot, proven-setting differences
        plan/analysis.json             Phase 1 library pass (PyMuPDF raw/processed, tables, categories)
        plan/routing.json              DIGITAL / SCANNED / MIXED per page with rule and evidence
        plan/layout.json               PP-DocLayout boxes, region decisions, review queue (per page)
        plan/jobs.json                 the OCR jobs (crop / page) with image SHA-256
        crops/<job_id>.png             every image a job would send (native pixels unless recorded); for
                                       page/body jobs this is the MASKED copy (visuals painted white)
        pages/<job_id>_original.png    the unmasked page image of each page/body job (audit, reproduction)
        review/process_now_*.json      jobs added by a reviewer ("Process now")
        service/service_*.json         audit: Qari endpoint, deployment mode and trust basis per execution
        raw/<job_id>/attempt_NN.intent.json / attempt_NN.result.json   every request and its outcome
        imports/import_*.json          offline result imports (hash-checked)
        verification/<job_id>/         digit second read: upscaled input copy, intent/result, skips (report only)
        document.md  document.json  validation_report.json  rag_chunks.jsonl
        execution_manifest.json        written LAST: its presence means the run is finalized

Images go only to the Qari service the administrator configured and trusts (QARI_DEPLOYMENT_MODE;
see qari_client.endpoint_info); SHA-256 hashes are kept for integrity and provenance, never as a user
approval. Qari is the only OCR model: if the service is unavailable the run
pauses (resumable) and the jobs can be exported for an offline Qari run and imported back. A request
that may have reached the service without a response is never resent automatically.
"""

from __future__ import annotations

import hashlib
import io
import json
import secrets
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lab import __version__
from lab.core.auto import config as auto_config
from lab.core.auto import images, masking, qari_client, regions, router, visual_crops
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.store import results_root
from lab.core.hybrid import analysis as hybrid_analysis
from lab.core.hybrid import layout as hybrid_layout
from lab.core.hybrid import qari as hybrid_qari

PIPELINE_VERSION = "1"
FOLDER = "automatic_extraction"
COMPLETED, SERVICE_ERROR, UNCONFIRMED, NOT_TRANSMITTED = "completed", "service_error", "unconfirmed", "not_transmitted"
Progress = Callable[[str, int, int], None]


class PipelineError(ValueError):
    pass


# ------------------------------------------------------------------ storage


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_once(path: Path, payload: dict | list | str | bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        data = payload
    elif isinstance(payload, str):
        data = payload.encode("utf-8", errors="surrogatepass")
    else:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    with path.open("xb") as handle:  # never overwrites
        handle.write(data)
    return path


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def auto_root(config: auto_config.AutoConfig, root: Path | None = None) -> Path:
    return Path(root or config.app.results_dir or results_root()) / FOLDER


def _new_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    while True:
        path = base / f"run_{datetime.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue


def is_finalized(run_dir: Path) -> bool:
    return (Path(run_dir) / "execution_manifest.json").is_file()


def list_runs(config: auto_config.AutoConfig, root: Path | None = None) -> list[dict]:
    base = auto_root(config, root)
    runs = []
    for folder in sorted(base.glob("run_*"), reverse=True) if base.is_dir() else []:
        info = read_json(folder / "plan" / "run.json") or {}
        runs.append({"run_dir": str(folder), "run_id": folder.name, "finalized": is_finalized(folder),
                     "document": (info.get("document") or {}).get("display_name"),
                     "sha256": (info.get("document") or {}).get("sha256"),
                     "pages": (info.get("document") or {}).get("page_count"), "created_at": info.get("created_at")})
    return runs


# ------------------------------------------------------------------ plan


def _page_png(document, page_number: int, dpi: int) -> tuple[bytes, dict]:
    page = document[page_number - 1]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    data = pix.tobytes("png")
    return data, {"method": "rendered", "dpi": dpi, "pixel_size": [pix.width, pix.height], "format": "png",
                  "sha256": sha256(data), "bytes": len(data), "native_pixels": False}


def _job(run_dir: Path, job_id: str, page: int, kind: str, purpose: str, data: bytes, meta: dict, **extra) -> dict:
    name = f"crops/{job_id}.png"
    write_once(run_dir / name, data)
    return {"job_id": job_id, "page": page, "kind": kind, "purpose": purpose, "image_file": name,
            "image_sha256": sha256(data), "image": meta, **extra}


def _region_jobs(run_dir: Path, page_number: int, selection: dict, page_image: images.PageImage, prefix: str,
                 max_pixels: int | None, **extra) -> list[dict]:
    jobs = []
    for region in selection["regions"]:
        if region["decision"] == regions.REMOVED:
            continue
        data, meta = images.crop_png(page_image, region["pixel_box"], max_image_pixels=max_pixels)
        x0, y0, x1, y1 = region["pixel_box"]
        sx = (page_image.bbox_pt[2] - page_image.bbox_pt[0]) / page_image.size[0]
        sy = (page_image.bbox_pt[3] - page_image.bbox_pt[1]) / page_image.size[1]
        region["crop_bbox_pt"] = [round(page_image.bbox_pt[0] + x0 * sx, 2), round(page_image.bbox_pt[1] + y0 * sy, 2),
                                  round(page_image.bbox_pt[0] + x1 * sx, 2), round(page_image.bbox_pt[1] + y1 * sy, 2)]
        job_id = f"{prefix}_{region['region_id'].split('_', 1)[1]}"
        region["job_id"] = job_id
        job = _job(run_dir, job_id, page_number, "crop", "region", data, meta, region_id=region["region_id"],
                   label=region["label"], automatic=region["decision"] in regions.PROCESSED,
                   **extra)
        jobs.append(job)
    return jobs


def _body_image(run_dir: Path, job_id: str, page_image: images.PageImage, selection: dict,
                max_pixels: int | None) -> tuple[bytes, dict, dict]:
    """The image for body-text OCR: a COPY of the page image with every automatically processed visual
    masked white (masking.py). The original page image is saved unchanged next to it for audit."""
    masked, record = masking.plan(selection, page_image.image)
    original, original_meta = images.encode_png(page_image.image, meta={"method": page_image.method,
                                                                         **page_image.meta})
    original_file = f"pages/{job_id}_original.png"
    write_once(run_dir / original_file, original)
    record.update(original_image_file=original_file, original_image_sha256=original_meta["sha256"],
                  original_image_px=original_meta["pixel_size"])
    data, meta = images.encode_png(masked if masked is not None else page_image.image, max_image_pixels=max_pixels,
                                   meta={"method": page_image.method, **page_image.meta,
                                         "masked": masked is not None})
    record.update(sent_image="masked copy" if masked is not None else "original (no visual to mask)",
                  masked_image_file=f"crops/{job_id}.png" if masked is not None else None,
                  masked_image_sha256=meta["sha256"] if masked is not None else None)
    return data, meta, record


def create_run(pdf_path: Path, *, display_name: str, config: auto_config.AutoConfig,
               detector: regions.LayoutDetector | None, root: Path | None = None,
               on_progress: Progress | None = None) -> Path:
    """Analyse, route, detect layout, select regions and prepare every job. Nothing is sent."""
    import pymupdf  # noqa: PLC0415

    pdf_path = Path(pdf_path)
    data = pdf_path.read_bytes()
    if len(data) > config.app.max_pdf_mb * 1024 * 1024:
        raise PipelineError(f"the PDF exceeds {config.app.max_pdf_mb} MB (AGENTICDOC_MAX_PDF_MB)")
    progress = on_progress or (lambda *_: None)
    with pymupdf.open(stream=data, filetype="pdf") as probe:
        if probe.needs_pass:
            raise PipelineError("the PDF is password protected")
        if len(probe) > config.app.max_pages:
            raise PipelineError(f"the PDF has {len(probe)} pages; the limit is {config.app.max_pages}")
    run_dir = _new_run_dir(auto_root(config, root))
    write_once(run_dir / "source.pdf", data)
    source = run_dir / "source.pdf"
    progress("analysis", 0, 1)
    analysis = hybrid_analysis.analyze_pdf(source, on_page=lambda n, t: progress("analysis", n, t))
    write_once(run_dir / "plan" / "analysis.json", analysis)
    proven = config.proven
    margin, max_pixels = proven.layout.crop_margin_px, config.qari.max_image_pixels
    render_dpi = proven.layout.digital_render_dpi  # 250: digital/mixed visuals rendered from the page
    jobs: list[dict] = []
    layout_pages: list[dict] = []
    with pymupdf.open(source) as document:
        routes = router.route_document(document, analysis)
        write_once(run_dir / "plan" / "routing.json", {"router_version": router.ROUTER_VERSION, "pages": routes,
                                                       "counts": router.counts(routes)})
        needs_layout = any(r["route"] == router.SCANNED or r["embedded_scan_regions"] for r in routes)
        detector_info = None
        if needs_layout:
            if detector is None:
                raise PipelineError("this PDF has scanned content but no layout detector is configured")
            setup = detector.setup()
            if not setup["ok"]:  # fail clearly instead of planning pages with no layout boxes
                raise PipelineError("Local PP-DocLayout is not ready (LAYOUT_MODEL_DIR="
                                    f"{detector.model_dir!r}): " + "; ".join(setup["problems"]))
            detector_info = detector.describe()
        reorder = analysis["settings"].get("reorder") or {}
        for index, route in enumerate(routes, start=1):
            number = route["page"]
            progress("layout", index, len(routes))
            prefix = f"p{number:03d}"
            page_layout: dict[str, Any] = {"page": number, "route": route["route"]}
            if route["route"] == router.SCANNED:
                page_image = images.page_image(document, number, render_dpi=config.layout.page_render_dpi)
                page_layout["page_image"] = {"method": page_image.method, "px": page_image.size,
                                             "bbox_pt": page_image.bbox_pt, **page_image.meta}
                try:
                    boxes = detector.detect(page_image.image)
                except Exception as exc:  # noqa: BLE001 - recorded; the page still gets full-page OCR
                    boxes, page_layout["detection_error"] = [], f"{type(exc).__name__}: {exc}"
                selection = regions.select_regions(
                    number, boxes, image_px=page_image.size, image_bbox_pt=page_image.bbox_pt,
                    labels=config.layout.labels, margin_px=margin, attach_titles=config.layout.attach_titles,
                    captions_below=config.layout.captions_below, policy=config.layout.selection_policy)
                jobs += _region_jobs(run_dir, number, selection, page_image, prefix, max_pixels)
                body, meta, masking_record = _body_image(run_dir, f"{prefix}_page", page_image, selection, max_pixels)
                jobs.append(_job(run_dir, f"{prefix}_page", number, "page", "scanned_body", body, meta, automatic=True,
                                 masking=masking_record))
                page_layout.update(boxes=boxes, selection=selection,
                                   overlapping_auto=regions.overlapping_auto(selection["regions"]))
            else:
                tables_doc = next((p.get("tables") for p in analysis["pages"] if p["page"] == number), None)
                elements, _layout_crops = hybrid_layout.extract_page_elements(  # 150 DPI crops unused
                    document[number - 1], number, reorder_config=reorder, tables_doc=tables_doc)
                page_layout["figures"] = [{k: v for k, v in f.items() if k != "phase2"} for f in elements["figures"]]
                embedded = []
                layers = visual_crops.page_layers(document[number - 1])
                for e_index, bbox in enumerate(route["embedded_scan_regions"]):
                    native = images.embedded_image(document, number, bbox)
                    if native is None:  # several layers (e.g. shadow) or not decodable: render the whole region
                        rendered, render_meta = visual_crops.render_region(document[number - 1], bbox, render_dpi)
                        native = images.PageImage(rendered, render_meta["clip_bbox"], visual_crops.RENDERED,
                                                  render_meta | {"reason": "embedded scan is not a single decodable "
                                                                           "image layer; rendered from the page"})
                    try:
                        boxes = detector.detect(native.image)
                        error = None
                    except Exception as exc:  # noqa: BLE001
                        boxes, error = [], f"{type(exc).__name__}: {exc}"
                    selection = regions.select_regions(
                        number, boxes, image_px=native.size, image_bbox_pt=native.bbox_pt,
                        labels=config.layout.labels, margin_px=margin, attach_titles=config.layout.attach_titles,
                        captions_below=config.layout.captions_below, policy=config.layout.selection_policy,
                        id_prefix=f"E{e_index}R")
                    e_prefix = f"{prefix}_E{e_index}"
                    jobs += _region_jobs(run_dir, number, selection, native, prefix, max_pixels, embedded_index=e_index)
                    body, meta, masking_record = _body_image(run_dir, f"{e_prefix}_body", native, selection, max_pixels)
                    jobs.append(_job(run_dir, f"{e_prefix}_body", number, "page", "embedded_scan_body", body, meta,
                                     automatic=True, embedded_index=e_index, bbox_pt=native.bbox_pt,
                                     masking=masking_record))
                    embedded.append({"index": e_index, "bbox_pt": native.bbox_pt, "source_bbox_pt": list(bbox),
                                     "status": "layout detected", "image_method": native.method,
                                     "detection_error": error, "boxes": boxes, "selection": selection,
                                     "image_px": native.size})
                page_layout["embedded_scans"] = embedded
                handled = [e["source_bbox_pt"] for e in embedded if e.get("status") == "layout detected"]
                figures = elements["figures"]
                merged = {m["kept"] for m in elements.get("image_merges") or []}
                def _is_scan(figure: dict) -> bool:
                    return any(all(abs(float(a) - float(b)) <= 2.0 for a, b in zip(figure["bbox"], box))
                               for box in handled)

                # Composites only among ordinary figures: an embedded scan (or page background) is processed on
                # its own and must never swallow the figures drawn on it.
                inside = visual_crops.contained_figures([f for f in figures if not _is_scan(f)])

                def _root(fid: str) -> str:
                    seen = set()
                    while fid in inside and fid not in seen:
                        seen.add(fid)
                        fid = inside[fid]
                    return fid

                composites = {fid: _root(fid) for fid in inside}
                page_layout["composites"] = composites
                for figure in figures:
                    if _is_scan(figure):
                        continue  # embedded scan: processed region by region above
                    if figure["id"] in composites:
                        continue  # a layer of a larger figure: covered by that figure's complete crop
                    members = [fid for fid, root in composites.items() if root == figure["id"]]
                    others = [f for f in figures if f["id"] != figure["id"] and f["id"] not in members]
                    job_id = f"{prefix}_{figure['id']}"
                    data_png, meta, candidate = visual_crops.figure_crop(
                        document, number, figure, layers=layers, other_figures=others,
                        merged=figure["id"] in merged, contains_figures=members, render_dpi=render_dpi,
                        min_dpi=config.layout.native_min_dpi, tolerance_pt=config.layout.native_bbox_tolerance_pt,
                        min_std=config.layout.crop_quality_min_std)
                    if candidate is not None:
                        meta["native_candidate_file"] = f"crops/audit/{job_id}_native_candidate.png"
                        write_once(run_dir / meta["native_candidate_file"], candidate)
                    if max_pixels and meta["pixel_size"][0] * meta["pixel_size"][1] > max_pixels:
                        from PIL import Image  # noqa: PLC0415

                        with Image.open(io.BytesIO(data_png)) as source:  # explicit setting only; recorded
                            data_png, resized = images.encode_png(source.convert("RGB"), max_image_pixels=max_pixels)
                        meta.update(resized=resized["resized"], pixel_size=resized["pixel_size"],
                                    sha256=resized["sha256"], bytes=resized["bytes"])
                    meta["includes_figures"] = members
                    jobs.append(_job(run_dir, job_id, number, "crop", "digital_figure", data_png, meta,
                                     figure_id=figure["id"], automatic=True, bbox_pt=meta["crop_bbox"],
                                     includes_figures=members))
                if route["category_b_rejoin"]:
                    png, meta = _page_png(document, number, hybrid_qari.QARI_DPI)
                    jobs.append(_job(run_dir, f"{prefix}_rejoin", number, "page", "category_b_rejoin", png, meta,
                                     automatic=True))
            layout_pages.append(page_layout)
    if len(jobs) > config.processing.max_jobs:
        raise PipelineError(f"{len(jobs)} OCR jobs exceed the limit of {config.processing.max_jobs} "
                            "(PROCESSING_MAX_JOBS)")
    write_once(run_dir / "plan" / "layout.json", {"detector": detector_info, "pages": layout_pages,
                                                  "proven_layout": proven.layout.model_dump()})
    write_once(run_dir / "plan" / "jobs.json", {"jobs": jobs})
    write_once(run_dir / "plan" / "run.json", {
        "schema_version": 1, "kind": "automatic_extraction_run", "pipeline_version": PIPELINE_VERSION,
        "run_id": run_dir.name, "created_at": _now(), "lab_version": __version__,
        "document": {"display_name": display_name, "sha256": sha256(data), "bytes": len(data),
                     "page_count": analysis["page_count"]},
        "config": config.snapshot(), "proven_differences": auto_config.proven_differences(config),
        "image_resizing": ("none (native pixels)" if not max_pixels else
                           f"downscale above {max_pixels} pixels (explicit setting)"),
    })
    return run_dir


# ------------------------------------------------------------------ state


def load_plan(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    plan = {name: read_json(run_dir / "plan" / f"{name}.json") for name in ("run", "analysis", "routing", "layout", "jobs")}
    if any(value is None for value in plan.values()):
        raise PipelineError(f"not a complete automatic extraction run: {run_dir}")
    return plan


def all_jobs(run_dir: Path, plan: dict | None = None) -> list[dict]:
    """Planned jobs plus jobs added by "Process now"."""
    plan = plan or load_plan(run_dir)
    jobs = list(plan["jobs"]["jobs"])
    requested = {payload["job_id"]: payload for _p, payload in ordered(Path(run_dir) / "review", "process_now") if payload}
    for job in jobs:
        if job["job_id"] in requested:
            job["requested"] = requested[job["job_id"]]
    return jobs


def runnable(job: dict) -> bool:
    return bool(job.get("automatic") or job.get("requested"))


def attempts(run_dir: Path, job_id: str) -> list[dict]:
    folder = Path(run_dir) / "raw" / job_id
    found = []
    for intent_path in sorted(folder.glob("attempt_[0-9][0-9].intent.json")) if folder.is_dir() else []:
        intent = read_json(intent_path) or {}
        result = read_json(intent_path.with_name(intent_path.name.replace(".intent.", ".result.")))
        found.append({"intent": intent, "result": result})
    return found


def latest(run_dir: Path, job_id: str) -> dict | None:
    """The latest outcome of a job (an intent without a result = interrupted -> unconfirmed)."""
    tried = attempts(run_dir, job_id)
    completed = [a["result"] for a in tried if a["result"] and a["result"].get("state") == COMPLETED]
    if completed:
        return completed[-1]
    if not tried:
        return None
    last = tried[-1]
    if last["result"] is None:
        return {"state": UNCONFIRMED, "error": {"kind": "interrupted",
                                                "message": "the process stopped during the request"},
                "attempt": last["intent"].get("attempt"), "request_id": last["intent"].get("request_id")}
    return last["result"]


def status(run_dir: Path) -> dict:
    plan = load_plan(run_dir)
    jobs = all_jobs(run_dir, plan)
    rows, counts = [], {"completed": 0, "pending": 0, "unconfirmed": 0, "failed": 0, "review": 0}
    for job in jobs:
        outcome = latest(run_dir, job["job_id"])
        state = (outcome or {}).get("state")
        if not runnable(job):
            bucket = "review"
        elif state == COMPLETED:
            bucket = "completed"
        elif state == UNCONFIRMED:
            bucket = "unconfirmed"
        elif state == SERVICE_ERROR:
            bucket = "failed"
        else:
            bucket = "pending"
        counts[bucket] += 1
        rows.append({"job_id": job["job_id"], "page": job["page"], "kind": job["kind"], "purpose": job["purpose"],
                     "region_id": job.get("region_id"), "bucket": bucket, "state": state,
                     "attempts": len(attempts(run_dir, job["job_id"]))})
    return {"run_id": Path(run_dir).name, "finalized": is_finalized(run_dir), "service_records": len(service_records(run_dir)),
            "counts": counts, "jobs": rows, "routing": plan["routing"]["counts"]}


# ------------------------------------------------------------------ service (trust + audit)


def service_records(run_dir: Path) -> list[dict]:
    """Audit records: which Qari service, deployment mode and trust basis each execution used."""
    return [payload for _p, payload in ordered(Path(run_dir) / "service", "service") if payload]


def check_service(config: auto_config.AutoConfig) -> dict:
    """The configured endpoint under the administrator's trust rules (no network, no user interaction).
    Raises QariUnavailable / QariUntrusted when nothing may be sent."""
    if not config.qari.automatic_enabled:
        raise qari_client.QariUnavailable("automatic Qari processing is disabled (QARI_AUTOMATIC_ENABLED=false); "
                                          "use the offline export/import")
    return qari_client.endpoint_from_config(config)


def _record_service(run_dir: Path, config: auto_config.AutoConfig, endpoint: dict, health: dict | None,
                    jobs: int) -> Path:
    return write_sequenced(Path(run_dir) / "service", "service", {
        "created_at": _now(), "endpoint": endpoint["display"], "deployment_mode": endpoint["deployment_mode"],
        "trust_basis": endpoint["trust_basis"], "external": endpoint["external"],
        "token_present": bool(config.qari_token()), "automatic_enabled": config.qari.automatic_enabled,
        "pending_jobs": jobs, "health": health})


# ------------------------------------------------------------------ execute


def _send(run_dir: Path, job: dict, client: qari_client.QariClient) -> dict:
    run_dir = Path(run_dir)
    folder = run_dir / "raw" / job["job_id"]
    number = len(attempts(run_dir, job["job_id"])) + 1
    request_id = f"{run_dir.name}:{job['job_id']}:a{number:02d}"
    data = (run_dir / job["image_file"]).read_bytes()
    if sha256(data) != job["image_sha256"]:
        raise PipelineError(f"{job['job_id']}: the saved image changed (SHA-256 mismatch)")
    write_once(folder / f"attempt_{number:02d}.intent.json", {
        "attempt": number, "request_id": request_id, "job_id": job["job_id"], "kind": job["kind"],
        "image_sha256": job["image_sha256"], "endpoint": client.endpoint["display"], "created_at": _now()})
    base = {"attempt": number, "request_id": request_id, "endpoint": client.endpoint["display"],
            "image_sha256": job["image_sha256"], "source": "service"}
    try:
        response = client.ocr(request_id, job["kind"], data)
        result = base | {"state": COMPLETED, "response": response}
    except qari_client.QariUnavailable as exc:
        result = base | {"state": NOT_TRANSMITTED, "error": exc.record()}
    except qari_client.QariUnconfirmed as exc:
        result = base | {"state": UNCONFIRMED, "error": exc.record()}
    except qari_client.QariServiceError as exc:
        result = base | {"state": SERVICE_ERROR, "error": exc.record()}
    write_once(folder / f"attempt_{number:02d}.result.json", result | {"created_at": _now()})
    return result


def pending_jobs(run_dir: Path) -> list[dict]:
    """Runnable jobs never sent, or whose last attempt was refused before transmission."""
    out = []
    for job in all_jobs(run_dir):
        if not runnable(job):
            continue
        outcome = latest(run_dir, job["job_id"])
        if outcome is None or outcome.get("state") == NOT_TRANSMITTED:
            out.append(job)
    return out


def execute(run_dir: Path, config: auto_config.AutoConfig, client: qari_client.QariClient, *,
            on_progress: Progress | None = None, should_stop: Callable[[], bool] | None = None,
            health: dict | None = None) -> dict:
    """Send every pending job to the configured, trusted Qari service (resumable; no approval step).
    Stops, with nothing sent, when the endpoint is untrusted, disabled or unavailable."""
    if is_finalized(run_dir):
        raise PipelineError("the run is finalized")
    summary = {"sent": 0, COMPLETED: 0, SERVICE_ERROR: 0, UNCONFIRMED: 0, NOT_TRANSMITTED: 0, "stopped": None}
    try:
        endpoint = check_service(config)
    except qari_client.QariClientError as exc:
        summary["stopped"] = f"not sent: {exc}"
        return summary
    if client.endpoint["display"] != endpoint["display"]:
        raise PipelineError("the client does not point to the configured Qari service")
    todo = pending_jobs(run_dir)
    if todo:
        _record_service(run_dir, config, endpoint, health, len(todo))
    progress = on_progress or (lambda *_: None)
    workers = max(1, config.qari.concurrency)

    def handle(result: dict) -> bool:
        summary["sent"] += 1
        summary[result["state"]] += 1
        progress("ocr", summary["sent"], len(todo))
        if result["state"] == NOT_TRANSMITTED:
            summary["stopped"] = "Qari service unavailable: " + result["error"]["message"]
            return False
        if result["state"] == SERVICE_ERROR and config.processing.on_failure == "stop":
            summary["stopped"] = "service error and PROCESSING_ON_FAILURE=stop"
            return False
        return True

    if workers == 1:
        for job in todo:
            if should_stop and should_stop():
                summary["stopped"] = "stopped by the user"
                break
            if not handle(_send(run_dir, job, client)):
                break
        return summary
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(todo), workers):
            if should_stop and should_stop():
                summary["stopped"] = "stopped by the user"
                break
            batch = list(pool.map(lambda j: _send(run_dir, j, client), todo[start:start + workers]))
            if not all([handle(r) for r in batch]):
                break
    return summary


def retry_job(run_dir: Path, job_id: str, config: auto_config.AutoConfig, client: qari_client.QariClient) -> dict:
    """Explicit retry of an unconfirmed or failed job (a new attempt with a new request id)."""
    if is_finalized(run_dir):
        raise PipelineError("the run is finalized")
    endpoint = check_service(config)
    if client.endpoint["display"] != endpoint["display"]:
        raise PipelineError("the client does not point to the configured Qari service")
    job = next((j for j in all_jobs(run_dir) if j["job_id"] == job_id), None)
    if job is None:
        raise PipelineError(f"unknown job {job_id}")
    outcome = latest(run_dir, job_id)
    if outcome and outcome.get("state") == COMPLETED:
        raise PipelineError(f"{job_id} already has a completed result")
    _record_service(run_dir, config, endpoint, None, 1)
    return _send(run_dir, job, client) | {"explicit_retry": True}


def recover_job(run_dir: Path, job_id: str, client: qari_client.QariClient) -> dict | None:
    """Ask the service for the result of an unconfirmed attempt (no image is sent)."""
    outcome = latest(run_dir, job_id)
    if not outcome or outcome.get("state") != UNCONFIRMED:
        raise PipelineError(f"{job_id} has no unconfirmed attempt")
    response = client.recover(outcome["request_id"])
    if response is None:
        return None
    job = next(j for j in all_jobs(run_dir) if j["job_id"] == job_id)
    if (response.get("image") or {}).get("sha256") != job["image_sha256"]:
        raise PipelineError("the recovered result is for a different image")
    number = outcome["attempt"]
    folder = Path(run_dir) / "raw" / job_id
    record = {"request_id": outcome["request_id"], "state": COMPLETED, "response": response, "source": "recovered",
              "recovery_of_attempt": number, "endpoint": client.endpoint["display"],
              "image_sha256": job["image_sha256"], "created_at": _now()}
    if not (folder / f"attempt_{number:02d}.result.json").exists():  # interrupted: complete that attempt
        return json.loads(write_once(folder / f"attempt_{number:02d}.result.json",
                                     record | {"attempt": number}).read_text(encoding="utf-8"))
    # The unconfirmed record stays unchanged; the recovered result is a new attempt that sent no image.
    next_number = len(attempts(run_dir, job_id)) + 1
    write_once(folder / f"attempt_{next_number:02d}.intent.json",
               {"attempt": next_number, "request_id": outcome["request_id"], "job_id": job_id, "kind": job["kind"],
                "image_sha256": job["image_sha256"], "recovery_of_attempt": number, "transmitted_image": False,
                "endpoint": client.endpoint["display"], "created_at": _now()})
    write_once(folder / f"attempt_{next_number:02d}.result.json", record | {"attempt": next_number})
    return record | {"attempt": next_number}


# ------------------------------------------------------------------ digit second read (report only)


VERIFY_DIR = "verification"
VERIFY_PURPOSES = frozenset({"region", "digital_figure"})


def verification_records(run_dir: Path, job_id: str) -> dict:
    """{"attempts": [{"intent", "result"}], "skipped": [...]} of the digit second read of one crop."""
    folder = Path(run_dir) / VERIFY_DIR / job_id
    found = {"attempts": [], "skipped": []}
    if not folder.is_dir():
        return found
    for intent_path in sorted(folder.glob("attempt_[0-9][0-9].intent.json")):
        result = read_json(intent_path.with_name(intent_path.name.replace(".intent.", ".result.")))
        found["attempts"].append({"intent": read_json(intent_path) or {}, "result": result})
    found["skipped"] = [payload for _p, payload in ordered(folder, "skipped") if payload]
    return found


def _transmitted(records: dict) -> bool:
    """True once a verification request may have reached the service (never sent again automatically)."""
    for attempt in records["attempts"]:
        state = (attempt["result"] or {}).get("state", UNCONFIRMED)  # intent without result: may have been sent
        if state != NOT_TRANSMITTED:
            return True
    return False


def verification_candidates(run_dir: Path) -> list[tuple[dict, dict]]:
    """Completed figure crops whose primary OCR contains Arabic-Indic / Persian digits."""
    from lab.core.auto import digits  # noqa: PLC0415

    out = []
    for job in all_jobs(run_dir):
        if job.get("kind") != "crop" or job.get("purpose") not in VERIFY_PURPOSES or not runnable(job):
            continue
        outcome = latest(run_dir, job["job_id"])
        if not outcome or outcome.get("state") != COMPLETED:
            continue
        if digits.has_eastern_digits((outcome.get("response") or {}).get("final_text") or ""):
            out.append((job, outcome))
    return out


def verification_requests_sent(run_dir: Path) -> int:
    folder = Path(run_dir) / VERIFY_DIR
    if not folder.is_dir():
        return 0
    return sum(1 for job_dir in folder.iterdir() if job_dir.is_dir()
               for attempt in verification_records(run_dir, job_dir.name)["attempts"]
               if (attempt["result"] or {}).get("state", UNCONFIRMED) != NOT_TRANSMITTED)


def _skip(run_dir: Path, job_id: str, reason: str) -> None:
    write_sequenced(Path(run_dir) / VERIFY_DIR / job_id, "skipped", {"reason": reason, "created_at": _now()})


def verification_image(run_dir: Path, job: dict, upscale: float) -> tuple[bytes, dict]:
    """A SEPARATE copy of the original crop, Lanczos-upscaled (the primary crop is never changed)."""
    from PIL import Image  # noqa: PLC0415

    path = Path(run_dir) / VERIFY_DIR / job["job_id"] / f"input_x{upscale:g}.png"
    if path.is_file():  # reused by a later attempt after a not-transmitted one
        data = path.read_bytes()
    else:
        with Image.open(Path(run_dir) / job["image_file"]) as source:
            image = source.convert("RGB")
            size = (max(1, round(image.width * upscale)), max(1, round(image.height * upscale)))
            buffer = io.BytesIO()
            image.resize(size, Image.Resampling.LANCZOS).save(buffer, format="PNG")
        data = buffer.getvalue()
        write_once(path, data)
    with Image.open(io.BytesIO(data)) as check:
        size = [check.width, check.height]
    return data, {"file": path.relative_to(run_dir).as_posix(), "sha256": sha256(data), "pixel_size": size,
                  "upscale": upscale, "resampling": "LANCZOS", "source_crop": job["image_file"],
                  "source_crop_sha256": job["image_sha256"]}


def verify_digits(run_dir: Path, config: auto_config.AutoConfig, client: qari_client.QariClient | None, *,
                  on_progress: Progress | None = None) -> dict:
    """Second read of figure crops with Arabic-Indic / Persian digits: at most ONE extra Qari request per
    crop (same model, prompt and settings; crop timeout), a per-run limit, never resent automatically.
    Report only: the primary OCR is never changed and a failure here never fails the extraction."""
    settings = config.verification
    summary = {"eligible": 0, "sent": 0, COMPLETED: 0, "skipped": 0, "already_verified": 0, "stopped": None}
    if is_finalized(run_dir):
        return summary | {"stopped": "the run is finalized"}
    candidates = verification_candidates(run_dir)
    summary["eligible"] = len(candidates)
    progress = on_progress or (lambda *_: None)
    stop_reason = None
    if not settings.digits_enabled:
        stop_reason = "digit verification is disabled (VERIFY_DIGITS_ENABLED=false)"
    elif client is None:
        stop_reason = "no Qari client available"
    else:
        try:
            endpoint = check_service(config)
            if client.endpoint["display"] != endpoint["display"]:
                stop_reason = "the client does not point to the configured Qari service"
        except qari_client.QariClientError as exc:
            stop_reason = f"Qari service not usable: {exc}"
    used = verification_requests_sent(run_dir)
    for index, (job, outcome) in enumerate(candidates, start=1):
        records = verification_records(run_dir, job["job_id"])
        if _transmitted(records):
            summary["already_verified"] += 1
            continue
        if stop_reason is None and used >= settings.max_requests:
            stop_reason = f"per-run verification limit reached ({settings.max_requests} requests)"
        if stop_reason is not None:
            _skip(run_dir, job["job_id"], stop_reason)
            summary["skipped"] += 1
            continue
        data, meta = verification_image(run_dir, job, settings.upscale)
        number = len(records["attempts"]) + 1
        request_id = f"{Path(run_dir).name}:{job['job_id']}:verify{number:02d}"
        folder = Path(run_dir) / VERIFY_DIR / job["job_id"]
        write_once(folder / f"attempt_{number:02d}.intent.json", {
            "attempt": number, "request_id": request_id, "job_id": job["job_id"], "kind": "crop",
            "purpose": "digit_verification", "input": meta, "primary_request_id": outcome.get("request_id"),
            "endpoint": client.endpoint["display"], "created_at": _now()})
        base = {"attempt": number, "request_id": request_id, "endpoint": client.endpoint["display"],
                "image_sha256": meta["sha256"]}
        try:
            result = base | {"state": COMPLETED, "response": client.ocr(request_id, "crop", data)}
        except qari_client.QariUnavailable as exc:
            result = base | {"state": NOT_TRANSMITTED, "error": exc.record()}
            stop_reason = f"Qari service unavailable: {exc}"
        except qari_client.QariUnconfirmed as exc:
            result = base | {"state": UNCONFIRMED, "error": exc.record()}
        except qari_client.QariServiceError as exc:
            result = base | {"state": SERVICE_ERROR, "error": exc.record()}
        write_once(folder / f"attempt_{number:02d}.result.json", result | {"created_at": _now()})
        if result["state"] != NOT_TRANSMITTED:
            used += 1
            summary["sent"] += 1
        if result["state"] == COMPLETED:
            summary[COMPLETED] += 1
        else:
            summary["skipped"] += result["state"] == NOT_TRANSMITTED
        progress("digit verification", index, len(candidates))
    summary["stopped"] = stop_reason
    summary["requests_sent_in_run"] = used
    return summary


# ------------------------------------------------------------------ review queue


def review_queue(run_dir: Path) -> list[dict]:
    """Regions not processed automatically or flagged for review, with image, boxes and reasons."""
    plan = load_plan(run_dir)
    jobs = {j.get("region_id"): j for j in all_jobs(run_dir, plan) if j.get("region_id")}
    queue = []
    for page in plan["layout"]["pages"]:
        selections = ([page["selection"]] if page.get("selection") else []) + [
            e["selection"] for e in page.get("embedded_scans") or [] if e.get("selection")]
        for selection in selections:
            for region in selection["regions"]:
                if region["decision"] not in {regions.REVIEW} | regions.FLAGGED:
                    continue
                job = jobs.get(region["region_id"]) or {}
                queue.append({"page": page["page"], "region_id": region["region_id"], "label": region["label"],
                              "decision": region["decision"], "status": region["status"],
                              "reasons": region["reasons"], "bbox_pt": region["box_pt_before_margin"],
                              "pixel_box": region["pixel_box"], "image_file": job.get("image_file"),
                              "job_id": job.get("job_id"), "requested": bool(job.get("requested")),
                              "proposed_expansion_pt": region.get("proposed_expansion_pt")})
    return queue


def process_now(run_dir: Path, region_id: str, *, reason: str = "") -> Path:
    """Add a review-queue region to the jobs (it is sent by the next execute)."""
    if is_finalized(run_dir):
        raise PipelineError("the run is finalized")
    job = next((j for j in all_jobs(run_dir) if j.get("region_id") == region_id), None)
    if job is None:
        raise PipelineError(f"unknown region {region_id}")
    if runnable(job):
        raise PipelineError(f"{region_id} is already processed or queued")
    return write_sequenced(Path(run_dir) / "review", "process_now", {
        "job_id": job["job_id"], "region_id": region_id, "reason": reason.strip(),
        "created_at": _now()})


# ------------------------------------------------------------------ offline export / import


OFFLINE_README = """Offline Qari batch for an Automatic Extraction run.

1. Copy this folder to the GPU host.
2. Run:  python qari_offline_batch.py <this folder>      (runpod/qari_offline_batch.py; same recipe as the service)
3. Bring back results.jsonl and import it in the app (Automatic Extraction -> Offline import).
Each line of results.jsonl: {"job_id", "image_sha256", "response": <the service response schema>}.
Images are checked by SHA-256 on import; a job that already has a completed result is not overwritten.
"""


def export_offline(run_dir: Path) -> bytes:
    """ZIP of the pending images (no text layer, no Ground Truth) for an offline Qari run."""
    jobs = pending_jobs(run_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        lines = []
        for job in jobs:
            archive.write(Path(run_dir) / job["image_file"], f"images/{job['job_id']}.png")
            lines.append(json.dumps({"job_id": job["job_id"], "kind": job["kind"], "image": f"images/{job['job_id']}.png",
                                     "image_sha256": job["image_sha256"]}, ensure_ascii=False))
        archive.writestr("jobs.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        archive.writestr("README.txt", OFFLINE_README)
        archive.writestr("run.json", json.dumps({"run_id": Path(run_dir).name, "jobs": len(jobs),
                                                 "created_at": _now()}, indent=2))
    return buffer.getvalue()


def import_offline(run_dir: Path, results_jsonl: bytes) -> dict:
    """Import offline results: hash-checked; completed jobs are never overwritten."""
    if is_finalized(run_dir):
        raise PipelineError("the run is finalized")
    jobs = {j["job_id"]: j for j in all_jobs(run_dir)}
    imported, skipped = [], []
    for number, line in enumerate(results_jsonl.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            skipped.append({"line": number, "reason": "not JSON"})
            continue
        job = jobs.get(item.get("job_id"))
        response = item.get("response") or {}
        if job is None:
            skipped.append({"line": number, "reason": f"unknown job {item.get('job_id')!r}"})
        elif item.get("image_sha256") != job["image_sha256"] or (response.get("image") or {}).get("sha256") != job["image_sha256"]:
            skipped.append({"line": number, "job_id": job["job_id"], "reason": "image SHA-256 does not match"})
        elif not isinstance(response.get("final_text"), str) or not response.get("attempts"):
            skipped.append({"line": number, "job_id": job["job_id"], "reason": "response lacks final_text/attempts"})
        elif (latest(run_dir, job["job_id"]) or {}).get("state") == COMPLETED:
            skipped.append({"line": number, "job_id": job["job_id"], "reason": "already completed (not overwritten)"})
        else:
            folder = Path(run_dir) / "raw" / job["job_id"]
            attempt = len(attempts(run_dir, job["job_id"])) + 1
            write_once(folder / f"attempt_{attempt:02d}.intent.json", {
                "attempt": attempt, "job_id": job["job_id"], "kind": job["kind"], "image_sha256": job["image_sha256"],
                "source": "offline_import", "transmitted_image": False, "created_at": _now()})
            write_once(folder / f"attempt_{attempt:02d}.result.json", {
                "attempt": attempt, "state": COMPLETED, "response": response, "source": "offline_import",
                "image_sha256": job["image_sha256"], "created_at": _now()})
            imported.append(job["job_id"])
    record = {"imported": imported, "skipped": skipped, "created_at": _now(),
              "sha256": sha256(results_jsonl)}
    write_sequenced(Path(run_dir) / "imports", "import", record)
    return record
