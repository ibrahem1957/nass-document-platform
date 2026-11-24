"""Phase 2: image, chart and screenshot extraction for Hybrid PDF Extraction (Benchmark Mode).

Builds on the approved layout revision (Phase 1, unchanged). Everything is append-only
under `<run>/enrichment/` and is allowed for finalized runs too (nothing existing changes):

    enrichment/set_NNNNNN/                 one region selection (write-once)
        regions.json                       regions, bboxes, detected type, crop SHA-256
        crops/<region_id>.png              the exact bytes every provider receives (250 DPI)
        exports/export_NNNNNN.json         manifests of exported crop ZIPs (for import checks)
        runs/<provider>_NNN/               hosted API (journals via APIEvaluationRunner) or Qari endpoint
        imports/import_NNNNNN/             manually imported texts ({region_id}.txt, unchanged)
        selections/selection_NNNNNN.json   preferred output per region + reason (append-only)
        references/ref_NNNNNN/             benchmark Ground Truth JSON (original bytes) + record
        matches/match_NNNNNN.json          confirmed region <-> reference matches (append-only)
    enrichment/documents/rev_NNNNNN/       unified page_NNN.md + full_document.md (+ revision_NNNNNN.json)

Provider behaviour is the provider's own: Qari keeps "Free OCR." and its parameters, Baidu
uses the document-parsing API without instructions, Oxen keeps its registry prompt. Quality
checks run after inference. Ground Truth is used only for evaluation, never for selection.
No request is sent during export, import, comparison, evaluation or recomputation.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from lab.core.e2.api import documents, full_pdf, host_policy, page_artifacts
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import check_authorization
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused, image_transmission_preview
from lab.core.e2.runner import new_run_id
from lab.core.hybrid import crops as crop_options
from lab.core.hybrid import numbers as number_checks
from lab.core.hybrid import qari, textcheck
from lab.local_pdf_extraction import diagnostics

ENRICH_VERSION = "1"
CROP_DPI = 250  # one crop per region, identical for every provider (Qari's documented DPI)
APPROVAL_HOURS = 24
APPROVAL_STATEMENT = (
    "I approve sending exactly the listed image-region crops (each identified by its SHA-256) of this exact "
    "document to the listed external provider destinations, one request per region, within the usage limits "
    "shown. The full PDF, other page regions, library text and Ground Truth are not sent, and nothing is "
    "resent automatically."
)
IMPORT_STATEMENT = (
    "I imported these texts myself. They are recorded as a manual import with the model name I declare; this "
    "does not prove which model produced them."
)
MATCH_STATEMENT = "I reviewed these region-to-reference matches; unconfirmed or rejected references are not used."
DOCUMENT_STATEMENT = (
    "I reviewed the unified page documents. Save them as a new enrichment revision (derived from saved outputs "
    "only; nothing is sent and no existing file changes)."
)
SELECTION_STATEMENT = "I chose these outputs myself; the choice is recorded with my reasons."
BILLING_NOTES = {
    "baidu-aistudio-paddleocr-vl-1.6": "per page of a submitted document; whether one crop counts as one page is "
                                       "not documented (UNKNOWN)",
    "oxen-qwen3-vl-4b-instruct": "compute-time billing; the preflight bound per request is used, actual charge "
                                 "from the response when reported (rate UNVERIFIED)",
    "qari": "your own server (no provider billing)",
}
# Optional PaddleOCR-VL extraction modes for Benchmark Mode. Only fields documented for the hosted
# AI Studio jobs API are sent (optionalPayload, a JSON string in the multipart form). The registry's
# own optional_payload (vendor defaults) is never changed; a mode only overlays these fields for
# the runs that explicitly choose it, and the exact payload is recorded in the plan, the journal
# (provider identity) and every saved result.
PADDLE_PROVIDER_ID = "baidu-aistudio-paddleocr-vl-1.6"
PADDLE_DOC_SOURCES = [
    "https://ai.baidu.com/ai-doc/AISTUDIO/Mml7n69e7 (async jobs API; sample optionalPayload)",
    "https://ai.baidu.com/ai-doc/AISTUDIO/Vmkz2nz1p (PaddleOCR-VL-1.5/1.6 API; useChartRecognition)",
]
PADDLE_MODES = {
    "standard": {"label": "Standard document parsing (registry defaults; no optional field)", "overlay": {}},
    "chart_recognition": {"label": "Chart recognition (useChartRecognition = true)",
                          "overlay": {"useChartRecognition": True}},
}
PADDLE_UNSUPPORTED = [
    "OCR for image blocks: no such field is documented for the hosted AI Studio API (use_ocr_for_image_block is "
    "a parameter of the local PaddleOCR pipeline); it is NOT sent and cannot be enabled on this endpoint",
]


def mode_spec(spec: ProviderSpec, mode: str) -> ProviderSpec:
    """The provider spec for one extraction mode (a copy; the registry entry is unchanged)."""
    if mode == "standard" or spec.id != PADDLE_PROVIDER_ID:
        if mode not in PADDLE_MODES:
            raise EnrichError(f"Unknown extraction mode {mode!r}")
        return spec
    if mode not in PADDLE_MODES:
        raise EnrichError(f"Unknown extraction mode {mode!r}")
    return spec.model_copy(update={"optional_payload": dict(spec.optional_payload) | PADDLE_MODES[mode]["overlay"]})


UNREADABLE = re.compile(r"(unreadable|illegible|not readable|\[\?\]|\?\?\?|غير مقروء|غير واضح)", re.IGNORECASE)


class EnrichError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_once(path: Path, payload: dict | bytes | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    elif isinstance(payload, str):
        data = payload.encode("utf-8", errors="surrogatepass")
    else:
        data = payload
    with path.open("xb") as handle:
        handle.write(data)
    return path


def _new_dir(base: Path, prefix: str, width: int = 6) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    sequence = len([p for p in base.glob(f"{prefix}_*") if p.is_dir()]) + 1
    while True:
        path = base / f"{prefix}_{sequence:0{width}d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            sequence += 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def enrichment_dir(run_dir: Path) -> Path:
    return Path(run_dir) / "enrichment"


# ------------------------------------------------------------------ regions


def detected_type(figure: dict) -> str:
    if figure.get("origin") == "vector":
        return "vector chart" + (" (suspected)" if figure.get("confidence") == "suspected" else "")
    if figure.get("possible_child_regions"):
        return "multi-figure image (possible sub-figures)"
    x0, y0, x1, y1 = figure["bbox"]
    if x1 - x0 > 0 and (y1 - y0) / (x1 - x0) < 0.35:
        return "wide image (KPI card or banner?)"
    return "raster image (chart, screenshot, table or photo; not classified)"


def candidate_regions(run_dir: Path) -> list[dict]:
    """Image/chart regions of the latest approved layout revision (read-only)."""
    from lab.core.hybrid import run as hybrid  # noqa: PLC0415 - avoid an import cycle

    latest = hybrid.latest_layout(run_dir)
    if latest is None:
        raise EnrichError("No approved layout revision: save one in Recompute layout first")
    folder = Path(latest["folder"])
    regions = []
    for path in sorted(folder.glob("page_*.layout.json")):
        saved = _read(path) or {}
        record, elements = saved.get("record") or {}, saved.get("elements") or {}
        page = int(elements.get("page") or record.get("page"))
        notes = record.get("needs_review") or []
        for figure in elements.get("figures") or []:
            uncertainty = [note for note in notes if figure["id"] in note]
            if figure.get("confidence") == "suspected":
                uncertainty.append("suspected vector region")
            if figure.get("review"):
                uncertainty.append(figure["review"])
            base = {"page": page, "figure_id": figure["id"], "origin": figure.get("origin"), "bbox": figure["bbox"],
                    "area_share": figure.get("area_share"), "detected_type": detected_type(figure),
                    "layout_revision": latest["revision"],
                    "layout_crop": str(folder / "figures" / figure["crop_file"]) if figure.get("crop_file") else None,
                    "default_selected": figure.get("origin") == "raster" and not uncertainty}
            regions.append(base | {"region_id": f"p{page}_{figure['id']}", "parent": None,
                                   "uncertain": bool(uncertainty), "uncertainty": uncertainty})
            for index, child in enumerate(figure.get("possible_child_regions") or [], start=1):
                regions.append(base | {"region_id": f"p{page}_{figure['id']}_c{index}", "bbox": child,
                                       "parent": f"p{page}_{figure['id']}", "detected_type": "possible sub-figure",
                                       "default_selected": False, "uncertain": True,
                                       "uncertainty": ["possible sub-figure from blank gutters; boundary unverified"],
                                       "layout_crop": None})
    return regions


def create_region_set(run_dir: Path, region_ids: list[str], *, pdf_path: Path | None, by: str,
                      resolution="auto", fmt: str = "auto") -> Path:
    """Make the selected crops once (local only) into a write-once set, with the chosen options:
    resolution "auto" | "native" | 150 | 200 | 250 and format "auto" | "png" | "jpeg" (see crops.py)."""
    if resolution not in crop_options.RESOLUTIONS or fmt not in crop_options.FORMATS:
        raise EnrichError(f"Unsupported crop options: resolution={resolution!r}, format={fmt!r}")
    import pymupdf

    from lab.core.hybrid import run as hybrid  # noqa: PLC0415

    if not by.strip():
        raise EnrichError("Enter your name")
    candidates = {item["region_id"]: item for item in candidate_regions(run_dir)}
    unknown = [rid for rid in region_ids if rid not in candidates]
    if unknown or not region_ids:
        raise EnrichError(f"Select at least one known region (unknown: {unknown})")
    metadata, _plan, _library = hybrid.load(run_dir)
    pdf, pdf_source = hybrid.layout_pdf(run_dir, metadata, pdf_path)
    folder = _new_dir(enrichment_dir(run_dir), "set")
    regions = []
    with pymupdf.open(pdf) as document:
        for index, rid in enumerate(region_ids, start=1):
            region = dict(candidates[rid])
            page = document[region["page"] - 1]
            try:
                data, options = crop_options.make_crop(document, page, region["bbox"], resolution=resolution, fmt=fmt)
            except ValueError as exc:
                raise EnrichError(f"Region {rid}: {exc}") from exc
            name = f"crops/{rid}.{crop_options.extension(options)}"
            _write_once(folder / name, data)
            page_area = (page.rect.width * page.rect.height) or 1.0
            if region.get("area_share") is None:
                x0, y0, x1, y1 = region["bbox"]
                region["area_share"] = round(max(0.0, x1 - x0) * max(0.0, y1 - y0) / page_area, 4)
            options["crop_sha256"] = _sha(data)
            region.update(index=index, crop_file=name, crop_sha256=options["crop_sha256"], crop_bytes=len(data),
                          crop_px=options["pixel_size"], crop_dpi=options["effective_dpi"], crop_options=options,
                          page_size_pt=[round(page.rect.width, 2), round(page.rect.height, 2)],
                          size_based_timeout_s=crop_options.size_based_timeout(region))
            regions.append(region)
    _write_once(folder / "regions.json", {
        "schema_version": 1, "kind": "hybrid_region_set", "enrich_version": ENRICH_VERSION, "set_id": folder.name,
        "run_id": Path(run_dir).name, "document": metadata["document"], "pdf_source": pdf_source,
        "crop_dpi": resolution if resolution in ("auto", "native") else int(resolution),
        "crop_request": {"resolution": resolution, "format": fmt, "jpeg_quality": crop_options.JPEG_QUALITY},
        "created_at": _now().isoformat(), "created_by": by.strip(), "regions": regions,
    })
    return folder


def region_sets(run_dir: Path) -> list[Path]:
    base = enrichment_dir(run_dir)
    return sorted(path.parent for path in base.glob("set_*/regions.json")) if base.is_dir() else []


def load_set(set_dir: Path) -> dict:
    data = _read(Path(set_dir) / "regions.json")
    if not data:
        raise EnrichError(f"Not a region set: {set_dir}")
    for region in data["regions"]:
        crop = Path(set_dir) / region["crop_file"]
        if not crop.is_file() or _sha(crop.read_bytes()) != region["crop_sha256"]:
            raise EnrichError(f"Crop {region['region_id']} is missing or changed; the set is not usable")
    return data


# ------------------------------------------------------------ export/import


def export_crops_zip(set_dir: Path, *, provider: str, model: str, by: str) -> bytes:
    """PNG crops + manifest.json. The manifest is also recorded in the set (for import checks)."""
    data = load_set(set_dir)
    regions = [{"region_id": r["region_id"], "page": r["page"], "bbox": r["bbox"], "crop_sha256": r["crop_sha256"],
                "crop_file": Path(r["crop_file"]).name, "crop_px": r["crop_px"], "detected_type": r["detected_type"],
                "crop_options": crop_options.region_options(r), "layout_box": r.get("layout_box")}
               for r in data["regions"]]
    manifest = {"schema_version": 1, "kind": "hybrid_crop_export", "set_id": data["set_id"], "run_id": data["run_id"],
                "document_sha256": data["document"]["sha256"], "crop_dpi": data["crop_dpi"],
                "intended_provider": provider, "intended_model": model, "exported_by": by.strip(),
                "exported_at": _now().isoformat(), "regions": regions,
                "instructions": "Write one UTF-8 file per region named {region_id}.txt; include this manifest.json "
                                "unchanged in the results ZIP."}
    manifest["manifest_sha256"] = _sha(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    write_sequenced(Path(set_dir) / "exports", "export", manifest)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for region in data["regions"]:
            archive.write(Path(set_dir) / region["crop_file"], f"crops/{Path(region['crop_file']).name}")
    return buffer.getvalue()


def _exports(set_dir: Path) -> list[dict]:
    return [payload for _path, payload in ordered(Path(set_dir) / "exports", "export") if payload]


def check_import(set_dir: Path, zip_bytes: bytes) -> dict:
    """Validate a results ZIP against an exported manifest. Never writes anything."""
    data = load_set(set_dir)
    report: dict[str, Any] = {"ok": False, "errors": [], "missing": [], "duplicates": [], "unknown": [],
                              "texts": {}, "manifest": None}
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        report["errors"].append("not a ZIP file")
        return report
    with archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if any(".." in Path(name).parts or name.startswith(("/", "\\")) for name in names):
            report["errors"].append("unsafe path in the ZIP")
            return report
        if sum(info.file_size for info in archive.infolist()) > 50_000_000:
            report["errors"].append("ZIP content larger than 50 MB")
            return report
        manifests = [name for name in names if Path(name).name == "manifest.json"]
        if len(manifests) != 1:
            report["errors"].append("the ZIP must contain exactly one manifest.json (the exported one)")
            return report
        try:
            manifest = json.loads(archive.read(manifests[0]).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            report["errors"].append("manifest.json is not valid UTF-8 JSON")
            return report
        report["manifest"] = manifest
        claimed = manifest.get("manifest_sha256")
        body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        if claimed != _sha(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")):
            report["errors"].append("manifest.json was modified after export")
        if claimed not in {item.get("manifest_sha256") for item in _exports(set_dir)}:
            report["errors"].append("manifest.json does not match any export recorded for this region set")
        if manifest.get("set_id") != data["set_id"] or manifest.get("document_sha256") != data["document"]["sha256"]:
            report["errors"].append("the manifest belongs to a different region set or document")
        set_regions = {r["region_id"]: r for r in data["regions"]}
        for item in manifest.get("regions") or []:
            region = set_regions.get(item.get("region_id"))
            if region is None or region["crop_sha256"] != item.get("crop_sha256"):
                report["errors"].append(f"manifest region {item.get('region_id')} does not match the set's crop")
        expected = {item.get("region_id") for item in manifest.get("regions") or []}
        seen: Counter = Counter()
        for name in names:
            if not name.lower().endswith(".txt"):
                continue
            stem = Path(name).stem
            seen[stem] += 1
            if stem not in expected:
                report["unknown"].append(name)
                continue
            raw = archive.read(name)
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                report["errors"].append(f"{name} is not UTF-8")
                continue
            report["texts"][stem] = raw
        report["duplicates"] = sorted(stem for stem, count in seen.items() if count > 1)
        report["missing"] = sorted(expected - set(report["texts"]))
        if report["unknown"]:
            report["errors"].append(f"unknown region files: {report['unknown']}")
        if report["duplicates"]:
            report["errors"].append(f"duplicate region files: {report['duplicates']}")
    report["ok"] = not report["errors"]
    return report


def import_results(set_dir: Path, zip_bytes: bytes, *, declared_model: str, reviewer: str, statement: str) -> Path:
    if statement != IMPORT_STATEMENT or not reviewer.strip() or not declared_model.strip():
        raise EnrichError("Enter the declared model, your name and tick the import statement")
    report = check_import(set_dir, zip_bytes)
    if not report["ok"]:
        raise EnrichError("Import refused: " + "; ".join(report["errors"]))
    folder = _new_dir(Path(set_dir) / "imports", "import")
    for rid, raw in report["texts"].items():
        _write_once(folder / "texts" / f"{rid}.txt", raw)  # unchanged bytes
    _write_once(folder / "manifest.json", report["manifest"])
    _write_once(folder / "record.json", {
        "schema_version": 1, "kind": "manual_import", "provenance": "manual_import",
        "declared_model": declared_model.strip(), "reviewer": reviewer.strip(), "statement": statement,
        "imported_at": _now().isoformat(), "zip_sha256": _sha(zip_bytes),
        "manifest_sha256": report["manifest"].get("manifest_sha256"),
        "regions_imported": sorted(report["texts"]), "regions_missing": report["missing"],
        "verification": "file identity and crop hashes verified against the export manifest; the producing model "
                        "is declared by the reviewer and NOT independently verified",
    })
    return folder


# ------------------------------------------------------------- provider runs


def capacity(spec: ProviderSpec, root: Path) -> dict:
    state = check_authorization(spec, pages=1, root=root)
    policy = state.policy or {}
    return {"allowed": state.allowed, "reasons": state.reasons, "remaining_requests": state.remaining_pages,
            "used_total": state.pages_used_total, "used_today": state.pages_used_today,
            "used_enablement": state.pages_used_enablement, "enablement_started_at": state.enablement_started_at,
            "limit_total": state.limit_enablement, "limit_day": state.limit_day,
            "limit_cumulative": state.limit_cumulative, "summary": state.describe(),
            "limit_source": policy.get("source"), "provider_quota": policy.get("provider_quota"),
            "estimated_usd_per_request": spec.pricing.estimated_usd_per_page,
            "price_verified": spec.pricing.price_verified, "billing": BILLING_NOTES.get(spec.id, "not documented")}


def _selected(data: dict, region_ids: list[str]) -> list[dict]:
    chosen = [region for region in data["regions"] if region["region_id"] in set(region_ids)]
    if not chosen or len(chosen) != len(set(region_ids)):
        raise EnrichError("Select regions of this set")
    return chosen


def _approval(run_dir: Path, *, by: str, statement: str, regions: list[dict], destination: str) -> None:
    if statement != APPROVAL_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the external transmission approval")
    now = _now()
    _write_once(Path(run_dir) / "approval.json", {
        "approved": True, "approved_by": by.strip(), "statement": statement, "destination": destination,
        "regions": {region["region_id"]: region["crop_sha256"] for region in regions},
        "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=APPROVAL_HOURS)).isoformat()})


def approval_valid(run_dir: Path) -> tuple[bool, str]:
    approval = _read(Path(run_dir) / "approval.json")
    if not approval or approval.get("statement") != APPROVAL_STATEMENT:
        return False, "no valid approval"
    try:
        if datetime.fromisoformat(approval["expires_at"]) <= _now():
            return False, "the approval has expired"
    except (KeyError, ValueError):
        return False, "the approval has no valid expiry"
    return True, ""


def region_timeouts(spec: ProviderSpec | None, regions: list[dict], override: float | None,
                    fallback_s: float | None = None) -> dict[str, float]:
    """Per-region request timeouts: the user's override (every provider), else the provider default
    (size-based 60 s / 180 s for Baidu AI Studio; registry timeout for the others)."""
    try:
        if override:
            value = crop_options.checked_timeout(spec, override)
            return {r["region_id"]: value for r in regions}
        return {r["region_id"]: crop_options.checked_timeout(spec, crop_options.default_timeout(r, spec, fallback_s))
                for r in regions}
    except ValueError as exc:
        raise EnrichError(str(exc)) from exc


def create_api_run(set_dir: Path, spec: ProviderSpec, region_ids: list[str], *, by: str, statement: str,
                   root: Path, mode: str = "standard", timeout_override: float | None = None) -> Path:
    if statement != APPROVAL_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the external transmission approval")
    data = load_set(set_dir)
    regions = _selected(data, region_ids)
    room = capacity(spec, root)
    if not room["allowed"] or room["remaining_requests"] < len(regions):
        raise EnrichError(f"{spec.provider}: {len(regions)} request(s) needed, {room['remaining_requests']} left "
                          "within the enablement limits; " + "; ".join(room["reasons"]))
    if not documents.authorization_state(data["document"]["sha256"], root=root)["allowed"]:
        raise EnrichError("The document is not authorized (exact SHA-256) for external transmission")
    effective = mode_spec(spec, mode)
    timeouts = region_timeouts(spec, regions, timeout_override)
    run_dir = _new_dir(Path(set_dir) / "runs", spec.id, width=3)
    _write_once(run_dir / "plan.json", {
        "schema_version": 1, "kind": "enrichment_api_run", "provider_id": spec.id, "identity": effective.identity(),
        "extraction_mode": mode, "extraction_mode_label": PADDLE_MODES[mode]["label"],
        "optional_payload": dict(effective.optional_payload),
        "unsupported_options": PADDLE_UNSUPPORTED if spec.id == PADDLE_PROVIDER_ID else [],
        "option_sources": PADDLE_DOC_SOURCES if spec.id == PADDLE_PROVIDER_ID else [],
        "endpoint": spec.endpoint, "set_id": data["set_id"], "document_sha256": data["document"]["sha256"],
        "regions": [{"index": r["index"], "region_id": r["region_id"], "page": r["page"],
                     "crop_sha256": r["crop_sha256"], "crop_file": r["crop_file"],
                     "crop_options": crop_options.region_options(r), "timeout_s": timeouts[r["region_id"]]}
                    for r in regions],
        "timeout_policy": ({"override_s": float(timeout_override)} if timeout_override else
                           {"small_region_s": crop_options.TIMEOUT_SMALL_S, "large_region_s": crop_options.TIMEOUT_LARGE_S,
                            "small_region_below_page_share": crop_options.SMALL_REGION_SHARE}
                           if spec.id in crop_options.SIZE_BASED_TIMEOUT_PROVIDERS else
                           {"registry_timeout_s": float(spec.limits.request_timeout_s)}),
        "requests": len(regions), "prompt_id": spec.prompt_id, "generation": dict(spec.generation),
        "billing": BILLING_NOTES.get(spec.id), "created_at": _now().isoformat()})
    _approval(run_dir, by=by, statement=statement, regions=regions, destination=spec.endpoint)
    return run_dir


def _run_journals(run_dir: Path) -> set[str]:
    return {str((_read(path) or {}).get("journal_id")) for path in (Path(run_dir) / "attempts").glob("*_attempt_[0-9][0-9].json")}


def _approve_known_host(run_dir: Path, spec: ProviderSpec, journal, root: Path) -> None:
    """Baidu: this run's approval covers HTTPS results on the registry's listed result hosts
    (or paddleocr-store-<n>.bj.bcebos.com) for journals of THIS run only. Others stay manual."""
    if not host_policy.supported(spec) or journal.journal_id not in _run_journals(run_dir):
        return
    url = journal.latest().get("result_url") if isinstance(journal.latest().get("result_url"), dict) else {}
    try:
        host = job_store.validate_result_hostname(str(url.get("host") or ""))
    except job_store.JobJournalError:
        return
    if (host != url.get("host") or url.get("scheme") != "https" or url.get("port") not in (None, 443)
            or not str(url.get("path") or "").startswith(host_policy.RESULT_PATH_PREFIX)):
        return
    listed = {item.lower() for item in spec.auto_result_hosts}
    if host not in listed and not re.fullmatch(host_policy.BAIDU_RESULT_HOST_PATTERN, host):
        return
    if host in job_store.approved_result_hosts(spec, root, journal_id=journal.journal_id):
        return
    approval = _read(Path(run_dir) / "approval.json") or {}
    path = job_store.approve_result_host(
        spec, host=host, approved_by=f"enrichment run approval ({approval.get('approved_by')})",
        statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal, root=root,
        via={"mode": "enrichment_run_approval", "run": str(run_dir)})
    write_sequenced(Path(run_dir) / "host_approvals", "host_approval",
                    {"journal_id": journal.journal_id, "host": host, "approval_record": str(path),
                     "created_at": _now().isoformat()})


def _continue_baidu(run_dir, runner, spec, journal_id, credential, root) -> None:
    journal = job_store.JobJournal(job_store.journal_dir(spec.id, journal_id, root))
    if not journal.events() or journal.state != job_store.AWAITING_RESULT_HOST_APPROVAL:
        return
    _approve_known_host(run_dir, spec, journal, root)
    host = (journal.latest().get("result_url") or {}).get("host")
    if host and host in runner.result_hosts(spec, journal_id):
        runner.resume(spec, journal_id, credential=credential, ground_truth=None)


Progress = Callable[[str, int, int, "dict | None"], None]
_SENDABLE = {None, "REFUSED_BEFORE_SENDING", "NOT_TRANSMITTED"}


def execute_api_run(run_dir: Path, runner: APIEvaluationRunner, spec: ProviderSpec, *, workspace_id: str, document,
                    pdf_path: Path, credential: str, root: Path, on_progress: Progress | None = None) -> dict:
    """Send each approved crop once, sequentially. Never resends; stops after 2 consecutive problems."""
    run_dir = Path(run_dir)
    plan = _read(run_dir / "plan.json") or {}
    if plan.get("provider_id") != spec.id:
        raise EnrichError("This run belongs to another provider")
    if not credential.strip():
        raise EnrichError(f"Enter the {spec.credential_label}")
    spec = _planned_spec(plan, spec)
    set_dir = run_dir.parent.parent
    stopped, problems = None, 0
    with full_pdf._exclusive(run_dir):
        total = len(plan["regions"])
        for position, item in enumerate(plan["regions"], start=1):
            key = {"page": item["index"], "image_sha256": item["crop_sha256"]}
            record = full_pdf.derive_page(run_dir, spec, key, root=root)
            if record["job_state"] not in _SENDABLE:
                continue
            ok, reason = approval_valid(run_dir)
            if not ok:
                stopped = reason
                break
            crop = set_dir / item["crop_file"]
            if _sha(crop.read_bytes()) != item["crop_sha256"]:
                stopped = f"crop {item['region_id']} changed; nothing was sent"
                break
            if on_progress:
                on_progress("sending", position, total, record)
            attempt = len(full_pdf.attempts(run_dir, item["index"])) + 1
            journal_id = new_run_id()
            intent = run_dir / "attempts" / f"page_{item['index']:03d}_attempt_{attempt:02d}.json"
            _write_once(intent, {"page": item["index"], "attempt": attempt, "journal_id": journal_id,
                                 "region_id": item["region_id"], "source_page": item["page"],
                                 "image_sha256": item["crop_sha256"], "created_at": _now().isoformat()})
            region_spec = crop_options.with_timeout(spec, item.get("timeout_s") or spec.limits.request_timeout_s)
            options = item.get("crop_options") or {}
            timing = {"timeout_s": region_spec.limits.request_timeout_s, "crop_options": options,
                      "send_started_at": _now().isoformat()}
            try:
                preview = image_transmission_preview(region_spec, document, pdf_path, crop, page=item["page"],
                                                     region_id=item["region_id"], dpi=options.get("effective_dpi"),
                                                     authorization_root=root)
                outcome = runner.run(region_spec, workspace_id, document, pdf_path, item["page"], credential=credential,
                                     confirmed_sha256=item["crop_sha256"], ground_truth=None, run_id=journal_id,
                                     artifact_context={"enrichment_run": str(run_dir), "region_id": item["region_id"],
                                                       "crop_options": options,
                                                       "timeout_s": region_spec.limits.request_timeout_s},
                                     preview=preview)
                timing["submit_returned_at"] = _now().isoformat()
                if outcome.state == job_store.AWAITING_RESULT_HOST_APPROVAL:
                    try:
                        _continue_baidu(run_dir, runner, region_spec, journal_id, credential, root)
                    except APIRunRefused:
                        pass
            except APIRunRefused as exc:
                _write_once(intent.with_name(f"{intent.stem}_refused.json"),
                            {"reason": str(exc), "created_at": _now().isoformat()})
                stopped = f"{item['region_id']} refused before sending: {exc}"
                break
            except Exception as exc:  # noqa: BLE001 - the journal decides what was sent
                _write_once(intent.with_name(f"{intent.stem}_local_error.json"),
                            {"type": type(exc).__name__, "message": str(exc)[:2000], "created_at": _now().isoformat()})
            timing["finished_at"] = _now().isoformat()
            journal = job_store.JobJournal(job_store.journal_dir(spec.id, journal_id, root))
            timing["journal_events"] = [{"state": e.get("state"), "at": e.get("at")} for e in journal.events()]
            _write_once(intent.with_name(f"{intent.stem}_timing.json"), timing)
            record = full_pdf.derive_page(run_dir, spec, key, root=root)
            if on_progress:
                on_progress("done", position, total, record)
            if record["job_state"] in {job_store.FAILED, job_store.SUBMISSION_UNCONFIRMED, job_store.RESUMABLE_ERROR,
                                       "NOT_TRANSMITTED"}:
                problems += 1
                if problems >= full_pdf.STOP_AFTER_CONSECUTIVE_PROBLEMS:
                    stopped = "stopped after 2 consecutive failed or unconfirmed regions; nothing is resent"
                    break
            else:
                problems = 0
    return {"stopped": stopped}


def _planned_spec(plan: dict, spec: ProviderSpec) -> ProviderSpec:
    """The spec with the run's recorded extraction mode; refuses if the payload would differ."""
    effective = mode_spec(spec, plan.get("extraction_mode") or "standard")
    if "optional_payload" in plan and dict(effective.optional_payload) != plan["optional_payload"]:
        raise EnrichError("The provider's request options changed since this run was approved; create a new run")
    return effective


def resume_api_run(run_dir: Path, runner: APIEvaluationRunner, spec: ProviderSpec, *, credential: str,
                   root: Path) -> list[str]:
    """Baidu: continue paused jobs from their Job IDs (no resubmission)."""
    messages = []
    plan = _read(Path(run_dir) / "plan.json") or {}
    spec = _planned_spec(plan, spec)
    for item in plan.get("regions") or []:
        record = full_pdf.derive_page(run_dir, spec, {"page": item["index"], "image_sha256": item["crop_sha256"]},
                                      root=root)
        if record["job_state"] not in {job_store.AWAITING_RESULT_HOST_APPROVAL, job_store.RESUMABLE_ERROR,
                                       job_store.SUBMITTED} or spec.protocol == "openai_chat_completions_v1":
            continue
        try:
            if record["job_state"] == job_store.AWAITING_RESULT_HOST_APPROVAL:
                _continue_baidu(run_dir, runner, spec, record["journal_id"], credential, root)
            else:
                runner.resume(spec, record["journal_id"], credential=credential, ground_truth=None)
        except APIRunRefused as exc:
            messages.append(f"{item['region_id']}: {exc}")
    return messages


def create_qari_run(set_dir: Path, region_ids: list[str], *, endpoint_url: str, model: str, by: str,
                    statement: str, root: Path, timeout_override: float | None = None) -> Path:
    if statement != APPROVAL_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the external transmission approval")
    data = load_set(set_dir)
    regions = _selected(data, region_ids)
    endpoint = qari.parse_endpoint(endpoint_url)
    timeouts = region_timeouts(None, regions, timeout_override, fallback_s=qari.QARI_TIMEOUT_S)
    if not endpoint["loopback"] and not documents.authorization_state(data["document"]["sha256"], root=root)["allowed"]:
        raise EnrichError("A remote Qari server requires the document's SHA-256 authorization")
    base = Path(set_dir) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    sequence = len(list(base.glob("qari_*"))) + 1
    while (base / f"qari_{sequence:03d}").exists():
        sequence += 1
    run_dir = base / f"qari_{sequence:03d}"
    qari.create_plan(run_dir, endpoint=endpoint, model=model, approved_by=by,
                     document_sha256=data["document"]["sha256"],
                     pages=[{"page": r["index"], "png": (Path(set_dir) / r["crop_file"]).read_bytes(),
                             "width": r["crop_px"][0], "height": r["crop_px"][1],
                             "dpi": crop_options.region_options(r).get("effective_dpi")} for r in regions])
    _write_once(run_dir / "regions.json", {"regions": [{"index": r["index"], "region_id": r["region_id"],
                                                        "page": r["page"], "crop_sha256": r["crop_sha256"],
                                                        "crop_options": crop_options.region_options(r),
                                                        "timeout_s": timeouts[r["region_id"]]}
                                                       for r in regions]})
    _approval(run_dir, by=by, statement=statement, regions=regions, destination=endpoint["chat_url"])
    return run_dir


def execute_qari_run(run_dir: Path, *, token: str | None, document_sha256: str, root: Path,
                     on_progress: Progress | None = None) -> dict:
    run_dir = Path(run_dir)
    regions = (_read(run_dir / "regions.json") or {}).get("regions") or []
    stopped, problems = None, 0
    for position, item in enumerate(regions, start=1):
        record = qari.page_record(run_dir, item["index"])
        if record["job_state"] not in {None, qari.NOT_TRANSMITTED}:
            continue
        ok, reason = approval_valid(run_dir)
        if not ok:
            stopped = reason
            break
        if on_progress:
            on_progress("sending", position, len(regions), record)
        authorized = documents.authorization_state(document_sha256, root=root)["allowed"]
        try:
            record = qari.send_page(run_dir, item["index"], token=token, document_authorized=authorized,
                                    timeout=float(item.get("timeout_s") or qari.QARI_TIMEOUT_S))
        except qari.QariError as exc:
            stopped = f"{item['region_id']} refused before sending: {exc}"
            break
        if on_progress:
            on_progress("done", position, len(regions), record)
        if record["job_state"] in {qari.UNCONFIRMED, qari.FAILED, qari.NOT_TRANSMITTED}:
            problems += 1
            if problems >= full_pdf.STOP_AFTER_CONSECUTIVE_PROBLEMS:
                stopped = "stopped after 2 consecutive failed or unconfirmed regions; nothing is resent"
                break
        else:
            problems = 0
    return {"stopped": stopped}


# ------------------------------------------------------------------ outputs


_IMG_SRC = re.compile(r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


def visible_text(text: str) -> str:
    """Text a reader would see: HTML tags and their attributes (image file names, sizes,
    styles) removed, entities decoded. Table cell text is kept. The raw text is never changed."""
    without_media = re.sub(r"<(img|br|hr)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    without_tags = re.sub(r"</?[A-Za-z][^>]*>", " ", without_media)
    return "\n".join(" ".join(line.split()) for line in html.unescape(without_tags).splitlines() if line.strip())


def content_kind(text: str) -> dict:
    """What an output actually contains (e.g. Paddle returning only an <img> reference)."""
    images = _IMG_SRC.findall(text)
    visible = visible_text(text)
    if not text.strip():
        kind = "empty"
    elif images and not visible.strip():
        kind = "html_image_reference_only"
    else:
        kind = "text"
    return {"kind": kind, "image_references": images, "visible_chars": len(visible),
            "note": ("the provider returned only an HTML image reference (no extracted text, labels or numbers); "
                     "image file names, sizes and HTML attributes are not extracted values") if kind ==
            "html_image_reference_only" else None}


def assess(text: str, *, finish_reason: str | None = None, truncated: bool | None = None,
           region_type: str = "") -> dict:
    """Post-inference quality checks (provider-independent). Never corrects anything.
    Numbers and repetition are checked on the visible text only (never on HTML attributes)."""
    shown = visible_text(text)
    kind = content_kind(text)
    quality = textcheck.vlm_quality(shown, finish_reason=finish_reason, truncated=truncated)
    numbers = [{"original": n["original"], "normalized": n["normalized"]} for n in textcheck.extract_numbers(shown)]
    reasons = list(quality["reasons"])
    if kind["kind"] == "html_image_reference_only":
        reasons = [kind["note"]]
    unreadable = len(UNREADABLE.findall(text))
    if unreadable:
        reasons.append(f"{unreadable} unreadable/uncertain marker(s) in the output")
    if "chart" in region_type and numbers and not any(ch.isalpha() for ch in shown):
        reasons.append("numbers without any label text (chart labels may be missing)")
    return {"needs_review": bool(reasons), "reasons": reasons, "numbers": numbers, "quality": quality,
            "content": kind, "visible_text": shown,
            "status": "Needs Review" if reasons else "No issues detected (not verified)",
            "note": "Image-derived numbers are listed for review; they are not compared with the text layer."}


def collect_outputs(set_dir: Path, spec_for: Callable[[str], ProviderSpec | None], root: Path) -> dict[str, list[dict]]:
    """Every saved output per region (API, endpoint, manual import), raw text unchanged."""
    data = load_set(set_dir)
    by_index = {r["index"]: r for r in data["regions"]}
    set_options = {r["region_id"]: crop_options.region_options(r) for r in data["regions"]}
    layout_boxes = {r["region_id"]: r.get("layout_box") for r in data["regions"]}
    found: dict[str, list[dict]] = {r["region_id"]: [] for r in data["regions"]}
    for run_dir in sorted((Path(set_dir) / "runs").glob("*")) if (Path(set_dir) / "runs").is_dir() else []:
        if (run_dir / "regions.json").is_file():  # Qari endpoint run
            plan = qari.load_plan(run_dir)
            for item in (_read(run_dir / "regions.json") or {}).get("regions") or []:
                record = qari.page_record(run_dir, item["index"])
                found.setdefault(item["region_id"], []).append({
                    "output_key": run_dir.name, "provider": "Qari-OCR endpoint", "model": plan["model"],
                    "provenance": "endpoint", "text": record["text"], "status": record["status"],
                    "job_state": record["job_state"], "finish_reason": record.get("finish_reason"),
                    "truncated": record.get("truncated_at_token_limit"), "tokens": record.get("tokens"),
                    "latency_s": record.get("processing_seconds"), "error": record.get("error"),
                    "crop_options": item.get("crop_options") or set_options.get(item["region_id"]),
                    "timeout_s": item.get("timeout_s", qari.QARI_TIMEOUT_S)})
            continue
        plan = _read(run_dir / "plan.json") or {}
        spec = spec_for(plan.get("provider_id") or "")
        if spec is not None:
            try:
                spec = _planned_spec(plan, spec)
            except EnrichError:
                pass  # outputs are still read; derive_page only needs the provider id
        for item in plan.get("regions") or []:
            if spec is None:
                record = {"text": "", "status": "Unknown", "job_state": "provider not configured"}
            else:
                record = full_pdf.derive_page(run_dir, spec, {"page": item["index"], "image_sha256": item["crop_sha256"]},
                                              root=root)
            found.setdefault(item["region_id"], []).append({
                "output_key": run_dir.name, "provider": (plan.get("identity") or {}).get("provider"),
                "model": (plan.get("identity") or {}).get("api_model_id"), "provenance": "hosted_api",
                "text": record.get("text") or "", "status": record.get("status"), "job_state": record.get("job_state"),
                "finish_reason": record.get("finish_reason"), "truncated": record.get("truncated_at_token_limit"),
                "tokens": record.get("tokens"), "latency_s": record.get("processing_seconds"),
                "cost": record.get("cost"), "error": record.get("error"), "journal_id": record.get("journal_id"),
                "job_id": record.get("job_id"), "run_dir": str(run_dir), "plan_index": item["index"],
                "extraction_mode": plan.get("extraction_mode") or "standard",
                "crop_options": item.get("crop_options") or set_options.get(item["region_id"]),
                "timeout_s": item.get("timeout_s", (spec.limits.request_timeout_s if spec else None)),
                "optional_payload": plan.get("optional_payload", (plan.get("identity") or {}).get("optional_payload")),
                "structured_blocks": structured_blocks(record.get("store_run_directory"), item["page"]),
                "artifacts": artifact_locations(spec, record.get("journal_id"), root) if spec else {}})
    for import_dir in sorted((Path(set_dir) / "imports").glob("import_*")) if (Path(set_dir) / "imports").is_dir() else []:
        record = _read(import_dir / "record.json") or {}
        for path in sorted((import_dir / "texts").glob("*.txt")):
            text = path.read_bytes().decode("utf-8-sig")
            found.setdefault(path.stem, []).append({
                "output_key": import_dir.name, "provider": "manual import", "model": record.get("declared_model"),
                "provenance": "manual_import", "text": text, "status": "Imported", "job_state": "IMPORTED",
                "finish_reason": None, "truncated": None, "tokens": None, "latency_s": None,
                "reviewer": record.get("reviewer"), "imported_at": record.get("imported_at"),
                "verification": record.get("verification"), "crop_options": set_options.get(path.stem)})
    types = {r["region_id"]: r["detected_type"] for r in by_index.values()}
    for rid, outputs in found.items():
        for output in outputs:
            output["assessment"] = assess(output["text"], finish_reason=output.get("finish_reason"),
                                          truncated=output.get("truncated"), region_type=types.get(rid, ""))
            output["usable"] = (bool(output["assessment"]["visible_text"].strip())
                                and output["job_state"] in {"COMPLETED", "IMPORTED"})
            output["layout_box"] = layout_boxes.get(rid)
            # Numbers with unit/format (validation only, stored next to the unchanged raw output).
            output["unit_validation"] = number_checks.validate(output["assessment"]["visible_text"])
            if output["unit_validation"]["needs_review"]:
                output["assessment"]["reasons"].append(
                    f"{len(output['unit_validation']['flags'])} number format flag(s) (see unit validation)")
                output["assessment"]["needs_review"] = True
                output["assessment"]["status"] = "Needs Review"
        conflicts = number_checks.unit_conflicts({o["output_key"]: o["unit_validation"] for o in outputs if o["usable"]})
        for output in outputs:
            output["unit_conflicts"] = conflicts
            if conflicts and output["usable"]:
                output["assessment"]["reasons"].append(
                    "unit differs between providers for " + ", ".join(c["value"] for c in conflicts[:8]))
                output["assessment"]["needs_review"] = True
                output["assessment"]["status"] = "Needs Review"
            extra = chart_review(output, types.get(rid, ""))
            if extra:
                output["assessment"]["reasons"] += [r for r in extra if r not in output["assessment"]["reasons"]]
                output["assessment"]["needs_review"] = True
                output["assessment"]["status"] = "Needs Review"
    return found


def structured_blocks(store_run_dir: str | None, page: int) -> list[dict]:
    """Layout blocks the provider returned (label + content as returned + visible text), read
    from the saved run record. Image paths, coordinates and HTML attributes are not values."""
    if not store_run_dir:
        return []
    run = _read(Path(store_run_dir) / "run.json") or {}
    response = (((run.get("environment") or {}).get("e2") or {}).get("page_raw") or {}).get(str(page), {}).get("response") or {}
    regions = (response.get("official_normalized") or {}).get("layout_regions") or []
    blocks = []
    for block in regions if isinstance(regions, list) else []:
        if not isinstance(block, dict):
            continue
        content = str(block.get("block_content") or "")
        blocks.append({"label": block.get("block_label"), "content": content, "visible_text": visible_text(content),
                       "bbox_in_crop_px": block.get("block_bbox")})
    return blocks


def chart_review(output: dict, region_type: str) -> list[str]:
    """Review reasons for chart regions: missing or empty chart values (never filled in)."""
    reasons = []
    charts = [b for b in output.get("structured_blocks") or [] if b.get("label") == "chart"]
    if output.get("extraction_mode") == "chart_recognition":
        if charts and not any(b["visible_text"].strip() for b in charts):
            reasons.append("chart recognition was requested but the chart block has no content (no chart values)")
        if not charts and ("chart" in region_type or "image" in region_type) and not output["assessment"]["numbers"]:
            reasons.append("chart recognition was requested but no chart block or value was returned")
    elif charts and not any(b["visible_text"].strip() for b in charts):
        reasons.append("the provider detected a chart but returned no chart content (chart recognition was off)")
    return reasons


def artifact_locations(spec: ProviderSpec, journal_id: str | None, root: Path) -> dict:
    """Where the saved records of one hosted request live (read-only lookup)."""
    if not journal_id:
        return {}
    journal_dir = job_store.journal_dir(spec.id, journal_id, root)
    events = job_store.JobJournal(journal_dir).events() if journal_dir.is_dir() else []
    first = events[0] if events else {}
    found = {"journal_dir": str(journal_dir) if events else None,
             "store_run_dir": first.get("run_directory"), "page_snapshot_dir": None}
    if first.get("document_id") and first.get("document_sha256") and first.get("page"):
        base = page_artifacts.page_directory(root, spec.id, str(first["document_id"]), str(first["document_sha256"]),
                                             int(first["page"])) / journal_id
        complete = [path for path in sorted(base.glob("snapshot-*")) if (path / "complete.json").is_file()]
        found["page_snapshot_dir"] = str(complete[-1]) if complete else None
    return found


def _find_key(value, key: str):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_key(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def outputs_export(set_dir: Path, spec_for: Callable[[str], ProviderSpec | None], root: Path) -> dict[str, bytes]:
    """ZIP entries (built in memory; nothing is written) with every saved provider output of a
    set: raw response files as saved, the parsed text, visible text, status and provenance."""
    data = load_set(set_dir)
    outputs = collect_outputs(set_dir, spec_for, root)
    entries: dict[str, bytes] = {}
    index = []
    for rid, items in outputs.items():
        for output in items:
            prefix = f"phase2_outputs/{data['set_id']}/{output['output_key']}/{rid}/"
            status = {key: output.get(key) for key in (
                "output_key", "provider", "model", "provenance", "status", "job_state", "job_id", "journal_id",
                "finish_reason", "truncated", "tokens", "latency_s", "cost", "error", "usable", "reviewer",
                "imported_at", "verification", "artifacts", "extraction_mode", "optional_payload",
                "structured_blocks", "crop_options", "timeout_s", "layout_box")}
            status["content"] = output["assessment"]["content"]
            status["quality"] = {"needs_review": output["assessment"]["needs_review"],
                                 "reasons": output["assessment"]["reasons"]}
            status["numbers_in_visible_text"] = output["assessment"]["numbers"]
            entries[prefix + "status.json"] = json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8")
            entries[prefix + "parsed_text.txt"] = output["text"].encode("utf-8", errors="surrogatepass")
            entries[prefix + "visible_text.txt"] = output["assessment"]["visible_text"].encode("utf-8")
            entries[prefix + "unit_validation.json"] = json.dumps(
                {"validation": output.get("unit_validation"), "unit_conflicts_in_region": output.get("unit_conflicts")},
                ensure_ascii=False, indent=2).encode("utf-8")
            artifacts = output.get("artifacts") or {}
            snapshot = Path(artifacts["page_snapshot_dir"]) if artifacts.get("page_snapshot_dir") else None
            if snapshot and snapshot.is_dir():
                for path in sorted(snapshot.iterdir()):
                    if path.is_file() and path.suffix in {".json", ".txt"}:
                        entries[prefix + f"provider_record/{path.name}"] = path.read_bytes()
                structured = _read(snapshot / "structured_output.json") or _read(snapshot / "raw_response.json")
                settings = _find_key(structured, "model_settings") if structured else None
                if settings is not None:
                    # As reported by the provider (e.g. use_chart_recognition); recorded, never changed.
                    status["provider_reported_model_settings"] = settings
                    entries[prefix + "status.json"] = json.dumps(status, ensure_ascii=False, indent=2).encode("utf-8")
            journal = Path(artifacts["journal_dir"]) if artifacts.get("journal_dir") else None
            if journal and journal.is_dir():
                for path in sorted(journal.glob("e*.json")):
                    entries[prefix + f"journal/{path.name}"] = path.read_bytes()
            index.append({"region_id": rid, "output_key": output["output_key"], "folder": prefix,
                          "content_kind": status["content"]["kind"], "status": output.get("status")})
    entries[f"phase2_outputs/{data['set_id']}/INDEX.json"] = json.dumps(
        {"set_id": data["set_id"], "generated_at": _now().isoformat(), "outputs": index,
         "note": "Copies of saved records for review; the originals stay where 'artifacts' points. Nothing was sent."},
        ensure_ascii=False, indent=2).encode("utf-8")
    return entries


def outputs_zip(set_dir: Path, spec_for: Callable[[str], ProviderSpec | None], root: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in outputs_export(set_dir, spec_for, root).items():
            archive.writestr(name, data)
    return buffer.getvalue()


def output_label(output: dict) -> str:
    if output["provenance"] == "manual_import":
        return f"{output['model']} (manual import, declared; {output['output_key']})"
    mode = output.get("extraction_mode")
    suffix = f" [{PADDLE_MODES[mode]['label'].split(' (')[0].lower()}]" if mode and mode != "standard" else ""
    return f"{output['provider']} · {output['model']}{suffix} ({output['output_key']})"


# --------------------------------------------------------------- selections


def save_selection(set_dir: Path, choices: dict[str, dict], *, by: str, statement: str) -> Path:
    """choices: {region_id: {"output_key": key | None, "reason": text}} (None = keep the placeholder)."""
    if statement != SELECTION_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the selection statement")
    data = load_set(set_dir)
    known = {r["region_id"] for r in data["regions"]}
    if set(choices) - known:
        raise EnrichError("Unknown regions in the selection")
    return write_sequenced(Path(set_dir) / "selections", "selection", {
        "schema_version": 1, "choices": choices, "by": by.strip(), "statement": statement,
        "created_at": _now().isoformat(), "note": "Chosen by the user; Ground Truth is never used to choose."})


def latest_selection(set_dir: Path) -> dict | None:
    records = [payload | {"path": str(path)} for path, payload in ordered(Path(set_dir) / "selections", "selection")
               if payload]
    return records[-1] if records else None


# ------------------------------------------------------------- references


def _strings(value: Any, skip: frozenset = frozenset({"id", "chart_type", "category_visual_order"})) -> list[str]:
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in skip:
                out += _strings(item, skip)
    elif isinstance(value, list):
        for item in value:
            out += _strings(item, skip)
    elif isinstance(value, str) and value.strip():
        out.append(value)
    return out


_LABEL_KEYS = frozenset({"title", "label", "unit", "legend", "categories", "name", "category"})


def _labels(value: Any) -> list[str]:
    out = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _LABEL_KEYS:
                out += _strings(item)
            elif isinstance(item, (dict, list)) and key not in {"tick_labels", "data"}:
                out += _labels(item)
    elif isinstance(value, list):
        for item in value:
            out += _labels(item)
    return list(dict.fromkeys(label for label in out if not re.fullmatch(r"[\d\s.,%٫٬٠-٩-]+", label)))


def reference_elements(document: dict) -> list[dict]:
    """Image-related reference elements from the AgenticDoc benchmark JSON (actual fields only):
    pages[].visual_elements.{charts,diagrams,images}[] and pages[].tables[]."""
    found = []
    for page in document.get("pages") or []:
        number = page.get("page_number")
        if not isinstance(number, int):
            continue
        visual = page.get("visual_elements") or {}
        for kind, items in (visual.items() if isinstance(visual, dict) else []):
            for index, item in enumerate(items if isinstance(items, list) else []):
                if not isinstance(item, dict):
                    continue
                element_id = str(item.get("id") or f"{kind}{index + 1}")
                text = "\n".join(_strings(item))
                found.append({"ref_id": f"p{number}:{kind}:{element_id}", "page": number, "kind": kind,
                              "element_id": element_id, "title": item.get("title"), "text": text,
                              "labels": _labels(item), "table_cells": [], "original": item})
        for table in page.get("tables") or []:
            if not isinstance(table, dict):
                continue
            cells = [str(cell.get("text")) for rows in (table.get("header_rows_as_rendered") or [],
                                                         table.get("body_rows_as_rendered") or [])
                     for row in rows for cell in (row or []) if isinstance(cell, dict) and str(cell.get("text") or "").strip()]
            found.append({"ref_id": f"p{number}:table:{table.get('id')}", "page": number, "kind": "table",
                          "element_id": str(table.get("id")), "title": table.get("caption"), "text": "\n".join(cells),
                          "labels": [], "table_cells": cells, "original": table})
    return found


def import_reference(set_dir: Path, data: bytes, *, filename: str, by: str) -> Path:
    try:
        document = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise EnrichError(f"Not UTF-8 JSON: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("pages"), list):
        raise EnrichError("Expected the AgenticDoc benchmark format with a 'pages' list")
    elements = reference_elements(document)
    if not elements:
        raise EnrichError("No pages[].visual_elements or pages[].tables entries were found")
    folder = _new_dir(Path(set_dir) / "references", "ref")
    _write_once(folder / "original.json", data)  # unchanged bytes
    _write_once(folder / "record.json", {"filename": filename, "sha256": _sha(data), "imported_by": by.strip(),
                                         "imported_at": _now().isoformat(), "elements": len(elements),
                                         "use": "evaluation only; never used to select outputs"})
    return folder


def load_reference(ref_dir: Path) -> list[dict]:
    return reference_elements(json.loads((Path(ref_dir) / "original.json").read_bytes().decode("utf-8-sig")))


def references(set_dir: Path) -> list[Path]:
    base = Path(set_dir) / "references"
    return sorted(base.glob("ref_*")) if base.is_dir() else []


def propose_matches(set_dir: Path, ref_dir: Path) -> list[dict]:
    """Candidates by page and type. The reference JSON has no coordinates, so a match is only
    *proposed* when it is the single candidate on a page with a single region; never applied."""
    data = load_set(set_dir)
    elements = load_reference(ref_dir)
    per_page = Counter(r["page"] for r in data["regions"] if r["parent"] is None)
    proposals = []
    for region in data["regions"]:
        compatible = [e for e in elements if e["page"] == region["page"]
                      and (e["kind"] != "table" or region["origin"] == "raster")]
        unique = len(compatible) == 1 and per_page[region["page"]] == 1 and region["parent"] is None
        proposals.append({"region_id": region["region_id"], "page": region["page"],
                          "detected_type": region["detected_type"],
                          "candidates": [{"ref_id": e["ref_id"], "kind": e["kind"], "title": e["title"]} for e in compatible],
                          "proposed": compatible[0]["ref_id"] if unique else None,
                          "basis": ("only candidate on this page (no coordinates in the reference)" if unique
                                    else "ambiguous or none: choose manually")})
    return proposals


def save_matches(set_dir: Path, ref_dir: Path, confirmations: dict[str, str | None], *, by: str, statement: str) -> Path:
    if statement != MATCH_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the match confirmation")
    valid = {e["ref_id"] for e in load_reference(ref_dir)}
    if any(value and value not in valid for value in confirmations.values()):
        raise EnrichError("A confirmed reference is not in this Ground Truth file")
    return write_sequenced(Path(set_dir) / "matches", "match", {
        "reference": Path(ref_dir).name, "confirmations": confirmations, "by": by.strip(),
        "statement": statement, "created_at": _now().isoformat()})


def confirmed_matches(set_dir: Path, ref_dir: Path) -> dict[str, str]:
    records = [p for _path, p in ordered(Path(set_dir) / "matches", "match") if p and p.get("reference") == Path(ref_dir).name]
    return {rid: ref for rid, ref in (records[-1]["confirmations"] if records else {}).items() if ref}


# --------------------------------------------------------------- evaluation


def _norm(text: str) -> str:
    return diagnostics.arabic_normalize(text).lower()


def region_metrics(reference: dict, text: str) -> dict:
    rates = diagnostics.error_rates(reference["text"], text)
    ref_tokens, hyp_tokens = Counter(_norm(reference["text"]).split()), Counter(_norm(text).split())
    matched_tokens = sum((ref_tokens & hyp_tokens).values())
    ref_numbers = Counter(n["normalized"] for n in textcheck.extract_numbers(reference["text"]))
    hyp_numbers = Counter(n["normalized"] for n in textcheck.extract_numbers(text))
    normalized = _norm(text)
    labels_found = [label for label in reference["labels"] if _norm(label) and _norm(label) in normalized]
    cells_found = [cell for cell in reference["table_cells"] if _norm(cell) and _norm(cell) in normalized]
    return {
        "cer": rates, "token_recall": matched_tokens / sum(ref_tokens.values()) if ref_tokens else None,
        "token_precision": matched_tokens / sum(hyp_tokens.values()) if hyp_tokens else None,
        "numbers": {"reference": sum(ref_numbers.values()), "matched": sum((ref_numbers & hyp_numbers).values()),
                    "missing": dict(ref_numbers - hyp_numbers), "not_in_reference": dict(hyp_numbers - ref_numbers)},
        "unit_mismatches": number_checks.unit_mismatches(reference["text"], text),
        "labels": {"reference": len(reference["labels"]), "found": len(labels_found),
                   "missing": [label for label in reference["labels"] if label not in labels_found]},
        "table_cells": {"reference": len(reference["table_cells"]), "found": len(cells_found)},
    }


def evaluate(set_dir: Path, outputs: dict[str, list[dict]], ref_dir: Path) -> dict:
    """Image-region metrics against CONFIRMED image references only (evaluation, never selection)."""
    elements = {e["ref_id"]: e for e in load_reference(ref_dir)}
    matches = confirmed_matches(set_dir, ref_dir)
    regions: dict[str, dict] = {}
    by_provider: dict[str, list[tuple[str, dict]]] = {}
    for rid, ref_id in matches.items():
        reference = elements.get(ref_id)
        if reference is None:
            continue
        regions[rid] = {"reference": ref_id, "outputs": {}}
        for output in outputs.get(rid, []):
            if not output["usable"]:
                continue
            metrics = region_metrics(reference, output["text"])
            regions[rid]["outputs"][output["output_key"]] = {"label": output_label(output)} | metrics
            by_provider.setdefault(output_label(output).split(" (")[0], []).append((rid, metrics))
    providers = list(by_provider)
    comparable = sorted(rid for rid in regions
                        if all(any(r == rid for r, _m in by_provider[p]) for p in providers)) if providers else []

    def aggregate(items: list[tuple[str, dict]], only: set | None = None) -> dict:
        chosen = [m for rid, m in items if only is None or rid in only]
        chars = sum(m["cer"]["arabic_normalized"]["reference_chars"] for m in chosen)
        edits = sum((m["cer"]["arabic_normalized"]["cer"] or 0) * m["cer"]["arabic_normalized"]["reference_chars"]
                    for m in chosen)
        numbers = sum(m["numbers"]["reference"] for m in chosen)
        return {"regions": len(chosen), "cer_normalized": edits / chars if chars else None,
                "numbers_matched": sum(m["numbers"]["matched"] for m in chosen), "numbers_reference": numbers,
                "numbers_not_in_reference": sum(sum(m["numbers"]["not_in_reference"].values()) for m in chosen),
                "unit_mismatches": sum(m["unit_mismatches"]["count"] for m in chosen),
                "labels_found": sum(m["labels"]["found"] for m in chosen),
                "labels_reference": sum(m["labels"]["reference"] for m in chosen),
                "table_cells_found": sum(m["table_cells"]["found"] for m in chosen),
                "table_cells_reference": sum(m["table_cells"]["reference"] for m in chosen)}

    return {"reference": Path(ref_dir).name, "regions": regions, "comparable_regions": comparable,
            "providers": {p: {"all_evaluated": aggregate(items), "comparable_only": aggregate(items, set(comparable))}
                          for p, items in by_provider.items()},
            "note": "Image-specific references only (confirmed matches). Aggregates cover only evaluated regions; "
                    "no overall accuracy is reported when references are incomplete."}


# --------------------------------------------------- unified page documents


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self.cell: list[str] | None = None
        self.spans = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in {"td", "th"}:
            self.cell = []
            if any(name in {"colspan", "rowspan"} and str(value) not in {"1", "None"} for name, value in attrs):
                self.spans = True

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(" ".join("".join(self.cell).split()))
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def html_tables_to_markdown(text: str) -> tuple[str, list[dict]]:
    """Simple HTML tables (no merged cells) -> Markdown tables; anything else is left unchanged."""
    notes = []

    def convert(match: re.Match) -> str:
        parser = _TableParser()
        parser.feed(match.group(0))
        rows = [row for row in parser.rows if row]
        width = max((len(row) for row in rows), default=0)
        if parser.spans or not rows or any(len(row) != width for row in rows):
            notes.append({"converted": False, "reason": "merged or irregular cells: kept as returned"})
            return match.group(0)
        notes.append({"converted": True, "rows": len(rows), "columns": width})
        escape = lambda value: html.unescape(value).replace("|", "\\|")  # noqa: E731
        lines = ["| " + " | ".join(escape(c) for c in rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(escape(c) for c in row) + " |" for row in rows[1:]]
        return "\n" + "\n".join(lines) + "\n"

    return re.sub(r"<table\b.*?</table>", convert, text, flags=re.IGNORECASE | re.DOTALL), notes


def transcriptions(set_dir: Path, outputs: dict[str, list[dict]], selection: dict | None) -> tuple[dict, dict]:
    data = load_set(set_dir)
    found: dict[str, dict] = {}
    notes: dict[str, dict] = {}
    choices = (selection or {}).get("choices") or {}
    for region in data["regions"]:
        choice = (choices.get(region["region_id"]) or {}).get("output_key")
        if not choice:
            continue
        output = next((o for o in outputs.get(region["region_id"], []) if o["output_key"] == choice), None)
        if output is None or not output["usable"]:
            notes[region["region_id"]] = {"placed": False, "reason": "the selected output is missing or unusable"}
            continue
        text, table_notes = html_tables_to_markdown(output["text"])
        key = region["parent"] or region["region_id"]
        model = (f"{output['model']} (manual import, declared)" if output["provenance"] == "manual_import"
                 else f"{output['provider']} {output['model']}")
        part = text if region["parent"] is None else f"Part {region['region_id'].rsplit('_c', 1)[1]}:\n{text}"
        entry = found.setdefault(key, {"text": "", "model": model, "sources": [], "bbox": None, "page": region["page"]})
        entry["text"] = (entry["text"] + "\n\n" + part).strip()
        if entry["model"] != model:
            entry["model"] = "several models"
        if region["parent"] is None:
            entry["bbox"] = region["bbox"]
        entry["sources"].append({"region_id": region["region_id"], "output_key": choice, "page": region["page"],
                                 "provider": output["provider"], "model": output["model"],
                                 "provenance": output["provenance"], "html_tables": table_notes,
                                 "numbers": output["assessment"]["numbers"],
                                 "needs_review": output["assessment"]["reasons"]})
    return found, notes


def compute_enriched(run_dir: Path, set_dir: Path, spec_for: Callable[[str], ProviderSpec | None], root: Path, *,
                     run_spec: ProviderSpec | None, pdf_path: Path | None = None) -> dict:
    """Unified page documents from saved outputs and the user's latest selection. No request."""
    from lab.core.hybrid import run as hybrid  # noqa: PLC0415

    outputs = collect_outputs(set_dir, spec_for, root)
    selection = latest_selection(set_dir)
    entries, notes = transcriptions(set_dir, outputs, selection)
    computed = hybrid.compute_layout(run_dir, run_spec, root, pdf_path=pdf_path,
                                     transcriptions={key: {"text": e["text"], "model": e["model"]}
                                                     for key, e in entries.items()})
    for page in computed["pages"]:
        record = page["record"]
        record["image_extraction"] = []
        figures = {f"p{page['page']}_{fig['id']}": fig for fig in (page["elements"].get("figures") or [])}
        blocks = page["elements"].get("blocks") or []
        for key, entry in entries.items():
            if entry["page"] != page["page"]:
                continue
            figure = figures.get(key)
            placed = figure is not None and (entry["bbox"] is None or all(
                abs(a - b) <= 1.0 for a, b in zip(figure["bbox"], entry["bbox"])))
            info = {"figure": key, "placed": placed, "sources": entry["sources"]}
            if not placed:
                page["markdown"] += (f"\n> 🖼️ **Image region {key}** (its position could not be matched to the current "
                                     f"layout)\n>\n" + "\n".join(f"> {line}" for line in entry["text"].splitlines())
                                     + f"\n>\n> <sub>(read by {entry['model']}; numbers need review)</sub>\n")
                record["needs_review"].append(f"image region {key} placed at the end: position not matched")
            else:
                inside = " ".join(b.get("raw_text") or "" for b in blocks
                                  if hybrid_layout_overlap(b["bbox"], figure["bbox"]))
                tokens = [t for t in _norm(entry["text"]).split() if len(t) > 1]
                shared = sum(1 for t in tokens if t in set(_norm(inside).split()))
                info["text_layer_overlap"] = round(shared / len(tokens), 3) if tokens else 0.0
                if tokens and shared / len(tokens) >= 0.6:
                    record["needs_review"].append(f"{key}: extracted text largely repeats the PDF text layer "
                                                  "(kept; review for duplication)")
            for source in entry["sources"]:
                record["needs_review"] += [f"{key} ({source['output_key']}): {reason}" for reason in source["needs_review"]]
            record["image_extraction"].append(info)
        for rid, note in notes.items():
            if rid.startswith(f"p{page['page']}_"):
                record["needs_review"].append(f"{rid}: {note['reason']}; placeholder kept")
        record["layout_status"] = "Needs Review" if record["needs_review"] else "No issues detected (automatic)"
    computed["full_document"] = "\n".join(f"--- Page {p['page']} ---\n\n{p['markdown']}" for p in computed["pages"])
    computed["enrichment"] = {"set": Path(set_dir).name, "selection": (selection or {}).get("path"),
                              "regions_with_output": sorted(entries)}
    return computed


def hybrid_layout_overlap(a, b) -> bool:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1])) > 0


def documents_dir(run_dir: Path) -> Path:
    return enrichment_dir(run_dir) / "documents"


def apply_enriched(run_dir: Path, set_dir: Path, spec_for, root: Path, *, run_spec, by: str, statement: str,
                   pdf_path: Path | None = None) -> Path:
    if statement != DOCUMENT_STATEMENT or not by.strip():
        raise EnrichError("Enter your name and tick the confirmation")
    computed = compute_enriched(run_dir, set_dir, spec_for, root, run_spec=run_spec, pdf_path=pdf_path)
    base = documents_dir(run_dir)
    folder = _new_dir(base, "rev")
    files = {}
    for page in computed["pages"]:
        files[f"page_{page['page']:03d}.md"] = page["markdown"].encode("utf-8", errors="surrogatepass")
        files[f"page_{page['page']:03d}.enrich.json"] = json.dumps(
            {"record": page["record"]}, ensure_ascii=False, indent=2).encode("utf-8", errors="surrogatepass")
    files["full_document.md"] = computed["full_document"].encode("utf-8", errors="surrogatepass")
    for name, data in files.items():
        _write_once(folder / name, data)
    sequence = int(folder.name.split("_")[1])
    _write_once(base / f"revision_{sequence:06d}.json", {
        "schema_version": 1, "kind": "hybrid_enrichment_revision", "revision": sequence, "folder": folder.name,
        "enrich_version": ENRICH_VERSION, "approved_by": by.strip(), "statement": statement,
        "created_at": _now().isoformat(), "sources": computed["sources"] | computed["enrichment"],
        "pages": [{"page": p["page"], "needs_review": p["record"]["needs_review"],
                   "image_extraction": p["record"].get("image_extraction")} for p in computed["pages"]],
        "files_sha256": {name: _sha(data) for name, data in files.items()}})
    return folder


def latest_enriched(run_dir: Path) -> dict | None:
    records = [payload for _path, payload in ordered(documents_dir(run_dir), "revision") if payload]
    if not records:
        return None
    latest = records[-1]
    folder = documents_dir(run_dir) / latest["folder"]
    pages = []
    for item in latest["pages"]:
        md = folder / f"page_{item['page']:03d}.md"
        record = (_read(folder / f"page_{item['page']:03d}.enrich.json") or {}).get("record") or item
        pages.append({"page": item["page"], "markdown": md.read_text(encoding="utf-8") if md.exists() else "",
                      "record": record})
    full = folder / "full_document.md"
    return {"revision": f"enrichment {latest['revision']}", "folder": str(folder), "pages": pages,
            "full_document": full.read_text(encoding="utf-8") if full.exists() else "", "approved": True,
            "kind": "enrichment", "approved_by": latest.get("approved_by"), "created_at": latest.get("created_at"),
            "layout_version": (latest.get("sources") or {}).get("layout_version")}
