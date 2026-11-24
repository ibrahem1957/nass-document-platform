"""Read-only "all pages" view data and exports (ZIP, CSV, HTML) for an E2-D batch.

Everything is built in memory from results that already exist: the batch status
(write-once item logs and job journals), each result's saved `run.json`, and the
exact page image saved with the result (`page.png` in an atomic page snapshot, or
the run's `transmitted/` copy). Nothing is sent anywhere, no request is retried, and
nothing is written to the results store. A missing output is shown as NOT AVAILABLE
with its recorded reason; it is never filled in.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import batch_report
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts

AVAILABLE = "AVAILABLE"
TIMEOUT_LABEL = "NOT AVAILABLE — request timed out"
CSV_FIELDS = [
    "document_id", "document_sha256", "language", "page", "provider_id", "provider", "status", "availability",
    "raw_text", "latency_ms", "cost_usd", "cost_source", "run_id", "item_id", "result_directory", "detail",
]

ImageLookup = Callable[[str, int], "Path | None"]


def _journal_error_codes(provider_id: str, journal_id: str | None, root: Path) -> list[str]:
    if not journal_id:
        return []
    journal = job_store.JobJournal(job_store.journal_dir(provider_id, journal_id, root))
    codes = []
    for event in journal.events():
        error = event.get("error")
        if isinstance(error, dict) and error.get("code"):
            codes.append(str(error["code"]))
    return codes


def _unavailable(state: dict, root: Path, image_sha256: str | None = None) -> str:
    codes = _journal_error_codes(state["provider_id"], state.get("journal_id"), root)
    if not codes and image_sha256:
        # No attempt in this batch (e.g. blocked by an earlier unfinished job): use the
        # recorded errors of earlier jobs for this exact page image and provider.
        for journal in job_store.journals(state["provider_id"], root):
            events = journal.events()
            if events and events[0].get("image_sha256") == image_sha256:
                codes += [str(event["error"]["code"]) for event in events
                          if isinstance(event.get("error"), dict) and event["error"].get("code")]
    detail = str(state.get("detail") or "")
    if "request_timeout" in codes or "deadline_exceeded" in codes or "request_timeout" in detail:
        return TIMEOUT_LABEL
    if state["status"] in {batch_mod.PENDING, batch_mod.INTERRUPTED_BEFORE_SEND}:
        return "NOT AVAILABLE — not processed"
    return f"NOT AVAILABLE — {state['status']}"


def _saved_image(run_dir: str | None, page: int) -> Path | None:
    """The exact PNG saved with a result (snapshot `page.png` or the run's sent copy)."""
    if not run_dir:
        return None
    base = Path(run_dir)
    snapshot = base / "page.png"
    if snapshot.is_file():
        return snapshot
    transmitted = base / "transmitted" / f"page_{page:04d}.png"
    if transmitted.is_file():
        return transmitted
    found = sorted((base / "transmitted").glob("*.png")) if (base / "transmitted").is_dir() else []
    return found[0] if len(found) == 1 else None


def collect(batch_dir: Path, root: Path) -> dict[str, Any]:
    """All documents, pages and provider outputs of one batch, in document order."""
    plan = batch_mod.load_plan(batch_dir)
    status = batch_mod.batch_status(batch_dir, root)
    states = {item["item_id"]: item for item in status["items"]}
    providers = [{"provider_id": provider["provider_id"], "label": provider["label"],
                  "short": page_artifacts.provider_folder(provider["provider_id"])}
                 for provider in plan["providers"]]
    order = {document["registry_id"]: index for index, document in enumerate(plan["documents"])}
    documents = {document["registry_id"]: document for document in plan["documents"]}
    pages: dict[tuple[str, int], dict] = {}
    for item in plan["items"]:
        key = (item["registry_id"], item["page"])
        state = states[item["item_id"]]
        entry = pages.setdefault(key, {
            "registry_id": item["registry_id"], "language": item["language"], "page": item["page"],
            "document_sha256": item["document_sha256"], "image_sha256": item["image_sha256"],
            "document_title": documents.get(item["registry_id"], {}).get("title"), "outputs": {},
            "image_candidates": [],
        })
        run = batch_report._run_payload(state.get("run_directory")) if state["status"] in batch_mod.DONE else None
        text = None
        if run and run.get("status") == "ok" and run.get("pages"):
            value = run["pages"][0].get("text")
            text = value if isinstance(value, str) else None
        usage = (run or {}).get("usage") or {}
        page_raw = ((((run or {}).get("environment") or {}).get("e2") or {}).get("page_raw") or {})
        raw_page = page_raw.get(str(item["page"])) if isinstance(page_raw, dict) else None
        response = (raw_page or {}).get("response") if isinstance(raw_page, dict) else None
        entry["outputs"][item["provider_id"]] = {
            "provider_id": item["provider_id"], "item_id": item["item_id"], "status": state["status"],
            "availability": AVAILABLE if text is not None else _unavailable(state, root, item["image_sha256"]),
            "text": text, "detail": state.get("detail"), "run_id": (run or {}).get("run_id"),
            "run_directory": state.get("run_directory"), "artifact_directory": state.get("artifact_directory"),
            "latency_ms": (run or {}).get("wall_ms"), "cost_usd": usage.get("cost_usd", state.get("cost_usd")),
            "cost_source": usage.get("cost_source", state.get("cost_source")),
            "finish_reason": response.get("finish_reason") if isinstance(response, dict) else None,
        }
        if text is not None:
            entry["image_candidates"].append(state.get("run_directory"))
    ordered = sorted(pages.values(), key=lambda entry: (order.get(entry["registry_id"], 99), entry["page"]))
    return {"batch_id": plan["batch_id"], "stage": plan["stage"], "plan_sha256": plan["plan_sha256"],
            "definition_title": plan["definition_title"], "documents": plan["documents"],
            "providers": providers, "pages": ordered,
            "counts": {"pages": len(ordered),
                       "outputs": sum(len(page["outputs"]) for page in ordered),
                       "available": sum(output["availability"] == AVAILABLE
                                        for page in ordered for output in page["outputs"].values())}}


def page_image(entry: dict, render: ImageLookup | None = None) -> Path | None:
    """Prefer the saved image of an existing result; render locally only as a fallback."""
    for run_dir in entry.get("image_candidates") or []:
        path = _saved_image(run_dir, entry["page"])
        if path is not None:
            return path
    return render(entry["registry_id"], entry["page"]) if render else None


def text_direction(language: str) -> str:
    return "rtl" if str(language).lower().startswith("arab") else "ltr"


# --------------------------------------------------------------------- CSV


def to_csv(collected: dict) -> bytes:
    labels = {provider["provider_id"]: provider["label"] for provider in collected["providers"]}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for page in collected["pages"]:
        for provider_id, output in page["outputs"].items():
            writer.writerow({
                "document_id": page["registry_id"], "document_sha256": page["document_sha256"],
                "language": page["language"], "page": page["page"], "provider_id": provider_id,
                "provider": labels.get(provider_id, provider_id), "status": output["status"],
                "availability": output["availability"], "raw_text": output["text"] or "",
                "latency_ms": "" if output["latency_ms"] is None else round(float(output["latency_ms"]), 1),
                "cost_usd": "" if output["cost_usd"] is None else output["cost_usd"],
                "cost_source": output["cost_source"] or "", "run_id": output["run_id"] or "",
                "item_id": output["item_id"], "result_directory": output["run_directory"] or "",
                "detail": "" if output["availability"] == AVAILABLE else (output["detail"] or ""),
            })
    # UTF-8 with BOM so spreadsheet programs display Arabic correctly.
    return buffer.getvalue().encode("utf-8-sig")


# --------------------------------------------------------------------- ZIP


def _metadata(collected: dict, page: dict, image: Path | None) -> dict:
    return {
        "batch_id": collected["batch_id"], "stage": collected["stage"], "plan_sha256": collected["plan_sha256"],
        "document_id": page["registry_id"], "document_sha256": page["document_sha256"],
        "document_title": page["document_title"], "language": page["language"], "page": page["page"],
        "page_image_sha256_planned": page["image_sha256"],
        "page_image_source": str(image) if image else None,
        "evidence_class": batch_report.API_LABEL,
        "outputs": {provider_id: {key: value for key, value in output.items() if key != "text"}
                    for provider_id, output in page["outputs"].items()},
    }


def to_zip(collected: dict, render: ImageLookup | None = None) -> bytes:
    """`<document>/page-NNN/{original.png, <provider>_raw.txt, metadata.json}` plus CSV and manifest."""
    shorts = {provider["provider_id"]: provider["short"] for provider in collected["providers"]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for page in collected["pages"]:
            folder = f"{page['registry_id']}/page-{page['page']:03d}"
            image = page_image(page, render)
            if image is not None and Path(image).is_file():
                archive.writestr(f"{folder}/original.png", Path(image).read_bytes(),
                                 compress_type=zipfile.ZIP_STORED)
            for provider_id, output in page["outputs"].items():
                if output["text"] is not None:
                    archive.writestr(f"{folder}/{shorts.get(provider_id, provider_id)}_raw.txt",
                                     output["text"].encode("utf-8"))
            archive.writestr(f"{folder}/metadata.json",
                             json.dumps(_metadata(collected, page, image), ensure_ascii=False, indent=2).encode("utf-8"))
        archive.writestr("results.csv", to_csv(collected))
        archive.writestr("manifest.json", json.dumps({
            "batch_id": collected["batch_id"], "stage": collected["stage"], "plan_sha256": collected["plan_sha256"],
            "generated_at": datetime.now(timezone.utc).isoformat(), "counts": collected["counts"],
            "providers": collected["providers"], "documents": collected["documents"],
            "note": "Built from existing saved results only; missing outputs are marked NOT AVAILABLE, never filled in.",
        }, ensure_ascii=False, indent=2).encode("utf-8"))
    return buffer.getvalue()


# -------------------------------------------------------------------- HTML

_CSS = """
:root { --border:#d0d4da; --muted:#5b6470; --bad:#a1260d; --ok:#1d6b35; --bg:#ffffff; --fg:#1b1f24; --panel:#f6f8fa; }
@media (prefers-color-scheme: dark) { :root { --border:#3a414b; --muted:#a4adb8; --bad:#ff8a73; --ok:#7bd88f;
  --bg:#14171b; --fg:#e7eaee; --panel:#1d2127; } }
body { font-family: system-ui, "Segoe UI", "Noto Sans", "Noto Naskh Arabic", Tahoma, sans-serif; margin: 0 16px 48px;
  background: var(--bg); color: var(--fg); }
h1 { font-size: 1.4rem; } h2 { margin-top: 2rem; border-bottom: 2px solid var(--border); }
.meta { color: var(--muted); font-size: .9rem; }
.page { border: 1px solid var(--border); border-radius: 8px; margin: 16px 0; padding: 12px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }
@media (max-width: 960px) { .grid { grid-template-columns: minmax(0, 1fr); } }
.cell h3 { font-size: 1rem; margin: 0 0 6px; }
.cell img { width: 100%; height: auto; border: 1px solid var(--border); }
.text { white-space: pre-wrap; unicode-bidi: plaintext; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px; max-height: 70vh; overflow: auto; font-size: .9rem; line-height: 1.5; }
.ok { color: var(--ok); } .bad { color: var(--bad); font-weight: 600; }
"""


def _image_tag(path: Path | None, alt: str, embed: bool) -> str:
    if path is None or not Path(path).is_file():
        return '<p class="bad">Original page image not available</p>'
    if not embed:
        return f'<p class="meta">Image not embedded (source: {html.escape(str(path))})</p>'
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img loading="lazy" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'


def to_html(collected: dict, render: ImageLookup | None = None, *, embed_images: bool = True) -> bytes:
    labels = {provider["provider_id"]: provider["label"] for provider in collected["providers"]}
    parts = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>E2-D comparison {html.escape(collected['batch_id'])}</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(collected['definition_title'])}: side-by-side comparison</h1>",
        f'<p class="meta">Batch {html.escape(collected["batch_id"])} (Stage {collected["stage"]}) · plan SHA-256 '
        f'{html.escape(collected["plan_sha256"])} · generated {datetime.now(timezone.utc).isoformat()} · '
        f'{collected["counts"]["available"]} of {collected["counts"]["outputs"]} outputs available · '
        f'{html.escape(batch_report.API_LABEL)}. Missing outputs are marked NOT AVAILABLE and never filled in.</p>',
    ]
    by_document: dict[str, list[dict]] = {}
    for page in collected["pages"]:
        by_document.setdefault(page["registry_id"], []).append(page)
    for document in collected["documents"]:
        pages = by_document.get(document["registry_id"]) or []
        if not pages:
            continue
        parts.append(f"<h2>{html.escape(str(document.get('language') or ''))} document — {len(pages)} pages "
                     f'<span class="meta">({html.escape(document["registry_id"])})</span></h2>')
        for page in pages:
            direction = text_direction(page["language"])
            image = page_image(page, render)
            parts.append(f'<section class="page" id="{html.escape(page["registry_id"])}-p{page["page"]}">'
                         f"<h3>{html.escape(page['registry_id'])} · page {page['page']}</h3><div class=\"grid\">")
            parts.append('<div class="cell"><h3>Original PDF page</h3>'
                         + _image_tag(image, f"{page['registry_id']} page {page['page']}", embed_images) + "</div>")
            for provider_id, output in page["outputs"].items():
                header = f"<h3>{html.escape(labels.get(provider_id, provider_id))}</h3>"
                if output["text"] is not None:
                    status = f'<p class="meta ok">{html.escape(output["status"])} · RAW output</p>'
                    body = f'<div class="text" dir="{direction}">{html.escape(output["text"])}</div>'
                else:
                    status = f'<p class="bad">{html.escape(output["availability"])}</p>'
                    body = f'<p class="meta">{html.escape(str(output["detail"] or ""))}</p>'
                parts.append(f'<div class="cell">{header}{status}{body}</div>')
            parts.append("</div></section>")
    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")


__all__ = ["AVAILABLE", "TIMEOUT_LABEL", "collect", "page_image", "text_direction", "to_csv", "to_html", "to_zip"]
