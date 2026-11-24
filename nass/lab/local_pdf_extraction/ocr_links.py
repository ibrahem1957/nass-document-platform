"""READ-ONLY link to earlier OCR/VLM page results (E2-D Baidu PaddleOCR-VL / Oxen Qwen3-VL).

Matching is by the full PDF SHA-256 only: a snapshot is used when its folder name
carries the SHA-256 prefix AND its metadata.json `document_sha256` equals the full
hash, its integrity check passes (`page_artifacts.verify_snapshot`, which only reads
and hashes files) and its state is COMPLETED. Anything uncertain is not shown.
Nothing is ever written under these folders.

Qari (E2-A local runs under `runs/`) stores whole-run results without a per-page
document SHA-256 in a page-snapshot layout; it is not linked here (see the report).
"""

from __future__ import annotations

from pathlib import Path

from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.page_artifacts import verify_snapshot


def find_previous_ocr(results_root: Path, document_sha256: str) -> dict[str, dict[int, dict]]:
    """{provider_id: {page: {"text", "model_id", "run_id", "snapshot", "saved_at", "label"}}} (latest per page)."""
    sha = document_sha256.lower()
    if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        return {}
    base = Path(results_root) / "pages"
    if not base.is_dir():
        return {}
    found: dict[str, dict[int, dict]] = {}
    for snapshot in sorted(base.glob(f"*/*--{sha[:12]}/page-*/run_*/snapshot-*")):
        if not snapshot.is_dir():
            continue
        checked = verify_snapshot(snapshot)
        metadata = checked.get("metadata") or {}
        if not checked["ok"] or metadata.get("document_sha256", "").lower() != sha:
            continue
        if metadata.get("state") != job_store.COMPLETED:
            continue
        try:
            page = int(metadata["page"])
            text = (snapshot / "raw_text.txt").read_bytes().decode("utf-8")
        except (KeyError, ValueError, OSError, UnicodeDecodeError):
            continue
        provider = str(metadata.get("provider_id") or snapshot.parts[-5])
        entry = {
            "text": text, "model_id": metadata.get("model_id"), "run_id": metadata.get("run_id"),
            "snapshot": str(snapshot), "saved_at": metadata.get("saved_at") or "",
            "label": f"OCR/VLM (earlier run, read-only) — {provider} · {metadata.get('model_id') or ''}",
        }
        current = found.setdefault(provider, {}).get(page)
        if current is None or entry["saved_at"] > current["saved_at"]:
            found[provider][page] = entry
    return found
