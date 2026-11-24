"""Read-only link to earlier OCR/VLM page snapshots (synthetic snapshot folders only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.core.e2.api import jobs as job_store
from lab.local_pdf_extraction import ocr_links

SHA = hashlib.sha256(b"same pdf").hexdigest()


def _snapshot(root: Path, provider: str, sha: str, page: int, text: str, *, state=job_store.COMPLETED,
              saved_at="2026-11-17T10:00:00+00:00", run="run_aaa") -> Path:
    path = root / "pages" / provider.split("-")[0] / f"traffic_ar--{sha[:12]}" / f"page-{page:04d}" / run / "snapshot-000001"
    path.mkdir(parents=True)
    files = {
        "page.png": b"png", "raw_text.txt": text.encode("utf-8"), "raw_response.json": b"{}",
        "status.json": b"{}", "job_journal.json": b"{}", "request_response.json": b"{}",
        "structured_output.json": b"{}",
        "metadata.json": json.dumps({"document_sha256": sha, "page": page, "state": state, "provider_id": provider,
                                     "model_id": "model-x", "run_id": run, "saved_at": saved_at}).encode(),
    }
    for name, data in files.items():
        (path / name).write_bytes(data)
    manifest = {"files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    digest = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    (path / "complete.json").write_text(json.dumps({"complete": True, "manifest_sha256": digest}), encoding="utf-8")
    return path


def test_only_verified_completed_full_sha_matches_are_linked(tmp_path) -> None:
    _snapshot(tmp_path, "baidu-paddleocr-vl", SHA, 1, "نص بايدو")
    _snapshot(tmp_path, "oxen-qwen3-vl", SHA, 1, "old", saved_at="2026-11-16T00:00:00+00:00", run="run_old")
    _snapshot(tmp_path, "oxen-qwen3-vl", SHA, 1, "new", saved_at="2026-11-17T00:00:00+00:00", run="run_new")
    _snapshot(tmp_path, "oxen-qwen3-vl", SHA, 2, "failed", state=job_store.FAILED)
    tampered = _snapshot(tmp_path, "baidu-paddleocr-vl", SHA, 3, "x", run="run_t")
    (tampered / "raw_text.txt").write_bytes(b"tampered")
    other = hashlib.sha256(b"other").hexdigest()
    collision = SHA[:12] + other[12:]  # same folder prefix, different full SHA-256
    _snapshot(tmp_path, "baidu-paddleocr-vl", collision, 4, "wrong document", run="run_c")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    found = ocr_links.find_previous_ocr(tmp_path, SHA)

    assert set(found) == {"baidu-paddleocr-vl", "oxen-qwen3-vl"}
    assert found["baidu-paddleocr-vl"][1]["text"] == "نص بايدو"
    assert found["oxen-qwen3-vl"][1]["text"] == "new"  # latest verified snapshot
    assert 2 not in found["oxen-qwen3-vl"]  # FAILED state ignored
    assert set(found["baidu-paddleocr-vl"]) == {1}  # tampered and SHA-collision pages ignored
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before  # read-only
    assert ocr_links.find_previous_ocr(tmp_path, "not-a-sha") == {}
    assert ocr_links.find_previous_ocr(tmp_path / "missing", SHA) == {}
