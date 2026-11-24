"""Local PDF Extraction Benchmark: extractors, storage, runner, resume, exports.

Everything runs on small synthetic PDFs in pytest's tmp_path; no project result
folder is used or touched.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path

import pytest

from lab.local_pdf_extraction import exports, extractors, runner, storage
from lab.local_pdf_extraction.extractors import PageOutput
from tests.lab import lpe_pdf_factory as factory

ALL = ["pymupdf", "pdfplumber", "pypdf", "pypdfium2", "pdfminer", "pymupdf_ar_reordered"]


@pytest.fixture
def pdf_bytes(tmp_path) -> bytes:
    return factory.small_pdf(tmp_path / "small.pdf", pages=3).read_bytes()


def _new_run(root: Path, data: bytes, methods=ALL, names=("تقرير المرور ٢٠٢٥.pdf",)) -> Path:
    return storage.create_run(root, [(name, data) for name in names], runner.method_entries(list(methods), {}),
                              {"reorder": {}})


def _snapshot(run_path: Path) -> dict[str, tuple[bytes, int]]:
    return {str(p.relative_to(run_path)): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in (run_path / "documents").rglob("page_*.*") if "_quarantine" not in p.parts}


# ---------------------------------------------------------------- extractors
def test_every_library_imports_and_its_api_runs(tmp_path) -> None:
    path = factory.small_pdf(tmp_path / "one.pdf", pages=1)
    for method in extractors.METHODS:
        status = extractors.availability(method)
        assert status["available"], (method.method_id, status["error"])
        assert status["version"]
        session = extractors.open_session(method.method_id, path, {"reorder": {}})
        try:
            assert session.page_count == 1
            output = session.extract(1)
        finally:
            session.close()
        assert isinstance(output.text, str) and "Road safety" in output.text, method.method_id


def test_raw_outputs_are_saved_byte_for_byte_and_only_processed_differs(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes)
    final = runner.run(run_path)
    assert final["status"] == "COMPLETED"
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    source = storage.original_pdf(run_path, key)
    for method_id in extractors.RAW_METHOD_IDS:
        session = extractors.open_session(method_id, source, {})
        try:
            for page in (1, 2, 3):
                expected = session.extract(page).text
                text_path, json_path = storage.page_paths(run_path, key, method_id, page)
                assert text_path.read_bytes() == expected.encode("utf-8", "surrogatepass")
                record = json.loads(json_path.read_text(encoding="utf-8"))
                assert record["output_type"] == "raw" and record["status"] == "COMPLETED"
                assert "reorder" not in record
        finally:
            session.close()
    processed = json.loads(storage.page_paths(run_path, key, "pymupdf_ar_reordered", 1)[1].read_text(encoding="utf-8"))
    assert processed["output_type"] == "processed"
    assert processed["reorder"]["reorder_status"] in ("not_applicable", "not_needed", "reconstructed", "uncertain")
    assert processed["reorder"]["verified_against_ground_truth"] is False
    for path in run_path.rglob("*.json"):
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert (storage.method_dir(run_path, key, "pymupdf") / "full_document.txt").read_text(encoding="utf-8").startswith("=== page 1 ===")


def test_save_and_reload_from_disk(tmp_path, pdf_bytes) -> None:
    root = tmp_path / "root"
    run_path = _new_run(root, pdf_bytes)
    runner.run(run_path)
    assert storage.list_runs(root) == [run_path]
    collected = exports.collect(run_path)  # a fresh load: only files on disk
    assert collected["status"]["status"] == "COMPLETED"
    manifest = collected["manifest"]
    assert manifest["documents"][0]["original_name"] == "تقرير المرور ٢٠٢٥.pdf"
    assert len(manifest["documents"][0]["sha256"]) == 64
    assert {m["method_id"] for m in manifest["methods"]} == set(ALL)
    assert all(m["library_version"] for m in manifest["methods"])
    assert manifest["settings"]["reorder"] == {}
    texts = [page["outputs"]["pypdf"]["text"] for page in collected["documents"][0]["pages"]]
    assert all("Road safety" in text for text in texts)
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["methods"]["pymupdf"]["completed_pages"] == 3


def test_document_folder_names_are_sanitized(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"],
                        names=("../../..\\evil<>name.pdf", "تقرير.pdf", "C:\\Windows\\x.pdf"))
    keys = [d["doc_key"] for d in storage.load_manifest(run_path)["documents"]]
    assert len(set(keys)) == 3
    for key in keys:
        folder = storage.document_dir(run_path, key)
        assert folder.parent == run_path / "documents" and folder.exists()
        assert key.isascii() and len(key) <= 40 and ".." not in key and "\\" not in key
    with pytest.raises(storage.StorageError):
        storage.document_dir(run_path, "../escape")
    with pytest.raises(storage.StorageError):
        storage.run_dir(tmp_path, "../../x")
    assert storage.new_run_id() != storage.new_run_id()


def test_rejects_non_pdf_upload(tmp_path) -> None:
    with pytest.raises(storage.StorageError):
        storage.create_run(tmp_path, [("x.pdf", b"not a pdf")], runner.method_entries(["pymupdf"], {}), {})


# ------------------------------------------------------- failure isolation
class _Failing:
    def __init__(self, path, settings):
        self.page_count = 3

    def extract(self, page):
        if page == 2:
            raise RuntimeError("synthetic library failure")
        return PageOutput(f"ok {page}")

    def close(self):
        pass


class _BrokenOpen:
    def __init__(self, path, settings):
        raise OSError("cannot open")


def test_failing_library_does_not_block_or_delete_others(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf", "pypdf", "pdfplumber"])
    registry = dict(extractors.SESSIONS, pymupdf=_Failing, pypdf=_BrokenOpen)
    final = runner.run(run_path, registry=registry)
    assert final["status"] == "PARTIAL"
    methods = final["scan"]["methods"]
    assert methods["pymupdf"] == {"valid": 2, "failed": 1, "missing": 0, "invalid": 0, "total": 3}
    assert methods["pypdf"]["failed"] == 3
    assert methods["pdfplumber"]["valid"] == 3
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    record = json.loads(storage.page_paths(run_path, key, "pymupdf", 2)[1].read_text(encoding="utf-8"))
    assert record["status"] == "FAILED" and "synthetic library failure" in record["error"]
    state = storage.load_state(run_path)
    assert state["methods"]["pypdf"]["status"] == "FAILED"
    assert state["methods"]["pdfplumber"]["status"] == "COMPLETED"
    assert not (run_path / storage.LOCK_NAME).exists()


# ---------------------------------------------------- interruption / resume
class _Interrupting:
    calls = 0

    def __init__(self, path, settings):
        self._inner = extractors.PyMuPDFSession(path, settings)
        self.page_count = self._inner.page_count

    def extract(self, page):
        _Interrupting.calls += 1
        if _Interrupting.calls == 2:
            raise KeyboardInterrupt  # simulates the process being stopped mid-run
        return self._inner.extract(page)

    def close(self):
        self._inner.close()


def test_interrupted_run_resume_only_missing_and_approval_for_corrupt(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf", "pypdf"])
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    _Interrupting.calls = 0
    with pytest.raises(KeyboardInterrupt):
        runner.run(run_path, registry=dict(extractors.SESSIONS, pymupdf=_Interrupting))
    status = storage.derive_status(run_path)
    assert status["status"] == "INTERRUPTED"
    assert status["label"] == "INTERRUPTED — 1/6 pages"
    assert storage.load_manifest(run_path)["status"] == "INTERRUPTED"

    # Resume #1: only the missing pages are processed; page 1 stays byte-identical.
    before = _snapshot(run_path)
    time.sleep(0.02)
    runner.run(run_path)
    after = _snapshot(run_path)
    for name, value in before.items():
        assert after[name] == value
    assert storage.derive_status(run_path)["status"] == "COMPLETED"

    # Corrupt one page and delete another.
    text_path, _ = storage.page_paths(run_path, key, "pypdf", 2)
    text_path.write_bytes(text_path.read_bytes() + b"tampered")
    for path in storage.page_paths(run_path, key, "pymupdf", 3):
        path.unlink()
    scan = storage.scan_run(run_path)
    assert scan["counts"]["missing"] == 1
    assert [(i["method_id"], i["page"], i["state"]) for i in scan["issues"]] == [("pypdf", 2, "invalid")]
    assert storage.derive_status(run_path)["status"] == "INTERRUPTED"

    # Resume #2 without approval: missing page re-extracted, corrupt page left alone.
    before = _snapshot(run_path)
    runner.run(run_path)
    assert storage.verify_page(run_path, key, "pymupdf", 3)[0] == "valid"
    assert storage.verify_page(run_path, key, "pypdf", 2)[0] == "invalid"
    after = _snapshot(run_path)
    for name, value in before.items():
        assert after[name] == value  # nothing valid (or the corrupt file) was rewritten

    # Resume #3 with explicit approval: corrupt page re-extracted, old files quarantined.
    request = runner.create_resume_request(run_path, None, [{"doc_key": key, "method_id": "pypdf", "page": 2}])
    before = _snapshot(run_path)
    runner.run(run_path, resume_request=request)
    assert storage.verify_page(run_path, key, "pypdf", 2)[0] == "valid"
    quarantine = list((storage.method_dir(run_path, key, "pypdf") / "_quarantine").iterdir())
    assert len(quarantine) == 2
    after = _snapshot(run_path)
    for name, value in before.items():
        if "pypdf" in name and "page_002" in name:
            continue
        assert after[name] == value
    assert storage.derive_status(run_path)["status"] == "COMPLETED"


def test_valid_page_is_never_overwritten(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"])
    runner.run(run_path)
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    with pytest.raises(storage.StorageError):
        storage.write_page_result(run_path, key, "pymupdf", 1, {"status": "COMPLETED"}, "x", replace_approved=True)


def test_hard_kill_leaves_stale_lock_that_needs_explicit_clearing(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"])
    (run_path / storage.LOCK_NAME).write_text(json.dumps({"pid": 999999, "host": __import__("socket").gethostname(),
                                                          "token": "dead", "heartbeat_epoch": 0}), encoding="utf-8")
    state = storage.load_state(run_path)
    state.update(status="RUNNING", attempt=1)
    storage.atomic_write_json(run_path / "state.json", state)
    status = storage.derive_status(run_path)
    assert status["status"] == "INTERRUPTED" and status["lock"]["stale"]
    with pytest.raises(storage.StorageError):
        runner.launch_worker(run_path)
    with pytest.raises(storage.StorageError):
        runner.run(run_path)
    storage.clear_stale_lock(run_path)
    assert not (run_path / storage.LOCK_NAME).exists()
    assert list((run_path / "logs").glob("stale_lock_cleared_*.json"))
    assert runner.run(run_path)["status"] == "COMPLETED"


def test_lock_prevents_a_second_worker(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"])
    token = storage.acquire_lock(run_path)
    try:
        assert storage.lock_status(run_path)["alive"]
        with pytest.raises(storage.StorageError):
            storage.acquire_lock(run_path)
        with pytest.raises(storage.StorageError):
            storage.clear_stale_lock(run_path)
    finally:
        storage.release_lock(run_path, token)


def test_cancel_request_stops_between_pages(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"])

    class _CancelAfterFirst(extractors.PyMuPDFSession):
        def extract(self, page):
            storage.request_cancel(run_path)
            return super().extract(page)

    final = runner.run(run_path, registry=dict(extractors.SESSIONS, pymupdf=_CancelAfterFirst))
    assert final["status"] == "INTERRUPTED"
    assert final["counts"]["valid"] == 1
    assert "cancel" in storage.load_state(run_path)["message"]


def test_detached_worker_process_completes(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf", "pypdfium2"])
    runner.launch_worker(run_path)
    deadline = time.time() + 90
    while time.time() < deadline:
        status = storage.derive_status(run_path)
        if status["status"] not in ("RUNNING", "PENDING"):
            break
        time.sleep(0.5)
    assert status["status"] == "COMPLETED", (run_path / "logs" / "worker_0001.log").read_text(errors="replace")


# ------------------------------------------------------------------ exports
def test_exports_from_saved_files(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes)
    runner.run(run_path)
    collected = exports.collect(run_path)
    csv_bytes = exports.to_csv(collected)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert csv_bytes.decode("utf-8-sig").count("\n") >= 3 * len(ALL)
    payload = json.loads(exports.to_json(collected))
    assert len(payload["pages"]) == 3 * len(ALL)
    report = exports.to_html(collected, lambda k, p: storage.page_image(run_path, k, p, 60)).decode("utf-8")
    assert "data:image/png;base64," in report and "PROCESSED output — not raw library output" in report
    for forbidden in ("http://", "https://", "<script", "<link"):
        assert forbidden not in report
    assert len(list((run_path / "page_images").rglob("*.png"))) == 3
    zipped = exports.to_zip(run_path, {"summary.csv": csv_bytes, "comparison.html": report.encode()})
    names = zipfile.ZipFile(io.BytesIO(zipped)).namelist()
    assert f"{run_path.name}/manifest.json" in names and f"{run_path.name}/exports/comparison.html" in names
    assert not any("page_images" in name for name in names)
    key = collected["documents"][0]["doc_key"]
    assert exports.document_txt(run_path, key, "pypdf").startswith(b"=== page 1 ===")
    exports.save_export(run_path, "summary.csv", csv_bytes)
    with pytest.raises(storage.StorageError):
        exports.save_export(run_path, "../x.csv", b"")


def test_ground_truth_scores_only_when_attached(tmp_path, pdf_bytes) -> None:
    run_path = _new_run(tmp_path / "root", pdf_bytes, methods=["pymupdf"])
    runner.run(run_path)
    assert exports.ground_truth_scores(exports.collect(run_path)) == {}
    key = storage.load_manifest(run_path)["documents"][0]["doc_key"]
    _, text = storage.read_page(run_path, key, "pymupdf", 1)
    storage.save_ground_truth(run_path, key, f"=== page 1 ===\n{text}\n=== page 2 ===\nشيء آخر", ["gt.txt"])
    with pytest.raises(storage.StorageError):
        storage.save_ground_truth(run_path, key, "x", ["gt.txt"])
    scores = exports.ground_truth_scores(exports.collect(run_path))[key]["pymupdf"]
    assert scores["level"] == "page"
    assert scores["pages"]["1"]["arabic_normalized"]["cer"] == 0.0
    assert scores["pages"]["2"]["none"]["cer"] > 0


def test_delete_requires_confirmation_and_only_one_run(tmp_path, pdf_bytes) -> None:
    root = tmp_path / "root"
    first = _new_run(root, pdf_bytes, methods=["pymupdf"])
    second = _new_run(root, pdf_bytes, methods=["pymupdf"])
    outside = tmp_path / "keep.txt"
    outside.write_text("keep", encoding="utf-8")
    with pytest.raises(storage.StorageError):
        storage.delete_run(root, first.name, "wrong")
    with pytest.raises(storage.StorageError):
        storage.delete_run(root, "..", "..")
    storage.delete_run(root, first.name, first.name)
    assert not first.exists() and second.exists() and outside.exists()


def test_results_root_uses_agenticdoc_results_dir(monkeypatch, tmp_path) -> None:
    from lab.local_pdf_extraction import config

    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path))
    assert config.benchmark_root() == tmp_path / "local_pdf_extraction"
    monkeypatch.delenv("AGENTICDOC_RESULTS_DIR")
    assert config.benchmark_root() == Path("D:/agenticdoc_e2/results") / "local_pdf_extraction"
    assert os.environ.get("AGENTICDOC_RESULTS_DIR") is None
