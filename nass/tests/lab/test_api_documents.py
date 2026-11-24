"""Exact-hash synthetic document authorization for hosted API evaluation.

All provider traffic is mocked; nothing leaves the machine."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fitz
import pytest

from lab.core.e2.api import documents
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused, verify_document_for_transmission
from lab.core.e2.store import ResultStore
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager
from tests.lab.api_doc_helpers import authorize, register_test_documents
from tests.lab.test_api_baidu import (
    SUBMIT,
    TOKEN,
    ScriptedTransport,
    _done,
    _enable,
    _in_process,
    _jsonl,
    _spec,
)

PROJECT = Path(__file__).resolve().parents[2]
DOWNLOADS = Path.home() / "Downloads"
BENCHMARKS = {
    "agenticdoc-arabic-benchmark-v1": DOWNLOADS / "AgenticDoc_Arabic_Benchmark.pdf",
    "agenticdoc-english-benchmark-v1": DOWNLOADS / "AgenticDoc_English_Benchmark.pdf",
}


def _ingest(settings, pdf: Path):
    manager = WorkspaceManager(settings)
    workspace = manager.create()
    ingest = IngestService(manager, settings)
    with Path(pdf).open("rb") as handle:
        document = ingest.ingest(workspace.workspace_id, handle, Path(pdf).name)
    return manager, workspace.workspace_id, document, ingest.original_path(workspace.workspace_id, document.doc_id)


def _new_pdf(path: Path, text: str) -> Path:
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(path)
    return path


@pytest.fixture
def env(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    store = ResultStore(tmp_path / "results")
    spec = _spec()
    _enable(spec, store)
    return settings, tmp_path, store, spec, monkeypatch


def _run(runner, spec, workspace_id, document, source, transport):
    preview = runner.preview(spec, workspace_id, document, source, 1)
    return runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                      confirmed_sha256=preview["file"]["sha256"], executor=_in_process(transport))


def _happy():
    return ScriptedTransport([SUBMIT, _done(), _jsonl("mocked text")])


# ---------------------------------------------------------------- registry


TRAFFIC = {
    "traffic-accident-report-ar-v1": (DOWNLOADS / "تحليل العوامل المؤثرة في الحوادث المرورية.pdf", 18),
    "traffic-accident-report-en-v1": (DOWNLOADS / "Traffic_Accident_Report_English.pdf", 13),
}


def test_shipped_registry_lists_exactly_the_verified_documents() -> None:
    registry = documents.load_registry()
    assert {entry.id for entry in registry.values()} == set(BENCHMARKS) | set(TRAFFIC)
    for entry in registry.values():
        assert entry.bytes > 0 and len(entry.sha256) == 64
        if entry.id in BENCHMARKS:
            assert entry.page_count == 8 and entry.ground_truth_sha256
        else:
            # No Ground Truth exists for the traffic pair; provenance must stay honest.
            assert entry.page_count == TRAFFIC[entry.id][1]
            assert entry.ground_truth_sha256 is None and entry.ground_truth_filename is None
            assert "not independently verified" in entry.provenance
    for entry_id, (path, _pages) in TRAFFIC.items():
        if path.is_file():
            entry = next(item for item in registry.values() if item.id == entry_id)
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry.sha256 and len(data) == entry.bytes
    for entry_id, path in BENCHMARKS.items():
        if not path.is_file():
            continue  # verified where the user's files exist
        entry = next(item for item in registry.values() if item.id == entry_id)
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry.sha256 and len(data) == entry.bytes
        gt = path.with_name(entry.ground_truth_filename)
        if gt.is_file():
            assert hashlib.sha256(gt.read_bytes()).hexdigest() == entry.ground_truth_sha256


# ------------------------------------------------------- unrelated / copied


def test_unrelated_pdf_is_blocked_even_inside_the_fixture_folder(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    register_test_documents(monkeypatch, tmp_path, PROJECT / "tests" / "lab" / "data" / "text.pdf")
    # An arbitrary PDF placed in a folder named exactly like the old trusted one.
    fixtures = tmp_path / "lab" / "fixtures" / "generated"
    fixtures.mkdir(parents=True)
    arbitrary = _new_pdf(fixtures / "real_contract.pdf", "CONFIDENTIAL agreement")
    manager, workspace_id, document, source = _ingest(settings, arbitrary)
    runner = APIEvaluationRunner(manager, settings, store)
    transport = _happy()
    with pytest.raises(APIRunRefused, match="not in the synthetic document registry"):
        _run(runner, spec, workspace_id, document, source, transport)
    assert transport.calls == []
    # The repository's own generated fixtures (trusted by folder before) are refused too.
    repo_fixture = PROJECT / "lab" / "fixtures" / "generated" / "en_single_column" / "document_pymupdf.pdf"
    if repo_fixture.is_file():
        manager, workspace_id, document, source = _ingest(settings, repo_fixture)
        with pytest.raises(APIRunRefused, match="not in the synthetic document registry"):
            APIEvaluationRunner(manager, settings, store).preview(spec, workspace_id, document, source, 1)
    # The runner has no folder-based trust left at all.
    assert "fixtures" not in (PROJECT / "lab" / "core" / "e2" / "api" / "runner.py").read_text(encoding="utf-8")


# ---------------------------------------------- authorization lifecycle


def test_eligible_but_unauthorized_is_blocked_then_authorized_runs_mocked(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    runner = APIEvaluationRunner(manager, settings, store)
    transport = _happy()
    with pytest.raises(APIRunRefused, match="No explicit authorization"):
        _run(runner, spec, workspace_id, document, source, transport)
    assert transport.calls == []
    record_path = authorize(pdf, store.root, days=3)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["document_sha256"] == document.sha256 and record["approved"] is True
    assert record["reviewer"] == "Tester" and record["declaration"] == documents.SYNTHETIC_DECLARATION
    assert record["document"]["filename"] == "text.pdf"
    assert datetime.fromisoformat(record["expires_at"]) > datetime.fromisoformat(record["approved_at"])
    outcome = _run(runner, spec, workspace_id, document, source, transport)
    assert outcome.state == job_store.COMPLETED
    # The worker sent the write-once copy of the confirmed bytes inside the run.
    sent = outcome.directory / "transmitted" / "page_0001.png"
    assert sent.is_file() and image_in_body(sent, transport)
    final = json.loads((outcome.directory / "final.json").read_text(encoding="utf-8"))
    assert "transmitted/page_0001.png" in final["files_sha256"]


def image_in_body(image: Path, transport) -> bool:
    return image.read_bytes() in transport.calls[0]["body"]


def test_revoked_or_expired_authorization_is_refused(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    runner = APIEvaluationRunner(manager, settings, store)
    authorize(pdf, store.root)
    documents.revoke_document(document.sha256, revoked_by="Tester", reason="test", root=store.root)
    with pytest.raises(APIRunRefused, match="revoked"):
        runner.preview(spec, workspace_id, document, source, 1)
    # Expired: a correctly formed approval whose expiry has passed.
    entry = documents.load_registry()[document.sha256]
    past = datetime.now(timezone.utc) - timedelta(days=2)
    documents._write_once(documents._auth_dir(store.root, document.sha256), {
        "schema_version": 1, "document_sha256": document.sha256, "document": entry.identity(),
        "approved": True, "reviewer": "Tester", "declaration": documents.SYNTHETIC_DECLARATION,
        "approved_at": past.isoformat(), "expires_at": (past + timedelta(days=1)).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    with pytest.raises(APIRunRefused, match="expired"):
        runner.preview(spec, workspace_id, document, source, 1)
    # Full audit trail kept: approval, revocation, expired approval.
    assert len(documents.authorization_records(document.sha256, store.root)) == 3


def test_authorization_requires_declaration_matching_bytes_and_registry(env, tmp_path) -> None:
    settings, _tmp, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    entry = documents.load_registry()[hashlib.sha256(pdf.read_bytes()).hexdigest()]
    kwargs = {"file_path": pdf, "reviewer": "Tester", "days_valid": 1, "root": store.root}
    with pytest.raises(documents.DocumentAuthorizationError, match="declaration"):
        documents.authorize_document(entry, declaration="yes, synthetic", **kwargs)
    other = _new_pdf(tmp_path / "other.pdf", "different bytes")
    with pytest.raises(documents.DocumentAuthorizationError, match="do not match"):
        documents.authorize_document(entry, declaration=documents.SYNTHETIC_DECLARATION,
                                     **(kwargs | {"file_path": other}))
    forged = entry.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(documents.DocumentAuthorizationError, match="not an eligible"):
        documents.authorize_document(forged, declaration=documents.SYNTHETIC_DECLARATION, **kwargs)
    with pytest.raises(documents.DocumentAuthorizationError, match="1..30"):
        documents.authorize_document(entry, declaration=documents.SYNTHETIC_DECLARATION, **(kwargs | {"days_valid": 31}))


# ---------------------------------------------------------- modified bytes


def test_modified_bytes_are_rejected_before_transmission(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    authorize(pdf, store.root)
    runner = APIEvaluationRunner(manager, settings, store)
    preview = runner.preview(spec, workspace_id, document, source, 1)
    # The workspace copy is altered after authorization and confirmation.
    source.write_bytes(source.read_bytes() + b"\n%tampered")
    transport = _happy()
    with pytest.raises(APIRunRefused, match="modified bytes"):
        runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                   confirmed_sha256=preview["file"]["sha256"], executor=_in_process(transport))
    assert transport.calls == []
    # A registry entry edited after authorization invalidates the approval.
    shutil.copyfile(pdf, source)
    registry_text = documents.DOCUMENT_REGISTRY.read_text(encoding="utf-8").replace('title = "test fixture"', 'title = "edited"')
    documents.DOCUMENT_REGISTRY.write_text(registry_text, encoding="utf-8")
    with pytest.raises(APIRunRefused, match="registry entry changed"):
        verify_document_for_transmission(document, source, root=store.root)


def test_changed_rendered_page_after_confirmation_is_rejected(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    authorize(pdf, store.root)
    runner = APIEvaluationRunner(manager, settings, store)
    real_preview = runner.preview

    def tampering_preview(*args):
        result = real_preview(*args)
        Path(result["image_path"]).write_bytes(b"\x89PNG\r\n\x1a\nnot the confirmed page")
        return result

    runner.preview = tampering_preview
    transport = _happy()
    confirmed = real_preview(spec, workspace_id, document, source, 1)["file"]["sha256"]
    with pytest.raises(APIRunRefused, match="changed after confirmation"):
        runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                   confirmed_sha256=confirmed, executor=_in_process(transport))
    assert transport.calls == []


# ------------------------------------------------------------- concurrency


def test_concurrent_requests_cannot_bypass_authorization(env) -> None:
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    runner = APIEvaluationRunner(manager, settings, store)
    transport = _happy()
    refused: list[str] = []
    barrier = threading.Barrier(10)

    def attempt() -> None:
        barrier.wait()
        try:
            _run(runner, spec, workspace_id, document, source, transport)
        except APIRunRefused as exc:
            refused.append(str(exc))

    threads = [threading.Thread(target=attempt) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(refused) == 10 and transport.calls == []  # unauthorized: nobody gets through


def test_revocation_between_preview_and_send_is_enforced(env) -> None:
    """A revocation that lands after the preview but before the locked send check wins."""
    settings, tmp_path, store, spec, monkeypatch = env
    pdf = PROJECT / "tests" / "lab" / "data" / "text.pdf"
    register_test_documents(monkeypatch, tmp_path, pdf)
    manager, workspace_id, document, source = _ingest(settings, pdf)
    authorize(pdf, store.root)
    runner = APIEvaluationRunner(manager, settings, store)
    real_preview = runner.preview

    def preview_then_revoke(*args):
        result = real_preview(*args)
        documents.revoke_document(document.sha256, revoked_by="Other session", reason="race", root=store.root)
        return result

    confirmed = real_preview(spec, workspace_id, document, source, 1)["file"]["sha256"]
    runner.preview = preview_then_revoke
    transport = _happy()
    with pytest.raises(APIRunRefused, match="revoked"):
        runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                   confirmed_sha256=confirmed, executor=_in_process(transport))
    assert transport.calls == [] and not (store.root / "api_jobs").exists()


# ------------------------------------------------------ the real benchmarks


@pytest.mark.parametrize("entry_id", sorted(BENCHMARKS))
def test_real_benchmark_blocked_until_authorized_then_runs_mocked(env, entry_id) -> None:
    path = BENCHMARKS[entry_id]
    if not path.is_file():
        pytest.skip(f"{path.name} not present on this machine")
    settings, _tmp, store, spec, _monkeypatch = env  # uses the SHIPPED registry
    manager, workspace_id, document, source = _ingest(settings, path)
    runner = APIEvaluationRunner(manager, settings, store)
    with pytest.raises(APIRunRefused, match="No explicit authorization"):
        runner.preview(spec, workspace_id, document, source, 1)
    entry = documents.load_registry()[document.sha256]
    assert entry.id == entry_id
    documents.authorize_document(entry, file_path=source, reviewer="Tester",
                                 declaration=documents.SYNTHETIC_DECLARATION, days_valid=1, root=store.root)
    transport = _happy()
    outcome = _run(runner, spec, workspace_id, document, source, transport)
    assert outcome.state == job_store.COMPLETED and len(transport.calls) == 3  # mocked only
