"""Full-PDF hosted API runs (simple OCR/VLM Evaluation view).

No network: the API worker is replaced by an in-process fake executor that records
exactly which page images it was given.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import fitz
import pytest

import lab.core.e2.api.runner as api_runner
from lab.core.e2.api import documents, full_pdf
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import REQUIRED_STATEMENT, record_enablement
from lab.core.e2.api.providers import load_providers
from lab.core.e2.api.runner import APIEvaluationRunner
from lab.core.e2.store import ResultStore
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager

KEY = "SECRET-KEY-full-pdf-7c1e"
OXEN = "oxen-qwen3-vl-4b-instruct"
PAGES = 3


def _pdf(path: Path, pages: int = PAGES) -> Path:
    document = fitz.open()
    for number in range(1, pages + 1):
        page = document.new_page()
        page.insert_text((72, 72), f"Synthetic test page {number}", fontsize=14)
    document.save(path)
    document.close()
    return path


def _ok(text: str, *, finish: str = "stop", truncated: bool = False) -> dict:
    return {"phase": "done", "ok": True, "job_id": f"chatcmpl-{hashlib.sha1(text.encode()).hexdigest()[:8]}",
            "result": {"text": text, "blocks": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                       "finish_reason": finish, "truncated": truncated, "reported_model": "qwen3-vl-4b-instruct",
                       "exchanges": []}}


UNCONFIRMED = {"phase": "error", "ok": False, "job_id": None, "resumable": False,
               "error": {"code": "request_timeout", "message": "no response within the timeout"}, "exchanges": []}
FAILED = {"phase": "error", "ok": False, "job_id": None, "resumable": False,
          "error": {"code": "http_400", "message": "bad request"}, "exchanges": []}


class FakeExecutor:
    """Returns scripted outcomes in order and records every image it was given."""

    def __init__(self, outcomes: list[dict]):
        self.outcomes = list(outcomes)
        self.sent: list[str] = []

    def factory(self):
        def execute(spec, credential, image_path, filename, job_id, result_hosts, on_event):
            assert credential == KEY
            self.sent.append(hashlib.sha256(Path(image_path).read_bytes()).hexdigest())
            return self.outcomes.pop(0)
        return execute


@pytest.fixture
def env(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setattr(documents, "DOCUMENT_REGISTRY", tmp_path / "no_registry.toml")
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)  # the page builds its own settings
    manager = WorkspaceManager(settings)
    workspace_id = manager.create().workspace_id
    ingest = IngestService(manager, settings)
    pdf = _pdf(tmp_path / "My_Test_Report.pdf")
    with pdf.open("rb") as handle:
        document = ingest.ingest(workspace_id, handle, pdf.name)
    source = ingest.original_path(workspace_id, document.doc_id)
    store = ResultStore(tmp_path / "results")
    spec = next(item for item in load_providers() if item.id == OXEN)
    runner = APIEvaluationRunner(manager, settings, store)
    return {"manager": manager, "workspace_id": workspace_id, "document": document, "source": source,
            "store": store, "spec": spec, "runner": runner, "monkeypatch": monkeypatch, "tmp": tmp_path}


def _register_and_authorize(env) -> None:
    documents.register_document(env["source"], display_name=env["document"].display_name, registered_by="Tester",
                                statement=documents.REGISTRATION_STATEMENT, root=env["store"].root)
    entry = documents.load_registry(root=env["store"].root)[env["document"].sha256]
    documents.authorize_document(entry, file_path=env["source"], reviewer="Tester",
                                 declaration=documents.USER_OWNED_DECLARATION, days_valid=1, root=env["store"].root)


def _enable(env, pages: int = 10) -> None:
    record_enablement(env["spec"], enabled=True, enabled_by="Tester", statement=REQUIRED_STATEMENT,
                      max_pages_total=pages, max_pages_per_day=pages, days_valid=1, root=env["store"].root)


def _start(env, outcomes: list[dict]) -> tuple[Path, FakeExecutor, dict]:
    fake = FakeExecutor(outcomes)
    env["monkeypatch"].setattr(api_runner, "subprocess_executor", fake.factory)
    plan, _images = full_pdf.build_plan(env["runner"], env["spec"], env["workspace_id"], env["document"],
                                        env["source"])
    assert fake.sent == []  # preparing the plan never transmits anything
    run_dir = full_pdf.create_run(env["store"].root, env["spec"], env["document"], plan, approved_by="Tester",
                                  statement=full_pdf.APPROVAL_STATEMENT, render_dpi=200)
    assert fake.sent == []  # creating the run never transmits anything
    return run_dir, fake, plan


def _run(env, run_dir: Path) -> dict:
    return full_pdf.run_pending(run_dir, env["runner"], env["spec"], workspace_id=env["workspace_id"],
                                document=env["document"], pdf_path=env["source"], credential=KEY,
                                root=env["store"].root)


def _all_bytes(root: Path) -> bytes:
    return b"".join(path.read_bytes() for path in root.rglob("*") if path.is_file())


# ---------------------------------------------------------------- registration


def test_ui_registration_makes_a_new_pdf_eligible_but_not_authorized(env) -> None:
    root = env["store"].root
    sha = env["document"].sha256
    assert sha not in documents.load_registry(root=root)
    with pytest.raises(documents.DocumentAuthorizationError):
        documents.register_document(env["source"], display_name="x.pdf", registered_by="Tester",
                                    statement="something else", root=root)
    entry = documents.register_document(env["source"], display_name=env["document"].display_name,
                                        registered_by="Tester", statement=documents.REGISTRATION_STATEMENT, root=root)
    assert entry.sha256 == sha and entry.page_count == PAGES
    assert entry.authorization_category == documents.CATEGORY_USER_OWNED
    assert documents.load_registry(root=root)[sha] == entry
    state = documents.authorization_state(sha, root=root)
    assert not state["allowed"] and "No explicit authorization" in " ".join(state["reasons"])
    # The synthetic declaration is never accepted for a UI-registered document.
    with pytest.raises(documents.DocumentAuthorizationError):
        documents.authorize_document(entry, file_path=env["source"], reviewer="Tester",
                                     declaration=documents.SYNTHETIC_DECLARATION, days_valid=1, root=root)
    documents.authorize_document(entry, file_path=env["source"], reviewer="Tester",
                                 declaration=documents.USER_OWNED_DECLARATION, days_valid=1, root=root)
    assert documents.authorization_state(sha, root=root)["allowed"]
    documents.withdraw_registration(sha, withdrawn_by="Tester", reason="test", root=root)
    assert sha not in documents.load_registry(root=root)
    assert not documents.authorization_state(sha, root=root)["allowed"]


def test_explicit_registry_path_ignores_ui_registrations(env, tmp_path) -> None:
    documents.register_document(env["source"], display_name="r.pdf", registered_by="Tester",
                                statement=documents.REGISTRATION_STATEMENT, root=env["store"].root)
    assert documents.load_registry(tmp_path / "missing.toml", root=env["store"].root) == {}


def test_non_pdf_cannot_be_registered(env, tmp_path) -> None:
    other = tmp_path / "not.pdf"
    other.write_bytes(b"hello")
    with pytest.raises(documents.DocumentAuthorizationError):
        documents.register_document(other, display_name="not.pdf", registered_by="Tester",
                                    statement=documents.REGISTRATION_STATEMENT, root=env["store"].root)


# ----------------------------------------------------------------- full runs


def test_full_run_sends_every_page_once_in_order_and_saves_the_folder(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    texts = ["الصفحة الأولى ٢٠٢٥\nline 2", "Page two 1,284,330", "  page three  \n"]
    run_dir, fake, plan = _start(env, [_ok(text) for text in texts])
    summary = _run(env, run_dir)
    assert fake.sent == [item["image_sha256"] for item in plan["pages"]]
    assert summary["counts"]["Success"] == PAGES and summary["finalized"]
    assert run_dir.parent.parent.name == "qwen3_vl" and run_dir.parent.name == "My_Test_Report"
    assert run_dir.name.startswith("run_") and run_dir.name.endswith("_001")
    for number, text in enumerate(texts, start=1):
        assert (run_dir / f"page_{number:03d}.txt").read_bytes() == text.encode("utf-8")  # exact, unmodified
        record = json.loads((run_dir / f"page_{number:03d}.json").read_text(encoding="utf-8"))
        assert record["status"] == "Success" and record["tokens"]["prompt"] == 100
        assert record["model_settings"]["prompt_id"] == env["spec"].prompt_id
        assert record["finish_reason"] == "stop" and record["truncated_at_token_limit"] is False
        assert "repetition" in record["indicators"]
    full_text = (run_dir / "full_text.txt").read_text(encoding="utf-8")
    assert full_text.index("Page 1 / 3") < full_text.index("Page 2 / 3") < full_text.index("Page 3 / 3")
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    assert results["finalized"] and [item["text"] for item in results["pages"]] == texts
    assert "page_001.txt" in results["files_sha256"]
    report = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "Page 3" in report and "Page two 1,284,330" in report
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["provider"]["provider_id"] == OXEN and metadata["settings"]["generation"]
    assert metadata["settings"]["prompt"]["sha256"] == plan["prompt"]["sha256"]
    approval = json.loads((run_dir / "approval.json").read_text(encoding="utf-8"))
    assert approval["plan_sha256"] == plan["plan_sha256"] and len(approval["image_sha256"]) == PAGES
    # The key is never written anywhere under the results root.
    assert KEY.encode() not in _all_bytes(env["store"].root)


def test_unconfirmed_page_is_never_resent_and_later_pages_continue(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, fake, _plan = _start(env, [_ok("one"), UNCONFIRMED, _ok("three")])
    summary = _run(env, run_dir)
    statuses = [item["status"] for item in summary["records"]]
    assert statuses == ["Success", "Uncertain", "Success"] and len(fake.sent) == 3
    assert summary["records"][1]["job_state"] == job_store.SUBMISSION_UNCONFIRMED
    assert not summary["finalized"] and not (run_dir / "page_002.txt").exists()
    # Continuing the run never resubmits the unconfirmed page.
    again = _run(env, run_dir)
    assert len(fake.sent) == 3 and again["records"][1]["job_state"] == job_store.SUBMISSION_UNCONFIRMED


def test_two_consecutive_problems_stop_the_run(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, fake, _plan = _start(env, [UNCONFIRMED, FAILED])
    summary = _run(env, run_dir)
    assert len(fake.sent) == 2 and "consecutive" in summary["stopped"]
    assert summary["records"][2]["status"] == "Not sent"


def test_explicit_retry_abandons_the_unconfirmed_job_and_sends_once(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, fake, _plan = _start(env, [_ok("one"), UNCONFIRMED, _ok("three"), _ok("two, retried")])
    _run(env, run_dir)
    with pytest.raises(full_pdf.FullPDFError):
        full_pdf.retry_page(run_dir, env["runner"], env["spec"], 2, workspace_id=env["workspace_id"],
                            document=env["document"], pdf_path=env["source"], credential=KEY,
                            root=env["store"].root, by="Tester", statement="no")
    with pytest.raises(full_pdf.FullPDFError):  # completed pages are never resent
        full_pdf.retry_page(run_dir, env["runner"], env["spec"], 1, workspace_id=env["workspace_id"],
                            document=env["document"], pdf_path=env["source"], credential=KEY,
                            root=env["store"].root, by="Tester", statement=full_pdf.RETRY_STATEMENT)
    record = full_pdf.retry_page(run_dir, env["runner"], env["spec"], 2, workspace_id=env["workspace_id"],
                                 document=env["document"], pdf_path=env["source"], credential=KEY,
                                 root=env["store"].root, by="Tester", statement=full_pdf.RETRY_STATEMENT)
    assert record["status"] == "Success" and record["attempts"] == 2 and len(fake.sent) == 4
    assert (run_dir / "page_002.txt").read_text(encoding="utf-8") == "two, retried"


def test_truncated_output_is_saved_and_marked_uncertain(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, _fake, _plan = _start(env, [_ok("a"), _ok("cut off", finish="length", truncated=True), _ok("c")])
    summary = _run(env, run_dir)
    record = summary["records"][1]
    assert record["status"] == "Uncertain" and "token limit" in record["status_reason"]
    assert (run_dir / "page_002.txt").read_text(encoding="utf-8") == "cut off"
    assert summary["finalized"]  # every job completed; the indicator does not block finalization


def test_run_requires_the_exact_approval_and_enough_usage_limit(env) -> None:
    _register_and_authorize(env)
    _enable(env, pages=2)  # fewer than the 3 pages of the PDF
    plan, _images = full_pdf.build_plan(env["runner"], env["spec"], env["workspace_id"], env["document"],
                                        env["source"])
    with pytest.raises(full_pdf.FullPDFError, match="statement"):
        full_pdf.create_run(env["store"].root, env["spec"], env["document"], plan, approved_by="Tester",
                            statement="I agree")
    with pytest.raises(full_pdf.FullPDFError, match="Usage limit"):
        full_pdf.create_run(env["store"].root, env["spec"], env["document"], plan, approved_by="Tester",
                            statement=full_pdf.APPROVAL_STATEMENT)
    tampered = dict(plan, endpoint="https://elsewhere.example")
    _enable(env)
    with pytest.raises(full_pdf.FullPDFError, match="modified"):
        full_pdf.create_run(env["store"].root, env["spec"], env["document"], tampered, approved_by="Tester",
                            statement=full_pdf.APPROVAL_STATEMENT)
    assert not (env["store"].root / "qwen3_vl").exists()  # a refused run creates no folder


def test_unauthorized_document_cannot_be_planned(env) -> None:
    _enable(env)
    with pytest.raises(api_runner.APIRunRefused):
        full_pdf.build_plan(env["runner"], env["spec"], env["workspace_id"], env["document"], env["source"])


def test_runs_are_write_once_and_numbered(env) -> None:
    _register_and_authorize(env)
    _enable(env, pages=20)
    first, _fake, _plan = _start(env, [_ok("1"), _ok("2"), _ok("3")])
    _run(env, first)
    snapshot = {path: path.read_bytes() for path in first.rglob("*") if path.is_file() and path.suffix != ".lock"}
    second, _fake2, _plan2 = _start(env, [_ok("x"), _ok("y"), _ok("z")])
    _run(env, second)
    assert second.name.endswith("_002") and second.parent == first.parent
    assert {path: path.read_bytes() for path in snapshot} == snapshot  # first run untouched
    with pytest.raises(full_pdf.FullPDFError):
        full_pdf.finalize(first, env["spec"], root=env["store"].root)
    with pytest.raises(full_pdf.FullPDFError):
        _run(env, first)  # a finalized run never sends again


def test_finalize_with_open_jobs_requires_explicit_abandon(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, _fake, _plan = _start(env, [_ok("one"), UNCONFIRMED, _ok("three")])
    _run(env, run_dir)
    with pytest.raises(full_pdf.FullPDFError, match="unfinished"):
        full_pdf.finalize(run_dir, env["spec"], root=env["store"].root, runner=env["runner"], abandon_by="Tester")
    full_pdf.finalize(run_dir, env["spec"], root=env["store"].root, runner=env["runner"], abandon_by="Tester",
                      statement=full_pdf.ABANDON_STATEMENT)
    record = json.loads((run_dir / "page_002.json").read_text(encoding="utf-8"))
    assert record["status"] == "Failed" and record["text_file"] is None
    assert "[no text:" in (run_dir / "full_text.txt").read_text(encoding="utf-8")


def test_zip_contains_the_folder_and_provisional_outputs_without_the_key(env) -> None:
    _register_and_authorize(env)
    _enable(env)
    run_dir, _fake, _plan = _start(env, [_ok("one"), UNCONFIRMED, _ok("three")])
    _run(env, run_dir)
    data = full_pdf.zip_bytes(run_dir, env["spec"], root=env["store"].root)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert {f"{run_dir.name}/{name}" for name in ("metadata.json", "plan.json", "approval.json",
                                                         "page_001.txt", "results.json", "full_text.txt",
                                                         "report.html")} <= names
        assert json.loads(archive.read(f"{run_dir.name}/results.json"))["finalized"] is False
        assert all(KEY.encode() not in archive.read(name) for name in names)
    assert not (run_dir / "results.json").exists()  # provisional outputs are never written


# ------------------------------------------------------------------------ UI


def test_page_defaults_to_the_simple_view_and_sends_nothing_on_open(env) -> None:
    from streamlit.testing.v1 import AppTest

    fake = FakeExecutor([])
    env["monkeypatch"].setattr(api_runner, "subprocess_executor", fake.factory)
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "7_OCR_VLM_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=60)
    app.session_state["workspace_id"] = env["workspace_id"]
    app.run()
    assert not app.exception
    assert next(item for item in app.radio if item.label == "View").value == "Full PDF run (hosted API)"
    assert any(item.label.startswith("Register this PDF") for item in app.button)
    assert not any(item.label == "Run Full PDF" for item in app.button)  # nothing to run before authorization
    _register_and_authorize(env)
    _enable(env)
    app.run()
    assert not app.exception
    prepare = next(item for item in app.button if item.label.startswith("Prepare the page images"))
    prepare.click().run()
    assert not app.exception
    run = next(item for item in app.button if item.label == "Run Full PDF")
    assert run.disabled  # no key, no name, no approval yet
    assert fake.sent == []
    assert not (env["store"].root / "qwen3_vl").exists()
