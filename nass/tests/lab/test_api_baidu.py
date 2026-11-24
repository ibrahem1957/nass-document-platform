"""E2-C hosted API integration tests. Every provider response is mocked: no test
opens a network connection."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lab.core.e2.airgap import OutboundNetworkBlocked
from lab.core.e2.api import API_INTEGRATION_ENABLED_BY_DEFAULT
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import (
    REQUIRED_STATEMENT,
    _append_usage_unlocked,
    _check_authorization_unlocked,
    AuthorizationError,
    check_authorization,
    provider_usage_lock,
    record_enablement,
    usage_records,
)
from lab.core.e2.api.client import (
    APIRequestError,
    BaiduPaddleOCRJobsClient,
    HTTPResponse,
    ResultHostNotApproved,
    UrllibTransport,
    allowlist_network_guard,
    redact,
    sanitize_url,
    validate_result_url,
)
from lab.core.e2.api.providers import ProviderSpec, load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused, subprocess_executor
from lab.core.e2.api.worker import execute_job
from lab.core.e2.store import ResultStore
from tests.lab.api_doc_helpers import authorize, register_test_documents
from tests.lab.test_e2a import _workspace_with_gt

TOKEN = "SECRET-TOKEN-do-not-store-7f3a9c"
RESULT_HOST = "results.example.test"
SIGNED = "SIGNEDVALUE9f2"


def _spec(**update) -> ProviderSpec:
    spec = {item.id: item for item in load_providers()}["baidu-aistudio-paddleocr-vl-1.6"]
    data = spec.model_dump() | {"result_hosts": [RESULT_HOST]} | update
    return ProviderSpec.model_validate(data)


def _json(payload: dict, status: int = 200, headers: dict | None = None) -> HTTPResponse:
    return HTTPResponse(status, {"Content-Type": "application/json", "X-Request-Id": "req-1"} | (headers or {}),
                        json.dumps(payload).encode(), 5.0)


def _jsonl(markdown: str) -> HTTPResponse:
    line = {"result": {"dataInfo": {"numPages": 1},
                       "layoutParsingResults": [{
                           "markdown": {"text": markdown, "images": {}},
                           "prunedResult": {"parsing_res_list": [
                               {"block_label": "text", "block_content": markdown, "block_order": 0,
                                "block_bbox": [1, 2, 3, 4]}]},
                           "outputImages": {"layout": f"https://{RESULT_HOST}/viz.png?sig={SIGNED}"}}]}}
    return HTTPResponse(200, {"Content-Type": "application/jsonl"}, (json.dumps(line) + "\n").encode(), 7.0)


def _done(host: str = RESULT_HOST, query: str = f"X-Sig={SIGNED}") -> HTTPResponse:
    return _json({"code": 0, "data": {"state": "done", "resultUrl": {
        "jsonUrl": f"https://{host}/r/job-123.jsonl?{query}",
        "markdownUrl": f"https://{host}/r/job-123.md?{query}"}}})


SUBMIT = _json({"code": 0, "data": {"jobId": "job-123"}})
RUNNING = _json({"code": 0, "data": {"state": "running"}})


class ScriptedTransport:
    """Returns queued responses and records every request (method, url, headers, body)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, *, headers, body, timeout, max_bytes):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body or b""})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def submissions(self) -> int:
        return sum(1 for call in self.calls if call["method"] == "POST")


def _happy(markdown: str = "page 1 Arabic ٢٠٢٥ and 1,284,330"):
    return ScriptedTransport([SUBMIT, RUNNING, _done(), _jsonl(markdown)])


def _in_process(transport, clock=None, wall_clock=None):
    def execute(spec, credential, image_path, filename, job_id, result_hosts, on_event):
        extra = {}
        if clock:
            extra["clock"] = clock
        if wall_clock:
            extra["wall_clock"] = wall_clock
        client = BaiduPaddleOCRJobsClient(spec, credential, transport, result_hosts=result_hosts,
                                          sleep=lambda _s: None, **extra)
        return execute_job(client, image_path=image_path, filename=filename, job_id=job_id, emit=on_event)

    return execute


@pytest.fixture
def lab(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    store = ResultStore(tmp_path / "results")
    # Exact-hash authorization, as a user would grant it (folders confer nothing).
    register_test_documents(monkeypatch, tmp_path, source)
    authorize(source, store.root)
    runner = APIEvaluationRunner(manager, settings, store)
    return runner, store, workspace_id, document, source, ground_truth


def _enable(spec, store, **kw):
    params = {"enabled": True, "enabled_by": "Tester", "statement": REQUIRED_STATEMENT,
              "max_pages_total": 10, "max_pages_per_day": 10, "days_valid": 1} | kw
    return record_enablement(spec, root=store.root, **params)


def _submit(runner, spec, workspace_id, document, source, transport, ground_truth=None, **kw):
    preview = runner.preview(spec, workspace_id, document, source, 1)
    return runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                      confirmed_sha256=preview["file"]["sha256"], ground_truth=ground_truth,
                      executor=_in_process(transport, **kw))


def _all_stored_text(root: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file())


# ------------------------------------------------------------------ registry


def test_registry_has_baidu_implemented_and_others_planned_only() -> None:
    specs = {item.id: item for item in load_providers()}
    baidu = specs["baidu-aistudio-paddleocr-vl-1.6"]
    assert API_INTEGRATION_ENABLED_BY_DEFAULT is False
    assert baidu.runnable and baidu.api_model_id == "PaddleOCR-VL-1.6"
    assert baidu.endpoint == "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    assert baidu.exact_revision_verified is False
    assert baidu.classification == "SAME MODEL FAMILY, REVISION UNVERIFIED"
    assert baidu.result_hosts == []  # never pre-approved: the user reviews the real host
    assert specs["featherless-qwen3-vl-4b-instruct"].runnable  # implemented in E2-C phase 2
    assert not specs["dedicated-qari-ocr-0.4.0"].runnable  # still planned
    with pytest.raises(ValueError, match="EXACT MATCH VERIFIED requires"):
        ProviderSpec.model_validate(baidu.model_dump() | {"classification": "EXACT MATCH VERIFIED"})
    with pytest.raises(ValueError, match="https"):
        ProviderSpec.model_validate(baidu.model_dump() | {"api_base_url": "http://paddleocr.aistudio-app.com"})


# ------------------------------------------------------------- authorization


def test_disabled_by_default_and_planned_providers_cannot_be_enabled(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    state = check_authorization(spec, pages=1, root=store.root)
    assert not state.allowed and any("disabled by default" in reason for reason in state.reasons)
    transport = _happy()
    with pytest.raises(APIRunRefused, match="disabled"):
        _submit(runner, spec, workspace_id, document, source, transport)
    assert transport.calls == [] and not (store.root / "runs").exists()
    planned = {item.id: item for item in load_providers()}["dedicated-qari-ocr-0.4.0"]
    with pytest.raises(AuthorizationError, match="planned"):
        _enable(planned, store)
    with pytest.raises(AuthorizationError, match="statement"):
        _enable(spec, store, statement="yes")


def test_revocation_expiry_and_endpoint_change_disable_the_provider(lab) -> None:
    _runner, store, *_ = lab
    spec = _spec()
    _enable(spec, store)
    assert check_authorization(spec, pages=1, root=store.root).allowed
    moved = _spec(api_base_url="https://other.example.test")
    assert any("endpoint" in reason for reason in check_authorization(moved, pages=1, root=store.root).reasons)
    time.sleep(0.01)
    record_enablement(spec, enabled=False, enabled_by="Tester", statement="", max_pages_total=0,
                      max_pages_per_day=0, days_valid=1, root=store.root)
    assert not check_authorization(spec, pages=1, root=store.root).allowed
    assert len(list((store.root / "api_authorizations" / spec.id).glob("*.json"))) == 2


# ----------------------------------------------------------- happy path


def test_one_confirmed_page_is_sent_scored_and_stored_write_once(lab) -> None:
    runner, store, workspace_id, document, source, ground_truth = lab
    spec = _spec()
    _enable(spec, store)
    preview = runner.preview(spec, workspace_id, document, source, 1)
    transport = _happy()
    outcome = _submit(runner, spec, workspace_id, document, source, transport, ground_truth)
    run, directory = outcome.run, outcome.directory
    assert outcome.state == job_store.COMPLETED
    assert run.status == "ok" and run.adapter.adapter_id == "api.baidu-aistudio-paddleocr-vl-1.6"
    assert run.adapter.kind == "api" and run.pages[0].text.startswith("page 1 Arabic")
    assert run.usage.cost_source == "estimated" and run.usage.cost_usd == 0.0
    assert run.environment["api"]["job_id"] == "job-123"
    assert run.environment["api"]["job_states"] == ["SUBMITTING", "SUBMITTED", "COMPLETED"]
    submit = transport.calls[0]
    image = Path(preview["image_path"]).read_bytes()
    assert hashlib.sha256(image).hexdigest() == preview["file"]["sha256"] and image in submit["body"]
    assert b'name="model"\r\n\r\nPaddleOCR-VL-1.6' in submit["body"]
    assert b'name="optionalPayload"\r\n\r\n{}' in submit["body"]
    assert submit["headers"]["Authorization"] == f"Bearer {TOKEN}"
    result_call = transport.calls[-1]
    assert result_call["url"].startswith(f"https://{RESULT_HOST}/") and "Authorization" not in result_call["headers"]
    final = json.loads((directory / "final.json").read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["summaries"][0]
    assert summary["kind"] == "hosted API" and summary["metrics"]["ref_chars"] > 0
    stored = _all_stored_text(store.root)
    assert TOKEN not in stored and SIGNED not in stored  # no secret, no signed query value
    assert run.environment["e2"]["page_raw"]["1"]["response"]["result_url"]["host"] == RESULT_HOST
    assert [entry["submitted"] for entry in usage_records(spec, root=store.root)] == [True, False]


def test_ground_truth_and_prior_outputs_never_reach_the_provider(lab) -> None:
    runner, store, workspace_id, document, source, ground_truth = lab
    spec = _spec()
    _enable(spec, store)
    canary = "CANARY-GROUND-TRUTH-9d1e"
    ground_truth = ground_truth.model_copy(
        update={"pages": [page.model_copy(update={"text": canary + page.text}) for page in ground_truth.pages]}
    )
    transport = _happy()
    _submit(runner, spec, workspace_id, document, source, transport, ground_truth)
    sent = b"".join(call["body"] for call in transport.calls) + "".join(call["url"] for call in transport.calls).encode()
    assert canary.encode() not in sent


def test_hosted_mode_rejects_non_allowlisted_or_replaced_pdf(lab, tmp_path) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    unknown = tmp_path / "unknown.pdf"
    unknown.write_bytes(source.read_bytes() + b"\n% synthetic marker absent\n")
    unknown_sha = hashlib.sha256(unknown.read_bytes()).hexdigest()
    unknown_doc = document.model_copy(update={"sha256": unknown_sha})
    with pytest.raises(APIRunRefused, match="not in the synthetic document registry"):
        runner.preview(spec, workspace_id, unknown_doc, unknown, 1)
    with pytest.raises(APIRunRefused, match="no longer matches"):
        runner.preview(spec, workspace_id, document, unknown, 1)


def test_confirmation_must_match_the_bytes_that_would_be_sent(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    transport = _happy()
    with pytest.raises(APIRunRefused, match="confirmed image"):
        runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                   confirmed_sha256="0" * 64, executor=_in_process(transport))
    with pytest.raises(APIRunRefused, match="Enter the"):
        runner.run(spec, workspace_id, document, source, 1, credential="  ",
                   confirmed_sha256="0" * 64, executor=_in_process(transport))
    assert transport.calls == [] and usage_records(spec, root=store.root) == []


# ------------------------------------------------ pause / approve / resume


def test_unapproved_result_host_pauses_then_resumes_after_approval_without_resubmitting(lab) -> None:
    runner, store, workspace_id, document, source, ground_truth = lab
    spec = _spec(result_hosts=[])  # the real registry state
    _enable(spec, store)
    transport = ScriptedTransport([SUBMIT, RUNNING, _done(host="bj.bcebos.example.com")])
    paused = _submit(runner, spec, workspace_id, document, source, transport, ground_truth)
    assert paused.state == job_store.AWAITING_RESULT_HOST_APPROVAL and paused.run is None
    journal = paused.journal
    summary = journal.summary()
    assert summary["job_id"] == "job-123" and summary["awaiting_host"] == "bj.bcebos.example.com"
    assert not any(call["url"].startswith("https://bj.bcebos") for call in transport.calls)  # nothing fetched
    assert not (paused.directory / "final.json").exists()  # not a terminal failure
    assert SIGNED not in _all_stored_text(store.root)
    # Resume is refused until the exact host is explicitly approved.
    with pytest.raises(APIRunRefused, match="Approve the result host"):
        runner.resume(spec, journal.journal_id, credential=TOKEN, executor=_in_process(ScriptedTransport([])))
    with pytest.raises(job_store.JobJournalError, match="exactly the reported host"):
        job_store.approve_result_host(spec, host="evil.example.com", approved_by="Tester",
                                      statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal, root=store.root)
    job_store.approve_result_host(spec, host="bj.bcebos.example.com", approved_by="Tester",
                                  statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal, root=store.root)
    resume_transport = ScriptedTransport([_done(host="bj.bcebos.example.com"), _jsonl("resumed text")])
    done = runner.resume(spec, journal.journal_id, credential=TOKEN, ground_truth=ground_truth,
                         executor=_in_process(resume_transport))
    assert done.state == job_store.COMPLETED and done.run.pages[0].text == "resumed text"
    assert resume_transport.submissions == 0 and transport.submissions == 1  # exactly one submission
    assert [event["state"] for event in journal.events()] == [
        "SUBMITTING", "SUBMITTED", "AWAITING_RESULT_HOST_APPROVAL", "RESUMING", "COMPLETED"]
    assert done.directory == paused.directory and (done.directory / "final.json").is_file()
    assert done.run.run_id == journal.journal_id
    # Usage: the page counted once; the resume added no page.
    assert check_authorization(spec, pages=1, root=store.root).pages_used_total == 1


def test_expired_result_url_is_refreshed_from_the_job_id(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    now = 2_000_000_000.0
    expired = _done(query=f"Expires={int(now) - 60}&X-Sig={SIGNED}")
    fresh = _done(query=f"Expires={int(now) + 900}&X-Sig={SIGNED}")
    transport = ScriptedTransport([SUBMIT, expired, fresh, _jsonl("fresh")])
    outcome = _submit(runner, spec, workspace_id, document, source, transport, wall_clock=lambda: now)
    assert outcome.state == job_store.COMPLETED and outcome.run.pages[0].text == "fresh"
    assert transport.submissions == 1
    assert [call["method"] for call in transport.calls] == ["POST", "GET", "GET", "GET"]
    # Storage rejects an un-dated URL (403): refresh once from the Job ID, then succeed.
    # (The first job is COMPLETED, so a deliberate re-evaluation of the page is allowed.)
    transport = ScriptedTransport([SUBMIT, _done(), HTTPResponse(403, {}, b"expired", 1.0), _done(), _jsonl("again")])
    outcome = _submit(runner, spec, workspace_id, document, source, transport)
    assert outcome.state == job_store.COMPLETED and transport.submissions == 1


def test_still_expired_url_pauses_as_resumable_and_resume_never_resubmits(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    now = 2_000_000_000.0
    stale = _done(query=f"Expires={int(now) - 60}")
    transport = ScriptedTransport([SUBMIT, stale, stale, stale])
    paused = _submit(runner, spec, workspace_id, document, source, transport, wall_clock=lambda: now)
    assert paused.state == job_store.RESUMABLE_ERROR
    assert paused.journal.latest()["error"]["code"] == "result_url_expired"
    resume = ScriptedTransport([_done(query=f"Expires={int(now) + 600}"), _jsonl("ok later")])
    done = runner.resume(spec, paused.journal.journal_id, credential=TOKEN,
                         executor=_in_process(resume, wall_clock=lambda: now))
    assert done.state == job_store.COMPLETED and resume.submissions == 0 and transport.submissions == 1


# ------------------------------------------------- duplicate prevention


def test_unfinished_job_blocks_resubmission_until_abandoned(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(result_hosts=[])
    _enable(spec, store)
    first = ScriptedTransport([SUBMIT, _done(host="storage.example.com")])
    paused = _submit(runner, spec, workspace_id, document, source, first)
    assert paused.state == job_store.AWAITING_RESULT_HOST_APPROVAL
    # Reruns, retries and double clicks: every new submission of this page is refused.
    again = _happy()
    for _attempt in range(3):
        with pytest.raises(APIRunRefused, match="unfinished job already exists"):
            _submit(runner, spec, workspace_id, document, source, again)
    assert again.calls == [] and first.submissions == 1
    with pytest.raises(APIRunRefused, match="name and a reason"):
        runner.abandon(spec, paused.journal.journal_id, by="", reason="")
    abandoned = runner.abandon(spec, paused.journal.journal_id, by="Tester", reason="host not acceptable")
    assert abandoned.state == job_store.ABANDONED
    assert json.loads((abandoned.directory / "final.json").read_text(encoding="utf-8"))["status"] == "failed"
    with pytest.raises(APIRunRefused, match="cannot be resumed"):
        runner.resume(spec, paused.journal.journal_id, credential=TOKEN, executor=_in_process(_happy()))
    # Only after an explicit abandon may the page be submitted again (a new job).
    fresh = ScriptedTransport([SUBMIT, _done(host="storage.example.com")])
    second = _submit(runner, spec, workspace_id, document, source, fresh)
    assert second.journal.journal_id != paused.journal.journal_id


def test_concurrent_reruns_claim_a_page_exactly_once(tmp_path) -> None:
    import threading

    spec = _spec()
    barrier = threading.Barrier(16)
    winners: list[str] = []
    refused: list[str] = []

    def attempt(index: int) -> None:
        barrier.wait()
        try:
            job_store.claim_submission(spec.id, "a" * 64, f"run_{index:02d}", tmp_path)
            winners.append(f"run_{index:02d}")
        except job_store.JobJournalError as exc:
            refused.append(str(exc))

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1 and len(refused) == 15
    # The winner's journal does not exist yet: still blocks a later claim.
    with pytest.raises(job_store.JobJournalError, match="in progress"):
        job_store.claim_submission(spec.id, "a" * 64, "run_late", tmp_path)


def test_concurrent_different_pages_cannot_exceed_provider_quota(tmp_path) -> None:
    import threading

    spec = _spec()
    record_enablement(
        spec, enabled=True, enabled_by="Tester", statement=REQUIRED_STATEMENT,
        max_pages_total=1, max_pages_per_day=1, days_valid=1, root=tmp_path,
    )
    barrier = threading.Barrier(12)
    winners: list[int] = []

    def reserve(index: int) -> None:
        barrier.wait()
        with provider_usage_lock(spec.id, tmp_path):
            state = _check_authorization_unlocked(spec, pages=1, root=tmp_path)
            if not state.allowed:
                return
            _append_usage_unlocked(
                spec,
                {"at": datetime.now(timezone.utc).isoformat(), "run_id": f"r{index}",
                 "pages": 1, "submitted": True},
                root=tmp_path,
            )
            winners.append(index)

    threads = [threading.Thread(target=reserve, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(winners) == 1
    assert check_authorization(spec, pages=1, root=tmp_path).pages_used_total == 1


def test_concurrent_run_index_appends_remain_valid_jsonl(tmp_path) -> None:
    import threading

    store = ResultStore(tmp_path / "results")
    barrier = threading.Barrier(20)

    def append(index: int) -> None:
        barrier.wait()
        store._append_index({"run_id": f"run-{index:02d}", "marker": "x" * 1000})

    threads = [threading.Thread(target=append, args=(index,)) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = [json.loads(line) for line in (store.root / "index.jsonl").read_text().splitlines()]
    assert {item["run_id"] for item in records} == {f"run-{index:02d}" for index in range(20)}


def test_unconfirmed_submission_is_never_resubmitted_or_resumed(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    # Submit timed out before any Job ID: the page may have been accepted.
    timeout = ScriptedTransport([APIRequestError("request_timeout", "Request timed out after 60s")])
    unconfirmed = _submit(runner, spec, workspace_id, document, source, timeout)
    assert unconfirmed.state == job_store.SUBMISSION_UNCONFIRMED
    assert not (unconfirmed.directory / "final.json").exists()
    again = _happy()
    with pytest.raises(APIRunRefused, match="unfinished job already exists"):
        _submit(runner, spec, workspace_id, document, source, again)
    with pytest.raises(APIRunRefused, match="cannot be resumed"):
        runner.resume(spec, unconfirmed.journal.journal_id, credential=TOKEN, executor=_in_process(again))
    assert again.calls == [] and timeout.submissions == 1
    assert runner.abandon(spec, unconfirmed.journal.journal_id, by="Tester",
                          reason="timed out; checked provider console").state == job_store.ABANDONED


# ------------------------------------------------------------- URL safety


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("http://results.example.test/r", "insecure_url"),
        ("https://user:pw@results.example.test/r", "unsafe_result_url"),
        ("https://results.example.test:8443/r", "unsafe_result_url"),
        ("https://10.0.0.5/r", "unsafe_result_url"),
        ("https://[::1]/r", "unsafe_result_url"),
        ("https://localhost/r", "unsafe_result_url"),
        ("https://other.example.test/r", "result_host_not_approved"),
    ],
)
def test_result_url_validation_rejects_unsafe_destinations(url, code) -> None:
    with pytest.raises(APIRequestError) as caught:
        validate_result_url(url, frozenset({RESULT_HOST}))
    assert caught.value.code == code


def test_sanitized_url_keeps_no_query_values_and_reads_expiry() -> None:
    info = sanitize_url(f"https://h.example.com/p/x.jsonl?Expires=2000000000&Signature={SIGNED}&token={TOKEN}")
    assert info["host"] == "h.example.com" and info["path"] == "/p/x.jsonl"
    assert info["query_parameter_names"] == ["Expires", "Signature", "token"]
    assert info["expires_at_epoch"] == 2_000_000_000.0
    assert SIGNED not in json.dumps(info) and TOKEN not in json.dumps(info)
    bce = sanitize_url("https://b.example.com/f?authorization=bce-auth-v1/ak/2026-11-17T10:00:00Z/1800/host/sig")
    assert bce["expires_at"].startswith("2026-11-17T10:30:00")


def test_result_host_approval_is_job_scoped_and_expires(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(result_hosts=[])
    _enable(spec, store)
    paused = _submit(
        runner, spec, workspace_id, document, source,
        ScriptedTransport([SUBMIT, _done(host="store.example.com")]),
    )
    approval = job_store.approve_result_host(
        spec, host="store.example.com", approved_by="Tester",
        statement=job_store.HOST_APPROVAL_STATEMENT, journal=paused.journal, root=store.root,
    )
    assert "store.example.com" in job_store.approved_result_hosts(
        spec, store.root, journal_id=paused.journal.journal_id
    )
    assert "store.example.com" not in job_store.approved_result_hosts(
        spec, store.root, journal_id="different-job"
    )
    record = json.loads(approval.read_text(encoding="utf-8"))
    record["expires_at"] = "2000-01-01T00:00:00+00:00"
    approval.write_text(json.dumps(record), encoding="utf-8")
    assert "store.example.com" not in job_store.approved_result_hosts(
        spec, store.root, journal_id=paused.journal.journal_id
    )


def test_host_approval_never_accepts_ips_or_local_names(lab) -> None:
    for bad in ("10.0.0.5", "127.0.0.1", "localhost", "files.internal", "intranet", "a/b.com"):
        with pytest.raises(job_store.JobJournalError):
            job_store.validate_result_hostname(bad)
    assert job_store.validate_result_hostname("BJ.BCEBOS.Example.com.") == "bj.bcebos.example.com"


def test_redirects_are_validated_and_api_redirects_refused(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(result_hosts=[RESULT_HOST, "cdn.example.test"])
    _enable(spec, store)
    good = ScriptedTransport([SUBMIT, _done(),
                              HTTPResponse(302, {"Location": "https://cdn.example.test/r/final.jsonl?sig=1"}, b"", 1.0),
                              _jsonl("via approved redirect")])
    assert _submit(runner, spec, workspace_id, document, source, good).state == job_store.COMPLETED
    bad = ScriptedTransport([SUBMIT, _done(), HTTPResponse(302, {"Location": "https://169.254.169.254/latest"}, b"", 1.0)])
    paused = _submit(runner, spec, workspace_id, document, source, bad)
    assert paused.state == job_store.RESUMABLE_ERROR
    assert paused.journal.latest()["error"]["code"] == "unsafe_redirect"
    api_redirect = ScriptedTransport([HTTPResponse(307, {"Location": "https://evil.example/jobs"}, b"", 1.0)])
    client = BaiduPaddleOCRJobsClient(spec, TOKEN, api_redirect, result_hosts=frozenset({RESULT_HOST}))
    with pytest.raises(APIRequestError) as caught:
        client.status("job-123")
    assert caught.value.code == "redirect_refused"


# ------------------------------------------------------------- failures


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (_json({"code": 401, "msg": f"invalid token {TOKEN}"}, 401), "auth_failed"),
        (_json({"code": 429, "msg": "daily quota exceeded"}, 429), "rate_limited"),
        (_json({"code": 500, "msg": "boom"}, 500), "provider_error"),
        (_json({"code": 17, "msg": "bad model", "data": {}}), "provider_error"),
        (HTTPResponse(200, {}, b"<html>not json</html>", 1.0), "response_format"),
    ],
)
def test_rejected_submissions_fail_terminally_without_leaking_the_token(lab, response, code) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome = _submit(runner, spec, workspace_id, document, source, ScriptedTransport([response]))
    assert outcome.state == job_store.FAILED and outcome.run.pages[0].error.startswith(code)
    assert json.loads((outcome.directory / "final.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert TOKEN not in _all_stored_text(store.root)
    assert check_authorization(spec, pages=1, root=store.root).pages_used_total == 1


def test_job_failure_is_terminal_and_poll_timeout_is_resumable(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(limits=_spec().limits.model_copy(update={"poll_timeout_s": 0.0}))
    _enable(spec, store)
    failed = _submit(runner, spec, workspace_id, document, source,
                     ScriptedTransport([SUBMIT, _json({"code": 0, "data": {"state": "failed", "errorMsg": "bad page"}})]))
    assert failed.state == job_store.FAILED and failed.run.pages[0].error.startswith("job_failed")
    ticks = iter(range(0, 10_000, 5))
    slow = ScriptedTransport([SUBMIT] + [RUNNING] * 3)
    timed_out = _submit(runner, spec, workspace_id, document, source, slow,
                        clock=lambda: float(next(ticks)))
    assert timed_out.state == job_store.RESUMABLE_ERROR and timed_out.journal.job_id == "job-123"


def test_page_and_cost_limits_are_enforced(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store, max_pages_total=1, max_pages_per_day=1)
    _submit(runner, spec, workspace_id, document, source, _happy())
    with pytest.raises(APIRunRefused, match="limit reached"):
        _submit(runner, spec, workspace_id, document, source, _happy())
    priced = _spec(pricing=spec.pricing.model_copy(update={"estimated_usd_per_page": 0.5}))
    assert any("exceeds the cap" in r for r in check_authorization(priced, pages=1, root=store.root).reasons)
    unknown = _spec(pricing=spec.pricing.model_copy(update={"estimated_usd_per_page": -1.0}))
    assert any("unknown" in r for r in check_authorization(unknown, pages=1, root=store.root).reasons)


# ------------------------------------------------------- network and secrets


def test_allowlist_guard_blocks_other_hosts_and_private_resolutions(monkeypatch) -> None:
    answers = {"api.example.test": "93.184.216.34", "rebind.example.test": "10.0.0.5"}

    def fake_getaddrinfo(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answers[host], 443))]

    connected: list[tuple] = []
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", lambda sock, address: connected.append(address))
    with allowlist_network_guard({"api.example.test", "rebind.example.test"}):
        socket.getaddrinfo("api.example.test", 443)
        socket.socket().connect(("93.184.216.34", 443))
        with pytest.raises(OutboundNetworkBlocked, match="non-public"):
            socket.getaddrinfo("rebind.example.test", 443)  # approved name, private address
        for call in (
            lambda: socket.getaddrinfo("evil.example", 443),
            lambda: socket.socket().connect(("10.0.0.5", 443)),
            lambda: socket.gethostbyname("evil.example"),
            lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53)),
        ):
            with pytest.raises(OutboundNetworkBlocked):
                call()
    assert connected == [("93.184.216.34", 443)]
    assert socket.getaddrinfo is fake_getaddrinfo  # restored


def test_transport_refuses_plain_http_and_redaction_covers_urls() -> None:
    with pytest.raises(APIRequestError, match="https"):
        UrllibTransport().request("GET", "http://example.test/", headers={}, body=None, timeout=1, max_bytes=10)
    cleaned = redact({"a": f"x {TOKEN} y", "u": "https://h.test/p?sig=1&t=2"}, [TOKEN])
    assert TOKEN not in json.dumps(cleaned) and cleaned["u"] == "https://h.test/p?[query-redacted]"
    assert isinstance(ResultHostNotApproved({"host": "x.example.com"}), APIRequestError)


def test_streamlit_api_mode_is_gated_previews_and_saves_one_page(settings, tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import lab.app.api_panel as panel
    import lab.core.e2.api.runner as api_runner
    from tests.lab.test_e2a import _hardware

    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("PADDLEOCR_ACCESS_TOKEN", raising=False)
    spec = _spec(result_hosts=[])
    monkeypatch.setattr(panel, "load_providers", lambda: [spec])
    transports = [ScriptedTransport([SUBMIT, _done(host="store.example.com")]),
                  ScriptedTransport([_done(host="store.example.com"), _jsonl("resumed in UI")])]
    used: list[ScriptedTransport] = []

    def next_executor():
        transport = transports.pop(0)
        used.append(transport)
        return _in_process(transport)

    monkeypatch.setattr(api_runner, "subprocess_executor", next_executor)
    _manager, workspace_id, _document, _source, _gt = _workspace_with_gt(settings)
    register_test_documents(monkeypatch, tmp_path, _source)  # eligible, NOT yet authorized
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "7_OCR_VLM_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=60)
    app.session_state["workspace_id"] = workspace_id
    app.session_state["e2_hardware"] = _hardware(ram=16.0)
    app.run()
    # The simple full-PDF view is the default; the single-page panel is the advanced view.
    next(item for item in app.radio if item.label == "View").set_value(
        "Advanced: single page, local models, manual review").run()
    next(item for item in app.radio if item.label == "Inference location").set_value("Hosted API (experimental)").run()
    assert not app.exception
    assert any("disabled by default" in item.value for item in app.warning)
    assert not any(item.label == "Send this one page to the API" for item in app.button)
    store = ResultStore(tmp_path / "results")
    _enable(spec, store)
    app.run()
    # Eligible but unauthorized: no preview, no Send button.
    assert not any(item.label == "Send this one page to the API" for item in app.button)
    assert any("No explicit authorization" in item.value for item in app.warning)
    # Authorize this exact document through the UI form (reviewer, hash prefix, declaration).
    from lab.core.e2.api import documents as doc_auth

    next(item for item in app.text_input if item.label == "Reviewer name").input("Tester")
    next(item for item in app.text_input if item.label.startswith("Type the first 12")).input(_document.sha256[:12])
    next(item for item in app.checkbox if item.label == doc_auth.SYNTHETIC_DECLARATION).check()
    next(item for item in app.button if item.label == "Authorize this exact document").click().run()
    assert not app.exception
    assert doc_auth.authorization_state(_document.sha256, root=store.root)["allowed"]
    assert next(item for item in app.button if item.label == "Send this one page to the API").disabled
    next(item for item in app.text_input if "access token" in item.label).input(TOKEN).run()
    next(item for item in app.checkbox if item.label.startswith("I confirm sending page 1")).check().run()
    next(item for item in app.button if item.label == "Send this one page to the API").click().run()
    assert not app.exception
    # Paused for host review: the exact host is shown and Send is now disabled.
    assert any("store.example.com" in item.value for item in app.error)
    assert next(item for item in app.button if item.label == "Send this one page to the API").disabled
    resume = next(item for item in app.button if item.label == "Resume this job (no resubmission)")
    assert resume.disabled  # host not approved yet
    open_job = next(item for item in job_store.journals(spec.id, store.root) if item.open)
    job_store.approve_result_host(spec, host="store.example.com", approved_by="Tester",
                                  statement=job_store.HOST_APPROVAL_STATEMENT, journal=open_job, root=store.root)
    app.run()
    next(item for item in app.button if item.label == "Resume this job (no resubmission)").click().run()
    assert not app.exception
    assert any("Saved write-once API run" in item.value for item in app.success)
    assert [transport.submissions for transport in used] == [1, 0]
    assert TOKEN not in _all_stored_text(tmp_path / "results")


# ----------------------------------------------- self-review regressions


def test_redaction_covers_escaped_json_urls_and_credential_params() -> None:
    body = r'{"jsonUrl":"https:\/\/bj.bcebos.com\/r\/x.jsonl?authorization=bce-auth-v1%2F' + SIGNED + '"}'
    cleaned = redact(body, [])
    assert SIGNED not in cleaned and r"https:\/\/bj.bcebos.com\/r\/x.jsonl?[query-redacted]" in cleaned
    loose = redact(f"see token={TOKEN}&x=1 and Signature={SIGNED}", [])
    assert TOKEN not in loose and SIGNED not in loose
    unicode_escaped = (
        r'{"jsonUrl":"https:\/\/store.example.com\/r\u003fsig\u003d'
        + SIGNED + r'\u0026token\u003d' + TOKEN + r'"}'
    )
    cleaned_unicode = redact(unicode_escaped, [])
    assert SIGNED not in cleaned_unicode and TOKEN not in cleaned_unicode


def test_unrelated_query_parameter_is_not_read_as_expiry() -> None:
    assert sanitize_url("https://h.example.com/p?e=5&x=1")["expires_at"] is None


def test_expired_url_on_unapproved_host_pauses_without_spending_refreshes(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(result_hosts=[])
    _enable(spec, store)
    transport = ScriptedTransport([SUBMIT, _done(host="new.example.com", query="Expires=1")])
    outcome = _submit(runner, spec, workspace_id, document, source, transport, wall_clock=lambda: 2e9)
    assert outcome.state == job_store.AWAITING_RESULT_HOST_APPROVAL
    assert [call["method"] for call in transport.calls] == ["POST", "GET"]  # no refresh requests


class _ExplodingTransport(ScriptedTransport):
    def request(self, method, url, **kwargs):
        if method == "GET":
            self.calls.append({"method": method, "url": url, "headers": {}, "body": b""})
            raise ValueError("unexpected decoder bug")
        return super().request(method, url, **kwargs)


def test_unexpected_exception_after_submit_keeps_the_job_resumable(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome = _submit(runner, spec, workspace_id, document, source, _ExplodingTransport([SUBMIT]))
    assert outcome.state == job_store.RESUMABLE_ERROR and outcome.journal.job_id == "job-123"
    assert outcome.journal.latest()["error"]["code"] == "client_exception"
    resumed = runner.resume(spec, outcome.journal.journal_id, credential=TOKEN,
                            executor=_in_process(ScriptedTransport([_done(), _jsonl("recovered")])))
    assert resumed.state == job_store.COMPLETED


def test_fresh_resume_claim_blocks_duplicate_concurrent_resume(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    paused = _submit(
        runner, spec, workspace_id, document, source,
        ScriptedTransport([SUBMIT, APIRequestError("network_error", "temporary")]),
    )
    assert paused.state == job_store.RESUMABLE_ERROR
    paused.journal.append(job_store.RESUMING, job_id=paused.journal.job_id, result_hosts=[])
    transport = ScriptedTransport([_done(), _jsonl("must not be fetched twice")])
    with pytest.raises(APIRunRefused, match="already being resumed"):
        runner.resume(
            spec, paused.journal.journal_id, credential=TOKEN, executor=_in_process(transport)
        )
    assert transport.calls == []


def test_resume_claim_stays_valid_for_the_whole_worker_deadline(lab) -> None:
    """Regression (final verification): a 120 s stale threshold let a second session
    start a parallel resume while the first was still legitimately polling."""
    from datetime import datetime, timedelta, timezone

    from lab.core.e2.api.runner import resume_claim_stale_s, worker_deadline_s

    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    assert resume_claim_stale_s(spec) > worker_deadline_s(spec) > job_store.CLAIM_STALE_S
    paused = _submit(runner, spec, workspace_id, document, source,
                     ScriptedTransport([SUBMIT, APIRequestError("network_error", "temporary")]))
    job_id = paused.journal.job_id
    busy_since = datetime.now(timezone.utc) - timedelta(seconds=200)  # past the old 120 s
    paused.journal.append(job_store.RESUMING, job_id=job_id, result_hosts=[], at=busy_since.isoformat())
    transport = ScriptedTransport([_done(), _jsonl("must not be fetched twice")])
    with pytest.raises(APIRunRefused, match="already being resumed"):
        runner.resume(spec, paused.journal.journal_id, credential=TOKEN, executor=_in_process(transport))
    assert transport.calls == []
    # Only after the worker deadline (plus margin) has certainly passed may it be taken over.
    gone = datetime.now(timezone.utc) - timedelta(seconds=resume_claim_stale_s(spec) + 5)
    paused.journal.append(job_store.RESUMING, job_id=job_id, result_hosts=[], at=gone.isoformat())
    assert runner.resume(spec, paused.journal.journal_id, credential=TOKEN,
                         executor=_in_process(transport)).state == job_store.COMPLETED


def test_revoked_result_host_blocks_resume(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec(result_hosts=[])
    _enable(spec, store)
    paused = _submit(runner, spec, workspace_id, document, source,
                     ScriptedTransport([SUBMIT, _done(host="store.example.com")]))
    job_store.approve_result_host(spec, host="store.example.com", approved_by="Tester",
                                  statement=job_store.HOST_APPROVAL_STATEMENT, journal=paused.journal, root=store.root)
    assert "store.example.com" in runner.result_hosts(spec)
    time.sleep(0.01)
    job_store.revoke_result_host(spec, host="store.example.com", revoked_by="Tester", root=store.root)
    assert "store.example.com" not in runner.result_hosts(spec)
    with pytest.raises(APIRunRefused, match="Approve the result host"):
        runner.resume(spec, paused.journal.journal_id, credential=TOKEN, executor=_in_process(ScriptedTransport([])))
    assert len(job_store.host_approval_records(spec, store.root)) == 2  # both kept


class SlowTransport:
    def request(self, *args, **kwargs):
        time.sleep(30)


class EchoTokenTransport:
    """Fails with a message containing the Authorization header (must be redacted)."""

    def request(self, method, url, *, headers, body, timeout, max_bytes):
        raise APIRequestError("network_error", f"refused {headers.get('Authorization')}")


def test_isolated_worker_deadline_and_secret_redaction(tmp_path) -> None:
    spec = _spec(limits=_spec().limits.model_copy(update={"request_timeout_s": 1, "poll_timeout_s": 0}))
    image = tmp_path / "page.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    events: list[dict] = []
    started = time.monotonic()
    slow = subprocess_executor("tests.lab.test_api_baidu:SlowTransport", deadline_s=3)(
        spec, TOKEN, image, "page_0001.png", None, frozenset({RESULT_HOST}), events.append)
    assert slow["phase"] == "error" and slow["error"]["code"] == "deadline_exceeded"
    assert slow["job_id"] is None and time.monotonic() - started < 25
    echo = subprocess_executor("tests.lab.test_api_baidu:EchoTokenTransport", deadline_s=60)(
        spec, TOKEN, image, "page_0001.png", None, frozenset({RESULT_HOST}), events.append)
    assert echo["phase"] == "error" and echo["error"]["code"] == "network_error"
    assert TOKEN not in json.dumps(echo) and events == []
