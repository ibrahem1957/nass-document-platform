"""Per-job result-host approval, resume without resubmission, and ambiguous-outcome
decisions in E2-D. Every provider exchange is mocked; nothing is sent anywhere."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.client import APIRequestError
from tests.lab import test_api_baidu as baidu
from tests.lab import test_api_featherless as feather
from tests.lab.test_e2d_batch import BAIDU_ID, FEATHER_ID, Env, _stored_text


class _Provider:
    """Each Baidu submission gets its own Job ID and its own result-storage host."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.count = 0
        self.host_of: dict[str, str] = {}

    def script(self, job_id):
        if job_id is None:
            with self.lock:
                self.count += 1
                new_id, host = f"1163059{self.count:011d}", f"paddleocr-store-{self.count}.bj.example.com"
            self.host_of[new_id] = host
            return [baidu._json({"code": 0, "data": {"jobId": new_id}}), baidu._done(host=host)]
        return [baidu._done(host=self.host_of[job_id]), baidu._jsonl(f"resumed {job_id}")]


@pytest.fixture
def paused(settings, tmp_path, monkeypatch):
    env = Env(settings, tmp_path, monkeypatch, baidu_hosts=())
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    provider = _Provider()
    executor = env.executor(baidu_script=provider.script)
    status = env.execute(batch_dir, executor)
    waiting = [item for item in status["items"] if item["status"] == batch_mod.NEEDS_HOST_APPROVAL]
    assert len(waiting) == 3 and status["overall"] == "WAITING_FOR_USER"
    return env, batch_dir, provider, executor, waiting


def _approve(env, batch_dir, item, **override):
    values = {"host": item["awaiting_host"], "job_id": item["job_id"], "approved_by": "Operator",
              "statement": job_store.HOST_APPROVAL_STATEMENT} | override
    return batch_mod.approve_item_result_host(batch_dir, env.specs[BAIDU_ID], item["item_id"], root=env.root,
                                              **values)


def _baidu_posts(env) -> int:
    return sum(provider == BAIDU_ID for provider, _ in env.sends)


def test_waiting_jobs_expose_job_host_and_sanitized_https_url(paused) -> None:
    env, _batch_dir, _provider, _executor, waiting = paused
    assert len({item["job_id"] for item in waiting}) == 3
    assert len({item["awaiting_host"] for item in waiting}) == 3
    for item in waiting:
        url = item["result_url"]
        assert url["scheme"] == "https" and url["host"] == item["awaiting_host"]
        assert url["path"].startswith("/r/") and url["query_parameter_names"] == ["X-Sig"]
    assert baidu.SIGNED not in _stored_text(env.root)  # the signature is never stored


def test_approval_is_exact_per_job_and_never_broadens(paused) -> None:
    env, batch_dir, _provider, _executor, waiting = paused
    first, second, _third = waiting
    spec = env.specs[BAIDU_ID]
    with pytest.raises(batch_mod.BatchError, match="exactly the host"):
        _approve(env, batch_dir, first, host=second["awaiting_host"])  # another job's host
    with pytest.raises(batch_mod.BatchError, match="exactly the host"):
        _approve(env, batch_dir, first, host="evil.example.net")
    with pytest.raises(job_store.JobJournalError, match="IP-address"):
        _approve(env, batch_dir, first, host="203.0.113.9")
    with pytest.raises(batch_mod.BatchError, match="Job ID"):
        _approve(env, batch_dir, first, job_id=second["job_id"])
    with pytest.raises(job_store.JobJournalError, match="statement"):
        _approve(env, batch_dir, first, statement="ok")
    with pytest.raises(job_store.JobJournalError, match="name"):
        _approve(env, batch_dir, first, approved_by=" ")
    with pytest.raises(batch_mod.BatchError, match="not a .* item of this batch"):
        _approve(env, batch_dir, first | {"item_id": "doc-ar__p0001__" + FEATHER_ID})
    assert job_store.host_approval_records(spec, env.root) == []  # nothing recorded by refusals

    _approve(env, batch_dir, first)
    statuses = {item["item_id"]: item for item in batch_mod.batch_status(batch_dir, env.root)["items"]}
    assert batch_mod.item_host_approved(spec, statuses[first["item_id"]], env.root)
    assert not batch_mod.item_host_approved(spec, statuses[second["item_id"]], env.root)
    # The approval is bound to the first job's journal: no other job (not even a new
    # job that reports the same host) gains it, and the registry stays unchanged.
    assert first["awaiting_host"] not in job_store.approved_result_hosts(
        spec, env.root, journal_id=second["journal_id"])
    assert first["awaiting_host"] not in job_store.approved_result_hosts(spec, env.root, journal_id="run_" + "d" * 18)
    assert spec.result_hosts == []
    with pytest.raises(batch_mod.BatchError, match="already approved"):
        _approve(env, batch_dir, first)
    (record,) = job_store.host_approval_records(spec, env.root)
    assert record["surfaced_by_journal"] == first["journal_id"] and record["surfaced_by_job_id"] == first["job_id"]


def test_resume_downloads_only_approved_jobs_and_never_resubmits(paused) -> None:
    env, batch_dir, provider, executor, waiting = paused
    posts = _baidu_posts(env)
    assert posts == 3
    first, *rest = waiting
    _approve(env, batch_dir, first)
    status = env.execute(batch_dir, executor)  # Resume
    by_id = {item["item_id"]: item for item in status["items"]}
    assert by_id[first["item_id"]]["status"] == batch_mod.COMPLETED
    assert by_id[first["item_id"]]["job_id"] == first["job_id"]  # same job, downloaded
    assert all(by_id[item["item_id"]]["status"] == batch_mod.NEEDS_HOST_APPROVAL for item in rest)
    for item in rest:
        _approve(env, batch_dir, item)
    status = env.execute(batch_dir, executor)
    assert status["overall"] == "COMPLETE"
    assert _baidu_posts(env) == posts  # no page image was submitted again
    assert provider.count == 3
    texts = {Path(item["run_directory"]).joinpath("raw_text.txt").read_text(encoding="utf-8")
             for item in status["items"] if item["provider_id"] == BAIDU_ID}
    assert texts == {f"resumed {item['job_id']}" for item in waiting}


def test_readiness_does_not_demand_page_quota_for_jobs_that_only_resume(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, baidu_hosts=())
    env.authorize_all()
    env.enable_all(max_pages_total=3, max_pages_per_day=3)  # exactly the 3 Stage 1 pages
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    provider = _Provider()
    env.execute(batch_dir, env.executor(baidu_script=provider.script))
    checks = {check["check"]: check for check in batch_mod.stage_readiness(
        batch_dir, specs=env.specs, sources=env.sources, root=env.root,
        credentials_present={BAIDU_ID: True, FEATHER_ID: True})}
    enablement = checks[f"Provider enablement {BAIDU_ID}"]
    assert enablement["ok"] and "needs 0 pages" in enablement["detail"]
    assert all(check["ok"] for check in checks.values()), [c for c in checks.values() if not c["ok"]]


def test_ambiguous_timeout_needs_a_separate_decision_with_diagnostics(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    stalled = APIRequestError(
        "request_timeout", "Request timed out after 120.0s during waiting_response_headers",
        diagnostics={"stages": {
            "connection": {"status": "completed", "elapsed_ms": 652.9},
            "sending_request": {"status": "completed", "elapsed_ms": 1896.6, "request_bytes": 339914},
            "waiting_response_headers": {"status": "timeout", "elapsed_ms": 120040.3}},
            "http_status": None, "response_bytes": 0, "request_id_headers": {}})
    calls = {"n": 0}

    def feather_script():
        calls["n"] += 1
        return [stalled] if calls["n"] == 1 else [feather._completion()]

    executor = env.executor(feather_script=feather_script)
    status = env.execute(batch_dir, executor)
    (undecided,) = [item for item in status["items"] if item["status"] == batch_mod.NEEDS_DECISION]
    facts = undecided["diagnostics"]
    assert facts["code"] == "request_timeout" and facts["stalled_stage"] == "waiting_response_headers"
    assert facts["upload_completed"] is True and facts["http_status"] is None
    assert facts["stage_elapsed_ms"]["waiting_response_headers"] == pytest.approx(120040.3)
    sent = len(env.sends)
    # Resume does not touch it, and a host approval cannot be used on it.
    status = env.execute(batch_dir, executor)
    still = next(item for item in status["items"] if item["item_id"] == undecided["item_id"])
    assert still["status"] == batch_mod.NEEDS_DECISION and len(env.sends) == sent
    with pytest.raises(batch_mod.BatchError, match="not waiting"):
        batch_mod.approve_item_result_host(batch_dir, env.specs[FEATHER_ID], undecided["item_id"], host="x.example.com",
                                           job_id="", approved_by="T", statement=job_store.HOST_APPROVAL_STATEMENT,
                                           root=env.root)
    # Abandoning is a decision only; a new attempt still needs its own authorization.
    env.runner.abandon(env.specs[FEATHER_ID], undecided["journal_id"], by="Operator", reason="checked dashboard")
    status = env.execute(batch_dir, executor)
    after = next(item for item in status["items"] if item["item_id"] == undecided["item_id"])
    assert after["status"] == batch_mod.ABANDONED and len(env.sends) == sent


# ---------------------------------------------------------------- Streamlit


def _app(env, monkeypatch, settings):
    from streamlit.testing.v1 import AppTest

    import lab.core.e2.api.providers as providers_mod
    import lab.core.e2.api.runner as api_runner

    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(env.root))
    monkeypatch.setattr(providers_mod, "load_providers", lambda path=None: list(env.specs.values()))
    monkeypatch.setattr(api_runner, "subprocess_executor", lambda *a, **k: env.executor())
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "8_Two_Stage_API_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state["workspace_id"] = env.workspace_id
    return app


def test_streamlit_shows_one_approval_form_per_job_with_its_details(paused, monkeypatch, settings) -> None:
    env, batch_dir, _provider, _executor, waiting = paused
    app = _app(env, monkeypatch, settings)
    app.run()
    assert not app.exception
    text = "\n".join(item.value for item in app.markdown)
    for item in waiting:
        assert item["job_id"] in text and item["awaiting_host"] in text and item["journal_id"] in text
        assert f"https://{item['awaiting_host']}/r/job-123.jsonl?X-Sig=<omitted>" in text
    assert baidu.SIGNED not in text
    labels = [button.label for button in app.button]
    for item in waiting:
        assert f"Approve {item['awaiting_host']} for job {item['job_id']} only" in labels
    assert any(subheader.value.startswith("Approval needed: result-storage host (3 job(s))")
               for subheader in app.subheader)

    first = waiting[0]
    key = f"1-{first['journal_id']}"
    sent = len(env.sends)
    # Wrong Job ID typed: refused, nothing recorded.
    app.text_input(key=f"e2d-hostjob-by-{key}").input("Operator")
    app.text_input(key=f"e2d-hostjob-typed-{key}").input(first["awaiting_host"])
    app.text_input(key=f"e2d-hostjob-job-{key}").input(waiting[1]["job_id"])
    app.checkbox(key=f"e2d-hostjob-agree-{key}").check()
    next(b for b in app.button if b.label == f"Approve {first['awaiting_host']} for job {first['job_id']} only"
         ).click().run()
    assert any("Job ID does not match" in error.value for error in app.error)
    assert job_store.host_approval_records(env.specs[BAIDU_ID], env.root) == []
    app.text_input(key=f"e2d-hostjob-job-{key}").input(first["job_id"])
    next(b for b in app.button if b.label == f"Approve {first['awaiting_host']} for job {first['job_id']} only"
         ).click().run()
    assert not app.exception
    (record,) = job_store.host_approval_records(env.specs[BAIDU_ID], env.root)
    assert record["surfaced_by_journal"] == first["journal_id"] and record["approved_by"] == "Operator"
    assert len(env.sends) == sent  # approving sends and downloads nothing
    assert any("approved for this job only" in item.value for item in app.success)
    assert "Resume Stage 1" in [button.label for button in app.button]
    assert baidu.TOKEN not in _stored_text(env.root)


def test_controls_hidden_while_running_appear_once_execution_stops(paused, monkeypatch, settings) -> None:
    env, batch_dir, _provider, _executor, waiting = paused
    app = _app(env, monkeypatch, settings)
    with batch_mod.executor_lock(batch_dir):  # a run in progress
        app.run()
        assert not app.exception
        assert not any(b.label.startswith("Approve ") and " for job " in b.label for b in app.button)
        assert any("appear here when execution stops" in caption.value for caption in app.caption)
    app.run()
    assert sum(b.label.startswith("Approve ") and " for job " in b.label for b in app.button) == 3
