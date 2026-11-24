"""E2-D two-stage hosted API batch evaluation. Every provider response is mocked:
no test opens a network connection, and nothing is sent anywhere."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import fitz
import pytest

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import batch_report
from lab.core.e2.api import documents
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts
from lab.core.e2.api.client import APIRequestError, HTTPResponse
from lab.core.e2.api.runner import APIEvaluationRunner
from lab.core.e2.store import ResultStore
from lab.core.groundtruth import GroundTruthStore
from lab.core.ingest import IngestService
from lab.core.workspace import WorkspaceManager
from tests.lab import test_api_baidu as baidu
from tests.lab import test_api_featherless as feather

BAIDU_ID = "baidu-aistudio-paddleocr-vl-1.6"
FEATHER_ID = "featherless-qwen3-vl-4b-instruct"
OXEN_ID = "oxen-qwen3-vl-4b-instruct"
OXEN_TOKEN = "OXEN-SECRET-5e6f7a"
CREDS = {BAIDU_ID: baidu.TOKEN, FEATHER_ID: feather.TOKEN, OXEN_ID: OXEN_TOKEN}
TEXT_LAYER_CANARY = "TEXTLAYERCANARY"
DOWNLOADS = Path.home() / "Downloads"


# ------------------------------------------------------------------ fixtures


def _oxen_spec(**update):
    from lab.core.e2.api.providers import ProviderSpec, load_providers

    spec = {item.id: item for item in load_providers()}[OXEN_ID]
    return ProviderSpec.model_validate(spec.model_dump() | update)


def _oxen_completion(text="page text", *, cost="0.012600000", currency="USD", status=200, model="qwen3-vl-4b-instruct"):
    """Shape documented at docs.oxen.ai (usage.cost is a decimal string with usage.currency)."""
    from lab.core.e2.api.client import HTTPResponse

    usage = {"prompt_tokens": 1830, "completion_tokens": 512, "total_tokens": 2342}
    if cost is not None:
        usage["cost"] = cost
    if currency is not None:
        usage["currency"] = currency
    payload = {"id": "chatcmpl-oxen-1", "object": "chat.completion", "created": 1, "model": model,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
               "usage": usage}
    return HTTPResponse(status, {"Content-Type": "application/json"}, json.dumps(payload).encode(), 8.0)


def _pdf(path: Path, pages: list[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    return path


def _definition(tmp_path: Path, providers=(BAIDU_ID, FEATHER_ID), options=(), **overrides) -> Path:
    values = {"max_concurrency": 2, "max_concurrency_per_provider": 1, "batch_deadline_minutes": 30,
              "ambiguous_stop_threshold": 2} | overrides
    text = f"""
id = "e2d-test"
title = "E2-D test"
providers = {json.dumps(list(providers))}
provider_options = {json.dumps(list(options))}
max_concurrency = {values['max_concurrency']}
max_concurrency_per_provider = {values['max_concurrency_per_provider']}
batch_deadline_minutes = {values['batch_deadline_minutes']}
ambiguous_stop_threshold = {values['ambiguous_stop_threshold']}
typical_prompt_tokens = 1500
typical_completion_tokens = 1200

[[documents]]
registry_id = "doc-ar"
language = "Arabic"
stage1_pages = [1, 3]
stage1_page_roles = {{ 1 = "prose", 3 = "table" }}

[[documents]]
registry_id = "doc-en"
language = "English"
stage1_pages = [2]
"""
    path = tmp_path / "definition.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _register(tmp_path: Path, monkeypatch, docs: dict[str, tuple[Path, int]], category: str = "synthetic") -> None:
    lines = []
    for registry_id, (pdf, pages) in docs.items():
        data = pdf.read_bytes()
        lines += ["[[documents]]", f'id = "{registry_id}"', f'filename = "{pdf.name}"',
                  f'sha256 = "{hashlib.sha256(data).hexdigest()}"', f"bytes = {len(data)}",
                  f"page_count = {pages}", 'title = "test"', 'provenance = "generated test data"',
                  'verified_on = "2026-11-17"', f'authorization_category = "{category}"', ""]
    registry = tmp_path / "registry.toml"
    registry.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(documents, "DOCUMENT_REGISTRY", registry)


class Env:
    def __init__(self, settings, tmp_path, monkeypatch, *, baidu_hosts=(baidu.RESULT_HOST,), second=None,
                 category="synthetic", options=(), oxen_cost="0.012600000", **definition):
        monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
        self.tmp = tmp_path
        self.settings = settings
        ar = _pdf(tmp_path / "ar.pdf", [f"Arabic page {n} total 1,284 rate 45% {TEXT_LAYER_CANARY}"
                                       for n in (1, 2, 3)])
        en = _pdf(tmp_path / "en.pdf", [f"English page {n} value 2025 {TEXT_LAYER_CANARY}" for n in (1, 2)])
        _register(tmp_path, monkeypatch, {"doc-ar": (ar, 3), "doc-en": (en, 2)}, category)
        self.second = second or FEATHER_ID
        self.oxen_cost = oxen_cost
        monkeypatch.setattr(batch_mod, "DEFINITION_PATH", _definition(
            tmp_path, providers=(BAIDU_ID, self.second), options=options, **definition))
        self.definition, self.definition_sha = batch_mod.load_definition()
        self.manager = WorkspaceManager(settings)
        self.workspace_id = self.manager.create().workspace_id
        self.ingest = IngestService(self.manager, settings)
        for pdf in (ar, en):
            with pdf.open("rb") as handle:
                self.ingest.ingest(self.workspace_id, handle, pdf.name)
        self.store = ResultStore(tmp_path / "results")
        self.root = self.store.root
        self.specs = {BAIDU_ID: baidu._spec(result_hosts=list(baidu_hosts)), FEATHER_ID: feather._spec(),
                      OXEN_ID: _oxen_spec()}
        self.runner = APIEvaluationRunner(self.manager, settings, self.store)
        self.sends: list[tuple[str, str]] = []  # (provider_id, image filename) per POST to the API
        self.lock = threading.Lock()

    @property
    def sources(self):
        pdfs = [(item, self.ingest.original_path(self.workspace_id, item.doc_id))
                for item in self.ingest.list_documents(self.workspace_id)]
        found, problems = batch_mod.resolve_sources(self.definition, pdfs)
        assert not problems
        return found

    def authorize_all(self):
        for source in self.sources.values():
            documents.authorize_document(source.entry, file_path=source.pdf_path, reviewer="Tester",
                                         declaration=source.entry.declaration, days_valid=1, root=self.root)

    def enable_all(self, **kw):
        for provider_id in (BAIDU_ID, FEATHER_ID, OXEN_ID):
            baidu._enable(self.specs[provider_id], self.store,
                          **({"max_pages_total": 50, "max_pages_per_day": 50} | kw))

    def plan(self, stage=1, prior=None):
        plan = batch_mod.build_plan(self.definition, self.definition_sha, stage, sources=self.sources,
                                    specs=self.specs, manager=self.manager, settings=self.settings,
                                    workspace_id=self.workspace_id, root=self.root, stage1_batch=prior)
        return batch_mod.save_plan(plan, self.root)

    def approve(self, batch_dir, budgets=None, **kw):
        plan = batch_mod.load_plan(batch_dir)
        budgets = budgets or {p["provider_id"]: p["max_total_cost_usd"] for p in plan["providers"]}
        return batch_mod.approve_plan(batch_dir, approved_by="Tester", statement=batch_mod.BATCH_APPROVAL_STATEMENT,
                                      typed_plan_prefix=plan["plan_sha256"][:12], budgets_usd=budgets,
                                      hours_valid=kw.pop("hours", 2), **kw)

    def executor(self, *, baidu_script=None, feather_script=None, delay=0.0, on_call=None):
        """Dispatch to a fresh mocked transport per call, by protocol."""
        def execute(spec, credential, image_path, filename, job_id, result_hosts, on_event):
            if on_call:
                on_call(spec)
            if delay:
                time.sleep(delay)
            if spec.protocol == "openai_chat_completions_v1":
                default = (_oxen_completion(f"Arabic page total 1,284 rate 45% {filename}", cost=self.oxen_cost)
                           if spec.id == OXEN_ID else feather._completion(
                               text=f"Arabic page total 1,284 rate 45% {filename}"))
                transport = feather.Scripted(feather_script() if feather_script else [default])
                result = feather._in_process(transport)(spec, credential, image_path, filename, job_id,
                                                        result_hosts, on_event)
            else:
                script = baidu_script(job_id) if baidu_script else (
                    [baidu.SUBMIT, baidu._done(), baidu._jsonl("Arabic page total 1,284 rate 45%")]
                    if job_id is None else [baidu._done(), baidu._jsonl("resumed text 1,284")])
                transport = baidu.ScriptedTransport(script)
                result = baidu._in_process(transport)(spec, credential, image_path, filename, job_id,
                                                      result_hosts, on_event)
            with self.lock:
                self.sends += [(spec.id, filename)] * sum(call["method"] == "POST" for call in transport.calls)
            return result

        return execute

    def execute(self, batch_dir, executor=None, credentials=None, run=True):
        if run:
            batch_mod.request_control(batch_dir, "RUN", by="Tester")
        return batch_mod.BatchRunner(self.runner, self.specs, executor=executor or self.executor()).execute(
            batch_dir, credentials=credentials or CREDS, sources=self.sources)


@pytest.fixture
def env(settings, tmp_path, monkeypatch):
    return Env(settings, tmp_path, monkeypatch)


def _statuses(env, batch_dir):
    return {item["item_id"]: item["status"] for item in batch_mod.batch_status(batch_dir, env.root)["items"]}


def _stored_text(root: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------- planning


def test_stage1_plan_has_exact_pages_hashes_destinations_and_costs(env) -> None:
    batch_dir = env.plan(1)
    plan = batch_mod.load_plan(batch_dir)
    assert sorted((item["registry_id"], item["page"], item["provider_id"]) for item in plan["items"]) == sorted(
        (doc, page, provider) for doc, page in [("doc-ar", 1), ("doc-ar", 3), ("doc-en", 2)]
        for provider in (BAIDU_ID, FEATHER_ID))
    for item in plan["items"]:
        source = env.sources[item["registry_id"]]
        rendered = batch_mod.render_page(env.manager, env.settings, env.workspace_id, source, item["page"])
        assert item["image_sha256"] == rendered.sha256  # exactly what the runner would confirm
        assert item["document_sha256"] == source.entry.sha256
        assert item["destination"] == env.specs[item["provider_id"]].endpoint
        assert item["action"] == "submit"
    feather_plan = next(p for p in plan["providers"] if p["provider_id"] == FEATHER_ID)
    assert feather_plan["max_cost_per_page_usd"] == 0.02 and feather_plan["max_total_cost_usd"] == pytest.approx(0.06)
    assert 0 < feather_plan["typical_total_cost_usd"] < feather_plan["max_total_cost_usd"]
    assert next(p for p in plan["providers"] if p["provider_id"] == BAIDU_ID)["max_total_cost_usd"] == 0
    assert plan["stage"] == 1 and not any(TEXT_LAYER_CANARY in json.dumps(item) for item in plan["items"])
    # Write-once and tamper-evident.
    with pytest.raises(FileExistsError):
        batch_mod._write_once(batch_dir / "plan.json", {})
    (batch_dir / "plan.json").chmod(0o666)
    data = json.loads((batch_dir / "plan.json").read_text(encoding="utf-8"))
    data["items"][0]["page"] = 2
    tampered = env.tmp / "tampered"
    tampered.mkdir()
    (tampered / "plan.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(batch_mod.BatchError, match="modified"):
        batch_mod.load_plan(tampered)


def test_sources_match_by_hash_never_by_name(env, tmp_path) -> None:
    (tmp_path / "copy").mkdir()
    impostor = _pdf(tmp_path / "copy" / "ar.pdf", ["different bytes", "x", "y"])
    with impostor.open("rb") as handle:
        env.ingest.ingest(env.workspace_id, handle, "ar.pdf")  # same name, different bytes
    sources = env.sources
    assert {key: value.entry.sha256 for key, value in sources.items()} == {
        key: hashlib.sha256(value.pdf_path.read_bytes()).hexdigest() for key, value in sources.items()}
    found, problems = batch_mod.resolve_sources(env.definition, [])
    assert not found and len(problems) == 2


# ---------------------------------------------------------- authorization


def test_nothing_is_sent_without_every_gate(env) -> None:
    batch_dir = env.plan(1)
    executor = env.executor()
    with pytest.raises(batch_mod.BatchError, match="not been approved"):
        env.execute(batch_dir, executor, run=False)
    env.approve(batch_dir)
    with pytest.raises(batch_mod.BatchError, match="Press Run"):
        env.execute(batch_dir, executor, run=False)
    # Approved and RUN, but providers disabled and documents unauthorized: all refused.
    status = env.execute(batch_dir, executor)
    assert env.sends == []
    assert set(_statuses(env, batch_dir).values()) == {batch_mod.REFUSED}
    env.enable_all()
    env.execute(batch_dir, executor)
    assert env.sends == [] and "No explicit authorization" in json.dumps(status["items"] + batch_mod.batch_status(
        batch_dir, env.root)["items"])
    # No credential for a provider: that provider is not even attempted.
    env.authorize_all()
    status = env.execute(batch_dir, executor, credentials={BAIDU_ID: baidu.TOKEN})
    assert {provider for provider, _ in env.sends} == {BAIDU_ID}
    assert all(item["status"] == batch_mod.COMPLETED for item in status["items"] if item["provider_id"] == BAIDU_ID)
    assert all(item["status"] == batch_mod.REFUSED for item in status["items"] if item["provider_id"] == FEATHER_ID)
    # A revoked batch approval stops everything.
    batch_mod.revoke_plan_approval(batch_dir, revoked_by="Tester", reason="test")
    with pytest.raises(batch_mod.BatchError, match="revoked"):
        env.execute(batch_dir, executor)


def test_approval_rules(env) -> None:
    batch_dir = env.plan(1)
    plan = batch_mod.load_plan(batch_dir)
    good = {"approved_by": "T", "statement": batch_mod.BATCH_APPROVAL_STATEMENT,
            "typed_plan_prefix": plan["plan_sha256"][:12], "budgets_usd": {BAIDU_ID: 0, FEATHER_ID: 0.06},
            "hours_valid": 1}
    for change, match in [({"statement": "ok"}, "statement"), ({"typed_plan_prefix": "000000000000"}, "12"),
                          ({"budgets_usd": {BAIDU_ID: 0, FEATHER_ID: 9}}, "Budget"),
                          ({"budgets_usd": {BAIDU_ID: 0, FEATHER_ID: 0.01}}, "single page"),
                          ({"hours_valid": 999}, "Validity"), ({"approved_by": " "}, "name")]:
        with pytest.raises(batch_mod.BatchError, match=match):
            batch_mod.approve_plan(batch_dir, **(good | change))
    assert not batch_mod.approval_state(batch_dir)["allowed"]
    batch_mod.approve_plan(batch_dir, **good)
    assert batch_mod.approval_state(batch_dir)["allowed"]
    # Stage approvals are separate: a Stage 2 plan is not approved by Stage 1's approval.
    env.authorize_all()
    stage2 = env.plan(2, prior=batch_dir)
    assert not batch_mod.approval_state(stage2)["allowed"]


# ---------------------------------------------------------- happy path


def test_stage1_runs_once_completes_and_never_duplicates(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    status = env.execute(batch_dir)
    assert status["overall"] == "COMPLETE" and status["done"] == 6
    assert sorted(provider for provider, _ in env.sends) == [BAIDU_ID] * 3 + [FEATHER_ID] * 3
    # Pressing Run again submits nothing: every item is complete.
    env.execute(batch_dir)
    assert len(env.sends) == 6
    stored = _stored_text(env.root)
    assert baidu.TOKEN not in stored and feather.TOKEN not in stored
    # Every attempt has an INTENT recorded before its journal; journals are complete.
    for item in status["items"]:
        assert item["journal_id"] and item["attempts"] == 1
        assert item["job_id"] == "job-123" or item["provider_id"] == FEATHER_ID
        journal = job_store.JobJournal(job_store.journal_dir(item["provider_id"], item["journal_id"], env.root))
        assert journal.state == job_store.COMPLETED
    assert status["providers"][FEATHER_ID]["submissions"] == 3


# ------------------------------------------------------- budget and quota


def test_budget_limit_stops_before_overspending(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir, budgets={BAIDU_ID: 0.0, FEATHER_ID: 0.04})
    status = env.execute(batch_dir)
    feather_items = [item for item in status["items"] if item["provider_id"] == FEATHER_ID]
    assert sum(item["status"] == batch_mod.COMPLETED for item in feather_items) == 2
    pending = [item for item in feather_items if item["status"] == batch_mod.PENDING]
    assert len(pending) == 1 and "budget" in pending[0]["detail"]
    assert sum(provider == FEATHER_ID for provider, _ in env.sends) == 2


def test_provider_quota_is_still_enforced_per_page(env) -> None:
    env.authorize_all()
    env.enable_all(max_pages_total=2, max_pages_per_day=2)
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    status = env.execute(batch_dir)
    for provider in (BAIDU_ID, FEATHER_ID):
        mine = [item for item in status["items"] if item["provider_id"] == provider]
        assert sum(item["status"] == batch_mod.COMPLETED for item in mine) == 2
        assert sum(item["status"] == batch_mod.REFUSED for item in mine) == 1
    assert len(env.sends) == 4
    checks = batch_mod.stage_readiness(batch_dir, specs=env.specs, sources=env.sources, root=env.root,
                                       credentials_present={BAIDU_ID: True, FEATHER_ID: True})
    assert not next(c for c in checks if c["check"].startswith("Provider enablement"))["ok"]


def test_submission_limit_from_approval(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    env.execute(batch_dir)
    item = next(i for i in batch_mod.load_plan(batch_dir)["items"] if i["provider_id"] == FEATHER_ID)
    with pytest.raises(batch_mod.BatchError, match="failed or abandoned"):
        batch_mod.authorize_item_retry(batch_dir, item["item_id"], by="T", statement=batch_mod.RETRY_STATEMENT,
                                       root=env.root)


# ----------------------------------------------------------- crash recovery


def test_crash_before_journal_is_safe_to_submit_but_crash_after_is_never_resent(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    items = batch_mod.load_plan(batch_dir)["items"]
    before, after = (next(i for i in items if i["provider_id"] == provider) for provider in (FEATHER_ID, BAIDU_ID))
    # Crash 1: INTENT recorded, process died before the job journal existed (nothing sent).
    batch_mod._item_log(batch_dir, before["item_id"]).append("INTENT", run_id="run_" + "a" * 18)
    # Crash 2: the journal says SUBMITTING with no Job ID (page may have been sent).
    run_id = "run_" + "b" * 18
    batch_mod._item_log(batch_dir, after["item_id"]).append("INTENT", run_id=run_id)
    job_store.JobJournal(job_store.journal_dir(BAIDU_ID, run_id, env.root)).append(
        job_store.SUBMITTING, image_sha256=after["image_sha256"], page=after["page"], run_directory="",
        document_sha256=after["document_sha256"], provider_id=BAIDU_ID)
    statuses = _statuses(env, batch_dir)
    assert statuses[before["item_id"]] == batch_mod.INTERRUPTED_BEFORE_SEND
    assert statuses[after["item_id"]] == batch_mod.NEEDS_DECISION
    status = env.execute(batch_dir)
    statuses = {item["item_id"]: item for item in status["items"]}
    assert statuses[before["item_id"]]["status"] == batch_mod.COMPLETED
    assert statuses[before["item_id"]]["attempts"] == 2
    assert statuses[after["item_id"]]["status"] == batch_mod.NEEDS_DECISION  # never auto-resent
    assert statuses[after["item_id"]]["attempts"] == 1
    assert sum(provider == BAIDU_ID for provider, _ in env.sends) == 2  # only the two untouched pages
    # The ambiguous attempt is charged at the bound until resolved.
    assert status["providers"][BAIDU_ID]["submissions"] == 3


# ------------------------------------------------- Baidu pause and resume


def test_baidu_pauses_for_result_host_and_resumes_without_resubmission(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, baidu_hosts=())
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    host = "store.example.com"
    script = lambda job_id: [baidu.SUBMIT, baidu._done(host=host)] if job_id is None else [  # noqa: E731
        baidu._done(host=host), baidu._jsonl("resumed 1,284")]
    status = env.execute(batch_dir, env.executor(baidu_script=script))
    waiting = [item for item in status["items"] if item["status"] == batch_mod.NEEDS_HOST_APPROVAL]
    assert len(waiting) == 3 and {item["awaiting_host"] for item in waiting} == {host}
    assert all(item["artifact_complete"] and not item["reusable_saved_result"] for item in waiting)
    assert all(Path(item["artifact_directory"]).is_dir() for item in waiting)
    assert status["overall"] == "WAITING_FOR_USER"
    submitted = sum(provider == BAIDU_ID for provider, _ in env.sends)
    assert submitted == 3
    with pytest.raises(job_store.JobJournalError, match="IP-address"):
        batch_mod.approve_batch_result_host(batch_dir, env.specs[BAIDU_ID], host="127.0.0.1", approved_by="T",
                                            statement=job_store.HOST_APPROVAL_STATEMENT, root=env.root)
    with pytest.raises(job_store.JobJournalError, match="statement"):
        batch_mod.approve_batch_result_host(batch_dir, env.specs[BAIDU_ID], host=host, approved_by="T",
                                            statement="ok", root=env.root)
    with pytest.raises(batch_mod.BatchError, match="No job"):
        batch_mod.approve_batch_result_host(batch_dir, env.specs[BAIDU_ID], host="evil.example.net",
                                            approved_by="T", statement=job_store.HOST_APPROVAL_STATEMENT, root=env.root)
    paths = batch_mod.approve_batch_result_host(batch_dir, env.specs[BAIDU_ID], host=host, approved_by="T",
                                                statement=job_store.HOST_APPROVAL_STATEMENT, root=env.root)
    assert len(paths) == 3  # one job-scoped approval per paused job, never provider-wide
    assert host not in job_store.approved_result_hosts(env.specs[BAIDU_ID], env.root, journal_id="run_" + "c" * 18)
    assert {item["status"] for item in batch_mod.batch_status(batch_dir, env.root)["items"]
            if item["provider_id"] == BAIDU_ID} == {batch_mod.NEEDS_HOST_APPROVAL}
    status = env.execute(batch_dir, env.executor(baidu_script=script))
    assert status["overall"] == "COMPLETE"
    assert sum(provider == BAIDU_ID for provider, _ in env.sends) == submitted  # no resubmission


# --------------------------------------------------- Featherless timeouts


def test_featherless_timeouts_are_ambiguous_stop_the_provider_and_are_never_retried(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    # The retry allowance needs its own budget: an ambiguous request is charged at the bound.
    env.approve(batch_dir, budgets={BAIDU_ID: 0.0, FEATHER_ID: 0.08}, retry_allowance=1)
    timeout =lambda: [APIRequestError("request_timeout", "Request timed out after 180s")]  # noqa: E731
    status = env.execute(batch_dir, env.executor(feather_script=timeout))
    feather_items = [item for item in status["items"] if item["provider_id"] == FEATHER_ID]
    assert sum(item["status"] == batch_mod.NEEDS_DECISION for item in feather_items) == 2
    ambiguous = [item for item in feather_items if item["status"] == batch_mod.NEEDS_DECISION]
    assert all(item["artifact_complete"] and not item["reusable_saved_result"] for item in ambiguous)
    skipped = [item for item in feather_items if item["status"] == batch_mod.PENDING]
    assert len(skipped) == 1 and "unknown outcome" in skipped[0]["detail"]
    assert sum(provider == FEATHER_ID for provider, _ in env.sends) == 2
    # Pressing Run again never resends an ambiguous item.
    env.execute(batch_dir, env.executor(feather_script=timeout))
    assert sum(provider == FEATHER_ID for provider, _ in env.sends) == 3  # only the skipped one was tried
    # The user abandons one, then explicitly authorizes a retry: it is sent exactly once more.
    ambiguous = next(item for item in batch_mod.batch_status(batch_dir, env.root)["items"]
                     if item["status"] == batch_mod.NEEDS_DECISION)
    env.runner.abandon(env.specs[FEATHER_ID], ambiguous["journal_id"], by="T", reason="checked dashboard")
    batch_mod.authorize_item_retry(batch_dir, ambiguous["item_id"], by="T", statement=batch_mod.RETRY_STATEMENT,
                                   root=env.root)
    status = env.execute(batch_dir)
    retried = next(item for item in status["items"] if item["item_id"] == ambiguous["item_id"])
    assert retried["status"] == batch_mod.COMPLETED and retried["attempts"] == 2
    assert sum(provider == FEATHER_ID for provider, _ in env.sends) == 4


# ------------------------------------------------------- Stage 1 -> 2 reuse


def test_stage2_reuses_only_exact_matches_and_needs_its_own_approval(env) -> None:
    env.authorize_all()
    env.enable_all()
    stage1 = env.plan(1)
    env.approve(stage1)
    env.execute(stage1)
    sent_stage1 = len(env.sends)
    stage2 = env.plan(2, prior=stage1)
    plan = batch_mod.load_plan(stage2)
    assert len(plan["items"]) == (3 + 2) * 2
    reused = [item for item in plan["items"] if item["action"] == "reuse"]
    assert len(reused) == 6, [item.get("reason") for item in plan["items"]]
    assert all(item["reuse_of"]["batch_id"] == batch_mod.load_plan(stage1)["batch_id"] for item in reused)
    with pytest.raises(batch_mod.BatchError, match="not been approved"):
        env.execute(stage2)
    env.approve(stage2)
    status = env.execute(stage2)
    assert status["overall"] == "COMPLETE"
    assert len(env.sends) - sent_stage1 == 4  # only the 2 new pages x 2 providers
    assert status["counts"][batch_mod.REUSED] == 6
    reused_statuses = [item for item in status["items"] if item["status"] == batch_mod.REUSED]
    assert all(
        item["artifact_complete"] and item["reusable_saved_result"]
        and Path(item["artifact_directory"]).is_dir()
        for item in reused_statuses
    )
    # Any mismatch disables reuse: a changed prompt/settings identity...
    env.specs[FEATHER_ID] = feather._spec(generation={"temperature": 0.0, "max_tokens": 4096})
    plan = batch_mod.load_plan(env.plan(2, prior=stage1))
    changed = [item for item in plan["items"] if item["provider_id"] == FEATHER_ID and item["reason"]
               and "fingerprint" in item["reason"]]
    # 3 Stage 1 results plus the 2 pages already completed by this Stage 2: every saved
    # Featherless result is refused because the settings identity changed.
    assert len(changed) == 5 and all(item["action"] == "submit" for item in changed)
    assert sum("Stage 1 result not reused" in item["reason"] for item in changed) == 3
    assert all("Saved completed result not reused" in item["reason"] for item in changed)
    # ...or a stored Stage 1 snapshot whose immutable content was modified.
    run_dir = Path(next(item for item in reused if item["provider_id"] == BAIDU_ID)["reuse_of"]["run_directory"])
    (run_dir / "raw_text.txt").chmod(0o666)
    (run_dir / "raw_text.txt").write_text("tampered", encoding="utf-8")
    plan = batch_mod.load_plan(env.plan(2, prior=stage1))
    broken = next(item for item in plan["items"] if item["item_id"] == next(
        i for i in reused if i["provider_id"] == BAIDU_ID)["item_id"])
    assert broken["action"] == "submit" and "not reusable" in broken["reason"]


def test_completed_page_survives_later_oxen_credit_exhaustion(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    credit_error = HTTPResponse(
        402,
        {"Content-Type": "application/json"},
        json.dumps({
            "error": {"type": "insufficient_credits", "title": "You have run out of credits."},
            "status": "error",
            "status_message": "insufficient_credits",
        }).encode(),
        1.0,
    )
    responses = iter([[_oxen_completion("durable first page")], [credit_error]])
    status = env.execute(batch_dir, env.executor(feather_script=lambda: next(responses)))
    oxen = [item for item in status["items"] if item["provider_id"] == OXEN_ID]
    completed = next(item for item in oxen if item["status"] == batch_mod.COMPLETED)
    failed = next(item for item in oxen if item["status"] == batch_mod.FAILED)
    pending = next(item for item in oxen if item["status"] == batch_mod.PENDING)
    assert completed["artifact_complete"] and completed["reusable_saved_result"]
    assert page_artifacts.verify_snapshot(Path(completed["artifact_directory"]))["reusable"]
    assert Path(completed["artifact_directory"]).is_relative_to(env.root / "pages" / "oxen")
    assert failed["artifact_complete"] and not failed["reusable_saved_result"]
    assert "run out of credits" in failed["detail"]
    assert "payment_required" in (Path(failed["artifact_directory"]) / "run.json").read_text(
        encoding="utf-8"
    )
    assert "insufficient credits" in pending["detail"]
    assert sum(provider == OXEN_ID for provider, _ in env.sends) == 2


def test_incomplete_and_corrupt_page_artifacts_are_never_reused(env) -> None:
    env.authorize_all()
    env.enable_all()
    stage1 = env.plan(1)
    env.approve(stage1)
    status = env.execute(stage1)
    completed = next(item for item in status["items"] if item["status"] == batch_mod.COMPLETED)
    artifact = Path(completed["artifact_directory"])
    incomplete = artifact.parent / ".incomplete-deadbeef"
    incomplete.mkdir()
    (incomplete / "complete.json").write_text('{"complete": true}', encoding="utf-8")
    (artifact / "raw_text.txt").write_text("corrupt after completion", encoding="utf-8")
    assert not page_artifacts.verify_snapshot(artifact)["ok"]
    visible = next(
        item for item in batch_mod.batch_status(stage1, env.root)["items"]
        if item["item_id"] == completed["item_id"]
    )
    assert visible["status"] == batch_mod.INCOMPLETE_ARTIFACT
    assert not visible["artifact_complete"] and ".incomplete-" in visible["artifact_directory"]
    stage2 = env.plan(2, prior=stage1)
    prior_item = next(
        item for item in batch_mod.load_plan(stage2)["items"]
        if item["provider_id"] == completed["provider_id"]
        and item["registry_id"] == completed["registry_id"]
        and item["page"] == completed["page"]
    )
    assert prior_item["action"] == "submit" and "not reusable" in prior_item["reason"]


def test_stage2_blocks_pages_with_unresolved_stage1_jobs(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, baidu_hosts=())
    env.authorize_all()
    env.enable_all()
    stage1 = env.plan(1)
    env.approve(stage1)
    env.execute(stage1, env.executor(baidu_script=lambda job_id: [baidu.SUBMIT, baidu._done(host="s.example.com")]))
    plan = batch_mod.load_plan(env.plan(2, prior=stage1))
    blocked = [item for item in plan["items"] if item["action"] == "blocked"]
    assert len(blocked) == 3 and all(item["provider_id"] == BAIDU_ID for item in blocked)
    assert all("unresolved" in item["reason"] for item in blocked)


# --------------------------------------------- concurrency and cancellation


def test_concurrency_is_bounded(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, max_concurrency=2, max_concurrency_per_provider=2)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    active: dict[str, int] = {"all": 0, BAIDU_ID: 0, FEATHER_ID: 0}
    peak: dict[str, int] = dict.fromkeys(active, 0)
    guard = threading.Lock()

    def on_call(spec):
        with guard:
            active["all"] += 1
            active[spec.id] += 1
            for key in active:
                peak[key] = max(peak[key], active[key])

    base = env.executor(delay=0.05, on_call=on_call)

    def counted(spec, *args):
        try:
            return base(spec, *args)
        finally:
            with guard:
                active["all"] -= 1
                active[spec.id] -= 1

    status = env.execute(batch_dir, counted)
    assert status["overall"] == "COMPLETE"
    assert peak["all"] <= 2 and peak[BAIDU_ID] <= 2 and peak[FEATHER_ID] <= 2


def test_pause_and_stop_halt_new_submissions_and_resume_continues(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, max_concurrency=1, max_concurrency_per_provider=1)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    calls = []

    def pause_after_first(spec):
        calls.append(spec.id)
        if len(calls) == 1:
            batch_mod.request_control(batch_dir, "PAUSE", by="Tester")

    status = env.execute(batch_dir, env.executor(on_call=pause_after_first))
    assert status["overall"] == "PAUSED"
    assert status["done"] == 1 and len(env.sends) == 1  # the in-flight page finished; nothing new started
    saved = next(item for item in status["items"] if item["status"] == batch_mod.COMPLETED)
    assert saved["artifact_complete"] and saved["reusable_saved_result"]
    assert page_artifacts.verify_snapshot(Path(saved["artifact_directory"]))["reusable"]
    with pytest.raises(batch_mod.BatchError, match="Press Run"):
        env.execute(batch_dir, run=False)
    batch_mod.request_control(batch_dir, "STOP", by="Tester")
    assert batch_mod.batch_status(batch_dir, env.root)["overall"] == "STOPPED"
    status = env.execute(batch_dir)  # explicit Resume
    assert status["overall"] == "COMPLETE" and len(env.sends) == 6


def test_one_execution_per_batch_and_deadline(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    batch_mod.request_control(batch_dir, "RUN", by="T")
    with batch_mod.executor_lock(batch_dir):
        assert batch_mod.executor_active(batch_dir)
        with pytest.raises(batch_mod.BatchError, match="already executing"):
            env.execute(batch_dir, run=False)
    assert not batch_mod.executor_active(batch_dir)
    ticks = iter([0.0] + [10_000.0] * 100)
    runner = batch_mod.BatchRunner(env.runner, env.specs, executor=env.executor(), clock=lambda: next(ticks))
    status = runner.execute(batch_dir, credentials=CREDS, sources=env.sources)
    assert env.sends == [] and all(item["status"] == batch_mod.PENDING for item in status["items"])
    assert any("deadline" in (item["detail"] or "") for item in status["items"])


# ------------------------------------------------ isolation and reports


def test_ground_truth_and_text_layer_never_reach_a_provider(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    bodies = []
    base = env.executor()

    def capture(spec, credential, image_path, filename, job_id, result_hosts, on_event):
        if image_path:
            bodies.append(Path(image_path).read_bytes())
        return base(spec, credential, image_path, filename, job_id, result_hosts, on_event)

    env.execute(batch_dir, capture)
    assert bodies and all(body.startswith(b"\x89PNG") for body in bodies)
    assert all(TEXT_LAYER_CANARY.encode() not in body for body in bodies)
    # Reports: no official metric without verified Ground Truth; qualitative labels instead.
    analysis = batch_report.analyze_batch(batch_dir, env.root, env.sources)
    assert all(row.get("official_metrics") is None for row in analysis["rows"])
    assert batch_report.NO_GT_NOTE in analysis["notes"]
    source = env.sources["doc-ar"]
    gt_store = GroundTruthStore(env.manager)
    payload = {"pages": [{"page_number": n, "full_text_in_reading_order": "Arabic page total 1,284 rate 45%"}
                         for n in (1, 2, 3)]}
    draft = gt_store.import_json(env.workspace_id, source.document, json.dumps(payload), verified=False, reviewer="t")
    analysis = batch_report.analyze_batch(batch_dir, env.root, env.sources, ground_truths={source.entry.sha256: draft})
    assert all(row.get("official_metrics") is None for row in analysis["rows"])
    verified = gt_store.import_json(env.workspace_id, source.document, json.dumps(payload), verified=True, reviewer="t")
    analysis = batch_report.analyze_batch(batch_dir, env.root, env.sources,
                                          ground_truths={source.entry.sha256: verified})
    assert any(row.get("official_metrics") for row in analysis["rows"] if row["registry_id"] == "doc-ar")
    assert all(row.get("official_metrics") is None for row in analysis["rows"] if row["registry_id"] == "doc-en")


def test_reports_are_qualitative_labelled_and_exportable(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    env.execute(batch_dir)
    analysis = batch_report.analyze_batch(batch_dir, env.root, env.sources)
    assert analysis["recommendation"]["verdict"] == "PRELIMINARY - no winner"
    assert len(analysis["rows"]) == 6 and all(row.get("analysis") for row in analysis["rows"])
    assert analysis["evidence_classes"]["reference"].startswith("PDF text layer")
    assert any("Compare the RAW" in task["task"] for task in analysis["review_tasks"])
    csv_text = batch_report.to_csv(analysis)
    assert csv_text.splitlines()[0].startswith("item_id,registry_id") and len(csv_text.splitlines()) == 7
    markdown = batch_report.to_markdown(analysis)
    assert "NOT Ground Truth" in markdown and "no winner" in markdown
    exported = batch_report.to_json(analysis) + csv_text + markdown
    assert baidu.TOKEN not in exported and feather.TOKEN not in exported
    report = env.store.write_report("e2d_stage1", markdown)
    with pytest.raises(Exception):
        batch_mod._write_once(report, {})


def test_text_analysis_folds_word_lam_alef_artefact_and_flags_values() -> None:
    reference = "تحليل السالمة واألنماط 1,284 حادث بنسبة 45% في 1/1/2021"
    good = "تحليل السلامة والأنماط ١٢٨٤ حادث بنسبة ٤٥٪ في 1/1/2021"
    result = batch_report.analyze_output(good, reference, {"image_objects": 0})
    assert result["text_layer_word_recall"] == 1.0
    assert result["values"]["numbers"]["missing"] == [] and result["values"]["percentages"]["recall"] == 1.0
    bad = "تحليل 9,999 " + "تكرار " * 40
    result = batch_report.analyze_output(bad, reference, {"image_objects": 2})
    assert result["values"]["numbers"]["missing"] == ["1284"]
    assert result["values"]["numbers"]["unsupported"] == ["9999"]
    assert "chart" in result["unsupported_values_note"] and result["repetition"]["flagged"]


# ----------------------------------------------- same-tick record ordering


def test_same_tick_approvals_and_revocations_keep_their_order(env, monkeypatch) -> None:
    """Regression: newest-record-wins must not depend on timestamps (Windows ticks repeat)."""
    from datetime import datetime as real_datetime, timezone as tz

    from lab.core.e2.api import authorization as auth_mod

    frozen = real_datetime.now(tz.utc)

    class Frozen(real_datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: ARG003 - every record gets the same instant
            return frozen

    for module in (documents, job_store, auth_mod, batch_mod):
        monkeypatch.setattr(module, "datetime", Frozen)
    source = env.sources["doc-ar"]
    for _ in range(5):
        documents.authorize_document(source.entry, file_path=source.pdf_path, reviewer="T",
                                     declaration=documents.SYNTHETIC_DECLARATION, days_valid=1, root=env.root)
        monkeypatch.setattr(documents, "datetime", real_datetime)  # expiry check uses the real clock
        assert documents.authorization_state(source.entry.sha256, root=env.root)["allowed"]
        monkeypatch.setattr(documents, "datetime", Frozen)
        documents.revoke_document(source.entry.sha256, revoked_by="T", reason="r", root=env.root)
        assert not documents.authorization_state(source.entry.sha256, root=env.root)["allowed"]
    spec = env.specs[FEATHER_ID]
    feather._enable(spec, env.store)
    auth_mod.record_enablement(spec, enabled=False, enabled_by="T", statement="", max_pages_total=0,
                               max_pages_per_day=0, days_valid=1, root=env.root)
    assert auth_mod.latest_enablement(spec, env.root)["enabled"] is False
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    batch_mod.revoke_plan_approval(batch_dir, revoked_by="T", reason="r")
    assert "revoked" in " ".join(batch_mod.approval_state(batch_dir)["reasons"])
    # Host approvals: many in one tick never collide; a same-tick revocation wins.
    baidu_spec = env.specs[BAIDU_ID]
    journals = []
    for index in range(4):
        journal = job_store.JobJournal(job_store.journal_dir(BAIDU_ID, f"run_{index:018x}", env.root))
        journal.append(job_store.SUBMITTING, image_sha256="x", page=1)
        journal.append(job_store.AWAITING_RESULT_HOST_APPROVAL, job_id=f"j{index}",
                       result_url={"host": "store.example.com"})
        journals.append(journal)
    paths = [job_store.approve_result_host(baidu_spec, host="store.example.com", approved_by="T",
                                           statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal,
                                           root=env.root) for journal in journals]
    assert len(set(paths)) == 4
    monkeypatch.setattr(job_store, "datetime", real_datetime)
    assert "store.example.com" in job_store.approved_result_hosts(baidu_spec, env.root,
                                                                   journal_id=journals[0].journal_id)
    monkeypatch.setattr(job_store, "datetime", Frozen)
    job_store.revoke_result_host(baidu_spec, host="store.example.com", revoked_by="T", root=env.root)
    monkeypatch.setattr(job_store, "datetime", real_datetime)
    assert "store.example.com" not in job_store.approved_result_hosts(baidu_spec, env.root,
                                                                       journal_id=journals[0].journal_id)


# ------------------------------------------------- real documents (local)


TRAFFIC = {"traffic-accident-report-ar-v1": DOWNLOADS / "تحليل العوامل المؤثرة في الحوادث المرورية.pdf",
           "traffic-accident-report-en-v1": DOWNLOADS / "Traffic_Accident_Report_English.pdf"}


@pytest.mark.skipif(not all(path.is_file() for path in TRAFFIC.values()), reason="user's traffic PDFs not present")
def test_real_traffic_reports_plan_locally_and_stay_blocked_until_authorized(settings, tmp_path, monkeypatch) -> None:
    """Real registry + real definition; local rendering only (nothing is sent)."""
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    definition, definition_sha = batch_mod.load_definition()
    manager = WorkspaceManager(settings)
    workspace_id = manager.create().workspace_id
    ingest = IngestService(manager, settings)
    for path in TRAFFIC.values():
        with path.open("rb") as handle:
            ingest.ingest(workspace_id, handle, path.name)
    pdfs = [(item, ingest.original_path(workspace_id, item.doc_id)) for item in ingest.list_documents(workspace_id)]
    sources, problems = batch_mod.resolve_sources(definition, pdfs)
    assert not problems and set(sources) == set(TRAFFIC)
    from lab.core.e2.api.providers import load_providers

    specs = {item.id: item for item in load_providers()}
    store = ResultStore(tmp_path / "results")
    plan = batch_mod.build_plan(definition, definition_sha, 1, sources=sources, specs=specs, manager=manager,
                                settings=settings, workspace_id=workspace_id, root=store.root)
    assert sorted({(item["registry_id"], item["page"]) for item in plan["items"]}) == [
        ("traffic-accident-report-ar-v1", 1), ("traffic-accident-report-ar-v1", 5),
        ("traffic-accident-report-ar-v1", 16), ("traffic-accident-report-en-v1", 1),
        ("traffic-accident-report-en-v1", 6), ("traffic-accident-report-en-v1", 12)]
    assert len(plan["items"]) == 12
    roles = {(item["registry_id"], item["page"]): item["page_features"] for item in plan["items"]}
    assert roles[("traffic-accident-report-ar-v1", 5)]["image_objects"] >= 1
    assert roles[("traffic-accident-report-ar-v1", 16)]["path_objects"] >= 100
    assert roles[("traffic-accident-report-en-v1", 12)]["path_objects"] >= 100
    batch_dir = batch_mod.save_plan(plan, store.root)
    checks = batch_mod.stage_readiness(batch_dir, specs=specs, sources=sources, root=store.root,
                                       credentials_present={})
    failing = {check["check"] for check in checks if not check["ok"]}
    assert any(name.startswith("Document authorization") for name in failing)
    assert "Batch approval" in failing and any(name.startswith("Provider enablement") for name in failing)
    stage2 = batch_mod.build_plan(definition, definition_sha, 2, sources=sources, specs=specs, manager=manager,
                                  settings=settings, workspace_id=workspace_id, root=store.root, stage1_batch=batch_dir)
    assert len(stage2["items"]) == 62
    assert all(item["action"] == "submit" for item in stage2["items"])  # Stage 1 never ran: nothing to reuse


# ------------------------------------------------------------ Streamlit


def test_streamlit_two_stage_flow_is_user_gated(settings, tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import lab.core.e2.api.providers as providers_mod
    import lab.core.e2.api.runner as api_runner

    env = Env(settings, tmp_path, monkeypatch)
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(env.root))
    for name in ("PADDLEOCR_ACCESS_TOKEN", "FEATHERLESS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(providers_mod, "load_providers", lambda path=None: list(env.specs.values()))
    monkeypatch.setattr(api_runner, "subprocess_executor", lambda *a, **k: env.executor())
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "8_Two_Stage_API_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state["workspace_id"] = env.workspace_id
    app.run()
    assert not app.exception
    button = lambda label: next(item for item in app.button if item.label == label)  # noqa: E731
    button("Create a new Stage 1 plan (renders pages locally; sends nothing)").click().run()
    assert not app.exception
    batch_dir = batch_mod.list_batches(env.root, "e2d-test", stage=1)[-1]
    assert button("Run Stage 1").disabled
    plan = batch_mod.load_plan(batch_dir)
    # Approve the exact plan through the form.
    key = plan["batch_id"]
    app.text_input(key=f"e2d-approve-by-{key}").input("Tester")
    app.text_input(key=f"e2d-approve-sha-{key}").input(plan["plan_sha256"][:12])
    app.checkbox(key=f"e2d-approve-agree-{key}").check()
    button("Approve this exact Stage 1 plan").click().run()
    assert not app.exception and batch_mod.approval_state(batch_dir)["allowed"]
    assert button("Run Stage 1").disabled  # documents, providers and credentials still missing
    env.authorize_all()
    env.enable_all()
    app.text_input(key=f"e2d-credential-{BAIDU_ID}").input(baidu.TOKEN)
    app.text_input(key=f"e2d-credential-{FEATHER_ID}").input(feather.TOKEN)
    app.text_input(key="e2d-operator-1").input("Tester")
    app.run()
    assert not button("Run Stage 1").disabled
    assert env.sends == []  # nothing happened until the user pressed Run
    button("Run Stage 1").click().run()
    deadline = time.monotonic() + 60
    while batch_mod.batch_status(batch_dir, env.root)["overall"] != "COMPLETE" and time.monotonic() < deadline:
        time.sleep(0.2)
    assert batch_mod.batch_status(batch_dir, env.root)["overall"] == "COMPLETE"
    assert len(env.sends) == 6
    assert batch_mod.list_batches(env.root, "e2d-test", stage=2) == []  # Stage 2 never auto-starts
    app.run()
    assert not app.exception
    assert any("PRELIMINARY" in item.value for item in app.subheader)
    assert baidu.TOKEN not in _stored_text(env.root) and feather.TOKEN not in _stored_text(env.root)
