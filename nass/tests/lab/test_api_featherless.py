"""Featherless (OpenAI-compatible) Qwen3-VL-4B provider: mocked tests only, no network."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.authorization import REQUIRED_STATEMENT, check_authorization, record_enablement
from lab.core.e2.api.client import APIRequestError, HTTPResponse, OpenAIChatClient, make_client
from lab.core.e2.api.providers import ProviderSpec, load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused
from lab.core.e2.api.worker import execute_job
from lab.core.e2.prompts import get_prompt
from lab.core.e2.specs import load_model_specs
from lab.core.e2.store import ResultStore
from tests.lab.api_doc_helpers import authorize, register_test_documents
from tests.lab.test_e2a import _workspace_with_gt

TOKEN = "FEATHERLESS-SECRET-1b2c3d"


def _spec(**update) -> ProviderSpec:
    spec = {item.id: item for item in load_providers()}["featherless-qwen3-vl-4b-instruct"]
    return ProviderSpec.model_validate(spec.model_dump() | update)


def _completion(
    text="page 1 text ٢٠٢٥", finish="stop", model="Qwen/Qwen3-VL-4B-Instruct",
    status=200, cost=None,
):
    payload = {"id": "chatcmpl-77", "object": "chat.completion", "model": model,
               "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
               "usage": {"prompt_tokens": 3812, "completion_tokens": 640, "total_tokens": 4452}}
    if cost is not None:
        payload["usage"]["cost"] = cost
    return HTTPResponse(status, {"Content-Type": "application/json"}, json.dumps(payload).encode(), 9.0)


class Scripted:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def request(self, method, url, *, headers, body, timeout, max_bytes, **options):
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body or b"",
                           "options": options})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _in_process(transport):
    def execute(spec, credential, image_path, filename, job_id, result_hosts, on_event):
        client = make_client(spec, credential, transport, result_hosts=result_hosts)
        return execute_job(client, image_path=image_path, filename=filename, job_id=job_id, emit=on_event)

    return execute


@pytest.fixture
def lab(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    store = ResultStore(tmp_path / "results")
    register_test_documents(monkeypatch, tmp_path, source)
    authorize(source, store.root)
    return APIEvaluationRunner(manager, settings, store), store, workspace_id, document, source, ground_truth


def _enable(spec, store, **kw):
    params = {"enabled": True, "enabled_by": "Tester", "statement": REQUIRED_STATEMENT,
              "max_pages_total": 10, "max_pages_per_day": 10, "days_valid": 1} | kw
    record_enablement(spec, root=store.root, **params)


def _submit(runner, spec, workspace_id, document, source, transport, ground_truth=None):
    preview = runner.preview(spec, workspace_id, document, source, 1)
    return runner.run(spec, workspace_id, document, source, 1, credential=TOKEN,
                      confirmed_sha256=preview["file"]["sha256"], ground_truth=ground_truth,
                      executor=_in_process(transport)), preview


def _stored(root: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file())


def test_registry_entry_matches_the_local_qwen_configuration() -> None:
    spec = _spec()
    local = {item.id: item for item in load_model_specs()}["qwen3-vl-4b"]
    assert spec.runnable and spec.protocol == "openai_chat_completions_v1"
    assert spec.endpoint == "https://api.featherless.ai/v1/chat/completions"
    assert spec.api_model_id == local.repo_id == "Qwen/Qwen3-VL-4B-Instruct"
    assert spec.prompt_id == local.prompt_id  # identical prompt to local inference
    assert spec.generation == {"temperature": 0.0, "max_tokens": local.generation["max_new_tokens"]}
    assert spec.exact_revision_verified is False and spec.pricing.price_verified is True
    assert spec.pricing.usd_per_million_input_tokens == 0.4
    assert spec.pricing.usd_per_million_output_tokens == 0.8
    assert spec.allowed_hosts == frozenset({"api.featherless.ai"})
    with pytest.raises(ValueError, match="unsupported generation"):
        ProviderSpec.model_validate(spec.model_dump() | {"generation": {"repetition_penalty": 1.1}})
    with pytest.raises(ValueError, match="result_hosts"):
        ProviderSpec.model_validate(spec.model_dump() | {"result_hosts": ["x.example.com"]})


def test_one_page_request_is_exact_scored_and_stored(lab) -> None:
    runner, store, workspace_id, document, source, ground_truth = lab
    spec = _spec()
    _enable(spec, store)
    transport = Scripted([_completion()])
    outcome, preview = _submit(runner, spec, workspace_id, document, source, transport, ground_truth)
    assert outcome.state == job_store.COMPLETED
    run = outcome.run
    assert run.adapter.adapter_id == "api.featherless-qwen3-vl-4b-instruct" and run.adapter.kind == "api"
    assert run.pages[0].text == "page 1 text ٢٠٢٥"
    # Exact request: prompt, one PNG data URI, registered generation options, nothing else.
    call = transport.calls[0]
    body = json.loads(call["body"])
    assert call["url"] == spec.endpoint and call["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert set(body) == {"model", "messages", "stream", "temperature", "max_tokens"}
    assert body["model"] == "Qwen/Qwen3-VL-4B-Instruct" and body["temperature"] == 0.0 and body["max_tokens"] == 2048
    parts = body["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": get_prompt("vlm_transcribe_v1").text}
    image = Path(preview["image_path"]).read_bytes()
    assert parts[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(image).decode()
    # Identity, termination, usage and cost (estimated from reported usage and verified price units).
    response = run.environment["e2"]["page_raw"]["1"]["response"]
    assert response["reported_model"] == "Qwen/Qwen3-VL-4B-Instruct" and response["reported_model_matches"]
    assert response["truncated_at_max_new_tokens"] is False and response["finish_reason"] == "stop"
    assert run.usage.prompt_tokens == 3812 and run.usage.completion_tokens == 640
    assert run.usage.cost_source == "estimated"
    assert run.usage.cost_usd == pytest.approx((3812 * 0.4 + 640 * 0.8) / 1e6)
    assert "UNVERIFIED" not in run.environment["api"]["cost"]["basis"]
    assert run.adapter.params["prompt_used_by_backend"] is True
    summary = json.loads((outcome.directory / "summary.json").read_text(encoding="utf-8"))["summaries"][0]
    assert summary["kind"] == "hosted API"
    stored = _stored(store.root)
    assert TOKEN not in stored and base64.b64encode(image).decode()[:200] not in stored  # no page bytes in logs


def test_truncation_and_model_mismatch_are_recorded(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome, _ = _submit(runner, spec, workspace_id, document, source,
                         Scripted([_completion(finish="length", model="qwen3-vl-4b-fp8")]))
    response = outcome.run.environment["e2"]["page_raw"]["1"]["response"]
    assert response["truncated_at_max_new_tokens"] is True
    assert response["reported_model"] == "qwen3-vl-4b-fp8" and response["reported_model_matches"] is False


def test_provider_reported_settled_cost_takes_precedence(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome, _ = _submit(
        runner, spec, workspace_id, document, source, Scripted([_completion(cost=0.0037)])
    )
    assert outcome.run.usage.cost_usd == pytest.approx(0.0037)
    assert outcome.run.usage.cost_source == "response"
    assert outcome.run.environment["e2"]["page_raw"]["1"]["response"]["usage"]["cost"] == 0.0037


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (HTTPResponse(401, {}, json.dumps({"error": {"message": f"bad key {TOKEN}"}}).encode(), 1.0), "auth_failed"),
        (HTTPResponse(402, {}, b'{"error":{"message":"insufficient credits"}}', 1.0), "payment_required"),
        (HTTPResponse(404, {}, b'{"error":{"message":"model not found"}}', 1.0), "model_not_found"),
        (HTTPResponse(429, {}, b'{"error":{"message":"concurrency"}}', 1.0), "rate_limited"),
        (HTTPResponse(302, {"Location": "https://evil.example/"}, b"", 1.0), "redirect_refused"),
        (HTTPResponse(200, {}, b'{"choices":[]}', 1.0), "response_format"),
    ],
)
def test_rejected_requests_fail_terminally_without_leaking_the_key(lab, response, code) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome, _ = _submit(runner, spec, workspace_id, document, source, Scripted([response]))
    assert outcome.state == job_store.FAILED and outcome.run.pages[0].error.startswith(code)
    assert TOKEN not in _stored(store.root)


def test_timeout_is_unconfirmed_never_retried_and_not_resumable(lab) -> None:
    runner, store, workspace_id, document, source, _gt = lab
    spec = _spec()
    _enable(spec, store)
    outcome, _ = _submit(runner, spec, workspace_id, document, source,
                         Scripted([APIRequestError("request_timeout", "Request timed out after 180s")]))
    assert outcome.state == job_store.SUBMISSION_UNCONFIRMED  # may have been processed and billed
    again = Scripted([_completion()])
    with pytest.raises(APIRunRefused, match="unfinished job already exists"):
        _submit(runner, spec, workspace_id, document, source, again)
    with pytest.raises(APIRunRefused, match="no job to resume"):
        runner.resume(spec, outcome.journal.journal_id, credential=TOKEN, executor=_in_process(again))
    assert again.calls == []
    runner.abandon(spec, outcome.journal.journal_id, by="Tester", reason="checked usage dashboard")
    assert _submit(runner, spec, workspace_id, document, source, again)[0].state == job_store.COMPLETED


def test_cost_cap_and_ground_truth_isolation(lab) -> None:
    runner, store, workspace_id, document, source, ground_truth = lab
    spec = _spec()
    _enable(spec, store)
    expensive = _spec(limits=spec.limits.model_copy(update={"max_cost_usd_per_run": 0.001}))
    assert any("exceeds the cap" in r for r in check_authorization(expensive, pages=1, root=store.root).reasons)
    canary = "CANARY-GT-FEATHERLESS-41"
    ground_truth = ground_truth.model_copy(
        update={"pages": [page.model_copy(update={"text": canary + page.text}) for page in ground_truth.pages]})
    transport = Scripted([_completion()])
    _submit(runner, spec, workspace_id, document, source, transport, ground_truth)
    assert canary.encode() not in transport.calls[0]["body"]


def test_resume_is_never_possible_for_chat_and_client_refuses_job_ids() -> None:
    client = OpenAIChatClient(_spec(), TOKEN, Scripted([]))
    with pytest.raises(APIRequestError, match="no job to resume"):
        client.run(None, None, job_id="x")


def test_both_providers_land_in_one_comparable_library(lab) -> None:
    from tests.lab.test_api_baidu import _done, _jsonl, SUBMIT, ScriptedTransport, _in_process as baidu_exec

    runner, store, workspace_id, document, source, ground_truth = lab
    baidu = {item.id: item for item in load_providers()}["baidu-aistudio-paddleocr-vl-1.6"]
    baidu = ProviderSpec.model_validate(baidu.model_dump() | {"result_hosts": ["results.example.test"]})
    feather = _spec()
    _enable(baidu, store)
    _enable(feather, store)
    preview = runner.preview(baidu, workspace_id, document, source, 1)
    first = runner.run(baidu, workspace_id, document, source, 1, credential=TOKEN,
                       confirmed_sha256=preview["file"]["sha256"], ground_truth=ground_truth,
                       executor=baidu_exec(ScriptedTransport([SUBMIT, _done(), _jsonl("paddle text")])))
    second, _ = _submit(runner, feather, workspace_id, document, source, Scripted([_completion()]), ground_truth)
    adapters = {item["final"]["adapter_id"] for item in store.runs()}
    assert adapters == {"api.baidu-aistudio-paddleocr-vl-1.6", "api.featherless-qwen3-vl-4b-instruct"}
    keys = [set(json.loads((o.directory / "summary.json").read_text(encoding="utf-8"))["summaries"][0]["metrics"])
            for o in (first, second)]
    assert keys[0] == keys[1]  # identical scoring for both providers
    # Same page image for both (the same render is what each provider received).
    assert first.journal.first["image_sha256"] == second.journal.first["image_sha256"]
