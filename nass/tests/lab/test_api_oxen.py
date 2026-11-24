"""Oxen.ai Qwen3-VL provider and the user-owned document category.

Every provider response is mocked (shapes from docs.oxen.ai); no test opens a network
connection, and nothing is sent anywhere."""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import documents
from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts
from lab.core.e2.api.client import APIRequestError, HTTPResponse, reported_usd_cost
from lab.core.e2.api.providers import ProviderSpec, load_providers
from lab.core.e2.api.runner import APIEvaluationRunner, APIRunRefused
from lab.core.e2.prompts import get_prompt
from lab.core.e2.store import ResultStore
from tests.lab import test_api_baidu as baidu
from tests.lab import test_api_featherless as feather
from tests.lab.api_doc_helpers import authorize, register_test_documents
from tests.lab.test_e2a import _workspace_with_gt
from tests.lab.test_e2d_batch import (
    BAIDU_ID,
    FEATHER_ID,
    OXEN_ID,
    OXEN_TOKEN,
    Env,
    _oxen_completion,
    _oxen_spec,
    _stored_text,
)

DOWNLOADS = Path.home() / "Downloads"


@pytest.fixture
def lab(settings, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    manager, workspace_id, document, source, ground_truth = _workspace_with_gt(settings)
    store = ResultStore(tmp_path / "results")
    register_test_documents(monkeypatch, tmp_path, source)
    authorize(source, store.root)
    spec = _oxen_spec()
    baidu._enable(spec, store)
    return APIEvaluationRunner(manager, settings, store), store, workspace_id, document, source, spec


def _submit(lab, transport):
    runner, _store, workspace_id, document, source, spec = lab
    preview = runner.preview(spec, workspace_id, document, source, 1)
    return runner.run(spec, workspace_id, document, source, 1, credential=OXEN_TOKEN,
                      confirmed_sha256=preview["file"]["sha256"], executor=feather._in_process(transport)), preview


# --------------------------------------------------------------- registry


def test_registry_entry_is_a_separate_oxen_provider() -> None:
    providers = {item.id: item for item in load_providers()}
    oxen, featherless = providers[OXEN_ID], providers[FEATHER_ID]
    assert oxen.runnable and oxen.endpoint == "https://hub.oxen.ai/api/ai/chat/completions"
    assert oxen.api_model_id == "qwen3-vl-4b-instruct" and oxen.allowed_hosts == {"hub.oxen.ai"}
    assert oxen.credential_env == "OXEN_API_KEY" != featherless.credential_env
    assert oxen.classification == "SAME MODEL FAMILY, REVISION UNVERIFIED" and not oxen.exact_revision_verified
    assert oxen.reported_revision == "not reported by the API"
    # Comparable with Featherless and the local registry entry.
    assert oxen.prompt_id == featherless.prompt_id and oxen.generation == featherless.generation
    assert oxen.local_counterpart == featherless.local_counterpart == "qwen3-vl-4b"
    pricing = oxen.pricing
    assert pricing.basis == "per_compute_second" and pricing.usd_per_compute_second == 0.0014
    assert pricing.compute_bound_usd() == pytest.approx(0.0014 * 180)
    assert oxen.limits.max_cost_usd_per_run >= pricing.estimated_usd_per_page >= pricing.compute_bound_usd()
    assert pricing.max_billable_seconds_per_request >= oxen.limits.request_timeout_s
    assert not pricing.price_verified and "user" in pricing.source  # honest about the rate's origin


def test_compute_pricing_bounds_are_validated() -> None:
    data = _oxen_spec().model_dump()
    with pytest.raises(ValidationError, match="rate x max billable"):
        ProviderSpec.model_validate(data | {"pricing": data["pricing"] | {"estimated_usd_per_page": 0.01}})
    with pytest.raises(ValidationError, match="needs a rate"):
        ProviderSpec.model_validate(data | {"pricing": data["pricing"] | {"usd_per_compute_second": None}})
    with pytest.raises(ValidationError, match="max_cost_usd_per_run"):
        ProviderSpec.model_validate(data | {"limits": data["limits"] | {"max_cost_usd_per_run": 0.05}})
    with pytest.raises(ValidationError, match="request timeout"):
        ProviderSpec.model_validate(data | {"pricing": data["pricing"] | {
            "max_billable_seconds_per_request": 60, "estimated_usd_per_page": 0.252}})


def test_reported_cost_parsing() -> None:
    assert reported_usd_cost("0.000120000", "USD") == pytest.approx(0.00012)
    assert reported_usd_cost(0.004) == 0.004  # Featherless: a number, no currency
    for value, currency in [("0.1", "EUR"), ("abc", "USD"), ("-1", None), ("NaN", None), (True, None), (None, None)]:
        assert reported_usd_cost(value, currency) is None


# ------------------------------------------------------------ single page


def test_one_page_request_matches_oxen_docs_and_reconciles_cost(lab) -> None:
    runner, store, *_ = lab
    transport = feather.Scripted([_oxen_completion("نص الصفحة 1,284", cost="0.012600000")])
    outcome, preview = _submit(lab, transport)
    assert outcome.state == job_store.COMPLETED
    (call,) = transport.calls
    assert call["method"] == "POST" and call["url"] == "https://hub.oxen.ai/api/ai/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {OXEN_TOKEN}"
    body = json.loads(call["body"])
    # Streaming (documented by Oxen); the preview shows exactly what is sent, and the
    # whole response must arrive within the unchanged 120 s budget.
    assert body["model"] == "qwen3-vl-4b-instruct" and body["stream"] is True
    assert preview["json_body"]["stream"] is True
    assert call["options"] == {"response_deadline_s": 120}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 2048
    text_part, image_part = body["messages"][0]["content"]
    assert text_part == {"type": "text", "text": get_prompt("vlm_transcribe_v1").text}
    url = image_part["image_url"]["url"]
    assert image_part["type"] == "image_url" and url.startswith("data:image/png;base64,")
    assert __import__("hashlib").sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest() == preview["file"]["sha256"]
    run = outcome.run
    assert run.pages[0].text == "نص الصفحة 1,284"
    assert run.usage.cost_usd == pytest.approx(0.0126) and run.usage.cost_source == "response"
    cost = run.environment["api"]["cost"]
    assert cost["reported_raw"] == "0.012600000" and cost["reported_currency"] == "USD"
    assert cost["exceeds_preflight_bound"] is False
    assert run.adapter.params["exact_revision_verified"] is False
    assert run.adapter.model_id == "Oxen.ai:qwen3-vl-4b-instruct"
    assert OXEN_TOKEN not in _stored_text(store.root)
    assert (Path(outcome.directory) / "final.json").is_file()


@pytest.mark.parametrize("cost,currency", [(None, None), ("0.5", "EUR"), ("garbage", "USD")])
def test_missing_or_unusable_cost_is_charged_at_the_compute_bound(lab, cost, currency) -> None:
    outcome, _ = _submit(lab, feather.Scripted([_oxen_completion(cost=cost, currency=currency)]))
    assert outcome.run.usage.cost_usd == pytest.approx(0.252) and outcome.run.usage.cost_source == "estimated"
    assert "compute-time bound" in outcome.run.environment["api"]["cost"]["basis"]


def test_charge_above_the_bound_is_flagged(lab) -> None:
    outcome, _ = _submit(lab, feather.Scripted([_oxen_completion(cost="0.900000000")]))
    assert outcome.run.environment["api"]["cost"]["exceeds_preflight_bound"] is True


def test_timeout_is_unconfirmed_never_retried(lab) -> None:
    runner, *_rest, spec = lab
    outcome, _ = _submit(lab, feather.Scripted([APIRequestError("request_timeout", "timed out after 120s")]))
    assert outcome.state == job_store.SUBMISSION_UNCONFIRMED
    again = feather.Scripted([_oxen_completion()])
    with pytest.raises(APIRunRefused, match="unfinished job"):
        _submit(lab, again)
    with pytest.raises(APIRunRefused, match="no job to resume"):
        runner.resume(spec, outcome.journal.journal_id, credential=OXEN_TOKEN, executor=feather._in_process(again))
    assert again.calls == []


@pytest.mark.parametrize("status,code", [(402, "payment_required"), (401, "auth_failed"), (404, "model_not_found")])
def test_provider_errors_fail_terminally_without_leaking_the_key(lab, status, code) -> None:
    runner, store, *_ = lab
    response = HTTPResponse(status, {"Content-Type": "application/json"},
                            json.dumps({"error": {"message": f"insufficient credits {OXEN_TOKEN}"}}).encode(), 3.0)
    outcome, _ = _submit(lab, feather.Scripted([response]))
    assert outcome.state == job_store.FAILED and code in outcome.run.error
    assert OXEN_TOKEN not in _stored_text(store.root)


def test_oxen_credit_error_without_message_is_failed_and_saved_atomically(lab) -> None:
    runner, store, *_ = lab
    response = HTTPResponse(
        402,
        {"Content-Type": "application/json"},
        json.dumps({
            "error": {"type": "insufficient_credits", "title": "You have run out of credits."},
            "status": "error",
            "status_message": "insufficient_credits",
        }).encode(),
        3.0,
    )
    outcome, _ = _submit(lab, feather.Scripted([response]))
    assert outcome.state == job_store.FAILED
    assert outcome.run is not None and "payment_required" in outcome.run.error
    assert outcome.artifact_directory is not None
    checked = page_artifacts.verify_snapshot(outcome.artifact_directory)
    assert checked["ok"] and not checked["reusable"]
    assert checked["metadata"]["provider_folder"] == "oxen"
    assert Path(checked["path"]).is_relative_to(store.root / "pages" / "oxen")
    assert "insufficient_credits" in (Path(checked["path"]) / "request_response.json").read_text(
        encoding="utf-8"
    )


def test_oxen_key_is_never_used_for_featherless_and_vice_versa(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    seen = []
    base = env.executor()

    def spy(spec, credential, *args):
        seen.append((spec.id, credential))
        return base(spec, credential, *args)

    # Only a Featherless key is present: no Oxen request is attempted with it.
    status = env.execute(batch_dir, spy, credentials={BAIDU_ID: baidu.TOKEN, FEATHER_ID: feather.TOKEN})
    assert {provider for provider, _ in seen} == {BAIDU_ID}
    assert all(item["status"] == batch_mod.PENDING for item in status["items"] if item["provider_id"] == OXEN_ID)
    status = env.execute(batch_dir, spy, credentials={BAIDU_ID: baidu.TOKEN, OXEN_ID: OXEN_TOKEN})
    assert {(provider, key) for provider, key in seen if provider == OXEN_ID} == {(OXEN_ID, OXEN_TOKEN)}
    assert status["overall"] == "COMPLETE"


# ------------------------------------------------------------------ batch


def test_baidu_plus_oxen_stage1_and_stage2_with_reconciled_budget(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID)
    env.authorize_all()
    env.enable_all()
    stage1 = env.plan(1)
    plan = batch_mod.load_plan(stage1)
    assert {p["provider_id"] for p in plan["providers"]} == {BAIDU_ID, OXEN_ID} == set(plan["selected_providers"])
    oxen = next(p for p in plan["providers"] if p["provider_id"] == OXEN_ID)
    assert oxen["max_cost_per_page_usd"] == pytest.approx(0.252)
    assert oxen["typical_total_cost_usd"] == pytest.approx(3 * 20 * 0.0014)
    # Budget below the worst case (3 x 0.252) but above the reconciled actual charges.
    env.approve(stage1, budgets={BAIDU_ID: 0.0, OXEN_ID: 0.30})
    status = env.execute(stage1)
    assert status["overall"] == "COMPLETE"  # 0.252 + 0.0126 + 0.0126 + ... fits only via reconciliation
    assert status["providers"][OXEN_ID]["committed_usd"] == pytest.approx(3 * 0.0126)
    assert all(item["artifact_complete"] and item["reusable_saved_result"] for item in status["items"])
    folders = {
        Path(item["artifact_directory"]).relative_to(env.root / "pages").parts[0]
        for item in status["items"]
    }
    assert folders == {"baidu", "oxen"}
    stage2 = env.plan(2, prior=stage1)
    plan2 = batch_mod.load_plan(stage2)
    assert sum(item["action"] == "reuse" for item in plan2["items"]) == 6
    env.approve(stage2, budgets={BAIDU_ID: 0.0, OXEN_ID: 0.30})
    status = env.execute(stage2)
    assert status["overall"] == "COMPLETE" and status["counts"][batch_mod.REUSED] == 6


def test_charge_above_bound_stops_submissions_to_that_provider(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID, oxen_cost="0.400000000")
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir, budgets={BAIDU_ID: 0.0, OXEN_ID: 5.0})
    status = env.execute(batch_dir)
    oxen_items = [item for item in status["items"] if item["provider_id"] == OXEN_ID]
    assert sum(item["status"] == batch_mod.COMPLETED for item in oxen_items) == 1
    assert sum(item["status"] == batch_mod.PENDING for item in oxen_items) == 2
    assert "price assumption is wrong" in next(i for i in oxen_items if i["status"] == batch_mod.PENDING)["detail"]
    assert all(item["status"] == batch_mod.COMPLETED for item in status["items"] if item["provider_id"] == BAIDU_ID)


def test_provider_selection_is_validated(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID, options=(BAIDU_ID, OXEN_ID, FEATHER_ID))
    assert env.definition.providers == [BAIDU_ID, OXEN_ID]
    with pytest.raises(batch_mod.BatchError, match="Not a provider option"):
        env.definition.selection(["unknown-provider"])
    with pytest.raises(batch_mod.BatchError, match="distinct"):
        env.definition.selection([OXEN_ID, OXEN_ID])
    plan = batch_mod.build_plan(env.definition, env.definition_sha, 1, sources=env.sources, specs=env.specs,
                                manager=env.manager, settings=env.settings, workspace_id=env.workspace_id,
                                root=env.root, providers=[BAIDU_ID, FEATHER_ID])
    assert plan["selected_providers"] == [BAIDU_ID, FEATHER_ID]


def test_shipped_definition_defaults_to_baidu_and_oxen() -> None:
    definition, _ = batch_mod.load_definition()
    assert definition.providers == [BAIDU_ID, OXEN_ID]
    assert set(definition.options) == {BAIDU_ID, OXEN_ID, FEATHER_ID}


# ------------------------------------------------- user-owned documents


def test_user_owned_category_requires_its_own_declaration(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID, category="user_owned_test")
    source = env.sources["doc-ar"]
    assert source.entry.authorization_category == "user_owned_test"
    assert source.entry.declaration == documents.USER_OWNED_DECLARATION
    kwargs = {"file_path": source.pdf_path, "reviewer": "T", "days_valid": 1, "root": env.root}
    with pytest.raises(documents.DocumentAuthorizationError, match="user-owned"):
        documents.authorize_document(source.entry, declaration=documents.SYNTHETIC_DECLARATION, **kwargs)
    state = documents.authorization_state(source.entry.sha256, root=env.root)
    assert not state["allowed"] and "No explicit authorization" in state["reasons"][0]
    path = documents.authorize_document(source.entry, declaration=documents.USER_OWNED_DECLARATION, **kwargs)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["authorization_category"] == "user_owned_test" and record["declaration"] == documents.USER_OWNED_DECLARATION
    assert documents.authorization_state(source.entry.sha256, root=env.root)["allowed"]
    # Same revocation and expiry semantics as synthetic documents.
    documents.revoke_document(source.entry.sha256, revoked_by="T", reason="done", root=env.root)
    assert not documents.authorization_state(source.entry.sha256, root=env.root)["allowed"]


def test_synthetic_entry_refuses_the_user_owned_declaration_and_category_changes_invalidate(
        settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch)
    source = env.sources["doc-ar"]
    kwargs = {"file_path": source.pdf_path, "reviewer": "T", "days_valid": 1, "root": env.root}
    with pytest.raises(documents.DocumentAuthorizationError, match="synthetic"):
        documents.authorize_document(source.entry, declaration=documents.USER_OWNED_DECLARATION, **kwargs)
    documents.authorize_document(source.entry, declaration=documents.SYNTHETIC_DECLARATION, **kwargs)
    assert documents.authorization_state(source.entry.sha256, root=env.root)["allowed"]
    # Re-classifying the entry after authorization never carries the old approval over.
    registry = documents.DOCUMENT_REGISTRY
    text = registry.read_text(encoding="utf-8").replace('authorization_category = "synthetic"',
                                                        'authorization_category = "user_owned_test"')
    registry.write_text(text, encoding="utf-8")
    state = documents.authorization_state(source.entry.sha256, root=env.root)
    assert not state["allowed"] and any("registry entry changed" in reason for reason in state["reasons"])


def test_user_owned_documents_stay_blocked_until_authorized_then_run(settings, tmp_path, monkeypatch) -> None:
    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID, category="user_owned_test")
    env.enable_all()
    batch_dir = env.plan(1)
    env.approve(batch_dir)
    status = env.execute(batch_dir)
    assert env.sends == [] and {item["status"] for item in status["items"]} == {batch_mod.REFUSED}
    env.authorize_all()  # uses each entry's own (user-owned) declaration
    assert env.execute(batch_dir)["overall"] == "COMPLETE"


TRAFFIC = {"traffic-accident-report-ar-v1", "traffic-accident-report-en-v1"}


def test_shipped_traffic_reports_are_user_owned_and_never_synthetic() -> None:
    registry = {entry.id: entry for entry in documents.load_registry().values()}
    for entry_id in TRAFFIC:
        entry = registry[entry_id]
        assert entry.authorization_category == "user_owned_test"
        assert entry.declaration == documents.USER_OWNED_DECLARATION != documents.SYNTHETIC_DECLARATION
        assert "not independently verified" in entry.provenance
    for entry_id in ("agenticdoc-arabic-benchmark-v1", "agenticdoc-english-benchmark-v1"):
        assert registry[entry_id].authorization_category == "synthetic"


# ------------------------------------------------------------- Streamlit


def test_streamlit_offers_baidu_plus_oxen_and_the_user_owned_declaration(settings, tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    import lab.core.e2.api.providers as providers_mod
    import lab.core.e2.api.runner as api_runner

    env = Env(settings, tmp_path, monkeypatch, second=OXEN_ID, category="user_owned_test",
              options=(BAIDU_ID, OXEN_ID, FEATHER_ID))
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(env.root))
    for name in ("PADDLEOCR_ACCESS_TOKEN", "FEATHERLESS_API_KEY", "OXEN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FEATHERLESS_API_KEY", "FEATHERLESS-ENV-must-not-be-used-for-oxen")
    monkeypatch.setattr(providers_mod, "load_providers", lambda path=None: list(env.specs.values()))
    monkeypatch.setattr(api_runner, "subprocess_executor", lambda *a, **k: env.executor())
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "8_Two_Stage_API_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state["workspace_id"] = env.workspace_id
    app.run()
    assert not app.exception
    assert app.multiselect(key="e2d-providers").value == [BAIDU_ID, OXEN_ID]
    labels = [item.label for item in app.checkbox]
    assert documents.USER_OWNED_DECLARATION in labels and documents.SYNTHETIC_DECLARATION not in labels
    assert any("Oxen API key" in item.label for item in app.text_input)
    # The Featherless environment key never satisfies Oxen.
    assert not any("FEATHERLESS_API_KEY" in item.value and "Oxen" in item.value for item in app.caption)
    # Authorize both documents through the UI with the user-owned declaration.
    for source in env.sources.values():
        sha = source.entry.sha256
        app.text_input(key=f"doc-auth-reviewer-{sha}").input("Tester")
        app.text_input(key=f"doc-auth-sha-{sha}").input(sha[:12])
        app.checkbox(key=f"doc-auth-agree-{sha}").check()
        # Each unauthorized document has its own form; the first remaining one is this one.
        next(item for item in app.button if item.label == "Authorize this exact document").click().run()
        assert not app.exception
        assert documents.authorization_state(sha, root=env.root)["allowed"]
    assert all(documents.authorization_state(s.entry.sha256, root=env.root)["allowed"] for s in env.sources.values())
    env.enable_all()
    next(item for item in app.button if item.label.startswith("Create a new Stage 1 plan")).click().run()
    batch_dir = batch_mod.list_batches(env.root, "e2d-test", stage=1)[-1]
    plan = batch_mod.load_plan(batch_dir)
    assert plan["selected_providers"] == [BAIDU_ID, OXEN_ID]
    key = plan["batch_id"]
    app.text_input(key=f"e2d-approve-by-{key}").input("Tester")
    app.text_input(key=f"e2d-approve-sha-{key}").input(plan["plan_sha256"][:12])
    app.number_input(key=f"e2d-budget-{key}-{OXEN_ID}").set_value(0.30)
    app.checkbox(key=f"e2d-approve-agree-{key}").check()
    next(item for item in app.button if item.label == "Approve this exact Stage 1 plan").click().run()
    assert batch_mod.approval_state(batch_dir)["allowed"]
    app.text_input(key=f"e2d-credential-{BAIDU_ID}").input(baidu.TOKEN)
    app.run()
    assert next(item for item in app.button if item.label == "Run Stage 1").disabled  # Oxen key still missing
    app.text_input(key=f"e2d-credential-{OXEN_ID}").input(OXEN_TOKEN)
    app.text_input(key="e2d-operator-1").input("Tester")
    app.run()
    assert env.sends == []
    next(item for item in app.button if item.label == "Run Stage 1").click().run()
    deadline = time.monotonic() + 60
    while batch_mod.batch_status(batch_dir, env.root)["overall"] != "COMPLETE" and time.monotonic() < deadline:
        time.sleep(0.2)
    assert batch_mod.batch_status(batch_dir, env.root)["overall"] == "COMPLETE"
    assert sorted({provider for provider, _ in env.sends}) == [BAIDU_ID, OXEN_ID]
    assert OXEN_TOKEN not in _stored_text(env.root)


@pytest.mark.skipif(not (DOWNLOADS / "Traffic_Accident_Report_English.pdf").is_file(), reason="user's PDF absent")
def test_real_traffic_report_refuses_the_synthetic_declaration(settings, tmp_path) -> None:
    entry = next(item for item in documents.load_registry().values() if item.id == "traffic-accident-report-en-v1")
    with pytest.raises(documents.DocumentAuthorizationError):
        documents.authorize_document(entry, file_path=DOWNLOADS / "Traffic_Accident_Report_English.pdf",
                                     reviewer="T", declaration=documents.SYNTHETIC_DECLARATION, days_valid=1,
                                     root=tmp_path / "results")
    assert not (tmp_path / "results" / "api_document_authorizations").exists() or not any(
        (tmp_path / "results" / "api_document_authorizations").rglob("*.json"))
