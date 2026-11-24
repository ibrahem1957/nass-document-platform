"""E2-D reuse of previously saved, completed page results (manual or batch).

Reuse is decided by exact document SHA-256 (upload-session document ids are ignored),
page, provider, full identity and snapshot integrity. All provider calls are mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api import batch_report
from lab.core.e2.api import jobs as job_store
from tests.lab.test_e2d_batch import BAIDU_ID, CREDS, OXEN_ID, Env


@pytest.fixture
def env(settings, tmp_path, monkeypatch):
    return Env(settings, tmp_path, monkeypatch, second=OXEN_ID)


def _manual_run(env: Env, provider_id: str, page: int = 1, registry_id: str = "doc-ar", executor=None):
    """A manual single-page run from a *separate upload session* of the same PDF."""
    source = env.sources[registry_id]
    workspace = env.manager.create().workspace_id
    with source.pdf_path.open("rb") as handle:
        env.ingest.ingest(workspace, handle, source.pdf_path.name)
    (document,) = env.ingest.list_documents(workspace)
    assert document.doc_id != source.document.doc_id and document.sha256 == source.entry.sha256
    pdf = env.ingest.original_path(workspace, document.doc_id)
    spec = env.specs[provider_id]
    preview = env.runner.preview(spec, workspace, document, pdf, page)
    return env.runner.run(spec, workspace, document, pdf, page, credential=CREDS[provider_id],
                          confirmed_sha256=preview["file"]["sha256"],
                          executor=executor or env.executor())


def _item(plan: dict, provider_id: str, page: int = 1, registry_id: str = "doc-ar") -> dict:
    return next(item for item in plan["items"] if item["provider_id"] == provider_id
                and item["page"] == page and item["registry_id"] == registry_id)


def _ready(env):
    env.authorize_all()
    env.enable_all()


def test_manual_results_from_another_session_are_reused_in_stage1_and_stage2(env) -> None:
    _ready(env)
    manual = {provider: _manual_run(env, provider) for provider in (BAIDU_ID, OXEN_ID)}
    assert all(outcome.state == job_store.COMPLETED for outcome in manual.values())
    sent_before = len(env.sends)

    stage1 = env.plan(1)
    plan = batch_mod.load_plan(stage1)
    for provider in (BAIDU_ID, OXEN_ID):
        item = _item(plan, provider)
        assert item["action"] == "reuse", item["reason"]
        reuse = item["reuse_of"]
        assert reuse["source"] == "saved_page_artifact"
        assert reuse["run_id"] == manual[provider].journal.journal_id
        assert Path(reuse["artifact_directory"]) == Path(manual[provider].artifact_directory)
        assert reuse["document_id_at_run"] != item["doc_id"]  # matched by SHA-256, not session id
        assert item["max_cost_usd"] == 0.0
    summary = {p["provider_id"]: p for p in plan["providers"]}
    assert summary[BAIDU_ID]["reuse_count"] == summary[OXEN_ID]["reuse_count"] == 1
    assert summary[OXEN_ID]["submit_count"] == 2  # Arabic p3 and English p2 only

    env.approve(stage1)
    status = env.execute(stage1)
    assert status["overall"] == "COMPLETE" and status["counts"][batch_mod.REUSED] == 2
    # Only the 2 non-reused pages x 2 providers were sent; page 1 was not repeated.
    assert len(env.sends) - sent_before == 4
    rows = {row["item_id"]: row for row in batch_report.analyze_batch(
        stage1, env.root, sources=env.sources)["rows"]}
    reused_row = rows[_item(plan, OXEN_ID)["item_id"]]
    assert reused_row["reused"] and reused_row["status"] == batch_mod.REUSED
    assert "analysis" in reused_row  # the saved text is analysed like a fresh result

    sent_stage1 = len(env.sends)
    stage2 = batch_mod.load_plan(env.plan(2, prior=stage1))
    for provider in (BAIDU_ID, OXEN_ID):
        item = _item(stage2, provider)
        assert item["action"] == "reuse"
        assert item["reuse_of"]["run_id"] == manual[provider].journal.journal_id
    assert sum(item["action"] == "reuse" for item in stage2["items"]) == 6  # all Stage 1 pages
    assert len(env.sends) == sent_stage1


def test_identity_mismatch_is_reported_and_never_reused(env) -> None:
    _ready(env)
    _manual_run(env, OXEN_ID)
    env.specs[OXEN_ID] = env.specs[OXEN_ID].model_copy(
        update={"generation": {"temperature": 0.0, "max_tokens": 4096}})
    item = _item(batch_mod.load_plan(env.plan(1)), OXEN_ID)
    assert item["action"] == "submit"
    assert "Saved completed result not reused" in item["reason"]
    assert "generation" in item["reason"] and "fingerprint" in item["reason"]


def test_tampered_missing_or_unfinished_results_are_never_reused(env) -> None:
    _ready(env)
    outcome = _manual_run(env, OXEN_ID)
    artifact = Path(outcome.artifact_directory)
    (artifact / "raw_text.txt").chmod(0o666)
    (artifact / "raw_text.txt").write_text("tampered", encoding="utf-8")
    item = _item(batch_mod.load_plan(env.plan(1)), OXEN_ID)
    assert item["action"] == "submit" and "modified raw_text.txt" in item["reason"]


def test_unconfirmed_or_failed_requests_are_never_reused(env) -> None:
    from lab.core.e2.api.client import APIRequestError

    _ready(env)
    unconfirmed = _manual_run(env, OXEN_ID, executor=env.executor(
        feather_script=lambda: [APIRequestError("request_timeout", "timed out")]))
    assert unconfirmed.state == job_store.SUBMISSION_UNCONFIRMED
    item = _item(batch_mod.load_plan(env.plan(1)), OXEN_ID)
    assert item["action"] == "submit" and item.get("reuse_of") is None


def test_legacy_result_without_a_page_snapshot_is_not_reused(env) -> None:
    """Like the pre-artifact Baidu manual run: integrity of an aggregate run alone is
    not enough, because its implementation version and rendering label were never
    recorded with a page snapshot."""
    _ready(env)
    outcome = _manual_run(env, BAIDU_ID)
    shutil.rmtree(Path(outcome.artifact_directory).parent.parent.parent)  # test data only
    item = _item(batch_mod.load_plan(env.plan(1)), BAIDU_ID)
    assert item["action"] == "submit" and item.get("reuse_of") is None


def test_other_documents_and_pages_are_not_matched(env) -> None:
    _ready(env)
    _manual_run(env, OXEN_ID, page=2)  # Arabic page 2 is not a Stage 1 page
    _manual_run(env, OXEN_ID, page=1, registry_id="doc-en")
    plan = batch_mod.load_plan(env.plan(1))
    assert _item(plan, OXEN_ID)["action"] == "submit"
    assert _item(plan, OXEN_ID, page=2, registry_id="doc-en")["action"] == "submit"


def test_reused_snapshot_altered_after_planning_is_not_shown_as_reused(env) -> None:
    _ready(env)
    outcome = _manual_run(env, OXEN_ID)
    stage1 = env.plan(1)
    artifact = Path(outcome.artifact_directory)
    (artifact / "raw_text.txt").chmod(0o666)
    (artifact / "raw_text.txt").write_text("tampered", encoding="utf-8")
    item = _item(batch_mod.load_plan(stage1), OXEN_ID)
    status = batch_mod.item_status(stage1, item, env.root)
    assert status["status"] == batch_mod.INCOMPLETE_ARTIFACT
    assert batch_mod.INCOMPLETE_ARTIFACT not in batch_mod.SUBMITTABLE  # never auto-resubmitted
    assert "create a new plan" in status["detail"]
