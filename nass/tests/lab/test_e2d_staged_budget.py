"""The per-batch spending cap and staged execution across batches.

The cap is a backend safety limit (not raised here). A plan whose worst case exceeds
it runs in stages: each batch sends pages only while its approved budget covers them,
and a follow-up plan reuses every completed page, so no completed request is ever
repeated. All provider calls are mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.core.e2.api import batch as batch_mod
from lab.core.e2.api.client import APIRequestError
from tests.lab.test_e2d_batch import BAIDU_ID, OXEN_ID, Env

BOUND = 0.252


@pytest.fixture
def env(settings, tmp_path, monkeypatch):
    # Oxen streams report no cost, so every page is charged at the bound (as observed live).
    return Env(settings, tmp_path, monkeypatch, second=OXEN_ID, oxen_cost=None)


def _approve(env, batch_dir, oxen_budget):
    return env.approve(batch_dir, budgets={BAIDU_ID: 0.0, OXEN_ID: oxen_budget})


def _oxen_sends(env) -> int:
    return sum(provider == OXEN_ID for provider, _ in env.sends)


def _items(env, batch_dir, provider=OXEN_ID):
    return [item for item in batch_mod.batch_status(batch_dir, env.root)["items"] if item["provider_id"] == provider]


def test_cap_is_enforced_by_the_backend_not_only_the_form(env) -> None:
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    assert batch_mod.MAX_BATCH_BUDGET_USD == 5.0  # unchanged
    with pytest.raises(batch_mod.BatchError, match=r"must be 0\.\.5\.0 USD"):
        _approve(env, batch_dir, 6.552)
    with pytest.raises(batch_mod.BatchError, match="single page"):
        _approve(env, batch_dir, BOUND - 0.01)
    _approve(env, batch_dir, 5.0)
    assert batch_mod.approval_state(batch_dir)["allowed"]


def test_staged_batches_stay_within_the_cap_and_never_repeat_completed_pages(env, monkeypatch) -> None:
    monkeypatch.setattr(batch_mod, "MAX_BATCH_BUDGET_USD", 0.6)  # room for 2 bounded Oxen pages
    env.authorize_all()
    env.enable_all()
    first = env.plan(1)
    plan = batch_mod.load_plan(first)
    oxen = next(p for p in plan["providers"] if p["provider_id"] == OXEN_ID)
    assert oxen["submit_count"] == 3 and oxen["max_total_cost_usd"] == pytest.approx(3 * BOUND)
    with pytest.raises(batch_mod.BatchError):
        _approve(env, first, oxen["max_total_cost_usd"])  # above the cap: refused
    _approve(env, first, 0.6)
    status = env.execute(first)
    assert _oxen_sends(env) == 2  # 2 x 0.252 <= 0.6 < 3 x 0.252
    assert status["providers"][OXEN_ID]["committed_usd"] == pytest.approx(2 * BOUND)
    assert status["providers"][OXEN_ID]["committed_usd"] <= 0.6
    pending = [item for item in _items(env, first) if item["status"] == batch_mod.PENDING]
    assert len(pending) == 1 and "budget" in pending[0]["detail"]
    assert all(item["status"] == batch_mod.COMPLETED for item in _items(env, first, BAIDU_ID))

    # Continuation: a new plan reuses everything already completed.
    second = env.plan(1)
    plan = batch_mod.load_plan(second)
    actions = {item["item_id"]: item["action"] for item in plan["items"]}
    assert actions[pending[0]["item_id"]] == "submit"
    assert sum(action == "reuse" for action in actions.values()) == 5  # 3 Baidu + 2 Oxen
    oxen = next(p for p in plan["providers"] if p["provider_id"] == OXEN_ID)
    assert oxen["submit_count"] == 1 and oxen["max_total_cost_usd"] == pytest.approx(BOUND)
    assert next(p for p in plan["providers"] if p["provider_id"] == BAIDU_ID)["submit_count"] == 0
    _approve(env, second, BOUND)
    status = env.execute(second)
    assert status["overall"] == "COMPLETE" and _oxen_sends(env) == 3
    total = sum(len(_items(env, path, provider)) for path in (second,) for provider in (OXEN_ID, BAIDU_ID))
    assert total == 6

    # Resuming the first batch later sends nothing (its budget is used up).
    batch_mod.request_control(first, "RUN", by="Tester")
    env.execute(first)
    assert _oxen_sends(env) == 3 and len(env.sends) == 6  # every page sent exactly once


def test_an_older_batch_with_budget_left_never_resends_a_page_completed_elsewhere(env) -> None:
    env.authorize_all()
    env.enable_all()
    first = env.plan(1)
    _approve(env, first, 1.0)
    # First run without the Oxen key: Baidu completes, no Oxen page is sent.
    batch_mod.request_control(first, "RUN", by="Tester")
    batch_mod.BatchRunner(env.runner, env.specs, executor=env.executor()).execute(
        first, credentials={BAIDU_ID: "baidu-token"}, sources=env.sources)
    assert _oxen_sends(env) == 0
    # A later plan sends the Oxen pages (and reuses the Baidu ones).
    second = env.plan(1)
    _approve(env, second, 1.0)
    assert env.execute(second)["overall"] == "COMPLETE"
    assert _oxen_sends(env) == 3
    # The first batch still has budget for all 3 Oxen pages, but sends none of them.
    batch_mod.request_control(first, "RUN", by="Tester")
    env.execute(first)
    assert _oxen_sends(env) == 3
    for item in _items(env, first):
        assert item["status"] == batch_mod.PENDING and "Already completed in run" in item["detail"]


def test_continuation_plan_blocks_pages_with_unfinished_jobs(env) -> None:
    env.authorize_all()
    env.enable_all()
    first = env.plan(1)
    _approve(env, first, 1.0)
    calls = {"n": 0}

    def oxen_script():
        calls["n"] += 1
        if calls["n"] == 1:
            return [APIRequestError("request_timeout", "Request timed out after 120.0s during waiting_response_headers")]
        from tests.lab.test_e2d_batch import _oxen_completion
        return [_oxen_completion("text", cost=None)]

    env.execute(first, env.executor(feather_script=oxen_script))
    (undecided,) = [item for item in _items(env, first) if item["status"] == batch_mod.NEEDS_DECISION]
    second = batch_mod.load_plan(env.plan(1))
    item = next(entry for entry in second["items"] if entry["item_id"] == undecided["item_id"])
    assert item["action"] == "blocked" and undecided["journal_id"] in item["reason"]
    oxen = next(p for p in second["providers"] if p["provider_id"] == OXEN_ID)
    assert oxen["submit_count"] == 0 and oxen["blocked_count"] == 1 and oxen["reuse_count"] == 2


def test_two_batches_of_one_evaluation_never_execute_at_once(env) -> None:
    env.authorize_all()
    env.enable_all()
    first = env.plan(1)
    second = env.plan(1)
    _approve(env, second, 1.0)
    batch_mod.request_control(second, "RUN", by="Tester")
    with batch_mod.executor_lock(first):
        with pytest.raises(batch_mod.BatchError, match="Another batch of this evaluation is executing"):
            batch_mod.BatchRunner(env.runner, env.specs, executor=env.executor()).execute(
                second, credentials={BAIDU_ID: "t", OXEN_ID: "t"}, sources=env.sources)
    assert env.sends == []


def test_form_explains_staging_when_the_plan_exceeds_the_cap(env, monkeypatch, settings) -> None:
    from streamlit.testing.v1 import AppTest

    import lab.core.e2.api.providers as providers_mod
    import lab.core.e2.api.runner as api_runner

    monkeypatch.setattr(batch_mod, "MAX_BATCH_BUDGET_USD", 0.6)
    env.authorize_all()
    env.enable_all()
    batch_dir = env.plan(1)
    monkeypatch.setenv("LAB_TMP_DIR", settings.tmp_dir)
    monkeypatch.setenv("AGENTICDOC_RESULTS_DIR", str(env.root))
    monkeypatch.setattr(providers_mod, "load_providers", lambda path=None: list(env.specs.values()))
    monkeypatch.setattr(api_runner, "subprocess_executor", lambda *a, **k: env.executor())
    page = Path(__file__).resolve().parents[2] / "lab" / "app" / "pages" / "8_Two_Stage_API_Evaluation.py"
    app = AppTest.from_file(str(page), default_timeout=90)
    app.session_state["workspace_id"] = env.workspace_id
    app.run()
    assert not app.exception
    staged = [item.value for item in app.info if "Staged execution" in item.value]
    assert len(staged) == 1 and OXEN_ID in staged[0] and "at most 2 of 3 pages" in staged[0]
    budget = app.number_input(key=f"e2d-budget-{batch_mod.load_plan(batch_dir)['batch_id']}-{OXEN_ID}")
    assert budget.value == pytest.approx(0.6) and budget.max == pytest.approx(0.6)  # defaults to the cap
    assert not batch_mod.approval_state(batch_dir)["allowed"]  # rendering approves nothing
