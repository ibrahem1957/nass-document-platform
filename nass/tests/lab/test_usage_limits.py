"""Internal usage limits: enablement-scoped budget, UTC daily limit, cumulative cap and
registry-configured ceilings. Synthetic ledgers only; no request is sent."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab.core.e2.api.authorization import (
    REQUIRED_STATEMENT,
    UNCONFIGURED_PAGE_CEILING,
    AuthorizationError,
    check_authorization,
    record_enablement,
    usage_policy,
)
from lab.core.e2.api.providers import load_providers

BAIDU = "baidu-aistudio-paddleocr-vl-1.6"


def _spec(**limits):
    spec = next(item for item in load_providers() if item.id == BAIDU)
    return spec.model_copy(update={"limits": spec.limits.model_copy(update=limits)}) if limits else spec


def _ledger(root: Path, spec, entries: list[dict]) -> Path:
    path = root / "api_usage" / f"{spec.id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    return path


def _submitted(at: datetime, count: int) -> list[dict]:
    return [{"at": at.isoformat(), "run_id": f"run_{i:018x}", "pages": 1, "submitted": True} for i in range(count)]


def _enable(spec, root, total=10, day=10):
    return record_enablement(spec, enabled=True, enabled_by="Tester", statement=REQUIRED_STATEMENT,
                             max_pages_total=total, max_pages_per_day=day, days_valid=1, root=root)


def test_baidu_policy_comes_from_the_registry() -> None:
    policy = usage_policy(_spec())
    assert policy["max_pages_per_enablement"] == 500 and policy["max_pages_per_day_ceiling"] == 100
    assert policy["default_enablement_days"] == 7 and policy["max_pages_cumulative"] == 2000
    assert "not provider quotas" in policy["source"]


def test_unconfigured_provider_keeps_the_previous_ceiling() -> None:
    spec = _spec(max_pages_per_enablement=None, max_pages_per_day_ceiling=None, max_pages_cumulative=None)
    assert usage_policy(spec)["max_pages_per_enablement"] == UNCONFIGURED_PAGE_CEILING


def test_enablement_budget_counts_only_since_the_enablement_started(tmp_path) -> None:
    spec = _spec()
    _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc) - timedelta(days=2), 96))  # earlier history
    _enable(spec, tmp_path, total=20, day=20)
    state = check_authorization(spec, pages=1, root=tmp_path)
    assert state.allowed and state.pages_used_total == 96 and state.pages_used_enablement == 0
    assert state.remaining_pages == 20


def test_daily_limit_counts_the_utc_day_across_enablements(tmp_path) -> None:
    spec = _spec()
    _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc) - timedelta(seconds=30), 5))
    _enable(spec, tmp_path, total=10, day=6)
    state = check_authorization(spec, pages=1, root=tmp_path)
    assert state.pages_used_enablement == 0 and state.pages_used_today == 5 and state.remaining_pages == 1
    _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc), 1))
    state = check_authorization(spec, pages=1, root=tmp_path)
    assert not state.allowed and any("Daily page limit reached" in r for r in state.reasons)


def test_enablement_budget_and_unreadable_timestamps_fail_closed(tmp_path) -> None:
    spec = _spec()
    _enable(spec, tmp_path, total=2, day=2)
    _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc), 1) + [{"at": "bad", "pages": 1, "submitted": True}])
    state = check_authorization(spec, pages=1, root=tmp_path)
    assert not state.allowed and any("Enablement page limit reached" in r for r in state.reasons)


def test_cumulative_cap_stops_re_enabling_from_raising_the_budget(tmp_path) -> None:
    spec = _spec(max_pages_cumulative=50)
    _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc) - timedelta(days=3), 50))
    _enable(spec, tmp_path, total=100, day=100)
    state = check_authorization(spec, pages=1, root=tmp_path)
    assert not state.allowed and any("Cumulative page limit reached" in r for r in state.reasons)
    assert state.remaining_pages == 0


def test_ceilings_are_enforced_and_lowered_ceilings_apply_at_once(tmp_path) -> None:
    spec = _spec()
    with pytest.raises(AuthorizationError, match="500"):
        _enable(spec, tmp_path, total=501, day=50)
    with pytest.raises(AuthorizationError):
        _enable(spec, tmp_path, total=200, day=101)
    _enable(spec, tmp_path, total=300, day=100)
    lowered = _spec(max_pages_per_enablement=50, max_pages_per_day_ceiling=20)
    state = check_authorization(lowered, pages=1, root=tmp_path)
    assert state.limit_enablement == 50 and state.limit_day == 20 and state.remaining_pages == 20


def test_documented_provider_quota_lowers_the_daily_ceiling() -> None:
    assert usage_policy(_spec(provider_documented_daily_pages=30))["max_pages_per_day_ceiling"] == 30


def test_checks_and_new_enablements_never_change_the_usage_ledger(tmp_path) -> None:
    spec = _spec()
    path = _ledger(tmp_path, spec, _submitted(datetime.now(timezone.utc) - timedelta(days=1), 7))
    before = path.read_bytes()
    _enable(spec, tmp_path, total=10, day=10)
    check_authorization(spec, pages=1, root=tmp_path)
    record_enablement(spec, enabled=False, enabled_by="Tester", statement="", max_pages_total=0,
                      max_pages_per_day=0, days_valid=1, root=tmp_path)
    _enable(spec, tmp_path, total=10, day=10)
    assert path.read_bytes() == before
    records = sorted((tmp_path / "api_authorizations" / spec.id).glob("enablement_*.json"))
    assert len(records) == 3  # every enablement record is kept
    assert json.loads(records[-1].read_text(encoding="utf-8"))["usage_policy"]["max_pages_per_enablement"] == 500
