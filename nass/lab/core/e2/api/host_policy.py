"""Opt-in, per-batch automatic approval of Baidu result-storage hosts (E2-D).

OFF by default: every paused Baidu job needs a manual, job-scoped host approval.
The operator may enable "Trust approved Baidu result hosts for this batch" once,
with a name, a statement and an expiry. While it is active, a paused Baidu job of
THIS batch is approved automatically only when ALL of these hold:

- the job journal is awaiting result-host approval, for this provider endpoint and
  model, with a recorded Job ID, and belongs to an item of this batch;
- the job paused after the policy was enabled, unless the operator explicitly chose
  to include jobs that were already pending;
- its recorded result URL is HTTPS on the default port, with a `/v1/job/` path;
- its host is a valid public DNS name (never an IP, localhost or local name) and is
  either (a) listed exactly in the registry `auto_result_hosts`, (b) a host the
  operator approved manually for another job of this batch, or (c) matches the
  narrow pattern `paddleocr-store-<1-2 digits>.bj.bcebos.com` recorded in the consent.

Anything else stays manual. An automatic approval is the SAME job-scoped, expiring
record a manual approval creates (jobs.approve_result_host), marked as automatic, and
is also written to this batch's audit log. Downloads keep every existing check: the
client accepts only HTTPS, default port, no credentials, no IP hosts and exactly the
host approved for that job (redirects are re-validated against it), and the network
guard connects only to that host and refuses non-public resolved addresses.
Submission, spending, other providers (e.g. Oxen) and saved results are unaffected.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lab.core.e2.api import jobs as job_store
from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced

POLICY_PROTOCOL = "baidu_paddleocr_jobs_v2"
MAX_POLICY_HOURS = 72
RESULT_PATH_PREFIX = "/v1/job/"
BAIDU_RESULT_HOST_PATTERN = r"^paddleocr-store-[0-9]{1,2}\.bj\.bcebos\.com$"
POLICY_STATEMENT = (
    "I authorize automatic result-host approval for the Baidu jobs of this batch until the expiry I set: a paused "
    "job's result may be downloaded without asking me per job only when Baidu's job-status API reported an HTTPS "
    "result URL on an exactly listed or previously approved Baidu result-storage host, or on a host matching "
    "paddleocr-store-<number>.bj.bcebos.com. Any other host still needs my manual approval, page images are still "
    "sent only through the approved plan, and every automatic approval is logged."
)


class HostPolicyError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _directory(batch_dir: Path) -> Path:
    return Path(batch_dir) / "host_policy"


def supported(spec: ProviderSpec) -> bool:
    return spec.protocol == POLICY_PROTOCOL


def enable_policy(batch_dir: Path, spec: ProviderSpec, *, enabled_by: str, statement: str,
                  hours_valid: int, include_pending: bool = False) -> Path:
    if not supported(spec):
        raise HostPolicyError("Automatic result-host approval is only available for Baidu result retrieval")
    if statement != POLICY_STATEMENT:
        raise HostPolicyError("The automatic-approval statement was not accepted")
    if not enabled_by.strip():
        raise HostPolicyError("Enter the name of the person enabling the policy")
    if not 1 <= int(hours_valid) <= MAX_POLICY_HOURS:
        raise HostPolicyError(f"Validity must be 1..{MAX_POLICY_HOURS} hours")
    if not (Path(batch_dir) / "plan.json").is_file():
        raise HostPolicyError("Not a batch directory")
    now = _now()
    return write_sequenced(_directory(batch_dir), "policy", {
        "schema_version": 1, "enabled": True, "batch": Path(batch_dir).name, "provider_id": spec.id,
        "endpoint": spec.endpoint, "api_model_id": spec.api_model_id,
        "listed_hosts": sorted({job_store.validate_result_hostname(host) for host in spec.auto_result_hosts}),
        "host_pattern": BAIDU_RESULT_HOST_PATTERN, "include_pending": bool(include_pending),
        "enabled_by": enabled_by.strip(), "statement": statement,
        "created_at": now.isoformat(), "expires_at": (now + timedelta(hours=int(hours_valid))).isoformat(),
    })


def revoke_policy(batch_dir: Path, spec: ProviderSpec, *, revoked_by: str) -> Path:
    if not revoked_by.strip():
        raise HostPolicyError("Enter the name of the person turning the policy off")
    return write_sequenced(_directory(batch_dir), "policy", {
        "schema_version": 1, "enabled": False, "batch": Path(batch_dir).name, "provider_id": spec.id,
        "endpoint": spec.endpoint, "revoked_by": revoked_by.strip(), "created_at": _now().isoformat(),
    })


def policy_state(batch_dir: Path, spec: ProviderSpec) -> dict[str, Any]:
    """The newest record for this provider decides; unreadable records fail closed (OFF)."""
    records = ordered(_directory(batch_dir), "policy")
    if any(payload is None for _path, payload in records):
        return {"active": False, "reasons": ["a policy record is unreadable"], "record": None, "path": None}
    mine = [(path, payload) for path, payload in records if payload.get("provider_id") == spec.id]
    if not mine:
        return {"active": False, "reasons": ["off (manual per-job approval)"], "record": None, "path": None}
    path, latest = mine[-1]
    reasons = []
    if not latest.get("enabled"):
        reasons.append(f"turned off by {latest.get('revoked_by')}")
    else:
        if latest.get("statement") != POLICY_STATEMENT:
            reasons.append("statement not recorded")
        if latest.get("batch") != Path(batch_dir).name:
            reasons.append("recorded for a different batch")
        if latest.get("endpoint") != spec.endpoint or latest.get("api_model_id") != spec.api_model_id:
            reasons.append("recorded for a different provider endpoint or model")
        if latest.get("host_pattern") != BAIDU_RESULT_HOST_PATTERN:
            reasons.append("recorded host pattern differs")
        try:
            if datetime.fromisoformat(latest["expires_at"]) <= _now():
                reasons.append("expired")
        except (KeyError, ValueError):
            reasons.append("no valid expiry")
    if not supported(spec):
        reasons.append("not a Baidu provider")
    return {"active": not reasons, "reasons": reasons, "record": latest, "path": str(path)}


def _manually_approved_hosts(spec: ProviderSpec, batch_journals: set[str], root: Path | None) -> set[str]:
    """Hosts the operator approved by hand for jobs of this batch (still valid)."""
    hosts: set[str] = set()
    for record in job_store.host_approval_records(spec, root):
        if (record.get("approved") and record.get("endpoint") == spec.endpoint
                and (record.get("approved_via") or {"mode": "manual"}).get("mode") == "manual"
                and record.get("surfaced_by_journal") in batch_journals):
            try:
                if datetime.fromisoformat(record["expires_at"]) > _now():
                    hosts.add(job_store.validate_result_hostname(record.get("host", "")))
            except (KeyError, ValueError, job_store.JobJournalError):
                continue
    return hosts


def check_job(batch_dir: Path, spec: ProviderSpec, journal: job_store.JobJournal, *,
              batch_journals: set[str], root: Path | None = None) -> dict[str, Any]:
    """Whether this paused job may be approved automatically, with reasons if not."""
    state = policy_state(batch_dir, spec)
    if not state["active"]:
        return {"eligible": False, "reasons": state["reasons"], "host": None}
    events = journal.events()
    if not events:
        return {"eligible": False, "reasons": ["empty job journal"], "host": None}
    reasons: list[str] = []
    first, latest = events[0], events[-1]
    record = state["record"]
    if journal.journal_id not in batch_journals:
        reasons.append("job does not belong to this batch")
    if latest.get("state") != job_store.AWAITING_RESULT_HOST_APPROVAL:
        reasons.append(f"job is {latest.get('state')}, not awaiting result-host approval")
    if first.get("provider_id") != spec.id or first.get("endpoint") != spec.endpoint:
        reasons.append("job was submitted to a different provider endpoint")
    if first.get("api_model_id") != spec.api_model_id:
        reasons.append("job was submitted for a different model")
    job_id = journal.job_id
    if not job_id or latest.get("job_id") not in (None, job_id):
        reasons.append("no matching Job ID recorded")
    if not record.get("include_pending"):
        try:
            if datetime.fromisoformat(latest["at"]) < datetime.fromisoformat(record["created_at"]):
                reasons.append("job was already pending when the policy was enabled (not included)")
        except (KeyError, ValueError):
            reasons.append("pause time unknown")
    url = latest.get("result_url") if isinstance(latest.get("result_url"), dict) else {}
    raw_host = str(url.get("host") or "")
    host = None
    try:
        host = job_store.validate_result_hostname(raw_host)
    except job_store.JobJournalError as exc:
        reasons.append(str(exc))
    if host is not None and host != raw_host:
        reasons.append("recorded host is not in canonical form")
    if url.get("scheme") != "https":
        reasons.append("result URL is not HTTPS")
    if url.get("port") not in (None, 443):
        reasons.append(f"result URL uses port {url.get('port')}")
    if not str(url.get("path") or "").startswith(RESULT_PATH_PREFIX):
        reasons.append(f"result URL path is not under {RESULT_PATH_PREFIX}")
    basis = None
    if host is not None:
        if host in set(record.get("listed_hosts") or []) & {h.lower() for h in spec.auto_result_hosts}:
            basis = "listed Baidu result host"
        elif host in _manually_approved_hosts(spec, batch_journals - {journal.journal_id}, root):
            basis = "host approved manually for another job of this batch"
        elif re.fullmatch(BAIDU_RESULT_HOST_PATTERN, host):
            basis = "matches paddleocr-store-<n>.bj.bcebos.com"
        else:
            reasons.append(f"host {host!r} is not a trusted Baidu result host; approve it manually")
    return {"eligible": not reasons, "reasons": reasons, "host": host, "job_id": job_id, "basis": basis,
            "policy_path": state["path"]}


def auto_approve(batch_dir: Path, spec: ProviderSpec, journal: job_store.JobJournal, *,
                 batch_journals: set[str], root: Path | None = None) -> Path | None:
    """Create the job-scoped approval for an eligible paused job; None leaves it manual."""
    checked = check_job(batch_dir, spec, journal, batch_journals=batch_journals, root=root)
    if not checked["eligible"]:
        return None
    if checked["host"] in job_store.approved_result_hosts(spec, root, journal_id=journal.journal_id):
        return None
    record = policy_state(batch_dir, spec)["record"]
    via = {"mode": "automatic_policy", "batch": Path(batch_dir).name, "policy_record": checked["policy_path"],
           "basis": checked["basis"], "policy_enabled_by": record.get("enabled_by"),
           "policy_expires_at": record.get("expires_at")}
    try:
        path = job_store.approve_result_host(
            spec, host=checked["host"], approved_by=f"automatic policy (enabled by {record.get('enabled_by')})",
            statement=job_store.HOST_APPROVAL_STATEMENT, journal=journal, root=root, via=via)
    except job_store.JobJournalError:
        return None
    write_sequenced(_directory(batch_dir), "auto_approval", {
        "schema_version": 1, "provider_id": spec.id, "endpoint": spec.endpoint, "journal_id": journal.journal_id,
        "job_id": checked["job_id"], "host": checked["host"], "basis": checked["basis"],
        "approval_record": str(path), "policy_record": checked["policy_path"], "created_at": _now().isoformat(),
    })
    return path


def auto_approvals(batch_dir: Path) -> list[dict]:
    return [payload for _path, payload in ordered(_directory(batch_dir), "auto_approval") if payload]


__all__ = [
    "BAIDU_RESULT_HOST_PATTERN", "HostPolicyError", "MAX_POLICY_HOURS", "POLICY_STATEMENT", "auto_approvals",
    "auto_approve", "check_job", "enable_policy", "policy_state", "revoke_policy", "supported",
]
