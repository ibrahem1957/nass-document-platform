"""Write-once job journal and result-host approvals for hosted API jobs.

One journal per run: `results/api_jobs/<provider_id>/<run_id>/eNNNN_<STATE>.json`.
Events are created with exclusive creation, so concurrent writers cannot overwrite
each other and the full audit trail is kept. The current state is the last event.

A page is submitted at most once per journal. While any journal for the same
provider and page image is unfinished, a new submission is refused: the user must
resume it, or explicitly abandon it.
"""

from __future__ import annotations

import ipaddress
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.store import results_root

SUBMITTING = "SUBMITTING"
SUBMITTED = "SUBMITTED"
AWAITING_RESULT_HOST_APPROVAL = "AWAITING_RESULT_HOST_APPROVAL"
RESUMABLE_ERROR = "RESUMABLE_ERROR"
RESUMING = "RESUMING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
ABANDONED = "ABANDONED"
# Derived, never written: SUBMITTING without a later SUBMITTED (crash mid-submit).
SUBMISSION_UNCONFIRMED = "SUBMISSION_UNCONFIRMED"

TERMINAL = frozenset({COMPLETED, FAILED, ABANDONED})
CLAIM_STALE_S = 120.0
HOST_APPROVAL_HOURS = 24
# SUBMISSION_UNCONFIRMED is also written explicitly when a submit timed out or the
# worker died before a Job ID arrived: the page may have been accepted, so the only
# way forward is an explicit abandon (never an automatic resubmission).
RESUMABLE = frozenset({SUBMITTED, AWAITING_RESULT_HOST_APPROVAL, RESUMABLE_ERROR, RESUMING})

HOST_APPROVAL_STATEMENT = (
    "I reviewed this exact result-storage hostname and approve downloading this "
    "provider's results from it over HTTPS."
)


class JobJournalError(RuntimeError):
    pass


def _root(root: Path | None) -> Path:
    return Path(root or results_root())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobJournal:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @property
    def journal_id(self) -> str:
        return self.directory.name

    def events(self) -> list[dict]:
        found = []
        for path in sorted(self.directory.glob("e*.json")):
            try:
                found.append(json.loads(path.read_text(encoding="utf-8")) | {"_file": path.name})
            except (OSError, ValueError):
                continue
        return found

    def append(self, state: str, **fields: Any) -> dict:
        self.directory.mkdir(parents=True, exist_ok=True)
        sequence = len(list(self.directory.glob("e*.json"))) + 1
        while True:
            path = self.directory / f"e{sequence:04d}_{state}.json"
            record = {"sequence": sequence, "state": state, "at": _now(), **fields}
            try:
                with path.open("x", encoding="utf-8", newline="\n") as handle:
                    json.dump(record, handle, ensure_ascii=False, indent=2)
                return record
            except FileExistsError:
                sequence += 1

    @property
    def first(self) -> dict:
        events = self.events()
        if not events:
            raise JobJournalError(f"Empty job journal {self.directory}")
        return events[0]

    def latest(self) -> dict:
        events = self.events()
        if not events:
            raise JobJournalError(f"Empty job journal {self.directory}")
        return events[-1]

    @property
    def job_id(self) -> str | None:
        return next((item["job_id"] for item in reversed(self.events()) if item.get("job_id")), None)

    @property
    def state(self) -> str:
        latest = self.latest()
        if latest["state"] == SUBMITTING and not self.job_id:
            return SUBMISSION_UNCONFIRMED
        return latest["state"]

    @property
    def open(self) -> bool:
        return self.state not in TERMINAL

    def summary(self) -> dict:
        first = self.first
        latest = self.latest()
        return {
            "journal_id": self.journal_id,
            "state": self.state,
            "job_id": self.job_id,
            "provider_id": first.get("provider_id"),
            "api_model_id": first.get("api_model_id"),
            "page": first.get("page"),
            "image_sha256": first.get("image_sha256"),
            "document_sha256": first.get("document_sha256"),
            "run_directory": first.get("run_directory"),
            "awaiting_host": latest.get("result_url", {}).get("host")
            if self.state == AWAITING_RESULT_HOST_APPROVAL
            else None,
            "latest": {key: value for key, value in latest.items() if key != "_file"},
            "events": len(self.events()),
        }


def journal_dir(provider_id: str, run_id: str, root: Path | None = None) -> Path:
    return _root(root) / "api_jobs" / provider_id / run_id


def journals(provider_id: str, root: Path | None = None) -> list[JobJournal]:
    base = _root(root) / "api_jobs" / provider_id
    if not base.is_dir():
        return []
    return [
        JobJournal(path) for path in sorted(base.iterdir())
        if path.is_dir() and not path.name.startswith("_") and any(path.glob("e*.json"))
    ]


def open_journals_for_image(provider_id: str, image_sha256: str, root: Path | None = None) -> list[JobJournal]:
    return [
        item for item in journals(provider_id, root)
        if item.open and item.first.get("image_sha256") == image_sha256
    ]


def claim_submission(provider_id: str, image_sha256: str, journal_id: str, root: Path | None = None) -> Path:
    """Atomically claim the right to submit this page image once.

    Claims are numbered per image and created with exclusive creation, so two
    concurrent clicks/reruns cannot both win. A claim is granted only when every
    earlier journal for this image is terminal.
    """
    open_items = open_journals_for_image(provider_id, image_sha256, root)
    if open_items:
        item = open_items[0]
        raise JobJournalError(
            f"An unfinished job already exists for this page image (journal {item.journal_id}, "
            f"state {item.state}, job {item.job_id}). Resume or abandon it; it is never resubmitted."
        )
    directory = _root(root) / "api_jobs" / provider_id / "_claims"
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob(f"claim_{image_sha256}_*.json"))
    for claim in existing:
        # A claim whose journal does not exist yet is a submission in progress (its
        # SUBMITTING record is written before anything is sent). It blocks new claims
        # until it is older than CLAIM_STALE_S, which can only mean a crash before
        # anything was transmitted.
        try:
            other = json.loads(claim.read_text(encoding="utf-8")).get("journal_id")
        except (OSError, ValueError):
            other = None
        journal = JobJournal(journal_dir(provider_id, other, root)) if other else None
        if journal is not None and journal.events():
            if journal.open:
                raise JobJournalError(
                    f"An unfinished job already exists for this page image (journal {other}); refused"
                )
            continue
        age = datetime.now(timezone.utc).timestamp() - claim.stat().st_mtime
        if age < CLAIM_STALE_S:
            raise JobJournalError("Another submission for this page image is in progress; refused")
    sequence = len(existing) + 1
    path = directory / f"claim_{image_sha256}_{sequence:04d}.json"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump({"journal_id": journal_id, "image_sha256": image_sha256, "at": _now()}, handle)
    except FileExistsError as exc:
        raise JobJournalError("Another submission for this page image started at the same time; refused") from exc
    return path


# --------------------------------------------------------- host approvals


def validate_result_hostname(host: str) -> str:
    """A result host must be a public DNS name (never an IP literal or local name)."""
    value = (host or "").strip().lower().rstrip(".")
    if not value or any(char in value for char in "/:@?# \\"):
        raise JobJournalError(f"Not a bare hostname: {host!r}")
    try:
        ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        pass
    else:
        raise JobJournalError(f"IP-address result hosts are never approved: {host!r}")
    if value in {"localhost"} or value.endswith((".localhost", ".local", ".internal", ".lan", ".home", ".corp")):
        raise JobJournalError(f"Local/private hostnames are never approved: {host!r}")
    if "." not in value:
        raise JobJournalError(f"Hostname must be fully qualified: {host!r}")
    return value


def approve_result_host(
    spec: ProviderSpec, *, host: str, approved_by: str, statement: str,
    journal: JobJournal, root: Path | None = None, via: dict | None = None,
) -> Path:
    """Approve exactly the host a paused job reported.

    Manual by default. `via` is set only by the user-enabled automatic policy
    (host_policy), which records which consent authorized this job's approval.
    """
    if statement != HOST_APPROVAL_STATEMENT:
        raise JobJournalError("The host-approval statement was not accepted")
    if not approved_by.strip():
        raise JobJournalError("Enter the name of the person approving the host")
    if journal.state != AWAITING_RESULT_HOST_APPROVAL:
        raise JobJournalError("Only a job awaiting result-host approval can approve a host")
    reported = journal.latest().get("result_url", {}).get("host")
    value = validate_result_hostname(host)
    if value != reported:
        raise JobJournalError(f"Approval must name exactly the reported host {reported!r}")
    # Strictly sequenced records: same-tick approvals/revocations never collide or reorder.
    return write_sequenced(_root(root) / "api_result_host_approvals" / spec.id, "approval", {
        "provider_id": spec.id,
        "endpoint": spec.endpoint,
        "host": value,
        "approved": True,
        "approved_by": approved_by.strip(),
        "statement": statement,
        "surfaced_by_journal": journal.journal_id,
        "surfaced_by_job_id": journal.job_id,
        "approved_via": via or {"mode": "manual"},
        "created_at": _now(),
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=HOST_APPROVAL_HOURS)).isoformat(),
    })


def revoke_result_host(spec: ProviderSpec, *, host: str, revoked_by: str, root: Path | None = None) -> Path:
    """Withdraw an approval (a newer write-once record; nothing is deleted)."""
    if not revoked_by.strip():
        raise JobJournalError("Enter the name of the person revoking the host")
    value = validate_result_hostname(host)
    return write_sequenced(_root(root) / "api_result_host_approvals" / spec.id, "approval", {
        "provider_id": spec.id, "endpoint": spec.endpoint, "host": value, "approved": False,
        "revoked_by": revoked_by.strip(), "created_at": _now()})


def host_approval_records(spec: ProviderSpec, root: Path | None = None) -> list[dict]:
    # Oldest first by record sequence (never by timestamp). Unreadable records are
    # ignored, which can only remove an approval, never add one.
    return [payload for _path, payload in ordered(_root(root) / "api_result_host_approvals" / spec.id, "approval")
            if payload is not None]


def approved_result_hosts(
    spec: ProviderSpec, root: Path | None = None, *, journal_id: str | None = None
) -> frozenset[str]:
    """Registry hosts plus explicit user approvals bound to this provider endpoint.

    For each host the newest record wins, so a revocation takes effect immediately.
    """
    hosts = {host.lower() for host in spec.result_hosts}
    latest: dict[str, bool] = {}
    for record in host_approval_records(spec, root):
        if record.get("endpoint") != spec.endpoint:
            continue
        # Dynamic result-storage approval is scoped to the job that surfaced it.
        # A new job must pause and obtain its own approval even for the same host.
        if (journal_id is not None and record.get("approved")
                and record.get("surfaced_by_journal") != journal_id):
            continue
        if record.get("approved"):
            try:
                if datetime.fromisoformat(record["expires_at"]) <= datetime.now(timezone.utc):
                    continue
            except (KeyError, ValueError):
                continue
        try:
            latest[validate_result_hostname(record.get("host", ""))] = bool(record.get("approved"))
        except JobJournalError:
            continue
    for host, approved in latest.items():
        if approved:
            hosts.add(host)
        else:
            hosts.discard(host)
    return frozenset(hosts)
