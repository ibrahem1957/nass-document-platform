"""Per-provider enablement records and an append-only usage ledger.

An enablement is a write-once JSON record created by the user. It is bound to the
provider's exact endpoint and API model id (changing either invalidates it), it
expires, and it caps pages and cost. Revocation is a newer record with
`enabled = false`; nothing is ever overwritten or deleted.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from lab.core.e2.api.providers import ProviderSpec
from lab.core.e2.api.records import ordered, write_sequenced
from lab.core.e2.store import results_root

MAX_ENABLEMENT_DAYS = 30
# Used only for a provider whose registry entry sets no internal ceiling (keeps the earlier
# behaviour for such providers). Configure `[providers.limits]` in providers_e2.toml instead.
UNCONFIGURED_PAGE_CEILING = 100
DEFAULT_ENABLEMENT_DAYS = 7


def usage_policy(spec: ProviderSpec) -> dict[str, Any]:
    """The internal usage limits for a provider and where each value comes from.

    These are this project's internal safety limits, not provider quotas. A provider's documented
    daily quota (when configured) lowers the daily ceiling; it is never raised above it."""
    limits = spec.limits
    per_enablement = limits.max_pages_per_enablement or UNCONFIGURED_PAGE_CEILING
    per_day = limits.max_pages_per_day_ceiling or per_enablement
    if limits.provider_documented_daily_pages:
        per_day = min(per_day, limits.provider_documented_daily_pages)
    configured = limits.max_pages_per_enablement is not None
    return {
        "max_pages_per_enablement": per_enablement,
        "max_pages_per_day_ceiling": min(per_day, per_enablement),
        "max_pages_cumulative": limits.max_pages_cumulative,
        "default_enablement_days": limits.default_enablement_days or DEFAULT_ENABLEMENT_DAYS,
        "provider_documented_daily_pages": limits.provider_documented_daily_pages,
        "source": ("providers_e2.toml [providers.limits] (internal limits, not provider quotas)" if configured
                   else f"not configured in providers_e2.toml: internal default ceiling {UNCONFIGURED_PAGE_CEILING}"),
        "provider_quota": (f"provider documents {limits.provider_documented_daily_pages} pages/day"
                           if limits.provider_documented_daily_pages else
                           "provider quota not documented here and never queried; the provider enforces it itself"),
        "counting": ("each submitted request counts 1 page, also when it failed or is unconfirmed; the enablement "
                     "budget counts submissions since the current enablement started; the daily limit counts the "
                     "UTC day across all enablements; the cumulative cap counts the whole history"),
    }
REQUIRED_STATEMENT = (
    "I confirm that only pages of documents I have explicitly authorized by exact SHA-256 "
    "(under the declaration recorded for each document) will be sent, that this is an "
    "experimental evaluation outside the air-gapped system, and that the provider "
    "receives the page images shown in the preview or the approved plan."
)


class AuthorizationError(PermissionError):
    pass


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


@contextmanager
def provider_usage_lock(provider_id: str, root: Path | None = None):
    """Serialize quota checks and usage writes across threads and processes.

    The file itself contains no credential or usage data.  Keeping the check and
    reservation under the same lock prevents two different page submissions from
    both observing the last remaining quota slot.
    """
    lock_path = _root(root) / "api_usage" / f".{provider_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    key = str(lock_path.resolve()).lower()
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.RLock())
    with thread_lock, lock_path.open("a+b") as handle:
        if handle.tell() == 0 and lock_path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt  # noqa: PLC0415

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # noqa: PLC0415

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _root(root: Path | None) -> Path:
    return Path(root or results_root())


def _enable_dir(root: Path | None, provider_id: str) -> Path:
    return _root(root) / "api_authorizations" / provider_id


def _ledger(root: Path | None, provider_id: str) -> Path:
    return _root(root) / "api_usage" / f"{provider_id}.jsonl"


def _write_once(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def record_enablement(
    spec: ProviderSpec,
    *,
    enabled: bool,
    enabled_by: str,
    statement: str,
    max_pages_total: int,
    max_pages_per_day: int,
    days_valid: int,
    root: Path | None = None,
) -> Path:
    if enabled:
        if not spec.runnable:
            raise AuthorizationError(f"{spec.id} is {spec.status}; it cannot be enabled")
        if statement != REQUIRED_STATEMENT:
            raise AuthorizationError("The confirmation statement was not accepted")
        if not enabled_by.strip():
            raise AuthorizationError("Enter the name of the person enabling this provider")
        policy = usage_policy(spec)
        ceiling, day_ceiling = policy["max_pages_per_enablement"], policy["max_pages_per_day_ceiling"]
        if not 1 <= max_pages_total <= ceiling or not 1 <= max_pages_per_day <= min(max_pages_total, day_ceiling):
            raise AuthorizationError(
                f"Page limits must satisfy 1 <= per-day <= total, total <= {ceiling} per enablement and per-day <= "
                f"{day_ceiling} ({policy['source']})")
        if not 1 <= days_valid <= MAX_ENABLEMENT_DAYS:
            raise AuthorizationError(f"Validity must be 1..{MAX_ENABLEMENT_DAYS} days")
    now = datetime.now(timezone.utc)
    record = {
        "schema_version": 1,
        "provider_id": spec.id,
        "api_model_id": spec.api_model_id,
        "endpoint": spec.endpoint,
        "enabled": enabled,
        "enabled_by": enabled_by.strip(),
        "statement": statement if enabled else "revoked",
        "max_pages_total": max_pages_total if enabled else 0,
        "max_pages_per_day": max_pages_per_day if enabled else 0,
        # Recorded for audit: the policy this enablement was created under.
        "usage_policy": usage_policy(spec) if enabled else None,
        "budget_scope": "submissions since created_at (this enablement); daily limit per UTC day",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(days=days_valid)).isoformat() if enabled else now.isoformat(),
    }
    # Strictly sequenced, so an enable and a disable in the same clock tick keep their order.
    return write_sequenced(_enable_dir(root, spec.id), "enablement", record)


def latest_enablement(spec: ProviderSpec, root: Path | None = None) -> dict | None:
    records = ordered(_enable_dir(root, spec.id), "enablement")
    if not records:
        return None
    path, payload = records[-1]
    if payload is None:
        # An unreadable latest record fails closed: the provider is treated as disabled.
        return {"enabled": False, "corrupt": True, "path": str(path), "max_pages_total": 0,
                "max_pages_per_day": 0, "expires_at": ""}
    return payload | {"path": str(path)}


def _usage_records_unlocked(spec: ProviderSpec, root: Path | None = None) -> list[dict]:
    path = _ledger(root, spec.id)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise AuthorizationError(f"Usage ledger is malformed; refusing requests: {path}") from exc
    return records


def usage_records(spec: ProviderSpec, root: Path | None = None) -> list[dict]:
    with provider_usage_lock(spec.id, root):
        return _usage_records_unlocked(spec, root)


def _append_usage_unlocked(spec: ProviderSpec, entry: dict, root: Path | None = None) -> None:
    path = _ledger(root, spec.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_usage(spec: ProviderSpec, entry: dict, root: Path | None = None) -> None:
    with provider_usage_lock(spec.id, root):
        _append_usage_unlocked(spec, entry, root)


@dataclass
class AuthorizationState:
    allowed: bool
    reasons: list[str]
    enablement: dict | None
    pages_used_total: int          # whole history (all enablements), shown separately
    pages_used_today: int          # UTC day, all enablements
    pages_used_enablement: int = 0  # since the current enablement started (its budget)
    enablement_started_at: str | None = None
    limit_enablement: int = 0
    limit_day: int = 0
    limit_cumulative: int | None = None
    policy: dict | None = None

    @property
    def remaining_pages(self) -> int:
        """Pages that may still be submitted now under every limit (0 when not enabled)."""
        if not (self.enablement or {}).get("enabled"):
            return 0
        values = [self.limit_enablement - self.pages_used_enablement, self.limit_day - self.pages_used_today]
        if self.limit_cumulative is not None:
            values.append(self.limit_cumulative - self.pages_used_total)
        return max(0, min(values))

    def describe(self) -> str:
        cumulative = (f"; cumulative {self.pages_used_total}/{self.limit_cumulative}" if self.limit_cumulative
                      else f"; history {self.pages_used_total} (no cumulative cap)")
        return (f"this enablement {self.pages_used_enablement}/{self.limit_enablement} since "
                f"{self.enablement_started_at or '-'}; today {self.pages_used_today}/{self.limit_day}{cumulative}; "
                f"{self.remaining_pages} left")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reasons": list(self.reasons),
            "enablement_path": (self.enablement or {}).get("path"),
            "pages_used_total": self.pages_used_total,
            "pages_used_today": self.pages_used_today,
            "pages_used_enablement": self.pages_used_enablement,
            "enablement_started_at": self.enablement_started_at,
            "limit_enablement": self.limit_enablement,
            "limit_day": self.limit_day,
            "limit_cumulative": self.limit_cumulative,
            "remaining_pages": self.remaining_pages,
            "policy_source": (self.policy or {}).get("source"),
        }


def _at(value: Any) -> datetime | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _check_authorization_unlocked(
    spec: ProviderSpec, *, pages: int, root: Path | None = None
) -> AuthorizationState:
    reasons: list[str] = []
    record = latest_enablement(spec, root)
    usage = _usage_records_unlocked(spec, root)
    # Every submitted request counts, successful or not: the provider saw the page.
    sent = [item for item in usage if item.get("submitted")]
    today = datetime.now(timezone.utc).date().isoformat()
    used_total = sum(int(item.get("pages", 0)) for item in sent)
    used_today = sum(int(item.get("pages", 0)) for item in sent if str(item.get("at", "")).startswith(today))
    policy = usage_policy(spec)
    started = _at((record or {}).get("created_at")) if record and record.get("enabled") else None
    # An entry without a readable timestamp counts against the budget (fail closed).
    used_window = sum(int(item.get("pages", 0)) for item in sent
                      if started is None or _at(item.get("at")) is None or _at(item.get("at")) >= started)
    # The record's limits, never above the current ceilings (a lowered ceiling applies at once).
    limit_window = min(int((record or {}).get("max_pages_total", 0)), policy["max_pages_per_enablement"])
    limit_day = min(int((record or {}).get("max_pages_per_day", 0)), policy["max_pages_per_day_ceiling"])
    cumulative = policy["max_pages_cumulative"]
    if not spec.runnable:
        reasons.append(f"Provider {spec.id} is {spec.status}; no request can be sent")
    if record is None:
        reasons.append("Provider is disabled: no enablement record exists (disabled by default)")
    else:
        if not record.get("enabled"):
            reasons.append("Provider was disabled by the latest enablement record")
        if record.get("provider_id") != spec.id or record.get("endpoint") != spec.endpoint or record.get("api_model_id") != spec.api_model_id:
            reasons.append("Enablement does not match the provider's current endpoint/model; enable again")
        try:
            expired = datetime.fromisoformat(record["expires_at"]) <= datetime.now(timezone.utc)
        except (KeyError, ValueError):
            expired = True
        if expired:
            reasons.append("Enablement has expired")
        if used_window + pages > limit_window:
            reasons.append(f"Enablement page limit reached ({used_window}/{limit_window} since "
                           f"{record.get('created_at')}; internal limit)")
        if used_today + pages > limit_day:
            reasons.append(f"Daily page limit reached ({used_today}/{limit_day} today, UTC; internal limit)")
        if cumulative is not None and used_total + pages > cumulative:
            reasons.append(f"Cumulative page limit reached ({used_total}/{cumulative} over the whole history; "
                           "internal cap in providers_e2.toml)")
    if pages > spec.limits.max_pages_per_request:
        reasons.append(f"At most {spec.limits.max_pages_per_request} page per request")
    if spec.pricing.estimated_usd_per_page < 0:
        reasons.append("Price per page is unknown; the cost cap cannot be enforced")
    elif spec.pricing.basis == "per_token" and not spec.pricing.price_verified:
        reasons.append("Per-token price units are unverified; the cost cap cannot be enforced")
    elif spec.pricing.estimated_usd_per_page * pages > spec.limits.max_cost_usd_per_run:
        reasons.append(
            f"Estimated cost {spec.pricing.estimated_usd_per_page * pages:.4f} USD exceeds the cap "
            f"{spec.limits.max_cost_usd_per_run:.4f} USD"
        )
    return AuthorizationState(
        not reasons, reasons, record, used_total, used_today, pages_used_enablement=used_window,
        enablement_started_at=(record or {}).get("created_at") if started else None,
        limit_enablement=limit_window if record else 0, limit_day=limit_day if record else 0,
        limit_cumulative=cumulative, policy=policy)


def check_authorization(
    spec: ProviderSpec, *, pages: int, root: Path | None = None
) -> AuthorizationState:
    with provider_usage_lock(spec.id, root):
        return _check_authorization_unlocked(spec, pages=pages, root=root)
