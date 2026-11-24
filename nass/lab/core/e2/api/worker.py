"""Isolated child process for one hosted API job step (submit-and-wait, or resume).

The credential arrives over the pipe after start (never on the command line or in
the environment), the outbound allowlist guard is installed before any request, and
only redacted records are sent back. `submitted` is sent the moment a Job ID exists,
so the parent persists it before anything else can fail. The parent enforces the
overall deadline.
"""

from __future__ import annotations

import importlib
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from lab.core.e2.api.client import (
    APIRequestError,
    ResultHostNotApproved,
    UrllibTransport,
    allowlist_network_guard,
    make_client,
    redact,
)
from lab.core.e2.api.providers import ProviderSpec

# Errors after a Job ID exists that do not invalidate the job: it can be resumed with
# the same Job ID (no resubmission). Anything else is terminal.
RESUMABLE_CODES = frozenset(
    {
        "poll_timeout",
        "request_timeout",
        "network_error",
        "network_blocked",
        "service_unavailable",
        "rate_limited",
        "auth_failed",
        "result_url_expired",
        "provider_error",
        "unsafe_redirect",
        "deadline_exceeded",
    }
)


def execute_job(
    client,
    *,
    image_path: Path | None,
    filename: str | None,
    job_id: str | None,
    emit: Callable[[dict], None],
) -> dict:
    """Run one step. Never submits when `job_id` is given (resume)."""
    submitted: list[str] = []

    def on_submitted(new_job_id: str) -> None:
        submitted.append(new_job_id)
        emit({"phase": "submitted", "job_id": new_job_id})

    try:
        result = client.run(image_path, filename, job_id=job_id, on_submitted=on_submitted)
        return {"phase": "done", "ok": True, "job_id": result.job_id, "result": asdict(result)}
    except ResultHostNotApproved as exc:
        return {
            "phase": "awaiting_result_host", "ok": False,
            "job_id": client.job_id or job_id or (submitted[0] if submitted else None),
            "result_url": exc.info, "exchanges": client.exchanges,
        }
    except APIRequestError as exc:
        known = client.job_id or job_id or (submitted[0] if submitted else None)
        return {
            "phase": "error", "ok": False, "job_id": known,
            "resumable": bool(known) and exc.code in RESUMABLE_CODES,
            "error": {
                "code": exc.code, "message": str(exc), "http_status": exc.http_status,
                "diagnostics": exc.diagnostics,
            },
            "exchanges": client.exchanges,
        }
    except Exception as exc:  # unexpected bug/parse error: never lose a known Job ID
        known = client.job_id or job_id or (submitted[0] if submitted else None)
        return {
            "phase": "error", "ok": False, "job_id": known, "resumable": bool(known),
            "error": {"code": "client_exception", "message": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-6000:]},
            "exchanges": client.exchanges,
        }


def _transport(path: str | None):
    if not path:
        return UrllibTransport()
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)()


def run_api_worker(conn, spec_data: dict, image_path: str | None, filename: str | None,
                   transport_path: str | None, job_id: str | None = None,
                   result_hosts: list[str] | None = None) -> None:
    token = ""
    try:
        token = str(conn.recv())
        spec = ProviderSpec.model_validate(spec_data)
        hosts = frozenset(result_hosts or spec.result_hosts)
        with allowlist_network_guard(spec.allowed_hosts | hosts):
            client = make_client(spec, token, _transport(transport_path), result_hosts=hosts)
            outcome = execute_job(
                client,
                image_path=Path(image_path) if image_path else None,
                filename=filename,
                job_id=job_id,
                emit=lambda message: conn.send(redact(message, [token])),
            )
        conn.send(redact(outcome, [token]))
    except Exception as exc:  # report, never leak the token
        conn.send(
            {
                "phase": "error", "ok": False, "job_id": job_id, "resumable": bool(job_id),
                "error": {
                    "code": "worker_error",
                    "message": redact(f"{type(exc).__name__}: {exc}", [token]),
                    "traceback": redact(traceback.format_exc()[-6000:], [token]),
                },
                "exchanges": [],
            }
        )
    finally:
        token = ""
        conn.close()
