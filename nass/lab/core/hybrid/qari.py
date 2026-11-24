"""Qari-OCR 0.4.0 through a user-run OpenAI-compatible server (local GPU or RunPod).

Nothing assumes Qari is installed. `check_ready` only lists the server's models
(GET /v1/models); it never starts inference. A page is sent only after the hybrid
run's explicit approval, one request per page, never retried automatically:

- loopback servers (localhost / 127.0.0.1 / ::1) may use http; any other host must use
  https, and its page images are sent only for a document authorized by exact SHA-256;
- no redirects are followed; the request goes only to the host shown in the plan;
- a request whose body was sent but got no response is SUBMISSION_UNCONFIRMED and is
  never resent automatically (an explicit retry is required).

Settings: DPI 250, max_new_tokens 4096 (max_tokens), do_sample=False (temperature 0),
prompt "Free OCR." (prompt id `qari_free_ocr_v1`).
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from lab.core.e2.prompts import get_prompt

JPEG_MAGIC = b"\xff\xd8\xff"  # JPEG start-of-image marker
QARI_MODEL = "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"
QARI_PROMPT_ID = "qari_free_ocr_v1"
QARI_DPI = 250
QARI_MAX_NEW_TOKENS = 4096
QARI_TIMEOUT_S = 300
MAX_RESPONSE_BYTES = 5_000_000
LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})

SENT = "COMPLETED"
FAILED = "FAILED"
UNCONFIRMED = "SUBMISSION_UNCONFIRMED"
NOT_TRANSMITTED = "NOT_TRANSMITTED"


class QariError(ValueError):
    pass


def settings() -> dict:
    prompt = get_prompt(QARI_PROMPT_ID)
    return {"model": QARI_MODEL, "dpi": QARI_DPI, "max_new_tokens": QARI_MAX_NEW_TOKENS, "do_sample": False,
            "request": {"max_tokens": QARI_MAX_NEW_TOKENS, "temperature": 0.0},
            "prompt": {"id": prompt.prompt_id, "sha256": prompt.sha256, "text": prompt.text},
            "timeout_s": QARI_TIMEOUT_S, "protocol": "OpenAI-compatible /v1/chat/completions"}


def parse_endpoint(url: str) -> dict:
    parts = urlsplit((url or "").strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise QariError("Enter the server URL, e.g. http://localhost:8000 or https://<pod>-8000.proxy.runpod.net")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise QariError("The server URL must not contain credentials, a query or a fragment")
    host = parts.hostname.lower()
    loopback = host in LOOPBACK
    if not loopback and parts.scheme != "https":
        raise QariError("A remote server (anything but localhost) must use https")
    base_path = parts.path.rstrip("/")
    if base_path.endswith("/v1"):
        base_path = base_path[:-3]
    return {"scheme": parts.scheme, "host": host, "port": parts.port, "loopback": loopback,
            "base_path": base_path, "display": f"{parts.scheme}://{parts.netloc}{base_path}",
            "chat_url": f"{parts.scheme}://{parts.netloc}{base_path}/v1/chat/completions"}


def _connection(endpoint: dict, timeout: float):
    if endpoint["scheme"] == "https":
        return http.client.HTTPSConnection(endpoint["host"], endpoint["port"] or 443, timeout=timeout,
                                           context=ssl.create_default_context())
    return http.client.HTTPConnection(endpoint["host"], endpoint["port"] or 80, timeout=timeout)


def _headers(token: str | None) -> dict:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def check_ready(url: str, token: str | None = None, *, model: str = QARI_MODEL, timeout: float = 10.0) -> dict:
    """List the server's models (no inference). Ready only if the Qari model is served."""
    try:
        endpoint = parse_endpoint(url)
    except QariError as exc:
        return {"ready": False, "status": "Unavailable", "error": str(exc), "models": []}
    connection = _connection(endpoint, timeout)
    try:
        connection.request("GET", f"{endpoint['base_path']}/v1/models", headers=_headers(token))
        response = connection.getresponse()
        body = response.read(1_000_000)
        if response.status != 200:
            return {"ready": False, "status": "Unavailable", "error": f"HTTP {response.status} from /v1/models",
                    "models": [], "endpoint": endpoint["display"]}
        models = [str(item.get("id")) for item in (json.loads(body.decode("utf-8")).get("data") or [])
                  if isinstance(item, dict)]
    except (OSError, http.client.HTTPException, ValueError) as exc:
        return {"ready": False, "status": "Unavailable", "error": f"{type(exc).__name__}: {exc}", "models": [],
                "endpoint": endpoint["display"]}
    finally:
        connection.close()
    listed = model in models
    return {"ready": listed, "status": "Ready" if listed else "Unavailable", "models": models,
            "endpoint": endpoint["display"], "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": None if listed else f"the server does not list the model {model!r}"}


def transcribe(endpoint: dict, token: str | None, png: bytes, *, model: str = QARI_MODEL,
               timeout: float = QARI_TIMEOUT_S) -> dict:
    """One request. `state` tells whether anything may have been received by the server."""
    prompt = get_prompt(QARI_PROMPT_ID)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": ("data:image/jpeg;base64," if png[:3] == JPEG_MAGIC
                                                        else "data:image/png;base64,") + base64.b64encode(png).decode()}},
            {"type": "text", "text": prompt.text},
        ]}],
        "max_tokens": QARI_MAX_NEW_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }).encode("utf-8")
    clock = time.perf_counter()
    send_started_at = datetime.now(timezone.utc).isoformat()
    timing = {"send_started_at": send_started_at, "timeout_s": timeout}
    connection = _connection(endpoint, timeout)
    try:
        try:
            connection.connect()
        except (OSError, http.client.HTTPException) as exc:
            return {"state": NOT_TRANSMITTED, "ok": False, "elapsed_ms": (time.perf_counter() - clock) * 1000,
                    "error": {"code": "connect_failed", "message": f"{type(exc).__name__}: {exc}"}} | timing
        try:
            connection.request("POST", f"{endpoint['base_path']}/v1/chat/completions", body=body,
                               headers=_headers(token))
            timing["send_finished_at"] = datetime.now(timezone.utc).isoformat()
            response = connection.getresponse()
            timing["response_received_at"] = datetime.now(timezone.utc).isoformat()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            return {"state": UNCONFIRMED, "ok": False, "elapsed_ms": (time.perf_counter() - clock) * 1000,
                    "error": {"code": "no_response", "message": f"{type(exc).__name__}: {exc}"}} | timing
    finally:
        connection.close()
    elapsed = (time.perf_counter() - clock) * 1000
    timing["elapsed_ms"] = elapsed
    if 300 <= response.status < 400:
        return {"state": FAILED, "ok": False, "elapsed_ms": elapsed,
                "error": {"code": f"http_{response.status}", "message": "redirects are not followed"}}
    if len(raw) > MAX_RESPONSE_BYTES:
        return {"state": FAILED, "ok": False, "elapsed_ms": elapsed,
                "error": {"code": "response_too_large", "message": f"more than {MAX_RESPONSE_BYTES} bytes"}}
    if response.status != 200:
        return {"state": FAILED, "ok": False, "elapsed_ms": elapsed,
                "error": {"code": f"http_{response.status}", "message": raw[:500].decode("utf-8", "replace")}}
    try:
        payload = json.loads(raw.decode("utf-8"))
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str):
            raise ValueError("no message content")
    except (ValueError, AttributeError) as exc:
        return {"state": FAILED, "ok": False, "elapsed_ms": elapsed,
                "error": {"code": "response_format", "message": str(exc)}}
    finish = choice.get("finish_reason")
    usage = payload.get("usage") or {}
    return {"state": SENT, "ok": True, "elapsed_ms": elapsed, "text": text, "finish_reason": finish,
            "truncated": finish == "length" if finish else None, "reported_model": payload.get("model"),
            "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")},
            "response_id": payload.get("id")}


# ----------------------------------------------------------- run records


def _write_once(path: Path, payload: dict | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def attempts(vlm_dir: Path, page: int) -> list[dict]:
    found = []
    for path in sorted((Path(vlm_dir) / "attempts").glob(f"page_{page:03d}_attempt_[0-9][0-9].json")):
        intent = _read(path) or {}
        intent["outcome"] = _read(path.with_name(f"{path.stem}_outcome.json"))
        found.append(intent)
    return found


def page_record(vlm_dir: Path, page: int) -> dict:
    """Normalized record (same keys as the hosted-API records used by the combiner)."""
    tried = attempts(vlm_dir, page)
    base = {"page": page, "attempts": len(tried), "text": "", "job_state": None, "status": "Not sent",
            "status_reason": "not sent yet", "tokens": {"prompt": None, "completion": None, "reported": False},
            "finish_reason": None, "truncated_at_token_limit": None, "processing_seconds": None, "error": None}
    if not tried:
        return base
    last = tried[-1]
    outcome = last.get("outcome")
    if outcome is None:  # intent without outcome: the process stopped mid-request
        return base | {"job_state": UNCONFIRMED, "status": "Uncertain",
                       "status_reason": "request outcome unknown (interrupted); never resent automatically"}
    state = outcome.get("state")
    record = base | {
        "job_state": state, "error": outcome.get("error"),
        "processing_seconds": round(float(outcome.get("elapsed_ms") or 0) / 1000, 3),
    }
    if state == SENT:
        text = outcome.get("text") or ""
        usage = outcome.get("usage") or {}
        record.update(text=text, finish_reason=outcome.get("finish_reason"),
                      truncated_at_token_limit=outcome.get("truncated"),
                      tokens={"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
                              "reported": usage.get("prompt_tokens") is not None},
                      status="Success", status_reason="completed")
    elif state == NOT_TRANSMITTED:
        record.update(status="Not sent", status_reason=f"not transmitted: {(outcome.get('error') or {}).get('message')}")
    elif state == UNCONFIRMED:
        record.update(status="Uncertain", status_reason="no response (the server may have received the page); "
                                                        "never resent automatically")
    else:
        error = outcome.get("error") or {}
        record.update(status="Failed", status_reason=f"{error.get('code')}: {error.get('message')}")
    return record


def create_plan(vlm_dir: Path, *, endpoint: dict, model: str, pages: list[dict], document_sha256: str,
                approved_by: str) -> None:
    """`pages`: [{page, png (bytes), width, height}] rendered at QARI_DPI. Images are stored write-once."""
    vlm_dir = Path(vlm_dir)
    vlm_dir.mkdir(parents=True, exist_ok=False)
    listed = []
    for item in pages:
        # The bytes are stored unchanged; JPEG crops (Phase 2 native/JPEG options) keep a .jpg name.
        name = f"images/page_{item['page']:03d}.{'jpg' if item['png'][:3] == JPEG_MAGIC else 'png'}"
        _write_once(vlm_dir / name, item["png"])
        listed.append({"page": item["page"], "file": name, "bytes": len(item["png"]),
                       "image_sha256": hashlib.sha256(item["png"]).hexdigest(),
                       "width": item.get("width"), "height": item.get("height"),
                       "dpi": item.get("dpi", QARI_DPI)})
    _write_once(vlm_dir / "plan.json", {
        "schema_version": 1, "kind": "qari_plan", "endpoint": {k: endpoint[k] for k in ("display", "chat_url",
                                                                                           "host", "loopback")},
        "model": model, "settings": settings(), "document_sha256": document_sha256, "pages": listed,
        "approved_by": approved_by, "created_at": datetime.now(timezone.utc).isoformat(),
        "not_sent": ["Ground Truth", "PDF text layer / library text", "other pages", "the PDF file"],
    })


def load_plan(vlm_dir: Path) -> dict:
    plan = _read(Path(vlm_dir) / "plan.json")
    if not plan:
        raise QariError(f"No Qari plan in {vlm_dir}")
    return plan


def send_page(vlm_dir: Path, page: int, *, token: str | None, document_authorized: bool, kind: str = "run",
              by: str | None = None, timeout: float = QARI_TIMEOUT_S) -> dict:
    vlm_dir = Path(vlm_dir)
    plan = load_plan(vlm_dir)
    item = next((entry for entry in plan["pages"] if entry["page"] == page), None)
    if item is None:
        raise QariError(f"Page {page} is not in the approved plan")
    endpoint = parse_endpoint(plan["endpoint"]["display"])
    if not endpoint["loopback"] and not document_authorized:
        raise QariError("The document is not authorized (exact SHA-256) for sending to a remote server")
    png = (vlm_dir / item["file"]).read_bytes()
    if hashlib.sha256(png).hexdigest() != item["image_sha256"]:
        raise QariError("The stored page image does not match the approved plan; nothing was sent")
    number = len(attempts(vlm_dir, page)) + 1
    intent = vlm_dir / "attempts" / f"page_{page:03d}_attempt_{number:02d}.json"
    _write_once(intent, {"page": page, "attempt": number, "kind": kind, "by": by,
                         "image_sha256": item["image_sha256"], "endpoint": plan["endpoint"]["display"],
                         "created_at": datetime.now(timezone.utc).isoformat()})
    called_at = datetime.now(timezone.utc).isoformat()
    outcome = transcribe(endpoint, token, png, model=plan["model"], timeout=timeout)
    # Timing is recorded for every outcome (send start/end, response); never used to resend.
    outcome.setdefault("send_started_at", called_at)
    outcome.setdefault("response_received_at", datetime.now(timezone.utc).isoformat()
                       if outcome.get("state") in {SENT, FAILED} else None)
    outcome["timeout_s"] = timeout
    _write_once(intent.with_name(f"{intent.stem}_outcome.json"), outcome)
    return page_record(vlm_dir, page)
