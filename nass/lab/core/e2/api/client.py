"""Hosted API transport, outbound allowlist and the Baidu AI Studio PaddleOCR client.

The Baidu protocol mirrors the official client shipped in paddleocr 3.7.0
(`paddleocr/_api_client`), which was read statically. It is not imported: importing
`paddleocr` pulls in PaddleX, and this client must work in any lab environment.

  POST {base}/api/v2/ocr/jobs   multipart: model, optionalPayload (JSON), file
       Authorization: Bearer <token>             -> {"code":0,"data":{"jobId":...}}
  GET  {base}/api/v2/ocr/jobs/{jobId}           -> data.state pending|running|done|failed
  GET  data.resultUrl.jsonUrl (pre-signed object storage, NO token) -> JSONL
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import json
import math
import mimetypes
import os
import re
import socket
import ssl
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Protocol
from urllib.parse import parse_qs, urljoin, urlsplit

from lab.core.e2 import airgap
from lab.core.e2.airgap import OutboundNetworkBlocked
from lab.core.e2.api.providers import ProviderSpec


JPEG_MAGIC = b"\xff\xd8\xff"  # JPEG start-of-image marker

class APIRequestError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, http_status: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        self.diagnostics = diagnostics or {}


# ----------------------------------------------------------------- redaction

_URL_QUERY = re.compile(r"(https?://[^\s\"'?#]+)\?[^\s\"'#]*")
# JSON encoders outside Python often escape "/" as "\/": https:\/\/host\/path?sig=...
_ESCAPED_URL_QUERY = re.compile(r"(https?:\\/\\/[^\s\"'?#]+)\?[^\s\"'#]*")
# Some JSON/HTML-safe encoders escape the query delimiter itself. Without this,
# a signed URL embedded as ``https:\/\/host\/p\u003fsig\u003d...`` survived
# redaction even though the ordinary and slash-escaped forms were covered.
_UNICODE_ESCAPED_URL_QUERY = re.compile(
    r"(https?(?::|\\u003[aA])(?:\\/|\\u002[fF]){2}[^\s\"']*?)(?:\\u003[fF])[^\s\"']*"
)
# Belt and braces: credential-like parameters anywhere (also percent/escaped forms).
_SENSITIVE_PARAM = re.compile(
    r"(?i)\b(authorization|signature|x-amz-signature|x-amz-credential|x-amz-security-token|"
    r"access_token|token|sig|key|credential|policy)=([^&\s\"'\\]+)"
)


def redact(value: Any, secrets: list[str]) -> Any:
    """Remove secrets and pre-signed URL query strings from anything to be stored."""
    if isinstance(value, dict):
        return {key: redact(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        text = value
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[REDACTED-SECRET]")
                escaped = json.dumps(secret, ensure_ascii=True)[1:-1]
                if escaped != secret:
                    text = text.replace(escaped, "[REDACTED-SECRET]")
        text = _URL_QUERY.sub(r"\1?[query-redacted]", text)
        text = _ESCAPED_URL_QUERY.sub(r"\1?[query-redacted]", text)
        text = _UNICODE_ESCAPED_URL_QUERY.sub(r"\1?[query-redacted]", text)
        return _SENSITIVE_PARAM.sub(r"\1=[redacted]", text)
    return value


# ------------------------------------------------------------ network guard


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    return ip.is_global and not ip.is_multicast


@contextmanager
def allowlist_network_guard(hosts: frozenset[str] | set[str]) -> Iterator[set[str]]:
    """Allow name resolution and connections only to `hosts` (and loopback).

    Connections are accepted only to addresses that resolved from an allowed host
    inside this guard. Datagram sends to non-loopback addresses are refused.
    """
    allowed_hosts = {host.lower() for host in hosts}
    allowed_ips: set[str] = set()
    originals = {name: getattr(owner, name) for owner, name in airgap._GUARDED_ATTRIBUTES}

    def _check_host(host: Any, what: str) -> None:
        if host is None or _is_loopback(str(host)) or str(host).lower() in allowed_hosts:
            return
        raise OutboundNetworkBlocked(f"API allowlist blocked {what} for host {host!r}")

    def _check_address(address: Any) -> None:
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if _is_loopback(host) or host in allowed_ips or host.lower() in allowed_hosts:
                return
        raise OutboundNetworkBlocked(f"API allowlist blocked connection to {address!r}")

    def _accept_public(host: Any, addresses: list[str]) -> None:
        # An approved public hostname must not resolve to a private, loopback,
        # link-local or reserved address (DNS rebinding / internal targets).
        if host is None or _is_loopback(str(host)):
            return
        unsafe = [address for address in addresses if not _is_public_ip(address)]
        if unsafe:
            raise OutboundNetworkBlocked(
                f"API allowlist blocked {host!r}: resolves to non-public address(es) {unsafe}"
            )
        allowed_ips.update(addresses)

    def getaddrinfo(host, *args, **kwargs):
        _check_host(host, "DNS resolution")
        result = originals["getaddrinfo"](host, *args, **kwargs)
        if host is not None and str(host).lower() in allowed_hosts:
            _accept_public(host, [item[4][0] for item in result])
        return result

    def gethostbyname(host):
        _check_host(host, "DNS resolution")
        address = originals["gethostbyname"](host)
        if str(host).lower() in allowed_hosts:
            _accept_public(host, [address])
        return address

    def gethostbyname_ex(host):
        _check_host(host, "DNS resolution")
        result = originals["gethostbyname_ex"](host)
        if str(host).lower() in allowed_hosts:
            _accept_public(host, list(result[2]))
        return result

    def gethostbyaddr(host):
        raise OutboundNetworkBlocked(f"API allowlist blocked reverse DNS for {host!r}")

    def connect(sock, address):
        _check_address(address)
        return originals["connect"](sock, address)

    def connect_ex(sock, address):
        _check_address(address)
        return originals["connect_ex"](sock, address)

    def sendto(sock, data, *args):
        target = args[-1] if args else None
        if not (isinstance(target, tuple) and target and _is_loopback(str(target[0]))):
            raise OutboundNetworkBlocked(f"API allowlist blocked datagram to {target!r}")
        return originals["sendto"](sock, data, *args)

    def create_connection(address, *args, **kwargs):
        if isinstance(address, tuple) and address:
            _check_host(address[0], "connection")
        return originals["create_connection"](address, *args, **kwargs)

    replacements = {
        "connect": connect,
        "connect_ex": connect_ex,
        "sendto": sendto,
        "create_connection": create_connection,
        "getaddrinfo": getaddrinfo,
        "gethostbyname": gethostbyname,
        "gethostbyname_ex": gethostbyname_ex,
        "gethostbyaddr": gethostbyaddr,
    }
    for owner, name in airgap._GUARDED_ATTRIBUTES:
        setattr(owner, name, replacements[name])
    try:
        yield allowed_ips
    finally:
        for owner, name in airgap._GUARDED_ATTRIBUTES:
            setattr(owner, name, originals[name])


# ---------------------------------------------------------------- transport


@dataclass
class HTTPResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    elapsed_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
        max_bytes: int,
    ) -> HTTPResponse: ...


_REQUEST_ID_HEADERS = {"x-request-id", "x-bce-request-id", "request-id", "x-trace-id"}
_KEEP_HEADERS = {"content-type", "date"} | _REQUEST_ID_HEADERS


class UrllibTransport:
    """Direct HTTPS transport with phase timings and no proxies or redirects.

    The legacy class name is retained for compatibility.  Using ``http.client``
    directly lets us distinguish DNS/TCP/TLS establishment, request upload,
    waiting for response headers, and reading the response body.  The configured
    timeout remains a socket-operation timeout exactly as before.
    """

    def __init__(self, *, connection_factory=None, clock=time.perf_counter) -> None:
        self._context = ssl.create_default_context()
        self._clock = clock
        self._connection_factory = connection_factory or (
            lambda host, port, timeout, context: http.client.HTTPSConnection(
                host, port=port, timeout=timeout, context=context
            )
        )

    def request(self, method, url, *, headers, body, timeout, max_bytes, response_deadline_s=None):
        """``timeout`` is the per-socket-operation (idle) timeout.

        ``response_deadline_s`` (streaming only) additionally bounds the whole response,
        headers plus body, measured from the end of the upload, so a stream that keeps
        trickling can never run longer than the non-streaming header wait did. The body
        is then read chunk by chunk, the first body byte is timed, and on failure the
        partial body is kept (redacted by the caller) as evidence.
        """
        parts = urlsplit(url)
        if parts.scheme != "https":
            raise APIRequestError("insecure_url", "Only https URLs are allowed")
        if not parts.hostname or parts.username or parts.password:
            raise APIRequestError("unsafe_url", "HTTPS URL must have a hostname and no embedded credentials")
        if parts.fragment:
            raise APIRequestError("unsafe_url", "HTTP request URLs must not contain fragments")
        port = parts.port or 443
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        started = self._clock()
        stages: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, Any] = {
            "timeout_s": timeout,
            "stages": stages,
            "http_status": None,
            "response_bytes": 0,
            "request_id_headers": {},
        }
        connection = self._connection_factory(parts.hostname, port, timeout, self._context)

        def elapsed(since: float) -> float:
            return max(0.0, (self._clock() - since) * 1000)

        streaming = response_deadline_s is not None
        chunks: list[bytes] = []
        response_window: dict[str, float] = {}

        def fail(stage: str, exc: BaseException, *, code: str = "request_timeout",
                 message: str | None = None) -> APIRequestError:
            stages[stage] = {"status": "timeout" if code == "request_timeout" else "error",
                             "elapsed_ms": elapsed(stage_started)}
            diagnostics["total_elapsed_ms"] = elapsed(started)
            if streaming and chunks:
                diagnostics["partial_body_text"] = b"".join(chunks).decode("utf-8", "replace")
            message = message or (
                f"Request timed out after {timeout}s during {stage}"
                if code == "request_timeout" else f"Connection failed during {stage}: {exc}"
            )
            return APIRequestError(
                code, message, http_status=diagnostics["http_status"], diagnostics=diagnostics
            )

        try:
            stage_started = self._clock()
            try:
                connection.connect()
            except (TimeoutError, socket.timeout) as exc:
                raise fail("connection", exc) from exc
            except OutboundNetworkBlocked as exc:
                raise fail("connection", exc, code="network_blocked") from exc
            except (OSError, http.client.HTTPException) as exc:
                raise fail("connection", exc, code="network_error") from exc
            stages["connection"] = {"status": "completed", "elapsed_ms": elapsed(stage_started),
                                    "includes": "DNS, TCP and TLS"}

            stage_started = self._clock()
            try:
                connection.putrequest(method, target)
                has_length = any(key.lower() == "content-length" for key in headers)
                for key, value in headers.items():
                    connection.putheader(key, value)
                if body is not None and not has_length:
                    connection.putheader("Content-Length", str(len(body)))
                connection.endheaders()
                if body:
                    connection.send(body)
            except (TimeoutError, socket.timeout) as exc:
                raise fail("sending_request", exc) from exc
            except OutboundNetworkBlocked as exc:
                raise fail("sending_request", exc, code="network_blocked") from exc
            except (OSError, http.client.HTTPException) as exc:
                raise fail("sending_request", exc, code="network_error") from exc
            stages["sending_request"] = {"status": "completed", "elapsed_ms": elapsed(stage_started),
                                         "request_bytes": len(body or b"")}

            stage_started = self._clock()
            if streaming:
                diagnostics["response_deadline_s"] = response_deadline_s
                response_window["end"] = stage_started + float(response_deadline_s)

            def remaining_s() -> float:
                return response_window["end"] - self._clock()

            def clamp_socket_timeout() -> None:
                # Never wait on one read beyond the overall response deadline.
                sock = getattr(connection, "sock", None)
                if streaming and sock is not None:
                    sock.settimeout(max(0.001, min(float(timeout), remaining_s())))

            try:
                clamp_socket_timeout()
                response = connection.getresponse()
            except (TimeoutError, socket.timeout) as exc:
                raise fail("waiting_response_headers", exc) from exc
            except OutboundNetworkBlocked as exc:
                raise fail("waiting_response_headers", exc, code="network_blocked") from exc
            except (OSError, http.client.HTTPException) as exc:
                raise fail("waiting_response_headers", exc, code="network_error") from exc
            status = int(response.status)
            response_headers = dict(response.headers.items())
            diagnostics["http_status"] = status
            diagnostics["request_id_headers"] = {
                key: value for key, value in response_headers.items()
                if key.lower() in _REQUEST_ID_HEADERS
            }
            stages["waiting_response_headers"] = {
                "status": "completed", "elapsed_ms": elapsed(stage_started)
            }

            stage_started = self._clock()
            size = 0
            # read1 returns whatever has arrived (one SSE chunk) instead of waiting for
            # 64 KiB, so stream progress is observable; plain reads otherwise.
            reader = getattr(response, "read1", None) if streaming else None
            reader = reader or response.read
            try:
                while True:
                    if streaming and remaining_s() <= 0:
                        raise fail(
                            "reading_response_body", TimeoutError(), message=(
                                f"Response not complete within {response_deadline_s}s after the "
                                "upload (streaming deadline)"
                            ),
                        )
                    clamp_socket_timeout()
                    chunk = reader(min(65536, max_bytes + 1 - size))
                    if not chunk:
                        break
                    if streaming and not chunks:
                        diagnostics["first_body_byte_ms"] = elapsed(stage_started)
                    chunks.append(chunk)
                    size += len(chunk)
                    diagnostics["response_bytes"] = size
                    if size > max_bytes:
                        break
            except APIRequestError:
                raise
            except (TimeoutError, socket.timeout) as exc:
                raise fail("reading_response_body", exc) from exc
            except (OSError, http.client.HTTPException) as exc:
                raise fail("reading_response_body", exc, code="network_error") from exc
            data = b"".join(chunks)
            stages["reading_response_body"] = {
                "status": "completed", "elapsed_ms": elapsed(stage_started), "response_bytes": size
            }
        finally:
            try:
                connection.close()
            except OSError:
                pass
        if len(data) > max_bytes:
            diagnostics["total_elapsed_ms"] = elapsed(started)
            raise APIRequestError(
                "response_too_large", f"Response exceeded {max_bytes} bytes",
                http_status=status, diagnostics=diagnostics,
            )
        diagnostics["total_elapsed_ms"] = elapsed(started)
        return HTTPResponse(status, response_headers, data, diagnostics["total_elapsed_ms"], diagnostics)


# ------------------------------------------------------- result URL safety

URL_EXPIRY_MARGIN_S = 30
MAX_URL_REFRESHES = 2
MAX_RESULT_REDIRECTS = 2


class ResultHostNotApproved(APIRequestError):
    """The job is done, but its result lives on a host the user has not approved."""

    def __init__(self, info: dict):
        super().__init__(
            "result_host_not_approved",
            f"The job's result is on host {info.get('host')!r}, which has not been approved. "
            "Nothing was downloaded; the job is paused for review.",
        )
        self.info = info


def _expiry_epoch(query: dict[str, list[str]]) -> float | None:
    """Best-effort expiry of common pre-signed URL formats (no value is stored)."""
    lowered = {key.lower(): values for key, values in query.items()}
    for key in ("expires", "x-expires"):
        if lowered.get(key) and lowered[key][0].isdigit():
            return float(lowered[key][0])
    if lowered.get("x-amz-date") and lowered.get("x-amz-expires"):
        try:
            start = datetime.strptime(lowered["x-amz-date"][0], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return start.timestamp() + float(lowered["x-amz-expires"][0])
        except ValueError:
            return None
    auth = (lowered.get("authorization") or [""])[0]
    match = re.match(r"bce-auth-v1/[^/]+/([0-9T:\-Z]+)/(\d+)/", auth)
    if match:
        try:
            start = datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            return start.timestamp() + float(match.group(2))
        except ValueError:
            return None
    return None


def sanitize_url(url: str) -> dict:
    """What may be stored about a URL: never query values (signatures, credentials)."""
    parts = urlsplit(url or "")
    query = parse_qs(parts.query, keep_blank_values=True)
    expires = _expiry_epoch(query)
    return {
        "scheme": parts.scheme,
        "host": (parts.hostname or "").lower(),
        "port": parts.port,
        "path": parts.path,
        "query_parameter_names": sorted(query),
        "expires_at_epoch": expires,
        "expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat() if expires else None,
    }


def validate_result_url(url: str, approved_hosts: frozenset[str]) -> dict:
    """HTTPS, default port, a public DNS name, and exactly an approved host."""
    parts = urlsplit(url or "")
    info = sanitize_url(url)
    host = info["host"]
    if parts.scheme != "https":
        raise APIRequestError("insecure_url", "Result URL is not https; refused")
    if parts.username or parts.password:
        raise APIRequestError("unsafe_result_url", "Result URL carries credentials in the authority; refused")
    if parts.port not in (None, 443):
        raise APIRequestError("unsafe_result_url", f"Result URL uses non-default port {parts.port}; refused")
    try:
        ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        pass
    else:
        raise APIRequestError("unsafe_result_url", f"Result URL host is an IP address ({host}); refused")
    if host == "localhost" or "." not in host:
        raise APIRequestError("unsafe_result_url", f"Result URL host {host!r} is not a public name; refused")
    if host not in approved_hosts:
        raise ResultHostNotApproved(info)
    return info


# ------------------------------------------------------------- Baidu client

def _multipart(fields: dict[str, str], file_field: str, filename: str, content: bytes) -> tuple[bytes, str]:
    boundary = "agenticdoc-" + os.urandom(12).hex()
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            + value.encode("utf-8")
            + b"\r\n"
        )
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode()
        + content
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


@dataclass
class BaiduJobResult:
    job_id: str
    text: str
    blocks: list[dict] | None
    raw_pages: list[dict]
    data_info: dict
    jsonl_text: str
    exchanges: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    official_normalized: dict = field(default_factory=dict)
    result_url: dict = field(default_factory=dict)  # sanitized: no query values
    # Chat protocols: provider-reported identity, termination and token usage.
    reported_model: str | None = None
    reported_model_matches: bool | None = None
    finish_reason: str | None = None
    truncated: bool | None = None
    usage: dict = field(default_factory=dict)


APIJobResult = BaiduJobResult


class BaiduPaddleOCRJobsClient:
    def __init__(
        self,
        spec: ProviderSpec,
        token: str,
        transport: Transport,
        *,
        result_hosts: frozenset[str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        if not token or not token.strip():
            raise APIRequestError("credential_missing", f"No {spec.credential_label} supplied")
        self.spec = spec
        self._token = token.strip()
        self.transport = transport
        # Result hosts: registry hosts plus explicit user approvals (never automatic).
        self.result_hosts = frozenset(result_hosts if result_hosts is not None else spec.result_hosts)
        self.sleep = sleep
        self.clock = clock
        self.wall_clock = wall_clock
        self.exchanges: list[dict] = []
        self.job_id: str | None = None

    # A record of one HTTP exchange; never contains the token or request headers.
    def _record(
        self, stage: str, method: str, url: str, response: HTTPResponse | None,
        note: str | None = None, sent_bytes: int | None = None,
        error: APIRequestError | None = None,
    ) -> None:
        diagnostics = response.diagnostics if response else (error.diagnostics if error else {})
        diagnostic_headers = diagnostics.get("request_id_headers") or {}
        response_headers = response.headers if response else diagnostic_headers
        self.exchanges.append(
            redact(
                {
                    "stage": stage,
                    "method": method,
                    "url": url,
                    "sent_bytes": sent_bytes,
                    "status": response.status if response else (error.http_status if error else None),
                    "elapsed_ms": (
                        response.elapsed_ms if response else diagnostics.get("total_elapsed_ms")
                    ),
                    "response_bytes": (
                        len(response.body) if response else diagnostics.get("response_bytes")
                    ),
                    "response_headers": {
                        key: value for key, value in response_headers.items()
                        if key.lower() in _KEEP_HEADERS
                    },
                    "response_body": response.body.decode("utf-8", "replace") if response else None,
                    "http_diagnostics": diagnostics,
                    "note": note,
                },
                [self._token],
            )
        )

    def _api(self, method: str, url: str, *, body: bytes | None = None,
             content_type: str | None = None, stage: str) -> dict:
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/json"}
        if content_type:
            headers["Content-Type"] = content_type
        try:
            response = self.transport.request(
                method, url, headers=headers, body=body,
                timeout=self.spec.limits.request_timeout_s,
                max_bytes=self.spec.limits.max_response_bytes,
            )
        except APIRequestError as exc:
            self._record(
                stage, method, url, None, note=f"{exc.code}: {exc}",
                sent_bytes=len(body or b""), error=exc,
            )
            raise
        self._record(stage, method, url, response, sent_bytes=len(body or b""))
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except ValueError:
            payload = None
        message = None
        if isinstance(payload, dict):
            message = payload.get("msg") or payload.get("errorMsg") or payload.get("message")
            if not message and isinstance(payload.get("data"), dict):
                message = payload["data"].get("errorMsg")
        if 300 <= response.status < 400:
            destination = sanitize_url(response.headers.get("Location") or response.headers.get("location") or "")
            raise APIRequestError(
                "redirect_refused",
                f"API host answered HTTP {response.status} redirecting to {destination.get('host')!r}; "
                "API calls never follow redirects",
                http_status=response.status,
            )
        if not 200 <= response.status < 300:
            code = {401: "auth_failed", 403: "auth_failed", 400: "invalid_request",
                    429: "rate_limited", 503: "service_unavailable", 504: "service_unavailable"}
            raise APIRequestError(
                code.get(response.status, "provider_error"),
                f"HTTP {response.status}: {redact(str(message or response.body[:300]), [self._token])}",
                http_status=response.status,
            )
        if not isinstance(payload, dict):
            raise APIRequestError("response_format", "Response body is not a JSON object")
        if payload.get("code", 0) not in (0, None):
            raise APIRequestError("provider_error", f"API code {payload.get('code')}: {message}")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise APIRequestError("response_format", "Response JSON lacks object field 'data'")
        return data

    def submit(self, image_path: Path, filename: str) -> str:
        content = Path(image_path).read_bytes()
        body, content_type = _multipart(
            {
                "model": self.spec.api_model_id,
                "optionalPayload": json.dumps(self.spec.optional_payload),
            },
            "file", filename, content,
        )
        data = self._api("POST", self.spec.endpoint, body=body, content_type=content_type, stage="submit")
        job_id = data.get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise APIRequestError("response_format", "Submit response lacks 'jobId'")
        return job_id

    def poll(self, job_id: str) -> dict:
        limits = self.spec.limits
        started = self.clock()
        interval = limits.poll_initial_interval_s
        while True:
            if self.clock() - started > limits.poll_timeout_s:
                raise APIRequestError("poll_timeout", f"Job {job_id} not done after {limits.poll_timeout_s:.0f}s")
            data = self._api("GET", f"{self.spec.endpoint}/{job_id}", stage="poll")
            state = data.get("state")
            if state == "done":
                return data
            if state == "failed":
                raise APIRequestError("job_failed", f"Job {job_id} failed: {data.get('errorMsg') or 'unknown error'}")
            if state not in {"pending", "running"}:
                raise APIRequestError("response_format", f"Unknown job state {state!r}")
            self.sleep(interval)
            interval = min(max(interval * 1.5, 0.0), limits.poll_max_interval_s)

    def status(self, job_id: str) -> dict:
        """One status request for an existing job (used to refresh result URLs)."""
        return self._api("GET", f"{self.spec.endpoint}/{job_id}", stage="status_refresh")

    def _result_url(self, done: dict) -> tuple[str, dict]:
        result_url = done.get("resultUrl") if isinstance(done.get("resultUrl"), dict) else {}
        url = result_url.get("jsonUrl")
        if not isinstance(url, str) or not url:
            raise APIRequestError("response_format", "Done job lacks resultUrl.jsonUrl")
        info = sanitize_url(url)
        if isinstance(result_url.get("markdownUrl"), str):
            info["markdown_url"] = sanitize_url(result_url["markdownUrl"])
        return url, info

    def download(self, job_id: str, done: dict) -> tuple[str, dict]:
        """Download the JSONL result of a done job. The page is never resubmitted.

        Expired (or storage-rejected) URLs are refreshed by re-reading the job status
        with the original Job ID. Redirects are followed only to validated hosts.
        """
        refreshes = 0
        current = done
        while True:
            url, info = self._result_url(current)
            # Host first: an unapproved host pauses for review (no refresh is spent on a
            # host that will not be contacted anyway).
            validate_result_url(url, self.result_hosts)
            expires = info.get("expires_at_epoch")
            if expires is not None and expires <= self.wall_clock() + URL_EXPIRY_MARGIN_S:
                if refreshes >= MAX_URL_REFRESHES:
                    raise APIRequestError("result_url_expired", "Result URL still expired after refreshing the job status")
                refreshes += 1
                self._record("result", "GET", url, None, note="skipped: URL expired; refreshing job status")
                current = self._refreshed(job_id)
                continue
            response = self._follow(url, info)
            if response.status in (401, 403, 404, 410):
                if refreshes >= MAX_URL_REFRESHES:
                    raise APIRequestError(
                        "result_url_expired",
                        f"Result storage answered HTTP {response.status} after {refreshes} refreshes",
                        http_status=response.status,
                    )
                refreshes += 1
                current = self._refreshed(job_id)
                continue
            if not 200 <= response.status < 300:
                raise APIRequestError("provider_error", f"Result download HTTP {response.status}",
                                      http_status=response.status)
            return response.body.decode("utf-8"), info | {"refreshes": refreshes}

    def _refreshed(self, job_id: str) -> dict:
        data = self.status(job_id)
        if data.get("state") != "done":
            raise APIRequestError("response_format", f"Job {job_id} is no longer done ({data.get('state')!r})")
        return data

    def _follow(self, url: str, info: dict) -> HTTPResponse:
        """GET a validated result URL (no token), following at most two validated redirects."""
        for _hop in range(MAX_RESULT_REDIRECTS + 1):
            try:
                response = self.transport.request(
                    "GET", url, headers={"Accept": "application/json"}, body=None,
                    timeout=self.spec.limits.request_timeout_s,
                    max_bytes=self.spec.limits.max_response_bytes,
                )
            except APIRequestError as exc:
                self._record(
                    "result", "GET", url, None, note=f"{exc.code}: {exc}", error=exc
                )
                raise
            self._record("result", "GET", url, response)
            if not 300 <= response.status < 400:
                return response
            location = response.headers.get("Location") or response.headers.get("location") or ""
            target = urljoin(url, location)
            try:
                validate_result_url(target, self.result_hosts)
            except ResultHostNotApproved as exc:
                raise APIRequestError(
                    "unsafe_redirect",
                    f"Result storage redirected to unapproved host {exc.info.get('host')!r}; refused",
                ) from exc
            except APIRequestError as exc:
                raise APIRequestError("unsafe_redirect", f"Unsafe redirect refused: {exc}") from exc
            url = target
        raise APIRequestError("unsafe_redirect", f"More than {MAX_RESULT_REDIRECTS} redirects; refused")

    def run(self, image_path: Path, filename: str, *, job_id: str | None = None,
            on_submitted: Callable[[str], None] | None = None) -> BaiduJobResult:
        """Submit (only when no job_id is given) or resume an existing job."""
        timings: dict[str, float] = {}
        started = time.perf_counter()
        if job_id is None:
            job_id = self.submit(image_path, filename)
            timings["submit_ms"] = (time.perf_counter() - started) * 1000
            if on_submitted:
                on_submitted(job_id)
        self.job_id = job_id
        poll_started = time.perf_counter()
        done = self.poll(job_id)
        timings["processing_ms"] = (time.perf_counter() - poll_started) * 1000
        fetch_started = time.perf_counter()
        jsonl_text, result_info = self.download(job_id, done)
        timings["result_ms"] = (time.perf_counter() - fetch_started) * 1000
        timings["total_ms"] = (time.perf_counter() - started) * 1000
        pages, data_info = parse_document_jsonl(jsonl_text)
        if len(pages) != 1:
            raise APIRequestError("response_format", f"Expected 1 parsed page, got {len(pages)}")
        page = pages[0]
        normalized = _normalize(page)
        return BaiduJobResult(
            job_id=job_id,
            text=normalized["text"],
            blocks=normalized["blocks"],
            raw_pages=[redact(item, [self._token]) for item in pages],
            data_info=redact(data_info, [self._token]),
            jsonl_text=redact(jsonl_text, [self._token]),
            exchanges=self.exchanges,
            timings_ms=timings,
            official_normalized=normalized["raw"],
            result_url=result_info,
        )


class OpenAIChatClient:
    """One synchronous OpenAI-compatible chat completion for one page image.

    There is no job: the request either returns the transcription or fails. A request
    whose outcome is unknown (timeout, dropped connection) may already have been
    processed and billed, so it is never retried automatically.
    """

    def __init__(self, spec: ProviderSpec, token: str, transport: Transport, *,
                 result_hosts: frozenset[str] | None = None, sleep=None, clock=None, wall_clock=None):
        if not token or not token.strip():
            raise APIRequestError("credential_missing", f"No {spec.credential_label} supplied")
        self.spec = spec
        self._token = token.strip()
        self.transport = transport
        self.exchanges: list[dict] = []
        self.job_id: str | None = None

    _record = BaiduPaddleOCRJobsClient._record

    def request_body(self, image_bytes: bytes) -> dict:
        from lab.core.e2.prompts import get_prompt  # noqa: PLC0415

        prompt = get_prompt(self.spec.prompt_id or "")
        mime = "image/jpeg" if image_bytes[:3] == JPEG_MAGIC else "image/png"  # PNG unless JPEG bytes
        data_uri = f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
        return {
            "model": self.spec.api_model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "stream": bool(self.spec.stream),
            **self.spec.generation,
        }

    def run(self, image_path: Path | None, filename: str | None, *, job_id: str | None = None,
            on_submitted: Callable[[str], None] | None = None) -> BaiduJobResult:
        if job_id is not None or image_path is None:
            raise APIRequestError("not_resumable", "Synchronous chat requests have no job to resume")
        body = json.dumps(self.request_body(Path(image_path).read_bytes())).encode("utf-8")
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        options: dict[str, Any] = {}
        if self.spec.stream:
            # Same 120 s budget as the non-streaming header wait, now covering the
            # whole response; never longer.
            options["response_deadline_s"] = self.spec.limits.request_timeout_s
            headers["Accept"] = "text/event-stream, application/json"
        try:
            response = self.transport.request(
                "POST", self.spec.endpoint, headers=headers, body=body,
                timeout=self.spec.limits.request_timeout_s, max_bytes=self.spec.limits.max_response_bytes,
                **options,
            )
        except APIRequestError as exc:
            partial = exc.diagnostics.get("partial_body_text")
            if isinstance(partial, str):
                # Evidence of how far the provider got. The outcome stays unconfirmed
                # (never retried); the partial text is recorded, not used as a result.
                exc.diagnostics["stream_progress"] = sse_summary(partial)
            self._record(
                "chat", "POST", self.spec.endpoint, None, note=f"{exc.code}: {exc}",
                sent_bytes=len(body), error=exc,
            )
            raise
        # Never store the base64 page in the exchange log (it is stored locally already).
        self._record("chat", "POST", self.spec.endpoint, response, sent_bytes=len(body))
        if 300 <= response.status < 400:
            raise APIRequestError("redirect_refused", f"API answered HTTP {response.status}; redirects are refused",
                                  http_status=response.status)
        content_type = next(
            (value for key, value in response.headers.items() if key.lower() == "content-type"), ""
        )
        is_sse = self.spec.stream and (
            "text/event-stream" in content_type.lower() or response.body.lstrip().startswith(b"data:")
        )
        if is_sse and 200 <= response.status < 300:
            try:
                payload = assemble_sse(response.body.decode("utf-8", "replace"))
            except APIRequestError as exc:
                exc.http_status = response.status
                exc.diagnostics = response.diagnostics
                raise
        else:
            try:
                payload = json.loads(response.body.decode("utf-8"))
            except ValueError:
                payload = None
        if not 200 <= response.status < 300:
            message = ""
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    message = (
                        error.get("message") or error.get("title") or error.get("type")
                        or payload.get("status_message") or payload.get("message") or ""
                    )
                else:
                    message = error or payload.get("status_message") or payload.get("message") or ""
                message = str(message)
            codes = {400: "invalid_request", 401: "auth_failed", 402: "payment_required", 403: "auth_failed",
                     404: "model_not_found", 429: "rate_limited", 503: "service_unavailable"}
            raise APIRequestError(codes.get(response.status, "provider_error"),
                                  f"HTTP {response.status}: {redact(message[:300], [self._token])}",
                                  http_status=response.status)
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or not payload["choices"]:
            raise APIRequestError("response_format", "Response lacks choices")
        choice = payload["choices"][0]
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, list):  # some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            raise APIRequestError("response_format", "Response choice lacks text content")
        self.job_id = str(payload.get("id") or "") or None
        finish = choice.get("finish_reason")
        reported = payload.get("model")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        elapsed = (time.perf_counter() - started) * 1000
        return BaiduJobResult(
            job_id=self.job_id or "",
            text=content,
            blocks=None,
            raw_pages=[redact({key: value for key, value in payload.items()}, [self._token])],
            data_info={},
            jsonl_text=redact(response.body.decode("utf-8", "replace"), [self._token]),
            exchanges=self.exchanges,
            timings_ms={"total_ms": elapsed},
            reported_model=reported,
            reported_model_matches=(str(reported).lower() == self.spec.api_model_id.lower()) if reported else None,
            finish_reason=finish,
            truncated=True if finish == "length" else False if finish == "stop" else None,
            usage={
                **{key: usage.get(key)
                   for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")},
                # Featherless reports a number; Oxen a decimal string plus `currency`.
                "cost": reported_usd_cost(usage.get("cost"), usage.get("currency")),
                "cost_reported_raw": usage.get("cost"),
                "currency": usage.get("currency"),
            },
        )


def _sse_events(text: str) -> tuple[list[dict], bool, list[str]]:
    """Parse OpenAI-style server-sent events: (json chunks, saw [DONE], unparsable)."""
    chunks: list[dict] = []
    bad: list[str] = []
    done = False
    for block in text.replace("\r\n", "\n").split("\n\n"):
        data = "\n".join(
            line[5:].lstrip(" ") for line in block.split("\n") if line.startswith("data:")
        )
        if not data:
            continue  # comments (": keep-alive"), event/id lines, blank keep-alives
        if data.strip() == "[DONE]":
            done = True
            continue
        try:
            value = json.loads(data)
        except ValueError:
            bad.append(data[:200])
            continue
        if isinstance(value, dict):
            chunks.append(value)
    return chunks, done, bad


def _sse_accumulate(chunks: list[dict]) -> dict[str, Any]:
    content: list[str] = []
    state: dict[str, Any] = {"id": None, "model": None, "finish_reason": None, "usage": None,
                             "error": None}
    for chunk in chunks:
        if chunk.get("error") is not None:
            state["error"] = chunk["error"]
        state["id"] = state["id"] or chunk.get("id")
        state["model"] = state["model"] or chunk.get("model")
        if isinstance(chunk.get("usage"), dict):
            state["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            piece = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(piece, list):
                piece = "".join(part.get("text", "") for part in piece if isinstance(part, dict))
            if isinstance(piece, str):
                content.append(piece)
            if choice.get("finish_reason"):
                state["finish_reason"] = choice["finish_reason"]
    state["content"] = "".join(content)
    return state


def sse_summary(text: str) -> dict[str, Any]:
    """How far an interrupted stream got (evidence only; never treated as a result)."""
    chunks, done, bad = _sse_events(text)
    state = _sse_accumulate(chunks)
    return {"events": len(chunks), "done_marker": done, "unparsable_events": len(bad),
            "completion_id": state["id"], "finish_reason": state["finish_reason"],
            "content_chars": len(state["content"]), "partial_content": state["content"],
            "usage": state["usage"], "error": state["error"]}


def assemble_sse(text: str) -> dict[str, Any]:
    """Rebuild a non-streaming chat completion from a complete SSE body.

    Raises when the stream reports an error or ended without a finish reason and the
    [DONE] marker (the provider stopped early). These are definite provider answers,
    so they are terminal failures, never automatic retries.
    """
    chunks, done, bad = _sse_events(text)
    state = _sse_accumulate(chunks)
    if state["error"] is not None:
        error = state["error"]
        message = error.get("message") if isinstance(error, dict) else error
        raise APIRequestError("provider_error", f"Stream reported an error: {str(message or '')[:300]}")
    if not chunks:
        raise APIRequestError("response_format", "Event stream contained no data events")
    if not done and not state["finish_reason"]:
        raise APIRequestError(
            "stream_incomplete",
            f"Event stream ended early after {len(chunks)} events and {len(state['content'])} characters "
            "(no finish_reason, no [DONE]); the provider may have billed it",
        )
    payload: dict[str, Any] = {
        "id": state["id"], "object": "chat.completion", "model": state["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": state["content"]},
                     "finish_reason": state["finish_reason"]}],
        "stream": {"events": len(chunks), "done_marker": done, "unparsable_events": len(bad)},
    }
    if state["usage"] is not None:
        payload["usage"] = state["usage"]
    return payload


def reported_usd_cost(value: Any, currency: Any = None) -> float | None:
    """A provider-reported charge in USD, or None when it is absent or not usable.

    Accepts a finite non-negative number or decimal string. A stated currency other
    than USD is never treated as dollars (the cost then falls back to the bound).
    """
    if currency is not None and str(currency).strip().upper() != "USD":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(Decimal(value.strip()))
        except (InvalidOperation, ValueError):
            return None
    else:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def make_client(spec: ProviderSpec, token: str, transport: Transport, **kwargs):
    """The client for a provider's protocol (only implemented protocols)."""
    if spec.protocol == "baidu_paddleocr_jobs_v2":
        return BaiduPaddleOCRJobsClient(spec, token, transport, **kwargs)
    if spec.protocol == "openai_chat_completions_v1":
        return OpenAIChatClient(spec, token, transport, **kwargs)
    raise APIRequestError("not_implemented", f"Protocol {spec.protocol!r} is not implemented")


def parse_document_jsonl(text: str) -> tuple[list[dict], dict]:
    pages: list[dict] = []
    data_info: dict = {}
    try:
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            result = json.loads(line)["result"]
            if isinstance(result.get("dataInfo"), dict):
                data_info.update(result["dataInfo"])
            for item in result["layoutParsingResults"]:
                if not isinstance(item.get("markdown"), dict) or not isinstance(item["markdown"].get("text"), str):
                    raise KeyError("markdown.text")
                pages.append(item)
    except (KeyError, TypeError, ValueError) as exc:
        raise APIRequestError("response_format", f"Malformed document parsing result: {exc}") from exc
    return pages, data_info


def _normalize(page: dict) -> dict:
    """Reuse the local official-pipeline normalizer on the hosted result."""
    from lab.core.e2.backends import normalize_official_pipeline_result  # noqa: PLC0415

    pruned = page.get("prunedResult")
    shim = SimpleNamespace(
        json={"res": pruned if isinstance(pruned, dict) else {}},
        markdown={"markdown_texts": page["markdown"]["text"]},
    )
    output = normalize_official_pipeline_result(shim)
    return {"text": output.text, "blocks": output.blocks, "raw": output.raw}
