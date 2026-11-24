"""HTTP client for runpod/qari_service.py. The URL comes from configuration only (no hardcoded host).

Outcomes are classified so the pipeline can stay resumable without ever guessing:
- QariUnavailable: nothing was transmitted (connection refused, DNS, 503 model not loaded, 429 queue
  full, 401 wrong token). The job stays pending; it may be retried later.
- QariUnconfirmed: the request body was (possibly) transmitted but no response arrived. The image is
  NEVER resent automatically; `recover` only asks the service for a completed result by request_id.
- QariServiceError: the service answered with an error (400/413/500). Recorded as a failed job.
There is no fallback model: if Qari cannot answer, the job is not done.

Endpoint trust (QARI_DEPLOYMENT_MODE, set by the administrator): local = loopback only; internal = a
company-internal server (no public IP address); development_external = https and a host listed in
QARI_TRUSTED_DEV_HOSTS. Anything else is refused before sending. Redirects are never followed. The
token is sent only in the Authorization header and is never logged or returned in records.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
from typing import Any
from urllib.parse import urlsplit

LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1"})
MAX_RESPONSE_BYTES = 10_000_000


class QariClientError(RuntimeError):
    kind = "error"

    def __init__(self, message: str, *, status: int | None = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status, self.detail = status, detail

    def record(self) -> dict:
        return {"kind": self.kind, "message": str(self), "status": self.status, "detail": self.detail}


class QariUnavailable(QariClientError):
    kind = "unavailable"


class QariUnconfirmed(QariClientError):
    kind = "unconfirmed"


class QariServiceError(QariClientError):
    kind = "service_error"


class QariUntrusted(QariUnavailable):
    """The configured endpoint is not trusted for this deployment mode (nothing is sent)."""
    kind = "untrusted"


DEPLOYMENT_MODES = ("local", "internal", "development_external")


def endpoint_info(url: str | None, *, deployment_mode: str = "local",
                  trusted_development_hosts: tuple[str, ...] | list[str] = ()) -> dict:
    """Parse the configured service URL and apply the administrator's trust rules (no network).

    local                 loopback host only (localhost / 127.0.0.1 / ::1); http or https
    internal              a company-internal server: loopback, private IP or internal host name; a
                          public IP address is refused; http or https
    development_external  https only, and the host must be listed exactly in trusted_development_hosts
    Any other URL is refused before anything is sent (QariUntrusted)."""
    if deployment_mode not in DEPLOYMENT_MODES:
        raise QariUntrusted(f"unknown QARI_DEPLOYMENT_MODE {deployment_mode!r} (use {', '.join(DEPLOYMENT_MODES)})")
    if not url or not url.strip():
        raise QariUnavailable("QARI_SERVICE_URL is not configured")
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise QariUnavailable(f"invalid Qari service URL (need http(s)://host[:port]): {url!r}")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise QariUnavailable("the Qari service URL must not contain credentials, a query or a fragment")
    host = parts.hostname.lower()
    loopback, private, public_ip = host in LOOPBACK, False, False
    try:
        address = ipaddress.ip_address(host)
        loopback = loopback or address.is_loopback
        private = address.is_private
        public_ip = address.is_global
    except ValueError:
        pass
    trusted_hosts = {h.strip().lower() for h in trusted_development_hosts or () if h.strip()}
    if deployment_mode == "local":
        if not loopback:
            raise QariUntrusted(f"QARI_DEPLOYMENT_MODE=local accepts only a loopback host, not {host!r}")
        basis = "loopback host (local mode)"
    elif deployment_mode == "internal":
        if public_ip:
            raise QariUntrusted(f"QARI_DEPLOYMENT_MODE=internal refuses the public IP address {host!r}")
        basis = "internal server configured by the administrator (internal mode)"
    else:
        if parts.scheme != "https":
            raise QariUntrusted("QARI_DEPLOYMENT_MODE=development_external requires https")
        if host not in trusted_hosts:
            raise QariUntrusted(f"{host!r} is not in QARI_TRUSTED_DEV_HOSTS; external endpoints are trusted only "
                                "when the administrator lists them explicitly")
        basis = "host listed in QARI_TRUSTED_DEV_HOSTS (development_external mode)"
    base = f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"
    return {"base_url": base, "host": host, "scheme": parts.scheme, "loopback": loopback, "private_ip": private,
            "deployment_mode": deployment_mode, "trust_basis": basis, "display": f"{parts.scheme}://{parts.netloc}",
            "external": deployment_mode == "development_external"}


class QariClient:
    def __init__(self, url: str | None, token: str | None, *, timeout_s: float = 180.0, connect_retries: int = 2,
                 deployment_mode: str = "local", trusted_development_hosts: tuple[str, ...] | list[str] = (),
                 transport: Any = None) -> None:
        import httpx  # noqa: PLC0415

        self.endpoint = endpoint_info(url, deployment_mode=deployment_mode,
                                      trusted_development_hosts=trusted_development_hosts)
        self.token = token or None
        self.timeout_s = float(timeout_s)
        self.connect_retries = int(connect_retries)
        self._httpx = httpx
        self._client = httpx.Client(base_url=self.endpoint["base_url"], follow_redirects=False,
                                    timeout=httpx.Timeout(self.timeout_s, connect=min(15.0, self.timeout_s)),
                                    transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QariClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @staticmethod
    def _detail(response) -> Any:
        try:
            return response.json().get("detail")
        except Exception:  # noqa: BLE001
            return response.text[:500]

    def health(self) -> dict:
        """GET /health. Never transmits document content."""
        httpx = self._httpx
        try:
            response = self._client.get("/health", timeout=min(15.0, self.timeout_s))
        except httpx.HTTPError as exc:
            raise QariUnavailable(f"Qari service not reachable: {type(exc).__name__}") from exc
        if response.status_code != 200:
            raise QariUnavailable(f"Qari health check returned HTTP {response.status_code}",
                                  status=response.status_code, detail=self._detail(response))
        return response.json()

    def ocr(self, request_id: str, kind: str, image: bytes) -> dict:
        """POST /v1/ocr with the image bytes unchanged. Raises a classified error on any failure."""
        httpx = self._httpx
        body = {"request_id": request_id, "kind": kind,
                "image_png_b64": base64.b64encode(image).decode("ascii"),
                "image_sha256": hashlib.sha256(image).hexdigest()}
        attempt = 0
        while True:
            try:
                response = self._client.post("/v1/ocr", json=body, headers=self._headers())
                break
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:  # nothing sent: safe to try again
                attempt += 1
                if attempt > self.connect_retries:
                    raise QariUnavailable(f"Qari service not reachable: {type(exc).__name__}") from exc
            except httpx.HTTPError as exc:  # the body may have been received: never resent automatically
                raise QariUnconfirmed(f"no response after sending ({type(exc).__name__}); the request is not "
                                      "resent automatically; use recover() or an explicit retry") from exc
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise QariServiceError("response too large", status=response.status_code)
        if response.status_code == 200:
            payload = response.json()
            if payload.get("request_id") != request_id or payload.get("image", {}).get("sha256") != body["image_sha256"]:
                raise QariServiceError("response does not match the request (request_id or image hash)",
                                       status=200)
            return payload
        detail = self._detail(response)
        if response.status_code in {401, 403, 429, 503}:
            raise QariUnavailable(f"Qari service refused the request (HTTP {response.status_code})",
                                  status=response.status_code, detail=detail)
        if 300 <= response.status_code < 400:
            raise QariUnavailable("the Qari service answered with a redirect (not followed)",
                                  status=response.status_code)
        raise QariServiceError(f"Qari service error (HTTP {response.status_code})", status=response.status_code,
                               detail=detail)

    def recover(self, request_id: str) -> dict | None:
        """Ask for a completed result of an unconfirmed request (no image is sent). None if unknown."""
        httpx = self._httpx
        try:
            response = self._client.get(f"/v1/results/{request_id}", headers=self._headers(),
                                        timeout=min(30.0, self.timeout_s))
        except httpx.HTTPError as exc:
            raise QariUnavailable(f"Qari service not reachable: {type(exc).__name__}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise QariUnavailable(f"recovery returned HTTP {response.status_code}", status=response.status_code)
        return response.json()


def from_config(config, *, transport: Any = None) -> QariClient:
    return QariClient(config.qari.service_url, config.qari_token(), timeout_s=config.qari.request_timeout_s,
                      connect_retries=config.qari.max_connect_retries,
                      deployment_mode=config.qari.deployment_mode,
                      trusted_development_hosts=config.qari.trusted_development_hosts, transport=transport)


def endpoint_from_config(config) -> dict:
    return endpoint_info(config.qari.service_url, deployment_mode=config.qari.deployment_mode,
                         trusted_development_hosts=config.qari.trusted_development_hosts)
