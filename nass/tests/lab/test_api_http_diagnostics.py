"""Phase-aware hosted-API HTTP diagnostics. All connections are mocked."""

from __future__ import annotations

import json
import socket

import pytest

from lab.core.e2.api.client import APIRequestError, OpenAIChatClient, UrllibTransport
from tests.lab.test_api_featherless import TOKEN, _completion, _spec


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


class _Response:
    def __init__(self, steps, *, status=200, headers=None) -> None:
        self.status = status
        self.headers = headers or {}
        self.steps = list(steps)

    def read(self, _size: int) -> bytes:
        value = self.steps.pop(0) if self.steps else b""
        if isinstance(value, BaseException):
            raise value
        return value


class _Connection:
    def __init__(self, *, fail_at=None, response=None) -> None:
        self.fail_at = fail_at
        self.response = response or _Response([b""])
        self.headers: dict[str, str] = {}
        self.body = b""
        self.closed = False

    def _fail(self, stage: str) -> None:
        if self.fail_at == stage:
            raise socket.timeout("mock timeout")

    def connect(self) -> None:
        self._fail("connection")

    def putrequest(self, method: str, target: str) -> None:
        self.method = method
        self.target = target

    def putheader(self, key: str, value: str) -> None:
        self.headers[key] = value

    def endheaders(self) -> None:
        self._fail("sending_request")

    def send(self, body: bytes) -> None:
        self.body += body

    def getresponse(self):
        self._fail("waiting_response_headers")
        return self.response

    def close(self) -> None:
        self.closed = True


def _transport(connection: _Connection) -> UrllibTransport:
    return UrllibTransport(
        connection_factory=lambda _host, _port, _timeout, _context: connection,
        clock=_Clock(),
    )


def _request(transport: UrllibTransport):
    return transport.request(
        "POST", "https://hub.oxen.ai/api/ai/chat/completions",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
        body=b"{}", timeout=120, max_bytes=5_000_000,
    )


def test_connection_timeout_is_phase_specific_and_redacted() -> None:
    connection = _Connection(fail_at="connection")
    with pytest.raises(APIRequestError) as raised:
        _request(_transport(connection))
    error = raised.value
    assert error.code == "request_timeout" and "during connection" in str(error)
    assert error.diagnostics["stages"] == {
        "connection": {"status": "timeout", "elapsed_ms": pytest.approx(10.0)}
    }
    assert error.diagnostics["http_status"] is None
    assert TOKEN not in json.dumps(error.diagnostics)
    assert connection.closed


def test_response_header_timeout_is_distinct_from_connection() -> None:
    connection = _Connection(fail_at="waiting_response_headers")
    with pytest.raises(APIRequestError) as raised:
        _request(_transport(connection))
    diagnostics = raised.value.diagnostics
    assert diagnostics["stages"]["connection"]["status"] == "completed"
    assert diagnostics["stages"]["sending_request"]["status"] == "completed"
    assert diagnostics["stages"]["waiting_response_headers"]["status"] == "timeout"
    assert "reading_response_body" not in diagnostics["stages"]
    assert diagnostics["http_status"] is None and diagnostics["response_bytes"] == 0


def test_body_timeout_preserves_status_size_and_request_id_in_redacted_record(tmp_path) -> None:
    response = _Response(
        [b'{"partial":', socket.timeout("mock timeout")],
        status=200,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": "req-oxen-123",
            "Set-Cookie": f"secret={TOKEN}",
        },
    )
    client = OpenAIChatClient(_spec(), TOKEN, _transport(_Connection(response=response)))
    image = tmp_path / "page.png"
    image.write_bytes(b"mock png")
    with pytest.raises(APIRequestError, match="reading_response_body") as raised:
        client.run(image, "page.png")
    diagnostics = raised.value.diagnostics
    assert diagnostics["http_status"] == 200 and diagnostics["response_bytes"] == 11
    assert diagnostics["request_id_headers"] == {"X-Request-Id": "req-oxen-123"}
    assert diagnostics["stages"]["reading_response_body"]["status"] == "timeout"
    (record,) = client.exchanges
    assert record["status"] == 200 and record["response_bytes"] == 11
    assert record["response_headers"] == {"X-Request-Id": "req-oxen-123"}
    assert record["response_body"] is None
    stored = json.dumps(record)
    assert TOKEN not in stored and "Authorization" not in stored and "Set-Cookie" not in stored


def test_success_records_all_phase_timings_status_and_response_size(tmp_path) -> None:
    expected = _completion().body
    response = _Response(
        [expected[:25], expected[25:], b""],
        status=200,
        headers={"Content-Type": "application/json", "Request-Id": "success-1"},
    )
    connection = _Connection(response=response)
    client = OpenAIChatClient(_spec(), TOKEN, _transport(connection))
    image = tmp_path / "page.png"
    image.write_bytes(b"mock png")
    result = client.run(image, "page.png")
    assert result.text
    (record,) = client.exchanges
    assert record["status"] == 200 and record["response_bytes"] == len(expected)
    assert record["response_headers"] == {
        "Content-Type": "application/json", "Request-Id": "success-1"
    }
    diagnostics = record["http_diagnostics"]
    assert diagnostics["response_bytes"] == len(expected)
    assert diagnostics["request_id_headers"] == {"Request-Id": "success-1"}
    assert set(diagnostics["stages"]) == {
        "connection", "sending_request", "waiting_response_headers", "reading_response_body"
    }
    assert all(item["status"] == "completed" for item in diagnostics["stages"].values())
    assert connection.headers["Content-Length"] == str(len(connection.body))
    assert TOKEN not in json.dumps(record)
