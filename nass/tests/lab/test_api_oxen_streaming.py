"""Oxen streaming (server-sent events) and the 120 s response budget.

Regression tests for the live failures of 2026-11-17: two non-streaming requests got
no response headers within 120 s after a complete upload. Every connection and
response here is mocked; nothing is sent anywhere and no credit is used.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from lab.core.e2.api import jobs as job_store
from lab.core.e2.api import page_artifacts
from lab.core.e2.api.client import (
    APIRequestError,
    HTTPResponse,
    UrllibTransport,
    assemble_sse,
    sse_summary,
)
from lab.core.e2.api.providers import ProviderSpec, load_providers
from tests.lab import test_api_featherless as feather
from tests.lab.test_api_oxen import _submit, lab  # noqa: F401 - pytest fixture
from tests.lab.test_e2d_batch import OXEN_ID, OXEN_TOKEN, _stored_text

# ------------------------------------------------------------------ helpers


def _sse(*events, done=True, keepalive=True) -> bytes:
    parts = [": keep-alive\n\n"] if keepalive else []
    parts += [f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events]
    if done:
        parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def _chunk(text=None, finish=None, usage=None, cid="chatcmpl-ox-1"):
    event = {"id": cid, "object": "chat.completion.chunk", "model": "qwen3-vl-4b-instruct",
             "choices": [{"index": 0, "delta": {} if text is None else {"content": text},
                          "finish_reason": finish}]}
    if usage is not None:
        event["usage"] = usage
    return event


def _stream_response(body: bytes, status=200) -> HTTPResponse:
    return HTTPResponse(status, {"Content-Type": "text/event-stream", "X-Request-Id": "ox-req-9"}, body, 5.0)


USAGE = {"prompt_tokens": 3700, "completion_tokens": 410, "total_tokens": 4110,
         "cost": "0.030800000", "currency": "USD"}


class _Clock:
    """Advances 0.01 s per reading; `jump` simulates time passing on the wire."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value

    def jump(self, seconds: float) -> None:
        self.value += seconds


class _Sock:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _StreamResponse:
    """Delivers SSE chunks one by one through read1 (as http.client does)."""

    def __init__(self, clock: _Clock, chunks, *, seconds_per_chunk=0.0, status=200) -> None:
        self.clock = clock
        self.chunks = list(chunks)
        self.seconds = seconds_per_chunk
        self.status = status
        self.headers = {"Content-Type": "text/event-stream", "X-Request-Id": "ox-req-9"}

    def read1(self, _size: int) -> bytes:
        self.clock.jump(self.seconds)
        value = self.chunks.pop(0) if self.chunks else b""
        if isinstance(value, BaseException):
            raise value
        return value

    def read(self, _size: int) -> bytes:  # must not be used when streaming
        raise AssertionError("streaming must read with read1")


class _Connection:
    def __init__(self, response, *, header_wait_s=0.0, clock=None) -> None:
        self.response = response
        self.header_wait_s = header_wait_s
        self.clock = clock
        self.sock = _Sock()
        self.body = b""

    def connect(self) -> None: ...
    def putrequest(self, method, target) -> None: ...
    def putheader(self, key, value) -> None: ...
    def endheaders(self) -> None: ...
    def close(self) -> None: ...

    def send(self, body: bytes) -> None:
        self.body += body

    def getresponse(self):
        if self.header_wait_s:
            self.clock.jump(self.header_wait_s)
        return self.response


def _transport(connection, clock) -> UrllibTransport:
    return UrllibTransport(connection_factory=lambda *_args: connection, clock=clock)


def _post(transport, **options):
    return transport.request(
        "POST", "https://hub.oxen.ai/api/ai/chat/completions",
        headers={"Authorization": f"Bearer {OXEN_TOKEN}"}, body=b"{}", timeout=120,
        max_bytes=5_000_000, **options,
    )


# ---------------------------------------------------------------- registry


def test_only_oxen_streams_and_other_identities_are_unchanged() -> None:
    providers = {item.id: item for item in load_providers()}
    oxen = providers[OXEN_ID]
    assert oxen.stream is True and oxen.identity()["stream"] is True
    assert oxen.limits.request_timeout_s == 120  # the budget was not increased
    for provider_id, spec in providers.items():
        if provider_id != OXEN_ID:
            # Existing runs' identities (Stage 2 reuse fingerprints) stay identical.
            assert spec.stream is False and "stream" not in spec.identity()
    baidu = providers["baidu-aistudio-paddleocr-vl-1.6"]
    with pytest.raises(ValidationError, match="only supported for chat"):
        ProviderSpec.model_validate(baidu.model_dump() | {"stream": True})


# --------------------------------------------------------------- transport


def test_streamed_response_is_read_incrementally_with_first_byte_timing() -> None:
    clock = _Clock()
    body = _sse(_chunk("نص "), _chunk("الصفحة", finish="stop"), _chunk(usage=USAGE))
    response = _StreamResponse(clock, [body[:40], body[40:], b""], seconds_per_chunk=3.0)
    connection = _Connection(response, header_wait_s=4.0, clock=clock)
    result = _post(_transport(connection, clock), response_deadline_s=120)
    assert result.body == body
    diagnostics = result.diagnostics
    assert diagnostics["response_deadline_s"] == 120
    assert diagnostics["stages"]["waiting_response_headers"]["elapsed_ms"] == pytest.approx(4010, abs=50)
    assert diagnostics["first_body_byte_ms"] == pytest.approx(3000, abs=50)
    # Every read's socket timeout is clamped to the remaining response budget (<= 120 s).
    assert connection.sock.timeouts and all(0 < value <= 120 for value in connection.sock.timeouts)
    assert connection.sock.timeouts == sorted(connection.sock.timeouts, reverse=True)


def test_trickling_stream_cannot_exceed_the_120s_budget_and_keeps_partial_evidence() -> None:
    clock = _Clock()
    pieces = [f"data: {json.dumps(_chunk(f'w{i} '))}\n\n".encode() for i in range(10)]
    # Each chunk arrives within the 120 s idle timeout, but together they never finish.
    response = _StreamResponse(clock, pieces, seconds_per_chunk=50.0)
    with pytest.raises(APIRequestError) as raised:
        _post(_transport(_Connection(response), clock), response_deadline_s=120)
    error = raised.value
    assert error.code == "request_timeout" and "streaming deadline" in str(error)
    assert error.http_status == 200
    diagnostics = error.diagnostics
    assert diagnostics["stages"]["reading_response_body"]["status"] == "timeout"
    assert diagnostics["partial_body_text"].count("data:") == 3  # 3 x 50 s > 120 s
    assert sse_summary(diagnostics["partial_body_text"])["partial_content"] == "w0 w1 w2 "


def test_non_streaming_transport_behaviour_is_unchanged() -> None:
    clock = _Clock()

    class _Plain(_StreamResponse):
        """Without a response deadline the transport must use plain read()."""

        def read(self, _size):
            return self.chunks.pop(0) if self.chunks else b""

        def read1(self, _size):
            raise AssertionError("read1 must not be used without streaming")

    connection = _Connection(_Plain(clock, [b'{"ok": true}', b""]))
    result = _post(_transport(connection, clock))
    assert result.body == b'{"ok": true}'
    assert "response_deadline_s" not in result.diagnostics and connection.sock.timeouts == []


# -------------------------------------------------------------- SSE parsing


def test_sse_assembly_handles_keepalives_crlf_usage_and_content_parts() -> None:
    body = _sse(_chunk("a"), {"id": "chatcmpl-ox-1", "choices": [{"delta": {"content": [
        {"type": "text", "text": "b"}]}}]}, _chunk(finish="stop"), _chunk(usage=USAGE))
    payload = assemble_sse(body.decode().replace("\n", "\r\n"))
    assert payload["choices"][0]["message"]["content"] == "ab"
    assert payload["choices"][0]["finish_reason"] == "stop"
    assert payload["usage"]["cost"] == "0.030800000" and payload["id"] == "chatcmpl-ox-1"
    with pytest.raises(APIRequestError) as raised:
        assemble_sse(_sse(_chunk("a"), done=False).decode())
    assert raised.value.code == "stream_incomplete"
    with pytest.raises(APIRequestError, match="Stream reported an error") as raised:
        assemble_sse(_sse(_chunk("a"), {"error": {"message": "model overloaded"}}).decode())
    assert raised.value.code == "provider_error"
    with pytest.raises(APIRequestError) as raised:
        assemble_sse(": keep-alive\n\n")
    assert raised.value.code == "response_format"


# --------------------------------------------------- runner + page artifacts


def test_streamed_page_completes_reconciles_cost_and_is_saved_under_pages_oxen(lab) -> None:  # noqa: F811
    _runner, store, *_ = lab
    body = _sse(_chunk("نص الصفحة "), _chunk("١٢٣", finish="stop"), _chunk(usage=USAGE))
    outcome, _preview = _submit(lab, feather.Scripted([_stream_response(body)]))
    assert outcome.state == job_store.COMPLETED
    run = outcome.run
    assert run.pages[0].text == "نص الصفحة ١٢٣"
    assert run.usage.cost_usd == pytest.approx(0.0308) and run.usage.cost_source == "response"
    artifact = Path(outcome.artifact_directory)
    assert artifact.relative_to(store.root).parts[:2] == ("pages", "oxen")
    checked = page_artifacts.verify_snapshot(artifact)
    assert checked["ok"] and checked["reusable"]
    assert (artifact / "raw_text.txt").read_text(encoding="utf-8") == "نص الصفحة ١٢٣"
    assert OXEN_TOKEN not in _stored_text(store.root)


def test_stream_that_ends_early_is_a_terminal_failure_not_a_retry(lab) -> None:  # noqa: F811
    _runner, store, *_ = lab
    body = _sse(_chunk("نص جزئي"), done=False)
    outcome, _ = _submit(lab, feather.Scripted([_stream_response(body)]))
    assert outcome.state == job_store.FAILED
    assert "stream_incomplete" in (outcome.run.error or "")
    # Billed conservatively at the compute-time bound, never treated as free.
    assert outcome.run.usage.cost_source == "estimated" and outcome.run.usage.cost_usd == pytest.approx(0.252)
    assert page_artifacts.verify_snapshot(Path(outcome.artifact_directory))["ok"]
    assert not page_artifacts.verify_snapshot(Path(outcome.artifact_directory))["reusable"]


def test_mid_stream_timeout_stays_unconfirmed_and_records_progress_without_the_key(lab) -> None:  # noqa: F811
    _runner, store, *_ = lab
    partial = _sse(_chunk("بداية النص"), done=False, keepalive=False).decode()
    error = APIRequestError(
        "request_timeout", "Response not complete within 120s after the upload (streaming deadline)",
        http_status=200,
        diagnostics={"stages": {"connection": {"status": "completed"},
                                "sending_request": {"status": "completed"},
                                "waiting_response_headers": {"status": "completed"},
                                "reading_response_body": {"status": "timeout"}},
                     "http_status": 200, "response_bytes": len(partial), "partial_body_text": partial,
                     "request_id_headers": {"X-Request-Id": "ox-req-9"}},
    )
    outcome, _ = _submit(lab, feather.Scripted([error]))
    assert outcome.state == job_store.SUBMISSION_UNCONFIRMED  # never retried automatically
    latest = outcome.journal.latest()
    progress = latest["error"]["diagnostics"]["stream_progress"]
    assert progress["partial_content"] == "بداية النص" and progress["done_marker"] is False
    assert progress["completion_id"] == "chatcmpl-ox-1"
    assert page_artifacts.verify_snapshot(Path(outcome.artifact_directory))["ok"]
    assert OXEN_TOKEN not in _stored_text(store.root)


def test_header_timeout_with_streaming_is_still_unconfirmed(lab) -> None:  # noqa: F811
    error = APIRequestError("request_timeout", "Request timed out after 120s during waiting_response_headers",
                            diagnostics={"stages": {"waiting_response_headers": {"status": "timeout"}}})
    outcome, _ = _submit(lab, feather.Scripted([error]))
    assert outcome.state == job_store.SUBMISSION_UNCONFIRMED
    assert "stream_progress" not in outcome.journal.latest()["error"]["diagnostics"]


def test_402_with_null_error_fields_is_terminal_payment_required(lab) -> None:  # noqa: F811
    # Regression: run_cebd62dc96a30594ca crashed on `None[:300]` and was misrecorded
    # as an unconfirmed submission.
    response = HTTPResponse(402, {"Content-Type": "application/json"},
                            json.dumps({"error": None, "status_message": None}).encode(), 3.0)
    outcome, _ = _submit(lab, feather.Scripted([response]))
    assert outcome.state == job_store.FAILED and "payment_required" in (outcome.run.error or "")


def test_streamed_error_status_still_parses_the_json_error(lab) -> None:  # noqa: F811
    response = HTTPResponse(402, {"Content-Type": "application/json"},
                            json.dumps({"error": {"message": "insufficient_credits"}}).encode(), 3.0)
    outcome, _ = _submit(lab, feather.Scripted([response]))
    assert outcome.state == job_store.FAILED
    assert "insufficient_credits" in (outcome.run.error or "")


def test_socket_timeout_in_header_wait_still_reports_the_stage() -> None:
    clock = _Clock()

    class _Stalled(_Connection):
        def getresponse(self):
            raise socket.timeout("mock")

    with pytest.raises(APIRequestError) as raised:
        _post(_transport(_Stalled(None), clock), response_deadline_s=120)
    assert "waiting_response_headers" in str(raised.value)
    assert raised.value.diagnostics["stages"]["waiting_response_headers"]["status"] == "timeout"
    assert "partial_body_text" not in raised.value.diagnostics
