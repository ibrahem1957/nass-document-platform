"""Qari service (recipe, HTTP layer with a fake engine) and client (classified errors, no fallback,
no automatic resend). No model is loaded and no real network is used."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import io
import json
import threading
from pathlib import Path

import httpx
import pytest
from PIL import Image

from lab.core.auto import qari_client

SERVICE_TOKEN = "svc-test-token-7f3a9c"
SERVICE_PATH = Path(__file__).resolve().parents[2] / "runpod" / "qari_service.py"


def _service():
    spec = importlib.util.spec_from_file_location("qari_service_under_test", SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # no torch/fastapi import at module level
    return module


def _png(size=(40, 20)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeEngine:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def describe(self):
        return {"engine": "fake"}

    def generate(self, image, *, max_time_s, extra):
        self.calls.append({"max_time_s": max_time_s, "extra": dict(extra), "size": image.size})
        text, flags = self.script.pop(0)
        return text, {"tokens": 10, "seconds": 1.0, "loop_stopped": False, "hit_limit": False, "timed_out": False,
                      "params": {"max_time": max_time_s, **extra}} | flags


# ------------------------------------------------------------------ recipe


def test_loop_detector_matches_v3() -> None:
    service = _service()
    assert service.loop_detected([7] * 5)
    assert not service.loop_detected([7] * 4)
    assert service.loop_detected([1, 2, 3] * 5)
    assert not service.loop_detected([1, 2, 3] * 4 + [1, 2])
    assert service.loop_detected(list(range(40)) * 5)
    assert not service.loop_detected(list(range(41)) * 4)


@pytest.mark.parametrize("flag", ["loop_stopped", "hit_limit", "timed_out"])
def test_one_retry_with_repetition_penalty_after_loop_limit_or_timeout(flag) -> None:
    service = _service()
    engine = FakeEngine([("aaaa", {flag: True}), ("clean", {})])
    outcome = service.transcribe(engine, Image.new("RGB", (10, 10)), "crop")
    assert outcome["final_text"] == "clean" and outcome["final_attempt"] == 2 and outcome["retried"]
    assert [a["text"] for a in outcome["attempts"]] == ["aaaa", "clean"]  # every attempt kept
    assert engine.calls[0]["extra"] == {} and engine.calls[1]["extra"] == {"repetition_penalty": 1.1}
    assert outcome["attempts"][1]["retry_reason"] == flag


def test_no_retry_when_clean_and_time_limits_per_kind() -> None:
    service = _service()
    engine = FakeEngine([("fine", {}), ("page", {})])
    crop = service.transcribe(engine, Image.new("RGB", (10, 10)), "crop")
    page = service.transcribe(engine, Image.new("RGB", (10, 10)), "page")
    assert not crop["retried"] and len(crop["attempts"]) == 1
    assert [c["max_time_s"] for c in engine.calls] == [30.0, 60.0]
    assert crop["settings"]["max_new_tokens"] == 1536 and crop["settings"]["prompt"] == "Free OCR."


def test_only_one_retry_even_if_the_retry_loops_again() -> None:
    service = _service()
    engine = FakeEngine([("x", {"loop_stopped": True}), ("y", {"loop_stopped": True})])
    outcome = service.transcribe(engine, Image.new("RGB", (10, 10)), "crop")
    assert len(engine.calls) == 2 and outcome["final_flags"]["loop_stopped"]


def test_proven_overrides_need_permission() -> None:
    service = _service()
    with pytest.raises(RuntimeError):
        service.proven_from_env({"QARI_PROVEN_OVERRIDES": '{"max_new_tokens": 10}'})
    proven, overrides = service.proven_from_env({"QARI_PROVEN_OVERRIDES": '{"max_new_tokens": 10}',
                                                 "QARI_ALLOW_PROVEN_OVERRIDES": "true"})
    assert proven.max_new_tokens == 10 and overrides == {"max_new_tokens": 10}


# ------------------------------------------------------------------ HTTP layer


def _app(engine, **env):
    pytest.importorskip("fastapi")
    service = _service()
    return service.create_app(engine, environ={"QARI_SERVICE_TOKEN": SERVICE_TOKEN, **env})


def _body(image: bytes, request_id="r1", kind="crop", sha=None):
    return {"request_id": request_id, "kind": kind, "image_png_b64": base64.b64encode(image).decode(),
            "image_sha256": sha or hashlib.sha256(image).hexdigest()}


def test_service_requires_a_token_to_start() -> None:
    pytest.importorskip("fastapi")
    with pytest.raises(RuntimeError, match="QARI_SERVICE_TOKEN"):
        _service().create_app(FakeEngine([]), environ={})


def test_service_health_auth_validation_and_replay() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    engine = FakeEngine([("text 1", {})])
    with TestClient(_app(engine)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ready" and SERVICE_TOKEN not in json.dumps(health)
        assert health["proven"]["max_new_tokens"] == 1536 and health["concurrency"] == 1
        image = _png()
        assert client.post("/v1/ocr", json=_body(image)).status_code == 401
        assert client.post("/v1/ocr", json=_body(image), headers={"Authorization": "Bearer nope"}).status_code == 401
        auth = {"Authorization": f"Bearer {SERVICE_TOKEN}"}
        assert client.post("/v1/ocr", json=_body(image, sha="0" * 64), headers=auth).status_code == 400
        response = client.post("/v1/ocr", json=_body(image), headers=auth).json()
        assert response["final_text"] == "text 1" and response["image"]["resized"] is False
        assert response["image"]["width"] == 40 and len(response["attempts"]) == 1
        replay = client.post("/v1/ocr", json=_body(image), headers=auth).json()
        assert replay["replayed"] and len(engine.calls) == 1  # the same request id never runs twice
        assert client.get("/v1/results/r1", headers=auth).json()["final_text"] == "text 1"
        assert client.get("/v1/results/unknown", headers=auth).status_code == 404


def test_service_error_is_reported_not_guessed() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class Broken(FakeEngine):
        def generate(self, image, *, max_time_s, extra):
            raise RuntimeError("CUDA out of memory")

    with TestClient(_app(Broken([]))) as client:
        response = client.post("/v1/ocr", json=_body(_png()), headers={"Authorization": f"Bearer {SERVICE_TOKEN}"})
        assert response.status_code == 500 and "out of memory" in response.json()["detail"]


def test_bounded_queue_returns_429() -> None:
    pytest.importorskip("fastapi")
    release = threading.Event()

    class Slow(FakeEngine):
        def generate(self, image, *, max_time_s, extra):
            release.wait(5)
            return "slow", {"tokens": 1, "seconds": 0.1, "loop_stopped": False, "hit_limit": False,
                            "timed_out": False, "params": {}}

    app = _app(Slow([]), QARI_SERVICE_MAX_QUEUE="0", QARI_SERVICE_CONCURRENCY="1")
    image = _png()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            headers = {"Authorization": f"Bearer {SERVICE_TOKEN}"}
            first = asyncio.create_task(client.post("/v1/ocr", json=_body(image, "a"), headers=headers))
            await asyncio.sleep(0.2)
            second = await client.post("/v1/ocr", json=_body(image, "b"), headers=headers)
            release.set()
            return (await first).status_code, second.status_code

    assert asyncio.run(scenario()) == (200, 429)


# ------------------------------------------------------------------ client


def _client(handler, url="http://127.0.0.1:8000", **kwargs):
    return qari_client.QariClient(url, "tok", transport=httpx.MockTransport(handler), **kwargs)


def _ok_response(request):
    body = json.loads(request.content)
    return httpx.Response(200, json={"request_id": body["request_id"], "final_text": "ok",
                                     "image": {"sha256": body["image_sha256"]}, "attempts": [{}]})


def test_client_success_sends_bearer_and_unchanged_image() -> None:
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        seen["image"] = base64.b64decode(json.loads(request.content)["image_png_b64"])
        return _ok_response(request)

    image = _png()
    assert _client(handler).ocr("r1", "crop", image)["final_text"] == "ok"
    assert seen == {"auth": "Bearer tok", "image": image}


@pytest.mark.parametrize("status, error", [(503, qari_client.QariUnavailable), (429, qari_client.QariUnavailable),
                                           (401, qari_client.QariUnavailable), (500, qari_client.QariServiceError),
                                           (400, qari_client.QariServiceError),
                                           (302, qari_client.QariUnavailable)])
def test_client_classifies_http_errors(status, error) -> None:
    client = _client(lambda request: httpx.Response(status, json={"detail": "x"}, headers={"Location": "http://x"}))
    with pytest.raises(error):
        client.ocr("r1", "crop", _png())


def test_connect_errors_are_retried_only_before_sending() -> None:
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(qari_client.QariUnavailable):
        _client(handler, connect_retries=2).ocr("r1", "crop", _png())
    assert len(calls) == 3


def test_no_response_after_sending_is_unconfirmed_and_never_resent() -> None:
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(qari_client.QariUnconfirmed):
        _client(handler, connect_retries=5).ocr("r1", "crop", _png())
    assert len(calls) == 1


def test_mismatched_response_is_a_service_error() -> None:
    def handler(request):
        return httpx.Response(200, json={"request_id": "other", "final_text": "x", "image": {"sha256": "0"}})

    with pytest.raises(qari_client.QariServiceError):
        _client(handler).ocr("r1", "crop", _png())


def test_endpoint_rules_and_no_hardcoded_host() -> None:
    with pytest.raises(qari_client.QariUnavailable, match="not configured"):
        qari_client.endpoint_info(None)
    with pytest.raises(qari_client.QariUnavailable):
        qari_client.endpoint_info("https://user:pw@127.0.0.1:8000")
    source = (Path(qari_client.__file__)).read_text(encoding="utf-8")
    assert "runpod.net" not in source and "proxy.runpod" not in source


def test_local_mode_accepts_only_loopback() -> None:
    info = qari_client.endpoint_info("http://127.0.0.1:8000", deployment_mode="local")
    assert info["loopback"] and info["deployment_mode"] == "local" and not info["external"]
    assert qari_client.endpoint_info("http://localhost:8000")["loopback"]  # local is the default
    for url in ("http://10.0.0.5:8000", "https://qari.example.com"):
        with pytest.raises(qari_client.QariUntrusted, match="loopback"):
            qari_client.endpoint_info(url, deployment_mode="local")


def test_internal_mode_accepts_internal_servers_but_not_public_ips() -> None:
    for url in ("http://10.0.0.5:8000", "https://qari.gpu.internal:8443", "http://192.168.1.20:8000"):
        info = qari_client.endpoint_info(url, deployment_mode="internal")
        assert info["deployment_mode"] == "internal" and not info["external"]
    with pytest.raises(qari_client.QariUntrusted, match="public IP"):
        qari_client.endpoint_info("http://8.8.8.8:8000", deployment_mode="internal")


def test_development_external_needs_https_and_an_explicitly_trusted_host() -> None:
    hosts = ("pod123-8000.proxy.example.net",)
    info = qari_client.endpoint_info("https://POD123-8000.proxy.example.net", deployment_mode="development_external",
                                     trusted_development_hosts=hosts)
    assert info["external"] and "QARI_TRUSTED_DEV_HOSTS" in info["trust_basis"]
    with pytest.raises(qari_client.QariUntrusted, match="https"):
        qari_client.endpoint_info("http://pod123-8000.proxy.example.net", deployment_mode="development_external",
                                  trusted_development_hosts=hosts)
    for url in ("https://evil.example.com", "https://pod123-8000.proxy.example.net.evil.com"):
        with pytest.raises(qari_client.QariUntrusted, match="not in QARI_TRUSTED_DEV_HOSTS"):
            qari_client.endpoint_info(url, deployment_mode="development_external", trusted_development_hosts=hosts)
    with pytest.raises(qari_client.QariUntrusted, match="unknown"):
        qari_client.endpoint_info("http://127.0.0.1:8000", deployment_mode="anything")


def test_untrusted_endpoint_is_refused_before_any_request() -> None:
    calls = []
    with pytest.raises(qari_client.QariUntrusted):
        qari_client.QariClient("https://qari.example.com", "tok", deployment_mode="local",
                               transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200)))
    assert calls == [] and issubclass(qari_client.QariUntrusted, qari_client.QariUnavailable)
    source = (Path(qari_client.__file__)).read_text(encoding="utf-8")
    assert "runpod.net" not in source and "proxy.runpod" not in source


def test_recover_asks_without_sending_the_image() -> None:
    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/results/r1"
        return httpx.Response(200, json={"request_id": "r1", "final_text": "late"})

    assert _client(handler).recover("r1")["final_text"] == "late"
    assert _client(lambda request: httpx.Response(404)).recover("r1") is None


def test_no_other_model_is_reachable_from_the_automatic_pipeline() -> None:
    folder = Path(qari_client.__file__).parent
    text = "\n".join(path.read_text(encoding="utf-8") for path in folder.glob("*.py"))
    for forbidden in ("baidu", "oxen", "featherless", "APIEvaluationRunner", "full_pdf"):
        assert forbidden.lower() not in text.lower(), forbidden


# ------------------------------------------------------------------ in-generation recipe (fake torch/transformers)


class _Row:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _Tensor:
    def __init__(self, rows):
        self.rows = rows

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, key):
        r, c = key
        if isinstance(r, int):
            return _Row(self.rows[r][c])
        return _Tensor([row[c] for row in self.rows[r]])


class _Inputs(dict):
    def to(self, device):
        return self


class _FakeModel:
    """Generates token by token, calling the stopping criteria after each token like transformers."""
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, *, input_ids, max_new_tokens, max_time, do_sample, stopping_criteria, **extra):
        self.calls.append({"max_new_tokens": max_new_tokens, "max_time": max_time, "do_sample": do_sample, **extra})
        stream = [1, 2, 3, 4] if extra.get("repetition_penalty") else [9] + [7] * 1000  # loop unless penalised
        ids = list(input_ids.rows[0])
        for token in stream[:max_new_tokens]:
            ids.append(token)
            if any(criterion(_Tensor([ids]), None) for criterion in stopping_criteria):
                break
        return _Tensor([ids])


class _FakeProcessor:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return _Inputs(input_ids=_Tensor([[100, 101, 102]]))

    def batch_decode(self, new, **kwargs):
        return [" ".join(str(t) for t in new.rows[0])]


def _fake_engine(monkeypatch):
    import contextlib
    import sys
    import types

    transformers = types.ModuleType("transformers")
    transformers.StoppingCriteria = type("StoppingCriteria", (), {})
    transformers.StoppingCriteriaList = list
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
                                  cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    service = _service()
    engine = object.__new__(service.TransformersEngine)
    engine.proven, engine.torch = service.PROVEN, torch
    engine.processor, engine.model = _FakeProcessor(), _FakeModel()
    return service, engine


def test_loop_is_stopped_inside_generation_with_the_v3_parameters(monkeypatch) -> None:
    service, engine = _fake_engine(monkeypatch)
    text, info = engine.generate(Image.new("RGB", (8, 8)), max_time_s=30.0, extra={})
    assert info["loop_stopped"] and info["tokens"] == 6  # "9" then five "7": stopped at the 5th repeat
    assert not info["hit_limit"] and not info["timed_out"]
    assert engine.model.calls[0] == {"max_new_tokens": 1536, "max_time": 30.0, "do_sample": False}
    content = engine.processor.messages[0]["content"]
    assert content[0]["type"] == "image" and content[1] == {"type": "text", "text": "Free OCR."}


def test_antirepeat_retry_through_the_real_engine_code(monkeypatch) -> None:
    service, engine = _fake_engine(monkeypatch)
    outcome = service.transcribe(engine, Image.new("RGB", (8, 8)), "page")
    assert [c.get("repetition_penalty") for c in engine.model.calls] == [None, 1.1]
    assert [c["max_time"] for c in engine.model.calls] == [60.0, 60.0]
    assert outcome["attempts"][0]["loop_stopped"] and outcome["attempts"][0]["text"] == "9 7 7 7 7 7"
    assert outcome["final_text"] == "1 2 3 4" and outcome["final_attempt"] == 2
    assert outcome["attempts"][1]["params"]["repetition_penalty"] == 1.1
