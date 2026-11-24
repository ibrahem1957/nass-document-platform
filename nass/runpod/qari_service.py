"""Qari OCR service (FastAPI): the proven `run_qari_v3.py antirepeat` recipe behind an HTTP API.

Same model, prompt and decoding as the benchmark:
  base  Qwen/Qwen3-VL-4B-Instruct + adapter NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct (PEFT)
  prompt "Free OCR.", greedy (do_sample=False), max_new_tokens 1536,
  max_time 30 s for crops / 60 s for full pages,
  loop stopper: the newest tokens are one n-gram (n = 1..40) repeated 5+ times,
  after a loop, a timeout or the token limit: ONE retry with repetition_penalty=1.1,
  every attempt is returned (text + tokens, seconds, loop_stopped, hit_limit, timed_out, params).
The image is used exactly as received (no resizing or re-encoding by this service).

Endpoints
  GET  /health                     no secrets; model/device/queue state (no auth needed)
  POST /v1/ocr                     Bearer token; JSON {request_id, kind: crop|page, image_png_b64, image_sha256}
  GET  /v1/results/{request_id}    Bearer token; a completed result kept in memory (lost-response recovery)

Operational settings (environment):
  QARI_SERVICE_TOKEN         required bearer token (the service refuses to start without one unless
                             QARI_SERVICE_ALLOW_NO_TOKEN=true, which is meant for loopback-only tests)
  QARI_DEVICE_MAP            transformers device_map (default "auto"); QARI_DTYPE float16|bfloat16|float32
  QARI_SERVICE_CONCURRENCY   simultaneous generations (default 1)
  QARI_SERVICE_MAX_QUEUE     waiting requests beyond that before 429 (default 8)
  QARI_SERVICE_MAX_IMAGE_MB  request image limit (default 40)
  QARI_SERVICE_RESULT_CACHE  completed results kept for /v1/results (default 256)
  QARI_MODEL_CACHE_DIR       optional Hugging Face cache directory (weights downloaded beforehand)
Proven settings are fixed in PROVEN below. Changing them requires QARI_ALLOW_PROVEN_OVERRIDES=true
and is reported in every response and in /health.

Run (GPU host):  uvicorn qari_service:app_from_environment --factory --host 0.0.0.0 --port 8000 --workers 1
Requirements:    torch, transformers (Qwen3-VL support), peft, accelerate, pillow, fastapi, uvicorn
"""

# No `from __future__ import annotations`: FastAPI must resolve the request model defined inside
# create_app() from real annotation objects.
import asyncio
import base64
import binascii
import contextlib
import hashlib
import hmac
import io
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

SERVICE_VERSION = "1"
RECIPE = "run_qari_v3.py antirepeat"


@dataclass(frozen=True)
class Proven:
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    adapter: str = "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"
    prompt: str = "Free OCR."
    max_new_tokens: int = 1536
    max_time_s: dict = field(default_factory=lambda: {"crop": 30.0, "page": 60.0})
    loop_reps: int = 5
    loop_max_n: int = 40
    retry: dict = field(default_factory=lambda: {"repetition_penalty": 1.1})
    do_sample: bool = False


PROVEN = Proven()
KINDS = ("crop", "page")


def proven_from_env(environ: dict[str, str] | None = None) -> tuple[Proven, dict[str, Any]]:
    """The proven settings, with overrides from QARI_PROVEN_OVERRIDES (JSON) only when explicitly allowed."""
    environ = os.environ if environ is None else environ
    raw = environ.get("QARI_PROVEN_OVERRIDES", "").strip()
    if not raw:
        return PROVEN, {}
    if environ.get("QARI_ALLOW_PROVEN_OVERRIDES", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError("QARI_PROVEN_OVERRIDES is set but QARI_ALLOW_PROVEN_OVERRIDES is not true")
    overrides = json.loads(raw)
    unknown = set(overrides) - set(asdict(PROVEN))
    if unknown:
        raise RuntimeError(f"unknown proven setting(s): {sorted(unknown)}")
    return Proven(**(asdict(PROVEN) | overrides)), overrides


# ------------------------------------------------------------------ recipe (no torch needed)


def loop_detected(ids: list[int], reps: int = PROVEN.loop_reps, max_n: int = PROVEN.loop_max_n) -> bool:
    """True when the newest ids are the same n-gram (n = 1..max_n) repeated `reps` times (v3 rule)."""
    for n in range(1, max_n + 1):
        span = n * reps
        if len(ids) < span:
            break
        tail = ids[-span:]
        unit = tail[-n:]
        if all(tail[i:i + n] == unit for i in range(0, span, n)):
            return True
    return False


class Engine(Protocol):
    def generate(self, image: Any, *, max_time_s: float, extra: dict) -> tuple[str, dict]: ...

    def describe(self) -> dict: ...


def transcribe(engine: Engine, image: Any, kind: str, proven: Proven = PROVEN) -> dict:
    """The antirepeat recipe: attempt 1; if it loop-stopped, hit the token limit or timed out, one
    retry with repetition_penalty; the final text is the last attempt. Every attempt is returned."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    max_time = float(proven.max_time_s[kind])
    attempts = []
    text, info = engine.generate(image, max_time_s=max_time, extra={})
    attempts.append({"attempt": 1, "text": text, **info})
    retried = bool(info.get("loop_stopped") or info.get("hit_limit") or info.get("timed_out"))
    if retried:
        text2, info2 = engine.generate(image, max_time_s=max_time, extra=dict(proven.retry))
        attempts.append({"attempt": 2, "text": text2, **info2,
                         "retry_reason": ", ".join(name for name in ("loop_stopped", "hit_limit", "timed_out")
                                                   if info.get(name))})
    final = attempts[-1]
    return {"final_text": final["text"], "final_attempt": final["attempt"], "retried": retried,
            "attempts": attempts,
            "final_flags": {name: bool(final.get(name)) for name in ("loop_stopped", "hit_limit", "timed_out")},
            "settings": {"kind": kind, "max_time_s": max_time, "max_new_tokens": proven.max_new_tokens,
                         "prompt": proven.prompt, "do_sample": proven.do_sample, "retry": dict(proven.retry),
                         "loop_stopper": {"max_n": proven.loop_max_n, "reps": proven.loop_reps}}}


class TransformersEngine:
    """Loads the model ONCE (process start) and generates exactly like run_qari_v3.py."""

    def __init__(self, proven: Proven = PROVEN, *, device_map: str = "auto", dtype: str = "float16",
                 cache_dir: str | None = None) -> None:
        import torch  # noqa: PLC0415 - GPU host only
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: PLC0415

        self.proven, self.torch = proven, torch
        self.processor = AutoProcessor.from_pretrained(proven.adapter, cache_dir=cache_dir)
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            proven.base_model, torch_dtype=getattr(torch, dtype), device_map=device_map, cache_dir=cache_dir)
        self.model = PeftModel.from_pretrained(base, proven.adapter, cache_dir=cache_dir)
        self.model.eval()
        self.dtype, self.device_map = dtype, device_map

    def describe(self) -> dict:
        cuda = self.torch.cuda.is_available()
        return {"engine": "transformers", "device": str(self.model.device), "device_map": self.device_map,
                "dtype": self.dtype, "cuda": cuda,
                "gpu": self.torch.cuda.get_device_name(0) if cuda else None}

    def generate(self, image: Any, *, max_time_s: float, extra: dict) -> tuple[str, dict]:
        from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: PLC0415

        torch, proven = self.torch, self.proven

        class LoopStopper(StoppingCriteria):
            def __init__(self, prompt_len: int) -> None:
                self.prompt_len, self.fired = prompt_len, False

            def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ARG002
                if loop_detected(input_ids[0, self.prompt_len:].tolist(), proven.loop_reps, proven.loop_max_n):
                    self.fired = True
                    return True
                return False

        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": proven.prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        stopper = LoopStopper(inputs["input_ids"].shape[1])
        params = dict(max_new_tokens=proven.max_new_tokens, max_time=max_time_s, do_sample=proven.do_sample, **extra)
        started = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **params, stopping_criteria=StoppingCriteriaList([stopper]))
        new = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        seconds = round(time.time() - started, 2)
        tokens = int(new.shape[1])
        info = {"tokens": tokens, "seconds": seconds, "loop_stopped": stopper.fired,
                "hit_limit": tokens >= proven.max_new_tokens,
                "timed_out": (not stopper.fired) and seconds >= max_time_s - 0.5, "params": dict(params)}
        del inputs, out, new
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text, info


# ------------------------------------------------------------------ HTTP layer


class _Results:
    """Completed results by request_id (bounded, in memory) for lost-response recovery."""

    def __init__(self, capacity: int) -> None:
        self.capacity, self.items, self.lock = capacity, OrderedDict(), threading.Lock()

    def put(self, request_id: str, value: dict) -> None:
        with self.lock:
            self.items[request_id] = value
            self.items.move_to_end(request_id)
            while len(self.items) > self.capacity:
                self.items.popitem(last=False)

    def get(self, request_id: str) -> dict | None:
        with self.lock:
            return self.items.get(request_id)


def _int_env(environ: dict, name: str, default: int) -> int:
    try:
        return int(environ.get(name, default))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def create_app(engine: Engine | None = None, *, engine_factory: Callable[[], Engine] | None = None,
               environ: dict[str, str] | None = None):
    """Build the FastAPI app. Tests pass a fake `engine`; production loads TransformersEngine at startup."""
    from fastapi import FastAPI, Header, HTTPException  # noqa: PLC0415
    from fastapi.concurrency import run_in_threadpool  # noqa: PLC0415
    from pydantic import BaseModel, Field  # noqa: PLC0415

    environ = dict(os.environ if environ is None else environ)
    proven, overrides = proven_from_env(environ)
    token = environ.get("QARI_SERVICE_TOKEN", "")
    allow_no_token = environ.get("QARI_SERVICE_ALLOW_NO_TOKEN", "").lower() in {"1", "true", "yes"}
    if not token and not allow_no_token:
        raise RuntimeError("QARI_SERVICE_TOKEN is not set (set QARI_SERVICE_ALLOW_NO_TOKEN=true only for "
                           "loopback-only tests)")
    concurrency = max(1, _int_env(environ, "QARI_SERVICE_CONCURRENCY", 1))
    max_queue = max(0, _int_env(environ, "QARI_SERVICE_MAX_QUEUE", 8))
    max_image_bytes = max(1, _int_env(environ, "QARI_SERVICE_MAX_IMAGE_MB", 40)) * 1024 * 1024
    results = _Results(max(1, _int_env(environ, "QARI_SERVICE_RESULT_CACHE", 256)))

    state: dict[str, Any] = {"engine": engine, "load_error": None, "loaded_at": None, "running": 0, "waiting": 0,
                             "served": 0, "failed": 0}
    semaphore = asyncio.Semaphore(concurrency)
    counter_lock = asyncio.Lock()

    def _factory() -> Engine:
        return TransformersEngine(proven, device_map=environ.get("QARI_DEVICE_MAP", "auto"),
                                  dtype=environ.get("QARI_DTYPE", "float16"),
                                  cache_dir=environ.get("QARI_MODEL_CACHE_DIR") or None)

    @contextlib.asynccontextmanager
    async def lifespan(_app):  # the model is loaded once per process, at startup
        if state["engine"] is None:
            try:
                state["engine"] = await run_in_threadpool(engine_factory or _factory)
            except Exception as exc:  # noqa: BLE001 - reported by /health; requests get 503
                state["load_error"] = f"{type(exc).__name__}: {exc}"
        state["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        yield

    app = FastAPI(title="Qari OCR service", version=SERVICE_VERSION, lifespan=lifespan)

    def _authorize(authorization: str | None) -> None:
        if not token:
            return
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    class OCRRequest(BaseModel):
        request_id: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:-]+$")
        kind: str = Field(pattern=r"^(crop|page)$")
        image_png_b64: str
        image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def _describe() -> dict:
        engine_now = state["engine"]
        return {"status": "ready" if engine_now is not None else ("error" if state["load_error"] else "loading"),
                "service_version": SERVICE_VERSION, "recipe": RECIPE,
                "model": {"base": proven.base_model, "adapter": proven.adapter, "prompt": proven.prompt},
                "proven": asdict(proven), "proven_overrides": overrides,
                "engine": engine_now.describe() if engine_now is not None else None,
                "load_error": state["load_error"], "loaded_at": state["loaded_at"],
                "concurrency": concurrency, "max_queue": max_queue, "running": state["running"],
                "waiting": state["waiting"], "served": state["served"], "failed": state["failed"],
                "auth": "bearer token required" if token else "no token (loopback test mode)"}

    @app.get("/health")
    async def health() -> dict:
        return _describe()

    @app.get("/v1/results/{request_id}")
    async def result(request_id: str, authorization: str | None = Header(default=None)) -> dict:
        _authorize(authorization)
        found = results.get(request_id)
        if found is None:
            raise HTTPException(status_code=404, detail="no completed result for this request_id")
        return found

    @app.post("/v1/ocr")
    async def ocr(request: OCRRequest, authorization: str | None = Header(default=None)) -> dict:
        _authorize(authorization)
        if state["engine"] is None:
            raise HTTPException(status_code=503, detail=f"model not loaded ({state['load_error'] or 'loading'})")
        cached = results.get(request.request_id)
        if cached is not None:  # the same request_id is never run twice
            return cached | {"replayed": True}
        try:
            data = base64.b64decode(request.image_png_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(status_code=400, detail="image_png_b64 is not valid base64") from exc
        if len(data) > max_image_bytes:
            raise HTTPException(status_code=413, detail=f"image exceeds {max_image_bytes // (1024 * 1024)} MB")
        if hashlib.sha256(data).hexdigest() != request.image_sha256:
            raise HTTPException(status_code=400, detail="image_sha256 does not match the image bytes")
        from PIL import Image  # noqa: PLC0415

        try:
            with Image.open(io.BytesIO(data)) as source:
                image_format = source.format
                image = source.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"unreadable image: {type(exc).__name__}") from exc
        async with counter_lock:
            if state["waiting"] + state["running"] >= concurrency + max_queue:
                raise HTTPException(status_code=429, detail="queue full; try again later",
                                    headers={"Retry-After": "30"})
            state["waiting"] += 1
        async with semaphore:
            async with counter_lock:
                state["waiting"] -= 1
                state["running"] += 1
            started = time.time()
            try:
                outcome = await run_in_threadpool(transcribe, state["engine"], image, request.kind, proven)
            except Exception as exc:  # noqa: BLE001 - reported to the client; nothing guessed
                state["failed"] += 1
                raise HTTPException(status_code=500,
                                    detail=f"generation failed: {type(exc).__name__}: {exc}") from exc
            finally:
                async with counter_lock:
                    state["running"] -= 1
        payload = {"request_id": request.request_id, "kind": request.kind, "service_version": SERVICE_VERSION,
                   "recipe": RECIPE, "model": {"base": proven.base_model, "adapter": proven.adapter},
                   "proven_overrides": overrides,
                   "image": {"sha256": request.image_sha256, "bytes": len(data), "format": image_format,
                             "width": image.width, "height": image.height, "resized": False},
                   "elapsed_s": round(time.time() - started, 2), **outcome}
        state["served"] += 1
        results.put(request.request_id, payload)
        return payload

    return app


def app_from_environment():
    """uvicorn factory: `uvicorn qari_service:app_from_environment --factory --host 0.0.0.0 --port 8000`."""
    try:
        return create_app()
    except RuntimeError as exc:  # configuration problem: fail loudly at startup, never silently
        raise SystemExit(f"qari_service: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover - convenience launcher
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app_from_environment(), host=os.environ.get("QARI_SERVICE_HOST", "127.0.0.1"),
                port=int(os.environ.get("QARI_SERVICE_PORT", "8000")), workers=1)
