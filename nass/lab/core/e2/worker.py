"""Child-process inference worker. The parent enforces load and per-page timeouts."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from lab.core.e2.airgap import enforce_offline_runtime
from lab.core.e2.backends import InferenceError, open_session
from lab.core.e2.hardware import peak_rss_mb
from lab.core.e2.specs import ModelSpec


def _error(exc: BaseException, default_code: str) -> dict:
    # The traceback is kept: library code can swallow the real cause (torch hides a
    # blocked numpy DLL behind "cannot import name NP_SUPPORTED_MODULES").
    return {
        "code": getattr(exc, "code", default_code),
        "type": type(exc).__name__,
        "message": str(exc)[:2000],
        "traceback": traceback.format_exc()[-8000:],
    }


def run_worker(
    conn,
    spec_data: dict,
    jobs: list[dict],
    prompt: str,
    page_timeout_s: float,
    cache: str | None,
) -> None:
    """Load once, then infer each page. `jobs` hold only page numbers and image paths."""
    # Must happen before optional model libraries are imported by open_session.
    enforce_offline_runtime()
    session = None
    try:
        spec = ModelSpec.model_validate(spec_data)
        started = time.perf_counter()
        try:
            session = open_session(
                spec, page_timeout_s=page_timeout_s, cache=Path(cache) if cache else None
            )
        except Exception as exc:  # report, never crash silently
            conn.send({"phase": "fatal", "error": _error(exc, "load_error")})
            return
        conn.send(
            {
                "phase": "loaded",
                "load_ms": (time.perf_counter() - started) * 1000,
                "meta": session.metadata(),
            }
        )
        for job in jobs:
            conn.send({"phase": "page_start", "page": job["page"]})
            wall = time.perf_counter()
            cpu = time.process_time()
            try:
                output = session.infer(Path(job["image_path"]), prompt)
                conn.send(
                    {
                        "phase": "page",
                        "page": job["page"],
                        "ok": True,
                        "text": output.text,
                        "raw": output.raw,
                        "prompt_tokens": output.prompt_tokens,
                        "completion_tokens": output.completion_tokens,
                        "blocks": output.blocks,
                        "elapsed_ms": (time.perf_counter() - wall) * 1000,
                        "cpu_ms": (time.process_time() - cpu) * 1000,
                        "peak_rss_mb": peak_rss_mb(),
                    }
                )
            except Exception as exc:
                conn.send(
                    {
                        "phase": "page",
                        "page": job["page"],
                        "ok": False,
                        "error": _error(
                            exc, exc.code if isinstance(exc, InferenceError) else "backend_error"
                        ),
                        "elapsed_ms": (time.perf_counter() - wall) * 1000,
                    }
                )
        conn.send({"phase": "done", "peak_rss_mb": peak_rss_mb(), "meta": session.metadata()})
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        conn.close()
