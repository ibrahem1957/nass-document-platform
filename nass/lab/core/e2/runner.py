"""E2-A inference runner: render -> isolated worker -> immutable RAW RunResult.

The runner's API deliberately has no ground-truth or extracted-text parameter:
inference inputs are the rendered page image and a static prompt only.
"""

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lab import __version__
from lab.core.e2 import REMOTE_INFERENCE_ENABLED, appcontrol
from lab.core.e2.airgap import load_manifest, preflight_deployment
from lab.core.e2.backends import Readiness, check_readiness, hf_snapshot_dir
from lab.core.e2.hardware import HardwareProfile, probe_hardware
from lab.core.e2.prompts import get_prompt
from lab.core.e2.render import render_pdf_page
from lab.core.e2.specs import ModelSpec
from lab.core.e2.worker import run_worker
from lab.core.models import AdapterInfo, LabDocument, PageResult, RunResult, TextBlock
from lab.core.settings import LabSettings
from lab.core.workspace import WorkspaceManager

E2_IMPLEMENTATION_VERSION = "3"
INPUT_POLICY = (
    "rendered page image only; element backends receive a static prompt; "
    "the official pipeline selects internal per-region prompts; no ground truth or extracted text"
)
PREFLIGHT_MAX_AGE_SECONDS = 300


def new_run_id() -> str:
    return f"run_{os.urandom(9).hex()}"


class E2RunRefused(RuntimeError):
    def __init__(self, readiness: Readiness | None, message: str):
        super().__init__(message)
        self.readiness = readiness


def adapter_info_for(spec: ModelSpec, backend_version: str | None = None) -> AdapterInfo:
    prompt = get_prompt(spec.prompt_id)
    prompt_used = spec.pipeline_mode != "official_full_page"
    return AdapterInfo(
        adapter_id=f"e2.{spec.id}",
        task="ocr",
        label=spec.label,
        kind="local",
        library=spec.backend,
        library_version=backend_version,
        license=spec.license,
        model_id=spec.pinned_ref,
        params={
            "prompt_id": prompt.prompt_id if prompt_used else None,
            "prompt_sha256": prompt.sha256 if prompt_used else None,
            "registry_prompt_id": prompt.prompt_id,
            "prompt_used_by_backend": prompt_used,
            "generation": dict(spec.generation),
            "weights_kind": spec.weights_kind,
            "pipeline_mode": spec.pipeline_mode,
            "pipeline_version": spec.pipeline_version,
            "pipeline_engine": spec.pipeline_engine,
            "layout_model": (
                f"{spec.layout_repo_id}@{spec.layout_revision}" if spec.layout_repo_id else None
            ),
            "base_model": (
                f"{spec.base_repo_id}@{spec.base_revision}" if spec.base_repo_id else None
            ),
        },
        available=True,
        unavailable_reason=None,
        supports_blocks=spec.pipeline_mode == "official_full_page",
        implementation_version=E2_IMPLEMENTATION_VERSION,
    )


def is_mock_run(run: RunResult) -> bool:
    return bool((run.environment.get("e2") or {}).get("is_mock"))


def is_e2_run(run: RunResult) -> bool:
    return run.adapter.adapter_id.startswith("e2.")


def save_run(manager: WorkspaceManager, workspace_id: str, run: RunResult) -> Path:
    path = manager.path(workspace_id) / "runs" / f"{run.run_id}.json"
    WorkspaceManager.atomic_write_json(path, run.model_dump(mode="json"))
    return path


class E2Runner:
    def __init__(
        self,
        manager: WorkspaceManager,
        settings: LabSettings,
        *,
        hardware: HardwareProfile | None = None,
        cache: Path | None = None,
    ):
        self.manager = manager
        self.settings = settings
        self.hardware = hardware or probe_hardware()
        self.cache = cache

    def readiness(self, spec: ModelSpec) -> Readiness:
        return check_readiness(spec, self.hardware, cache=self.cache)

    def verified_preflight(self, spec: ModelSpec, report: dict | None = None) -> dict | None:
        """Compute or validate fresh file-backed evidence before any real inference."""
        if spec.is_mock:
            return None
        manifest = load_manifest()
        current = report or preflight_deployment(
            manifest, spec.id, environment="local", hardware=self.hardware
        )
        checked_at = current.get("checked_at")
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(checked_at)).total_seconds()
        except (TypeError, ValueError):
            age = PREFLIGHT_MAX_AGE_SECONDS + 1
        allowed_warnings = all(
            warning.startswith("wheel_hashes_unverified:")
            for warning in current.get("warnings", [])
        )
        components_verified = bool(current.get("components")) and all(
            item.get("status") == "HASH_VERIFIED" and not item.get("errors")
            for item in current["components"]
        )
        valid = (
            current.get("deployment_id") == spec.id
            and current.get("manifest_id") == manifest["manifest_id"]
            and current.get("manifest_sha256") == manifest["_manifest_sha256"]
            and 0 <= age <= PREFLIGHT_MAX_AGE_SECONDS
            and current.get("status")
            in {"DEPENDENCIES_VERIFIED", "OFFLINE_TESTED", "READY_FOR_AIRGAP"}
            and not current.get("errors")
            and allowed_warnings
            and components_verified
            and bool((current.get("runtime") or {}).get("ok"))
            and bool((current.get("hardware") or {}).get("ok"))
        )
        if not valid:
            raise E2RunRefused(
                self.readiness(spec),
                f"{spec.id} refused: fresh artifact/dependency preflight did not pass "
                f"(status={current.get('status')}, errors={current.get('errors')}, "
                f"warnings={current.get('warnings')}"
                + (
                    f", declared blockers={current.get('missing_or_unverified')}"
                    if current.get("status") == "BLOCKED"
                    else ""
                )
                + ")",
            )
        mismatched = self._unverified_load_directories(spec, manifest, current)
        if mismatched:
            raise E2RunRefused(
                self.readiness(spec),
                f"{spec.id} refused: the model would load files the preflight did not hash "
                f"({mismatched}). Point HF_HOME/the cache at the verified snapshot roots.",
            )
        return current

    def _unverified_load_directories(
        self, spec: ModelSpec, manifest: dict, report: dict
    ) -> list[str]:
        """Directories the session will load that the preflight did not verify."""
        if spec.backend == "ollama":
            return []
        loads = [(spec.repo_id, spec.revision)]
        if spec.base_repo_id and spec.base_revision:
            loads.append((spec.base_repo_id, spec.base_revision))
        if spec.layout_repo_id and spec.layout_revision:
            loads.append((spec.layout_repo_id, spec.layout_revision))
        repo_by_component = {
            item["component_id"]: (item["repo_id"], item["revision"])
            for item in manifest["components"]
        }
        verified = {
            repo_by_component.get(item.get("component_id")): Path(item.get("root", "")).resolve()
            for item in report.get("components", [])
            if item.get("status") == "HASH_VERIFIED"
        }
        problems = []
        for repo, revision in loads:
            directory = hf_snapshot_dir(repo, revision, self.cache).resolve()
            if verified.get((repo, revision)) != directory:
                problems.append(f"{repo}@{revision[:12]} -> {directory}")
        return problems

    def run(
        self,
        workspace_id: str,
        document: LabDocument,
        pdf_path: Path,
        spec: ModelSpec,
        pages: list[int],
        *,
        allow_mock: bool = False,
        dpi: int | None = None,
        progress: Callable[[int, int], None] | None = None,
        run_id: str | None = None,
        on_page: Callable[[int, dict], None] | None = None,
        preflight_report: dict | None = None,
    ) -> RunResult:
        if spec.backend == "remote":
            # E2-A has no remote client; REMOTE_INFERENCE_ENABLED is a hard False.
            raise E2RunRefused(
                None, f"Remote inference is disabled in E2-A (switch={REMOTE_INFERENCE_ENABLED})"
            )
        if spec.is_mock and not allow_mock:
            raise E2RunRefused(None, "Mock models run only in explicit architecture tests")
        if document.kind != "pdf":
            raise E2RunRefused(None, "E2-A renders PDF pages only")
        readiness = self.readiness(spec)
        if not readiness.runnable:
            raise E2RunRefused(readiness, f"{spec.id} is not ready: {readiness.status}")
        verified_preflight = self.verified_preflight(spec, preflight_report)
        prompt = get_prompt(spec.prompt_id)
        image_dir = self.manager.path(workspace_id) / "e2_images" / document.doc_id
        started_at = datetime.now(timezone.utc)
        rendered = [
            render_pdf_page(pdf_path, page, image_dir, dpi=dpi or self.settings.e2_render_dpi)
            for page in pages
        ]
        jobs = [{"page": item.page, "image_path": item.path} for item in rendered]
        run_clock = time.monotonic()
        outcome = self._execute(spec, jobs, prompt.text, progress, on_page)
        if outcome["fatal"] or outcome["errors"]:
            self._attach_application_control_evidence(outcome, time.monotonic() - run_clock)
        meta = outcome["meta"]
        page_results = []
        for item in rendered:
            record = outcome["pages"].get(item.page)
            if record is None:
                page_results.append(
                    PageResult(
                        page=item.page,
                        text="",
                        blocks=None,
                        status="error",
                        error=outcome.get("skipped_reason") or "not run",
                        elapsed_ms=None,
                    )
                )
            elif record.get("ok"):
                text = record["text"]  # exact model response, never altered
                blocks = [
                    TextBlock(
                        order=int(block["order"]),
                        text=str(block.get("text") or ""),
                        bbox=tuple(block["bbox"]) if block.get("bbox") else None,
                    )
                    for block in (record.get("blocks") or [])
                ]
                page_results.append(
                    PageResult(
                        page=item.page,
                        text=text,
                        blocks=blocks or None,
                        status="ok" if text.strip() else "empty",
                        error=None,
                        elapsed_ms=record.get("elapsed_ms"),
                        extraction_cpu_ms=record.get("cpu_ms"),
                    )
                )
            else:
                error = record["error"]
                page_results.append(
                    PageResult(
                        page=item.page,
                        text="",
                        blocks=None,
                        status="error",
                        error=f"{error['code']}: {error['message']}",
                        elapsed_ms=record.get("elapsed_ms"),
                    )
                )
        statuses = {page.status for page in page_results}
        if outcome["fatal"]:
            status = "timeout" if outcome["fatal"]["code"] == "load_timeout" else "error"
        elif "error" in statuses:
            status = "partial" if statuses - {"error"} else "error"
        else:
            status = "ok"
        page_wall = [page.elapsed_ms or 0.0 for page in page_results]
        environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "lab_version": __version__,
            "e2": {
                "is_mock": spec.is_mock,
                "input_policy": INPUT_POLICY,
                "pipeline_mode": spec.pipeline_mode,
                "spec": spec.model_dump(mode="json"),
                "prompt": (
                    {
                        "id": prompt.prompt_id,
                        "sha256": prompt.sha256,
                        "text": prompt.text,
                        "used_by_backend": True,
                    }
                    if spec.pipeline_mode != "official_full_page"
                    else {
                        "id": None,
                        "sha256": None,
                        "text": None,
                        "used_by_backend": False,
                        "registry_prompt_id": prompt.prompt_id,
                        "reason": "official pipeline selects per-region prompts internally",
                    }
                ),
                "inference_config": dict(spec.generation),
                "preflight": verified_preflight,
                "readiness": readiness.to_dict(),
                "hardware": {
                    "cpu": self.hardware.cpu,
                    "logical_cpus": self.hardware.logical_cpus,
                    "ram_total_gb": self.hardware.ram_total_gb,
                    "ram_available_gb": self.hardware.ram_available_gb,
                    "gpus": self.hardware.gpus,
                    "cuda_available": self.hardware.cuda_available,
                },
                "images": [item.to_dict() for item in rendered],
                "session": meta,
                "load_ms": outcome.get("load_ms"),
                "peak_rss_mb": outcome.get("peak_rss_mb"),
                "peak_vram_mb": (meta or {}).get("peak_vram_mb"),
                "vram_note": (
                    None if self.hardware.cuda_available else "not measurable: no CUDA device"
                ),
                "page_raw": {str(page): data for page, data in outcome["raw"].items()},
                "errors": outcome["errors"],
                "timeouts": {
                    "load_s": self.settings.e2_load_timeout_s,
                    "page_s": self.settings.e2_page_timeout_s,
                },
            },
        }
        backend_version = (meta or {}).get("transformers") or (meta or {}).get("ollama_version")
        return RunResult(
            run_id=run_id or new_run_id(),
            doc_id=document.doc_id,
            doc_sha256=document.sha256,
            adapter=adapter_info_for(spec, backend_version),
            variant="raw",
            pages=page_results,
            page_mapping="exact",
            started_at=started_at,
            wall_ms=sum(page_wall),
            cpu_ms=sum(page.extraction_cpu_ms or 0 for page in page_results) or None,
            usage=None,
            status=status,
            error=(outcome["fatal"] or {}).get("message"),
            cache_hit=False,
            environment=environment,
            startup_ms=float(outcome.get("load_ms") or 0.0),
        )

    @staticmethod
    def _attach_application_control_evidence(outcome: dict, elapsed_s: float) -> None:
        """Add Windows Code Integrity blocks seen during this run (read-only evidence).

        A blocked native module often surfaces as an unrelated ImportError (torch hides
        a blocked numpy DLL), so the block itself is recorded next to the failure.
        """
        blocks = appcontrol.recent_blocks(elapsed_s + 10, Path(sys.prefix))
        if not blocks:
            return
        files = ", ".join(item["file"] for item in blocks)
        evidence = {
            "code": "application_control_blocked",
            "page": None,
            "message": (
                "Windows application control blocked native module(s) of this environment "
                f"during the run: {files}. The exception above is a consequence; resolve the "
                "block through IT (do not disable security controls)."
            ),
            "application_control_blocks": blocks,
        }
        outcome["errors"].append(evidence)
        if outcome["fatal"]:
            outcome["fatal"] = dict(outcome["fatal"]) | {
                "message": f"{outcome['fatal'].get('message')} [root cause evidence: "
                f"application control blocked {files}]",
                "application_control_blocks": blocks,
            }
        if outcome.get("skipped_reason"):
            outcome["skipped_reason"] += f" (application control blocked {files})"

    def _execute(
        self,
        spec: ModelSpec,
        jobs: list[dict],
        prompt: str,
        progress: Callable[[int, int], None] | None,
        on_page: Callable[[int, dict], None] | None = None,
    ) -> dict:
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=run_worker,
            args=(
                child,
                spec.model_dump(mode="json"),
                jobs,
                prompt,
                float(self.settings.e2_page_timeout_s),
                str(self.cache) if self.cache else None,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        outcome: dict = {
            "pages": {},
            "raw": {},
            "errors": [],
            "fatal": None,
            "meta": None,
            "load_ms": None,
            "peak_rss_mb": None,
            "skipped_reason": None,
        }
        deadline = time.perf_counter() + self.settings.e2_load_timeout_s
        current_page: int | None = None
        done = 0
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    code = "load_timeout" if outcome["meta"] is None else "page_timeout"
                    limit = (
                        self.settings.e2_load_timeout_s
                        if code == "load_timeout"
                        else self.settings.e2_page_timeout_s
                    )
                    message = f"{code}: exceeded {limit}s" + (
                        f" on page {current_page}" if current_page else ""
                    )
                    outcome["errors"].append({"code": code, "page": current_page, "message": message})
                    if code == "load_timeout":
                        outcome["fatal"] = {"code": code, "message": message}
                    elif current_page is not None:
                        outcome["pages"][current_page] = {
                            "ok": False,
                            "error": {"code": code, "message": f"exceeded {limit}s"},
                            "elapsed_ms": limit * 1000.0,
                        }
                        if on_page:
                            on_page(current_page, outcome["pages"][current_page])
                    outcome["skipped_reason"] = f"not run: worker stopped after {code}"
                    break
                if not parent.poll(min(0.1, remaining)):
                    if not process.is_alive() and not parent.poll(0):
                        outcome["errors"].append(
                            {
                                "code": "worker_exit",
                                "page": current_page,
                                "message": f"worker exited with code {process.exitcode}",
                            }
                        )
                        if outcome["meta"] is None:
                            outcome["fatal"] = {
                                "code": "worker_exit",
                                "message": f"worker exited with code {process.exitcode}",
                            }
                        outcome["skipped_reason"] = "not run: worker exited"
                        break
                    continue
                try:
                    message = parent.recv()
                except EOFError:
                    if outcome["meta"] is None and outcome["fatal"] is None:
                        outcome["fatal"] = {"code": "worker_exit", "message": "worker closed"}
                    break
                phase = message.get("phase")
                if phase == "fatal":
                    outcome["fatal"] = message["error"]
                    outcome["errors"].append({**message["error"], "page": None})
                    outcome["skipped_reason"] = f"not run: {message['error']['code']}"
                    break
                if phase == "loaded":
                    outcome["meta"] = message["meta"]
                    outcome["load_ms"] = message["load_ms"]
                    deadline = time.perf_counter() + self.settings.e2_page_timeout_s
                elif phase == "page_start":
                    current_page = message["page"]
                    deadline = time.perf_counter() + self.settings.e2_page_timeout_s
                elif phase == "page":
                    page = message["page"]
                    outcome["pages"][page] = message
                    if on_page:
                        on_page(page, message)
                    if message.get("ok"):
                        outcome["raw"][page] = {
                            "response": message.get("raw"),
                            "prompt_tokens": message.get("prompt_tokens"),
                            "completion_tokens": message.get("completion_tokens"),
                            "blocks": message.get("blocks"),
                            "peak_rss_mb_after_page": message.get("peak_rss_mb"),
                            "cpu_ms": message.get("cpu_ms"),
                            "elapsed_ms": message.get("elapsed_ms"),
                        }
                    else:
                        outcome["errors"].append({**message["error"], "page": page})
                    current_page = None
                    done += 1
                    if progress:
                        progress(done, len(jobs))
                elif phase == "done":
                    outcome["peak_rss_mb"] = message.get("peak_rss_mb")
                    outcome["meta"] = message.get("meta") or outcome["meta"]
                    break
        finally:
            parent.close()
            if process.is_alive():
                process.join(2)
            if process.is_alive():
                process.terminate()
                process.join(5)
        return outcome
