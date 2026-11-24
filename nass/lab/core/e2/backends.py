"""OCR/VLM inference backends: readiness checks and inference sessions.

Every backend receives exactly two inputs per page: a rendered page image and a
static prompt template. No backend can reach a non-loopback host:
- transformers runs offline from a local, revision-pinned Hugging Face snapshot;
- ollama talks only to a loopback endpoint, with proxies disabled;
- remote is refused unconditionally in E2-A.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from lab.core.e2 import REMOTE_INFERENCE_ENABLED
from lab.core.e2.airgap import OFFLINE_ENVIRONMENT as AIRGAP_OFFLINE_ENVIRONMENT
from lab.core.e2.hardware import HardwareProfile, hf_cache_dir, package_version
from lab.core.e2.specs import ModelSpec, is_loopback_url

ReadinessStatus = Literal[
    "ready",
    "mock",
    "disabled",
    "not_installed",
    "not_downloaded",
    "insufficient_hardware",
    "server_unavailable",
    "remote_disabled",
]

# Single source of truth for library offline controls (re-exported for callers).
OFFLINE_ENVIRONMENT = AIRGAP_OFFLINE_ENVIRONMENT


class InferenceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class RemoteInferenceDisabled(InferenceError):
    def __init__(self) -> None:
        super().__init__(
            "remote_disabled",
            "Remote inference is disabled in E2-A; no document or image may leave this machine.",
        )


@dataclass
class Readiness:
    model_id: str
    status: ReadinessStatus
    reasons: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def runnable(self) -> bool:
        return self.status in {"ready", "mock"}

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InferenceOutput:
    text: str
    raw: dict[str, Any]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    blocks: list[dict[str, Any]] | None = None


class Session(Protocol):
    def metadata(self) -> dict[str, Any]: ...

    def infer(self, image_path: Path, prompt: str) -> InferenceOutput: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------- helpers


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def hf_snapshot_dir(repo_id: str, revision: str, cache: Path | None = None) -> Path:
    org_name = repo_id.replace("/", "--")
    return (cache or hf_cache_dir()) / f"models--{org_name}" / "snapshots" / revision


def missing_snapshot_files(snapshot: Path, weights_kind: str) -> list[str]:
    """Files required for an offline load that are absent from a pinned snapshot."""
    if not snapshot.is_dir():
        return [str(snapshot)]
    if weights_kind == "peft_lora_adapter":
        required = ["adapter_config.json", "adapter_model.safetensors"]
        return [name for name in required if not (snapshot / name).is_file()]
    missing = [] if (snapshot / "config.json").is_file() else ["config.json"]
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError):
            return missing + ["model.safetensors.index.json (unreadable)"]
        missing += sorted(name for name in shards if not (snapshot / name).is_file())
    elif not (snapshot / "model.safetensors").is_file():
        missing.append("model.safetensors")
    return missing


def missing_named_files(snapshot: Path, required: list[str]) -> list[str]:
    """Check an explicitly pinned non-Transformers snapshot without downloading."""
    if not snapshot.is_dir():
        return [str(snapshot)]
    return [name for name in required if not (snapshot / name).is_file()]


def _local_opener() -> urllib.request.OpenerDirector:
    # An empty ProxyHandler stops HTTP(S)_PROXY from routing "localhost" elsewhere.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def local_http_json(url: str, payload: dict | None = None, *, timeout: float = 5.0) -> dict:
    """GET/POST JSON to a loopback URL only. Refuses any other host before connecting."""
    if not is_loopback_url(url):
        raise InferenceError("network_refused", f"Refusing non-loopback URL: {url}")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with _local_opener().open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_ollama_payload(
    tag: str, prompt: str, image_bytes: bytes, generation: dict[str, Any]
) -> dict[str, Any]:
    """The complete request body: model tag, static prompt, one image, options."""
    return {
        "model": tag,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
            }
        ],
        "stream": False,
        "options": dict(generation),
    }


# --------------------------------------------------------------- readiness


def _hardware_reasons(spec: ModelSpec, hardware: HardwareProfile) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    blocked = False
    gpu_ok = hardware.cuda_available and hardware.max_vram_gb >= spec.min_vram_gb
    if spec.backend in {"transformers", "paddleocr_vl"} and not gpu_ok:
        if not spec.cpu_feasible:
            blocked = True
            reasons.append(
                f"No CUDA GPU with >= {spec.min_vram_gb:g} GB VRAM and the model is not CPU-feasible."
            )
        else:
            reasons.append(
                "No usable CUDA GPU: inference would run on CPU (expect minutes per page)."
            )
    # Installed RAM decides whether the machine can host the model at all. Available
    # RAM changes from minute to minute with other applications, so it only warns:
    # a transient condition must never label a model "unsupported on current hardware"
    # or hide a real blocker such as missing weights.
    total = hardware.ram_total_gb
    available = hardware.ram_available_gb
    if total is not None and total < spec.min_ram_gb:
        blocked = True
        reasons.append(
            f"Installed RAM {total:.1f} GB is below the estimated {spec.min_ram_gb:g} GB needed."
        )
    elif available is not None and available < spec.min_ram_gb:
        reasons.append(
            f"Available RAM is {available:.1f} GB now, below the estimated {spec.min_ram_gb:g} GB "
            "needed: close other applications before running (Windows can page, but the run may "
            "be very slow or fail)."
        )
    elif available is not None and available < spec.min_ram_gb + 3:
        reasons.append(
            f"Available RAM {available:.1f} GB leaves little headroom over the estimated "
            f"{spec.min_ram_gb:g} GB; close other applications before running."
        )
    return blocked, reasons


def _transformers_readiness(
    spec: ModelSpec, hardware: HardwareProfile, cache: Path | None
) -> Readiness:
    reasons: list[str] = []
    steps: list[str] = []
    details: dict[str, Any] = {}
    missing_packages = [name for name in spec.required_packages if package_version(name) is None]
    if missing_packages:
        reasons.append("Missing Python packages: " + ", ".join(missing_packages))
        steps.append(
            "After approval, install CPU PyTorch and the listed packages into the lab venv "
            "(e.g. pip install torch --index-url https://download.pytorch.org/whl/cpu; "
            "pip install " + " ".join(name for name in missing_packages if name != "torch") + ")."
        )
    transformers_version = package_version("transformers")
    details["transformers"] = transformers_version
    if (
        spec.min_transformers
        and transformers_version
        and _version_tuple(transformers_version) < _version_tuple(spec.min_transformers)
    ):
        missing_packages.append("transformers>=" + spec.min_transformers)
        reasons.append(f"transformers {transformers_version} < required {spec.min_transformers}")
    snapshots = [(spec.repo_id, spec.revision, spec.weights_kind)]
    if spec.weights_kind == "peft_lora_adapter":
        snapshots.append((spec.base_repo_id or "", spec.base_revision or "", "full"))
    missing_weights: list[str] = []
    for repo, revision, kind in snapshots:
        snapshot = hf_snapshot_dir(repo, revision, cache)
        missing = missing_snapshot_files(snapshot, kind)
        details[f"snapshot:{repo}"] = {"path": str(snapshot), "missing": missing}
        if missing:
            missing_weights.append(f"{repo}@{revision[:12]}")
    if missing_weights:
        reasons.append("Weights not in the local cache: " + ", ".join(missing_weights))
        steps.append(
            f"After explicit approval, download about {spec.download_gb:g} GB: "
            + "; ".join(
                f"huggingface-cli download {repo} --revision {revision}"
                for repo, revision, _ in snapshots
            )
        )
    blocked, hardware_reasons = _hardware_reasons(spec, hardware)
    reasons.extend(hardware_reasons)
    if blocked:
        status: ReadinessStatus = "insufficient_hardware"
    elif missing_packages:
        status = "not_installed"
    elif missing_weights:
        status = "not_downloaded"
    else:
        status = "ready"
    return Readiness(spec.id, status, reasons, steps, details)


def _paddleocr_vl_readiness(
    spec: ModelSpec, hardware: HardwareProfile, cache: Path | None
) -> Readiness:
    """Readiness for the official pipeline; this function is strictly offline."""
    reasons: list[str] = []
    steps: list[str] = []
    details: dict[str, Any] = {
        "pipeline_mode": spec.pipeline_mode,
        "pipeline_version": spec.pipeline_version,
        "engine": spec.pipeline_engine,
        "automatic_downloads": "disabled by explicit local model directories",
    }
    missing_packages = [name for name in spec.required_packages if package_version(name) is None]
    if missing_packages:
        reasons.append("Missing Python packages: " + ", ".join(missing_packages))
        steps.append(
            "After the layout-weight approval, install the reviewed integration dependencies "
            "in the isolated E2 environment: pip install \"paddleocr[doc-parser]==3.7.0\"."
        )
    transformers_version = package_version("transformers")
    details["transformers"] = transformers_version
    if (
        spec.min_transformers
        and transformers_version
        and _version_tuple(transformers_version) < _version_tuple(spec.min_transformers)
    ):
        missing_packages.append("transformers>=" + spec.min_transformers)
        reasons.append(f"transformers {transformers_version} < required {spec.min_transformers}")
    paddleocr_version = package_version("paddleocr")
    details["paddleocr"] = paddleocr_version
    if (
        spec.min_paddleocr
        and paddleocr_version
        and _version_tuple(paddleocr_version) < _version_tuple(spec.min_paddleocr)
    ):
        missing_packages.append("paddleocr>=" + spec.min_paddleocr)
        reasons.append(f"paddleocr {paddleocr_version} < required {spec.min_paddleocr}")

    root = cache or hf_cache_dir()
    vl_dir = hf_snapshot_dir(spec.repo_id, spec.revision, root)
    vl_missing = missing_snapshot_files(vl_dir, spec.weights_kind)
    layout_dir = hf_snapshot_dir(spec.layout_repo_id or "", spec.layout_revision or "", root)
    layout_missing = missing_named_files(layout_dir, spec.layout_required_files)
    details["vl_snapshot"] = {"path": str(vl_dir), "missing": vl_missing}
    details["layout_snapshot"] = {"path": str(layout_dir), "missing": layout_missing}

    if vl_missing:
        reasons.append(f"Approved VL weights are not complete at {vl_dir}")
        steps.append(
            "Restore the already-authorized PaddleOCR-VL 1.6 snapshot at the pinned revision; "
            "do not substitute another model."
        )
    if layout_missing:
        exact_bytes = spec.layout_download_bytes or 0
        reasons.append(
            f"Additional layout weights are absent: {spec.layout_repo_id}@{spec.layout_revision} "
            f"({exact_bytes:,} bytes). Explicit user approval is required before download."
        )
        steps.append(
            "After separate explicit approval only: hf download "
            f"{spec.layout_repo_id} --revision {spec.layout_revision}"
        )

    blocked, hardware_reasons = _hardware_reasons(spec, hardware)
    reasons.extend(hardware_reasons)
    if blocked:
        status: ReadinessStatus = "insufficient_hardware"
    elif missing_packages:
        status = "not_installed"
    elif vl_missing or layout_missing:
        status = "not_downloaded"
    else:
        status = "ready"
    return Readiness(spec.id, status, reasons, steps, details)


def _ollama_readiness(spec: ModelSpec, hardware: HardwareProfile) -> Readiness:
    assert spec.endpoint and spec.ollama_tag
    reasons: list[str] = []
    details: dict[str, Any] = {"endpoint": spec.endpoint}
    try:
        version = local_http_json(spec.endpoint.rstrip("/") + "/api/version", timeout=3)
        tags = local_http_json(spec.endpoint.rstrip("/") + "/api/tags", timeout=3)
    except (OSError, ValueError, urllib.error.URLError, InferenceError) as exc:
        return Readiness(
            spec.id,
            "server_unavailable",
            [f"Local Ollama server not reachable at {spec.endpoint}: {exc}"],
            ["Start Ollama locally (it must listen on 127.0.0.1 only)."],
            details,
        )
    details["ollama_version"] = version.get("version")
    models = {item.get("name"): item for item in tags.get("models", [])}
    entry = models.get(spec.ollama_tag)
    blocked, hardware_reasons = _hardware_reasons(spec, hardware)
    reasons.extend(hardware_reasons)
    if entry is None:
        return Readiness(
            spec.id,
            "insufficient_hardware" if blocked else "not_downloaded",
            reasons + [f"Model tag {spec.ollama_tag} is not pulled in the local Ollama."],
            [f"After explicit approval (~{spec.download_gb:g} GB): ollama pull {spec.ollama_tag}"],
            details,
        )
    details["digest"] = entry.get("digest")
    details["size_bytes"] = entry.get("size")
    return Readiness(
        spec.id, "insufficient_hardware" if blocked else "ready", reasons, [], details
    )


def check_readiness(
    spec: ModelSpec, hardware: HardwareProfile, *, cache: Path | None = None
) -> Readiness:
    if not spec.enabled:
        return Readiness(spec.id, "disabled", ["Disabled in models_e2.toml"])
    if spec.backend == "remote":
        return Readiness(
            spec.id,
            "remote_disabled",
            ["Remote inference is disabled in E2-A (documents never leave this machine)."],
        )
    if spec.backend == "mock":
        return Readiness(
            spec.id, "mock", ["MOCK backend for architecture tests; not real inference."]
        )
    if spec.backend == "ollama":
        return _ollama_readiness(spec, hardware)
    if spec.backend == "paddleocr_vl":
        return _paddleocr_vl_readiness(spec, hardware, cache)
    return _transformers_readiness(spec, hardware, cache)


# ---------------------------------------------------------------- sessions


class MockSession:
    """Deterministic architecture-test session. Records exactly what it received."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def metadata(self) -> dict[str, Any]:
        return {"backend": "mock", "is_mock": True, "device": "none"}

    def infer(self, image_path: Path, prompt: str) -> InferenceOutput:
        if self.spec.mock_delay_s:
            time.sleep(self.spec.mock_delay_s)
        if self.spec.mock_fail:
            raise InferenceError("backend_error", "Mock backend configured to fail")
        image_bytes = Path(image_path).read_bytes()
        return InferenceOutput(
            text=self.spec.mock_response or "",
            raw={
                "mock": True,
                "received": {
                    "prompt": prompt,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                },
            },
        )

    def close(self) -> None:
        return None


class OllamaSession:
    def __init__(self, spec: ModelSpec, timeout_s: float):
        assert spec.endpoint and spec.ollama_tag
        self.spec = spec
        self.timeout_s = timeout_s
        base = spec.endpoint.rstrip("/")
        self._base = base
        tags = local_http_json(base + "/api/tags", timeout=5)
        entry = next(
            (item for item in tags.get("models", []) if item.get("name") == spec.ollama_tag),
            None,
        )
        if entry is None:
            raise InferenceError("not_downloaded", f"{spec.ollama_tag} is not pulled locally")
        self._meta = {
            "backend": "ollama",
            "ollama_version": local_http_json(base + "/api/version", timeout=5).get("version"),
            "digest": entry.get("digest"),
            "quantization": (entry.get("details") or {}).get("quantization_level"),
            "is_mock": False,
        }

    def metadata(self) -> dict[str, Any]:
        return dict(self._meta)

    def infer(self, image_path: Path, prompt: str) -> InferenceOutput:
        payload = build_ollama_payload(
            self.spec.ollama_tag or "", prompt, Path(image_path).read_bytes(), self.spec.generation
        )
        response = local_http_json(self._base + "/api/chat", payload, timeout=self.timeout_s)
        try:
            loaded = local_http_json(self._base + "/api/ps", timeout=5)
            memory = [
                {"size": item.get("size"), "size_vram": item.get("size_vram")}
                for item in loaded.get("models", [])
                if item.get("name") == self.spec.ollama_tag
            ]
        except (OSError, ValueError, urllib.error.URLError, InferenceError):
            memory = []
        message = response.get("message") or {}
        raw = {key: value for key, value in response.items() if key != "message"}
        raw["message_role"] = message.get("role")
        raw["server_memory"] = memory
        return InferenceOutput(
            text=str(message.get("content", "")),
            raw=raw,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

    def close(self) -> None:
        return None


class TransformersSession:
    """Offline PyTorch inference from pinned local snapshots (never downloads)."""

    def __init__(self, spec: ModelSpec, cache: Path | None = None):
        os.environ.update(OFFLINE_ENVIRONMENT)
        import torch  # noqa: PLC0415 - optional heavy dependency
        import transformers  # noqa: PLC0415

        self.spec = spec
        self.torch = torch
        weights_repo, weights_revision = spec.repo_id, spec.revision
        adapter_dir = None
        if spec.weights_kind == "peft_lora_adapter":
            adapter_dir = hf_snapshot_dir(spec.repo_id, spec.revision, cache)
            weights_repo, weights_revision = spec.base_repo_id or "", spec.base_revision or ""
        weights_dir = hf_snapshot_dir(weights_repo, weights_revision, cache)
        for directory, kind in ((weights_dir, "full"), (adapter_dir, spec.weights_kind)):
            if directory is not None and missing_snapshot_files(directory, kind):
                raise InferenceError("not_downloaded", f"Incomplete local snapshot: {directory}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16
        model_class = getattr(transformers, spec.model_class or "AutoModelForImageTextToText")
        model = model_class.from_pretrained(
            str(weights_dir),
            torch_dtype=dtype,
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
        )
        if adapter_dir is not None:
            from peft import PeftModel  # noqa: PLC0415

            model = PeftModel.from_pretrained(model, str(adapter_dir), local_files_only=True)
        self.model = model.to(self.device).eval()
        processor_dir = adapter_dir if adapter_dir is not None else weights_dir
        self.processor = transformers.AutoProcessor.from_pretrained(
            str(processor_dir),
            local_files_only=True,
            trust_remote_code=spec.trust_remote_code,
        )
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._meta = {
            "backend": "transformers",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": package_version("peft") if adapter_dir is not None else None,
            "device": self.device,
            "dtype": str(dtype),
            "weights_dir": str(weights_dir),
            "adapter_dir": str(adapter_dir) if adapter_dir else None,
            # transformers 5 resolves legacy "...Fast" processor names to a torchvision or
            # PIL backend depending on what is installed; the two resize differently, so
            # the class actually used is recorded with every run.
            "image_processor": type(getattr(self.processor, "image_processor", None)).__name__,
            "is_mock": False,
        }

    def metadata(self) -> dict[str, Any]:
        meta = dict(self._meta)
        if self.device == "cuda":
            meta["peak_vram_mb"] = self.torch.cuda.max_memory_allocated() / 1024**2
        return meta

    def infer(self, image_path: Path, prompt: str) -> InferenceOutput:
        from PIL import Image  # noqa: PLC0415

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generation = {"max_new_tokens": 2048, "do_sample": False}
        generation.update(self.spec.generation)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[-1]
        new_tokens = generated[0][prompt_length:]
        text = self.processor.decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        new_count = int(new_tokens.shape[-1])
        return InferenceOutput(
            text=text,
            raw={
                "generation": generation,
                "new_tokens": new_count,
                # Hitting the cap means the transcription may be cut off.
                "truncated_at_max_new_tokens": new_count >= int(generation["max_new_tokens"]),
                "image_grid_thw": (
                    inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else None
                ),
            },
            prompt_tokens=int(prompt_length),
            completion_tokens=int(new_tokens.shape[-1]),
        )

    def close(self) -> None:
        self.model = None


def _json_safe(value: Any) -> Any:
    """Convert official pipeline values to lossless-enough JSON primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


# Documented defaults of the installed vendor code, read from source (not executed):
# paddlex 3.7.2 configs/pipelines/PaddleOCR-VL-1.6.yaml, paddleocr_vl/pipeline.py and
# doc_vlm/predictor.py. Recorded with every official run so results stay interpretable.
OFFICIAL_VENDOR_DEFAULTS: dict[str, Any] = {
    "source": "paddleocr==3.7.0 / paddlex==3.7.2 (PaddleOCR-VL-1.6.yaml, pipeline.py, predictor.py)",
    "use_doc_preprocessor": False,
    "use_layout_detection": True,
    "use_chart_recognition": False,
    "use_seal_recognition": False,
    "format_block_content": False,
    "merge_layout_blocks": True,
    "use_queues": True,
    "markdown_ignore_labels": [
        "number", "footnote", "header", "header_image", "footer", "footer_image", "aside_text",
    ],
    "layout_threshold": 0.3,
    "max_new_tokens": 4096,
    "region_min_pixels": 112896,
    "region_max_pixels": 1003520,
    "repetition_penalty": "unsupported by the local model (ignored with a warning)",
    "temperature": "unsupported by the local model (ignored with a warning)",
    "top_p": "unsupported by the local model (ignored with a warning)",
    "vl_backend": "native",
}


def normalize_official_pipeline_result(
    result: Any, *, inference_config: dict[str, Any] | None = None
) -> InferenceOutput:
    """Preserve official JSON/Markdown and expose reading-order blocks to the lab."""
    json_value = getattr(result, "json", result)
    if callable(json_value):
        json_value = json_value()
    markdown_value = getattr(result, "markdown", None)
    if callable(markdown_value):
        markdown_value = markdown_value()
    payload = _json_safe(json_value)
    markdown = _json_safe(markdown_value)
    body = payload.get("res", payload) if isinstance(payload, dict) else {}
    parsing = body.get("parsing_res_list", []) if isinstance(body, dict) else []

    markdown_text = ""
    if isinstance(markdown, dict):
        candidate = markdown.get("markdown_texts", "")
        if isinstance(candidate, str):
            markdown_text = candidate
        elif isinstance(candidate, list):
            markdown_text = "\n\n".join(str(item) for item in candidate)
    raw_blocks: list[dict[str, Any]] = []
    ordered_blocks: list[dict[str, Any]] = []
    for item in parsing if isinstance(parsing, list) else []:
        if not isinstance(item, dict):
            continue
        content = str(item.get("block_content") or "")
        order = item.get("block_order")
        explicit_order = isinstance(order, int)
        if not explicit_order:
            order = None
        coordinate = item.get("block_bbox")
        bbox = None
        if isinstance(coordinate, list) and len(coordinate) >= 4:
            flat = coordinate
            if coordinate and isinstance(coordinate[0], list):
                points = [point for point in coordinate if isinstance(point, list) and len(point) >= 2]
                if points:
                    xs = [float(point[0]) for point in points]
                    ys = [float(point[1]) for point in points]
                    flat = [min(xs), min(ys), max(xs), max(ys)]
            if len(flat) >= 4 and all(isinstance(value, (int, float)) for value in flat[:4]):
                bbox = [float(value) for value in flat[:4]]
        block = {
            "order": order,
            "order_status": "explicit" if explicit_order else "unknown",
            "text": content,
            "bbox": bbox,
            "label": item.get("block_label"),
            "block_id": item.get("block_id"),
            "truncation_status": "unknown",
            "raw": item,
        }
        raw_blocks.append(block)
        if explicit_order:
            ordered_blocks.append(block)
    ordered_blocks.sort(key=lambda item: item["order"])
    order_complete = len(ordered_blocks) == len(raw_blocks)
    assembly_warning = None
    if not markdown_text and order_complete:
        markdown_text = "\n\n".join(item["text"] for item in ordered_blocks if item["text"])
    elif not markdown_text and not order_complete:
        assembly_warning = "No official Markdown and one or more blocks lack explicit reading order"
    return InferenceOutput(
        text=markdown_text,
        raw={
            "official_pipeline_json": payload,
            "official_pipeline_markdown": markdown,
            "layout_regions": [item["raw"] for item in raw_blocks],
            "structured_tables": [
                item["raw"]
                for item in raw_blocks
                if str(item.get("label", "")).lower() == "table"
            ],
            "reading_order_status": "explicit" if order_complete else "incomplete_or_unknown",
            "assembly_warning": assembly_warning,
            "inference_config": dict(inference_config or {}),
            "truncated_at_max_new_tokens": None,
            "truncation_status": "unknown_not_reported_by_official_pipeline",
            "block_truncation": [
                {
                    "block_id": item["block_id"],
                    "order": item["order"],
                    "status": item["truncation_status"],
                }
                for item in raw_blocks
            ],
        },
        blocks=ordered_blocks if order_complete else None,
    )


class PaddleOCRVLFullPageSession:
    """Official v1.6 layout -> crop -> VL recognition -> assembly pipeline."""

    def __init__(self, spec: ModelSpec, cache: Path | None = None):
        os.environ.update(OFFLINE_ENVIRONMENT)
        root = cache or hf_cache_dir()
        self.spec = spec
        self.layout_dir = hf_snapshot_dir(
            spec.layout_repo_id or "", spec.layout_revision or "", root
        )
        self.vl_dir = hf_snapshot_dir(spec.repo_id, spec.revision, root)
        layout_missing = missing_named_files(self.layout_dir, spec.layout_required_files)
        vl_missing = missing_snapshot_files(self.vl_dir, spec.weights_kind)
        if layout_missing or vl_missing:
            raise InferenceError(
                "not_downloaded",
                f"Official pipeline assets incomplete; layout={layout_missing}, vl={vl_missing}",
            )
        # PaddleX creates ~/.paddlex (temp, locks) on import; keep it beside the result
        # store on the evaluation drive unless the operator chose a location.
        from lab.core.e2.store import results_root  # noqa: PLC0415

        os.environ.setdefault(
            "PADDLE_PDX_CACHE_HOME", str(Path(results_root()).resolve().parent / "paddlex_home")
        )
        try:
            from paddleocr import PaddleOCRVL  # noqa: PLC0415
        except ImportError as exc:
            raise InferenceError("not_installed", "paddleocr[doc-parser] is not installed") from exc
        settings = dict(spec.generation)
        overrides = spec.official_overrides
        # Identity and local paths are always explicit (no hub lookup); every other option
        # is passed only when the spec overrides it, so `vendor_default` runs exactly the
        # documented PaddleOCR-VL-1.6 pipeline configuration.
        self._predict_overrides = dict(overrides["predict"])
        self.pipeline = PaddleOCRVL(
            pipeline_version=spec.pipeline_version,
            engine=spec.pipeline_engine,
            layout_detection_model_name=spec.layout_model_name,
            layout_detection_model_dir=str(self.layout_dir),
            vl_rec_model_name="PaddleOCR-VL-1.6-0.9B",
            vl_rec_model_dir=str(self.vl_dir),
            use_layout_detection=True,
            **overrides["constructor"],
        )
        self._effective = {
            "configuration": spec.configuration,
            "explicit_identity": {
                "pipeline_version": spec.pipeline_version,
                "engine": spec.pipeline_engine,
                "layout_detection_model_name": spec.layout_model_name,
                "vl_rec_model_name": "PaddleOCR-VL-1.6-0.9B",
                "use_layout_detection": True,
            },
            "constructor_overrides": dict(overrides["constructor"]),
            "predict_overrides": dict(overrides["predict"]),
            "requested_but_ignored_by_local_model": dict(overrides["requested_but_ignored_locally"]),
            "vendor_defaults_reference": OFFICIAL_VENDOR_DEFAULTS,
        }
        self._meta = {
            "backend": "paddleocr_vl",
            "pipeline_mode": "official_full_page",
            "pipeline_version": spec.pipeline_version,
            "engine": spec.pipeline_engine,
            "layout_model": f"{spec.layout_repo_id}@{spec.layout_revision}",
            "layout_dir": str(self.layout_dir),
            "vl_model": spec.pinned_ref,
            "vl_dir": str(self.vl_dir),
            "paddleocr": package_version("paddleocr"),
            "paddlex": package_version("paddlex"),
            "transformers": package_version("transformers"),
            "is_mock": False,
            "inference_config": settings,
            "effective_configuration": self._effective,
        }

    def metadata(self) -> dict[str, Any]:
        return dict(self._meta)

    def infer(self, image_path: Path, prompt: str) -> InferenceOutput:
        del prompt  # The official pipeline selects per-region prompts from layout labels.
        results = list(
            self.pipeline.predict(input=str(image_path), **self._predict_overrides)
        )
        if len(results) != 1:
            raise InferenceError("pipeline_output", f"Expected one page result, got {len(results)}")
        return normalize_official_pipeline_result(
            results[0], inference_config=dict(self._effective)
        )

    def close(self) -> None:
        self.pipeline = None


def open_session(spec: ModelSpec, *, page_timeout_s: float, cache: Path | None = None) -> Session:
    # Remote stays refused even if the switch is flipped: E2-A has no remote client.
    if spec.backend not in {"transformers", "paddleocr_vl", "ollama", "mock"} or (
        spec.backend == "remote" and not REMOTE_INFERENCE_ENABLED
    ):
        raise RemoteInferenceDisabled()
    if spec.backend == "mock":
        return MockSession(spec)
    if spec.backend == "ollama":
        return OllamaSession(spec, page_timeout_s)
    if spec.backend == "paddleocr_vl":
        return PaddleOCRVLFullPageSession(spec, cache)
    return TransformersSession(spec, cache)
