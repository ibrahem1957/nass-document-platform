from __future__ import annotations

import ipaddress
import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator

from lab.core.models import ContractModel
from lab.core.settings import LAB_DIR

MODELS_CONFIG = LAB_DIR / "config" / "models_e2.toml"
Backend = Literal["transformers", "paddleocr_vl", "ollama", "remote", "mock"]
PipelineMode = Literal["element", "official_full_page"]
Configuration = Literal["vendor_default", "experimental"]

# Overrides the official PaddleOCRVL API accepts (PaddleOCR 3.7.0 signatures). Anything
# absent from a spec's `generation` is left to the vendor default and never passed.
OFFICIAL_CONSTRUCTOR_OPTIONS = frozenset(
    {
        "use_doc_orientation_classify",
        "use_doc_unwarping",
        "use_chart_recognition",
        "use_seal_recognition",
        "use_ocr_for_image_block",
        "format_block_content",
        "merge_layout_blocks",
        "markdown_ignore_labels",
        "use_queues",
        "layout_threshold",
        "layout_nms",
        "layout_unclip_ratio",
        "layout_merge_bboxes_mode",
    }
)
OFFICIAL_PREDICT_OPTIONS = frozenset(
    {
        "max_new_tokens",
        "repetition_penalty",
        "temperature",
        "top_p",
        "min_pixels",
        "max_pixels",
        "prompt_label",
        "layout_shape_mode",
    }
)
# PaddleX 3.7.2 doc_vlm/predictor.py: the local (native/transformers) model warns and
# ignores these, so they are recorded as requested-but-ineffective, never as applied.
LOCALLY_IGNORED_OPTIONS = frozenset({"repetition_penalty", "temperature", "top_p"})


class SpecError(ValueError):
    pass


def is_loopback_url(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = parts.hostname or ""
    if parts.scheme not in {"http", "https"} or not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class ModelSpec(ContractModel):
    """One OCR/VLM configuration. Immutable per run; copied into run metadata."""

    id: str
    label: str
    backend: Backend
    repo_id: str
    revision: str
    weights_kind: Literal["full", "peft_lora_adapter", "quantized_gguf", "mock"]
    license: str
    languages: str
    prompt_id: str
    download_gb: float
    min_ram_gb: float
    min_vram_gb: float
    cpu_feasible: bool
    required_packages: list[str] = Field(default_factory=list)
    enabled: bool = True
    base_repo_id: str | None = None
    base_revision: str | None = None
    model_class: str | None = None
    trust_remote_code: bool = False
    min_transformers: str | None = None
    min_paddleocr: str | None = None
    ollama_tag: str | None = None
    endpoint: str | None = None
    notes: str | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    pipeline_mode: PipelineMode = "element"
    pipeline_version: str | None = None
    pipeline_engine: str | None = None
    layout_model_name: str | None = None
    layout_repo_id: str | None = None
    layout_revision: str | None = None
    layout_download_bytes: int | None = None
    layout_required_files: list[str] = Field(default_factory=list)
    # Official mode only: vendor defaults (no overrides) or a labelled experiment.
    configuration: Configuration | None = None
    # Mock specs exist only to validate the architecture in tests; their output
    # is never real inference and is labelled as such everywhere.
    mock_response: str | None = None
    mock_delay_s: float = 0.0
    mock_fail: bool = False

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        if not value or not all(ch.isalnum() or ch in "-._" for ch in value):
            raise ValueError("model id may contain only letters, digits, '-', '.', '_'")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "ModelSpec":
        if self.weights_kind == "peft_lora_adapter" and not (
            self.base_repo_id and self.base_revision
        ):
            raise ValueError("an adapter needs a pinned base_repo_id and base_revision")
        if self.backend == "ollama":
            if not self.ollama_tag or not self.endpoint:
                raise ValueError("ollama models need ollama_tag and endpoint")
            if not is_loopback_url(self.endpoint):
                raise ValueError("ollama endpoint must be a loopback address in E2-A")
        if self.backend == "mock" and self.weights_kind != "mock":
            raise ValueError("mock backend requires weights_kind = 'mock'")
        if self.backend != "mock" and self.mock_response is not None:
            raise ValueError("mock_response is only allowed on mock specs")
        if self.pipeline_mode == "official_full_page":
            if self.backend != "paddleocr_vl":
                raise ValueError("official_full_page mode requires the paddleocr_vl backend")
            required = (
                self.pipeline_version,
                self.pipeline_engine,
                self.layout_model_name,
                self.layout_repo_id,
                self.layout_revision,
            )
            if not all(required) or not self.layout_required_files:
                raise ValueError("official_full_page mode needs pinned layout model metadata")
            if self.pipeline_version != "v1.6":
                raise ValueError("E2-B2 supports only the reviewed PaddleOCR-VL v1.6 pipeline")
            unknown = set(self.generation) - OFFICIAL_CONSTRUCTOR_OPTIONS - OFFICIAL_PREDICT_OPTIONS
            if unknown:
                raise ValueError(f"unsupported official pipeline options: {sorted(unknown)}")
            if self.configuration is None:
                raise ValueError("official_full_page mode must declare configuration")
            if self.configuration == "vendor_default" and self.generation:
                raise ValueError("a vendor_default configuration may not override any option")
            if self.configuration == "experimental" and not self.generation:
                raise ValueError("an experimental configuration must state its overrides")
        return self

    @property
    def official_overrides(self) -> dict[str, dict[str, Any]]:
        """Split official-pipeline overrides into constructor and predict arguments."""
        return {
            "constructor": {
                key: value for key, value in self.generation.items()
                if key in OFFICIAL_CONSTRUCTOR_OPTIONS
            },
            "predict": {
                key: value for key, value in self.generation.items()
                if key in OFFICIAL_PREDICT_OPTIONS
            },
            "requested_but_ignored_locally": {
                key: value for key, value in self.generation.items()
                if key in LOCALLY_IGNORED_OPTIONS
            },
        }

    @property
    def is_mock(self) -> bool:
        return self.backend == "mock"

    @property
    def pinned_ref(self) -> str:
        return f"{self.repo_id}@{self.revision}"

    @property
    def storage_revision(self) -> str:
        """Revision key that separates element and full-pipeline evidence."""
        if self.pipeline_mode == "official_full_page":
            return f"vl-{self.revision}__layout-{self.layout_revision}"
        return self.revision


def load_model_specs(path: Path | None = None) -> list[ModelSpec]:
    source = path or MODELS_CONFIG
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for entry in raw.get("models", []):
        try:
            spec = ModelSpec.model_validate(entry)
        except ValueError as exc:
            raise SpecError(f"Invalid model entry {entry.get('id')!r}: {exc}") from exc
        if spec.id in seen:
            raise SpecError(f"Duplicate model id {spec.id!r}")
        if spec.is_mock:
            raise SpecError("Mock specs are not allowed in the real model registry")
        seen.add(spec.id)
        specs.append(spec)
    return specs


def mock_spec(
    model_id: str = "mock-echo",
    response: str = "MOCK OUTPUT",
    *,
    delay_s: float = 0.0,
    fail: bool = False,
) -> ModelSpec:
    """A clearly labelled architecture-test model. Never real inference."""
    return ModelSpec(
        id=model_id,
        label=f"MOCK {model_id} (architecture test, not real inference)",
        backend="mock",
        repo_id="mock/none",
        revision="mock",
        weights_kind="mock",
        license="n/a",
        languages="n/a",
        prompt_id="vlm_transcribe_v1",
        download_gb=0.0,
        min_ram_gb=0.0,
        min_vram_gb=0.0,
        cpu_feasible=True,
        mock_response=response,
        mock_delay_s=delay_s,
        mock_fail=fail,
    )
