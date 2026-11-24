"""Hosted API provider registry (separate from the local model registry)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from lab.core.models import ContractModel
from lab.core.settings import LAB_DIR

PROVIDERS_CONFIG = LAB_DIR / "config" / "providers_e2.toml"
IMPLEMENTED_PROTOCOLS = frozenset({"baidu_paddleocr_jobs_v2", "openai_chat_completions_v1"})
# OpenAI-compatible request options we know the protocol defines. Anything else is
# refused at load time, so no undocumented parameter is ever sent.
OPENAI_GENERATION_KEYS = frozenset({"temperature", "max_tokens", "top_p", "seed", "stop"})


class ProviderError(ValueError):
    pass


class ProviderPricing(ContractModel):
    basis: Literal["per_page", "per_token", "per_instance_hour", "per_compute_second"]
    # Worst-case estimate used by the cost cap BEFORE sending. A negative value means
    # "unknown": a run is refused rather than guessed.
    estimated_usd_per_page: float
    source: str
    # Per-token rates (per_token basis) used to compute cost from reported usage.
    usd_per_million_input_tokens: float | None = None
    usd_per_million_output_tokens: float | None = None
    # Compute-time billing (per_compute_second basis). The preflight bound is
    # rate x max_billable_seconds_per_request; the actual charge is reconciled from the
    # provider-reported cost of each response.
    usd_per_compute_second: float | None = Field(default=None, gt=0)
    max_billable_seconds_per_request: float | None = Field(default=None, gt=0)
    price_verified: bool = False

    @model_validator(mode="after")
    def _compute_bound(self) -> "ProviderPricing":
        if self.basis == "per_compute_second":
            if self.usd_per_compute_second is None or self.max_billable_seconds_per_request is None:
                raise ValueError("per_compute_second pricing needs a rate and max_billable_seconds_per_request")
            if self.estimated_usd_per_page + 1e-12 < self.compute_bound_usd():
                raise ValueError("estimated_usd_per_page must be at least rate x max billable seconds")
        return self

    def compute_bound_usd(self) -> float | None:
        if self.basis != "per_compute_second" or self.usd_per_compute_second is None:
            return None
        return self.usd_per_compute_second * float(self.max_billable_seconds_per_request or 0)

    def cost_from_usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
        if self.basis != "per_token" or prompt_tokens is None or completion_tokens is None:
            return None
        if self.usd_per_million_input_tokens is None or self.usd_per_million_output_tokens is None:
            return None
        return (prompt_tokens * self.usd_per_million_input_tokens
                + completion_tokens * self.usd_per_million_output_tokens) / 1_000_000


class ProviderLimits(ContractModel):
    max_pages_per_request: int = Field(ge=1, le=1)
    max_requests_per_run: int = Field(ge=1, le=1)
    request_timeout_s: float = Field(gt=0, le=600)
    poll_timeout_s: float = Field(ge=0, le=3600)
    poll_initial_interval_s: float = Field(ge=0)
    poll_max_interval_s: float = Field(ge=0)
    max_response_bytes: int = Field(gt=0, le=100_000_000)
    max_cost_usd_per_run: float = Field(ge=0)
    # Internal usage policy of this project (NOT provider quotas), read by authorization.py.
    # max_pages_per_enablement: ceiling for one enablement's budget (counted from its start).
    # max_pages_per_day_ceiling: ceiling for the per-day limit (UTC day, all enablements).
    # max_pages_cumulative: cap over the whole usage history, so disabling and re-enabling
    #   cannot raise the internal budget without limit (None = no cumulative cap).
    # provider_documented_daily_pages: the provider's documented daily quota, when known.
    max_pages_per_enablement: int | None = Field(default=None, ge=1, le=1_000_000)
    max_pages_per_day_ceiling: int | None = Field(default=None, ge=1, le=1_000_000)
    max_pages_cumulative: int | None = Field(default=None, ge=1, le=10_000_000)
    default_enablement_days: int | None = Field(default=None, ge=1, le=30)
    provider_documented_daily_pages: int | None = Field(default=None, ge=1)


class ProviderSpec(ContractModel):
    id: str
    label: str
    provider: str
    status: Literal["implemented", "planned"]
    protocol: str
    api_base_url: str
    api_path: str
    api_model_id: str
    requested_model: str
    local_counterpart: str | None
    reported_revision: str
    exact_revision_verified: bool
    classification: Literal[
        "EXACT MATCH VERIFIED",
        "SAME MODEL FAMILY, REVISION UNVERIFIED",
        "DIFFERENT QUANTIZATION OR PIPELINE",
        "NOT VERIFIED",
    ]
    credential_env: str
    credential_label: str
    result_hosts: list[str] = Field(default_factory=list)
    # Exact result-storage hostnames (no wildcards) that a user-enabled automatic
    # approval policy may approve for a paused job. Never used without that policy,
    # never for submission, and only for the Baidu jobs protocol.
    auto_result_hosts: list[str] = Field(default_factory=list)
    optional_payload: dict[str, Any] = Field(default_factory=dict)
    # Prompted (chat) protocols: the static registry prompt and generation options.
    prompt_id: str | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    # Chat protocols only: request server-sent events (`stream: true`, documented by
    # the provider). The idle timeout still applies per read, and the whole response
    # (headers + body) must arrive within request_timeout_s after the upload.
    stream: bool = False
    data_residency: str
    protocol_source: str
    pricing: ProviderPricing
    limits: ProviderLimits

    @model_validator(mode="after")
    def _consistent(self) -> "ProviderSpec":
        parts = urlsplit(self.api_base_url)
        if parts.scheme != "https" or not parts.hostname or parts.path not in {"", "/"}:
            raise ValueError("api_base_url must be a bare https://host URL")
        if any(urlsplit(f"https://{host}").hostname != host.lower() for host in self.result_hosts):
            raise ValueError("result_hosts must be bare host names")
        if self.auto_result_hosts:
            if self.protocol != "baidu_paddleocr_jobs_v2":
                raise ValueError("auto_result_hosts is only supported for the Baidu jobs protocol")
            for host in self.auto_result_hosts:
                if (host != host.strip().lower() or "*" in host or host.count(".") < 2
                        or urlsplit(f"https://{host}").hostname != host
                        or all(part.isdigit() for part in host.split("."))):
                    raise ValueError(f"auto_result_hosts entries must be exact public hostnames: {host!r}")
        if self.classification == "EXACT MATCH VERIFIED" and not self.exact_revision_verified:
            raise ValueError("EXACT MATCH VERIFIED requires exact_revision_verified = true")
        if self.status == "implemented" and self.protocol not in IMPLEMENTED_PROTOCOLS:
            raise ValueError(f"protocol {self.protocol!r} has no implementation")
        if self.protocol == "openai_chat_completions_v1":
            if not self.prompt_id:
                raise ValueError("chat protocols need a static registry prompt_id")
            unknown = set(self.generation) - OPENAI_GENERATION_KEYS
            if unknown:
                raise ValueError(f"unsupported generation options {sorted(unknown)}")
            if self.result_hosts:
                raise ValueError("synchronous chat protocols return results inline; result_hosts must be empty")
        elif self.stream:
            raise ValueError("stream is only supported for chat protocols")
        if self.pricing.basis == "per_compute_second":
            if self.limits.max_cost_usd_per_run + 1e-12 < self.pricing.estimated_usd_per_page:
                raise ValueError("max_cost_usd_per_run must cover the per-request compute-time bound")
            if float(self.pricing.max_billable_seconds_per_request or 0) < self.limits.request_timeout_s:
                raise ValueError("max_billable_seconds_per_request must be at least the request timeout")
        return self

    @property
    def api_host(self) -> str:
        return str(urlsplit(self.api_base_url).hostname).lower()

    @property
    def endpoint(self) -> str:
        return self.api_base_url.rstrip("/") + self.api_path

    @property
    def adapter_id(self) -> str:
        return f"api.{self.id}"

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset({self.api_host, *(host.lower() for host in self.result_hosts)})

    @property
    def runnable(self) -> bool:
        return self.status == "implemented" and self.protocol in IMPLEMENTED_PROTOCOLS

    def identity(self) -> dict[str, Any]:
        """Everything recorded with a run to describe what was (and was not) verified."""
        identity = {
            "provider": self.provider,
            "provider_id": self.id,
            "api_model_id": self.api_model_id,
            "endpoint": self.endpoint,
            "requested_model": self.requested_model,
            "reported_revision": self.reported_revision,
            "exact_revision_verified": self.exact_revision_verified,
            "classification": self.classification,
            "local_counterpart": self.local_counterpart,
            "optional_payload": dict(self.optional_payload),
            "prompt_id": self.prompt_id,
            "generation": dict(self.generation),
            "data_residency": self.data_residency,
            "protocol": self.protocol,
            "protocol_source": self.protocol_source,
        }
        # Added only when enabled, so identities of existing non-streaming runs (and
        # their Stage 2 reuse fingerprints) are unchanged.
        if self.stream:
            identity["stream"] = True
        return identity


def load_providers(path: Path | None = None) -> list[ProviderSpec]:
    source = path or PROVIDERS_CONFIG
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    specs: list[ProviderSpec] = []
    seen: set[str] = set()
    for entry in raw.get("providers", []):
        try:
            spec = ProviderSpec.model_validate(entry)
        except ValueError as exc:
            raise ProviderError(f"Invalid provider {entry.get('id')!r}: {exc}") from exc
        if spec.id in seen:
            raise ProviderError(f"Duplicate provider id {spec.id!r}")
        seen.add(spec.id)
        specs.append(spec)
    return specs
