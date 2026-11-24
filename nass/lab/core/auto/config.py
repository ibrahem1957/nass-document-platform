"""Typed configuration for Automatic Extraction Mode (local machine or GPU cluster, by configuration).

Sources, later ones win: built-in defaults -> TOML file (`AGENTICDOC_AUTO_CONFIG`, default
lab/config/auto_extraction.toml) -> environment variables. Secrets (the Qari token) come only from
the environment and are never written to any file or manifest.

Two kinds of settings:
- PROVEN: the benchmark-validated extraction configuration (Qari model, prompt, decoding, limits,
  loop stopper, retry; PP-DocLayout model; 60 px native-pixel PNG crops). Changing any of these
  requires `AUTO_ALLOW_PROVEN_OVERRIDES=true`, and every change is recorded in the manifest.
- OPERATIONAL: endpoints, devices, concurrency, limits, paths, review policies. Safe to adjust.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lab.core.settings import LAB_DIR

CONFIG_ENV = "AGENTICDOC_AUTO_CONFIG"
DOTENV_ENV = "AGENTICDOC_ENV_FILE"
DEFAULT_DOTENV = LAB_DIR.parent / ".env"  # nass/.env (git-ignored)
DEFAULT_CONFIG_FILE = LAB_DIR / "config" / "auto_extraction.toml"


class ConfigError(ValueError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ------------------------------------------------------------------ proven


class ProvenQari(_Model):
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    adapter: str = "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"
    prompt: str = "Free OCR."
    do_sample: bool = False
    max_new_tokens: int = 1536
    max_time_crop_s: float = 30.0
    max_time_page_s: float = 60.0
    loop_max_ngram: int = 40
    loop_repeats: int = 5
    retry_repetition_penalty: float = 1.1
    retries_after_loop_timeout_or_limit: int = 1


class ProvenLayout(_Model):
    model_name: str = "PP-DocLayout_plus-L"
    crop_margin_px: int = 60
    crop_format: str = "png"
    crop_pixels: str = "native"
    unwarping: bool = False
    orientation_correction: bool = False
    # Digital/mixed visuals rendered from the PDF page (the previously validated resolution).
    digital_render_dpi: int = 250


class ProvenSettings(_Model):
    qari: ProvenQari = ProvenQari()
    layout: ProvenLayout = ProvenLayout()


# ------------------------------------------------------------- operational


class AppSettings(_Model):
    environment: Literal["local", "staging", "production"] = "local"
    results_dir: str | None = None  # None -> AGENTICDOC_RESULTS_DIR / the lab default
    max_pdf_mb: int = Field(default=200, ge=1, le=10_000)
    max_pages: int = Field(default=500, ge=1, le=100_000)


class LayoutSettings(_Model):
    # Local PP-DocLayout_plus-L *_infer directory (downloaded beforehand; never downloaded by the app).
    # Windows paths are accepted as written, e.g. C:\Users\<you>\models\PP-DocLayout_plus-L_infer.
    model_dir: str | None = None

    @field_validator("model_dir", mode="before")
    @classmethod
    def _model_dir(cls, value: Any) -> Any:
        if value is None or not str(value).strip():
            return None
        text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"').strip("'")))
        return str(Path(text))  # normalizes separators: C:\... and C:/... both work on Windows
    device: str = "cpu"
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    selection_policy: Literal["confident_only", "review_all"] = "confident_only"
    labels: tuple[str, ...] = ("chart", "image", "table")
    attach_titles: bool = True
    captions_below: bool = False
    page_render_dpi: int = Field(default=200, ge=72, le=600)  # pages without an embedded raster only
    # Digital/mixed visuals: native image bytes only for a clean single layer of at least this resolution
    native_min_dpi: float = Field(default=200.0, ge=50, le=1200)
    native_bbox_tolerance_pt: float = Field(default=2.0, ge=0, le=20)
    crop_quality_min_std: float = Field(default=3.0, ge=0, le=50)  # luminance std below this = blank/flat

    @field_validator("device")
    @classmethod
    def _device(cls, value: str) -> str:
        if value == "cpu" or value.startswith(("cuda", "gpu")) or value in {"npu", "xpu", "mps"} or ":" in value:
            return value
        raise ValueError(f"unsupported device {value!r} (use cpu, cuda:0, gpu:0, ...)")


class QariSettings(_Model):
    service_url: str | None = None
    token_env: str = "QARI_SERVICE_TOKEN"  # the token itself is read from this environment variable
    request_timeout_s: float = Field(default=180.0, gt=0, le=3600)
    max_connect_retries: int = Field(default=2, ge=0, le=10)  # only before anything is sent
    concurrency: int = Field(default=1, ge=1, le=64)
    max_image_pixels: int | None = Field(default=None, ge=10_000)  # None = native pixels, no resizing
    # Endpoint trust is configured by the administrator, never by the person running an extraction:
    #   local                 the service runs on this machine (loopback host only; http allowed)
    #   internal              a company-internal GPU server (private network; http or https)
    #   development_external  an external development service (e.g. RunPod): https only, and its host
    #                         must be listed in trusted_development_hosts
    deployment_mode: Literal["local", "internal", "development_external"] = "local"
    automatic_enabled: bool = True  # false: runs are planned and paused (offline export/import still works)
    trusted_development_hosts: tuple[str, ...] = ()  # exact host names, development_external only

    @field_validator("trusted_development_hosts", mode="before")
    @classmethod
    def _hosts(cls, value: Any) -> Any:
        if isinstance(value, str):
            return tuple(h.strip().lower() for h in value.split(",") if h.strip())
        return tuple(str(h).strip().lower() for h in value or () if str(h).strip())


class ProcessingSettings(_Model):
    on_failure: Literal["continue", "stop"] = "continue"  # after a Qari service error (unavailable always pauses)
    max_jobs: int = Field(default=2000, ge=1, le=100_000)  # OCR jobs per run (bounded work queue)


class ValidationSettings(_Model):
    # flag: finalize with flags in validation_report.json; block_finalize: errors or pending review block it
    review_policy: Literal["flag", "block_finalize"] = "flag"


class VerificationSettings(_Model):
    """Optional second read of figure crops that contain Arabic-Indic / Persian digits (REPORT ONLY).
    Same Qari model, prompt and generation settings; at most one extra request per crop; per-run limit."""
    digits_enabled: bool = True
    max_requests: int = Field(default=30, ge=0, le=10_000)  # per run; 0 = never send (recorded as skipped)
    upscale: float = Field(default=1.5, ge=1.0, le=3.0)     # Lanczos upscaling of the verification copy


class AutoConfig(_Model):
    app: AppSettings = AppSettings()
    layout: LayoutSettings = LayoutSettings()
    qari: QariSettings = QariSettings()
    processing: ProcessingSettings = ProcessingSettings()
    validation: ValidationSettings = ValidationSettings()
    verification: VerificationSettings = VerificationSettings()
    proven: ProvenSettings = ProvenSettings()
    allow_proven_overrides: bool = False
    proven_overrides: dict[str, Any] = Field(default_factory=dict)

    def qari_token(self) -> str | None:
        return os.environ.get(self.qari.token_env) or None

    def snapshot(self) -> dict[str, Any]:
        """Manifest form: every setting, never the token."""
        data = self.model_dump(mode="json")
        data["qari"]["token_present"] = bool(self.qari_token())
        return data


# Environment variables -> (section, field, type)
ENVIRONMENT: dict[str, tuple[str, str, Any]] = {
    "AGENTICDOC_ENV": ("app", "environment", str),
    "AGENTICDOC_RESULTS_DIR": ("app", "results_dir", str),
    "AGENTICDOC_MAX_PDF_MB": ("app", "max_pdf_mb", int),
    "AGENTICDOC_MAX_PAGES": ("app", "max_pages", int),
    "LAYOUT_MODEL_DIR": ("layout", "model_dir", str),
    "LAYOUT_DEVICE": ("layout", "device", str),
    "LAYOUT_SCORE_THRESHOLD": ("layout", "score_threshold", float),
    "LAYOUT_SELECTION_POLICY": ("layout", "selection_policy", str),
    "LAYOUT_PAGE_RENDER_DPI": ("layout", "page_render_dpi", int),
    "LAYOUT_NATIVE_MIN_DPI": ("layout", "native_min_dpi", float),
    "LAYOUT_NATIVE_BBOX_TOLERANCE_PT": ("layout", "native_bbox_tolerance_pt", float),
    "LAYOUT_CROP_QUALITY_MIN_STD": ("layout", "crop_quality_min_std", float),
    "QARI_SERVICE_URL": ("qari", "service_url", str),
    "QARI_REQUEST_TIMEOUT_S": ("qari", "request_timeout_s", float),
    "QARI_MAX_CONNECT_RETRIES": ("qari", "max_connect_retries", int),
    "QARI_CONCURRENCY": ("qari", "concurrency", int),
    "QARI_MAX_IMAGE_PIXELS": ("qari", "max_image_pixels", int),
    "QARI_DEPLOYMENT_MODE": ("qari", "deployment_mode", str),
    "QARI_AUTOMATIC_ENABLED": ("qari", "automatic_enabled", bool),
    "QARI_TRUSTED_DEV_HOSTS": ("qari", "trusted_development_hosts", str),
    "PROCESSING_ON_FAILURE": ("processing", "on_failure", str),
    "PROCESSING_MAX_JOBS": ("processing", "max_jobs", int),
    "VALIDATION_REVIEW_POLICY": ("validation", "review_policy", str),
    "VERIFY_DIGITS_ENABLED": ("verification", "digits_enabled", bool),
    "VERIFY_DIGITS_MAX_REQUESTS": ("verification", "max_requests", int),
    "VERIFY_DIGITS_UPSCALE": ("verification", "upscale", float),
    "AUTO_ALLOW_PROVEN_OVERRIDES": ("", "allow_proven_overrides", bool),
}
RETIRED_ENVIRONMENT = ("QARI_TRUSTED_INTERNAL", "QARI_REQUIRE_DOCUMENT_AUTHORIZATION")
# The app reads the configuration when a run starts; these are cached by the loaded layout model.
RESTART_REQUIRED = {"LAYOUT_MODEL_DIR", "LAYOUT_DEVICE", "LAYOUT_SCORE_THRESHOLD"}


def _cast(value: str, kind: Any) -> Any:
    if kind is bool:
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"not a boolean: {value!r}")
    if value.strip() == "" or value.strip().lower() == "none":
        return None
    return kind(value)


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


_SECRET_KEYS = {"token", "api_key", "password", "secret", "authorization"}


def _has_secret_key(data: Any) -> bool:
    if isinstance(data, dict):
        return any(str(k).lower() in _SECRET_KEYS or _has_secret_key(v) for k, v in data.items())
    return False


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (no escape processing, so Windows paths stay intact); comments and blanks skipped."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_dotenv_into_process() -> list[str]:
    """Load nass/.env (or $AGENTICDOC_ENV_FILE) into os.environ for variables that are NOT set
    already (the real environment always wins). Returns the names loaded; values are never logged."""
    file = Path(os.environ.get(DOTENV_ENV) or DEFAULT_DOTENV)
    if not file.is_file():
        return []
    loaded = []
    for key, value in read_dotenv(file).items():
        if key not in os.environ and value != "" and not value.startswith("<"):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def load_config(path: str | Path | None = None, environ: dict[str, str] | None = None) -> AutoConfig:
    """With `environ=None` (the app), nass/.env is loaded first; an explicit `environ` (tests)
    is used exactly as given and no .env file is read."""
    if environ is None:
        load_dotenv_into_process()
    environ = dict(os.environ if environ is None else environ)
    data: dict[str, Any] = AutoConfig().model_dump(mode="python")
    file = Path(path or environ.get(CONFIG_ENV) or DEFAULT_CONFIG_FILE)
    if file.is_file():
        with file.open("rb") as handle:
            data = _merge(data, tomllib.load(handle))
    elif path or environ.get(CONFIG_ENV):
        raise ConfigError(f"configuration file not found: {file}")
    for name, (section, field, kind) in ENVIRONMENT.items():
        if name in environ:
            try:
                value = _cast(environ[name], kind)
            except (ValueError, ConfigError) as exc:
                raise ConfigError(f"{name}: {exc}") from exc
            if section:
                data[section][field] = value
            else:
                data[field] = value
    retired = [name for name in RETIRED_ENVIRONMENT if name in environ]
    if retired:  # never reinterpret an old trust setting silently
        raise ConfigError(", ".join(retired) + " is no longer used: set QARI_DEPLOYMENT_MODE=local|internal|"
                          "development_external (and QARI_TRUSTED_DEV_HOSTS for development_external) instead")
    overrides = data.get("proven_overrides") or {}
    if overrides and not data.get("allow_proven_overrides"):
        raise ConfigError("proven_overrides are set but AUTO_ALLOW_PROVEN_OVERRIDES is not true")
    if overrides:
        data["proven"] = _merge(data["proven"], overrides)
    if file.is_file() and _has_secret_key(data):
        raise ConfigError("tokens must not be stored in the configuration file; use QARI_SERVICE_TOKEN")
    try:
        config = AutoConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc
    if config.qari.deployment_mode == "development_external" and config.app.environment == "production":
        raise ConfigError("QARI_DEPLOYMENT_MODE=development_external is not allowed when AGENTICDOC_ENV=production "
                          "(production uses a local or internal Qari service)")
    if config.qari.deployment_mode == "development_external" and not config.qari.trusted_development_hosts:
        raise ConfigError("QARI_DEPLOYMENT_MODE=development_external needs QARI_TRUSTED_DEV_HOSTS (the exact "
                          "host names the administrator trusts)")
    return config


def layout_model_status(config: AutoConfig) -> dict[str, Any]:
    """Readiness of the local PP-DocLayout model directory (checks files only; never loads or downloads)."""
    from lab.core.hybrid import layout_regions  # noqa: PLC0415

    model_dir = config.layout.model_dir
    if not model_dir:
        return {"ok": False, "model_dir": None,
                "problems": [r"LAYOUT_MODEL_DIR is not set (e.g. LAYOUT_MODEL_DIR=C:\Users\<you>\models"
                             r"\PP-DocLayout_plus-L_infer in nass\.env)"]}
    if not Path(model_dir).is_dir():
        return {"ok": False, "model_dir": model_dir,
                "problems": [f"LAYOUT_MODEL_DIR does not exist or is not a directory: {model_dir}"]}
    return layout_regions.doclayout_setup(model_dir, config.proven.layout.model_name)


def proven_differences(config: AutoConfig) -> dict[str, Any]:
    """Every proven setting that differs from the benchmark configuration (recorded in manifests)."""
    reference = ProvenSettings().model_dump()
    current = config.proven.model_dump()
    differences = {}
    for section, values in reference.items():
        for key, value in values.items():
            if current[section][key] != value:
                differences[f"{section}.{key}"] = {"proven": value, "configured": current[section][key]}
    return differences
