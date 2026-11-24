from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ExistingDocument:
    document_id: str
    label: str
    nass_document: Path
    prepared_document: Path
    source_pdf: Path


@dataclass(frozen=True)
class StudioConfig:
    project_root: Path
    jobs_root: Path
    nass_root: Path
    nass_python: Path
    docprep_root: Path
    docprep_python: Path
    max_upload_mb: int
    stage_timeout_s: int
    existing_documents: tuple[ExistingDocument, ...]


CURRENT_PYTHON = "current"


def _path(value: Any, label: str, *, must_exist: bool = True, base: Path | None = None) -> Path:
    """Absolute paths are used as written; relative paths resolve against `base` (the Studio project
    folder), never against the process working directory."""
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = base / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise ConfigError(f"{label} does not exist: {path}")
    return path


def _python(value: Any, label: str, base: Path) -> Path:
    """`"current"` selects the interpreter running the Studio (one shared virtual environment)."""
    if value == CURRENT_PYTHON:
        return Path(sys.executable).resolve()
    return _path(value, label, base=base)


def load_config(path: Path | str) -> StudioConfig:
    config_path = Path(path).resolve(strict=True)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read Studio configuration: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != "document_processing_studio_config":
        raise ConfigError("incompatible Studio configuration")
    if value.get("schema_version") != 1:
        raise ConfigError("unsupported Studio configuration schema_version")
    project_root = config_path.parent.parent
    jobs_value = value.get("jobs_root", "data/jobs")
    if not isinstance(jobs_value, str):
        raise ConfigError("jobs_root must be a path string")
    jobs_root = (project_root / jobs_value).resolve() if not Path(jobs_value).is_absolute() else Path(jobs_value).resolve()
    max_upload_mb = value.get("max_upload_mb", 200)
    timeout = value.get("stage_timeout_s", 3600)
    if not isinstance(max_upload_mb, int) or not 1 <= max_upload_mb <= 500:
        raise ConfigError("max_upload_mb must be between 1 and 500")
    if not isinstance(timeout, int) or timeout < 30:
        raise ConfigError("stage_timeout_s must be at least 30")
    documents = []
    ids: set[str] = set()
    for item in value.get("existing_documents", []):
        if not isinstance(item, dict):
            raise ConfigError("invalid existing document entry")
        document_id, label = item.get("id"), item.get("label")
        if not isinstance(document_id, str) or not document_id.replace("-", "").isalnum():
            raise ConfigError("existing document id must be alphanumeric with optional hyphens")
        if document_id in ids or not isinstance(label, str) or not label:
            raise ConfigError("duplicate existing document id or missing label")
        ids.add(document_id)
        documents.append(ExistingDocument(
            document_id, label,
            _path(item.get("nass_document"), f"{document_id} Nass document", base=project_root),
            _path(item.get("prepared_document"), f"{document_id} prepared document", base=project_root),
            _path(item.get("source_pdf"), f"{document_id} source PDF", base=project_root),
        ))
    return StudioConfig(
        project_root=project_root,
        jobs_root=jobs_root,
        nass_root=_path(value.get("nass_root"), "nass_root", base=project_root),
        nass_python=_python(value.get("nass_python"), "nass_python", project_root),
        docprep_root=_path(value.get("docprep_root"), "docprep_root", base=project_root),
        docprep_python=_python(value.get("docprep_python"), "docprep_python", project_root),
        max_upload_mb=max_upload_mb,
        stage_timeout_s=timeout,
        existing_documents=tuple(documents),
    )
