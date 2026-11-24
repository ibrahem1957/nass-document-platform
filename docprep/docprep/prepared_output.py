"""Atomic, immutable Component 4 output."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .output import OutputExistsError, safe_name, timestamp
from .prepared_config import DEFAULT_PREPARED_OUTPUT_ROOT, PreparedConfig
from .prepared_inputs import PreparedInputs, rehash_prepared_inputs
from .prepared_model import PreparedResult
from .prepared_reports import render_prepared_html, render_prepared_markdown
from .prepared_validation import validate_prepared, validation_summary

REQUIRED_ARTIFACTS = (
    "prepared_document.json", "prepared_document_validation.json", "prepared_document_report.md",
    "prepared_document_report.html",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prepared_outputs(result: PreparedResult, inputs: PreparedInputs, *,
                           out_root: Path | str = DEFAULT_PREPARED_OUTPUT_ROOT, stamp: str | None = None,
                           config: PreparedConfig | None = None) -> tuple[Path, list[dict[str, Any]]]:
    config = config or PreparedConfig()
    result.input_hashes_after = rehash_prepared_inputs(inputs)
    result.validation = validate_prepared(result, inputs)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{safe_name(result.run_id)}__{stamp or timestamp()}"
    if target.exists():
        raise OutputExistsError(f"prepared output already exists (never overwritten): {target}")
    staging = root / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        validation_data = {
            "schema_version": 1, "kind": "docprep_prepared_document_validation",
            "source_run_id": result.run_id, "summary": validation_summary(result.validation),
            "checks": result.validation,
        }
        contents = {
            "prepared_document.json": json.dumps(result.document, ensure_ascii=False, indent=2),
            "prepared_document_validation.json": json.dumps(validation_data, ensure_ascii=False, indent=2),
            "prepared_document_report.md": render_prepared_markdown(result),
            "prepared_document_report.html": render_prepared_html(result),
        }
        output_hashes = {}
        for name, value in contents.items():
            path = staging / name
            path.write_text(value, encoding="utf-8", newline="\n")
            output_hashes[name] = _sha256(path)
        manifest = {
            "schema_version": 1, "kind": "docprep_prepared_document_manifest",
            "component": "Unified Prepared Document Builder", "docprep_version": __version__,
            "prepared_document_schema_version": config.schema_version,
            "source_run_id": result.run_id,
            "source_run_dir": str(inputs.run.run_dir), "structure_output_dir": str(inputs.structure_dir),
            "relationship_output_dir": str(inputs.relations_dir), "quality_output_dir": str(inputs.quality_dir),
            "input_hashes_before": result.input_hashes_before,
            "input_hashes_after": result.input_hashes_after,
            "inputs_unchanged": result.input_hashes_before == result.input_hashes_after,
            "configuration": config.to_dict(),
            "processing_configuration_sha256": result.document["provenance"]["processing_configuration_sha256"],
            "output_hashes": output_hashes,
            "validation_summary": validation_summary(result.validation),
            "required_artifacts": [*REQUIRED_ARTIFACTS, "prepared_document_manifest.json"],
            "written_at": datetime.now(timezone.utc).isoformat(),
            "scope": "unified prepared representation only; no correction, chunking, embeddings, retrieval or inference",
        }
        (staging / "prepared_document_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n",
        )
        if target.exists():
            raise OutputExistsError(f"prepared output already exists (never overwritten): {target}")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if config.read_only_files:
        for path in target.iterdir():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return target, result.validation
