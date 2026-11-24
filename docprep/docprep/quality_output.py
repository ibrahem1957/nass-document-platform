"""Atomic write-once output for Component 3."""

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
from .quality_config import DEFAULT_QUALITY_OUTPUT_ROOT, QualityConfig
from .quality_inputs import QualityInputs
from .quality_model import QualityResult
from .quality_reports import render_quality_html, render_quality_markdown
from .quality_validation import validate_quality, validation_summary

ARTIFACTS = ("quality_findings.json", "quality_summary.json", "quality_inventory.json",
             "quality_report.md", "quality_report.html")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_quality_outputs(result: QualityResult, inputs: QualityInputs, *,
                          out_root: Path | str = DEFAULT_QUALITY_OUTPUT_ROOT, stamp: str | None = None,
                          config: QualityConfig | None = None) -> tuple[Path, list[dict[str, Any]]]:
    config = config or QualityConfig()
    result.validation = validate_quality(result, inputs)
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{safe_name(result.run_id)}__{stamp or timestamp()}"
    if target.exists():
        raise OutputExistsError(f"quality output already exists (never overwritten): {target}")
    staging = root / f".{target.name}.partial-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        findings_json = {
            "schema_version": 1, "kind": "docprep_quality_findings", "docprep_version": __version__,
            "source_run_id": result.run_id, "findings": [item.to_dict() for item in result.findings],
            "evidence_audit": result.evidence_audit,
            "review_sources": {source_id: {"page": result.nodes[source_id].get("page"),
                                            "text": result.nodes[source_id].get("text")}
                               for source_id in sorted({value for item in result.findings for value in item.source_ids})
                               if source_id in result.nodes},
            "validation": {"summary": validation_summary(result.validation), "checks": result.validation},
        }
        summary_json = {"schema_version": 1, "kind": "docprep_quality_summary",
                        "source_run_id": result.run_id, **result.summary,
                        "validation_summary": validation_summary(result.validation)}
        inventory_json = {"schema_version": 1, "kind": "docprep_quality_inventory",
                          "source_run_id": result.run_id,
                          "inventory": [item.to_dict() for item in result.inventory]}
        contents = {
            "quality_findings.json": json.dumps(findings_json, ensure_ascii=False, indent=2),
            "quality_summary.json": json.dumps(summary_json, ensure_ascii=False, indent=2),
            "quality_inventory.json": json.dumps(inventory_json, ensure_ascii=False, indent=2),
            "quality_report.md": render_quality_markdown(result),
            "quality_report.html": render_quality_html(result),
        }
        hashes = {}
        for name, text in contents.items():
            path = staging / name
            path.write_text(text, encoding="utf-8", newline="\n")
            hashes[name] = _sha256(path)
        manifest = {
            "schema_version": 1, "kind": "docprep_quality_manifest",
            "component": "Document Quality & Validation Layer", "docprep_version": __version__,
            "source_run_id": result.run_id, "source_run_dir": result.run_dir,
            "structure_output_dir": result.structure_dir, "relationship_output_dir": result.relations_dir,
            "input_hashes_before": result.hashes_before, "input_hashes_after": result.hashes_after,
            "inputs_unchanged": result.hashes_before == result.hashes_after,
            "configuration": config.to_dict(), "output_hashes": hashes,
            "validation_summary": validation_summary(result.validation), "warnings": result.warnings,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "scope": "quality findings only; no OCR correction, rewriting, chunking, embeddings, retrieval or inference",
        }
        (staging / "quality_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                                       encoding="utf-8", newline="\n")
        if target.exists():
            raise OutputExistsError(f"quality output already exists (never overwritten): {target}")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if config.read_only_files:
        for path in target.iterdir():
            os.chmod(path, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return target, result.validation

