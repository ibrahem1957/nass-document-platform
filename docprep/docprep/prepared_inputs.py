"""Strict, read-only loading of Component 4 upstream artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inputs import InputError, RunInputs, load_run
from .quality_inputs import RELATION_FILES, STRUCTURE_FILES
from .relation_inputs import SOURCE_FILES, hash_existing

QUALITY_FILES = (
    "quality_findings.json", "quality_summary.json", "quality_inventory.json", "quality_manifest.json",
)

EXPECTED_CONTRACTS = {
    "document": (1, "automatic_extraction_document"),
    "section_tree": (1, "docprep_section_tree"),
    "assignments": (1, "docprep_assignments"),
    "structure_manifest": (1, "docprep_manifest"),
    "relationships": (1, "docprep_content_relationships"),
    "visuals": (1, "docprep_visual_inventory"),
    "lists": (1, "docprep_list_relationships"),
    "relation_manifest": (1, "docprep_relationship_manifest"),
    "quality_findings": (1, "docprep_quality_findings"),
    "quality_summary": (1, "docprep_quality_summary"),
    "quality_inventory": (1, "docprep_quality_inventory"),
    "quality_manifest": (1, "docprep_quality_manifest"),
}


@dataclass(frozen=True)
class PreparedInputs:
    run: RunInputs
    structure_dir: Path
    relations_dir: Path
    quality_dir: Path
    section_tree: dict[str, Any]
    assignments: dict[str, Any]
    structure_manifest: dict[str, Any]
    relationships: dict[str, Any]
    visuals: dict[str, Any]
    lists: dict[str, Any]
    relation_manifest: dict[str, Any]
    quality_findings: dict[str, Any]
    quality_summary: dict[str, Any]
    quality_inventory: dict[str, Any]
    quality_manifest: dict[str, Any]
    hashes: dict[str, dict[str, str]]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"{path} must contain a JSON object")
    return value


def _require(root: Path, names: tuple[str, ...], label: str) -> None:
    if not root.is_dir():
        raise InputError(f"{label} folder not found: {root}")
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise InputError(f"incomplete {label}: missing {', '.join(missing)}")


def _contract(label: str, value: dict[str, Any]) -> None:
    expected_version, expected_kind = EXPECTED_CONTRACTS[label]
    if value.get("schema_version") != expected_version:
        raise InputError(
            f"incompatible {label} schema_version {value.get('schema_version')!r}; expected {expected_version}"
        )
    if value.get("kind") != expected_kind:
        raise InputError(f"incompatible {label} kind {value.get('kind')!r}; expected {expected_kind!r}")


def _require_list(value: dict[str, Any], key: str, label: str) -> None:
    if not isinstance(value.get(key), list):
        raise InputError(f"{label} lacks required list '{key}'")


def _verify_recorded_path(label: str, recorded: Any, actual: Path) -> None:
    if not isinstance(recorded, str) or not recorded:
        return
    expected = os.path.normcase(os.path.abspath(recorded))
    received = os.path.normcase(os.path.abspath(actual))
    if expected != received:
        raise InputError(f"{label} lineage mismatch: manifest records {recorded!r}, received {str(actual)!r}")


def load_prepared_inputs(run_dir: Path | str, structure_dir: Path | str, relations_dir: Path | str,
                         quality_dir: Path | str) -> PreparedInputs:
    run = load_run(run_dir)
    structure, relations, quality = Path(structure_dir), Path(relations_dir), Path(quality_dir)
    _require(structure, STRUCTURE_FILES, "Structure output")
    _require(relations, RELATION_FILES, "Relationship output")
    _require(quality, QUALITY_FILES, "Quality output")
    values = {
        "document": run.document,
        "section_tree": _read_object(structure / "section_tree.json"),
        "assignments": _read_object(structure / "assignments.json"),
        "structure_manifest": _read_object(structure / "manifest.json"),
        "relationships": _read_object(relations / "relationships.json"),
        "visuals": _read_object(relations / "visuals.json"),
        "lists": _read_object(relations / "lists.json"),
        "relation_manifest": _read_object(relations / "manifest.json"),
        "quality_findings": _read_object(quality / "quality_findings.json"),
        "quality_summary": _read_object(quality / "quality_summary.json"),
        "quality_inventory": _read_object(quality / "quality_inventory.json"),
        "quality_manifest": _read_object(quality / "quality_manifest.json"),
    }
    for label, value in values.items():
        _contract(label, value)
    identities = {
        str(values["section_tree"].get("run_id")), str(values["assignments"].get("run_id")),
        str(values["structure_manifest"].get("source_run_id")),
        str(values["relationships"].get("source_run_id")), str(values["visuals"].get("source_run_id")),
        str(values["lists"].get("source_run_id")), str(values["relation_manifest"].get("source_run_id")),
        str(values["quality_findings"].get("source_run_id")),
        str(values["quality_summary"].get("source_run_id")),
        str(values["quality_inventory"].get("source_run_id")),
        str(values["quality_manifest"].get("source_run_id")),
    }
    identities.discard("None")
    if identities != {run.run_id}:
        raise InputError(f"upstream layers do not all belong to source run {run.run_id}: {sorted(identities)}")
    _verify_recorded_path("Structure source", values["structure_manifest"].get("source_run_dir"), run.run_dir)
    _verify_recorded_path("Relationships source", values["relation_manifest"].get("source_run_dir"), run.run_dir)
    _verify_recorded_path("Relationships Structure", values["relation_manifest"].get("structure_output_dir"), structure)
    _verify_recorded_path("Quality source", values["quality_manifest"].get("source_run_dir"), run.run_dir)
    _verify_recorded_path("Quality Structure", values["quality_manifest"].get("structure_output_dir"), structure)
    _verify_recorded_path(
        "Quality Relationships", values["quality_manifest"].get("relationship_output_dir"), relations,
    )
    for value, key, label in (
        (values["section_tree"], "sections", "section_tree"),
        (values["section_tree"], "lists", "section_tree"),
        (values["assignments"], "source_inventory", "assignments"),
        (values["assignments"], "assignments", "assignments"),
        (values["relationships"], "nodes", "relationships"),
        (values["relationships"], "relationships", "relationships"),
        (values["visuals"], "visuals", "visuals"),
        (values["quality_findings"], "findings", "quality_findings"),
        (values["quality_inventory"], "inventory", "quality_inventory"),
    ):
        _require_list(value, key, label)
    return PreparedInputs(
        run, structure, relations, quality, values["section_tree"], values["assignments"],
        values["structure_manifest"], values["relationships"], values["visuals"], values["lists"],
        values["relation_manifest"], values["quality_findings"], values["quality_summary"],
        values["quality_inventory"], values["quality_manifest"],
        {"source": hash_existing(run.run_dir, SOURCE_FILES),
         "structure": hash_existing(structure, STRUCTURE_FILES),
         "relations": hash_existing(relations, RELATION_FILES),
         "quality": hash_existing(quality, QUALITY_FILES)},
    )


def rehash_prepared_inputs(inputs: PreparedInputs) -> dict[str, dict[str, str]]:
    return {
        "source": hash_existing(inputs.run.run_dir, SOURCE_FILES),
        "structure": hash_existing(inputs.structure_dir, STRUCTURE_FILES),
        "relations": hash_existing(inputs.relations_dir, RELATION_FILES),
        "quality": hash_existing(inputs.quality_dir, QUALITY_FILES),
    }
