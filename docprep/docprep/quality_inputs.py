"""Read-only loader for the three Component 3 input layers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inputs import InputError, RunInputs, load_run
from .relation_inputs import SOURCE_FILES, hash_existing

STRUCTURE_FILES = ("section_tree.json", "assignments.json", "manifest.json")
RELATION_FILES = ("relationships.json", "visuals.json", "lists.json", "manifest.json")


@dataclass(frozen=True)
class QualityInputs:
    run: RunInputs
    structure_dir: Path
    relations_dir: Path
    section_tree: dict[str, Any]
    assignments: dict[str, Any]
    structure_manifest: dict[str, Any]
    relationships: dict[str, Any]
    visuals: dict[str, Any]
    lists: dict[str, Any]
    relation_manifest: dict[str, Any]
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


def load_quality_inputs(run_dir: Path | str, structure_dir: Path | str,
                        relations_dir: Path | str) -> QualityInputs:
    run = load_run(run_dir)
    structure, relations = Path(structure_dir), Path(relations_dir)
    _require(structure, STRUCTURE_FILES, "Structure output")
    _require(relations, RELATION_FILES, "Relationship output")
    tree = _read_object(structure / "section_tree.json")
    assignments = _read_object(structure / "assignments.json")
    structure_manifest = _read_object(structure / "manifest.json")
    relationship_data = _read_object(relations / "relationships.json")
    visuals = _read_object(relations / "visuals.json")
    lists = _read_object(relations / "lists.json")
    relation_manifest = _read_object(relations / "manifest.json")
    identities = {
        str(tree.get("run_id")), str(assignments.get("run_id")),
        str(structure_manifest.get("source_run_id")), str(relationship_data.get("source_run_id")),
        str(visuals.get("source_run_id")), str(lists.get("source_run_id")),
        str(relation_manifest.get("source_run_id")),
    }
    identities.discard("None")
    if identities != {run.run_id}:
        raise InputError(f"input layers do not all belong to source run {run.run_id}: {sorted(identities)}")
    if not isinstance(relationship_data.get("nodes"), list) or not isinstance(
        relationship_data.get("relationships"), list
    ):
        raise InputError("relationships.json lacks nodes or relationships")
    return QualityInputs(
        run, structure, relations, tree, assignments, structure_manifest, relationship_data, visuals, lists,
        relation_manifest,
        {"source": hash_existing(run.run_dir, SOURCE_FILES),
         "structure": hash_existing(structure, STRUCTURE_FILES),
         "relations": hash_existing(relations, RELATION_FILES)},
    )


def rehash_quality_inputs(inputs: QualityInputs) -> dict[str, dict[str, str]]:
    return {
        "source": hash_existing(inputs.run.run_dir, SOURCE_FILES),
        "structure": hash_existing(inputs.structure_dir, STRUCTURE_FILES),
        "relations": hash_existing(inputs.relations_dir, RELATION_FILES),
    }

