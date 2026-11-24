"""Read-only loading for Nass and Document Structure artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inputs import InputError, RunInputs, load_run, sha256_file

STRUCTURE_FILES = ("section_tree.json", "assignments.json", "manifest.json")
SOURCE_FILES = (
    "document.json", "validation_report.json", "execution_manifest.json", "plan/layout.json", "source.pdf",
)


@dataclass(frozen=True)
class RelationshipInputs:
    run: RunInputs
    structure_dir: Path
    section_tree: dict[str, Any]
    assignments: dict[str, Any]
    structure_manifest: dict[str, Any]
    source_hashes: dict[str, str]
    structure_hashes: dict[str, str]


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


def hash_existing(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def load_relationship_inputs(run_dir: Path | str, structure_dir: Path | str) -> RelationshipInputs:
    run = load_run(run_dir)
    structure = Path(structure_dir)
    if not structure.is_dir():
        raise InputError(f"structure output folder not found: {structure}")
    missing = [name for name in STRUCTURE_FILES if not (structure / name).is_file()]
    if missing:
        raise InputError(f"incomplete structure output: missing {', '.join(missing)}")
    tree = _read_object(structure / "section_tree.json")
    assignments = _read_object(structure / "assignments.json")
    manifest = _read_object(structure / "manifest.json")
    expected = {str(tree.get("run_id")), str(assignments.get("run_id")), str(manifest.get("source_run_id"))}
    expected.discard("None")
    if expected != {run.run_id}:
        raise InputError(f"structure output does not belong to source run {run.run_id}: {sorted(expected)}")
    if not isinstance(assignments.get("source_inventory"), list) or not isinstance(
        assignments.get("assignments"), list
    ):
        raise InputError("assignments.json lacks source_inventory or assignments")
    return RelationshipInputs(
        run=run,
        structure_dir=structure,
        section_tree=tree,
        assignments=assignments,
        structure_manifest=manifest,
        source_hashes=hash_existing(run.run_dir, SOURCE_FILES),
        structure_hashes=hash_existing(structure, STRUCTURE_FILES),
    )

