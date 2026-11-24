"""Data contracts for the Content Relationship Builder.

The relationship component never mutates source elements.  Any item that does not
already have a Nass or Structure identifier is represented as an auditable derived
node with source provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RelationNode:
    node_id: str
    node_type: str
    page: int | None
    text: str | None
    source_id: str | None
    start: int | None = None
    end: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    review_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_flags"] = list(self.review_flags)
        return data


@dataclass
class Relationship:
    rel_id: str
    type: str
    source_id: str
    target_id: str
    evidence: dict[str, Any]
    confidence: str
    method: str
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    source_type: str | None = None
    target_type: str | None = None
    source_page: int | None = None
    target_page: int | None = None
    review_recommended: bool = False
    limitations: list[str] = field(default_factory=list)
    source_validation_state: str | None = None
    target_validation_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationResult:
    run_id: str
    source_run_dir: str
    structure_dir: str
    nodes: dict[str, RelationNode]
    relationships: list[Relationship]
    visuals: list[dict[str, Any]]
    lists: dict[str, Any]
    evidence_audit: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    warnings: list[str]
    source_hashes_before: dict[str, str]
    structure_hashes_before: dict[str, str]
    source_hashes_after: dict[str, str] = field(default_factory=dict)
    structure_hashes_after: dict[str, str] = field(default_factory=dict)

    def relationship_dicts(self) -> list[dict[str, Any]]:
        return [relation.to_dict() for relation in self.relationships]

