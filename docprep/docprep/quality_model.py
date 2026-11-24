"""Component 3 data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class QualityFinding:
    finding_id: str
    category: str
    subtype: str
    severity: str
    confidence: str
    source_ids: list[str]
    page_numbers: list[int]
    evidence: dict[str, Any]
    explanation: str
    suggested_review_action: str
    status: str
    offsets: list[dict[str, Any]] = field(default_factory=list)
    related_ids: list[str] = field(default_factory=list)
    relationship_ids: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    method: str = "heuristic"
    priority_score: int = 0
    priority_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryRecord:
    source_id: str
    page: int
    element_type: str
    validation_state: str | None
    structure_assignment: dict[str, Any] | None
    relationship_ids: list[str]
    categories_attempted: list[str]
    skipped_checks: dict[str, str]
    geometry_reliability: str = "unavailable"
    finding_ids: list[str] = field(default_factory=list)
    rag_flags: list[str] = field(default_factory=list)
    evidence_reliability: str = "unassessed"
    evidence_reliability_reasons: list[str] = field(default_factory=list)
    # Separate dimensions (diagnostic, backward compatible): source text reliability, structural placement,
    # relationship confidence, evidence completeness and retrieval readiness.
    reliability_dimensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityResult:
    run_id: str
    run_dir: str
    structure_dir: str
    relations_dir: str
    findings: list[QualityFinding]
    inventory: list[InventoryRecord]
    summary: dict[str, Any]
    nodes: dict[str, dict[str, Any]]
    warnings: list[str]
    hashes_before: dict[str, dict[str, str]]
    hashes_after: dict[str, dict[str, str]]
    evidence_audit: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)

