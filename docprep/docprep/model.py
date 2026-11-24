"""Core data structures shared by the stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceElement:
    """One original Nass element, unchanged. Represented exactly once in the source inventory."""

    element_id: str
    page: int
    element_type: str
    reading_order: int | None
    bbox: list[float] | None
    text: str | None
    validation_state: str | None
    source: str | None
    order: int                      # position in the document-wide Nass order
    raw: dict[str, Any]


@dataclass
class DerivedSpan:
    """A span derived from one original element's text. Offsets refer to the exact original text."""

    span_id: str
    element_id: str
    start: int
    end: int
    text: str                       # exactly original_text[start:end]
    cut_reason: str                 # why this span starts here
    role_hint: str                  # heading | caption | list_item | paragraph | unknown
    support: str                    # text | layout | both | none
    basis: list[str]
    geometry: dict[str, Any] | None
    uncertain: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Signal:
    name: str
    contribution: float
    detail: str
    category: str                   # source_type | typography | layout | pattern | position | context | semantic | style

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "contribution": round(self.contribution, 4), "detail": self.detail,
                "category": self.category}


@dataclass
class Unit:
    """An assignable content unit: an original element, or one derived span of an element."""

    unit_id: str
    element: SourceElement
    span: DerivedSpan | None
    order: tuple[int, int]
    text: str                       # verbatim text of the unit
    page: int
    route: str
    direction: str | None
    features: dict[str, Any] = field(default_factory=dict)
    typography: Any = None          # typography.ElementTypography | None
    layout: Any = None              # layout.LayoutMatch | None
    furniture: Any = None           # furniture.FurnitureInfo | None

    @property
    def element_id(self) -> str:
        return self.element.element_id

    @property
    def span_id(self) -> str | None:
        return self.span.span_id if self.span else None

    @property
    def clean(self) -> str:
        return " ".join(self.text.split())

    @property
    def bbox(self) -> list[float] | None:
        """Element box for whole elements; derived spans never inherit the element box."""
        return None if self.span else self.element.bbox

    @property
    def element_type(self) -> str:
        return self.element.element_type

    @property
    def validation_state(self) -> str | None:
        return self.element.validation_state


@dataclass
class HeadingCandidate:
    candidate_id: str
    unit_ids: list[str]
    page: int
    text: str                       # display title (verbatim parts joined by a space)
    parts: list[str]                # verbatim text of each source unit
    signals: list[Signal]
    score: float
    status: str                     # promoted | rejected | excluded
    exclusions: list[str]
    confidence: str | None          # detection confidence for promoted headings
    ambiguous: bool
    review: bool
    conflicts: list[str]
    semantic: dict[str, Any] | None
    style_key: Any
    pattern_class: str = "plain"
    numbering: dict[str, Any] | None = None
    level: int | None = None              # final (normalized) section level
    raw_level: int | None = None          # level from the style / numbering class, before normalization
    level_confidence: str | None = None
    level_basis: list[str] = field(default_factory=list)
    level_conflicts: list[str] = field(default_factory=list)
    level_ambiguous: bool = False
    is_document_title: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "unit_ids": self.unit_ids, "page": self.page, "text": self.text,
            "parts": self.parts, "status": self.status, "score": round(self.score, 4),
            "signals": [s.to_dict() for s in self.signals], "exclusions": self.exclusions,
            "confidence": self.confidence, "ambiguous": self.ambiguous, "review_recommended": self.review,
            "conflicts": self.conflicts, "semantic": self.semantic,
            "style_key": list(self.style_key) if isinstance(self.style_key, tuple) else self.style_key,
            "pattern_class": self.pattern_class, "numbering": self.numbering, "level": self.level,
            "raw_level": self.raw_level, "level_confidence": self.level_confidence, "level_basis": self.level_basis,
            "level_conflicts": self.level_conflicts, "level_ambiguous": self.level_ambiguous,
            "is_document_title": self.is_document_title, "method": "heuristic", "notes": self.notes,
        }
