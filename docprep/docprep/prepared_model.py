"""In-memory result for Component 4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedResult:
    run_id: str
    document: dict[str, Any]
    input_hashes_before: dict[str, dict[str, str]]
    input_hashes_after: dict[str, dict[str, str]]
    validation: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
