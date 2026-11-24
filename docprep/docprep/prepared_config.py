"""Configuration contract for the Unified Prepared Document Builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_PREPARED_OUTPUT_ROOT = Path(r"D:\agenticdoc_e2\results\prepared_documents")
PREPARED_SCHEMA_VERSION = "1.1.0"


@dataclass(frozen=True)
class PreparedConfig:
    output_root: str = str(DEFAULT_PREPARED_OUTPUT_ROOT)
    schema_version: str = PREPARED_SCHEMA_VERSION
    upstream_schema_version: int = 1
    preserve_source_text: bool = True
    include_upstream_evidence: bool = True
    read_only_files: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
