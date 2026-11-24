"""Configuration for the deterministic Document Quality layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_QUALITY_OUTPUT_ROOT = Path(r"D:\agenticdoc_e2\results\document_quality")


@dataclass(frozen=True)
class QualityConfig:
    output_root: str = str(DEFAULT_QUALITY_OUTPUT_ROOT)
    low_ocr_confidence: float = 0.60
    near_duplicate_ratio: float = 0.90
    near_duplicate_min_chars: int = 40
    sparse_visual_terms: int = 3
    vocabulary_min_same_script_tokens: int = 150   # independent same-script text needed to verify OCR words
    vocabulary_min_visual_script_tokens: int = 3   # OCR word tokens of a script before it is assessed
    # Edit distance 3-4 is only plausible OCR damage when most characters still agree (else: a different word).
    vocabulary_large_distance_min_similarity: float = 0.6
    numeric_label_min_shared_terms: int = 2
    cross_page_bottom_y: float = 700.0
    cross_page_top_y: float = 120.0
    read_only_files: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

