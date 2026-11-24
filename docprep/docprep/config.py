"""Configuration: every threshold and weight is configurable (code defaults < TOML file < explicit overrides)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import tomllib

DEFAULT_OUTPUT_ROOT = r"D:\agenticdoc_e2\results\document_structure"
OUTPUT_ROOT_ENV = "DOCPREP_OUTPUT_ROOT"
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config" / "docprep.toml"


class ConfigError(ValueError):
    """Invalid configuration file or value."""


@dataclass
class Thresholds:
    promote: float = 0.45          # a candidate with this score or more becomes a heading
    high: float = 0.70             # detection confidence "high" at or above this score
    medium: float = 0.55           # detection confidence "medium" at or above this score (else "low")
    reject_floor: float = 0.25     # rejected candidates below this are listed as "weak"
    ambiguity_band: float = 0.08   # |score - promote| below this marks the decision as ambiguous


@dataclass
class Weights:
    nass_heading_type: float = 0.30
    size_large: float = 0.35
    size_medium: float = 0.25
    size_smaller_than_body: float = -0.15
    bold: float = 0.25
    distinct_color: float = 0.10
    mixed_sizes: float = -0.05
    layout_doc_title: float = 0.35
    layout_paragraph_title: float = 0.25
    layout_body_text: float = -0.05
    layout_header_top: float = 0.05
    numbering: float = 0.20
    multilevel_numbering: float = 0.05
    ordinal: float = 0.20
    colon_short: float = 0.15
    no_terminal_punctuation: float = 0.05
    terminal_period: float = -0.10
    short_standalone: float = 0.10
    bullet_prefixed: float = -0.10
    span_heading_hint: float = 0.10
    title_echo: float = 0.15
    gap_before: float = 0.05
    followed_by_body: float = 0.05
    followed_by_same_style_candidate: float = -0.05
    introduces_numbered_sequence: float = 0.05  # same weight as followed_by_body
    centered: float = 0.05
    first_on_first_page: float = 0.10
    style_consistency: float = 0.10
    # A distinctive (non-body) colour/weight used document-wide almost only by short heading-like lines.
    repeated_heading_style: float = 0.15
    semantic_max: float = 0.10
    semantic_min: float = -0.05
    fake_bold: float = 0.05                     # low weight: supporting evidence only
    numbered_list_sequence: float = -0.10       # numbered line directly followed by the next numbered line


@dataclass
class TypographyConfig:
    enabled: bool = True
    size_ratio_large: float = 1.40
    size_ratio_medium: float = 1.15
    size_ratio_small: float = 0.90
    bold_share: float = 0.80          # share of characters in bold for an element to count as bold
    size_tolerance: float = 0.25      # pt; sizes this close count as the same size
    size_class_tolerance: float = 0.5 # pt; heading style classes
    distinct_size_gap: float = 1.0    # pt between heading classes for a "high" level confidence
    color_distance: int = 60          # RGB distance for a "distinct" colour
    bbox_tolerance: float = 2.0       # pt added around element boxes when attaching spans
    min_span_overlap: float = 0.5     # share of a span's area inside the element box
    min_char_match: float = 0.80      # share of span characters found in the element text
    min_element_coverage: float = 0.60  # share of element characters covered by attached spans
    min_consistency: float = 0.70     # share of characters at the dominant size


@dataclass
class LayoutConfig:
    enabled: bool = True
    min_score: float = 0.50           # detector confidence below this halves the layout contribution
    min_overlap: float = 0.50         # containment needed to associate a box with an element
    ambiguity_delta: float = 0.15     # two boxes this close in containment with different roles = ambiguous


@dataclass
class PatternConfig:
    max_heading_words: int = 20
    max_heading_chars: int = 200
    short_words: int = 10
    colon_heading_max_words: int = 10
    max_heading_lines: int = 3
    repeated_style_min_members: int = 3      # heading-like lines sharing one distinctive style
    repeated_style_min_pages: int = 2        # ... spread over at least this many pages
    repeated_style_min_share: float = 0.80   # share of all lines in that style that are heading-like
    fragment_max_chars: int = 2
    list_intro_min_words: int = 5          # floor of the derived "longer than a heading" word count
    list_intro_heading_factor: float = 1.5  # x median word count of the document's anchor headings


@dataclass
class ListConfig:
    trailing_marker_max_words: int = 12     # RTL lines with an end-of-line marker ("Power BI -")
    continuation_max_words: int = 8         # short line continuing the previous list item
    continuation_max_gap_lines: float = 1.0  # vertical gap, in line heights of the previous item
    continuation_min_x_overlap: float = 0.3  # horizontal overlap with the previous item (share of the shorter)


@dataclass
class AsideConfig:
    enabled: bool = True
    max_width_share: float = 0.60           # of the document's main body-column width
    body_min_words: int = 12                # paragraphs used to measure the body column
    min_vertical_overlap: float = 0.30      # overlap with the neighbouring element (share of the label height)
    min_horizontal_gap: float = 4.0         # pt between the label and its neighbour
    multicolumn_min_units: int = 3          # narrow text units on both halves of a page = multi-column page


@dataclass
class FurnitureConfig:
    top_band: float = 0.10            # top share of the page height
    bottom_band: float = 0.92         # elements starting below this share are in the bottom band
    min_repeat_pages: int = 2
    min_repeat_share: float = 0.30


@dataclass
class SegmentationConfig:
    enabled: bool = True
    element_types: list[str] = field(default_factory=lambda: ["page_text"])
    sources: list[str] = field(default_factory=lambda: ["qari_full_page"])
    colon_label_max_words: int = 8
    title_echo_min_words: int = 3
    title_echo_max_words: int = 12


@dataclass
class SemanticConfig:
    enabled: bool = True
    verifier: str = "heuristic"        # "heuristic" or "none"
    lookahead_units: int = 2
    assignment_warnings: bool = True   # flag paragraphs that lexically match the next heading (never moved)
    assignment_min_shared_terms: int = 2
    common_term_section_share: float = 0.30   # terms present in at least this share of sections are ignored
    min_sections_for_df: int = 3               # below this many sections no term is treated as common
    min_heading_terms: int = 2                 # fewer meaningful words in a heading: the check is skipped
    skip_when_next_heading_uncertain: bool = True  # never use a low-confidence heading as evidence


@dataclass
class HierarchyConfig:
    max_levels: int = 4
    detect_document_title: bool = True
    title_search_units: int = 3
    numbered_run_indent_tolerance: float = 6.0


@dataclass
class OutputConfig:
    root: str = DEFAULT_OUTPUT_ROOT
    read_only_files: bool = True


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    weights: Weights = field(default_factory=Weights)
    typography: TypographyConfig = field(default_factory=TypographyConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    patterns: PatternConfig = field(default_factory=PatternConfig)
    lists: ListConfig = field(default_factory=ListConfig)
    aside: AsideConfig = field(default_factory=AsideConfig)
    furniture: FurnitureConfig = field(default_factory=FurnitureConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(target: Any, values: dict[str, Any], path: str) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"unknown configuration key: {path}{key}")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{key} must be a table")
            _merge(current, value, f"{path}{key}.")
            continue
        if isinstance(current, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{path}{key} must be true or false")
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigError(f"{path}{key} must be a number")
            value = type(current)(value) if isinstance(current, float) else value
        elif isinstance(current, list):
            if not isinstance(value, list):
                raise ConfigError(f"{path}{key} must be a list")
        elif isinstance(current, str) and not isinstance(value, str):
            raise ConfigError(f"{path}{key} must be a string")
        setattr(target, key, value)


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None, *,
                use_default_file: bool = True) -> Config:
    """Defaults, then the TOML file (explicit path, else config/docprep.toml when present), then overrides.

    The output root may also come from the DOCPREP_OUTPUT_ROOT environment variable (applied before overrides).
    """
    config = Config()
    file = path if path is not None else (DEFAULT_CONFIG_FILE if use_default_file and DEFAULT_CONFIG_FILE.is_file() else None)
    if file is not None:
        try:
            data = tomllib.loads(Path(file).read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read configuration {file}: {exc}") from exc
        _merge(config, data, "")
    env_root = os.environ.get(OUTPUT_ROOT_ENV)
    if env_root:
        config.output.root = env_root
    if overrides:
        _merge(config, overrides, "")
    t = config.thresholds
    if not (0 <= t.reject_floor <= t.promote <= t.medium <= t.high):
        raise ConfigError("thresholds must satisfy 0 <= reject_floor <= promote <= medium <= high")
    return config
