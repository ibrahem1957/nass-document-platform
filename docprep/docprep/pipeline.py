"""Stage orchestration: inputs -> units -> signals -> headings -> levels -> tree -> validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .aside import detect_asides
from .config import Config
from .detection import detect_headings
from .diagnostics import layout_heading_diagnostics
from .features import build_units, compute_features
from .furniture import detect_furniture
from .hierarchy import assign_levels
from .inputs import RunInputs, hash_inputs, load_run
from .layout import load_layout, match_bbox
from .model import DerivedSpan, HeadingCandidate, Unit
from .semantic import SemanticVerifier, make_verifier
from .tree import TreeResult, build_tree, section_to_dict
from .typography import (
    BodyProfile,
    associate_page,
    attach_fake_bold,
    body_fake_bold_share,
    body_profile,
    read_pdf,
)
from .validation import validate

DEFAULT_PAGE_SIZE = (612.0, 792.0)


@dataclass
class StructureResult:
    inputs: RunInputs
    config: Config
    units: list[Unit]
    spans: dict[str, list[DerivedSpan]]
    candidates: list[HeadingCandidate]
    levels: list[dict[str, Any]]
    tree: TreeResult
    body: BodyProfile
    page_sizes: dict[int, tuple[float, float]]
    page_size_source: dict[int, str]
    findings: list[dict[str, Any]]
    warnings: list[str]
    started_at: str
    finished_at: str
    hashes_after: dict[str, str | None] = field(default_factory=dict)
    layout_diagnostics: list[Any] = field(default_factory=list)   # diagnostics.HeadingRegionDiagnostic


def _page_sizes(inputs: RunInputs, pdf_sizes, layout_sizes) -> tuple[dict[int, tuple[float, float]], dict[int, str]]:
    sizes, source = {}, {}
    for page in range(1, inputs.page_count + 1):
        if page in pdf_sizes:
            sizes[page], source[page] = pdf_sizes[page], "source.pdf"
        elif page in layout_sizes:
            sizes[page], source[page] = layout_sizes[page], "plan/layout.json page image"
        else:
            boxes = [e.bbox for e in inputs.elements if e.page == page and e.bbox]
            if boxes:
                w = max(DEFAULT_PAGE_SIZE[0], max(b[2] for b in boxes))
                h = max(DEFAULT_PAGE_SIZE[1], max(b[3] for b in boxes))
                sizes[page], source[page] = (w, h), "estimated from element boxes (US Letter minimum)"
            else:
                sizes[page], source[page] = DEFAULT_PAGE_SIZE, "default US Letter (no geometry)"
    return sizes, source


def build_structure(run_dir: Path | str, config: Config, *, verifier: SemanticVerifier | None = None,
                    use_pdf: bool = True) -> StructureResult:
    started = datetime.now(timezone.utc).isoformat()
    inputs = load_run(run_dir)
    warnings = list(inputs.warnings)
    verifier = verifier or make_verifier(config.semantic.verifier, config.semantic.enabled)

    layout_boxes, layout_sizes, layout_warnings = load_layout(inputs.layout if config.layout.enabled else None)
    warnings += layout_warnings
    routes = {n: str(p.get("route") or "UNKNOWN") for n, p in inputs.pages.items()}
    text_pages = {n for n, r in routes.items() if r in ("DIGITAL", "MIXED")}
    pdf_sizes: dict[int, tuple[float, float]] = {}
    pdf = None
    if use_pdf and config.typography.enabled and inputs.pdf_path is not None:
        pdf = read_pdf(inputs.pdf_path, text_pages)
        pdf_sizes = pdf.page_sizes
        warnings += pdf.warnings
    elif not use_pdf:
        warnings.append("typography disabled for this run (--no-pdf)")
    page_sizes, size_source = _page_sizes(inputs, pdf_sizes, layout_sizes)

    units, spans = build_units(inputs.elements, inputs.pages, layout_boxes, config)

    # Typography: whole elements on digital/mixed pages only (never derived spans, never image-only pages).
    unit_by_id = {u.unit_id: u for u in units}
    if pdf is not None and pdf.spans:
        for page, page_spans in pdf.spans.items():
            items = [(u.unit_id, u.bbox, u.text) for u in units
                     if u.page == page and u.span is None and u.bbox and u.clean
                     and u.element_type not in ("figure", "table")]
            typographies = associate_page(items, page_spans, config.typography)
            if page in pdf.trace_pages:
                attach_fake_bold(items, pdf.traces.get(page, []), typographies, config.typography)
            else:
                for typography in typographies.values():
                    typography.fake_bold_note = "text trace unavailable for this page"
            for unit_id, typography in typographies.items():
                unit_by_id[unit_id].typography = typography
    text_typographies = [u.typography for u in units if u.typography is not None
                         and u.element_type in ("text", "heading", "paragraph", "title")]
    body = body_profile(text_typographies)
    body.fake_bold_share = body_fake_bold_share(text_typographies)

    # Layout matches for whole elements with a real (not page-wide) box.
    if config.layout.enabled:
        for unit in units:
            if unit.span is None and unit.bbox and unit.page in layout_boxes:
                height = page_sizes[unit.page][1]
                if unit.bbox[3] - unit.bbox[1] <= 0.6 * height:
                    unit.layout = match_bbox(unit.bbox, layout_boxes[unit.page], config.layout)

    compute_features(units, page_sizes, config)
    furniture = detect_furniture(units, page_sizes, inputs.page_count, config.furniture)
    for unit in units:
        unit.furniture = furniture.get(unit.unit_id)
    for unit_id, info in detect_asides(units, page_sizes, config.aside, config.patterns).items():
        unit_by_id[unit_id].features["aside"] = info.to_dict()

    candidates = detect_headings(units, body, config, verifier)
    by_id = {u.unit_id: u for u in units}
    ordered_content = [u.unit_id for u in units if u.furniture is None and u.clean
                       and u.element_type not in ("figure", "table")]
    levels = assign_levels(candidates, by_id, body, ordered_content, config)
    tree = build_tree(units, candidates, inputs.validation_report, config)
    warnings += tree.warnings
    heading_units = {a.unit_id: a.section_id for a in tree.assignments if a.role == "heading"}
    layout_diagnostics = layout_heading_diagnostics(units, layout_boxes if config.layout.enabled else {},
                                                    candidates, heading_units, config)

    hashes_after = hash_inputs(inputs.run_dir)
    findings = validate(inputs.elements, units, spans, candidates, tree, inputs.page_count,
                        inputs.input_hashes, hashes_after)
    return StructureResult(inputs=inputs, config=config, units=units, spans=spans, candidates=candidates,
                           levels=levels, tree=tree, body=body, page_sizes=page_sizes, page_size_source=size_source,
                           findings=findings, warnings=warnings, started_at=started,
                           finished_at=datetime.now(timezone.utc).isoformat(), hashes_after=hashes_after,
                           layout_diagnostics=layout_diagnostics)


# ------------------------------------------------------------------ serialization

def _outline(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {s["section_id"]: s for s in sections}

    def node(sid: str) -> dict[str, Any]:
        s = by_id[sid]
        return {"section_id": sid, "title": s["title"], "level": s["level"], "page_start": s["page_start"],
                "page_end": s["page_end"], "children": [node(c) for c in s["children"]]}

    return [node(s["section_id"]) for s in sections if s["parent"] is None]


def section_tree_json(result: StructureResult) -> dict[str, Any]:
    cands = {c.candidate_id: c for c in result.candidates}
    lists = {node.list_id: node for node in result.tree.lists}
    sections = [section_to_dict(s, cands[s.heading_candidate_id], lists) for s in result.tree.sections]
    unassigned = [{"unit_id": a.unit_id, "element_id": a.element_id, "span_id": a.span_id, "page": a.page,
                   "role": a.role, "text": next(u.text for u in result.units if u.unit_id == a.unit_id),
                   "reasons": a.reasons, "validation_state": a.validation_state}
                  for a in result.tree.assignments if a.section_id is None]
    counts = {status: sum(1 for c in result.candidates if c.status == status)
              for status in ("promoted", "rejected", "excluded")}
    doc = result.inputs.document.get("document") or {}
    return {
        "schema_version": 1, "kind": "docprep_section_tree", "docprep_version": __version__,
        "run_id": result.inputs.run_id, "source_run_dir": str(result.inputs.run_dir),
        "document": {"display_name": doc.get("display_name"), "page_count": result.inputs.page_count,
                     "sha256": doc.get("sha256")},
        "outline": _outline(sections), "sections": sections,
        "lists": [node.to_dict() for node in result.tree.lists], "unassigned": unassigned,
        "heading_candidate_counts": counts,
        "heading_candidates": [c.to_dict() for c in result.candidates],
        "style_levels": result.levels, "body_typography": result.body.to_dict(),
        "layout_heading_diagnostics": [d.to_dict() for d in result.layout_diagnostics],
    }


def assignments_json(result: StructureResult) -> dict[str, Any]:
    represented: dict[str, list[str]] = {}
    for unit in result.units:
        represented.setdefault(unit.element_id, []).append(unit.unit_id)
    inventory = []
    for element in result.inputs.elements:
        inventory.append({
            "element_id": element.element_id, "page": element.page, "type": element.element_type,
            "reading_order": element.reading_order, "document_order": element.order, "bbox_pt": element.bbox,
            "validation_state": element.validation_state, "source": element.source, "text": element.text,
            "represented_by": represented.get(element.element_id, []),
            "derived_spans": [s.to_dict() for s in result.spans.get(element.element_id, [])],
        })
    units = {u.unit_id: u for u in result.units}
    assignments = []
    for a in result.tree.assignments:
        unit = units[a.unit_id]
        record = a.to_dict()
        record["text"] = unit.text
        record["bbox_pt"] = unit.bbox
        record["source_run_id"] = result.inputs.run_id
        record["typography"] = unit.typography.to_dict() if unit.typography is not None else None
        record["layout"] = unit.layout.to_dict() if unit.layout is not None else None
        assignments.append(record)
    return {"schema_version": 1, "kind": "docprep_assignments", "docprep_version": __version__,
            "run_id": result.inputs.run_id, "source_inventory": inventory, "assignments": assignments}
