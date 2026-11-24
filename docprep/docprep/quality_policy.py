"""Shared policy for separating content reliability from structural and relationship review.

Evidence reliability describes whether the *source text or visual content* can be trusted as evidence.
Structural placement (section/heading/list uncertainty) and relationship confidence are recorded as separate
dimensions: they matter for navigation and linking, but they do not make the extracted words less reliable.
"""

from __future__ import annotations

from typing import Any

# Categories whose findings describe where content sits or how it links, not whether its text is reliable.
STRUCTURAL_CATEGORIES = {"structure_quality", "cross_page_continuity"}
RELATIONSHIP_CATEGORIES = {"relationship_quality"}
# Findings that describe missing or probably incomplete evidence rather than corrupted text.
COMPLETENESS_SUBTYPES = {
    "missing_extraction", "recorded_known_missing", "probable_missing_numeric_evidence", "chart_without_values",
    "sparse_visual_ocr", "incomplete_region_coverage", "layout_coverage_limitation",
}
STRUCTURAL_SUBTYPES = {"ambiguous_reading_order"}

DIMENSIONS = ("source_text_reliability", "structural_placement", "relationship_confidence",
              "evidence_completeness")


def reliability_dimension(category: str | None, subtype: str | None) -> str:
    """The reliability dimension a finding belongs to."""
    if category in RELATIONSHIP_CATEGORIES:
        return "relationship_confidence"
    if category in STRUCTURAL_CATEGORIES:
        return "structural_placement"
    if subtype in STRUCTURAL_SUBTYPES:
        return "structural_placement"
    if subtype in COMPLETENESS_SUBTYPES:
        return "evidence_completeness"
    return "source_text_reliability"


def affects_content_reliability(
    category: str | None, subtype: str | None, confidence: str | None, evidence: dict[str, Any] | None,
) -> bool:
    """Return whether a finding may change element/visual content (evidence) reliability.

    Relationship and structural findings remain attached and reviewable in their own dimensions, but they
    describe link quality or section placement rather than the integrity of the source content. Script-level
    "unverifiable" vocabulary notes are informational and never lower reliability.
    """
    if category in RELATIONSHIP_CATEGORIES or category in STRUCTURAL_CATEGORIES or subtype in STRUCTURAL_SUBTYPES:
        return False
    if subtype == "unverifiable_script_vocabulary":
        return False
    if subtype == "ocr_vocabulary_noise":
        return confidence == "high" and (evidence or {}).get("evidence_strength") == "strong"
    return True


# Element-level RAG flags. One map shared by the Quality and Prepare stages, so a flag can never be produced by
# one stage and dropped by the other. A flag marks uncertain or incomplete evidence; it never states an error.
RAG_FLAG_BY_SUBTYPE = {
    "bidi_order_suspect": "bidi_order_suspect",
    "ocr_repetition_loop": "ocr_repetition_loop",
    "probable_missing_numeric_evidence": "probable_missing_numeric_evidence",
    "ocr_vocabulary_noise": "ocr_noise",
    "chart_without_values": "chart_without_values",
    "source_content_anomaly": "source_anomaly",
    "unverified_anomaly": "needs_review",
    "missing_extraction": "known_missing",
    "recorded_known_missing": "known_missing",
    # A table whose text has no verified row/column structure: cell-level associations are unavailable.
    "flattened_unstructured_table": "table_cells_unknown",
    # A saved region border cuts another layout block: some of the element's content may be missing.
    "incomplete_region_coverage": "possible_incomplete_content",
}
# Relationship-category findings that nevertheless concern the element's own text: flagged only when the
# upstream finding recommends review (never for geometric risk alone).
FOREIGN_TEXT_SUBTYPES = {"possible_caption_margin_leakage"}


def rag_flag(category: str | None, subtype: str | None, confidence: str | None,
             evidence: dict[str, Any] | None) -> str | None:
    """The RAG flag a finding puts on its source element, or None."""
    if subtype in FOREIGN_TEXT_SUBTYPES:
        upstream = (evidence or {}).get("component2_finding") or {}
        review = (upstream.get("severity") == "warning"
                  or bool((upstream.get("evidence") or {}).get("review_recommended"))
                  or bool((evidence or {}).get("review_recommended")))
        return "possible_foreign_text" if review else None
    if not affects_content_reliability(category, subtype, confidence, evidence):
        return None
    return RAG_FLAG_BY_SUBTYPE.get(str(subtype))


def unflagged_content_concern(category: str | None, subtype: str | None, severity: str | None,
                              status: str | None, confidence: str | None, evidence: dict[str, Any] | None) -> bool:
    """A recorded source-text or completeness concern that carries no RAG flag (e.g. a page-level layout
    warning). It must still keep the element out of `no_recorded_concern`."""
    return (affects_content_reliability(category, subtype, confidence, evidence)
            and rag_flag(category, subtype, confidence, evidence) is None
            and severity not in {"info"} and status not in {"not_applicable", "resolved"})


def finding_reason(finding: dict[str, Any]) -> str:
    """A precise, stable reason string for one finding (shared by Quality and Prepare)."""
    parts = [(f"{finding.get('subtype')} ({finding.get('category')}, {finding.get('severity')}, "
              f"status {finding.get('status')}; finding {finding.get('finding_id')})")]
    offsets = [item for item in finding.get("offsets") or [] if isinstance(item, dict)]
    if offsets:
        spans = ", ".join(f"{item.get('start')}-{item.get('end')}" for item in offsets[:3])
        parts.append(f"offsets {spans}" + (" ..." if len(offsets) > 3 else ""))
    return "; ".join(parts)
