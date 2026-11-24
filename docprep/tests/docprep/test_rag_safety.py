"""Pre-RAG safety contract: content risks become element-level RAG flags, tables expose their cell-structure
state, ambiguous number separators claim no value, and the prepared document is fully schema-validated.
All content is synthetic."""

from __future__ import annotations

import pytest

from docprep.prepared_builder import _table_structure
from docprep.prepared_validation import _schema_check
from docprep.quality_policy import rag_flag, unflagged_content_concern
from docprep.relation_text import match_numeric_tokens, numeric_tokens

from .helpers import el
from .test_prepare import _build as _build_prepared
from .test_quality import _build as _build_quality
from .test_quality import _find


def _record(result, source_id):
    return next(item for item in result.inventory if item.source_id == source_id)


def _readiness(record):
    return record.reliability_dimensions["retrieval_readiness"]["state"]


# ------------------------------------------------------------------------------------------ RAG flags

def test_flattened_table_is_flagged_table_cells_unknown(tmp_path):
    result, *_ = _build_quality(tmp_path, [el("t", 1, 0, "Line delay riders Line 1 3.2 38,920", type="table",
                                              source="qari_crop")])
    assert _find(result, "flattened_unstructured_table")
    record = _record(result, "t")
    assert "table_cells_unknown" in record.rag_flags
    assert record.evidence_reliability == "medium"  # uncertain, not declared erroneous
    assert _readiness(record) == "use_with_flags"
    assert result.nodes["t"]["text"] == "Line delay riders Line 1 3.2 38,920"


def test_incomplete_region_on_text_bearing_visual_is_flagged(tmp_path):
    validation = {"flags": [{"page": 1, "region_id": "p1_R1", "code": "incomplete_region",
                             "message": "the region border cuts another block; content may be missing"}]}
    elements = [el("body", 1, 0, "page text", type="page_text", provenance={"job_id": "p001_page"}),
                el("fig", 1, 1, "network facts 1.2M riders", type="figure", provenance={"job_id": "p1_R1"})]
    result, *_ = _build_quality(tmp_path, elements, validation=validation)
    record = _record(result, "fig")
    assert "possible_incomplete_content" in record.rag_flags and _readiness(record) == "use_with_flags"
    assert "possible_incomplete_content" not in _record(result, "body").rag_flags  # scope stays on the region


@pytest.mark.parametrize("severity, flagged", [("warning", True), ("info", False)])
def test_margin_leakage_flags_foreign_text_only_when_review_is_recommended(tmp_path, severity, flagged):
    finding = {"severity": severity, "code": "possible_caption_margin_leakage", "items": ["fig"],
               "message": "layout blocks inside the crop margin", "evidence": {"review_recommended": flagged}}
    result, *_ = _build_quality(tmp_path, [el("fig", 1, 0, "Figure 13: Log form name card", type="figure")],
                                relationship_findings=[finding])
    record = _record(result, "fig")
    assert ("possible_foreign_text" in record.rag_flags) is flagged
    if flagged:
        assert _readiness(record) == "use_with_flags"
    assert result.nodes["fig"]["text"] == "Figure 13: Log form name card"


def test_unflagged_content_finding_is_never_no_recorded_concern(tmp_path):
    validation = {"flags": [{"page": 1, "element_id": "fig", "code": "layout_review",
                             "message": "image inside another image without frame evidence; both kept"}]}
    result, *_ = _build_quality(tmp_path, [el("fig", 1, 0, "chart labels 12 14", type="figure")],
                                validation=validation)
    assert _find(result, "nonfatal_layout_warning")
    record = _record(result, "fig")
    assert not record.rag_flags
    assert _readiness(record) == "content_concern_recorded"
    assert record.reliability_dimensions["retrieval_readiness"]["unflagged_content_concerns"]


def test_flag_policy_is_shared_and_conservative():
    assert rag_flag("table_quality", "flattened_unstructured_table", "high", {}) == "table_cells_unknown"
    assert rag_flag("source_extraction", "incomplete_region_coverage", "high", {}) == "possible_incomplete_content"
    leak = {"component2_finding": {"severity": "warning"}}
    assert rag_flag("relationship_quality", "possible_caption_margin_leakage", "high", leak) == "possible_foreign_text"
    assert rag_flag("relationship_quality", "possible_caption_margin_leakage", "high",
                    {"component2_finding": {"severity": "info"}}) is None
    assert rag_flag("relationship_quality", "unresolved_caption_candidate", "high", {}) is None
    assert unflagged_content_concern("source_extraction", "nonfatal_layout_warning", "low", "unverified", "high", {})
    assert not unflagged_content_concern("numeric_integrity", "numeric_agreement", "info", "not_applicable",
                                         "high", {})


# ------------------------------------------------------------------------------------------ tables

def test_table_without_cells_disallows_cell_level_evidence():
    element = {"source_extraction": {"cells": None, "cells_note": "the OCR output is not a Markdown/HTML table"},
               "rag_flags": ["table_cells_unknown"]}
    state = _table_structure(element, {"rag_flags": []})
    assert state["table_structure"] == "none" and state["cell_level_evidence_allowed"] is False
    assert "must not be used as verified cell evidence" in state["table_consumer_rule"]


def test_table_with_source_cells_allows_cell_level_evidence():
    element = {"source_extraction": {"cells": [{"row": 0, "col": 0, "text": "A"}], "cells_note": "find_tables"},
               "rag_flags": []}
    state = _table_structure(element, {"rag_flags": []})
    assert state["table_structure"] == "source_cells" and state["cell_level_evidence_allowed"] is True


def test_cells_contradicted_by_a_flattened_flag_are_not_trusted():
    element = {"source_extraction": {"cells": [{"row": 0, "col": 0, "text": "A"}]}, "rag_flags": []}
    state = _table_structure(element, {"rag_flags": ["table_cells_unknown"]})
    assert state["table_structure"] == "none"


# ------------------------------------------------------------------------------------------ numbers

@pytest.mark.parametrize("raw, value, dimension", [
    ("٦,٤", None, "ambiguous_number"),
    ("٢,١٥", None, "ambiguous_number"),
    ("6,4", None, "ambiguous_number"),
    ("12,5", None, "ambiguous_number"),
    ("1,250", "1250", "scalar"),
    ("1,234,567", "1234567", "scalar"),
    ("٣,٤٥٠", "3450", "scalar"),
    ("٣٬٤٥٠", "3450", "scalar"),
    ("1٬250", "1250", "scalar"),
    ("٦٫٤", "6.4", "scalar"),
    ("٢٫١٥٪", "2.15", "percent"),
    ("1,234.5", "1234.5", "scalar"),
    ("38,920", "38920", "scalar"),
])
def test_locale_aware_separators(raw, value, dimension):
    token = numeric_tokens(raw)[0]
    assert token.raw == raw and token.dimension == dimension
    if value is None:
        assert token.value.startswith("ambiguous:") and token.normalization.startswith("ambiguous")
    else:
        assert token.value == value


def test_ambiguous_numbers_never_match_or_conflict(tmp_path):
    assert match_numeric_tokens("متوسط العمر ٦,٤ سنة", "متوسط العمر 64 سنة") == []
    assert match_numeric_tokens("average age 6,4 years", "average age 6,4 years") == []
    # Verified matching is unchanged.
    assert match_numeric_tokens("recorded 49.58% of cases", "49.58%")[0]["equality_asserted"] is True


# ------------------------------------------------------------------------------------------ schema

def test_prepared_document_is_fully_schema_validated(tmp_path):
    result, *_ = _build_prepared(tmp_path)
    check = next(item for item in result.validation if item["code"] == "json_schema_valid")
    assert check["severity"] == "ok"


def test_schema_violation_is_a_validation_error(tmp_path):
    result, *_ = _build_prepared(tmp_path)
    broken = dict(result.document)
    broken["elements"] = [dict(result.document["elements"][0], unexpected_field=True)]
    check = _schema_check(broken)
    assert check["severity"] == "error" and check["items"]
    missing = dict(result.document)
    missing.pop("quality")
    assert _schema_check(missing)["severity"] == "error"
