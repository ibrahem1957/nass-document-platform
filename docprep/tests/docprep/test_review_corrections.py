"""Regression tests for corrections made while validating the scanned-PDF fixes. All content is synthetic."""

from __future__ import annotations

from docprep.relation_builder import Builder
from docprep.relation_text import numeric_tokens

from .helpers import el
from .test_quality import _build as _build_quality
from .test_quality import _find
from .test_relations import _build, _types


def test_figure_and_table_label_numbers_are_identifiers_not_values():
    for text in ("الشكل 11: أداء الخطوط", "Figure 11: Route Performance", "الجدول 3 يوضح", "Table 4 lists"):
        assert [token.dimension for token in numeric_tokens(text)] == ["identifier"], text
    # A measured value next to the label stays a value.
    tokens = numeric_tokens("الشكل 11: بلغ التأخير 3.2 دقيقة")
    assert [token.dimension for token in tokens][0] == "identifier"
    assert any(token.raw.startswith("3.2") and token.dimension != "identifier" for token in tokens)


def _named(score):
    return {"accepted": True, "score": score, "source": "internal_title", "reason": "synthetic named reference"}


def test_paragraph_naming_two_visuals_equally_creates_no_describes(tmp_path, monkeypatch):
    monkeypatch.setattr(Builder, "_named_visual_reference", lambda self, paragraph, visual: _named(0.92))
    elements = [
        el("p", 1, 0, "The following tables show route performance and station comparison."),
        el("t1", 1, 1, "Route Performance delay riders", type="table"),
        el("t2", 1, 2, "Station Comparison revenue riders", type="table"),
    ]
    result, _, _ = _build(tmp_path, elements, sections={item["id"]: "s1" for item in elements})
    assert not _types(result, "describes")
    finding = next(item for item in result.findings if item.get("code") == "ambiguous_description_target")
    assert "names competing visuals" in finding["message"]


def test_clearly_stronger_named_reference_still_describes(tmp_path, monkeypatch):
    monkeypatch.setattr(Builder, "_named_visual_reference",
                        lambda self, paragraph, visual: _named(0.92 if visual.element_id == "t1" else 0.70))
    elements = [
        el("p", 1, 0, "The following table shows route performance."),
        el("t1", 1, 1, "Route Performance delay riders", type="table"),
        el("t2", 1, 2, "Station Comparison revenue riders", type="table"),
    ]
    result, _, _ = _build(tmp_path, elements, sections={item["id"]: "s1" for item in elements})
    assert [(item.source_id, item.target_id) for item in _types(result, "describes")] == [("p", "t1")]


def test_text_bearing_split_visual_keeps_its_own_weak_link(tmp_path):
    split = {"type": "containment", "decision": "split into inner visuals", "inner": [3, 4]}
    elements = [
        el("container", 1, 0, "", type="figure", state="pending_review",
           provenance={"job_id": "p001_R0", "decisions": split}),
        el("left", 1, 1, "left panel labels", type="figure", provenance={"job_id": "p001_R1"}),
        el("right", 1, 2, "right panel labels", type="figure", provenance={"job_id": "p001_R2"}),
        el("p", 1, 3, "Nearby explanatory material without a semantic visual reference."),
    ]
    layout = {"pages": [{"page": 1, "selection": {"regions": [
        {"region_id": "p1_R0", "job_id": "p001_R0", "block_index": 1, "label": "image"},
        {"region_id": "p1_R1", "job_id": "p001_R1", "block_index": 3, "label": "image"},
        {"region_id": "p1_R2", "job_id": "p001_R2", "block_index": 4, "label": "image"},
    ], "others": []}}]}
    result, _, _ = _build(tmp_path, elements, layout=layout, sections={item["id"]: "s1" for item in elements})
    assert {(r.source_id, r.target_id) for r in _types(result, "composite_part")} == {
        ("left", "container"), ("right", "container")}
    weak = _types(result, "co_located_in_section")
    assert any(item.target_id == "right" for item in weak)
    assert not any(item.evidence.get("redirected_from_composite_part") in {"left", "right"} for item in weak)


def test_large_edit_distance_between_unrelated_short_words_is_not_ocr_noise(tmp_path):
    # Two exact context anchors, but «متوسط» and «ومستوى» are different words (low character similarity).
    elements = [
        el("body", 1, 0, "زمن الانتظار في المحطات الرئيسية ومستوى الازدحام في المحطات خلال اليوم."),
        el("fig", 1, 1, "زمن الانتظار في المحطات الرئيسية متوسط الازدحام المحطات", type="figure",
           source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build_quality(tmp_path, elements)
    assert not [item for item in _find(result, "ocr_vocabulary_noise")
                if item.evidence["original_token"] == "متوسط"]
    audit = [item for item in result.evidence_audit if item["check"] == "ocr_vocabulary_noise"
             and item["evidence"].get("token") == "متوسط"]
    assert audit and audit[0]["status"] != "flagged"
    assert result.nodes["fig"]["text"].startswith("زمن الانتظار")


def test_large_edit_distance_with_high_character_similarity_is_still_flagged(tmp_path):
    visual = "التنبؤ المبكر بلحظالية وجود إصابات"
    elements = [
        el("body", 1, 0, "التنبؤ المبكر باحتمالية وجود إصابات في الحوادث."),
        el("fig", 1, 1, visual, type="figure", source="qari_crop", model={"label": "ocr"}),
    ]
    result, *_ = _build_quality(tmp_path, elements)
    finding = next(item for item in _find(result, "ocr_vocabulary_noise")
                   if item.evidence["original_token"] == "بلحظالية")
    assert finding.evidence["edit_distance"] >= 3
    assert result.nodes["fig"]["text"] == visual
