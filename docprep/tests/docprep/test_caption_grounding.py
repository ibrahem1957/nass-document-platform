"""Caption grounding for scanned pages: caption spans (offsets known, coordinates unknown), saved figure_title
regions and visual regions. All content is synthetic."""

from __future__ import annotations

from .helpers import el
from .test_relations import _build, _types

PAGE = [0, 0, 612, 792]
TOP = [130, 200, 480, 390]      # visual 1
BOTTOM = [130, 440, 480, 635]   # visual 2


def _page(caption_1: str, caption_2: str, intro: str = "Synthetic report page. ") -> tuple[str, list[dict]]:
    text = intro + caption_1 + " " + caption_2
    first = len(intro)
    second = first + len(caption_1) + 1
    spans = [
        {"span_id": "page_span_001", "element_id": "page", "start": 0, "end": first, "text": intro,
         "role_hint": "paragraph", "uncertain": False, "geometry": None},
        {"span_id": "page_span_002", "element_id": "page", "start": first, "end": first + len(caption_1) + 1,
         "text": caption_1 + " ", "role_hint": "caption", "uncertain": False, "geometry": None},
        {"span_id": "page_span_003", "element_id": "page", "start": second, "end": len(text), "text": caption_2,
         "role_hint": "caption", "uncertain": False, "geometry": None},
    ]
    return text, spans


def _layout(titles: list[dict], regions: list[dict] | None = None) -> dict:
    regions = regions if regions is not None else [
        {"region_id": "p1_R0", "job_id": "p001_R0", "block_index": 0, "label": "chart", "bbox_pt": TOP},
        {"region_id": "p1_R1", "job_id": "p001_R1", "block_index": 3, "label": "chart", "bbox_pt": BOTTOM},
    ]
    return {"pages": [{"page": 1, "boxes": [], "selection": {"regions": regions, "others": titles}}]}


def _title(block: int, bbox: list[float], **extra) -> dict:
    return {"block_index": block, "label": "figure_title", "bbox_pt": bbox, "score": 0.9, **extra}


def _elements(text: str, visual_1: str, visual_2: str, **provenance_1) -> list[dict]:
    return [
        el("page", 1, 0, text, type="page_text", bbox=PAGE, source="qari_full_page"),
        el("v1", 1, 1, visual_1, type="figure", bbox=TOP, provenance={"job_id": "p001_R0", **provenance_1}),
        el("v2", 1, 2, visual_2, type="figure", bbox=BOTTOM, provenance={"job_id": "p001_R1"}),
    ]


# Captions below their visuals (the "captions_below" policy is off, so Nass attached neither).
BELOW_TITLES = [_title(4, [230, 406, 385, 419]), _title(2, [230, 651, 387, 663])]
CAPTION_1 = "Figure 1: Daily trips by route"
CAPTION_2 = "Figure 2: Waiting time at stations"
VISUAL_1 = "Number of daily trips by route 18,450 12,730 Red Blue"
VISUAL_2 = "Average waiting time at main stations 16 12 8 minutes"


def _sections(elements):
    return {item["id"]: "s1" for item in elements}


def test_full_page_ocr_box_never_makes_the_page_a_caption(tmp_path):
    # Page OCR that starts with a caption pattern and whose page-sized box "overlaps" the figure.
    elements = [el("page", 1, 0, "Figure 3: Route performance. The page continues with body text.",
                   type="page_text", bbox=PAGE, source="qari_full_page"),
                el("fig", 1, 1, "Route performance delay riders", type="figure", bbox=TOP)]
    result, _, _ = _build(tmp_path, elements, sections=_sections(elements))
    assert not [item for item in _types(result, "caption_of") if item.source_id == "page"]


def test_convergent_caption_span_is_linked_without_inventing_geometry(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2)
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(BELOW_TITLES),
                          sections=_sections(elements))
    captions = {item.target_id: item for item in _types(result, "caption_of")}
    assert set(captions) == {"v1", "v2"}
    for relation in captions.values():
        assert relation.confidence == "medium" and relation.review_recommended
        assert relation.evidence["span_geometry_used"] is False and relation.evidence["span_geometry"] is None
        node = result.nodes[relation.source_id]
        assert node.node_type == "caption_span" and text[node.start:node.end] == node.text
        assert relation.evidence["layout_title_region"]["source"] == "measured_by_docprep_from_saved_boxes"
    assert result.nodes["page"].text == text


def test_title_region_between_two_visuals_at_similar_distances_stays_unresolved(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2)
    # Visual 1 caption region sits 16 pt below v1 and 18 pt above v2 (< 5 pt margin); no other region near v1.
    titles = [_title(4, [230, 406, 385, 422]), _title(2, [230, 651, 387, 663])]
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(titles),
                          sections=_sections(elements))
    assert "v1" not in {item.target_id for item in _types(result, "caption_of")}
    unresolved = [item for item in result.findings if item["code"] == "unresolved_caption_candidate"
                  and item["evidence"]["span_id"] == "page_span_002"]
    assert unresolved and unresolved[0]["evidence"]["failed_checks"]


def test_caption_inside_another_visual_crop_stays_unresolved(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    # v2's crop OCR contains caption 1 verbatim: the caption may belong to either visual.
    elements = _elements(text, VISUAL_1, CAPTION_1 + " " + VISUAL_2)
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(BELOW_TITLES),
                          sections=_sections(elements))
    assert "v1" not in {item.target_id for item in _types(result, "caption_of")}
    finding = next(item for item in result.findings if item["code"] == "unresolved_caption_candidate"
                   and item["evidence"]["span_id"] == "page_span_002")
    assert any("verbatim" in check for check in finding["evidence"]["failed_checks"])


def test_title_region_attached_to_the_visual_is_not_used_as_its_caption(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2)
    regions = [
        {"region_id": "p1_R0", "job_id": "p001_R0", "block_index": 0, "label": "chart", "bbox_pt": TOP,
         "title_attachments": [{"block_index": 4, "position": "above"}]},
        {"region_id": "p1_R1", "job_id": "p001_R1", "block_index": 3, "label": "chart", "bbox_pt": BOTTOM},
    ]
    # Block 4 is the visual's own attached title (above it); no unattached region next to v1 remains.
    titles = [_title(4, [230, 180, 385, 195]), _title(2, [230, 651, 387, 663])]
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(titles, regions),
                          sections=_sections(elements))
    assert "v1" not in {item.target_id for item in _types(result, "caption_of")}


def test_weak_word_agreement_stays_unresolved(tmp_path):
    text, spans = _page("Figure 1: Complaint distribution", CAPTION_2)
    elements = _elements(text, "Complaint reasons per quarter delay crowding", VISUAL_2)
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(BELOW_TITLES),
                          sections=_sections(elements))
    assert "v1" not in {item.target_id for item in _types(result, "caption_of")}


def test_arabic_caption_with_arabic_digits(tmp_path):
    text, spans = _page("الشكل ١: الرحلات اليومية حسب المسار", "الشكل ٢: زمن الانتظار في المحطات", "تقرير تجريبي. ")
    elements = _elements(text, "عدد الرحلات اليومية حسب المسار 18,450 المسار الأحمر",
                         "متوسط زمن الانتظار في المحطات الرئيسية 16 12 دقائق")
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(BELOW_TITLES),
                          sections=_sections(elements))
    captions = {item.target_id: item for item in _types(result, "caption_of")}
    assert set(captions) == {"v1", "v2"}
    assert captions["v1"].evidence["caption_pattern"] == ["figure", "1"]


def test_saved_nass_title_association_is_reported_as_the_source(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2)
    association = {"schema": 1, "status": "candidate_not_attached_by_policy", "candidates": []}
    titles = [_title(4, [230, 406, 385, 419], title_association=association),
              _title(2, [230, 651, 387, 663], title_association=association)]
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(titles),
                          sections=_sections(elements))
    relation = next(item for item in _types(result, "caption_of") if item.target_id == "v1")
    assert relation.evidence["layout_title_region"]["source"] == "nass_title_association"


def test_older_run_without_layout_metadata_is_readable_and_links_nothing(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2)
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, sections=_sections(elements))
    assert not _types(result, "caption_of")
    assert any(item["code"] == "unresolved_caption_candidate" for item in result.findings)


def test_composite_member_target_stays_unresolved(tmp_path):
    text, spans = _page(CAPTION_1, CAPTION_2)
    elements = _elements(text, VISUAL_1, VISUAL_2,
                         decisions=[{"type": "containment", "decision": "outer visual preferred", "inner": [9]}])
    result, _, _ = _build(tmp_path, elements, derived_spans={"page": spans}, layout=_layout(BELOW_TITLES),
                          sections=_sections(elements))
    assert "v1" not in {item.target_id for item in _types(result, "caption_of")}
